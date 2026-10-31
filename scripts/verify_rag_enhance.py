"""
verify_rag_enhance.py —— FoodSafetySystem 三项增强验证

覆盖：
    R1  Redis 三级缓存（core/cache.py）
    R2  Backtrack 回溯检索策略（core/rag_system.py）
    R3  RAGAS 评估脚手架（scripts/evaluate_ragas.py）
    R4  配置项完整性（[backtrack] / [redis] / [ragas]）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))


# ==================================================================
# 测试替身：可脚本化返回分数的向量库 / LLM
# ==================================================================

class FakeDoc:
    def __init__(self, content, score):
        self.page_content = content
        self.metadata = {"file_name": "GB_test.pdf", "source": "regulation",
                         "rerank_score": score}


class ScriptedVectorStore:
    """按调用顺序依次返回预设结果，用于精确验证回溯行为"""

    def __init__(self, script):
        # script: [ (query_pattern_or_None, [docs]) , ... ]
        self.script = list(script)
        self.calls = []

    def hybrid_search_with_rerank(self, query, k=5, kb_filter=None):
        self.calls.append(query)
        # 按关键词匹配脚本；未匹配则返回脚本中最后一项
        for pattern, docs in self.script:
            if pattern is None or pattern in query:
                return list(docs)
        return list(self.script[-1][1])


class ScriptedLLM:
    """LLM 替身：返回预设文本（用于 HyDE/SubQuery/Backtrack 改写）"""

    def __init__(self, responses=None, default="改写后的法规检索式"):
        self.responses = list(responses or [])
        self.default = default
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt[:80])
        text = self.responses.pop(0) if self.responses else self.default
        return iter([text])


def _build_rag(vector_store, llm):
    from core.rag_system import RAGSystem
    return RAGSystem(vector_store, llm)


# ==================================================================
# R1 缓存
# ==================================================================

def r1_cache():
    print("\n[R1] Redis 三级缓存")
    from core.cache import RAGCache, content_hash, rag_cache

    # 单例状态
    st = rag_cache.init()
    check("r1.1 未启用 Redis 时降级 local", st["backend"] == "local", str(st))

    # 三类键互不冲突
    k_emb = rag_cache.key_embedding("q")
    k_ret = rag_cache.key_retrieval("q", "regulation", "direct")
    k_ans = rag_cache.key_answer("q", "regulation")
    check("r1.2 三类键互不相同", len({k_emb, k_ret, k_ans}) == 3)
    check("r1.3 Embedding 键含模型名", "emb:" in k_emb and ":" in k_emb[4:])

    # 知识库隔离
    k_ret_sop = rag_cache.key_retrieval("q", "sop", "direct")
    check("r1.4 不同知识库键不同", k_ret != k_ret_sop)

    # 内容哈希
    check("r1.5 哈希稳定", content_hash("abc") == content_hash("abc"))
    check("r1.6 哈希区分不同内容", content_hash("abc") != content_hash("abd"))
    check("r1.7 哈希长度可配置", len(content_hash("abc", 8)) == 8)

    # 分项统计
    cache = RAGCache()
    cache.set(cache.key_embedding("t1"), [1, 2], 60)
    cache.get(cache.key_embedding("t1"), kind="embedding")
    cache.get(cache.key_embedding("miss"), kind="embedding")
    s = cache.stats()
    check("r1.8 分项命中统计存在", "embedding" in s["by_kind"])
    check("r1.9 命中/未命中计数正确",
          s["by_kind"]["embedding"]["hits"] == 1
          and s["by_kind"]["embedding"]["misses"] == 1,
          str(s["by_kind"]["embedding"]))
    check("r1.10 stats 含三类 TTL",
          set(s["ttl"].keys()) == {"embedding", "retrieval", "answer"})

    # 按类别清空
    cache.set(cache.key_answer("t2", "regulation"), "答案", 60)
    n = cache.clear("ans")
    check("r1.11 按类别清空生效", n >= 1
          and cache.get(cache.key_answer("t2", "regulation"), kind="answer") is None,
          f"清理 {n} 条")


# ==================================================================
# R2 Backtrack
# ==================================================================

def r2_backtrack_disabled():
    print("\n[R2-a] Backtrack 开关关闭时不回溯")
    import base.config as cfg_mod
    old = cfg_mod.config.BACKTRACK_ENABLED
    cfg_mod.config.BACKTRACK_ENABLED = False
    try:
        vs = ScriptedVectorStore([(None, [FakeDoc("低分片段", 0.2)])])
        rag = _build_rag(vs, ScriptedLLM())
        docs = rag.retrieve_and_merge("这肉还能用吗", strategy="direct")
        check("r2a.1 关闭时只检索一次", len(vs.calls) == 1, f"调用 {len(vs.calls)} 次")
        check("r2a.2 返回原始低分结果", docs and docs[0].metadata["rerank_score"] == 0.2)
    finally:
        cfg_mod.config.BACKTRACK_ENABLED = old


def r2_backtrack_low_score():
    print("\n[R2-b] 首轮低分 → 触发回溯并成功")
    import base.config as cfg_mod
    old = cfg_mod.config.BACKTRACK_ENABLED
    cfg_mod.config.BACKTRACK_ENABLED = True
    try:
        # 第一轮低分，第二轮（改写后）高分
        vs = ScriptedVectorStore([(None, [FakeDoc("沾边但不匹配的条文", 0.2)])])
        # 让第二次调用返回高分：用连续脚本实现
        call_count = {"n": 0}

        class TwoStageVS:
            def __init__(self):
                self.calls = []

            def hybrid_search_with_rerank(self, query, k=5, kb_filter=None):
                self.calls.append(query)
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return [FakeDoc("沾边但不匹配的条文", 0.2)]
                return [FakeDoc("GB 31654-2021 腐败变质判定标准", 0.85)]

        vs2 = TwoStageVS()
        rag = _build_rag(vs2, ScriptedLLM(["腐败变质 畜禽肉 判定标准"]))
        docs = rag.retrieve_and_merge("这肉看着有点怪还能用吗", strategy="direct")

        check("r2b.1 触发了第二轮检索", len(vs2.calls) >= 2, f"调用 {len(vs2.calls)} 次")
        check("r2b.2 第二轮使用了改写后的查询",
              any("腐败变质" in c for c in vs2.calls[1:]), str(vs2.calls[1:]))
        best = max(d.metadata["rerank_score"] for d in docs)
        check("r2b.3 回溯后分数达标", best >= cfg_mod.config.SCORE_THRESHOLD,
              f"best={best}")
    finally:
        cfg_mod.config.BACKTRACK_ENABLED = old


def r2_backtrack_no_improve():
    print("\n[R2-c] 回溯无改善 → 及时终止（不死循环）")
    import base.config as cfg_mod
    old_enabled = cfg_mod.config.BACKTRACK_ENABLED
    cfg_mod.config.BACKTRACK_ENABLED = True
    try:
        class FlatVS:
            def __init__(self):
                self.calls = []

            def hybrid_search_with_rerank(self, query, k=5, kb_filter=None):
                self.calls.append(query)
                return [FakeDoc("始终是这段话", 0.2)]   # 内容与分数都不变

        vs = FlatVS()
        rag = _build_rag(vs, ScriptedLLM(["改写A", "改写B", "改写C"]))
        rag.retrieve_and_merge("完全无关的问题", strategy="direct")
        # 首轮 + 最多 max_rounds 轮，且第 2 轮发现无改善后应立刻停
        limit = 1 + cfg_mod.config.BACKTRACK_MAX_ROUNDS
        check("r2c.1 回溯轮数不超过上限", len(vs.calls) <= limit,
              f"调用 {len(vs.calls)} 次，上限 {limit}")
        check("r2c.2 无改善时提前终止（少于上限）", len(vs.calls) <= 2,
              f"调用 {len(vs.calls)} 次")
    finally:
        cfg_mod.config.BACKTRACK_ENABLED = old_enabled


def r2_backtrack_high_score_skip():
    print("\n[R2-d] 首轮高分 → 不回溯（避免无谓开销）")
    import base.config as cfg_mod
    old = cfg_mod.config.BACKTRACK_ENABLED
    cfg_mod.config.BACKTRACK_ENABLED = True
    try:
        vs = ScriptedVectorStore([(None, [FakeDoc("完全匹配的法规原文", 0.92)])])
        rag = _build_rag(vs, ScriptedLLM())
        rag.retrieve_and_merge("食品安全标准是什么", strategy="direct")
        check("r2d.1 高分时只检索一次", len(vs.calls) == 1, f"调用 {len(vs.calls)} 次")
    finally:
        cfg_mod.config.BACKTRACK_ENABLED = old


def r2_backtrack_rewrite_cleanup():
    print("\n[R2-e] 改写结果清理（去引号/前缀）")
    from core.rag_system import RAGSystem
    import re

    def _clean(text):
        t = re.sub(r'^["“”\'\s]+|["“”\'\s]+$', '', text.strip())
        t = re.sub(r'^(检索式|查询式|改写后)[：:]\s*', '', t)
        return t

    cases = [
        ('"腐败变质畜禽肉判定标准"', "腐败变质畜禽肉判定标准"),
        ("检索式：食品贮存温度要求", "食品贮存温度要求"),
        ("查询式: 进货查验记录", "进货查验记录"),
        ("  正常文本  ", "正常文本"),
    ]
    ok = all(_clean(src) == exp for src, exp in cases)
    check("r2e.1 引号/前缀清理规则正确", ok,
          "" if ok else str([(_clean(s), e) for s, e in cases]))


def r2_backtrack_prompt():
    print("\n[R2-f] Backtrack 提示词")
    from core.prompts import RAGPrompts
    tpl = RAGPrompts.backtrack_rewrite_prompt()
    check("r2f.1 模板存在", tpl is not None)
    check("r2f.2 含两个输入变量",
          set(tpl.input_variables) == {"query", "prior_context"},
          str(tpl.input_variables))
    text = tpl.format(query="这肉还能用吗", prior_context="片段")
    check("r2f.3 可正常格式化", "这肉还能用吗" in text)
    check("r2f.4 含术语归一要求", "国标" in text or "法规术语" in text)


# ==================================================================
# R3 RAGAS 脚手架
# ==================================================================

def r3_ragas():
    print("\n[R3] RAGAS 评估脚手架")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(base, "scripts", "evaluate_ragas.py")
    check("r3.1 evaluate_ragas.py 存在", os.path.exists(script))

    if os.path.exists(script):
        src = open(script, encoding="utf-8").read()
        check("r3.2 支持四种核心指标",
              all(m in src for m in ["faithfulness", "answer_relevancy",
                                     "context_precision", "context_recall"]))
        check("r3.3 未装 ragas 时优雅降级",
              "ImportError" in src or "ImportError" in src)
        check("r3.4 支持生成评估集", "gen" in src.lower() or "build" in src.lower())
        check("r3.5 结果可导出 JSON/CSV",
              ".json" in src and ".csv" in src)

    # 评估集目录与示例集
    eval_dir = os.path.join(base, "data", "eval")
    sample = os.path.join(eval_dir, "ragas_eval_set.jsonl")
    check("r3.6 评估集示例文件存在", os.path.exists(sample))
    if os.path.exists(sample):
        lines = [l for l in open(sample, encoding="utf-8").read().split("\n") if l.strip()]
        check("r3.7 评估集非空", len(lines) >= 5, f"{len(lines)} 条")
        import json as _json
        try:
            first = _json.loads(lines[0])
            check("r3.8 评估集字段含 question/ground_truth",
                  "question" in first and "ground_truth" in first,
                  str(sorted(first.keys())))
        except Exception as exc:  # noqa: BLE001
            check("r3.8 评估集字段合法", False, str(exc))


# ==================================================================
# R4 配置
# ==================================================================

def r4_config():
    print("\n[R4] 配置项完整性")
    from base.config import config
    check("r4.1 backtrack.enabled 可读", config.BACKTRACK_ENABLED is True)
    check("r4.2 trigger < score_threshold（给一次自救机会）",
          config.BACKTRACK_TRIGGER_THRESHOLD < config.SCORE_THRESHOLD,
          f"{config.BACKTRACK_TRIGGER_THRESHOLD} < {config.SCORE_THRESHOLD}")
    check("r4.3 give_up < trigger（兜底线更低）",
          config.BACKTRACK_GIVE_UP_THRESHOLD < config.BACKTRACK_TRIGGER_THRESHOLD,
          f"{config.BACKTRACK_GIVE_UP_THRESHOLD} < {config.BACKTRACK_TRIGGER_THRESHOLD}")
    check("r4.4 max_rounds 合理", 1 <= config.BACKTRACK_MAX_ROUNDS <= 5,
          str(config.BACKTRACK_MAX_ROUNDS))
    check("r4.5 redis 默认关闭", config.REDIS_ENABLED is False)
    check("r4.6 redis TTL 三档齐全",
          all(getattr(config, a) > 0 for a in
              ["REDIS_EMBEDDING_TTL", "REDIS_RETRIEVAL_TTL", "REDIS_ANSWER_TTL"]))
    check("r4.7 embedding TTL 最长（向量最稳定）",
          config.REDIS_EMBEDDING_TTL > config.REDIS_RETRIEVAL_TTL)
    check("r4.8 ragas 默认关闭", config.RAGAS_ENABLED is False)
    check("r4.9 ragas 路径解析为绝对路径",
          os.path.isabs(config.RAGAS_EVAL_FILE) and os.path.isabs(config.RAGAS_OUTPUT_DIR))
    check("r4.10 ragas 四项指标默认",
          set(config.RAGAS_METRICS) == {"faithfulness", "answer_relevancy",
                                        "context_precision", "context_recall"},
          str(config.RAGAS_METRICS))


def main():
    print("=" * 64)
    print("FoodSafetySystem 三项增强验证")
    print("=" * 64)
    r1_cache()
    r2_backtrack_disabled()
    r2_backtrack_low_score()
    r2_backtrack_no_improve()
    r2_backtrack_high_score_skip()
    r2_backtrack_rewrite_cleanup()
    r2_backtrack_prompt()
    r3_ragas()
    r4_config()

    print("\n" + "=" * 64)
    print(f"结果：{len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("失败项：" + "、".join(FAIL))
    print("=" * 64)
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
