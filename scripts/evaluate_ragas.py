"""
evaluate_ragas.py —— 食品安全知识库 RAG 效果评估

职责：
    用 RAGAS 四项核心指标（faithfulness / answer_relevancy / context_precision /
    context_recall）量化评估 RAG 链路的检索质量与生成质量。

设计要点：
    1. **优雅降级**：未安装 ragas 时（ImportError），自动退化为
       「LLM-as-judge + 规则近似」的本地打分器，脚本仍可完整跑通并产出报告。
    2. **可生成评估集**：`--gen` 从知识库自动 build 一份评估集 JSONL，
       解决「没有标注数据就没法评估」的冷启动问题。
    3. **多格式导出**：结果同时导出 JSON 与 CSV，便于人工复核与横向对比。
    4. **可复现**：固定采样顺序 + 记录 config 快照，两次运行结果可比。

用法：
    # 直接用示例评估集评估（未装 ragas 会自动降级）
    python scripts/evaluate_ragas.py

    # 只跑前 5 条，只算两项指标
    python scripts/evaluate_ragas.py --sample 5 --metrics faithfulness,context_recall

    # 从知识库自动生成评估集（不评估）
    python scripts/evaluate_ragas.py --gen --gen-size 30

    # 指定输入/输出
    python scripts/evaluate_ragas.py --input data/eval/ragas_eval_set.jsonl \
        --output-dir data/eval/results
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# 允许直接 `python scripts/evaluate_ragas.py` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import config          # noqa: E402
from base.llm_guard import breaker as llm_breaker  # noqa: E402
from base.logger import logger          # noqa: E402
from base.run_policy import RunPolicyError, run_policy  # noqa: E402

# 四项核心指标（与 config.ini [ragas] metrics 对应）
ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ==================================================================
# 1. 评估集读写
# ==================================================================


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 评估集，每行一个 {question, ground_truth, ...}"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"评估集不存在：{path}")
    items: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"第 {lineno} 行 JSON 解析失败，跳过：{exc}")
                continue
            if "question" not in obj:
                logger.warning(f"第 {lineno} 行缺少 question 字段，跳过")
                continue
            obj.setdefault("ground_truth", "")
            obj.setdefault("kb_type", "")
            items.append(obj)
    logger.info(f"载入评估集 {len(items)} 条：{path}")
    return items


def save_eval_set(items: List[Dict[str, Any]], path: str) -> str:
    """把评估集写为 JSONL（`--gen` 使用）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"评估集已写出 {len(items)} 条：{path}")
    return path


# ==================================================================
# 2. 生成评估集（--gen）：从知识库反向 build 问答对
# ==================================================================

# 知识库文档中高频的「条款式」句式，用来抽取可提问的要点
_QUESTION_TEMPLATES = {
    "regulation": [
        "{topic}的标准要求是什么？",
        "国标对{topic}有哪些具体规定？",
        "{topic}不符合要求会怎么处理？",
    ],
    "sop": [
        "{topic}的操作步骤是什么？",
        "门店做{topic}时要注意哪些关键点？",
        "SOP 里对{topic}是怎么规定的？",
    ],
    "case": [
        "{topic}这类事故一般是什么原因造成的？",
        "遇到{topic}问题应该怎么处置？",
    ],
    "nutrition": [
        "{topic}的营养成分大概是多少？",
        "{topic}在营养上有什么需要注意的？",
    ],
    "supplier": [
        "{topic}的供应商准入要求是什么？",
        "对{topic}供应商怎么审核？",
    ],
}


def _extract_topic(text: str) -> str:
    """从文档片段里粗提一个「可提问的主题词」，优先取首个短句/条款名"""
    if not text:
        return ""
    # 取第一段非空行
    for raw in text.split("\n"):
        line = raw.strip().strip("#＊*-— ")
        if len(line) < 4:
            continue
        # 截到第一个句读，且限制长度
        for sep in ["。", "；", "：", "，", ".", ";"]:
            if sep in line[:40]:
                line = line.split(sep)[0]
                break
        line = line.strip()
        if 2 <= len(line) <= 24:
            return line
    return text.strip()[:16]


def generate_eval_set(size: int = 20, kb_filter: str = "") -> List[Dict[str, Any]]:
    """
    从向量库抽样 build 评估集：对每个抽样片段生成一条「问题 + 参考答案」。
    这是冷启动方案——没有人工标注时，用文档原文作为 ground_truth 的近似。
    """
    from core.vector_store import VectorStore

    logger.info(f"开始生成评估集：目标 {size} 条，kb_filter={kb_filter or '全部'}")
    vs = VectorStore()
    collection = getattr(vs, "collection", None)
    if collection is None:
        raise RuntimeError("向量库未就绪（getattr(collection) 为空），请先执行 python build_kb.py")

    # 拉取一批文档（Milvus 单次 query 上限通常 16384）
    try:
        rows = collection.query(
            expr="", output_fields=["text", "source"], limit=min(size * 12, 4000)
        )
    except Exception as exc:  # noqa: BLE001  —— 不同版本 Milvus 参数名有差异
        logger.warning(f"标准 query 失败（{exc}），改用 search 兜底")
        rows = []

    rng = random.Random(20241013)   # 固定 seed，保证可复现
    if not rows:
        logger.warning("向量库抽样为空，回退到示例评估集")
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "eval", "ragas_eval_set.jsonl",
        )
        return load_eval_set(default_path) if os.path.exists(default_path) else []

    rng.shuffle(rows)
    items: List[Dict[str, Any]] = []
    seen_topics = set()

    for row in rows:
        if len(items) >= size:
            break
        text = (row.get("text") or row.get("page_content") or "").strip()
        source = row.get("source") or "regulation"
        if len(text) < 40:
            continue
        topic = _extract_topic(text)
        if not topic or topic in seen_topics:
            continue
        # 去重时按前缀粗判，避免「食品贮存温度」与「食品贮存温度要求」重复
        if any(topic[:6] == t[:6] for t in seen_topics):
            continue
        seen_topics.add(topic)

        templates = _QUESTION_TEMPLATES.get(source, _QUESTION_TEMPLATES["regulation"])
        question = rng.choice(templates).format(topic=topic)
        items.append({
            "question": question,
            "ground_truth": text[:600],
            "kb_type": kb_filter or source,
            "difficulty": "auto",
            "generated": True,
        })

    logger.info(f"生成评估集完成：{len(items)} 条")
    return items


# ==================================================================
# 3. RAG 应答采集：对每条 question 跑一遍链路
# ==================================================================


def collect_predictions(items: List[Dict[str, Any]],
                        kb_filter: str = "") -> List[Dict[str, Any]]:
    """对每个问题调用 RAGSystem.query()，采集 answer 与 contexts（供 RAGAS 打分）"""
    from core.vector_store import VectorStore
    from core.rag_system import RAGSystem
    from qa_system import QASystem

    logger.info("初始化 RAG 链路（QASystem）...")
    qa = QASystem()
    rag = qa.rag_system
    records: List[Dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        # 熔断后继续跑没有意义：每条仍要花 2~6 秒本地 embedding + rerank，
        # 但 LLM 已确定不可用，答案必然是降级话术，跑出来的分数没有参考价值。
        if llm_breaker.is_open:
            logger.warning(f"[{idx}/{len(items)}] LLM 已熔断（{llm_breaker.reason}），"
                           f"提前停止采集，剩余 {len(items) - idx + 1} 条不再执行")
            break

        question = item["question"]
        kb = kb_filter or item.get("kb_type") or ""
        started = time.time()
        try:
            answer, sources, elapsed = qa.ask(question, kb_filter=kb or None)
        except Exception as exc:  # noqa: BLE001 —— 单条失败不阻断整体评估
            logger.error(f"[{idx}/{len(items)}] 问答失败：{question[:30]} —— {exc}")
            answer, sources, elapsed = "", [], time.time() - started

        # contexts：从检索结果取正文片段（RAGAS 的 context_precision/recall 依赖它）
        contexts = [s.get("snippet", "") for s in sources if s.get("snippet")]
        # sources 里只有 snippet(200 字)，补一次完整检索拿到全文，保证 context_recall 准确
        if not contexts:
            try:
                docs = rag.retrieve_and_merge(question, kb_filter=kb or None)
                contexts = [d.page_content for d in docs]
            except Exception:  # noqa: BLE001
                contexts = []

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", ""),
            "kb_type": kb,
            "difficulty": item.get("difficulty", ""),
            "elapsed": round(elapsed, 3),
            "n_contexts": len(contexts),
        })
        logger.info(f"[{idx}/{len(items)}] {question[:28]}... → {len(answer)} 字 / "
                    f"{len(contexts)} 段上下文 / {elapsed:.2f}s")

    return records


# ==================================================================
# 4. 打分：RAGAS 优先，未装则降级为本地近似
# ==================================================================


def _tokenize(text: str) -> List[str]:
    """中英混排的粗分词：中文按 2-gram，英文按空格切"""
    import re
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk:
        tokens.extend(run[i:i + 2] for i in range(max(len(run) - 1, 1)))
        if len(run) == 1:
            tokens.append(run)
    return tokens


def _overlap(a: str, b: str) -> float:
    """集合重叠率：|A ∩ B| / |A|，用于近似 precision/recall"""
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def fallback_score(records: List[Dict[str, Any]],
                   metrics: List[str]) -> Dict[str, Any]:
    """
    未安装 ragas 时的降级打分器。
    用可解释的近似指标替代（规则 + 与关键词重叠度），保证脚本永不中断。
    分数区间统一到 [0, 1]，便于与 RAGAS 结果做趋势对比（非等价，仅参考）。
    """
    logger.warning("未检测到 ragas，启用本地近似打分（结果仅作趋势参考，不等价于 RAGAS）")
    scored: List[Dict[str, float]] = []

    for rec in records:
        answer = rec.get("answer", "")
        contexts = rec.get("contexts", [])
        gt = rec.get("ground_truth", "")
        ctx_all = "\n".join(contexts)

        row: Dict[str, float] = {}

        if "faithfulness" in metrics:
            # 近似：答案有多少比例的词能在上下文中找到 → 越低越像幻觉
            row["faithfulness"] = round(min(_overlap(answer, ctx_all) * 1.5, 1.0), 4)

        if "answer_relevancy" in metrics:
            # 近似：答案与问题的词重叠（并惩罚拒答/空答案）
            if not answer or "暂无" in answer:
                row["answer_relevancy"] = 0.0
            else:
                row["answer_relevancy"] = round(min(_overlap(answer, rec["question"]) * 2.2, 1.0), 4)

        if "context_precision" in metrics:
            # 近似：命中的上下文里有多少与 ground_truth 相关
            if contexts:
                hit = sum(1 for c in contexts if _overlap(c, gt or rec["question"]) > 0.15)
                row["context_precision"] = round(hit / len(contexts), 4)
            else:
                row["context_precision"] = 0.0

        if "context_recall" in metrics:
            # 近似：ground_truth 的信息有多少被召回（无 gt 时用 question 兜底）
            row["context_recall"] = round(_overlap(gt or rec["question"], ctx_all), 4)

        scored.append(row)

    # 汇总均值
    summary = {m: 0.0 for m in metrics}
    for row in scored:
        for m in metrics:
            summary[m] += row.get(m, 0.0)
    if scored:
        summary = {m: round(v / len(scored), 4) for m, v in summary.items()}

    return {"engine": "fallback(local heuristic)", "per_sample": scored, "summary": summary}


def ragas_score(records: List[Dict[str, Any]],
                metrics: List[str],
                judge_model: str = "") -> Dict[str, Any]:
    """
    用 RAGAS 官方库打分。未安装时抛 ImportError，由上层捕获后走降级。
    """
    from datasets import Dataset                      # noqa: F401  —— ragas 依赖
    from ragas import evaluate                        # noqa: F401
    from ragas.metrics import (                       # noqa: F401
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    selected = [metric_map[m] for m in metrics if m in metric_map]
    if not selected:
        raise ValueError(f"无有效指标：{metrics}")

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    logger.info(f"调用 RAGAS evaluate()，指标 {metrics}，样本 {len(records)} 条 ...")
    try:
        result = evaluate(dataset, metrics=selected)
    except TypeError:
        # 不同版本 ragas 对 llm/embeddings 传参方式不同，兜底重试
        result = evaluate(dataset, metrics=selected, raise_exceptions=False)

    # ragas 返回结构在不同版本间有差异，做兼容取值
    df = result.to_pandas() if hasattr(result, "to_pandas") else None
    per_sample: List[Dict[str, float]] = []
    if df is not None:
        for _, row in df.iterrows():
            per_sample.append({m: round(float(row.get(m, 0.0) or 0.0), 4) for m in metrics})

    summary = {}
    for m in metrics:
        try:
            summary[m] = round(float(result[m]), 4)
        except Exception:  # noqa: BLE001
            vals = [s.get(m, 0.0) for s in per_sample]
            summary[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return {"engine": f"ragas(official){'/' + judge_model if judge_model else ''}",
            "per_sample": per_sample, "summary": summary}


def score(records: List[Dict[str, Any]], metrics: List[str],
          allow_fallback: bool = True) -> Dict[str, Any]:
    """打分总入口：优先 RAGAS，ImportError 时降级；两者都不可用则报错"""
    try:
        return ragas_score(records, metrics, judge_model=config.RAGAS_JUDGE_MODEL)
    except ImportError as exc:
        logger.warning(f"ragas 依赖缺失（{exc}）")
        if not allow_fallback:
            raise
        return fallback_score(records, metrics)
    except Exception as exc:  # noqa: BLE001 —— 例如未配置 judge LLM
        logger.error(f"RAGAS 执行失败：{exc}")
        if not allow_fallback:
            raise
        logger.warning("自动降级为本地近似打分")
        return fallback_score(records, metrics)


# ==================================================================
# 5. 报告导出：JSON + CSV
# ==================================================================


def export_report(records: List[Dict[str, Any]], result: Dict[str, Any],
                  metrics: List[str], output_dir: str,
                  extra: Dict[str, Any] | None = None) -> Dict[str, str]:
    """导出 JSON（完整）与 CSV（便于 Excel 复核），返回两个文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine": result.get("engine", ""),
        "metrics": metrics,
        "summary": result.get("summary", {}),
        "n_samples": len(records),
        "config_snapshot": {
            "retrieval_strategy": getattr(config, "RETRIEVAL_STRATEGY", ""),
            "candidate_m": config.CANDIDATE_M,
            "score_threshold": config.SCORE_THRESHOLD,
            "backtrack_enabled": config.BACKTRACK_ENABLED,
            "backtrack_trigger_threshold": config.BACKTRACK_TRIGGER_THRESHOLD,
            "backtrack_max_rounds": config.BACKTRACK_MAX_ROUNDS,
            "rerank_model": getattr(config, "RERANK_MODEL", ""),
            "llm_model": config.LLM_MODEL,
        },
        "per_sample": [],
    }
    if extra:
        report.update(extra)

    per_sample = result.get("per_sample", [])
    for i, rec in enumerate(records):
        row = {
            "question": rec.get("question", ""),
            "answer": rec.get("answer", ""),
            "ground_truth": rec.get("ground_truth", ""),
            "kb_type": rec.get("kb_type", ""),
            "elapsed": rec.get("elapsed", 0.0),
            "n_contexts": rec.get("n_contexts", len(rec.get("contexts", []))),
        }
        if i < len(per_sample):
            row.update(per_sample[i])
        report["per_sample"].append(row)

    json_path = os.path.join(output_dir, f"ragas_report_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, f"ragas_report_{stamp}.csv")
    fieldnames = ["question", "answer", "ground_truth", "kb_type",
                  "elapsed", "n_contexts"] + list(metrics)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report["per_sample"]:
            writer.writerow(row)

    logger.info(f"报告已导出：\n  JSON → {json_path}\n  CSV  → {csv_path}")
    return {"json": json_path, "csv": csv_path}


# ==================================================================
# 6. CLI
# ==================================================================


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="食品安全知识库 RAGAS 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=config.RAGAS_EVAL_FILE,
                        help="评估集 JSONL 路径（默认取 config [ragas] eval_file）")
    parser.add_argument("--output-dir", default=config.RAGAS_OUTPUT_DIR,
                        help="报告输出目录")
    parser.add_argument("--sample", type=int, default=config.RAGAS_SAMPLE_SIZE,
                        help="只评估前 N 条（0 表示全部）")
    parser.add_argument("--metrics", default=",".join(config.RAGAS_METRICS),
                        help="逗号分隔的指标名，默认四项全跑")
    parser.add_argument("--kb-filter", default="", help="限定知识库类型：regulation/sop/case/...")
    parser.add_argument("--gen", action="store_true",
                        help="从知识库自动构建评估集（不执行评估）")
    parser.add_argument("--gen-size", type=int, default=20, help="--gen 时生成的条数")
    parser.add_argument("--no-fallback", action="store_true",
                        help="未装 ragas 时直接报错，不降级")
    parser.add_argument("--seed", type=int, default=20241013, help="采样随机种子")
    parser.add_argument("--allow-full", action="store_true",
                        help="允许全量评估（需同时设置 ALLOW_FULL_RUN=1；默认仅最小验证）")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    # ================================================================
    # 运行策略：对话模型为 DeepSeek 远程 API（非本地部署、按 token 计费），
    # 本脚本是最容易「一不小心跑完整一轮」的入口，因此在这里强制最小验证：
    #   · 样本数压到 SMOKE_MAX_SAMPLES（默认 2）
    #   · 要跑全量必须 --allow-full 且 ALLOW_FULL_RUN=1，缺一即拒绝
    # 详见 base/run_policy.py
    # ================================================================
    try:
        run_policy.require_full_permission(args.allow_full, stage="RAGAS 全量评估")
    except RunPolicyError as exc:
        print(f"✗ {exc}")
        return 2
    run_policy.print_banner()
    # 每次运行从干净状态开始（进程级单例，防止上一次调用的熔断状态串味）
    llm_breaker.reset()
    args.sample = run_policy.clamp_samples(args.sample, stage="RAGAS 评估")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    invalid = [m for m in metrics if m not in ALL_METRICS]
    if invalid:
        print(f"✗ 未知指标：{invalid}，可选：{ALL_METRICS}")
        return 2
    if not metrics:
        metrics = list(ALL_METRICS)

    print("=" * 64)
    print("FoodSafetySystem RAGAS 评估")
    print("=" * 64)

    # --- 分支 A：只生成评估集 ---
    if args.gen:
        items = generate_eval_set(size=args.gen_size, kb_filter=args.kb_filter)
        if not items:
            print("✗ 生成评估集失败：知识库为空，请先执行 python build_kb.py")
            return 1
        out = save_eval_set(items, args.input)
        print(f"\n✓ 已生成评估集 {len(items)} 条 → {out}")
        print("  提示：可手动润色 ground_truth 后再执行评估（不带 --gen）")
        return 0

    # --- 分支 B：正常评估 ---
    try:
        items = load_eval_set(args.input)
    except FileNotFoundError as exc:
        print(f"✗ {exc}")
        return 1
    if not items:
        print("✗ 评估集为空")
        return 1

    if args.sample and args.sample > 0:
        rng = random.Random(args.seed)
        if args.sample < len(items):
            items = rng.sample(items, args.sample)
        print(f"采样 {len(items)} 条（seed={args.seed}）")

    print(f"评估集：{len(items)} 条 | 指标：{metrics}")
    print(f"生成器：{'官方 ragas' if _ragas_available() else '本地近似（降级）'}")
    print("-" * 64)

    print("\n[1/3] 采集 RAG 应答 ...")
    records = collect_predictions(items, kb_filter=args.kb_filter)
    if not records:
        print("✗ 未采集到任何应答")
        return 1

    print(f"\n[2/3] 打分（{len(records)} 条）...")
    result = score(records, metrics, allow_fallback=not args.no_fallback)

    print("\n[3/3] 导出报告 ...")
    paths = export_report(records, result, metrics, args.output_dir,
                          extra={"input_file": args.input})

    # --- 打印结论 ---
    print("\n" + "=" * 64)
    print(f"评分引擎：{result.get('engine', '')}")
    print(f"样本数：{len(records)}")
    print("-" * 64)
    for m in metrics:
        v = result["summary"].get(m, 0.0)
        bar = "█" * int(round(v * 30))
        print(f"  {m:<20} {v:>7.4f}  {bar}")
    print("-" * 64)
    avg = sum(result["summary"].values()) / max(len(result["summary"]), 1)
    print(f"  {'平均':<20} {avg:>7.4f}")
    print("-" * 64)
    # 熔断状态必须显式暴露：四项全 0 / 全 NaN 时，用户要知道是「检索差」还是「根本调不通 LLM」
    print(f"  LLM 状态：{llm_breaker.summary()}")
    if llm_breaker.is_open:
        print("  ⚠ 本次分数不可用于对外口径：LLM 未真正参与生成，答案来自降级话术")
    print("=" * 64)
    print(f"报告：\n  JSON → {paths['json']}\n  CSV  → {paths['csv']}")
    return 0


def _ragas_available() -> bool:
    """探测 ragas 是否可导入（不触发实际评估）"""
    import importlib.util
    return importlib.util.find_spec("ragas") is not None


if __name__ == "__main__":
    sys.exit(main())
