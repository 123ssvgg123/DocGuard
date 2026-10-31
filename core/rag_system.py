"""
    RAG 问答核心：core/rag_system.py

    职责：把「检索 → 重排 → 提示词组装 → LLM 生成」串成完整闭环。

    检索策略（对应需求基线的 4 种策略，四种全部实现）：
        - 直接检索：标准问题，直接检索知识库
        - HyDE    ：口语化/模糊提问，先生成假设解答再检索
        - SubQuery：复合问题，拆成子问题分别检索再合并
        - Backtrack：首轮检索分数偏低时，把提问改写为法规条文术语再检索一轮
                     （解决「用户口语表述」与「国标正式用语」之间的语义鸿沟）

    策略选择由「形态判断（口语化/复合）+ 意图分类（模型优先，缺失自动降级为规则）」共同决定；
    Backtrack 不作为路由目标，而是叠加在任意策略之上的一层**补救机制**：
    当选定策略检索回来的最高分低于阈值时自动触发。

    缓存层（core/cache.py）：检索结果与答案均可缓存，Redis 不可用时自动降级为进程内缓存。
"""
import re
import time

from base.config import config
from base.llm_guard import LLMUnavailable, breaker as llm_breaker, is_fallback_text
from base.logger import logger
from core.cache import rag_cache
from core.prompts import RAGPrompts
from core.intent_classifier import build_intent_classifier, INTENT_TO_STRATEGY
# 行级权限：检索结果出口二次校验（需求 4.2）
from core.access_control import filter_docs_by_acl


class RAGSystem:
    """封装 RAG 系统的核心逻辑"""

    def __init__(self, vector_store, llm):
        # 向量库对象
        self.vector_store = vector_store
        # 大模型调用函数（可调用对象，接收 prompt 返回 token 生成器）
        self.llm = llm
        # RAG 主提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 提示词最大长度
        self.max_prompt_length = config.MAX_PROMPT_LENGTH
        # 意图分类器（优先加载 models/intent-classifier/，不存在则用规则）
        self.intent_classifier = build_intent_classifier()

    # ============ LLM 文本获取（带守卫） ============

    def _llm_text(self, prompt, purpose=""):
        """
        取一段 LLM 文本输出，失败时抛 LLMUnavailable 而不是返回降级话术。

        为什么不能让降级话术流下去（2026-10-30 事故）：
            call_llm 出错时会 yield「抱歉，服务暂时不可用…」。HyDE / SubQuery /
            Backtrack 三处都是 ``"".join(self.llm(...))`` 直接拿去当检索式，
            于是这句道歉话被切成子查询去做向量检索，必然空召回 → 分数低 →
            再触发 Backtrack → 再打一次 LLM → 再失败。一次余额不足被放大成
            整条链路的无效重试。

        三道防线：
            1. 熔断中直接抛，不发请求（``llm_breaker.check``）
            2. 输出命中兜底标记 → 判定不可用并熔断
            3. 空输出同样视为不可用
        """
        where = purpose or "LLM 文本生成"
        llm_breaker.check(where)
        try:
            text = "".join(self.llm(prompt))
        except LLMUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 —— 生成器内部异常统一转成不可用
            llm_breaker.trip(f"{where} 异常：{exc}")
            raise LLMUnavailable(f"{where} 调用失败：{exc}") from exc

        if is_fallback_text(text):
            llm_breaker.trip(f"{where} 返回降级话术")
            raise LLMUnavailable(f"{where} 返回降级话术，判定 LLM 不可用")
        llm_breaker.record_success()
        return text

    # ============ 三种检索策略 ============

    def _retrieve_direct(self, query, kb_filter=None, acl=None):
        """直接检索：标准问题直接走向量检索"""
        logger.info(f"使用直接检索策略 (查询: '{query}')")
        return self.vector_store.hybrid_search_with_rerank(
            query, k=config.RETRIEVAL_K, kb_filter=kb_filter, acl=acl
        )

    def _retrieve_with_hyde(self, query, kb_filter=None, acl=None):
        """HyDE 检索：先生成假设解答，用假设解答的语义去检索"""
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        try:
            hyde_template = RAGPrompts.hyde_prompt()
            hypo_answer = self._llm_text(
                hyde_template.format(query=query), purpose="HyDE 假设解答")
            logger.info(f"HyDE 生成的假设解答: '{hypo_answer[:100]}...'")
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=config.RETRIEVAL_K, kb_filter=kb_filter, acl=acl
            )
        except LLMUnavailable as e:
            logger.warning(f"HyDE 不可用（{e}），本次直接走检索，不再重试")
            return self._retrieve_direct(query, kb_filter, acl=acl)
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}，回退到直接检索")
            return self._retrieve_direct(query, kb_filter, acl=acl)

    def _retrieve_with_subqueries(self, query, kb_filter=None, acl=None):
        """SubQuery 检索：拆成子问题分别检索，再按内容去重合并"""
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        try:
            subquery_template = RAGPrompts.subquery_prompt()
            subqueries_text = self._llm_text(
                subquery_template.format(query=query), purpose="SubQuery 子查询拆分")
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询，回退到直接检索")
                return self._retrieve_direct(query, kb_filter, acl=acl)

            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=config.RETRIEVAL_K, kb_filter=kb_filter, acl=acl
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个父块")

            # 基于内容去重
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
            logger.info(f"子查询合并去重后共 {len(unique_docs)} 个父块")
            return unique_docs
        except LLMUnavailable as e:
            logger.warning(f"SubQuery 不可用（{e}），本次直接走检索，不再重试")
            return self._retrieve_direct(query, kb_filter, acl=acl)
        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}，回退到直接检索")
            return self._retrieve_direct(query, kb_filter, acl=acl)

    def select_strategy(self, query):
        """
        策略路由：判定走哪种检索策略。

        路由优先级（自上而下，先命中先返回）：
            1. 口语化/模糊表达  → HyDE      （"这肉看着有点怪还能用吗"）
            2. 明显复合/多问句  → SubQuery  （"开火锅店要满足哪些条件、还要办什么证"）
            3. 意图分类结果     → 按 INTENT_TO_STRATEGY 映射
            4. 无把握           → 直接检索

        为什么前两条要放在意图分类之前：
            意图分类回答的是"问的是哪个业务领域"，回答不了"这句话有多模糊、有几个诉求"。
            "这肉看着有点怪还能用吗"的意图是「SOP 指导」，但它真正需要的是 HyDE；
            而"顾客中毒了要怎么处理、要上报谁"的意图是「事故处置」，需要的是 SubQuery。
            所以形态判断（口语化/复合）先于领域判断（意图）。

        说明：并列判断只用「且、以及、分别、同时」这类明确的并列连词。
            不要把「和 / 与」作为特征词——它们在中文里同时表示"和、跟、同"之外的
            高频含义（如"和面""与生俱来"），一旦用子串匹配会让几乎所有问句都误判为复合问题。

        注：Backtrack（第四种策略）不在此路由 —— 它是叠加在任意策略之上的补救层，
            由 retrieve_and_merge → _maybe_backtrack 在分数不足时自动触发。
        """
        # 口语化、模糊表达的特征词
        colloquial_markers = ['有点', '怪', '感觉', '好像', '能不能用', '还能用', '能不能吃',
                              '这样', '那个', '怎么回事', '啥', '咋']
        # 复合问题的特征：明确并列或连续问句
        composite_markers = ['哪些条件', '需要满足', '分别', '以及', '同时', '还要', '并且']
        question_marks = len(re.findall(r'[？?]', query))

        if any(m in query for m in colloquial_markers):
            return 'hyde'
        if question_marks >= 2 or any(m in query for m in composite_markers):
            return 'subquery'

        # 意图分类（模型优先，缺失时自动用规则）
        intent = self.intent_classifier.predict(query)
        strategy = INTENT_TO_STRATEGY.get(intent, 'direct')
        logger.info(f"意图识别结果：{intent or '未识别'}（分类器：{self.intent_classifier.name}）"
                    f"→ 检索策略：{strategy}")
        return strategy

    def retrieve_and_merge(self, query, kb_filter=None, strategy=None, acl=None):
        """按策略检索并返回最终用于组装提示词的父块列表（含行级权限过滤）"""
        if not strategy:
            strategy = self.select_strategy(query)
        logger.info(f"检索策略选择结果: {strategy} (查询: '{query}', "
                    f"角色: {acl.role if acl else 'off'})")

        # ---- L2 缓存：检索结果（同一问题反复检索直接命中，省 Milvus + Rerank 开销）----
        # ⚠️ 缓存键必须带权限维度（需求 4.2 第 ③ 条），否则跨角色会命中彼此的含密结果。
        acl_key = acl.cache_key if acl is not None else "acl-off"
        cache_key = rag_cache.key_retrieval(query, kb_filter or "", strategy, acl_key=acl_key)
        cached = rag_cache.get(cache_key, kind="retrieval")
        if cached is not None:
            from core.cache import _restore_docs
            docs = _restore_docs(cached)
            logger.info(f"[缓存] 检索结果命中（{len(docs)} 个父块）")
            # 缓存出口再校验一次：旧版本写入的缓存条目里没有权限维度
            return filter_docs_by_acl(docs, acl)[:config.CANDIDATE_M]

        if strategy == 'hyde':
            docs = self._retrieve_with_hyde(query, kb_filter, acl=acl)
        elif strategy == 'subquery':
            docs = self._retrieve_with_subqueries(query, kb_filter, acl=acl)
        else:
            docs = self._retrieve_direct(query, kb_filter, acl=acl)

        # ---- Backtrack 回溯：分数不够先自救一轮，而不是直接拒答 ----
        docs = self._maybe_backtrack(query, docs, kb_filter, strategy, acl=acl)

        # 出口再校验一次权限，然后统一截断到 candidate_m 个
        docs = filter_docs_by_acl(docs, acl)
        final_docs = docs[:config.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_docs)} 个父块作为上下文")

        # 写入缓存（仅缓存「分数达标」的结果，避免把失败检索固化下来）
        if final_docs and self._best_score(final_docs) >= config.SCORE_THRESHOLD:
            from core.cache import _snapshot_docs
            try:
                rag_cache.set(cache_key, _snapshot_docs(final_docs),
                              config.REDIS_RETRIEVAL_TTL)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[缓存] 检索结果写入失败: {exc}")

        return final_docs

    # ============ Backtrack 回溯检索策略 ============

    @staticmethod
    def _best_score(docs):
        """取文档列表中的最高 rerank 分数"""
        if not docs:
            return 0.0
        return max(doc.metadata.get('rerank_score', 0.0) for doc in docs)

    def _maybe_backtrack(self, query, docs, kb_filter, strategy, acl=None):
        """
        Backtrack 回溯检索（第四种检索策略）。

        触发条件：当前最高 rerank 分数 < BACKTRACK_TRIGGER_THRESHOLD
        执行动作：让 LLM 把口语提问改写为「法规条文术语」的检索式，再检索一轮；
                 把两轮结果按分数合并去重，取更优者。

        为什么需要它：
            RAG 的典型失败模式不是"检索不到"，而是"检索到了沾边但不匹配的条文"——
            用户问"这肉看着有点怪还能用吗"，知识库里存的是"腐败变质畜禽肉的判定
            与处置要求"。两者语义距离远，向量检索容易召回到无关内容，进而触发拒答。
            回溯一轮把口语翻译成术语，能从根上解决这类表述差异。

        终止条件（任一满足即停）：
            ① 分数已达 SCORE_THRESHOLD
            ② 已达 MAX_ROUNDS 上限
            ③ 改写没带来任何新增文档
            ④ 分数低于 GIVE_UP_THRESHOLD（判定回溯无效，交给上层拒答）
        """
        if not config.BACKTRACK_ENABLED:
            return docs

        # LLM 已熔断时回溯没有意义：改写是纯 LLM 动作，只会再多打几次注定失败的请求
        if llm_breaker.is_open:
            logger.warning(f"[Backtrack] LLM 已熔断（{llm_breaker.reason}），跳过回溯")
            return docs

        best = self._best_score(docs)
        if best >= config.SCORE_THRESHOLD:
            return docs  # 首轮就达标，无需回溯

        if best >= config.BACKTRACK_TRIGGER_THRESHOLD:
            # 分数在 [trigger, score_threshold) 之间：接近达标，但仍给一次机会
            logger.info(f"[Backtrack] 首轮最高分 {best:.4f} < 阈值 "
                        f"{config.SCORE_THRESHOLD}，触发回溯改写")

        merged = list(docs or [])
        current_query = query

        for round_no in range(1, config.BACKTRACK_MAX_ROUNDS + 1):
            try:
                # 把上一轮的低分片段作为改写参考（提升改写质量）
                prior_context = ""
                if config.BACKTRACK_USE_PRIOR_DOCS and merged:
                    prior_context = "\n".join(
                        doc.page_content[:150] for doc in merged[:3])

                rewrite_tpl = RAGPrompts.backtrack_rewrite_prompt()
                rewritten = self._llm_text(
                    rewrite_tpl.format(query=current_query, prior_context=prior_context),
                    purpose="Backtrack 检索式改写",
                ).strip().replace("\n", " ")

                # 清理 LLM 可能带出的引号/前缀
                rewritten = re.sub(r'^["“”\'\s]+|["“”\'\s]+$', '', rewritten)
                rewritten = re.sub(r'^(检索式|查询式|改写后)[：:]\s*', '', rewritten)

                if not rewritten or rewritten == current_query:
                    logger.info("[Backtrack] 改写结果无效或与原文相同，终止回溯")
                    break

                logger.info(f"[Backtrack] 第 {round_no} 轮改写："
                            f"'{current_query}' → '{rewritten}'")

                new_docs = self.vector_store.hybrid_search_with_rerank(
                    rewritten, k=config.RETRIEVAL_K, kb_filter=kb_filter, acl=acl
                )
                logger.info(f"[Backtrack] 第 {round_no} 轮检索到 {len(new_docs)} 个父块")

                # 按内容去重合并（保留更高分的版本）
                by_content = {}
                for doc in [*merged, *new_docs]:
                    key = doc.page_content
                    old = by_content.get(key)
                    if old is None or doc.metadata.get('rerank_score', 0) > \
                            old.metadata.get('rerank_score', 0):
                        by_content[key] = doc
                before = len(merged)
                merged = sorted(by_content.values(),
                                key=lambda d: d.metadata.get('rerank_score', 0.0),
                                reverse=True)[:config.BACKTRACK_MERGE_K]

                new_best = self._best_score(merged)
                logger.info(f"[Backtrack] 合并后 {before} → {len(merged)} 个父块，"
                            f"最高分 {best:.4f} → {new_best:.4f}")

                if new_best >= config.SCORE_THRESHOLD:
                    logger.info(f"[Backtrack] 第 {round_no} 轮回溯成功，分数达标")
                    return merged

                if len(merged) == before and new_best <= best:
                    logger.info("[Backtrack] 回溯未带来改善，终止")
                    break

                best = new_best
                current_query = rewritten
            except LLMUnavailable as exc:
                # LLM 不可用 → 立刻停，不要进下一轮再打一次注定失败的请求
                logger.warning(f"[Backtrack] 第 {round_no} 轮改写不可用（{exc}），终止回溯")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[Backtrack] 第 {round_no} 轮回溯失败: {exc}，返回已有结果")
                break

        if best < config.BACKTRACK_GIVE_UP_THRESHOLD:
            logger.warning(f"[Backtrack] 回溯后最高分 {best:.4f} 仍低于兜底线 "
                           f"{config.BACKTRACK_GIVE_UP_THRESHOLD}，将触发拒答")
        return merged

    # ============ 生成答案 ============

    @staticmethod
    def build_context(docs):
        """把检索到的父块拼成上下文字符串，并保留来源标注"""
        if not docs:
            return ""
        blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get('file_name') or doc.metadata.get('source') or '未知来源'
            blocks.append(f"[资料{i}｜来源：{source}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    def _check_relevance(self, docs):
        """阈值判断：全部父块 rerank 分数低于阈值，视为未命中，触发拒答"""
        if not docs:
            return False
        best = max(doc.metadata.get('rerank_score', 0.0) for doc in docs)
        logger.info(f"最高 rerank 分数：{best:.4f}，阈值：{config.SCORE_THRESHOLD}")
        return best >= config.SCORE_THRESHOLD

    def generate_answer(self, query, kb_filter=None, history=None, acl=None):
        """
        生成答案（流式），支持对话历史与行级权限过滤。
        :param acl: 调用方权限上下文（core.access_control.ACLContext）；
                    None 表示不做权限过滤（行为与改造前一致）
        :return: (token 生成器, 检索到的来源文档列表)
        """
        start_time = time.time()
        logger.info(f"generate_answer 开始处理查询: '{query}', 知识库过滤: {kb_filter}")

        # 校验并限制历史轮数
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-config.HISTORY_ROUNDS:]
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 构造历史上下文
        history_context = ""
        if history:
            history_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])

        # 检索（权限过滤在 vector_store 的召回阶段与 retrieve_and_merge 的出口各做一次）
        docs = self.retrieve_and_merge(query, kb_filter=kb_filter, acl=acl)
        retrieved_time = time.time() - start_time
        logger.info(f"检索耗时 {retrieved_time:.2f}s，命中 {len(docs)} 个父块")

        # 阈值未命中 → 拒答（幻觉控制）
        if not self._check_relevance(docs):
            logger.warning("检索结果相关性不足，触发拒答")
            refusal = (f"抱歉，知识库中暂无与该问题相关的权威依据，为避免给出不准确的信息，"
                       f"建议您联系品控部门核实，或拨打食品安全咨询热线：{config.SERVICE_PHONE}。")
            return iter([refusal]), []

        context = self.build_context(docs)
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=config.SERVICE_PHONE
        )
        # 超长截断
        if len(prompt_input) > self.max_prompt_length:
            logger.warning(f"提示长度 {len(prompt_input)} 超过 {self.max_prompt_length}，进行截断")
            prompt_input = prompt_input[:self.max_prompt_length]

        logger.info(f"提示词组装完成，长度 {len(prompt_input)}，准备调用 LLM")
        return self.llm(prompt_input), docs

    def query(self, query, kb_filter=None, history=None, acl=None):
        """对外统一入口：返回 (答案文本, 来源列表, 耗时)"""
        start = time.time()
        stream, docs = self.generate_answer(query, kb_filter=kb_filter, history=history, acl=acl)
        answer = "".join(stream)
        elapsed = time.time() - start
        sources = self.format_sources(docs)
        logger.info(f"问答完成，耗时 {elapsed:.2f}s")
        return answer, sources, elapsed

    @staticmethod
    def format_sources(docs):
        """把来源文档格式化为前端可展示的结构"""
        sources = []
        for doc in docs:
            sources.append({
                "file_name": doc.metadata.get('file_name', ''),
                "kb_type": doc.metadata.get('source', ''),
                "score": round(doc.metadata.get('rerank_score', 0.0), 4),
                "snippet": doc.page_content[:200]
            })
        return sources
