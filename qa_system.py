"""
    问答系统装配层：qa_system.py

    职责：把「LLM 客户端 + 向量库 + RAG 流程 + 会话历史 + 问候语短路」装配为对外的统一问答服务。
        这是唯一做装配的地方，业务逻辑都在 core/ 下，符合「入口只装配、逻辑下沉」的约定。
"""
import re

from openai import OpenAI

from base.config import config
from base.llm_guard import breaker as llm_breaker
from base.logger import logger
from base.run_policy import RunPolicyError, run_policy
from core.vector_store import VectorStore
from core.rag_system import RAGSystem

# 日常问候用语模式与回复：命中则直接返回，避免无意义的检索与 LLM 开销
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是食安通食品安全智能助手，可以帮你查法规标准、SOP 操作规范、原料判定和事故处置流程。请问有什么可以帮你？"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是食安通食品安全智能助手，基于集团食品安全知识库提供法规查询、SOP 指导和事故处置建议。"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗|在不)",
        "response": "我在！有什么食品安全问题可以直接问我，比如原料验收标准、冷藏温度要求、食品留样规范等。"
    },
]


class QASystem:
    """食品安全知识库问答系统"""

    def __init__(self):
        logger.info("初始化 QASystem ...")
        # 1. 初始化 LLM 客户端（OpenAI 兼容协议，指向 DeepSeek）
        self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        # 2. 初始化向量库
        self.vector_store = VectorStore()
        # 3. 初始化 RAG 系统
        self.rag_system = RAGSystem(self.vector_store, self.call_llm)
        # 4. 会话历史：{session_id: [{"question":.., "answer":..}, ...]}
        self.sessions = {}
        logger.info("QASystem 初始化完成")

    @staticmethod
    def check_greeting(query):
        """检查是否为日常问候用语，命中则返回模板回复，否则返回 None"""
        query_text = (query or "").strip()
        for item in GREETING_PATTERNS:
            if re.match(item["pattern"], query_text, re.IGNORECASE):
                return item["response"]
        return None

    def call_llm(self, prompt):
        """
        调用大模型，流式输出 token 生成器。
        内部统一处理异常：出错时返回兜底话术，避免上游拿到空结果。

        ⚠️ 运行策略（见 base/run_policy.py）：对话模型是 DeepSeek 远程 API（非本地部署、
        按 token 计费），因此每次调用前先扣「单次运行调用预算」，超限即停，
        保证一次运行不会演变成完整一轮全量流程。

        ⚠️ 熔断（见 base/llm_guard.py）：一旦确认 LLM 不可用（余额不足/鉴权失败），
        后续调用直接返回兜底话术、**不再发起 HTTP 请求**，避免整条链路反复撞 402。
        注意兜底话术本身不能再被上游当检索词用 —— 检索侧走 RAGSystem._llm_text()。
        """
        if llm_breaker.is_open:
            logger.error(f"LLM 已熔断（{llm_breaker.reason}），本次不再发起请求")
            yield f"抱歉，服务暂时不可用，请稍后重试或拨打咨询热线：{config.SERVICE_PHONE}。"
            return

        try:
            run_policy.llm.spend(where="QASystem.call_llm")
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是食品安全领域的专业助手，回答必须严谨、有依据。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
                stream=True
            )
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
                if token:
                    yield token
        except RunPolicyError as e:
            # 预算耗尽属于「策略预期内」的停止，不是故障：明确告知而不是伪装成服务异常
            logger.error(f"运行策略拦截：{e}")
            # 预算耗尽后继续调用没有意义，一并熔断，省下后续的本地检索开销
            llm_breaker.trip(f"预算耗尽：{e}")
            yield "（本次运行已达最小验证的远程调用上限，已停止继续生成。"
            yield "如需完整问答，请显式放开 ALLOW_FULL_RUN=1。）"
        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}")
            llm_breaker.trip(f"call_llm 失败：{e}")
            yield f"抱歉，服务暂时不可用，请稍后重试或拨打咨询热线：{config.SERVICE_PHONE}。"

    # ============ 会话历史管理 ============

    def get_history(self, session_id):
        """获取指定会话的历史消息"""
        return self.sessions.get(session_id, [])

    def clear_history(self, session_id):
        """清除指定会话的历史消息"""
        self.sessions.pop(session_id, None)
        logger.info(f"已清除会话历史: {session_id}")

    def _append_history(self, session_id, question, answer):
        """把本轮问答追加到会话历史，并限制轮数"""
        if session_id is None:
            return
        history = self.sessions.setdefault(session_id, [])
        history.append({"question": question, "answer": answer})
        # 只保留最近 N 轮
        if len(history) > config.HISTORY_ROUNDS:
            self.sessions[session_id] = history[-config.HISTORY_ROUNDS:]

    # ============ 对外问答入口 ============

    def ask(self, query, kb_filter=None, session_id=None, acl=None):
        """非流式问答，返回 (答案, 来源列表, 耗时)。acl 为行级权限上下文（可空）"""
        # 问候语短路：不检索、不调用 LLM
        greeting = self.check_greeting(query)
        if greeting:
            self._append_history(session_id, query, greeting)
            return greeting, [], 0.0

        history = self.get_history(session_id)
        answer, sources, elapsed = self.rag_system.query(
            query, kb_filter=kb_filter, history=history, acl=acl)
        self._append_history(session_id, query, answer)
        return answer, sources, elapsed

    def ask_stream(self, query, kb_filter=None, session_id=None, acl=None):
        """
        流式问答，逐 token 产出 (token, is_last, meta)。
        is_last 为 True 时 meta 中含完整来源列表，供前端展示引用。
        acl 为行级权限上下文（core.access_control.ACLContext），None 表示不过滤。
        """
        # 问候语短路：不检索、不调用 LLM
        greeting = self.check_greeting(query)
        if greeting:
            self._append_history(session_id, query, greeting)
            yield greeting, False, {}
            yield "", True, {"sources": []}
            return

        history = self.get_history(session_id)
        stream, docs = self.rag_system.generate_answer(
            query, kb_filter=kb_filter, history=history, acl=acl)

        answer = ""
        for token in stream:
            answer += token
            yield token, False, {}

        sources = self.rag_system.format_sources(docs)
        self._append_history(session_id, query, answer)
        yield "", True, {"sources": sources}
