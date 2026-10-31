# -*- coding: utf-8 -*-
"""
LLM 可用性守卫：base/llm_guard.py

为什么要有这个模块
------------------
2026-10-30 事故复盘：账户余额不足后，`QASystem.call_llm` 把异常吞掉并
yield 一句「抱歉，服务暂时不可用…」的兜底话术。上游 `_retrieve_with_subqueries`
没做校验，直接把这段话当"子查询"去做向量检索：

    _retrieve_with_subqueries - 76 - 生成的子查询: ['抱歉，服务暂时不可用，请稍后重试…']
    hybrid_search_with_rerank - 293 - query:抱歉,服务...

结果是一条必定空召回的无效检索 → 分数低 → 触发 Backtrack 改写 → 再打一次 LLM
→ 再 402 → 再生一句兜底话术。**一次余额不足被放大成整条链路的无效重试**，
单轮评估的请求数从预期的 20 次涨到 48 次，叠加 6 轮参数扫描后累计上百次。

本模块只做两件事
----------------
1. **识别兜底话术**：`is_fallback_text()` 判断一段"LLM 输出"其实是失败话术，
   不该被当成正常文本继续往下传。
2. **熔断**：同一次运行内，LLM 一旦确认不可用（余额不足 / 鉴权失败 / 连续异常），
   后续调用直接抛异常、**不再发 HTTP 请求**。

不负责的事：重试策略、降级文案、指标上报 —— 那些由各调用层自己管。

用法
----
::

    from base.llm_guard import LLMUnavailable, breaker, is_fallback_text

    breaker.check("检索改写")                 # 已熔断 → 立刻抛 LLMUnavailable
    text = "".join(llm(prompt))
    if is_fallback_text(text):
        breaker.trip("检索改写返回兜底话术")
        raise LLMUnavailable("...")
    breaker.record_success()
"""
from __future__ import annotations

from typing import List

__all__ = [
    "LLMUnavailable",
    "LLMBreaker",
    "breaker",
    "is_fallback_text",
    "FALLBACK_MARKERS",
    "AUTH_ERROR_CODES",
]

# 兜底话术的稳定片段。命中任意一条即判定「这不是模型真实输出」。
# 新增降级文案时请同步补充这里，否则又会漏检。
FALLBACK_MARKERS: List[str] = [
    "抱歉，服务暂时不可用",          # qa_system.call_llm 的异常兜底
    "本次运行已达最小验证的远程调用上限",  # run_policy 预算耗尽的兜底
]

# 这些 HTTP 状态码代表「账户/密钥层面不可用」，重试和切备用模型都无意义
AUTH_ERROR_CODES = frozenset({401, 402, 403})


def is_fallback_text(text: str) -> bool:
    """判断一段文本是否是降级兜底话术（而非模型真实输出）"""
    if not text:
        return True  # 空输出同样不能当有效结果用
    head = text[:120]
    return any(marker in head for marker in FALLBACK_MARKERS)


class LLMUnavailable(RuntimeError):
    """
    LLM 在本次运行内不可用。

    - 余额不足 / 鉴权失败 → 应当直接中止调用链，不要再试
    - 检索链路应当把它当作「这次改写拿不到结果」处理：回退 direct 检索或终止回溯
    """


class LLMBreaker:
    """
    单次运行内的熔断器。

    规则：
        · 任何一次 ``trip()`` 后立刻打开（fail-fast），因为绝大多数失败原因是
          账户级的，继续重试只会刷错误日志、白跑本地检索。
        · 打开后 ``check()`` 直接抛 ``LLMUnavailable``，不发请求。
        · ``record_success()`` 只清成功计数，不自动关闸 —— 避免抖动时反复试探。
          需要恢复请显式 ``reset()``（例如换了 key 之后）。
    """

    def __init__(self) -> None:
        self._open = False
        self._reason = ""
        self._blocked = 0  # 熔断期间被拦下的调用次数，便于收尾时汇报

    # ---------------- 状态查询 ----------------

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def blocked_count(self) -> int:
        """熔断打开后被拦下的调用次数（没发出去的请求）"""
        return self._blocked

    # ---------------- 动作 ----------------

    def check(self, where: str = "") -> None:
        """熔断中 → 抛 LLMUnavailable；正常 → 什么都不做"""
        if self._open:
            self._blocked += 1
            raise LLMUnavailable(
                f"LLM 已在本次运行内熔断（{self._reason}），跳过调用{f'：{where}' if where else ''}")

    def trip(self, reason: str) -> None:
        """标记 LLM 不可用。幂等：已经打开时只保留首次原因。"""
        if not self._open:
            self._open = True
            self._reason = reason

    def record_success(self) -> None:
        """一次成功调用。不清熔断状态，只作为健康信号留痕。"""
        return None

    def reset(self) -> None:
        """显式恢复（换 key / 充值之后用）"""
        self._open = False
        self._reason = ""
        self._blocked = 0

    def summary(self) -> str:
        """给收尾报告用的一句话结论"""
        if not self._open:
            return "LLM 调用正常，未触发熔断"
        return f"LLM 已熔断（{self._reason}），期间拦下 {self._blocked} 次无用调用"


# 进程级单例：一次运行共用一个熔断状态
breaker = LLMBreaker()
