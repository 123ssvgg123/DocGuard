# -*- coding: utf-8 -*-
"""
运行策略模块：base/run_policy.py —— 远程模型「最小验证模式」

为什么要有这个模块
------------------
本项目的对话模型走 **DeepSeek 官方 API**（远程服务、按 token 计费），
**不是本地部署**；只有 embedding（bge-m3）与 rerank（bge-reranker-large）
是本地权重。历史事故：一次全量评估跑批（20 条 × 4 指标，faithfulness
还要按句拆开判）把账户额度跑空，之后所有生成请求直接 402。

因此本项目立一条硬约定：

    **每次运行程序都不做完整一轮全量流程，只执行最小限度的验证。**

并且这条约定必须由代码强制，不能只写在文档里靠人自觉。

三条硬限制
----------
1. 样本上限：单次运行最多 ``SMOKE_MAX_SAMPLES`` 条（默认 2）
2. 调用预算：单次运行最多 ``SMOKE_MAX_LLM_CALLS`` 次远程调用（默认 6）
3. 全量闸门：要跑全量，必须**同时**满足
   「环境变量 ``ALLOW_FULL_RUN=1``」+「命令行 ``--allow-full``」，缺一即拒绝

用法
----
::

    from base.run_policy import run_policy, RunPolicyError

    run_policy.print_banner()          # 启动时打印当前策略
    n = run_policy.clamp_samples(20)   # 20 → 2，并 warn 一次
    run_policy.LLM_BUDGET.spend()      # 每次远程调用前扣一次，超了抛异常

放开全量（仅限人工确认成本后，临时用）
--------------------------------------
::

    # Windows
    set ALLOW_FULL_RUN=1
    # Linux / macOS
    export ALLOW_FULL_RUN=1
"""
from __future__ import annotations

import os
from typing import Optional

# 本模块可能早于 base.config 被导入（例如入口脚本先 import run_policy），
# 因此自己再兜一次 .env：只补缺失的键，不覆盖已有环境变量（load_dotenv 默认行为）
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:  # 未装 python-dotenv 时退回纯环境变量，不影响默认策略
    pass

__all__ = [
    "run_policy",
    "RunPolicyError",
    "LLMBudget",
]


class RunPolicyError(RuntimeError):
    """违反运行策略（超样本上限 / 超调用预算 / 未授权全量）时抛出"""


def _env_int(name: str, default: int) -> int:
    """读整型环境变量，非法值回退默认（配置写错不应该把程序搞崩）"""
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str) -> bool:
    """读布尔环境变量：1/true/yes/on 视为真"""
    return (os.getenv(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


class LLMBudget:
    """单次运行的远程调用预算计数器（进程级单例，见文件末尾 LLM_BUDGET）"""

    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = limit if limit is not None else _env_int("SMOKE_MAX_LLM_CALLS", 6)
        self.used = 0
        self.tripped = False

    # ---- 状态 ----
    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    def reset(self, limit: Optional[int] = None) -> None:
        """重置计数（测试或人工放开全量时用）"""
        if limit is not None:
            self.limit = limit
        self.used = 0
        self.tripped = False

    # ---- 扣费 ----
    def spend(self, n: int = 1, *, where: str = "") -> int:
        """
        调用远程模型前扣一次预算；超限抛 RunPolicyError。

        设计成「先扣后调」而不是「调完再记」：一旦超限，请求根本发不出去，
        不会出现「已经花钱了才发现超预算」的情况。
        """
        if self.used + n > self.limit:
            self.tripped = True
            raise RunPolicyError(
                f"远程模型调用预算已用尽（上限 {self.limit} 次，已用 {self.used} 次）"
                f"{'，触发点：' + where if where else ''}。"
                "本模型为远程 API 按量计费，非本地部署，按运行策略禁止继续调用。"
                "如确需更多调用，请显式设置 ALLOW_FULL_RUN=1 并调整 SMOKE_MAX_LLM_CALLS。"
            )
        self.used += n
        return self.used

    def snapshot(self) -> dict:
        return {"limit": self.limit, "used": self.used,
                "remaining": self.remaining, "tripped": self.tripped}


class RunPolicy:
    """运行策略门面：集中读取环境变量，提供样本钳制与全量闸门"""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """重新读取环境变量（.env 改动后无需重启进程）"""
        #: 是否仅允许最小验证（默认 True；只有 ALLOW_FULL_RUN=1 才为 False）
        self.smoke_only: bool = not _env_flag("ALLOW_FULL_RUN")
        #: 单次运行样本上限
        self.max_samples: int = _env_int("SMOKE_MAX_SAMPLES", 2)
        #: 单次运行远程调用上限
        self.max_llm_calls: int = _env_int("SMOKE_MAX_LLM_CALLS", 6)
        self.llm = LLMBudget(self.max_llm_calls)

    # ---- 样本钳制 ----
    def clamp_samples(self, requested: int, *, stage: str = "", minimum: int = 1) -> int:
        """
        把请求的样本数压到策略允许的上限内。

        :param requested: 调用方想跑的条数（0 或负数视为「全部」）
        :param stage: 调用场景名，仅用于日志
        :return: 实际允许跑的条数
        """
        if not self.smoke_only:
            return requested
        want = requested if requested and requested > 0 else self.max_samples
        allowed = max(min(want, self.max_samples), minimum)
        if want != allowed:
            print(f"[run_policy] 最小验证模式：{stage or '本次运行'} 请求 {want} 条 → "
                  f"压到 {allowed} 条（上限 SMOKE_MAX_SAMPLES={self.max_samples}）")
        return allowed

    # ---- 全量闸门 ----
    def require_full_permission(self, cli_flag: bool, *, stage: str = "") -> bool:
        """
        校验是否允许全量运行。

        :param cli_flag: 命令行是否显式传了 --allow-full
        :return: True 表示允许全量
        :raises RunPolicyError: 传了 --allow-full 但环境变量没放开
        """
        if not cli_flag:
            return False
        if self.smoke_only:
            raise RunPolicyError(
                f"{stage or '全量运行'}被拒绝：命令行传了 --allow-full，"
                "但环境变量 ALLOW_FULL_RUN 未设为 1。"
                "本模型为远程 API（非本地部署），全量跑批会产生不可控费用，"
                "必须两道闸门同时打开。"
            )
        return True

    # ---- 展示 ----
    def banner(self) -> str:
        """返回策略说明（多行文本），供入口打印"""
        mode = "最小验证（默认）" if self.smoke_only else "全量（已人工放开）"
        lines = [
            "=" * 64,
            "运行策略：远程模型最小验证模式",
            f"  模式        : {mode}",
            f"  样本上限    : {self.max_samples} 条/次",
            f"  调用预算    : {self.max_llm_calls} 次/次运行（已用 {self.llm.used}）",
            "  说明        : 对话模型为 DeepSeek 远程 API（按量计费），非本地部署，",
            "                因此默认不跑完整一轮流程，只做连通性与正确性抽验。",
            "  放开全量    : ALLOW_FULL_RUN=1 + 命令行 --allow-full",
            "=" * 64,
        ]
        return "\n".join(lines)

    def print_banner(self) -> None:
        print(self.banner())

    def snapshot(self) -> dict:
        return {
            "smoke_only": self.smoke_only,
            "max_samples": self.max_samples,
            "max_llm_calls": self.max_llm_calls,
            "llm_budget": self.llm.snapshot(),
        }


#: 进程级单例：所有模块共享同一个预算计数器
run_policy = RunPolicy()

#: 预算对象的快捷别名（供调用最频繁的 LLM 链路直接使用）
LLM_BUDGET = run_policy.llm
