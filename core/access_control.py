"""
    多用户内容隔离：core/access_control.py

    职责：实现需求 `4.2 多用户内容隔离` 中的「行级权限」部分，把「谁在问」翻译成
         一组可拼进 Milvus 过滤表达式的条件，并在检索结果出口做二次校验。

    需求给了 4 条隔离策略，本模块负责其中 3 条（第 1 条按项目既有取舍用 kb_type 语义替代）：

        ┌── 策略 ──────────────────────────────────────────────────────────────┐
        │ ① Collection 级物理隔离    → 本项目用「单集合 + kb_type 字段过滤」语义替代 │
        │ ② metadata 行级权限过滤     → 本模块 build_filter_expr()               │
        │ ③ 缓存 key 按角色/部门隔离   → 本模块 ACLContext.cache_key             │
        │ ④ FAQ visible_to 二次校验   → 本模块 passes_visible_to() / filter_docs │
        └──────────────────────────────────────────────────────────────────────┘

    为什么「过滤」和「二次校验」都要做（纵深防御）：
        · Milvus 过滤发生在召回阶段，决定**能不能被检索到**；
        · 但检索结果会进 L2 缓存（core/cache.py），缓存里可能残留换权限前写入的旧结果；
        · 且 `visible_to` 是「按文档白名单」的语义，比「按密级」更细，不方便全部塞进
          一个标量表达式（需要 OR 一大串角色）。
        因此出口再校验一次，成本是内存里几次字符串比较，几乎无延迟——
        需求 Q&A 里对这条的要求是「隔离查询额外延迟 < 5ms」。

    隔离模型（三层，同一条 chunk 上同时生效）：
        access_level  密级    public < internal < restricted，角色映射到「可见密级集合」
        department    归属部门 all / quality / kitchen / purchasing / management
        visible_to    角色白名单（逗号分隔），非空时**优先于密级规则**，FAQ 专用
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from base.config import config
from base.logger import logger

# ---- 常量 ----
#: 密级取值（顺序即等级，仅用于展示与校验，判定一律走角色映射表）
ACCESS_LEVELS: Tuple[str, ...] = ("public", "internal", "restricted")
#: 表示「所有部门可见」的部门取值
DEPARTMENT_ALL = "all"
#: 文档头部声明块：<!-- key: value -->，用于按文件覆盖知识库默认权限
_FRONT_MATTER_RE = re.compile(r"^\s*<!--\s*([A-Za-z_]+)\s*:\s*(.*?)\s*-->\s*$")


# ==================================================================
# 前台声明块（按文件覆盖默认权限）
# ==================================================================

def parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """
    解析文档开头的 `<!-- key: value -->` 声明块。

    只认「文件最开头的连续若干行」，遇到第一个非声明行即停止——
    避免把正文里出现的 HTML 注释误当成配置。

    :return: (声明字典, 去掉声明块后的正文)
    """
    if not text:
        return {}, ""
    lines = text.split("\n")
    meta: Dict[str, str] = {}
    idx = 0
    for i, line in enumerate(lines):
        if not line.strip():
            idx = i + 1
            continue
        m = _FRONT_MATTER_RE.match(line)
        if not m:
            break
        meta[m.group(1).lower()] = m.group(2)
        idx = i + 1
    if not meta:
        return {}, text
    return meta, "\n".join(lines[idx:])


def resolve_doc_acl(kb_type: str, front_matter: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    计算一条文档最终的权限三元组。

    优先级：**文件级声明块 > 知识库默认表（config.ACCESS_KB_DEFAULTS）> 兜底 internal/all**。

    :param kb_type: regulation / sop / case / nutrition / supplier / faq
    :param front_matter: parse_front_matter() 的产物
    :return: {"access_level": ..., "department": ..., "visible_to": ...}
    """
    fm = front_matter or {}
    default_level, default_dept = config.ACCESS_KB_DEFAULTS.get(
        kb_type, ("internal", DEPARTMENT_ALL))

    level = (fm.get("access_level") or default_level).strip().lower()
    if level not in ACCESS_LEVELS:
        logger.warning(f"[权限] 文档声明的 access_level='{level}' 非法（知识库 {kb_type}），"
                       f"回退为 {default_level}")
        level = default_level

    dept = (fm.get("department") or default_dept).strip().lower() or DEPARTMENT_ALL
    # visible_to 允许为空（表示「按密级规则」）；非空时按逗号切分成角色白名单
    visible_to = (fm.get("visible_to") or "").strip()
    if visible_to:
        roles = [r.strip().lower() for r in visible_to.split(",") if r.strip()]
        unknown = [r for r in roles if r not in config.ACCESS_ROLE_LEVELS]
        if unknown:
            logger.warning(f"[权限] 文档 {kb_type} 的 visible_to 含未定义角色 {unknown}，"
                           f"这些角色将无法检索到该文档")
        visible_to = ",".join(roles)
    return {"access_level": level, "department": dept, "visible_to": visible_to}


# ==================================================================
# 调用方身份
# ==================================================================

@dataclass(frozen=True)
class ACLContext:
    """一次查询的权限上下文（不可变，便于做缓存 key 与打日志）"""

    role: str
    access_levels: Tuple[str, ...]
    departments: Tuple[str, ...]
    enabled: bool = True
    #: 部门来源（role / explicit），便于 /api/query 回显与排障
    department_source: str = "role"

    @property
    def cache_key(self) -> str:
        """
        缓存隔离键（需求 4.2 的第 ③ 条）。

        键里必须同时含「角色」和「实际可见范围」：
            只放角色是不够的——同一个角色可能在调用时显式指定了不同部门
            （例如店长按「后厨」口径查一次、按「采购」口径再查一次），
            这两次的结果集不同，不能互相命中。
        可见范围用短哈希，避免 key 过长；角色本身保留可读，便于运维排查。
        """
        if not self.enabled:
            return "acl-off"
        raw = f"{self.role}|{','.join(self.access_levels)}|{','.join(self.departments)}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"{self.role}-{digest}"

    def describe(self) -> Dict[str, Any]:
        """给 /api/query 响应与健康检查用的可读描述"""
        return {
            "role": self.role,
            "access_levels": list(self.access_levels),
            "departments": list(self.departments),
            "enabled": self.enabled,
            "department_source": self.department_source,
            "cache_key": self.cache_key,
        }


def build_acl(role: Optional[str] = None, department: Optional[str] = None) -> ACLContext:
    """
    根据调用方角色（可选部门）构造权限上下文。

    :param role: 角色名，取值见 config.ini `[access] role_levels`；
                 为空时用 config.ACCESS_DEFAULT_ROLE；未定义的角色一律退化为默认角色
                 （**保守策略：不认识的调用方拿到最小可见范围，而不是全库**）。
    :param department: 显式指定部门，用于「店长按后厨口径查询」这类跨部门视角；
                       为空时按角色的部门清单。
    """
    if not config.ACCESS_ENABLED:
        # 关闭隔离时给一个「全可见」上下文，保证行为与改造前完全一致（零回归）
        return ACLContext(role="acl-disabled", access_levels=tuple(ACCESS_LEVELS),
                          departments=(DEPARTMENT_ALL,), enabled=False)

    raw_role = (role or "").strip().lower() or config.ACCESS_DEFAULT_ROLE
    levels = config.ACCESS_ROLE_LEVELS.get(raw_role)
    if levels is None:
        logger.warning(f"[权限] 未定义的角色 '{raw_role}'，回退为默认角色 "
                       f"'{config.ACCESS_DEFAULT_ROLE}'（按最小可见范围处理）")
        raw_role = config.ACCESS_DEFAULT_ROLE
        levels = config.ACCESS_ROLE_LEVELS.get(raw_role, ("public",))

    if department:
        depts: Tuple[str, ...] = (DEPARTMENT_ALL, department.strip().lower())
        source = "explicit"
    else:
        depts = tuple([DEPARTMENT_ALL] + list(config.ACCESS_ROLE_DEPARTMENTS.get(raw_role, ())))
        source = "role"

    # 去重且保持顺序（all 永远在第一位，便于过滤表达式可读）
    seen: List[str] = []
    for d in depts:
        if d and d not in seen:
            seen.append(d)

    return ACLContext(role=raw_role, access_levels=tuple(levels),
                      departments=tuple(seen), enabled=True, department_source=source)


# ==================================================================
# 过滤表达式与出口校验
# ==================================================================

def _quote_list(values: Sequence[str]) -> str:
    """拼成 Milvus 表达式里的字符串列表：["a", "b"]"""
    inner = ", ".join('"%s"' % str(v).replace('"', '') for v in values)
    return f"[{inner}]"


def build_filter_expr(kb_filter: Any = None, acl: Optional[ACLContext] = None) -> str:
    """
    拼装 Milvus 标量过滤表达式（多个条件用 `and` 连接）。

    生成示例：
        kb_type == "regulation" and access_level in ["public", "internal"]
                                 and department in ["all", "quality"]

    :param kb_filter: str / list / None，沿用既有语义（None = 全库）
    :param acl: 权限上下文；None 或 enabled=False 时不附加权限条件
    """
    parts: List[str] = []

    if isinstance(kb_filter, (list, tuple, set)):
        kb_list = list(kb_filter)
        if kb_list:
            parts.append(f"kb_type in {_quote_list(kb_list)}")
    elif kb_filter:
        parts.append(f'kb_type == "{str(kb_filter)}"')

    if acl is not None and acl.enabled:
        parts.append(f"access_level in {_quote_list(acl.access_levels)}")
        parts.append(f"department in {_quote_list(acl.departments)}")

    return " and ".join(parts)


def passes_visible_to(metadata: Optional[Dict[str, Any]], acl: Optional[ACLContext]) -> bool:
    """
    出口二次校验：判断一条命中的文档是否对当前角色可见。

    规则：
        · 权限关闭 / 未传 acl            → 放行
        · visible_to 为空                 → 放行（按密级规则，召回阶段已过滤）
        · visible_to 含 "all" 或当前角色  → 放行
        · 否则                            → 拦截并记日志（便于发现配置错误）
    """
    if acl is None or not acl.enabled:
        return True
    raw = str((metadata or {}).get("visible_to", "") or "").strip()
    if not raw:
        return True
    roles = {r.strip().lower() for r in raw.split(",") if r.strip()}
    if DEPARTMENT_ALL in roles or acl.role in roles:
        return True
    logger.info(f"[权限] 命中文档因 visible_to={raw} 对角色 '{acl.role}' 不可见，已剔除")
    return False


def filter_docs_by_acl(docs: Iterable[Any], acl: Optional[ACLContext]) -> List[Any]:
    """
    对检索结果做出口过滤（缓存命中后**也必须**走这一步）。

    注意：这里过滤的是「文档」，必须保留原顺序——重排分数是有序的，
          重新排序会破坏 Top-M 的语义。
    """
    if acl is None or not acl.enabled:
        return list(docs or [])
    kept = []
    for doc in docs or []:
        if passes_visible_to(getattr(doc, "metadata", None), acl):
            kept.append(doc)
    return kept


def acl_summary() -> Dict[str, Any]:
    """权限模型摘要（供 /health 与验证脚本输出，便于一眼看清配置）"""
    return {
        "enabled": config.ACCESS_ENABLED,
        "default_role": config.ACCESS_DEFAULT_ROLE,
        "access_levels": list(ACCESS_LEVELS),
        "role_levels": {k: list(v) for k, v in config.ACCESS_ROLE_LEVELS.items()},
        "role_departments": {k: list(v) for k, v in config.ACCESS_ROLE_DEPARTMENTS.items()},
        "kb_defaults": {k: list(v) for k, v in config.ACCESS_KB_DEFAULTS.items()},
    }


if __name__ == '__main__':
    # 自检：把权限模型与过滤表达式打印出来，便于人工核对
    import json

    print("=" * 66)
    print("多用户内容隔离自检")
    print("=" * 66)
    print(json.dumps(acl_summary(), ensure_ascii=False, indent=2))

    print("\n各角色的过滤表达式：")
    for r in config.ACCESS_ROLE_LEVELS:
        acl = build_acl(role=r)
        print(f"  {r:10s} -> {build_filter_expr('regulation', acl)}")
        print(f"  {'':10s}    cache_key={acl.cache_key}")

    print("\nvisible_to 出口校验：")
    acl_kitchen = build_acl(role='kitchen')
    cases = [
        ({}, True, "visible_to 为空 → 放行"),
        ({"visible_to": "all"}, True, "白名单 all → 放行"),
        ({"visible_to": "kitchen,quality"}, True, "命中角色 → 放行"),
        ({"visible_to": "quality"}, False, "未命中角色 → 拦截"),
    ]
    for meta, expect, desc in cases:
        got = passes_visible_to(meta, acl_kitchen)
        flag = "OK " if got == expect else "FAIL"
        print(f"  [{flag}] {desc}（期望 {expect}，实际 {got}）")

    print("\n按文件覆盖权限：")
    text = "<!-- access_level: restricted -->\n<!-- visible_to: quality,admin -->\n# 正文\n条款内容"
    fm, body = parse_front_matter(text)
    print("  声明块 =", fm)
    print("  正文   =", repr(body[:20]))
    print("  解析结果 =", resolve_doc_acl('sop', fm))
