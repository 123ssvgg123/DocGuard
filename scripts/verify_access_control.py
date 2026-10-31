"""
    多用户内容隔离回归验证：scripts/verify_access_control.py

    职责：在不依赖 Milvus / 模型 / 网络的前提下，验证需求 4.2「多用户内容隔离」的
         四条策略是否按预期工作。这是本项目少数「纯内存断言」的验证脚本，
         可以放进 CI 每次都跑。

    覆盖项（与需求 4.2 的四条策略一一对应）：
        ① 单集合 + kb_type 语义隔离      —— 过滤表达式里出现 kb_type 条件
        ② metadata 行级权限过滤           —— access_level / department 条件按角色变化
        ③ 缓存 key 按角色/部门隔离        —— 不同角色的检索缓存键不相等
        ④ FAQ visible_to 二次校验         —— 白名单之外的角色拿不到该文档
        附加：按文件声明块覆盖知识库默认权限；未定义角色的保守退化；关闭开关时零回归

    用法：
        python scripts/verify_access_control.py            # 全部断言
        python scripts/verify_access_control.py --verbose  # 额外打印生成的过滤表达式
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import config                                    # noqa: E402
from core.access_control import (ACCESS_LEVELS, ACLContext, build_acl,       # noqa: E402
                                 build_filter_expr, filter_docs_by_acl,
                                 parse_front_matter, passes_visible_to,
                                 resolve_doc_acl, acl_summary)
from core.cache import rag_cache                                   # noqa: E402


class _Doc:
    """轻量 Document 替身，避免为了跑断言而 import langchain"""

    def __init__(self, content, **meta):
        self.page_content = content
        self.metadata = meta


RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print("=" * 74)
    print("多用户内容隔离回归验证（需求 4.2）")
    print("=" * 74)
    print(f"隔离开关 ACCESS_ENABLED = {config.ACCESS_ENABLED}，"
          f"兜底角色 = {config.ACCESS_DEFAULT_ROLE}")
    print(f"角色清单 = {list(config.ACCESS_ROLE_LEVELS)}")
    print()

    # ---------------- 策略 ② metadata 行级权限过滤 ----------------
    print("策略 ② metadata 行级权限过滤（access_level / department）")
    acl_public = build_acl(role='public')
    acl_staff = build_acl(role='staff')
    acl_quality = build_acl(role='quality')
    acl_supplier = build_acl(role='supplier')

    e_public = build_filter_expr('regulation', acl_public)
    e_staff = build_filter_expr('regulation', acl_staff)
    e_quality = build_filter_expr('case', acl_quality)
    if args.verbose:
        for r, e in (('public', e_public), ('staff', e_staff), ('quality', e_quality)):
            print(f"      {r:9s} -> {e}")

    check("public 角色只可见 public 密级",
          'access_level in ["public"]' in e_public, e_public)
    check("staff 角色可见 public + internal",
          'access_level in ["public", "internal"]' in e_staff, e_staff)
    check("quality 角色额外可见 restricted",
          'restricted' in e_quality and 'access_level in' in e_quality, e_quality)
    check("department 条件按角色附加",
          'department in ["all"]' in e_public and 'department in ["all", "quality"]' in e_quality)
    check("kb_type 条件保留（策略 ① 的语义隔离）",
          'kb_type == "regulation"' in e_staff and 'kb_type == "case"' in e_quality)
    check("多条件用 and 连接",
          e_staff.count(' and ') == 2, e_staff)

    # 显式指定 department 时应覆盖角色的部门清单
    acl_explicit = build_acl(role='staff', department='purchasing')
    check("显式 department 覆盖角色默认部门",
          'purchasing' in build_filter_expr(None, acl_explicit)
          and acl_explicit.department_source == 'explicit')
    check("全库检索（kb_filter=None）不产生 kb_type 条件",
          build_filter_expr(None, acl_staff).startswith('access_level in'))

    # ---------------- 策略 ③ 缓存键隔离 ----------------
    print("\n策略 ③ 检索缓存键按角色/部门隔离")
    keys = {
        'public': rag_cache.key_retrieval('冷藏温度标准', 'regulation', 'direct', acl_public.cache_key),
        'staff': rag_cache.key_retrieval('冷藏温度标准', 'regulation', 'direct', acl_staff.cache_key),
        'quality': rag_cache.key_retrieval('冷藏温度标准', 'regulation', 'direct', acl_quality.cache_key),
    }
    if args.verbose:
        for k, v in keys.items():
            print(f"      {k:9s} -> {v}")
    check("不同角色的检索缓存键互不相同", len(set(keys.values())) == 3,
          f"{len(set(keys.values()))} 个唯一键 / 3")
    check("同一角色两次构造得到同一个键（稳定）",
          acl_staff.cache_key == build_acl(role='staff').cache_key)
    check("同角色不同 department 的键也不同",
          acl_staff.cache_key != acl_explicit.cache_key)
    check("关闭隔离时退化为 acl-off（保持旧行为）",
          ACLContext(role='x', access_levels=('public',), departments=('all',),
                     enabled=False).cache_key == 'acl-off')

    # ---------------- 策略 ④ visible_to 出口校验 ----------------
    print("\n策略 ④ visible_to 白名单出口二次校验")
    doc_ok = _Doc("公开条款", access_level='public', department='all', visible_to='')
    doc_faq = _Doc("只给品控看", access_level='public', department='all',
                   visible_to='quality,admin')
    doc_all = _Doc("所有人可见", access_level='public', department='all', visible_to='all')

    check("visible_to 为空 → 放行", passes_visible_to(doc_ok.metadata, acl_staff))
    check("白名单命中角色 → 放行", passes_visible_to(doc_faq.metadata, acl_quality))
    check("白名单未命中角色 → 拦截", not passes_visible_to(doc_faq.metadata, acl_staff))
    check("visible_to=all → 所有角色放行", passes_visible_to(doc_all.metadata, acl_public))
    check("出口过滤保持原顺序（不能打乱重排分数）",
          [d.page_content for d in filter_docs_by_acl([doc_faq, doc_ok, doc_all], acl_staff)]
          == ["公开条款", "所有人可见"])
    check("关闭隔离时出口一律放行",
          len(filter_docs_by_acl([doc_faq], None)) == 1)

    # ---------------- 附加：按文件覆盖默认权限 ----------------
    print("\n附加：文件声明块覆盖知识库默认权限")
    default_case = resolve_doc_acl('case')
    check("case 库默认 restricted/quality",
          default_case == {"access_level": "restricted", "department": "quality",
                           "visible_to": ""}, str(default_case))

    text = ("<!-- access_level: internal -->\n"
            "<!-- department: kitchen -->\n"
            "<!-- visible_to: kitchen,admin -->\n"
            "# 冷菜间操作规范\n正文内容")
    fm, body = parse_front_matter(text)
    overridden = resolve_doc_acl('case', fm)
    check("声明块覆盖密级/部门/白名单",
          overridden == {"access_level": "internal", "department": "kitchen",
                         "visible_to": "kitchen,admin"}, str(overridden))
    check("声明块从正文中剥离", body.strip().startswith('# 冷菜间操作规范'), repr(body[:20]))

    bad = resolve_doc_acl('sop', {"access_level": "top-secret"})
    check("非法密级回退知识库默认值",
          bad['access_level'] == config.ACCESS_KB_DEFAULTS['sop'][0], str(bad))

    # ---------------- 附加：未定义角色的保守退化 ----------------
    print("\n附加：未定义角色的保守退化")
    acl_unknown = build_acl(role='超级管理员')
    check("未定义角色退化为兜底角色（不是全库开放）",
          acl_unknown.role == config.ACCESS_DEFAULT_ROLE, acl_unknown.role)
    check("兜底角色不含 restricted",
          'restricted' not in acl_unknown.access_levels, str(acl_unknown.access_levels))
    check("兜底角色不是 admin",
          acl_unknown.role != 'admin')

    # ---------------- 附加：权限模型自洽 ----------------
    print("\n附加：权限模型自洽性")
    for role, levels in config.ACCESS_ROLE_LEVELS.items():
        if not set(levels) <= set(ACCESS_LEVELS):
            check(f"角色 {role} 的密级取值合法", False, str(levels))
            break
    else:
        check("所有角色的密级取值都在 {public, internal, restricted} 内", True)

    for kb, (level, dept) in config.ACCESS_KB_DEFAULTS.items():
        if kb not in config.KB_TYPES:
            check(f"kb_defaults 里的 {kb} 是已定义的知识库", False)
            break
        if level not in ACCESS_LEVELS:
            check(f"kb_defaults 里 {kb} 的密级合法", False, level)
            break
    else:
        check("kb_defaults 覆盖的知识库与密级均合法", True)

    # ---------------- 汇总 ----------------
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 74)
    print(f"共 {len(RESULTS)} 项断言，通过 {len(RESULTS) - len(failed)} 项，失败 {len(failed)} 项")
    if failed:
        print("失败项：")
        for n in failed:
            print("  -", n)
    print("=" * 74)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
