"""
    跨领域验证：scripts/verify_cross_domain.py

    目的：证明「建库 → 切分 → 召回 → 阈值 → 拒答」这条链路与领域解耦。
        同一份代码，同一套边界规则，同一套拒答判据，分别跑在
        食品安全语料 与 劳动用工语料 上；两个领域都能"答得出、拒得掉"，
        才说明复用的是链路而不是语料。

    ⚠️ 本脚本**不调用任何远程模型，不依赖 Milvus**，token 成本为 0。
       它验证的是链路契约与边界规则，不是 BGE-M3 的语义召回质量。
       向量召回质量由 build_kb + ragas 评估链路另行验证（需 Docker Milvus）。

    三档验证：
        A. 领域定义与语料覆盖：manifest 合法、每个 kb_type 有语料
        B. 评估集完整性：in_domain 的答案关键词真在目标文件里；out_of_domain 无出处
        C. 词法召回 + 拒答分离度：in_domain top1 命中、out_of_domain 分数低
           —— 判据不是"固定阈值"，而是"两类问题的分数是否可分"

    用法：
        python scripts/verify_cross_domain.py
        python scripts/verify_cross_domain.py --json
        python scripts/verify_cross_domain.py --max-files 4     # 限制每库文件数（默认 8）

    退出码：0 通过 / 1 失败
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.domain import DOMAINS_DIR, DomainSpec, list_domains  # noqa: E402

# ==================== 检索与切分引擎（来自 eval_kit） ====================
# 抽成独立包的动机：这套离线验证不该被生产依赖绑死（生产要 langchain + Milvus），
# 而边界有没有坏、该命中的有没有命中这两件事，任何机器上都该能零成本跑。
from eval_kit import DEFAULT_SEPARATORS, LexicalIndex, split_document  # noqa: E402


# ==================== 语料装载 ====================

def _kb_of_file(name: str, domain: DomainSpec) -> Optional[str]:
    """按 manifest 里的 file_prefix 判归属；无前缀的按目录/关键词兜底。"""
    for kb in domain.kb_types:
        if kb.file_prefix and name.startswith(kb.file_prefix):
            return kb.id
    # food_safety 的 nutrition / faq 无前缀，靠目录名判定
    for kb in domain.kb_types:
        if kb.id in name.lower():
            return kb.id
    return domain.kb_types[-1].id if domain.kb_types else None


def collect_corpus(domain: DomainSpec, max_files: int) -> List[Dict]:
    """按 kb_type 收集语料文件。"""
    base = domain.abs_corpus_dir()
    files: List[Dict] = []
    if not os.path.isdir(base):
        return files

    def push(path: str, kb_type: str):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            return
        files.append({"path": path, "name": os.path.basename(path),
                      "kb_type": kb_type, "text": text})

    if domain.domain_id == "food_safety":
        # 主领域语料按 data/<kb>_data/ 分目录
        for kb in domain.kb_types:
            d = os.path.join(base, f"{kb.id}_data")
            if not os.path.isdir(d):
                continue
            names = sorted(n for n in os.listdir(d) if n.endswith(".md"))
            for n in names[:max_files]:
                push(os.path.join(d, n), kb.id)
    else:
        names = sorted(n for n in os.listdir(base) if n.endswith((".md", ".txt")))
        for n in names:
            kb = _kb_of_file(n, domain)
            if kb:
                push(os.path.join(base, n), kb)
    return files


# ==================== 主流程 ====================

PARENT_SIZE = 512
PARENT_OVERLAP = 128
CHILD_SIZE = 120
CHILD_OVERLAP = 30

# 对齐 core/text_splitter.py 的 CHILD_CHUNK_SIZE_OVERRIDE：
# FAQ 是问答对，原子语义单元不能被 120 字切散，否则会召回"只有问题没答案"的残块。
CHILD_SIZE_OVERRIDE = {"faq": 512}


def check_domain(domain: DomainSpec, max_files: int, errors: List[str], lines: List[str]) -> Dict:
    """跑一个领域的 A/B/C 三档验证，返回指标。"""
    lines.append(f"\n【领域】{domain.domain_id} · {domain.display_name}（{domain.status}）")
    lines.append(f"  kb_type：{', '.join(domain.kb_type_ids())}")

    # ---------- A. 语料覆盖 ----------
    files = collect_corpus(domain, max_files)
    if not files:
        errors.append(f"[{domain.domain_id}] 未找到任何语料（{domain.corpus_dir}）")
        return {}

    covered = {f["kb_type"] for f in files}
    missing = [k for k in domain.kb_type_ids() if k not in covered]
    lines.append(f"  语料文件：{len(files)} 个，覆盖 kb_type {len(covered)}/{len(domain.kb_type_ids())}")
    if missing:
        errors.append(f"[{domain.domain_id}] 以下 kb_type 无任何语料：{missing}")

    # ---------- 切分（领域声明优先，未声明用内置） ----------
    index = LexicalIndex()
    chunk_total = 0
    boundary_issues: List[str] = []
    for f in files:
        seps = domain.separators_for(f["kb_type"]) or DEFAULT_SEPARATORS
        size = (domain.child_chunk_size_for(f["kb_type"])
                or CHILD_SIZE_OVERRIDE.get(f["kb_type"])
                or CHILD_SIZE)
        chunks = split_document(f["text"], seps, size, CHILD_OVERLAP)
        if not chunks:
            boundary_issues.append(f"{f['name']} 切分结果为空")
            continue
        # 边界规则自检：子块不应显著超过目标大小（重叠部分放宽 30%）
        over = [c for c in chunks if len(c) > size * 1.3]
        if over:
            boundary_issues.append(f"{f['name']} 有 {len(over)} 块超过 {int(size * 1.3)} 字")
        for i, c in enumerate(chunks):
            index.add(f"{f['name']}#{i}", c, file=f["name"], kb_type=f["kb_type"])
        chunk_total += len(chunks)

    index.build()
    lines.append(f"  子块总数：{chunk_total}（父块 {PARENT_SIZE}/{PARENT_OVERLAP}，"
                 f"子块 {CHILD_SIZE}/{CHILD_OVERLAP}）")
    if boundary_issues:
        for b in boundary_issues:
            errors.append(f"[{domain.domain_id}] 切分边界异常：{b}")
    else:
        lines.append("  边界规则：通过（无超长块、无空块）")

    # ---------- B. 评估集完整性 ----------
    golden_path = domain.eval_golden_set
    if not golden_path:
        errors.append(f"[{domain.domain_id}] manifest 未声明 eval.golden_set")
        return {}
    abs_golden = golden_path if os.path.isabs(golden_path) else os.path.join(ROOT, golden_path)
    if not os.path.isfile(abs_golden):
        errors.append(f"[{domain.domain_id}] 评估集不存在：{golden_path}")
        return {}

    items = []
    with open(abs_golden, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    in_domain = [i for i in items if i.get("type") == "in_domain"]
    out_domain = [i for i in items if i.get("type") == "out_of_domain"]
    lines.append(f"  评估集：{len(items)} 条（in_domain {len(in_domain)} / "
                 f"out_of_domain {len(out_domain)}）")

    # 关键词必须真出现在标注的目标文件里，否则评估集本身是错的
    by_name = {f["name"]: f["text"] for f in files}
    bad_cases = []
    for item in in_domain:
        target = item.get("expect_file")
        if not target or target not in by_name:
            bad_cases.append(f"{item['id']} 标注的目标文件 {target} 不在语料内")
            continue
        miss = [kw for kw in item.get("expect_keywords", []) if kw not in by_name[target]]
        if miss:
            bad_cases.append(f"{item['id']} 关键词 {miss} 未出现在 {target}")
    for b in bad_cases:
        errors.append(f"[{domain.domain_id}] 评估集标注有误：{b}")
    if not bad_cases:
        lines.append("  评估集标注：通过（答案关键词均可在目标文件定位）")

    # ---------- C. 词法召回 + 拒答分离度 ----------
    hits1 = hits3 = 0
    in_scores: List[float] = []
    for item in in_domain:
        res = index.search(item["question"], top_k=3)
        if not res:
            in_scores.append(0.0)
            continue
        in_scores.append(res[0][0])
        files_top3 = [r[1]["file"] for r in res]
        if files_top3 and files_top3[0] == item.get("expect_file"):
            hits1 += 1
        if item.get("expect_file") in files_top3:
            hits3 += 1

    out_scores: List[float] = []
    for item in out_domain:
        res = index.search(item["question"], top_k=1)
        out_scores.append(res[0][0] if res else 0.0)

    n_in = max(1, len(in_domain))
    min_in = min(in_scores) if in_scores else 0.0
    max_out = max(out_scores) if out_scores else 0.0
    separable = min_in > max_out

    lines.append(f"  召回 hit@1：{hits1}/{n_in}，hit@3：{hits3}/{n_in}")
    lines.append(f"  分数分离：in_domain 最低 {min_in:.4f} ｜ out_of_domain 最高 {max_out:.4f}"
                 f" ｜ 间隔 {min_in - max_out:+.4f} → {'可分离' if separable else '不可分离'}")

    if hits1 == 0:
        errors.append(f"[{domain.domain_id}] hit@1 为 0，召回链路不可用")
    if not separable:
        errors.append(f"[{domain.domain_id}] in/out 分数不可分，拒绝阈值无意义"
                      f"（in_min={min_in:.4f} ≤ out_max={max_out:.4f}）")

    return {
        "domain": domain.domain_id,
        "files": len(files),
        "chunks": chunk_total,
        "in_domain": len(in_domain),
        "out_of_domain": len(out_domain),
        "hit_at_1": hits1,
        "hit_at_3": hits3,
        "in_min_score": round(min_in, 4),
        "out_max_score": round(max_out, 4),
        "separable": separable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="跨领域链路验证（零模型调用）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--max-files", type=int, default=8,
                        help="每个知识库最多读取的文件数（默认 8，控制运行时间）")
    args = parser.parse_args()

    errors: List[str] = []
    lines: List[str] = []
    metrics: List[Dict] = []

    lines.append("=" * 72)
    lines.append("  跨领域链路验证（离线 · 零模型调用 · 不依赖 Milvus）")
    lines.append("=" * 72)

    try:
        domains = list_domains()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 领域定义加载失败：{exc}")
        return 1

    if len(domains) < 2:
        errors.append(f"只发现 {len(domains)} 个领域；跨领域验证至少需要 2 个")
    lines.append(f"发现领域 {len(domains)} 个：{', '.join(domains.keys())}")

    for domain in domains.values():
        m = check_domain(domain, args.max_files, errors, lines)
        if m:
            metrics.append(m)

    # ---------- 跨域一致性 ----------
    lines.append("\n" + "-" * 72)
    lines.append("  跨域一致性")
    lines.append("-" * 72)
    if len(metrics) >= 2:
        ok = all(m["separable"] for m in metrics)
        lines.append(f"  同一套切分/召回/拒答代码在 {len(metrics)} 个领域上均"
                     f"{'可分离' if ok else '存在不可分离领域'}")
        lines.append(f"  → 复用的是链路，不是语料")
    else:
        errors.append("不足两个领域完成对比")

    lines.append("\n" + "=" * 72)
    for e in errors:
        lines.append(f"[FAIL] {e}")
    lines.append(f"结论：{'通过' if not errors else '未通过'}（错误 {len(errors)} 项）")
    lines.append("=" * 72)

    if args.json:
        print(json.dumps({"ok": not errors, "metrics": metrics, "errors": errors},
                         ensure_ascii=False, indent=2))
    else:
        print("\n".join(lines))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
