"""
    递归切分：eval_kit/chunking.py

    定位：生产切分器（core/text_splitter.py）的**同边界规则离线实现**。
        生产侧依赖 langchain_text_splitters，装不上 langchain 的机器就跑不了评估；
        而边界规则本身是用不了重依赖的——它只是"按优先级找分隔符 + 合并到上限 + 加相邻重叠"。

    ⚠️ 明确边界：这个实现不替代生产切分器，也不追求与 LangChain 逐字符一致。
       它保证的是同一套 separator 列表会切出同样量级的块，
       从而让"边界有没有被改坏"这件事可以在任何机器上零成本验证。
"""
import re
from typing import List

# 默认分隔符优先级：段落 → 行 → 句 → 分句 → 逗号
DEFAULT_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。|！|？",
    r"；|;\s",
    r"，|,\s",
]

# 对齐生产侧 CHILD_CHUNK_SIZE_OVERRIDE：FAQ 是原子的问答对，不能被常规块大小切散，
# 否则会召回"只有问题没答案"或"只有答案没提问"的残块。
CHILD_SIZE_OVERRIDE = {"faq": 512}


def merge_split(text: str, separators: List[str], size: int) -> List[str]:
    """
    递归切分 + 合并到块大小上限，**不加重叠**。

    重叠由上层统一加。早期版本在递归里也加了一次、父层又加一次，
    结果 120 字的块被撑到 180 字，边界自检直接报"超长块"——
    教训是：重叠属于"最终输出形态"，不该混进"切分中间态"。
    """
    if not text.strip():
        return []
    if len(text) <= size:
        return [text]

    for i, sep in enumerate(separators):
        try:
            parts = re.split(f"({sep})", text) if "|" in sep else text.split(sep)
        except re.error:
            # 领域 manifest 里的正则可能写错，跳过该级而不是整体失败
            continue
        if len(parts) <= 1:
            continue

        merged: List[str] = []
        buf = ""
        for p in parts:
            if not p:
                continue
            if len(p) > size:
                if buf:
                    merged.append(buf)
                    buf = ""
                merged.extend(merge_split(p, separators[i + 1:], size))
            elif len(buf) + len(p) <= size:
                buf += p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        if merged:
            return [c for c in merged if c.strip()]

    # 没有任何分隔符命中 → 按大小硬切
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def split_document(text: str, separators: List[str], size: int, overlap: int) -> List[str]:
    """对外接口：递归切分后统一加相邻重叠。"""
    chunks = merge_split(text, separators or DEFAULT_SEPARATORS, size)
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        out.append(prev[-overlap:] + cur)
    return out


def size_histogram(chunks: List[str]) -> dict:
    """块长度分布，用于快速判断"是不是又被某个分隔符吃掉了"。"""
    if not chunks:
        return {"count": 0}
    lens = sorted(len(c) for c in chunks)
    return {
        "count": len(lens),
        "min": lens[0],
        "p50": lens[len(lens) // 2],
        "max": lens[-1],
        "empty": sum(1 for c in chunks if not c.strip()),
    }
