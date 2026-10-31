"""
    文本切分器：core/text_splitter.py

    针对食品安全文档结构特点设计的分块策略（对应需求基线的「分块策略」要求）：
        - 法规标准：按条款层级分块（第X条 / X.X 款 / （一）项），保留条款编号
        - SOP 规范：按操作步骤分块（步骤 1. / 第一步 / 1、），保留适用岗位
        - 案例报告：按事件结构分块（事件概述 / 原因分析 / 处置过程 / 教训总结）
        - 营养资料：按主题分块（小标题 / 段落）
        - FAQ：按问答对分块（Q: / A:）

    原理：走「父块 → 子块」两级切分。父子块都落库，
        命中子块后用它找回父块内容作为上下文，兼顾检索精度与上下文完整。
"""
import re
from typing import List, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from base.config import config


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # 按分隔符切分，保留分隔符
    if separator:
        if keep_separator:
            # 括号在正则中用于保留分隔符
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """
    中文递归字符切分器：优先按段落/句号等中文标点切分，避免句子被截断。
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本并返回块列表"""
        final_chunks = []
        # 选择合适的分隔符
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # 递归合并小块，控制每块大小
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks


# ============ 食品安全领域专用的结构化分隔符 ============

# 法规条款：匹配「第一条」「第十二条」「3.1」「3.1.2」「（一）」等层级标记
REGULATION_SEPARATORS = [
    r"\n第[一二三四五六七八九十百零]+条",
    r"\n第[一二三四五六七八九十百零]+章",
    r"\n\d+\.\d+\.\d+\s",
    r"\n\d+\.\d+\s",
    r"\n（[一二三四五六七八九十]+）",
    "\n\n",
    "\n",
    "。|！|？",
    r"；|;\s",
    r"，|,\s",
]

# SOP 操作步骤：匹配「步骤1」「第一步」「1、」「1.」等步骤标记
SOP_SEPARATORS = [
    r"\n步骤\s*\d+",
    r"\n第[一二三四五六七八九十]+步",
    r"\n\d+[、.]\s*(?=[^\d])",
    "\n\n",
    "\n",
    "。|！|？",
    r"；|;\s",
    r"，|,\s",
]

# 案例报告：按事件结构分块
CASE_SEPARATORS = [
    r"\n(事件概述|事故概述|基本情况)",
    r"\n(原因分析|原因调查|致病原因)",
    r"\n(处置过程|处理过程|应对措施)",
    r"\n(教训总结|整改措施|经验教训)",
    "\n\n",
    "\n",
    "。|！|？",
    r"；|;\s",
]

# FAQ 问答对：按「问答对」整体分块
#
# ⚠️ 关键：这里【只】用 Q/问 作为分隔符，绝不要把 A/答 也列进来。
#    原因：若把 A 也当分隔符，一个 Q 与其 A 会被切成两块——
#      - "Q: 冰箱冷藏温度标准是多少？"  → 一块（只有问题，没有答案）
#      - "A: 根据 GB 31654-2021，冷藏应 0~4℃" → 另一块（只有答案，没有提问）
#    这样检索命中"问题壳"时拿不到答案正文，rerank 分数会异常低（实测低至 0.015），
#    而且会产生大量形如 "Q:" 的 2 字符垃圾碎片占满召回名额。
#    只按 Q 切分，则每块 = 一个完整问答对（Q + 其 A），语义自洽，检索命中即得答案。
FAQ_SEPARATORS = [
    r"\nQ[:：]",
    r"\n问[:：]",
    "\n\n",      # 兜底：若问答对之间用空行分隔
    "\n",
]

# 营养资料 / 供应商资料 / 默认：按段落与小标题切分
DEFAULT_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。|！|？",
    r"；|;\s",
    r"，|,\s",
]

# 知识库类型 -> 分隔符 映射
KB_SEPARATOR_MAP = {
    'regulation': REGULATION_SEPARATORS,
    'sop': SOP_SEPARATORS,
    'case': CASE_SEPARATORS,
    'faq': FAQ_SEPARATORS,
    'nutrition': DEFAULT_SEPARATORS,
    'supplier': DEFAULT_SEPARATORS,
}

# 子块大小覆盖表：默认用 config.CHILD_CHUNK_SIZE，个别知识库类型需要更大（或更小）的块
#
# ⚠️ FAQ 必须放大：问答对是原子的语义单元，"Q:" 与其 "A:" 一旦被 chunk_size 限制切散，
#    检索就会命中"只有问题没答案"或"只有答案没提问"的残块，rerank 分数异常低（实测 0.015），
#    还会产生 "Q:" 这类 2 字符垃圾碎片占满召回名额。给足空间让整对问答落在同一子块。
CHILD_CHUNK_SIZE_OVERRIDE = {
    'faq': 512,
}


def get_parent_splitter(kb_type: str):
    """按知识库类型取父块切分器（保留结构，块较大）"""
    separators = KB_SEPARATOR_MAP.get(kb_type, DEFAULT_SEPARATORS)
    return ChineseRecursiveTextSplitter(
        separators=separators,
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        is_separator_regex=True,
    )


def get_child_splitter(kb_type: str):
    """按知识库类型取子块切分器（块较小，用于精准向量检索）"""
    separators = KB_SEPARATOR_MAP.get(kb_type, DEFAULT_SEPARATORS)
    chunk_size = CHILD_CHUNK_SIZE_OVERRIDE.get(kb_type, config.CHILD_CHUNK_SIZE)
    return ChineseRecursiveTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        is_separator_regex=True,
    )


if __name__ == '__main__':
    demo = "第一条 为保障食品安全，制定本法。\n第二条 食品生产经营者应当依照法律法规从事生产经营活动。"
    sp = ChineseRecursiveTextSplitter(chunk_size=40, chunk_overlap=8)
    for c in sp.split_text(demo):
        print(repr(c))
