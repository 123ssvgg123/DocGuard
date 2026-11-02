"""
    eval_kit · 检索链路离线验证工具包

    起因：调 RAG 的时候，最贵的不是写代码，是"每改一次都要重新跑一遍评估"。
        RAGAS 一套四指标跑 20 条问题，就是几十次模型调用；而很多时候你要回答的
        只是一个更便宜的问题——「检索到底有没有命中」「拒答边界还在不在」。

    这个包只做那件便宜的事：零模型调用、零向量库依赖，纯本地跑。
        它不能替代 RAGAS（那是评答案质量的），它评的是**链路是否还活着**：
            · 切分边界有没有被改坏（超长块 / 空块）
            · 该命中的有没有命中（hit@1 / hit@3）
            · 该拒答的分数是不是真的低（in/out 分数可分性）

    三个模块：
        chunking   递归切分（与生产切分器同边界规则的可离线实现）
        lexical    字符二元组 TF-IDF 检索（代理"召回"这一步）
        report     结果汇总与判据

    用法见 README.md。
"""
from eval_kit.chunking import (
    CHILD_SIZE_OVERRIDE,
    DEFAULT_SEPARATORS,
    merge_split,
    split_document,
)
from eval_kit.lexical import LexicalIndex, bigrams

__all__ = [
    "CHILD_SIZE_OVERRIDE",
    "DEFAULT_SEPARATORS",
    "merge_split",
    "split_document",
    "LexicalIndex",
    "bigrams",
]

__version__ = "0.1.0"
