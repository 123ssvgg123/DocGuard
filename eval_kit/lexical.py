"""
    词法检索：eval_kit/lexical.py

    定位：**召回这一步的代理**。生产侧用 BGE-M3 稠密+稀疏混招，跑它需要 Milvus 与
        本地模型（16G 内存机器上很容易 OOM）。做回归时你通常不需要语义质量，
        只需要回答「该命中的文件还排在第一吗」「不该命中的分数是不是还很低」。

    实现：字符二元组 + TF-IDF + 余弦。中文不分词也能用，因为二元组天然覆盖了
        大部分固定搭配（"试用期""经济补偿""48 小时"）。

    ⚠️ 局限（必须说清楚）：
        · 不做同义改写，问法和原文用词差太远会漏——这正是生产侧要 HyDE / 子查询的原因
        · 分数不能与 BGE-Reranker 的 0~1 分数互相换算，只能在本工具内横向比较
"""
import math
from collections import Counter
from typing import Dict, List, Tuple

_PUNCT_RE = __import__("re").compile(
    r"[\s，。、；：？！“”‘’（）《》【】·\-—…,.;:?!()\[\]{}]+"
)


def bigrams(text: str) -> List[str]:
    """字符二元组。单字文本退化为单字，避免空向量。"""
    clean = _PUNCT_RE.sub("", text.lower())
    if len(clean) < 2:
        return [clean] if clean else []
    return [clean[i:i + 2] for i in range(len(clean) - 1)]


class LexicalIndex:
    """最小可用的词法召回索引。"""

    def __init__(self) -> None:
        self.docs: List[Dict] = []
        self._tf: List[Counter] = []
        self._df: Counter = Counter()
        self._idf: Dict[str, float] = {}
        self._built = False

    def add(self, doc_id: str, text: str, **meta) -> None:
        tf = Counter(bigrams(text))
        if not tf:
            return
        self.docs.append({"id": doc_id, "text": text, **meta})
        self._tf.append(tf)
        for term in tf:
            self._df[term] += 1
        self._built = False

    def build(self) -> None:
        n = max(1, len(self.docs))
        self._idf = {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in self._df.items()}
        self._built = True

    def _vec(self, tf: Counter) -> Dict[str, float]:
        vec = {t: (1 + math.log(c)) * self._idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
        """返回 [(余弦分数, 文档)]，按分数降序。"""
        if not self._built:
            self.build()
        qtf = Counter(bigrams(query))
        if not qtf or not self.docs:
            return []
        qv = self._vec(qtf)
        scored: List[Tuple[float, Dict]] = []
        for i, tf in enumerate(self._tf):
            dv = self._vec(tf)
            small, big = (qv, dv) if len(qv) <= len(dv) else (dv, qv)
            score = sum(v * big.get(t, 0.0) for t, v in small.items())
            scored.append((score, self.docs[i]))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.docs)
