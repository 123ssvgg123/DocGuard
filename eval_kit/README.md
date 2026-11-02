# eval_kit · 检索链路的离线回归工具

> 只回答三个便宜的问题，**零模型调用、零向量库依赖**：
> 切分边界有没有被改坏？该命中的有没有命中？该拒答的分数是不是真的低？

## 为什么需要它

调 RAG 的时候最贵的不是写代码，是"每改一次都要重跑评估"。
RAGAS 一套四指标跑 20 条问题就是几十次模型调用；但你要回答的往往是更靠前的那个问题：
**这次改动有没有把检索本身弄坏**。

那件事不该花 token，也不该依赖 Milvus 起没起来。

| | RAGAS / 人工评审 | eval_kit |
|---|---|---|
| 评什么 | 答案质量（faithfulness / relevancy） | 链路是否还活着（边界 / 命中 / 拒答分离度） |
| 成本 | 每轮几十次模型调用 | 0 |
| 依赖 | LLM API + 向量库 + 本地模型 | 无 |
| 什么时候跑 | 发版前 | 每次改切分、改阈值、改召回策略 |

两者不是替代关系：eval_kit 先挡住"低级坏"，RAGAS 再评"答得好不好"。

## 三块内容

```
chunking.py   递归切分（与生产切分器同边界规则的离线实现）+ 块长度分布
lexical.py    字符二元组 TF-IDF 余弦召回（代理"召回"这一步）
```

判据不用固定分数，用**可分性**：

```
in_domain 问题的最低分  >  out_of_domain 问题的最高分
```

只要这个不等式成立，拒答阈值就有意义；不成立说明链路失效——这时候调阈值是自欺欺人。

## 用法

```python
from eval_kit import split_document, LexicalIndex, DEFAULT_SEPARATORS

chunks = split_document(text, DEFAULT_SEPARATORS, size=120, overlap=30)

index = LexicalIndex()
for i, c in enumerate(chunks):
    index.add(f"doc#{i}", c, file="SOP-CCP02.md")
index.build()

top3 = index.search("留样食品在什么温度下保存多少小时？", top_k=3)
print(top3[0][0], top3[0][1]["file"])
```

真实用例见 `scripts/verify_cross_domain.py`（跨领域验证，17 条评估集 + 10 条食安评估集）。

## 刻意不做的三件事

1. **不做同义改写**：问法和原文用词差太远就会漏——这正是生产侧要 HyDE / 子查询的原因，
   工具不该假装自己解决了这个问题。
2. **不换算分数**：这里的余弦分数与 BGE-Reranker 的 0~1 不能互相映射，
   只能在本工具内横向比较。混用会得出"阈值调到 0.35 就安全了"这种假结论。
3. **不替代生产切分器**：`langchain_text_splitters` 有它自己的合并细节，
   这里只保证同一套 separator 切出同量级的块，用于检测"边界被改坏"，不用于逐字符对拍。

## 已知坑

- **重叠要加在最后一层**：递归里加一次、父层再加一次，120 字的块会被撑到 180 字，
  边界自检直接报超长块。重叠是输出形态，不是中间态。
- **FAQ 必须单独放大块**：问答对是原子语义单元，被 120 字切散会召回
  "只有问题没答案"的残块，rerank 分数异常低（实测 0.015），还会让 "Q:" 这类
  两字符碎片占满召回名额。见 `CHILD_SIZE_OVERRIDE`。
