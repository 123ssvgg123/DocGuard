# 食品安全知识库系统 — 流程图总览

> 依据 `D:/heima/FoodSafetySystem` 源码梳理，覆盖 **系统分层**、**离线建库**、**离线数据集构建**、**在线问答** 四条主线。
> 所有图使用 Mermaid 语法，可在支持 Mermaid 的 Markdown 阅读器（Typora / VS Code / GitHub / 语雀）中直接渲染。

---

## 目录

- [0. 系统全貌](#0-系统全貌)
- [1. 系统分层与模块依赖](#1-系统分层与模块依赖)
- [2. 离线建库流程](#2-离线建库流程)
- [3. 离线训练数据集构建流程](#3-离线训练数据集构建流程)
- [4. 在线问答主流程](#4-在线问答主流程)
- [5. 检索策略路由详解](#5-检索策略路由详解)
- [6. 混合检索 + 重排详解](#6-混合检索--重排详解)
- [7. 两层拒答防线](#7-两层拒答防线)
- [8. 数据模型与 Milvus Schema](#8-数据模型与-milvus-schema)

---

## 0. 系统全貌

系统由 **三条独立链路** 组成，共用同一份 `data/*.md` 知识库原文与同一套 `core/` 模块：

```mermaid
flowchart LR
    subgraph SRC["📁 知识库原文 data/*.md"]
        D1[regulation_data 法规]
        D2[sop_data 操作规范]
        D3[case_data 事故案例]
        D4[nutrition_data 营养]
        D5[supplier_data 供应商]
        D6[faq_data 问答]
    end

    subgraph L1["① 离线建库 build_kb.py"]
        A1[加载 + 父子切分]
        A2[BGE-M3 向量化]
        A3[(Milvus 向量库\nfoodsafety_kb)]
        A1 --> A2 --> A3
    end

    subgraph L2["② 离线数据集 build_dataset.py"]
        B1[intent 意图分类集]
        B2[rerank 重排三元组]
        B3[sft 指令微调集]
    end

    subgraph L3["③ 在线问答 app.py / run_cli.py"]
        C1[HTTP / WebSocket]
        C2[RAG 检索 + 重排]
        C3[DeepSeek LLM 生成]
        C1 --> C2 --> C3
    end

    SRC --> L1
    SRC --> L2
    A3 -->|hybrid_search| C2
    B1 -.训练.-> C2
```

**三条链路的关系：**

| 链路 | 触发方式 | 产物 | 是否调用 LLM |
|---|---|---|---|
| ① 离线建库 | `python build_kb.py` | Milvus 集合 `foodsafety_kb` | ❌ 只用 Embedding 模型 |
| ② 离线数据集 | `python build_dataset.py` | `data/dataset/{intent,rerank,sft}/` | ❌ 纯规则 + 原文摘录 |
| ③ 在线问答 | 启动 `app.py` 后请求接口 | 答案 + 引用来源 | ✅ DeepSeek |

---

## 1. 系统分层与模块依赖

```mermaid
flowchart TD
    subgraph ENTRY["🖥️ 入口层"]
        APP["app.py\nFastAPI 服务\nPOST /api/query · WS /api/stream"]
        CLI["run_cli.py\n命令行验证 -q / -k / --debug"]
        WEB["static/index.html\n单页问答前端"]
        WEB -.WebSocket.-> APP
    end

    subgraph ASSEMBLE["🔧 装配层"]
        QA["qa_system.py · QASystem\n组装 LLM / VectorStore / RAGSystem\n会话历史 + 问候语短路"]
    end

    subgraph CORE["⚙️ 核心层 core/"]
        RAG["rag_system.py\nRAGSystem\n策略路由 · 流程编排"]
        VS["vector_store.py\nVectorStore\n向量化 · 入库 · 混合检索 · 重排"]
        DP["document_processor.py\n加载 · 父子切分 · ID 生成"]
        TS["text_splitter.py\n各类库专用分隔符"]
        IC["intent_classifier.py\n模型优先 · 规则降级"]
        PR["prompts.py\nRAGPrompts · HyDE · SubQuery"]
        DSB["dataset_builder.py\n三类数据集构建"]
    end

    subgraph BASE["🧱 基础设施层 base/"]
        CFG["config.py\n.env > config.ini > fallback"]
        LOG["logger.py\nsetup_logger"]
    end

    subgraph EXT["☁️ 外部依赖"]
        MILVUS[("Milvus 2.5\nlocalhost:19530")]
        DS["DeepSeek API\ndeepseek-chat"]
        MODELS["本地模型\nbge-m3 / bge-reranker-large\nintent-classifier"]
    end

    APP --> QA
    CLI --> QA
    QA --> RAG
    QA --> VS
    QA --> DS
    RAG --> VS
    RAG --> IC
    RAG --> PR
    RAG --> CFG
    DP --> TS
    DP --> CFG
    VS --> CFG
    VS --> MILVUS
    VS --> MODELS
    IC --> MODELS
    DSB --> TS
    DSB --> CFG
    QA --> LOG
    RAG --> LOG
    VS --> LOG
```

**依赖方向：** 入口层 → 装配层 → 核心层 → 基础设施层 → 外部依赖。
核心层各模块之间只有 `rag_system` 依赖 `vector_store` / `intent_classifier` / `prompts`，其余互不依赖，**无循环引用**。

---

## 2. 离线建库流程

对应脚本 `build_kb.py`，核心函数链：`VectorStore()` → `delete_collection()`（可选）→ `process_all_knowledge_bases()` → `add_documents()` → `count()`。

```mermaid
flowchart TD
    START([python build_kb.py]) --> ARG{解析参数\n--rebuild / --dir}

    ARG --> INIT["VectorStore()\n加载 BGE-M3 + Reranker\n连接 Milvus"]
    INIT --> REBUILD{--rebuild ?}

    REBUILD -->|是| DEL["delete_collection()\n删除旧集合"]
    REBUILD -->|否| LOAD
    DEL --> LOAD

    LOAD["process_all_knowledge_bases()\n或 process_documents(dir)"] --> SCAN

    subgraph SCAN["逐库处理（6 个目录循环）"]
        direction TB
        S1["load_documents_from_directory()\nMarkdownLoader 纯文本读取\n目录名推导 kb_type\n（去掉 _data 后缀）"]
        S2["_doc_key(kb_type, file_name)\nmd5('kb/file')[:12]\n→ 生成唯一前缀"]
        S3["get_parent_splitter(kb_type)\n父切分 size=512 overlap=128"]
        S4["get_child_splitter(kb_type)\n子切分 size=120 overlap=30\nFAQ 覆盖为 512"]
        S5["父块 ID: {doc_key}_p{j}\n子块 ID: {doc_key}_p{j}_c{k}\n子块存 parent_id + parent_content"]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    LOAD --> S1
    S5 --> CHK{"ID 全局唯一性自检\n重复 → 抛 ValueError"}

    CHK -->|有重复| ERR([❌ 中断报错\n避免 upsert 静默覆盖])
    CHK -->|通过| ADD

    ADD["add_documents(docs)\n_ensure_collection() 写前自愈\nBGE-M3 双向量化\n(dense 1024 + sparse)\nclient.upsert()"] --> CNT

    CNT["count()\nquery(filter='', count(*))\n瞬时 0 时重试 3 次"] --> VERIFY{"total < count ?"}

    VERIFY -->|是| WARN["⚠️ 警告 ID 覆盖\n汇总日志"]
    VERIFY -->|否| OK["✅ 入库成功\n打印各库块数"]
    WARN --> END([完成])
    OK --> END
```

**关键设计点：**

- **父子两级切分**：父块（512 字）保证上下文完整，子块（120 字）保证检索精度；命中子块后用 `parent_content` 还原父块送 LLM。
- **ID 唯一性**：用 `md5(kb_type/file_name)[:12]` 做前缀，杜绝早期「222 个子块只剩 93 个唯一 ID」的撞车问题。
- **写库幂等**：用 `upsert` 而非 `insert`，同一份文档重复构建不会产生重复数据。
- **FAQ 特例**：`FAQ_SEPARATORS` 只按 `\nQ[:：]` / `\n问[:：]` 切分（不能把 `A/答` 当分隔符，否则切出「只有问题没答案」的残块）；`CHILD_CHUNK_SIZE_OVERRIDE = {'faq': 512}` 放大 FAQ 子块。

---

## 3. 离线训练数据集构建流程

对应脚本 `build_dataset.py` + `core/dataset_builder.py`，**全程不调用 LLM**，靠规则 + 原文摘录。

```mermaid
flowchart TD
    START([python build_dataset.py]) --> ARG["解析参数\n--only {all,intent,rerank,sft}\n--eval-ratio 0.2\n--out-dir data/dataset"]

    ARG --> COLLECT["collect_knowledge_files()\n扫描 6 个库目录下所有 .md"]

    COLLECT --> BRANCH{--only 分支}

    BRANCH -->|intent| I1
    BRANCH -->|rerank| R1
    BRANCH -->|sft| F1
    BRANCH -->|all| I1

    subgraph INTENT["① 意图分类数据集"]
        direction TB
        I1["IntentDatasetBuilder\n6 类标签\n法规查询/事故处置/营养咨询\n供应商合规/SOP指导/闲聊"]
        I2["SEED_QUERIES 种子问题\n+ INTENT_KEYWORDS 特征词\n弱标注匹配"]
        I3["_stable_bucket()\nmd5 稳定哈希\n8:2 切分 train/eval"]
        I4["write_jsonl +\nwrite_label_map"]
        I1 --> I2 --> I3 --> I4
    end

    subgraph RERANK["② 重排数据集"]
        direction TB
        R1["RerankDatasetBuilder\n构造 query/positive/negatives 三元组"]
        R2["is_question / parse_faq_pairs /\nparse_doc_sections 提取候选"]
        R3["hard negative\n同库内相似但非答案的块"]
        R4["write_jsonl"]
        R1 --> R2 --> R3 --> R4
    end

    subgraph SFT["③ 指令微调数据集"]
        direction TB
        F1["SFTDatasetBuilder"]
        F2["instruction/input/output\noutput 直接摘自知识库原文"]
        F3["write_jsonl"]
        F1 --> F2 --> F3
    end

    I4 --> README
    R4 --> README
    F3 --> README

    README["write_dataset_readme()\n生成数据集说明"] --> END([输出 data/dataset/])
```

**稳定哈希的意义**：`_stable_bucket` 用 `md5` 而非 Python 内置 `hash()`，避免 `PYTHONHASHSEED` 每次运行不同导致 train/eval 划分漂移、评估集泄漏。

---

## 4. 在线问答主流程

对应 `app.py` → `qa_system.py` → `core/rag_system.py` → `core/vector_store.py`。

```mermaid
flowchart TD
    REQ([用户提问]) --> ENTRY{请求方式}

    ENTRY -->|HTTP| H1["POST /api/query\n非流式"]
    ENTRY -->|WebSocket| W1["WS /api/stream\n流式"]

    W1 --> WSEND["先发 {type: start}"]
    WSEND --> QASK
    H1 --> QASK

    QASK["qa_system.ask / ask_stream\n生成或复用 session_id"] --> GREET{"check_greeting()\n问候语短路？"}

    GREET -->|命中 你好/你是谁/在吗| GOUT["直接返回预设话术\n❌ 不检索 ❌ 不调 LLM"]
    GOUT --> HIST
    GREET -->|未命中| HIST

    HIST["get_history(session_id)\n取最近 HISTORY_ROUNDS=5 轮"] --> RAGQ

    RAGQ["rag_system.query()\n或 generate_answer()"] --> STRAT["select_strategy(query)\n→ 见第 5 节"]

    STRAT --> RETR["retrieve_and_merge()\n按策略检索\n→ 见第 6 节"]

    RETR --> RELEV{"_check_relevance(docs)\nRerank 最高分 ≥ 0.5 ?"}

    RELEV -->|否| REFUSE["两层拒答\n返回品控部门 + 咨询热线话术\n❌ 不调用 LLM"]
    RELEV -->|是| CTX

    CTX["build_context(docs)\n拼装检索上下文"] --> PROMPT["rag_prompt.format\ncontext + history + question + phone\n六条回答要求硬约束"]

    PROMPT --> LLM["self.llm(prompt_input)\nDeepSeek 流式生成"]

    LLM -->|流式| TOK["逐 token 推送\n{type: token}"]
    LLM -->|非流式| ANS

    TOK --> ANS["累积完整答案"]
    ANS --> SRC["format_sources(docs)\nfile_name / kb_type / score / snippet"]
    SRC --> APPEND["_append_history()\n写入会话历史"]

    APPEND --> OUT([返回答案 + 引用来源])
    REFUSE --> OUT2([返回拒答话术 + 空来源])
```

**接口清单：**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回 `static/index.html` |
| GET | `/health` | 健康检查 |
| GET | `/api/kb_types` | 返回 6 大知识库选项 |
| POST | `/api/create_session` | 创建会话 ID |
| GET | `/api/history/{session_id}` | 读取会话历史 |
| DELETE | `/api/history/{session_id}` | 清空会话历史 |
| POST | `/api/query` | 非流式问答 |
| WS | `/api/stream` | 流式问答（start / token / end / error） |

**WebSocket 消息类型：**

```mermaid
sequenceDiagram
    participant U as 前端 index.html
    participant A as app.py
    participant Q as QASystem
    participant R as RAGSystem

    U->>A: WS 连接 + 发送 query
    A->>U: {type: "start"}
    A->>Q: ask_stream(query, kb_filter, session_id)
    Q->>Q: check_greeting → get_history
    Q->>R: generate_answer(query, kb_filter, history)
    R->>R: select_strategy → retrieve_and_merge → 阈值校验
    R-->>Q: 生成器 stream + docs
    loop 每个 token
        Q-->>A: token, False, {}
        A->>U: {type: "token", content: "..."}
    end
    Q-->>A: "", True, {sources}
    A->>U: {type: "end", sources: [...]}
    Note over A,U: 异常时发送 {type: "error"}
```

---

## 5. 检索策略路由详解

`RAGSystem.select_strategy()` 的**路由优先级：形态判断 → 意图分类**。这是本项目最容易误解的一点：**先看句式形态，再看语义意图**。

```mermaid
flowchart TD
    Q([query]) --> NORM["normalize_query()\nNFKC 归一化\n修复康熙部首/异体字/全角"]

    NORM --> C1{"含口语化标记?\n有点/怪/感觉/好像/\n能不能用/还能用/能不能吃/\n怎么回事/啥/咋"}

    C1 -->|是| HYDE["策略 = hyde\n假设解答：先让 LLM 生成\n假设性答案再检索"]

    C1 -->|否| C2{"问号 ≥ 2 个?\n或含复合标记?\n哪些条件/需要满足/分别/\n以及/同时/还要/并且"}

    C2 -->|是| SUBQ["策略 = subquery\n子问题拆分：拆成多个子问题\n分别检索后合并去重"]

    C2 -->|否| IC["intent_classifier.predict(query)"]

    IC --> ICCHK{"模型可用?"}
    ICCHK -->|是| MODEL["ModelIntentClassifier\n微调模型"]
    ICCHK -->|否| RULE["RuleIntentClassifier\n特征词规则降级"]

    MODEL --> MAP["INTENT_TO_STRATEGY\n查表"]
    RULE --> MAP

    MAP --> RESULT["策略映射\n法规查询/SOP指导/营养咨询/\n供应商合规/闲聊 → direct\n事故处置 → subquery"]

    HYDE --> EXEC
    SUBQ --> EXEC
    RESULT --> EXEC

    EXEC["执行对应检索"]
    EXEC --> FALLBACK{"HyDE / SubQuery\n是否失败或返回空?"}
    FALLBACK -->|是| DIR["回退 direct 直接检索"]
    FALLBACK -->|否| DONE([进入重排])
    DIR --> DONE
```

**三种策略对比：**

| 策略 | 触发条件 | 做法 | 失败兜底 |
|---|---|---|---|
| `direct` | 默认 / 大部分意图 | 直接用原 query 检索 | — |
| `hyde` | 含口语化标记 | LLM 先生成假设性答案，用假设答案检索 | 回退 direct |
| `subquery` | 多问号 / 复合标记 / 事故处置意图 | 拆成多个子问题分别检索，按 `page_content` 去重合并 | 回退 direct |

**意图 → 策略映射表（`INTENT_TO_STRATEGY`）：**

| 意图 | 策略 |
|---|---|
| 法规查询 | direct |
| SOP 指导 | direct |
| 营养咨询 | direct |
| 供应商合规 | direct |
| 闲聊 | direct |
| **事故处置** | **subquery** |

---

## 6. 混合检索 + 重排详解

对应 `core/vector_store.py` 的 `hybrid_search_with_rerank()`。核心思想：**双路召回 → 粗排融合 → 子块升父块 → 精排截断**。

```mermaid
flowchart TD
    Q([query]) --> NORM["normalize_query()\n入库与查询必须用同一函数"]

    NORM --> EMB["BGE-M3 编码\n同时输出两种向量"]
    EMB --> DV["dense 向量\n1024 维"]
    EMB --> SV["sparse 向量\n稀疏权重词典"]

    DV --> DR["AnnSearchRequest\ndense_vector · IP\nnprobe=32 · limit=k=5"]
    SV --> SR["AnnSearchRequest\nsparse_vector · IP\nlimit=k=5"]

    DR --> HS["client.hybrid_search()\n双路 ANN 并发召回"]
    SR --> HS

    HS --> RK["WeightedRanker(1.0, 0.7)\ndense 权重 1.0\nsparse 权重 0.7\n粗排融合"]

    RK --> CHILD["得到 Top-K 子块\n带 parent_id / parent_content"]

    CHILD --> UNIQ["_get_unique_parent_docs()\n按 parent_id 去重升父块\n同一父块只保留一次"]

    UNIQ --> PAIR["构造 [query, parent_content] 配对"]

    PAIR --> CE["CrossEncoder\nbge-reranker-large\n逐对精排打分"]

    CE --> SORT["按分数降序排序\n取 Top-M = 3"]

    SORT --> META["写入 doc.metadata['rerank_score']"]

    META --> OUT([返回候选父块列表])
```

**参数说明：**

| 参数 | 值 | 位置 | 含义 |
|---|---|---|---|
| `PARENT_CHUNK_SIZE` | 512 | config | 父块大小 |
| `CHILD_CHUNK_SIZE` | 120 | config | 子块大小（FAQ 覆盖为 512） |
| `RETRIEVAL_K` | 5 | config | 每路 ANN 召回数 |
| `CANDIDATE_M` | 3 | config | 重排后保留父块数 |
| `SCORE_THRESHOLD` | 0.5 | config | 拒答阈值 |
| `NPROBE` | 32 | config | IVF_FLAT 探查簇数（nlist=128，约 1/4） |
| `WeightedRanker` | (1.0, 0.7) | 代码 | dense / sparse 融合权重 |

**为什么要有 `WeightedRanker` 这一步：** dense 向量擅长语义相似（「冷链温度」≈「冷藏要求」），sparse 向量擅长关键词精确命中（标准编号 `GB 31605-2020`）。两者分数尺度不同，必须先融合成统一排序，再做 CrossEncoder 精排。

---

## 7. 两层拒答防线

用户明确要求「不编造、缺依据就拒答」，系统用**两层**保证：

```mermaid
flowchart TD
    subgraph L1["第一层：检索阈值拒答（代码硬逻辑）"]
        direction TB
        A1["hybrid_search_with_rerank 返回候选"] --> A2{"Rerank 最高分\n≥ SCORE_THRESHOLD 0.5 ?"}
        A2 -->|否| A3["直接返回拒答话术\n'知识库中暂无权威依据，\n建议联系品控部门核实，\n或拨打咨询热线 {SERVICE_PHONE}'\n❌ 完全不调用 LLM"]
        A2 -->|是| A4["进入提示词组装"]
    end

    subgraph L2["第二层：提示词硬约束（LLM 语义层）"]
        direction TB
        B1["rag_prompt 六条要求"] --> B2["① 严格基于检索结果"]
        B1 --> B3["② 强制标注标准编号"]
        B1 --> B4["③ 步骤化输出"]
        B1 --> B5["④ 信息不足拒答"]
        B1 --> B6["⑤ 口语化还原意图"]
        B1 --> B7["⑥ 缺关键数值必须拒答"]
    end

    A4 --> B1
    B1 --> C["LLM 生成"]
    C --> D{"LLM 判断\n信息是否充分?"}
    D -->|不足| E["LLM 主动拒答\n或说明缺失项"]
    D -->|充分| F([输出带引用的答案])
```

| 防线 | 位置 | 触发方式 | 优点 | 局限 |
|---|---|---|---|---|
| 第一层 | `_check_relevance()` | 代码判断分数 | 确定性高、省 token | 只能判断「有没有」，不能判断「够不够」 |
| 第二层 | `core/prompts.py` 第 4、6 条 | LLM 语义理解 | 能识别「答非所问」「缺具体数值」 | 依赖模型遵循指令的能力 |

**两层互补**：第一层拦掉「知识库里根本没有的问题」，第二层拦掉「有相关内容但不足以回答的问题」（例如只说了要留样、但没给留样时间和温度）。

---

## 8. 数据模型与 Milvus Schema

**统一 Collection：`foodsafety_kb`**（Milvus 2.5，database=`foodsafety`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR(100) | 主键，`{doc_key}_p{j}_c{k}` |
| `text` | VARCHAR(65535) | 子块文本（用于检索命中） |
| `dense_vector` | FLOAT_VECTOR(1024) | BGE-M3 dense 向量 |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BGE-M3 sparse 向量 |
| `parent_id` | VARCHAR | 指向父块 |
| `parent_content` | VARCHAR(65535) | 父块全文（送 LLM 用） |
| `kb_type` | VARCHAR | 6 大知识库标量过滤字段 |
| `file_name` | VARCHAR | 来源文件 |
| `timestamp` | INT64 | 写入时间 |

**索引：**

| 索引 | 字段 | 类型 | 度量 | 参数 |
|---|---|---|---|---|
| `dense_index` | dense_vector | IVF_FLAT | IP | `nlist=128` |
| `sparse_index` | sparse_vector | SPARSE_INVERTED_INDEX | IP | `drop_ratio_build=0.2` |

**6 大知识库（`kb_type` → 目录 → 中文名）：**

```mermaid
flowchart LR
    COL[("foodsafety_kb\n统一集合")]
    COL --> K1["regulation\n法规标准"]
    COL --> K2["sop\n操作规范"]
    COL --> K3["case\n事故案例"]
    COL --> K4["nutrition\n营养知识"]
    COL --> K5["supplier\n供应商合规"]
    COL --> K6["faq\n实操问答"]
```

**统一 Collection 的好处：** 一次检索可跨库召回（`kb_filter=None`），也可用 `expr="kb_type == 'regulation'"` 精确过滤到单库——对应前端知识库下拉框。

---

## 附：核心源码文件索引

| 文件 | 职责 |
|---|---|
| `app.py` | FastAPI 服务入口，挂载静态页，HTTP + WebSocket 接口 |
| `qa_system.py` | 装配层，会话历史管理，问候语短路，LLM 异常兜底 |
| `run_cli.py` | 命令行验证工具 |
| `core/rag_system.py` | RAG 主流程编排，策略路由，拒答判断 |
| `core/vector_store.py` | 向量化、入库、混合检索、重排 |
| `core/document_processor.py` | 文档加载、父子切分、ID 生成与自检 |
| `core/text_splitter.py` | 各库专用分隔符与切分器 |
| `core/intent_classifier.py` | 意图识别（模型优先，规则降级） |
| `core/prompts.py` | RAG / HyDE / SubQuery 三类提示词 |
| `core/dataset_builder.py` | 意图 / 重排 / SFT 三类数据集构建 |
| `base/config.py` | 配置加载（.env > config.ini > fallback） |
| `base/logger.py` | 日志配置 |
| `build_kb.py` | 离线建库脚本（支持 `--rebuild`） |
| `build_dataset.py` | 离线数据集构建脚本 |
| `static/index.html` | 单页问答前端 |
