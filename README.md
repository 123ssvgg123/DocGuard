# 食安通 · 食品安全知识库系统（Food Safety Knowledge Base System）

基于 RAG 架构的餐饮食品安全智能问答系统。整合食品安全法规、SOP 操作规范、事故处置案例、
营养学资料、供应商资质与高频 FAQ 六大知识库，为品控人员、门店管理人员和一线操作人员
提供秒级的法规查询、SOP 指导与事故处置服务。

- 需求基线：见 [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md)（解析自场景文档 1.47）
- 启动验证：见 [`RUN.md`](RUN.md)

---

## 一、目录结构

```
FoodSafetySystem/
├── app.py                      # FastAPI 服务入口（HTTP + WebSocket 接口 + 前端托管）
├── build_kb.py                 # 知识库构建入口（导入 → 切分 → 向量化 → 入库）
├── build_dataset.py            # 数据集构建入口（知识库 → 意图/重排/指令微调训练数据）
├── train_intent.py             # 意图分类模型训练入口（→ models/intent-classifier/）
├── run_cli.py                  # 命令行问答入口（验证 RAG 效果）
├── qa_system.py                # 问答系统装配层（LLM + 向量库 + 会话历史 + 问候语短路）
├── config.ini                  # 主配置文件
├── .env                        # 环境变量覆盖（优先级高于 config.ini）
├── requirements.txt            # 依赖清单
│
├── base/                       # 基础设施层
│   ├── config.py               #   配置管理 + 计算设备自动探测（resolve_device）
│   └── logger.py               #   日志管理
│
├── core/                       # RAG 核心逻辑
│   ├── text_splitter.py        #   中文递归切分器 + 各知识库专用分隔符
│   ├── document_processor.py   #   文档加载与父子两级切分
│   ├── vector_store.py         #   向量化存储 + 混合检索 + Rerank
│   ├── prompts.py              #   提示词模板（RAG / HyDE / SubQuery）
│   ├── dataset_builder.py      #   训练数据集构建（意图分类 / 重排 / 指令微调）
│   ├── intent_classifier.py    #   意图识别推理（模型优先，缺失自动回退规则）
│   └── rag_system.py           #   RAG 主流程（策略路由 → 检索 → 组装 → 生成）
│
├── data/                       # 知识数据（目录名去掉 _data 后缀即知识库类型）
│   ├── regulation_data/        #   法规标准库
│   ├── sop_data/               #   SOP 规范库
│   ├── case_data/              #   事故处置案例库
│   ├── nutrition_data/         #   营养学知识库
│   ├── supplier_data/          #   供应商资质库
│   ├── faq_data/               #   FAQ 问答库
│   └── dataset/                #   训练数据集产物（由 build_dataset.py 生成，不入库）
│
├── static/index.html           # 问答交互页面
├── models/                     # 本地模型（bge-m3 / bge-reranker-large / intent-classifier）
├── logs/                       # 运行日志
└── docs/                       # 文档（REQUIREMENTS.md 需求基线 / FLOW.md 流程图）
```

## 二、各模块职责

| 模块 | 职责 | 关键设计 |
|------|------|----------|
| `base/config.py` | 全局配置的唯一来源 | 优先级 `.env` > `config.ini` > 代码默认值；业务代码零硬编码；`resolve_device()` 运行时自动探测计算设备（cuda > mps > xpu > cpu） |
| `base/logger.py` | 统一日志输出（控制台 + 文件） | 与参考项目一致的格式 |
| `core/text_splitter.py` | 中文递归切分 | 为法规/SOP/案例/FAQ 分别定义结构化分隔符，保留条款与步骤边界 |
| `core/document_processor.py` | 加载 + 父子两级切分 | 目录名推导知识库类型；父块定上下文、子块定检索精度 |
| `core/vector_store.py` | 向量化、入库与检索 | BGE-M3 稠密+稀疏双向量；Milvus 统一集合用 `kb_type` 区分 6 库；混合检索 + Reranker 精排 |
| `core/prompts.py` | 提示词模板 | RAG 主模板强制法规引用与拒答，并含"缺关键数值必须拒答"防幻觉硬约束；HyDE / SubQuery 模板用于检索增强 |
| `core/dataset_builder.py` | 训练数据集构建 | 从知识库原文挖掘意图分类 / 重排 / 指令微调三类数据；output 一律摘自原文，不用 LLM 生成；意图集支持「种子 + 模板扩写」，并按来源分层切分评估集 |
| `core/intent_classifier.py` | 意图识别推理 | 加载 `models/intent-classifier/` 做 6 类意图预测；模型缺失时自动降级为规则分类器 |
| `core/rag_system.py` | RAG 主流程编排 | 形态判断（口语化/复合）+ 意图识别 → 策略路由 → 检索 → 阈值判断 → 上下文组装 → 流式生成 |
| `qa_system.py` | 装配层 | 组装 LLM 客户端 / 向量库 / RAG 流程 / 会话历史；问候语短路；对外暴露 `ask` 与 `ask_stream` |
| `build_kb.py` | 离线建库入口 | 支持增量、全量重建、单库导入 |
| `build_dataset.py` | 离线数据加工入口 | 把 `data/` 知识库转为训练数据集；支持 `--only intent/rerank/sft` 分任务构建、`--no-augment` 关闭模板扩写 |
| `train_intent.py` | 意图分类训练入口 | 微调基座产出 `models/intent-classifier/`；训练集内部切验证集、固定测试集只评一次；输出准确率/macro-F1/混淆矩阵 |
| `run_cli.py` | 命令行验证 | 单次提问 / 交互问答 / 展示引用来源 |
| `app.py` | 服务入口（唯一） | 仅装配与转发；HTTP 非流式 `POST /api/query`、WebSocket 流式 `WS /api/stream` |

### 数据流

```
【离线-建库】data/*.md  →  document_processor（加载 + 父子切分）
                       →  vector_store.add_documents（BGE-M3 向量化）
                       →  Milvus（子块向量 + 父块内容 + 元数据）

【离线-数据集】data/*.md  →  build_dataset.py
                          →  ├── intent/  意图分类数据集（6 类意图，供训练意图识别模型）
                             ├── rerank/  重排数据集（query/positive/negatives 三元组）
                             └── sft/     指令微调数据集（instruction/input/output）

【在线】用户提问  →  qa_system.ask / ask_stream
                 →  问候语短路判断（命中直接返回，不检索）
                 →  rag_system（形态判断 + 意图识别 → 策略路由 → hybrid_search_with_rerank → 阈值校验）
                 →  prompts.rag_prompt（上下文 + 历史 + 问题）
                 →  LLM 流式生成  →  答案 + 引用来源
```

## 三、配置集中管理

所有关键配置集中在 `config.ini` / `.env`，改配置无需改代码：

| 配置分组 | 内容 |
|----------|------|
| `[server]` | 服务监听地址与端口 |
| `[milvus]` | Milvus 主机、端口、数据库名、集合名 |
| `[llm]` | 大模型名、API Key、Base URL |
| `[model]` | Embedding / Rerank / 意图分类模型目录名、推理设备（留空 = 自动探测） |
| `[retrieval]` | 父子块大小与重叠、召回数量 K、候选数量 M、拒答阈值 |
| `[app]` | 咨询热线、历史轮数、提示词最大长度 |
| `[train]` | 意图分类训练：基座模型、轮数、学习率、批大小、验证集占比、随机种子 |

## 四、数据处理环节：训练数据集构建

项目除了「建库用」的知识处理（`build_kb.py`），还有一条**面向模型训练**的数据加工链路
（`build_dataset.py`），产出三类训练数据：

| 数据集 | 用途 | 样本格式 | 划分 |
|--------|------|----------|------|
| `data/dataset/intent/` | 训练**意图识别模型**（6 类意图分类） | `{"text": "…", "label": 1, "label_name": "SOP 指导"}` | train / eval |
| `data/dataset/rerank/` | 微调 **BGE-Reranker**（提升 Top-K 精度） | `{"query": "…", "positive": "…", "negatives": ["…"]}` | train / eval |
| `data/dataset/sft/` | **LoRA 指令微调**对话模型 | `{"instruction": "…", "input": "", "output": "…"}` | train / eval |

**三条关键设计原则：**

1. **数据全部来自知识库原文，不调用大模型生成**。用模型自己产的答案去训练模型，
   会把上一代的幻觉固化成「标准答案」。`output` 字段一律摘自 `data/` 的 FAQ 问答对。
2. **意图样本的评估集独立切出**。按文本哈希切分，评估集绝不出现在训练集中，
   避免「用规则训练、又用规则评测」的自证式评估。
3. **重排负样本用 hard negative**。从同一知识库的其他文档取样（主题相近但非答案），
   比随机负例更有训练价值。

```bash
# 构建全部数据集
python build_dataset.py

# 只构建意图分类数据集（默认开启模板扩写）
python build_dataset.py --only intent

# 关闭模板扩写，仅用知识库真实问句 + 人工种子
python build_dataset.py --only intent --no-augment
```

产物结构、字段含义、类别分布统计见 `data/dataset/README.md`（脚本自动生成）。

### 意图数据的扩充机制（重要）

知识库中真实问句只有几十条，6 分类下每类不到 10 条，直接训练等于让模型背答案。
因此意图数据集采用**三来源混合**：

| 来源（`source` 字段） | 规模 | 说明 |
|---|---|---|
| `auto_keyword` | 10+ | 从知识库挖掘的真实问句，用特征词规则弱标注 |
| `seed` | 100+ | 人工编写的种子问句，覆盖规则命中不到的口语表达 |
| `template` | 500+ | 主体词 × 疑问句式 机械组合（如 `{餐具}怎么清洗消毒`） |

**评估集净化**：模板样本句式分布比真实提问窄，若占满评估集会让分数虚高。
因此切分时会控制模板样本在评估集中的占比，优先用真实样本填充评估集，
同时保证训练集至少保留一半真实样本（避免「掏空训练集」只剩模板）。

## 五、模型训练环节：意图分类

> 需求基线里的「模型训练」环节。输入端是 `data/dataset/intent/`，输出端是
> `models/intent-classifier/`，推理侧（`core/intent_classifier.py`）会自动加载接管。

```bash
# 完整训练（默认 10 轮，基座用本地 bge-m3）
python train_intent.py

# 数据体检：只打印类别/来源分布，不训练
python train_intent.py --dry-run

# 常用参数
python train_intent.py --epochs 15 --batch-size 8     # 调整轮数与批大小
python train_intent.py --device cpu                    # 强制走 CPU
python train_intent.py --lr 3e-5                       # 调整学习率
python train_intent.py --base bge-small-zh-v1.5        # 换更轻量的基座
```

### 数据集职责划分（整个训练链路的重点）

| 数据集 | 来源 | 用途 | 使用次数 |
|--------|------|------|----------|
| 训练集（train） | `intent_train.jsonl` 内切 | 更新参数 | 每轮都用 |
| **验证集（val）** | 从**训练集内部**再切 20% | 每轮评估、早停、**选最优 checkpoint** | 每轮都用 |
| **固定测试集（test）** | `intent_eval.jsonl` | **只在最后评一次**，衡量真实泛化能力 | **全程仅 1 次** |

> 为什么要把 val 和 test 分开：如果反复拿 `intent_eval.jsonl` 调超参，
> 就等于间接过拟合到测试集，最终报出的分数会失真。所以训练全程用 val 选模型，
> 只在收尾时用 test 报一次最终成绩。

### 输出产物

```
models/intent-classifier/
├── config.json                 # 模型配置
├── model.safetensors           # 权重
├── tokenizer.json 等            # 分词器
├── train_report.json           # 训练报告：超参、数据规模、准确率、macro-F1、逐类指标、混淆矩阵
└── _checkpoints/               # 训练中间 checkpoint（可删）
```

训练结束会打印**混淆矩阵**（行=真实标签，列=预测标签），用于定位
「哪个意图最容易被误判成哪个」——这比一个总分更有诊断价值。

### 计算设备自动探测

`base/config.py` 提供 `resolve_device()`，优先级 **cuda > mps(Apple) > xpu(Intel) > cpu**：

```python
from base.config import resolve_device, describe_device
device = resolve_device()          # 自动探测
print(describe_device(device))     # 'cuda (NVIDIA GeForce RTX 4060 Laptop GPU, 显存 8.0GB)'
```

- 配置留空（默认）→ 自动探测；`config.ini` 的 `[model] device` 或环境变量 `MODEL_DEVICE`
  填了值则**强制生效**（用于「有卡但想跑 CPU 基准」这类场景）。
- ⚠️ **常见坑**：装成 `+cpu` 版 torch 时 `torch.cuda.is_available()` 恒为 False。
  版本号**没有** `+cpu` 后缀也不代表就是 CUDA 版，务必用
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 实测。

### 意图识别：规则 → 模型的升级路径

项目**默认**用规则做意图识别（零模型依赖，开箱即用）。训练完模型后，
重启服务即自动接管，**无需改任何业务代码**：

| 阶段 | 意图识别方式 | 需要的动作 |
|------|-------------|-----------|
| 开箱即用 | 特征词规则（`RuleIntentClassifier`） | 无 |
| 已训练模型 | 微调分类模型（`ModelIntentClassifier`） | 运行 `python train_intent.py`，重启 |

加载策略：优先尝试模型目录 → 不存在/依赖缺失/加载失败 → 自动降级为规则，并在日志写明原因。
所以**没训练模型时系统照常跑**，不会因为缺模型而起不来。

意图与检索策略的映射（对应需求基线「6 类意图驱动 4 种检索策略动态路由」）：

| 意图 | 检索策略 | 理由 |
|------|----------|------|
| 法规查询 / SOP 指导 / 营养咨询 / 供应商合规 | 直接检索 | 问题单一明确 |
| 事故处置 | SubQuery | 问句常含「怎么处理 + 上报谁 + 保留什么证据」多个子诉求，拆分检索召回更全 |
| 闲聊 | 直接检索（兜底） | 实际在 `qa_system` 的问候语短路层就被拦下 |

> 注意路由优先级：**口语化/复合形态判断先于意图分类**。
> 「这肉看着有点怪还能用吗」的意图是 SOP 指导，但它真正需要的是 HyDE；
> 「顾客中毒了要怎么处理、要上报谁」的意图是事故处置，需要的是 SubQuery。

## 六、与参考项目（D:/heima/RAG）的约定对齐

| 维度 | 对齐方式 |
|------|----------|
| 目录组织 | 沿用扁平分层：`base/`（基础设施）+ `core/`（核心逻辑）+ 根目录入口脚本 |
| 配置文件 | 沿用 `config.ini` + `.env` 双轨，`base/config.py` 统一读取，`.env` 优先 |
| 技术栈 | Python + LangChain 切分 + Milvus + BGE-M3 + BGE-Reranker + DeepSeek(OpenAI 兼容) + FastAPI |
| 代码风格 | 中文注释、"导入 / 定义 / 使用"分段注释、文件头 docstring 标注职责 |
| 检索实现 | 沿用参考项目 `hybrid_search_with_rerank` 的稠密+稀疏双路召回与 WeightedRanker 粗排写法 |
| 模型路径 | 沿用本地模型目录约定 `models/bge-m3`、`models/bge-reranker-large` |

**差异说明**：本项目按「最小可用」约束做了裁剪——未引入 MySQL / Redis / BM25 三级缓存、
OCR 文档加载器、RAGAS 评估等模块，统一以 `kb_type` 字段在单集合内区分知识库，
替代参考项目的多集合物理隔离。详见 `docs/REQUIREMENTS.md` 第七节。

> 关于模型训练范围：本项目实现了**意图分类模型**的完整训练链路
> （`build_dataset.py` 造数据 → `train_intent.py` 训练 → `core/intent_classifier.py` 自动接管）。
> 重排（rerank）与指令微调（sft）两个数据集**只构建数据、不提供训练脚本**——
> 前者受限于当前数据量（几十条正样本微调 568M 的 Reranker 会过拟合），
> 后者需要 1k+ 条人工/强模型标注数据且本机显存不足，均不适合在当前规模下训练。
> 模型缺失时意图识别自动回退规则，**不训练也能跑通完整问答闭环**。

## 六、快速开始

详见 [`RUN.md`](RUN.md)。最简流程：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 准备模型（复制参考项目的本地模型）
mkdir -p models
cp -r D:/heima/RAG/rag_qa/models/bge-m3 models/
cp -r D:/heima/RAG/rag_qa/models/bge-reranker-large models/

# 构建知识库
python build_kb.py

# 构建训练数据集（意图分类默认开启模板扩写）
python build_dataset.py

# 训练意图分类模型（可选；训练后重启服务自动接管意图识别）
python train_intent.py

# 验证（二选一）
python run_cli.py -q "这批鸡胸肉到货温度8℃还能用吗"
python app.py          # 浏览器访问 http://127.0.0.1:8080
```

> PowerShell 用户注意：PowerShell 5.1 不支持 `&&`，请逐条执行；模型复制用
> `New-Item -ItemType Directory -Force -Path models` +
> `Copy-Item -Recurse -Force D:\heima\RAG\rag_qa\models\bge-m3 models\`。
> 完整双终端写法见 [`RUN.md`](RUN.md)。
