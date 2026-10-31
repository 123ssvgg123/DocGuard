# 食品安全知识库系统 · 运行说明

## 环境要求

- Python 3.10（与参考项目 D:/heima/RAG 保持一致）
- Milvus 2.5（本机 19530 端口，需预先创建数据库 `foodsafety`）
- 本地模型目录 `models/`（从参考项目复制，见下方「模型准备」）
- 大模型：DeepSeek 云端 API（默认 `deepseek-chat`），需联网
- GPU（可选）：训练意图分类模型时有 NVIDIA 显卡会快很多，没有也能用 CPU 跑

> **⚠️ 终端选择很重要**：本文命令给了 **Git Bash** 与 **PowerShell** 两套写法。
> PowerShell 5.1 **不支持 `&&` 串联命令**（会报 `标记“&&”不是此版本中的有效语句分隔符`），
> 也不支持 `mkdir -p` / `cp -r` 这类 Unix 语法。请按自己所用终端选对应写法。

## 依赖安装

**Git Bash / CMD 兼容写法：**

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**PowerShell 写法（命令是单条，无需改）：**

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

> 若清华源对个别包返回 403，可改用阿里源：
> `pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/`

### torch 需要单独装（选 CUDA 还是 CPU）

`requirements.txt` 里虽然锁了 `torch==2.10.0`，但**构建版本（CUDA / CPU）要按你的显卡单独装**，
这是 PyTorch 官方的要求。装完务必实测：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**有 NVIDIA 显卡（推荐，训练/检索都快）：**

```bash
# 按驱动支持的 CUDA 版本选一条（用 nvidia-smi 查看驱动支持的最高 CUDA 版本）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129
```

**只用 CPU：**

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

期望输出形如：`2.10.0+cu129 True`（CUDA 版）或 `2.10.0+cpu False`（CPU 版）。

> ⚠️ 版本号**没有** `+cpu` 后缀也不代表就是 CUDA 版。务必用上面的命令实测
> `torch.cuda.is_available()`，否则容易出现「以为在用 GPU、其实一直跑在 CPU」。
>
> 项目代码不依赖具体是哪个构建：`base/config.py` 的 `resolve_device()` 会自动探测，
> 有可用 GPU 就用 GPU，否则退 CPU，无需改代码。

## 模型准备

本项目复用参考项目 `D:/heima/RAG/rag_qa/models/` 下的本地模型。

**Git Bash 写法：**

```bash
mkdir -p models
cp -r D:/heima/RAG/rag_qa/models/bge-m3 models/
cp -r D:/heima/RAG/rag_qa/models/bge-reranker-large models/
```

**PowerShell 写法（逐条执行，不要用 `&&` 串联）：**

```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
Copy-Item -Recurse -Force D:\heima\RAG\rag_qa\models\bge-m3 models\
Copy-Item -Recurse -Force D:\heima\RAG\rag_qa\models\bge-reranker-large models\
```

> 如果只想把 Terminal 切到 Git Bash 一劳永逸：VS Code/PyCharm 的终端下拉箭头 →
> Select Default Profile → Git Bash，之后本文所有 `bash` 写法都能直接用。

模型目录结构：

```
models/
├── bge-m3/                 # Embedding 模型（稠密 1024 维 + 稀疏向量）
└── bge-reranker-large/     # Rerank 重排模型
```

## Milvus 数据库准备

本项目默认使用数据库 `foodsafety`。若 Milvus 中尚未创建，需先创建（Milvus 默认库为 default）：

```python
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
client.create_database("foodsafety")
```

或直接在 `config.ini` / `.env` 中把 `database_name` 改为已有数据库名（如 `default`）：

**Git Bash：**

```bash
echo "MILVUS_DATABASE_NAME=default" >> .env
```

**PowerShell：**

```powershell
Add-Content -Path ".env" -Value "MILVUS_DATABASE_NAME=default"
```

## 构建知识库

```bash
# 增量导入 data/ 下全部知识库
python build_kb.py

# 全量重建（先删集合再导入）
python build_kb.py --rebuild

# 只导入单个知识库
python build_kb.py --dir regulation_data
```

> 上述命令只调用 `python`，Git Bash / PowerShell 写法一致。

### ⚠️ 什么时候必须 `--rebuild`（重要）

向量库里的每一块都是**切分那一刻的产物**。切分策略一变，新旧块就不是同一套东西了，
此时用增量模式会留下**旧块残留**，检索时会和正确的新块一起参与打分，导致命中率异常。

| 你改了什么 | 是否需要 `--rebuild` |
|------------|---------------------|
| 只改 `data/` 下文档的正文内容 | 不需要（同 ID 覆盖），但改过切分规则后建议加 `--rebuild` 稳妥 |
| 改了 `core/text_splitter.py` 的**分隔符**（如 FAQ 的 `FAQ_SEPARATORS`） | **必须** `--rebuild` |
| 改了**块大小 / 重叠**（`PARENT_CHUNK_SIZE`、`CHILD_CHUNK_SIZE`、`CHILD_CHUNK_SIZE_OVERRIDE`） | **必须** `--rebuild` |
| 改了 Embedding 模型或向量维度 | **必须** `--rebuild` |
| 只改了提示词 / 阈值 / LLM 配置 | 不需要（改完**重启服务**即可） |

## 构建训练数据集（可选）

如果后续要做模型微调（意图分类 / 重排 / 对话模型），先把知识库转成训练数据：

```bash
# 构建全部数据集（意图分类 + 重排 + 指令微调）
python build_dataset.py

# 只构建某一类
python build_dataset.py --only intent    # 只构建意图分类数据集（默认开启模板扩写）
python build_dataset.py --only rerank    # 只构建重排数据集
python build_dataset.py --only sft       # 只构建指令微调数据集

# 关闭模板扩写（只用知识库真实问句 + 人工种子，数据量会小很多）
python build_dataset.py --only intent --no-augment

# 调整评估集占比（默认 0.2）
python build_dataset.py --eval-ratio 0.3
```

> 该脚本**不调用大模型**，纯本地文本处理，秒级完成；产物在 `data/dataset/` 下，
> 该目录已加入 `.gitignore`（可随时从 `data/` 一键重建）。
> 详细格式说明见 `data/dataset/README.md`（脚本自动生成）。

**意图数据的三个来源**（`source` 字段）：`auto_keyword`（知识库真实问句）、
`seed`（人工种子）、`template`（主体词 × 句式扩写）。扩写把意图样本从几十条
提升到 600+ 条，让 6 分类训练具备可行性。

## 训练意图分类模型（可选）

> 需求基线的「模型训练」环节。输入 `data/dataset/intent/`，输出 `models/intent-classifier/`。

```bash
# 先体检数据（打印类别分布与来源分布，不训练）
python train_intent.py --dry-run

# 完整训练（默认 10 轮，基座用本地 models/bge-m3）
python train_intent.py

# 常用参数
python train_intent.py --epochs 15            # 训练轮数
python train_intent.py --batch-size 8         # 批大小（CPU 训练建议调小）
python train_intent.py --lr 3e-5              # 学习率
python train_intent.py --device cpu           # 强制指定设备
python train_intent.py --base bge-small-zh-v1.5   # 换更轻量的基座（需先放入 models/）
python train_intent.py --output models/my-intent # 自定义输出目录
```

**训练输出**（`models/intent-classifier/`）：

```
config.json / model.safetensors / tokenizer 相关文件
train_report.json    # 超参、数据规模、准确率、macro-F1、逐类指标、混淆矩阵
_checkpoints/        # 中间 checkpoint（可删）
```

**训练时会打印**：每轮验证集指标（accuracy / macro-F1）→ 最终测试集指标 →
**混淆矩阵**（定位「哪个意图容易被误判成哪个」）。

### 训练集 / 验证集 / 测试集三者关系（重要）

| 集合 | 来源 | 用途 | 使用次数 |
|------|------|------|----------|
| 训练集 | `intent_train.jsonl` 内切 80% | 更新参数 | 每轮 |
| **验证集** | 从**训练集内部**再切 20% | 早停、**选最优 checkpoint** | 每轮 |
| **固定测试集** | `intent_eval.jsonl` | **只在最后评一次** | **全程 1 次** |

不要拿 `intent_eval.jsonl` 反复调超参——那等于间接过拟合，最后报的分数不可信。

### 训练设备自动探测

本项目不把设备写死。`base/config.py` 的 `resolve_device()` 会运行时探测：
**cuda > mps > xpu > cpu**。

```bash
# 查看当前会用什么设备
python base/config.py
# 输出：MODEL_DEVICE = cuda -> cuda (NVIDIA GeForce RTX 4060 Laptop GPU, 显存 8.0GB)
```

`config.ini` 的 `[model] device` 留空 = 自动探测；填 `cpu` / `cuda` 可强制指定。

> ⚠️ **必查**：装成 `+cpu` 版 torch 时 `torch.cuda.is_available()` 恒为 False。
> 版本号**没有** `+cpu` 后缀也不代表是 CUDA 版，务必实测：
> `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### 训练报 Segfault / 进程崩溃？

脚本已内置两处稳定性设置（`dataloader_num_workers=0`、CPU 时 `torch.set_num_threads(4)`），
若仍崩溃，按以下顺序排查：

1. **确认不是资源不足**。bge-m3 是 568M 参数的模型，训练时反向传播需要数 GB 内存。
   在内存吃紧或受限环境（容器、CI 沙箱）里容易直接被 OOM Killer 杀掉（表现为 Segfault 或静默退出）。
   先关闭其他占内存的程序（Docker Desktop、Milvus、浏览器等）再试。

2. **手动限制线程数**：

```bash
# Linux / Git Bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train_intent.py

# PowerShell
$env:OMP_NUM_THREADS=2; $env:MKL_NUM_THREADS=2; python train_intent.py
```

3. **换更轻的基座**（最有效）。568M 的基座在 CPU 上既是速度瓶颈也是内存瓶颈。
   下载 `bge-small-zh-v1.5`（约 95M）放入 `models/`，然后：

```bash
python train_intent.py --base bge-small-zh-v1.5
```

4. **在真实终端（非沙箱/容器）里跑**。部分受限执行环境对 torch 的多线程内存映射有限制，
   会在训练循环中段错误；普通终端没有这个问题。

### 关于重排与指令微调

`build_dataset.py` 会一并产出 `rerank/` 与 `sft/` 数据，但**本项目未提供对应训练脚本**：

- **重排**：当前只有几十组正样本，微调 568M 的 bge-reranker-large 会过拟合，
  效果可能不如原版。需先把数据扩到 500+ 组再考虑训练。
- **指令微调**：需 1k+ 条高质量标注数据，且本机显存不足以 LoRA 微调 7B 模型。
  系统走 RAG 路线（检索 → 拼上下文 → DeepSeek 生成），无需微调本地对话模型。

**关于意图识别模型的接入**：项目默认走规则意图识别，无需任何模型即可运行。
`train_intent.py` 训练完成后模型自动落到 `models/intent-classifier/`，重启服务即接管；
模型不存在、依赖缺失或加载失败时会**自动回退到规则**，不影响问答功能。

## 验证问答效果

### 方式一：命令行（推荐先验证）

```bash
# 单次提问
python run_cli.py -q "这批鸡胸肉到货温度8℃还能用吗"

# 指定知识库
python run_cli.py -q "冰箱冷藏温度标准是多少" -k sop

# 查看检索来源片段
python run_cli.py -q "食品留样要留多少克" --debug

# 交互式问答
python run_cli.py
```

### 方式二：Web 页面

```bash
python app.py
```

浏览器打开 http://127.0.0.1:8080 ，在输入框提问，回答下方会展示引用的知识库来源。

### 方式三：HTTP 接口

```bash
curl -X POST http://127.0.0.1:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"冷藏食品到货温度标准是多少","kb_type":"regulation"}'
```

**PowerShell 写法**：`curl` 是 `Invoke-WebRequest` 的别名，且单引号内 JSON 会被吃掉，
用 `curl.exe` + here-string 更稳：

```powershell
$body = '{"query":"冷藏食品到货温度标准是多少","kb_type":"regulation"}'
curl.exe -X POST http://127.0.0.1:8080/api/query -H "Content-Type: application/json" -d $body
```

## 配置修改位置

所有关键配置集中在两处（优先级：`.env` > `config.ini` > 代码默认值）：

| 配置项 | config.ini 位置 | .env 变量 |
|--------|----------------|-----------|
| LLM 模型 / API Key | `[llm]` | `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` |
| Milvus 地址 / 库名 / 集合名 | `[milvus]` | `MILVUS_HOST` 等 |
| Embedding / Rerank / 意图分类 模型名 | `[model]` | `EMBEDDING_MODEL` / `RERANK_MODEL` / `INTENT_MODEL` |
| 分块大小 / 召回数 / **拒答阈值** / **检索簇数** | `[retrieval]` | `PARENT_CHUNK_SIZE` / **`SCORE_THRESHOLD`** / **`NPROBE`** 等 |
| 服务端口 / 咨询热线 | `[server]` `[app]` | `APP_PORT` / `SERVICE_PHONE` |

### 检索精度调节：nprobe（召回率 vs 延迟）

稠密向量用 **IVF_FLAT** 索引 + `nlist=128`（把向量聚成 128 个簇）。
检索时 `nprobe` 决定实际扫描几个簇 —— **它直接影响召回率**：

| nprobe | 扫描比例 | 效果 |
|--------|---------|------|
| 10（旧值） | 8% | 快，但会漏掉落在未扫描簇里的正确答案 |
| **32（当前）** | 25% | 召回率明显提升，延迟增加可忽略 —— **推荐值** |
| 64 | 50% | 更全，数据量大时开始有延迟感 |
| 128 | 100% | 等价暴力检索，最准但最慢 |

经验公式：**`nprobe` ≈ `nlist` 的 1/4 ~ 1/3**。改这个值**不需要重建库**（它只作用于检索时），
改完**重启服务**即可生效。

### 拒答机制（两层防线）

系统对"知识库答不了的问题"有**两道拦截**，都不会报错，而是返回礼貌拒答话术：

| 层 | 位置 | 触发条件 | 效果 |
|----|------|----------|------|
| 1. 阈值拒答 | `config.ini` `score_threshold`（当前 **0.5**） | Rerank 最高分低于阈值 | 判定"未命中"，**不调用 LLM**，直接回复"知识库中暂无与该问题相关的权威依据…" |
| 2. 提示词硬约束 | `core/prompts.py` 第 4、6 条 | 检索到了内容但**答不对题**（如追问具体天数/数值，库里只有定性描述） | 由 LLM 判断后回复"知识库中暂无该数值的权威依据…" |

**阈值调节建议**：BGE-Reranker 输出 0~1 分数。
- `0.3` 偏松 → 边缘问题容易被"沾边条款"蒙混，幻觉风险高；
- `0.5`（当前）→ 平衡值，推荐起点；
- `0.6~0.7` → 更保守，适合合规问答等零容忍场景，代价是可能误拒边缘问题。

> 注：第 2 层是纯提示词约束，用于堵住"库里有相关条款但缺关键数值"这类阈值拦不住的灰色情况
> （典型案例：问"熟鸡蛋放几天不能吃"，库里只有"熟食常温超 2 小时不得供餐"——
> 阈值过得了，但语料里没有天数，此时靠第 6 条硬约束拒答）。

## 常见问题

**Q: 报错 `标记“&&”不是此版本中的有效语句分隔符`？**
你在 PowerShell 5.1 里跑 bash 语法了。三条出路：① 用本文的 PowerShell 写法；
② 把 Terminal 切到 Git Bash；③ 升级到 PowerShell 7（支持 `&&`）。

**Q: 报错连不上 Milvus？**
确认 Milvus 已启动：`docker ps | grep milvus`（PowerShell 用 `docker ps | Select-String milvus`），
并已在容器内创建 `foodsafety` 数据库。

**Q: 浏览器报 `ERR_ADDRESS_INVALID`，提示"无法访问 http://0.0.0.0:8080/"？**
`0.0.0.0` 是服务端**监听地址**（表示监听本机所有网卡），**不是浏览器能访问的地址**。
改在地址栏输入 **http://127.0.0.1:8080** 或 **http://localhost:8080** 即可。
这通常是因为直接照抄了 `config.ini` 里的 `host = 0.0.0.0` 去访问。
服务本身是正常的（能解析出端口说明进程已启动）。

**Q: 首次启动很慢？**
BGE-M3 与 Reranker 模型首次加载需要时间（CPU 环境约 30-60 秒），属正常现象。

**Q: 检索结果为空或总是拒答？**
先确认知识库已构建：`python build_kb.py`，并检查集合内向量数量（构建脚本会打印）。

**Q: 某个问题明明是库里的，却被拒答了（rerank 分数低到 0.0x）？**
踩坑记录，按顺序排查：

1. **服务没重启**。改完 `.py` 代码后不重启，进程内存里跑的还是旧模块，
   磁盘代码再对也不生效。先 `Ctrl+C` 再 `python app.py`。
2. **切分把问句和答句切散了**（最隐蔽）。典型症状：命中"只有问题没答案"的残块，
   rerank 分数极低（实测 0.0151），日志里还能看到 `子块 len=2: 'Q:'` 这类垃圾碎片。
   根因是 `core/text_splitter.py` 的 `FAQ_SEPARATORS` 里多写了 `A/答`——
   把 `\nA[:：]` 当分隔符就会把一问一答劈成两块。**只按 `Q/问` 切**，让 Q+A 落在同一块。
   同时 FAQ 要用 `CHILD_CHUNK_SIZE_OVERRIDE = {'faq': 512}` 放大子块，
   否则默认 `chunk_size=120` 装不下一个完整问答对（约 200 字），照样会被切碎。
   改完这两处**必须** `python build_kb.py --rebuild`。
3. **阈值定太高**。看日志里打印的"最高 rerank 分数"：若分数正常（如 0.7+）却仍拒答，
   检查 `score_threshold` 是不是被调到了 0.8 之类；若分数本身就是 0.0x，那是检索/切分问题，
   调低阈值治标不治本。

> 自查命令：`python run_cli.py -q "你的问题" --debug`，会打印每条的 rerank 分数与来源片段。

**Q: 想换云端模型？**
只改 `.env` 中 `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` 即可，无需改代码。
当前默认接入 DeepSeek（`deepseek-chat`，https://api.deepseek.com/v1 ）。
如需切换其他供应商，把 `LLM_BASE_URL` 换成对应的 OpenAI 兼容地址即可（如通义千问百炼平台的
compatible-mode 地址），模型名同步替换。

**Q: 日志里打印「未找到意图分类模型目录 …，使用规则分类器」，是报错吗？**
不是报错，是**正常降级提示**。项目默认用规则做意图识别，不需要任何模型。
`python build_dataset.py` 生成的 `data/dataset/intent/` 只是**训练数据**，不是模型本体。
要启用模型分类，需要先拿这份数据训练一个分类模型，再把产物放到 `models/intent-classifier/`。
在此之前，规则分类器会照常工作，问答功能完全不受影响。

**Q: `python build_dataset.py` 报某个意图类别为 0？**
说明 `data/` 知识库里缺少该类问句。比如「闲聊」类在知识库中天然不存在（知识库不会有「你好」），
所以由人工编写的种子样本（`core/dataset_builder.py` 的 `SEED_QUERIES`）补充。
若业务类别缺失，往 `data/faq_data/` 增补对应主题的问答对后重跑脚本即可。

---

## 附：密钥与运行策略（2026-10-30 更新）

### 1. 密钥已更换（名称 = tool）

对话模型仍然接 DeepSeek，密钥换成新的一把，名称标记为 `tool`。
代码里所有出现位置都已同步，换 key 时按这份清单改，不会漏：

| 位置 | 键名 | 说明 |
|---|---|---|
| `.env` | `LLM_API_KEY` | 运行时实际生效的密钥 |
| `.env` | `LLM_API_KEY_NAME` | 密钥名称（仅标识，不参与鉴权） |
| `config.ini` | `[llm] api_key` | 没有 `.env` 时的兜底 |
| `config.ini` | `[llm] api_key_name` | 同上，名称标识 |

优先级：环境变量 / `.env` > `config.ini` > 代码 fallback。
换 key 只改 `.env` 一处即可生效，其余两处是兜底与留痕。

### 2. 运行策略：只做最小验证，不跑完整一轮

对话模型是 **DeepSeek 远程 API（非本地部署、按 token 计费）**，
embedding / rerank 才是本地权重。为避免一次运行把额度跑空，
新增硬限制模块 **`base/run_policy.py`**，默认生效、不可绕过（除显式双闸门）：

| 限制 | 默认值 | 环境变量 |
|---|---|---|
| 单次运行样本上限 | 2 条 | `SMOKE_MAX_SAMPLES` |
| 单次运行远程调用预算 | 6 次 | `SMOKE_MAX_LLM_CALLS` |
| 全量运行 | 禁止 | `ALLOW_FULL_RUN=0` |

落地位置：

- `QASystem.call_llm`：每次远程调用前扣预算，超限即停止生成并提示，不再发请求
- `scripts/evaluate_ragas.py`：`--sample` 自动压到 2；传 `--allow-full` 也会被拒
- `scripts/eval_ragas_rejudge.py`：`--limit` 默认即 2
- `app.py`：启动时打印当前策略横幅

确实需要全量时，两道闸门必须同时打开（跑完记得改回）：

```bash
# Windows
set ALLOW_FULL_RUN=1
python scripts/evaluate_ragas.py --sample 20 --allow-full
# 跑完恢复
set ALLOW_FULL_RUN=0
```

> 只开环境变量不传 `--allow-full`，或只传 `--allow-full` 不开环境变量，都会被拒绝。
> 这样设计是为了防止「手滑跑全量」——两个动作都做，说明人确实是有意的。

### 3. LLM 熔断：失败话术不允许被当成检索词（2026-10-30 补充）

预算闸门只解决了「跑几条」，没解决「一条要打几次」。复盘 08:30 那轮
20 条样本打出 48 次请求、其中大量是无效重试，根因是**降级话术污染检索链路**：

```
call_llm 402 → yield「抱歉，服务暂时不可用…」
            → _retrieve_with_subqueries 把它当子查询去检索（必然空召回）
            → 分数低 → 触发 Backtrack → 再打一次 LLM → 再 402 → 循环
```

为此新增 **`base/llm_guard.py`**：

| 能力 | 说明 |
|---|---|
| `is_fallback_text()` | 识别降级话术（含预算耗尽文案），不当有效输出用 |
| `LLMUnavailable` | 明确表示「本次运行内 LLM 不可用」，不再伪装成正常文本 |
| `breaker.check()` | 熔断中直接抛异常，**不发 HTTP 请求** |
| `breaker.trip()` | 首次失败即熔断（fail-fast），避免逐条试探 |

落地位置：

- `core/rag_system.py` 新增 `_llm_text()`：HyDE / SubQuery / Backtrack 三处
  改写调用全部改走它，命中兜底话术或异常即抛 `LLMUnavailable`
  → HyDE/SubQuery 回退 direct 检索，Backtrack 立即终止本轮
- `core/rag_system.py:_maybe_backtrack` 入口检查熔断，已熔断直接跳过回溯
- `qa_system.py:call_llm` 在熔断期间直接返回话术、不再发请求；失败时 `trip()`
- `scripts/evaluate_ragas.py`：采集中途检测到熔断即 `break`，不再空跑剩余样本；
  收尾打印熔断状态，明确标注「本次分数不可用于对外口径」

效果对比（20 条样本）：

| | 修复前 | 修复后 |
|---|---|---|
| 余额不足时的请求数 | 48 次（逐条撞 402） | 1 次（第 2 条起熔断拦截） |
| 无效本地检索 | 20 条全跑（每条 2~6s） | 第 2 条即停 |
| 分数可信度 | 静默产出无意义数值 | 显式标注「不可对外」 |

