"""
    配置管理模块：base/config.py

    统一管理全项目配置，读取优先级：
        环境变量(.env / shell) > config.ini > 代码内 fallback 默认值

    说明：所有关键配置（模型、向量库、路径、检索参数）都集中在此，
        业务代码一律通过 `from base.config import config` 引用，不散落硬编码。
"""
# 导入配置解析库
import configparser
# 导入路径操作库
import os

# 项目根目录（base/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 尝试加载 .env（未安装 python-dotenv 时静默跳过，不影响 config.ini 读取）
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except ImportError:
    pass


# ==================================================================
# 多用户内容隔离（需求 4.2）的默认权限模型
# ==================================================================
# 格式约定（config.ini 的 [access] 段用同一套格式，便于运维直接改配置）：
#     role_levels      = 角色:密级1|密级2;角色2:...
#     role_departments = 角色:部门1|部门2;...   （留空表示只可见 all 部门的公共内容）
#     kb_defaults      = 知识库:默认密级:归属部门;...
#
# 密级语义：public（对外可查，如国标）< internal（内部规范，如 SOP）
#           < restricted（受限，如事故调查案例）
_DEFAULT_ROLE_LEVELS = (
    'public:public;'
    'staff:public|internal;'
    'kitchen:public|internal;'
    'quality:public|internal|restricted;'
    'supplier:public|internal;'
    'admin:public|internal|restricted'
)
_DEFAULT_ROLE_DEPARTMENTS = (
    'public:;'
    'staff:;'
    'kitchen:kitchen;'
    'quality:quality;'
    'supplier:purchasing;'
    'admin:quality|kitchen|purchasing|management'
)
_DEFAULT_KB_DEFAULTS = (
    'regulation:public:all;'
    'sop:internal:all;'
    'case:restricted:quality;'
    'nutrition:public:all;'
    'supplier:internal:purchasing;'
    'faq:public:all'
)


def _parse_pipe_map(raw, fallback_raw):
    """
    解析 `k:v1|v2;k2:v3` 形式的一行配置为 {k: (v1, v2)}。

    为什么用这种紧凑格式而不是多行 INI：
        configparser 不支持嵌套结构，写多行又要自定义解析；
        「分号分项、冒号分键值、竖线分列表」一行能表达完，且好读好改。
    解析失败（或结果为空）时回退到代码内默认值，保证配置写错不会让服务起不来。
    """
    def _parse(text):
        out = {}
        for item in str(text or '').split(';'):
            item = item.strip()
            if not item or ':' not in item:
                continue
            key, _, values = item.partition(':')
            key = key.strip().lower()
            vals = tuple(v.strip().lower() for v in values.split('|') if v.strip())
            if key:
                out[key] = vals
        return out

    parsed = _parse(raw)
    return parsed or _parse(fallback_raw)


def _parse_kb_defaults(raw, fallback_raw):
    """解析 `kb_type:access_level:department;...` → {kb_type: (密级, 部门)}"""
    def _parse(text):
        out = {}
        for item in str(text or '').split(';'):
            parts = [p.strip().lower() for p in item.split(':') if p.strip()]
            if len(parts) == 3:
                out[parts[0]] = (parts[1], parts[2])
        return out

    parsed = _parse(raw)
    return parsed or _parse(fallback_raw)


class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=None):
        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 项目根目录
        self.PROJECT_ROOT = PROJECT_ROOT
        # 日志目录
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        # 数据目录
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'data')
        # 模型目录
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, 'models')
        # 静态资源目录（前端问答页面）
        self.STATIC_DIR = os.path.join(self.PROJECT_ROOT, 'static')

        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')
        # 读取配置文件 config.ini
        self.config.read(config_file, encoding='utf-8')

        # --> 从.env文件中获取配置，找不到则从config.ini中获取，再找不到使用 fallback 备选值
        # 服务器配置
        # 监听地址
        self.APP_HOST = os.getenv('APP_HOST', self.config.get('server', 'host', fallback='0.0.0.0'))
        # 监听端口
        self.APP_PORT = int(os.getenv('APP_PORT', self.config.getint('server', 'port', fallback=8080)))

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = int(os.getenv('MILVUS_PORT', self.config.getint('milvus', 'port', fallback=19530)))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='foodsafety'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name',
                                                                fallback='foodsafety_kb'))

        # LLM 配置（DeepSeek，OpenAI 兼容接口）
        # LLM 模型名
        self.LLM_MODEL = os.getenv('LLM_MODEL', self.config.get('llm', 'model', fallback='deepseek-chat'))
        # LLM API 密钥
        self.LLM_API_KEY = os.getenv('LLM_API_KEY', self.config.get('llm', 'api_key', fallback=''))
        # LLM API 密钥的「名称」（DeepSeek 控制台上的 key 名称，仅作标识，不参与鉴权）
        # 换 key 时同步改这里，便于排查「这条日志是哪把 key 打的」
        self.LLM_API_KEY_NAME = os.getenv(
            'LLM_API_KEY_NAME', self.config.get('llm', 'api_key_name', fallback='tool'))
        # LLM API 地址
        self.LLM_BASE_URL = os.getenv('LLM_BASE_URL',
                                      self.config.get('llm', 'base_url',
                                                      fallback='https://api.deepseek.com/v1'))

        # ==================== 运行策略（远程模型最小验证） ====================
        # ⚠️ 对话模型是 DeepSeek 远程 API，不是本地部署、按 token 计费。
        #    因此默认只做「最小验证」，不跑完整一轮全量流程；硬限制在
        #    base/run_policy.py 里强制执行（样本上限 / 调用预算 / 全量闸门）。
        # 是否仅允许最小验证：默认 True，只有 ALLOW_FULL_RUN=1 才放开全量
        self.ALLOW_FULL_RUN = os.getenv(
            'ALLOW_FULL_RUN', self.config.get('run_policy', 'allow_full_run', fallback='0')
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        # 单次运行样本上限（条）
        self.SMOKE_MAX_SAMPLES = int(os.getenv(
            'SMOKE_MAX_SAMPLES',
            self.config.getint('run_policy', 'max_samples', fallback=2)))
        # 单次运行远程模型调用上限（次）
        self.SMOKE_MAX_LLM_CALLS = int(os.getenv(
            'SMOKE_MAX_LLM_CALLS',
            self.config.getint('run_policy', 'max_llm_calls', fallback=6)))

        # 模型配置
        # Embedding 模型名（models/ 下的子目录名）
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', self.config.get('model', 'embedding_model', fallback='bge-m3'))
        # Rerank 模型名（models/ 下的子目录名）
        self.RERANK_MODEL = os.getenv('RERANK_MODEL', self.config.get('model', 'rerank_model',
                                                                     fallback='bge-reranker-large'))
        # 推理/训练设备：由 resolve_device() 在运行时决定最终值。
        # 优先级：环境变量 MODEL_DEVICE（非空才生效）> config.ini [model] device > 自动探测
        #
        # ⚠️ 这里必须用 or 兜底：.env 里常见的写法是 `MODEL_DEVICE=`（留空占位），
        #    此时 os.getenv 返回空串 ''，若直接采用就会把 config.ini 里显式配置的
        #    设备（如 cpu）覆盖成「未配置」，进而走自动探测拿到 cuda —— 表现为
        #    「改了 config.ini 却不生效」。故空串统一视为「未设置」。
        self.MODEL_DEVICE = (
            os.getenv('MODEL_DEVICE', '').strip()
            or self.config.get('model', 'device', fallback='').strip()
        )
        # Embedding 模型绝对路径
        self.EMBEDDING_MODEL_PATH = os.path.join(self.MODELS_DIR, self.EMBEDDING_MODEL)
        # Rerank 模型绝对路径
        self.RERANK_MODEL_PATH = os.path.join(self.MODELS_DIR, self.RERANK_MODEL)
        # 意图分类模型名（models/ 下的子目录名）
        # 说明：该目录不存在时系统自动回退到规则意图识别，不影响问答功能。
        #      训练数据集见 data/dataset/intent/，训练好后把模型放入该目录即可自动接管。
        self.INTENT_MODEL = os.getenv('INTENT_MODEL', self.config.get('model', 'intent_model',
                                                                     fallback='intent-classifier'))
        # 意图分类模型绝对路径
        self.INTENT_MODEL_PATH = os.path.join(self.MODELS_DIR, self.INTENT_MODEL)

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = int(os.getenv('PARENT_CHUNK_SIZE',
                                               self.config.getint('retrieval', 'parent_chunk_size', fallback=512)))
        # 子块大小
        self.CHILD_CHUNK_SIZE = int(os.getenv('CHILD_CHUNK_SIZE',
                                              self.config.getint('retrieval', 'child_chunk_size', fallback=120)))
        # 父块重叠大小
        self.PARENT_CHUNK_OVERLAP = int(os.getenv('PARENT_CHUNK_OVERLAP',
                                                  self.config.getint('retrieval', 'parent_chunk_overlap',
                                                                     fallback=128)))
        # 子块重叠大小
        self.CHILD_CHUNK_OVERLAP = int(os.getenv('CHILD_CHUNK_OVERLAP',
                                                 self.config.getint('retrieval', 'child_chunk_overlap', fallback=30)))
        # 混合检索召回数量
        self.RETRIEVAL_K = int(os.getenv('RETRIEVAL_K',
                                         self.config.getint('retrieval', 'retrieval_k', fallback=5)))
        # 重排后最终候选父块数量
        self.CANDIDATE_M = int(os.getenv('CANDIDATE_M',
                                         self.config.getint('retrieval', 'candidate_m', fallback=3)))
        # 相似度阈值：低于该分数视为未命中，触发拒答
        self.SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD',
                                               self.config.getfloat('retrieval', 'score_threshold', fallback=0.5)))
        # IVF_FLAT 检索时扫描的聚类簇数（nlist=128 时建议 1/4~1/3，即 32~43；越大越准越慢）
        self.NPROBE = int(os.getenv('NPROBE',
                                    self.config.getint('retrieval', 'nprobe', fallback=32)))

        # Backtrack 回溯检索策略
        # 说明：首轮最高分低于 trigger_threshold 时，把问题改写成更贴近法规条文的
        #       检索式再检索一轮，避免「表述差异」导致直接拒答。
        self.BACKTRACK_ENABLED = os.getenv(
            'BACKTRACK_ENABLED',
            self.config.get('backtrack', 'enabled', fallback='true')).lower() == 'true'
        # 触发回溯的分数阈值
        self.BACKTRACK_TRIGGER_THRESHOLD = float(os.getenv(
            'BACKTRACK_TRIGGER_THRESHOLD',
            self.config.getfloat('backtrack', 'trigger_threshold', fallback=0.45)))
        # 最大回溯轮数
        self.BACKTRACK_MAX_ROUNDS = int(os.getenv(
            'BACKTRACK_MAX_ROUNDS',
            self.config.getint('backtrack', 'max_rounds', fallback=2)))
        # 兜底线：回溯后仍低于此分则放弃（拒答）
        self.BACKTRACK_GIVE_UP_THRESHOLD = float(os.getenv(
            'BACKTRACK_GIVE_UP_THRESHOLD',
            self.config.getfloat('backtrack', 'give_up_threshold', fallback=0.30)))
        # 回溯时是否参考上一轮的低分片段
        self.BACKTRACK_USE_PRIOR_DOCS = os.getenv(
            'BACKTRACK_USE_PRIOR_DOCS',
            self.config.get('backtrack', 'use_prior_docs', fallback='true')).lower() == 'true'
        # 回溯合并的候选块数量
        self.BACKTRACK_MERGE_K = int(os.getenv(
            'BACKTRACK_MERGE_K',
            self.config.getint('backtrack', 'merge_k', fallback=8)))

        # Redis 三级缓存（L1 进程内 → L2 Redis → L3 回源）
        self.REDIS_ENABLED = os.getenv(
            'REDIS_ENABLED',
            self.config.get('redis', 'enabled', fallback='false')).lower() == 'true'
        self.REDIS_HOST = os.getenv('REDIS_HOST',
                                    self.config.get('redis', 'host', fallback='localhost'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT',
                                        self.config.getint('redis', 'port', fallback=6379)))
        self.REDIS_DB = int(os.getenv('REDIS_DB',
                                      self.config.getint('redis', 'db', fallback=1)))
        self.REDIS_PASSWORD = os.getenv(
            'REDIS_PASSWORD', self.config.get('redis', 'password', fallback=''))
        # 各类缓存的 TTL（秒）
        self.REDIS_EMBEDDING_TTL = int(os.getenv(
            'REDIS_EMBEDDING_TTL',
            self.config.getint('redis', 'embedding_ttl', fallback=86400)))
        self.REDIS_RETRIEVAL_TTL = int(os.getenv(
            'REDIS_RETRIEVAL_TTL',
            self.config.getint('redis', 'retrieval_ttl', fallback=1800)))
        self.REDIS_ANSWER_TTL = int(os.getenv(
            'REDIS_ANSWER_TTL',
            self.config.getint('redis', 'answer_ttl', fallback=3600)))
        self.REDIS_KEY_PREFIX = os.getenv(
            'REDIS_KEY_PREFIX', self.config.get('redis', 'key_prefix', fallback='foodsafety:'))
        self.REDIS_MAX_ENTRIES = int(os.getenv(
            'REDIS_MAX_ENTRIES',
            self.config.getint('redis', 'max_entries', fallback=10000)))

        # RAGAS 评估配置（scripts/evaluate_ragas.py 使用，离线批处理）
        self.RAGAS_ENABLED = os.getenv(
            'RAGAS_ENABLED',
            self.config.get('ragas', 'enabled', fallback='false')).lower() == 'true'
        _eval_file = os.getenv(
            'RAGAS_EVAL_FILE',
            self.config.get('ragas', 'eval_file', fallback='data/eval/ragas_eval_set.jsonl'))
        # 相对路径统一解析到项目根目录
        self.RAGAS_EVAL_FILE = (_eval_file if os.path.isabs(_eval_file)
                                else os.path.join(self.PROJECT_ROOT, _eval_file))
        _out_dir = os.getenv(
            'RAGAS_OUTPUT_DIR',
            self.config.get('ragas', 'output_dir', fallback='data/eval/results'))
        self.RAGAS_OUTPUT_DIR = (_out_dir if os.path.isabs(_out_dir)
                                 else os.path.join(self.PROJECT_ROOT, _out_dir))
        self.RAGAS_SAMPLE_SIZE = int(os.getenv(
            'RAGAS_SAMPLE_SIZE',
            self.config.getint('ragas', 'sample_size', fallback=0)))
        self.RAGAS_CONCURRENCY = int(os.getenv(
            'RAGAS_CONCURRENCY',
            self.config.getint('ragas', 'concurrency', fallback=1)))
        self.RAGAS_JUDGE_MODEL = os.getenv(
            'RAGAS_JUDGE_MODEL', self.config.get('ragas', 'judge_model', fallback='')).strip()
        self.RAGAS_METRICS = [
            m.strip() for m in os.getenv(
                'RAGAS_METRICS',
                self.config.get('ragas', 'metrics',
                                fallback='faithfulness,answer_relevancy,'
                                         'context_precision,context_recall')).split(',')
            if m.strip()
        ]

        # 应用配置
        # 食品安全咨询热线
        self.SERVICE_PHONE = os.getenv('SERVICE_PHONE',
                                       self.config.get('app', 'service_phone', fallback='400-000-0000'))
        # 对话历史轮数上限
        self.HISTORY_ROUNDS = int(os.getenv('HISTORY_ROUNDS',
                                            self.config.getint('app', 'history_rounds', fallback=5)))
        # 提示词最大字符数
        self.MAX_PROMPT_LENGTH = int(os.getenv('MAX_PROMPT_LENGTH',
                                               self.config.getint('app', 'max_prompt_length', fallback=4096)))

        # 数据集输出目录（build_dataset.py 生成的训练数据）
        self.DATASET_DIR = os.path.join(self.DATA_DIR, 'dataset')

        # 日志文件路径
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')

        # 知识库类型（键 = data/ 下目录名去掉 _data 后缀；值 = 中文展示名）
        # 全项目唯一的类型清单来源，app.py 的知识库下拉/清单接口均由此派生
        self.KB_TYPES = {
            'regulation': '法规标准库',
            'sop': 'SOP 规范库',
            'case': '事故处置案例库',
            'nutrition': '营养学知识库',
            'supplier': '供应商资质库',
            'faq': 'FAQ 问答库',
        }

        # ============ 多用户内容隔离（需求 4.2 行级权限过滤）============
        # 角色 → 可见密级集合、角色 → 所属部门、知识库 → 默认密级与归属部门。
        # 这三项是权限模型的唯一来源，由 core/access_control.py 消费，业务代码不直接读。
        self.ACCESS_ENABLED = os.getenv(
            'ACCESS_ENABLED',
            self.config.get('access', 'enabled', fallback='true')).lower() == 'true'
        self.ACCESS_DEFAULT_ROLE = os.getenv(
            'ACCESS_DEFAULT_ROLE',
            self.config.get('access', 'default_role', fallback='staff')).strip().lower()
        self.ACCESS_ROLE_LEVELS = _parse_pipe_map(
            os.getenv('ACCESS_ROLE_LEVELS',
                      self.config.get('access', 'role_levels', fallback=_DEFAULT_ROLE_LEVELS)),
            _DEFAULT_ROLE_LEVELS)
        self.ACCESS_ROLE_DEPARTMENTS = _parse_pipe_map(
            os.getenv('ACCESS_ROLE_DEPARTMENTS',
                      self.config.get('access', 'role_departments',
                                      fallback=_DEFAULT_ROLE_DEPARTMENTS)),
            _DEFAULT_ROLE_DEPARTMENTS)
        self.ACCESS_KB_DEFAULTS = _parse_kb_defaults(
            os.getenv('ACCESS_KB_DEFAULTS',
                      self.config.get('access', 'kb_defaults', fallback=_DEFAULT_KB_DEFAULTS)),
            _DEFAULT_KB_DEFAULTS)
        # 兜底角色必须在映射表里，否则会出现「默认角色看不到任何东西」这种很难查的现象。
        # 这里不做静默兜底：找不到就用 staff，再找不到就取映射表里的第一个角色。
        if self.ACCESS_DEFAULT_ROLE not in self.ACCESS_ROLE_LEVELS:
            self.ACCESS_DEFAULT_ROLE = ('staff' if 'staff' in self.ACCESS_ROLE_LEVELS
                                        else next(iter(self.ACCESS_ROLE_LEVELS), 'staff'))

        # ============ 知识库增量更新的清单文件（core/incremental.py）============
        # 记录每个源文件的内容哈希与其写入 Milvus 的主键，用于算出「新增/变更/删除」。
        self.KB_MANIFEST_FILE = os.path.join(self.DATA_DIR, '.kb_manifest.json')

        # 训练相关配置（train_intent.py 使用）
        # 意图分类训练基座模型路径：默认复用本地 models/bge-m3，无需联网下载。
        # 配置值填 'auto' 或留空 = 用本地 embedding 模型；填相对名（如 bge-small-zh-v1.5）
        # 则解析为 models/ 下的同名目录；填绝对路径则原样使用。
        _base = os.getenv('INTENT_BASE_MODEL',
                          self.config.get('train', 'intent_base_model', fallback='auto')).strip()
        if not _base or _base.lower() == 'auto':
            self.INTENT_BASE_MODEL = self.EMBEDDING_MODEL_PATH
        elif os.path.isabs(_base):
            self.INTENT_BASE_MODEL = _base
        else:
            self.INTENT_BASE_MODEL = os.path.join(self.MODELS_DIR, _base)
        # 训练轮数
        self.TRAIN_EPOCHS = int(os.getenv('TRAIN_EPOCHS',
                                          self.config.getint('train', 'epochs', fallback=10)))
        # 学习率
        self.TRAIN_LR = float(os.getenv('TRAIN_LR',
                                        self.config.getfloat('train', 'learning_rate', fallback=2e-5)))
        # 训练批大小
        self.TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE',
                                              self.config.getint('train', 'batch_size', fallback=8)))
        # 从训练集内部再切出的验证集占比（用于早停与选最优 checkpoint，
        # 与 intent_eval.jsonl 这个固定测试集严格区分，避免污染测试集）
        self.TRAIN_VAL_RATIO = float(os.getenv('TRAIN_VAL_RATIO',
                                               self.config.getfloat('train', 'val_ratio', fallback=0.2)))
        # 随机种子
        self.TRAIN_SEED = int(os.getenv('TRAIN_SEED',
                                        self.config.getint('train', 'seed', fallback=42)))


def resolve_device(preferred=None):
    """
    自动探测本机可用的计算设备，返回 torch 设备字符串。

    探测优先级：显式配置 > cuda > mps(Apple) > xpu(Intel) > cpu

    设计意图：
        - 训练与推理都不该把设备写死。同一份代码在无 GPU 的机器上要能自动退到 CPU，
          在有 NVIDIA 显卡的机器上要自动吃到 GPU，无需人工改配置。
        - 显式配置（config.ini 的 [model] device / 环境变量 MODEL_DEVICE）一旦给出，
          即**强制生效**，用于「有卡但想强制走 CPU 做基准测试」这类场景。

    :param preferred: 显式指定的设备（如 'cuda' / 'cpu'）。为空则走自动探测。
    :return: 'cuda' | 'mps' | 'xpu' | 'cpu'
    """
    # 1. 显式指定优先
    preferred = (preferred or '').strip().lower()
    if preferred:
        return preferred

    # 2. 尝试探测：torch 未安装或导入失败时直接退 CPU，不让训练/推理因缺依赖而崩
    try:
        import torch
    except ImportError:
        return 'cpu'

    # CUDA（NVIDIA）：不仅要求编译时带 CUDA，还要求运行时有可用显卡
    # 注意：装成 +cpu 版 torch 时 torch.cuda.is_available() 恒为 False，
    #      这是常见踩坑——版本号无 +cpu 后缀不代表就能用 GPU。
    if getattr(torch.version, 'cuda', None) and torch.cuda.is_available():
        return 'cuda'

    # MPS（Apple Silicon）
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return 'mps'

    # XPU（Intel Arc 等）
    xpu = getattr(torch, 'xpu', None)
    if xpu is not None and xpu.is_available():
        return 'xpu'

    return 'cpu'


def describe_device(device=None):
    """
    返回设备的人类可读描述，用于启动日志/训练脚本打印。

    例：'cuda (NVIDIA GeForce RTX 4060 Laptop GPU, 显存 8.0GB)' / 'cpu (未检测到可用 GPU)'
    """
    device = device or config.MODEL_DEVICE or resolve_device()
    try:
        import torch
    except ImportError:
        return f'{device} (torch 未安装)'

    if device == 'cuda' and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f'cuda ({name}, 显存 {total:.1f}GB)'
        except Exception:
            return 'cuda (信息读取失败)'
    if device == 'mps':
        return 'mps (Apple Silicon)'
    if device == 'xpu':
        return 'xpu (Intel GPU)'

    # device == 'cpu'：要区分「机器没卡」和「有卡但显式指定了 cpu」——
    # 后者是最常见的情形（见 config.ini [model] device 的注释说明），
    # 若笼统写成「未检测到可用 GPU」会误导排查方向。
    cuda_build = getattr(torch.version, 'cuda', None)
    try:
        gpu_ok = bool(cuda_build and torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_ok = False

    if gpu_ok:
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return (f'cpu (显式指定；本机有可用的 {name} {total:.1f}GB，'
                    f'如需 GPU 请改 config.ini 的 [model] device)')
        except Exception:  # noqa: BLE001
            return 'cpu (显式指定；本机有可用 GPU)'
    return f'cpu (未检测到可用 GPU，torch 构建版本 {cuda_build or "cpu"})'


config = Config()

# 解析最终设备：显式配置优先，否则自动探测。全项目统一通过 config.MODEL_DEVICE 读取。
config.MODEL_DEVICE = resolve_device(config.MODEL_DEVICE)


if __name__ == '__main__':
    print('LLM_MODEL       =', config.LLM_MODEL)
    print('MILVUS_COLLECTION_NAME =', config.MILVUS_COLLECTION_NAME)
    print('EMBEDDING_MODEL_PATH   =', config.EMBEDDING_MODEL_PATH)
    print('INTENT_MODEL_PATH      =', config.INTENT_MODEL_PATH)
    print('INTENT_BASE_MODEL      =', config.INTENT_BASE_MODEL)
    print('DATA_DIR        =', config.DATA_DIR)
    print('DATASET_DIR     =', config.DATASET_DIR)
    print('RETRIEVAL_K     =', config.RETRIEVAL_K, type(config.RETRIEVAL_K))
    print('SCORE_THRESHOLD =', config.SCORE_THRESHOLD)
    print('NPROBE          =', config.NPROBE)
    print('MODEL_DEVICE    =', config.MODEL_DEVICE, '->', describe_device())
    print('KB_TYPES        =', config.KB_TYPES)
    print('ACCESS_ENABLED  =', config.ACCESS_ENABLED,
          'default_role =', config.ACCESS_DEFAULT_ROLE)
    print('ACCESS_ROLE_LEVELS      =', config.ACCESS_ROLE_LEVELS)
    print('ACCESS_ROLE_DEPARTMENTS =', config.ACCESS_ROLE_DEPARTMENTS)
    print('ACCESS_KB_DEFAULTS      =', config.ACCESS_KB_DEFAULTS)
    print('KB_MANIFEST_FILE        =', config.KB_MANIFEST_FILE)
