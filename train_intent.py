"""
    意图分类模型训练入口：train_intent.py

    对应需求基线的「模型训练」环节——用 data/dataset/intent/ 的数据微调一个
    6 分类模型，产出模型放到 models/intent-classifier/，core/intent_classifier.py
    会自动加载并接管策略路由（模型缺失时仍回退规则，不影响问答功能）。

    用法：
        python train_intent.py                    # 默认配置训练
        python train_intent.py --epochs 15        # 指定轮数
        python train_intent.py --device cpu       # 强制走 CPU
        python train_intent.py --base models/bge-m3  # 指定基座模型
        python train_intent.py --dry-run          # 只加载数据、检查标签分布，不训练

    数据集职责划分（重要，勿混用）：
        intent_eval.jsonl  —— **固定测试集**，只在最终评估时用一次，绝不参与调参。
                              反复拿它调超参 = 间接过拟合，测试分数会失真。
        训练集内部再切 val —— **验证集**，用于每轮评估、选最优 checkpoint、早停。

    设备选择：
        调用 base.config.resolve_device() 自动探测（cuda > mps > xpu > cpu），
        也可用 --device 或 config.ini 的 [model] device 强制指定。
"""
import argparse
import json
import os
import sys
from collections import Counter

from base.config import config, resolve_device, describe_device
from base.logger import logger

# 标签数量，与 core/dataset_builder.py 的 INTENT_LABELS 保持一致
NUM_LABELS = 6


def load_jsonl(path):
    """读取 JSONL 文件，返回记录列表"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_label_names():
    """
    从 label_map.json 读取标签中文名（按 id 排序），保证与数据集口径一致。
    找不到时回退到 dataset_builder 里的定义。
    """
    label_map_path = os.path.join(config.DATASET_DIR, 'intent', 'label_map.json')
    if os.path.isfile(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = sorted(data.get('labels', []), key=lambda x: x['id'])
        return [item['name'] for item in labels]

    from core.dataset_builder import INTENT_LABELS
    return [name for _, _, name in INTENT_LABELS]


def split_train_val(train_records, val_ratio, seed):
    """
    从训练集内部切出验证集（**不动测试集**）。

    用随机打乱后切片而非按哈希：因为这里已经是在「训练集」内部划分，
    不需要跨进程稳定性，随机切分更自然。固定 seed 保证可复现。
    """
    import random
    records = train_records[:]
    random.Random(seed).shuffle(records)
    val_size = max(1, int(len(records) * val_ratio))
    return records[val_size:], records[:val_size]


def check_dataset(train_path, test_path):
    """加载并体检数据集：检查文件存在、标签范围、类别分布、来源分布"""
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        logger.error(f"数据集不存在，请先运行：python build_dataset.py --only intent")
        logger.error(f"  期望路径：{train_path}")
        logger.error(f"            {test_path}")
        sys.exit(1)

    train = load_jsonl(train_path)
    test = load_jsonl(test_path)

    for name, records in (('训练集', train), ('测试集', test)):
        invalid = [r for r in records if not (0 <= int(r.get('label', -1)) < NUM_LABELS)]
        if invalid:
            logger.error(f"{name}存在 {len(invalid)} 条非法标签（越界或缺失），请重建数据集")
            sys.exit(1)

    return train, test


def report_distribution(records, label_names, title):
    """打印类别分布与来源分布，用于发现类别不均衡"""
    label_counter = Counter(int(r['label']) for r in records)
    source_counter = Counter(r.get('source', 'unknown') for r in records)
    logger.info(f"{title}：共 {len(records)} 条")
    for label_id, name in enumerate(label_names):
        cnt = label_counter.get(label_id, 0)
        ratio = cnt / len(records) * 100 if records else 0
        logger.info(f"    [{label_id}] {name:<8} {cnt:>4} 条  ({ratio:5.1f}%)")
    logger.info(f"    来源分布：{dict(source_counter)}")

    # 类别不均衡提醒：最大类与最小类差距超过 3 倍时提示
    counts = [label_counter.get(i, 0) for i in range(len(label_names))]
    nonzero = [c for c in counts if c > 0]
    if nonzero and max(nonzero) > min(nonzero) * 3:
        logger.warning(f"检测到明显类别不均衡（最多 {max(nonzero)} 条 vs 最少 {min(nonzero)} 条），"
                       f"可考虑补充少数类样本，或训练时使用类别权重。")


def compute_metrics(predictions, labels, num_labels):
    """
    计算分类指标：准确率、macro-F1、每类 F1、混淆矩阵。

    手写实现而不依赖 sklearn 的 classification_report，是为了顺带产出
    6×6 混淆矩阵——定位「哪个意图最容易被误判成哪个」比一个总分有用得多。
    """
    total = len(labels)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / total if total else 0.0

    # 混淆矩阵 matrix[真实][预测]
    matrix = [[0] * num_labels for _ in range(num_labels)]
    for pred, label in zip(predictions, labels):
        matrix[label][pred] += 1

    # 每类 P / R / F1
    per_class = []
    for i in range(num_labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(num_labels)) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(matrix[i]),
        })

    # macro-F1：各类 F1 的算术平均（不按样本数加权，少数类同样重要）
    macro_f1 = sum(item['f1'] for item in per_class) / num_labels if num_labels else 0.0

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'matrix': matrix,
    }


def print_confusion_matrix(matrix, label_names):
    """打印混淆矩阵，行=真实标签，列=预测标签"""
    logger.info("混淆矩阵（行=真实，列=预测）：")
    header = "        " + "".join(f"{name[:4]:>7}" for name in label_names)
    logger.info(header)
    for i, name in enumerate(label_names):
        row = "".join(f"{matrix[i][j]:>7}" for j in range(len(label_names)))
        logger.info(f"{name[:6]:<8}{row}")


def available_ram_gb():
    """
    返回 (可用物理内存GB, 总物理内存GB)；取不到时返回 (None, None)。

    用途是给 CPU 训练做内存预估。psutil 缺失时不阻断训练，只是跳过守护。
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / 1024 ** 3, vm.total / 1024 ** 3
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(description='食品安全知识库 · 意图分类模型训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--val-ratio', type=float, default=None,
                        help='从训练集内部切出的验证集占比')
    parser.add_argument('--device', type=str, default=None,
                        help='强制指定设备：cpu / cuda（默认自动探测）')
    parser.add_argument('--base', type=str, default=None,
                        help='基座模型路径（默认取 config.INTENT_BASE_MODEL，即本地 bge-m3）')
    parser.add_argument('--output', type=str, default=None,
                        help='模型输出目录（默认 models/intent-classifier）')
    parser.add_argument('--max-len', type=int, default=48,
                        help='最大序列长度。意图问句都很短，48 足够；调大更慢（padding 到定长，'
                             'attention 开销按长度平方涨）。实测语料 643 条最长仅 20 token，'
                             '与推理端 core/intent_classifier.py 的 max_length=64 完全等价')
    parser.add_argument('--optim', type=str, default=None,
                        choices=['adamw_torch', 'sgd', 'adafactor'],
                        help='优化器。adamw_torch 默认最稳；机器内存紧张（如 ≤8G 可用的 CPU）'
                             '时用 sgd——AdamW 要为全部参数额外保存两份状态，约等于再吃一份模型内存')
    # default=None 而不是 0：需要区分「用户显式要求全量微调」与「没传参」，
    # 后者才允许内存守护逻辑自动降级。
    parser.add_argument('--freeze-layers', type=int, default=None,
                        help='冻结 embeddings 与前 N 层 transformer，0 = 全量微调。'
                             '内存极紧时用（例如 20 表示只训最后 4 层 + 分类头）。'
                             '不传时若可用内存不足会自动降级')
    parser.add_argument('--dry-run', action='store_true',
                        help='只体检数据集、打印分布，不真正训练')
    args = parser.parse_args()

    epochs = args.epochs or config.TRAIN_EPOCHS
    batch_size = args.batch_size or config.TRAIN_BATCH_SIZE
    lr = args.lr or config.TRAIN_LR
    val_ratio = args.val_ratio if args.val_ratio is not None else config.TRAIN_VAL_RATIO
    base_model = args.base or config.INTENT_BASE_MODEL
    output_dir = args.output or config.INTENT_MODEL_PATH

    device = resolve_device(args.device or config.MODEL_DEVICE)

    # flush=True：输出被重定向到文件时 print 默认全缓冲，banner 会跑到训练日志后面，误导排查
    print("=" * 60, flush=True)
    print("食安通 · 意图分类模型训练", flush=True)
    print(f"基座模型：{base_model}", flush=True)
    print(f"训练设备：{device} -> {describe_device(device)}", flush=True)
    print(f"输出目录：{output_dir}", flush=True)
    print("=" * 60, flush=True)

    label_names = load_label_names()
    logger.info(f"标签空间（{len(label_names)} 类）：{label_names}")

    train_path = os.path.join(config.DATASET_DIR, 'intent', 'intent_train.jsonl')
    test_path = os.path.join(config.DATASET_DIR, 'intent', 'intent_eval.jsonl')
    train_all, test_records = check_dataset(train_path, test_path)

    report_distribution(train_all, label_names, "训练集（原始）")
    report_distribution(test_records, label_names, "固定测试集")

    if len(train_all) < 60:
        logger.warning(f"训练集仅 {len(train_all)} 条，样本偏少，模型容易过拟合。"
                       f"建议运行 python build_dataset.py --only intent 开启模板扩写。")

    if args.dry_run:
        logger.info("--dry-run 模式，仅完成数据体检，未进行训练。")
        return

    # ---------- 依赖导入（放在体检之后，避免缺依赖时连数据检查都做不了）----------
    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error(f"训练依赖缺失：{e}")
        logger.error("请安装：pip install torch transformers datasets scikit-learn accelerate")
        sys.exit(1)

    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("请求了 cuda 但当前 torch 不可用（可能是 +cpu 构建），自动回退 cpu")
        device = 'cpu'

    # 线程数与 CPU 训练稳定性：
    # 在 Windows/WSL 上，torch 默认会开满所有核心，实测偶发 Segmentation fault。
    # 这里主动收敛到 4 线程（可由 TRAIN_THREADS 覆盖），稳定性优先。
    if device == 'cpu':
        import os as _os
        threads = int(_os.getenv('TRAIN_THREADS', '4'))
        torch.set_num_threads(threads)
        logger.info(f"CPU 训练：已把 torch 线程数设为 {threads}（避免多线程偶发段错误）")

    # ---------- 训练配置的内存自适应（CPU 场景）----------
    # bge-m3 是 568M 参数的 XLM-R large。fp32 全量微调的内存账：
    #   权重 2.3G（必须常驻）+ 梯度 2.3G + AdamW 两份状态 4.6G ≈ 9G 峰值，再叠加 activation。
    # 内存不足时 PyTorch 申请失败往往不给 Python 异常，而是让进程以
    # 0xC0000005(access violation) / 139(segfault) 退出——表现就是「进度条刚出 0/xxx 就闪退，无 traceback」。
    # 所以这里先按可用内存预估，自动收敛到「SGD + 冻结底层」的省内存组合，
    # 避免每次都要人工记得加参数。显式传参时一律尊重用户的选择。
    FULL_FT_MIN_RAM_GB = 12.0

    optim_name = args.optim
    freeze_layers = args.freeze_layers if args.freeze_layers is not None else 0

    if device == 'cpu':
        avail, total = available_ram_gb()
        if avail is not None:
            logger.info(f"物理内存：可用 {avail:.1f}G / 共 {total:.1f}G")
            if avail < FULL_FT_MIN_RAM_GB:
                auto = False
                if args.optim is None:
                    optim_name = 'sgd'
                    auto = True
                    logger.warning(
                        f"可用 {avail:.1f}G < 全量微调经验下限 {FULL_FT_MIN_RAM_GB:.0f}G，"
                        f"自动改用 SGD（AdamW 要为每个可训参数额外存两份状态，约再吃一份模型内存）")
                if args.freeze_layers is None:
                    freeze_layers = 20
                    auto = True
                    logger.warning(
                        f"自动冻结 embeddings 与前 {freeze_layers} 层，只训最后几层 + 分类头以压缩梯度与优化器状态")
                if auto and args.lr is None:
                    lr = 0.01
                    logger.warning(f"SGD 在此规模上需要更大学习率，自动把 lr 提到 {lr}（原 {config.TRAIN_LR}）")
                elif args.optim == 'adamw_torch' and args.freeze_layers is None:
                    logger.warning("你显式指定了 adamw_torch 但内存偏紧；若仍以 0xC0000005 闪退，"
                                   "请去掉该参数让脚本自动降级，或手动加 --optim sgd --freeze-layers 20")

    optim_name = optim_name or 'adamw_torch'
    logger.info(f"训练超参：epochs={epochs} batch={batch_size} lr={lr} "
                f"max_len={args.max_len} optim={optim_name} freeze_layers={freeze_layers}")

    # 切分验证集（从训练集内部，不动测试集）
    train_records, val_records = split_train_val(train_all, val_ratio, config.TRAIN_SEED)
    logger.info(f"数据划分：训练 {len(train_records)} / 验证 {len(val_records)} / 测试 {len(test_records)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class IntentDataset(Dataset):
        """把 jsonl 记录转成模型输入。__getitem__ 返回的是张量，不是 list。"""

        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            record = self.records[idx]
            encoded = tokenizer(
                record['text'],
                truncation=True,
                max_length=args.max_len,
                padding='max_length',
            )
            # 关键：values() 拿到的是 python list，batch 拼接才不会被当索引；
            #       若直接传 tokenizer 原生返回的 BatchEncoding 会在 collate 时出错。
            item = {key: torch.tensor(value) for key, value in encoded.items()}
            item['labels'] = torch.tensor(int(record['label']))
            return item

    train_dataset = IntentDataset(train_records)
    val_dataset = IntentDataset(val_records)
    test_dataset = IntentDataset(test_records)

    logger.info(f"加载基座模型：{base_model}（num_labels={NUM_LABELS}）")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=NUM_LABELS)

    # ---------- 可选：冻结底层，降低训练内存 ----------
    # bge-m3 是 568M 参数的 XLM-R large 结构，fp32 全量微调时「模型 + 梯度 + AdamW 两份状态」
    # 三项相加约 9G；CPU 机器可用内存不足时会在 backward 阶段直接 segfault（无报错、无栈）。
    # 冻结底层后梯度只覆盖可训参数，内存占用随之下降。
    if freeze_layers > 0:
        encoder = getattr(model, 'roberta', None) or getattr(model, 'bert', None)
        if encoder is None:
            logger.warning("当前模型结构无法识别 encoder，忽略 --freeze-layers")
        else:
            frozen = 0
            for p in encoder.embeddings.parameters():
                p.requires_grad = False
            frozen += 1
            for i, layer in enumerate(encoder.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
                    frozen += 1
            total_layers = len(encoder.encoder.layer)
            logger.info(f"冻结 {frozen} 个模块（embeddings + 前 {freeze_layers}/{total_layers} 层）")

    # 训练前把内存账算清楚：出问题时能直接看出是超参问题还是机器扛不住
    all_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_gb = all_params * 4 / 1024 ** 3
    grad_gb = trainable * 4 / 1024 ** 3
    if optim_name == 'adamw_torch':
        state_gb = trainable * 8 / 1024 ** 3
    elif optim_name == 'sgd':
        state_gb = 0.0
    else:  # adafactor 只保留行/列二阶统计
        state_gb = trainable * 0.5 / 1024 ** 3
    logger.info(f"可训参数 {trainable/1e6:.1f}M / 总 {all_params/1e6:.1f}M")
    logger.info(f"预估训练态内存：权重 {weight_gb:.1f}G + 梯度 {grad_gb:.1f}G "
                f"+ 优化器状态 {state_gb:.1f}G ≈ {weight_gb + grad_gb + state_gb:.1f}G（不含 activation）")

    def compute_metrics_fn(eval_pred):
        """传给 Trainer 的回调：每轮拿 logits 与真实标签算指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = compute_metrics(predictions.tolist(), labels.tolist(), NUM_LABELS)
        # Trainer 只接受扁平标量，这里只回传总分；详细指标训练后单独算
        return {'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']}

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, '_checkpoints')
    best_dir = os.path.join(ckpt_dir, 'best')

    # 上一轮训练可能留下同名目录，先清掉，避免读到陈旧权重
    if os.path.isdir(best_dir):
        import shutil as _shutil
        _shutil.rmtree(best_dir, ignore_errors=True)

    class KeepBestCallback(TrainerCallback):
        """
        每次验证后若 macro-F1 创新高，就把权重覆盖保存到 _checkpoints/best。

        为什么不用 load_best_model_at_end 自带的机制：
          1) 它在 train() 内部把 best 权重整份读进内存，和训练后权重叠加，低内存机器会被顶死；
          2) 它配合 save_total_limit 时的「不删 best」只对 epoch 级 checkpoint 生效，
             而我们关掉它之后，save_total_limit 会按数量直接删旧 checkpoint，最优轮次可能被删掉。
        单独存一份 best 目录，既不占额外内存峰值，也不受 checkpoint 轮转影响。
        """

        def __init__(self, save_dir):
            self.save_dir = save_dir
            self.best_score = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            score = (metrics or {}).get('eval_macro_f1')
            if score is None:
                return control
            if self.best_score is None or score > self.best_score:
                model = kwargs.get('model')
                if model is not None:
                    self.best_score = score
                    model.save_pretrained(self.save_dir)
                    logger.info(f"epoch {state.epoch:.0f}：验证 macro-F1 {score:.4f} 创新高，"
                                f"权重已保存至 {self.save_dir}")
            return control

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        # 关掉 load_best_model_at_end：它会在 train() 内部再加载一份完整的 568M 参数
        # 权重（约 2.3G）进内存，和训练结束时仍驻留的那份叠加。可用内存不足的 CPU
        # 机器会被直接顶死——表现为训练进度条走完之后进程静默退出、没有 traceback。
        # 最优权重的挑选取到 train() 之后手动做（先释放优化器再加载），峰值只多一份。
        load_best_model_at_end=False,
        save_only_model=True,
        seed=config.TRAIN_SEED,
        report_to=[],
        # 优化器：默认 adamw_torch（自适应学习率，小数据上收敛更快）；
        # 内存受限时用 sgd 可省下两份参数状态的内存，代价是学习率需调大才收敛。
        optim=optim_name,
        # 数据加载用主进程（不开子进程）：
        # 本机是小数据集（几百条），多进程 DataLoader 收益为零，
        # 反而在 Windows / 沙箱环境下容易触发段错误。设为 0 最稳。
        dataloader_num_workers=0,
        # 显式指定设备：TrainingArguments 没有 device 参数，走 use_cpu / no_cuda
        use_cpu=(device == 'cpu'),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[KeepBestCallback(best_dir)],
    )

    logger.info("开始训练……")
    train_result = trainer.train()
    train_loss = train_result.metrics.get('train_loss')
    logger.info(f"训练完成：train_loss={train_loss if train_loss is not None else float('nan'):.4f}，"
                f"global_step={trainer.state.global_step}")

    # ---------- 训练后：挑最优 checkpoint 并加载 ----------
    # 对应 TrainingArguments 里 load_best_model_at_end=False 的说明：
    # 不在 train() 内部加载 best，避免「训练后权重」和「best 权重」两份同时驻留内存。
    # 顺序：先释放优化器/调度器 → 从验证历史挑最优轮次 → 丢掉当前权重 → 加载 best。
    import gc as _gc
    trainer.optimizer = None
    trainer.lr_scheduler = None
    _gc.collect()

    steps_per_epoch = len(trainer.get_train_dataloader())
    best_metrics = None
    for item in trainer.state.log_history:
        if 'eval_macro_f1' not in item:
            continue
        # 严格大于：并列时保留更早那一轮（更简单的模型），与「偏好早停」一致
        if best_metrics is None or item['eval_macro_f1'] > best_metrics['eval_macro_f1']:
            best_metrics = item

    best_path = None
    # 优先用回调单独保存的 best（不受 save_total_limit 轮转影响）；
    # 没有时再按验证历史推算 epoch 级 checkpoint 序号兜底。
    if os.path.isfile(os.path.join(best_dir, 'config.json')):
        best_path = best_dir
    elif best_metrics is not None:
        step = int(round(best_metrics['epoch'] * steps_per_epoch))
        candidate = os.path.join(ckpt_dir, f'checkpoint-{step}')
        if os.path.isdir(candidate):
            best_path = candidate

    if best_path and best_metrics is not None:
        logger.info(f"最优验证轮次：epoch {best_metrics['epoch']:.0f}（验证 macro-F1 "
                    f"{best_metrics['eval_macro_f1']:.4f}，eval_loss {best_metrics['eval_loss']:.4f}）-> {best_path}")

    if best_path:
        trainer.model = None
        trainer.model_wrapped = None
        del model
        _gc.collect()
        model = AutoModelForSequenceClassification.from_pretrained(best_path)
        model.to(device)
        trainer.model = model
        trainer.model_wrapped = model
        weights_epoch = int(round(best_metrics['epoch'])) if best_metrics else int(round(epochs))
        logger.info("已加载最优 checkpoint，后续评估与保存均基于该权重")
    else:
        weights_epoch = int(round(epochs))
        logger.warning("未找到可用的最优 checkpoint（可能被 save_total_limit 清理），"
                       "沿用最后一轮的权重继续评估；报告中的 weights_epoch 会如实标注")

    # ---------- 最终评估：在「固定测试集」上评一次 ----------
    logger.info("=" * 60)
    logger.info("在固定测试集上做最终评估（该测试集训练全程未参与调参）")
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1).tolist()
    labels = predictions_output.label_ids.tolist()

    final = compute_metrics(preds, labels, NUM_LABELS)
    logger.info(f"测试集准确率  : {final['accuracy']:.4f}")
    logger.info(f"测试集 macro-F1: {final['macro_f1']:.4f}")
    for i, name in enumerate(label_names):
        item = final['per_class'][i]
        logger.info(f"    [{i}] {name:<8} P={item['precision']:.3f} R={item['recall']:.3f} "
                    f"F1={item['f1']:.3f} 支持={item['support']}")
    print_confusion_matrix(final['matrix'], label_names)

    # ---------- 保存模型与训练报告 ----------
    # 只保存模型本体/tokenizer 到 output_dir，checkpoint 中间产物单独放 _checkpoints，
    # 避免 core/intent_classifier.py 加载时读到多余目录。
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    report = {
        'base_model': base_model,
        'device': device,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'max_len': args.max_len,
        'optim': optim_name,
        'freeze_layers': freeze_layers,
        # 最优验证轮次：说明「跑满 N 轮」与「实际最优」是否一致，便于判断过拟合程度
        'best_epoch': int(round(best_metrics['epoch'])) if best_metrics else None,
        # 实际保存到 output_dir 的权重来自哪一轮（best 被清理时会退化为最后一轮，如实标注）
        'weights_epoch': weights_epoch,
        'val_history': [
            {'epoch': round(item['epoch'], 2),
             'eval_loss': round(item['eval_loss'], 4),
             'accuracy': round(item['eval_accuracy'], 4),
             'macro_f1': round(item['eval_macro_f1'], 4)}
            for item in trainer.state.log_history if 'eval_macro_f1' in item
        ],
        'data': {
            'train': len(train_records),
            'val': len(val_records),
            'test': len(test_records),
            'train_source': dict(Counter(r.get('source', 'unknown') for r in train_records)),
            'test_source': dict(Counter(r.get('source', 'unknown') for r in test_records)),
        },
        'metrics': {
            'accuracy': final['accuracy'],
            'macro_f1': final['macro_f1'],
            'per_class': [
                {'label': label_names[i], **final['per_class'][i]}
                for i in range(NUM_LABELS)
            ],
        },
        'confusion_matrix': final['matrix'],
        'label_names': label_names,
    }
    report_path = os.path.join(output_dir, 'train_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"训练报告已保存 -> {report_path}")

    print("\n" + "=" * 60)
    print("训练完成")
    print(f"  模型目录：{output_dir}")
    print(f"  测试集准确率：{final['accuracy']:.4f}　macro-F1：{final['macro_f1']:.4f}")
    print(f"  重启 app.py 后，意图识别将自动切换到模型（日志会打印「意图分类模型加载成功」）")
    print("=" * 60)


if __name__ == '__main__':
    main()
