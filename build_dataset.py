"""
    数据集构建入口：build_dataset.py

    对应需求基线的「数据预处理 → 构建数据集」环节，把 data/ 下的知识库
    转化为三个下游任务可用的训练数据：

        1. 意图分类数据集  data/dataset/intent/     —— 训练意图识别模型
        2. 重排数据集      data/dataset/rerank/     —— 微调 BGE-Reranker
        3. 指令微调数据集  data/dataset/sft/        —— LoRA 微调对话模型

    用法：
        python build_dataset.py                 # 构建全部数据集
        python build_dataset.py --only intent   # 只构建意图分类数据集
        python build_dataset.py --eval-ratio 0.2  # 指定评估集占比

    说明：本脚本**不调用任何大模型**，全部数据来自知识库原文，保证可追溯、可复现。
"""
import argparse
import os
import sys
from datetime import datetime

from base.config import config
from base.logger import logger


def collect_knowledge_files():
    """
    遍历 data/ 下各知识库目录，返回 {kb_type: [(file_name, text), ...]}。
    kb_type 即目录名去掉 _data 后缀，与 Milvus 里的 kb_type 字段口径一致。
    """
    kb_files = {}
    for dir_name in sorted(os.listdir(config.DATA_DIR)):
        dir_path = os.path.join(config.DATA_DIR, dir_name)
        if not os.path.isdir(dir_path) or not dir_name.endswith('_data'):
            continue
        kb_type = dir_name[:-len('_data')]
        files = []
        for file_name in sorted(os.listdir(dir_path)):
            if not file_name.lower().endswith('.md'):
                continue
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                files.append((file_name, f.read()))
        if files:
            kb_files[kb_type] = files
    return kb_files


def build_intent_dataset(kb_files, out_dir, eval_ratio, augment=True):
    """构建意图分类数据集：从全部知识库中挖掘问句做弱标注 + 模板扩写"""
    from core.dataset_builder import IntentDatasetBuilder, write_jsonl, write_label_map

    logger.info("=" * 60)
    logger.info("【1/3】构建意图分类数据集")

    # 问句来源：FAQ 的 Q + 其余文档里的标题/正文中形似问句的句子
    from core.dataset_builder import parse_faq_pairs, parse_doc_sections, is_question

    questions = []
    for kb_type, files in kb_files.items():
        for _, text in files:
            if kb_type == 'faq':
                for question, _ in parse_faq_pairs(text):
                    questions.append(question)
            # 标题若本身就是问句（如「烹饪后放置超过 2 小时的食品还能吃吗？」），也可作为样本
            for title, _ in parse_doc_sections(text):
                if is_question(title):
                    questions.append(title)

    # 去重后交给构建器
    questions = list(dict.fromkeys(questions))
    logger.info(f"共挖掘出 {len(questions)} 个候选问句")

    builder = IntentDatasetBuilder(questions, eval_ratio=eval_ratio, augment=augment)
    train, eval_set = builder.build()

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, 'intent_train.jsonl'), train)
    write_jsonl(os.path.join(out_dir, 'intent_eval.jsonl'), eval_set)
    write_label_map(os.path.join(out_dir, 'label_map.json'))

    train_stats = IntentDatasetBuilder.stats(train)
    eval_stats = IntentDatasetBuilder.stats(eval_set)
    train_src = IntentDatasetBuilder.source_stats(train)
    eval_src = IntentDatasetBuilder.source_stats(eval_set)
    logger.info(f"训练集类别分布：{train_stats}")
    logger.info(f"评估集类别分布：{eval_stats}")
    logger.info(f"训练集来源分布：{train_src}")
    logger.info(f"评估集来源分布：{eval_src}")

    return {
        'train_size': len(train),
        'eval_size': len(eval_set),
        'train_stats': train_stats,
        'eval_stats': eval_stats,
        'train_src': train_src,
        'eval_src': eval_src,
    }


def build_rerank_dataset(kb_files, out_dir, eval_ratio):
    """构建重排数据集：FAQ 问答对作正样本，同库其他文档作 hard negative"""
    from core.dataset_builder import (
        RerankDatasetBuilder, parse_faq_pairs, parse_doc_sections, write_jsonl
    )

    logger.info("=" * 60)
    logger.info("【2/3】构建重排数据集")

    faq_pairs = []
    doc_sections = []
    for kb_type, files in kb_files.items():
        for file_name, text in files:
            if kb_type == 'faq':
                faq_pairs.extend(parse_faq_pairs(text))
            else:
                doc_sections.extend(parse_doc_sections(text))

    logger.info(f"FAQ 问答对（正样本）：{len(faq_pairs)} 条")
    logger.info(f"文档段落（补充正样本）：{len(doc_sections)} 条")

    builder = RerankDatasetBuilder(faq_pairs, doc_sections)
    triples = builder.build()

    # 按稳定哈希划分训练/评估，避免同一 query 跨集
    from core.dataset_builder import _stable_bucket
    bucket_limit = int(eval_ratio * 100)
    train = [t for t in triples if _stable_bucket(t['query']) >= bucket_limit]
    eval_set = [t for t in triples if _stable_bucket(t['query']) < bucket_limit]

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, 'rerank_train.jsonl'), train)
    write_jsonl(os.path.join(out_dir, 'rerank_eval.jsonl'), eval_set)

    logger.info(f"重排训练集 {len(train)} 条，评估集 {len(eval_set)} 条")
    return {'train_size': len(train), 'eval_size': len(eval_set)}


def build_sft_dataset(kb_files, out_dir, eval_ratio):
    """构建指令微调数据集：output 一律摘自知识库原文"""
    from core.dataset_builder import SFTDatasetBuilder, parse_faq_pairs, write_jsonl

    logger.info("=" * 60)
    logger.info("【3/3】构建指令微调数据集")

    faq_pairs = []
    for kb_type, files in kb_files.items():
        if kb_type != 'faq':
            continue
        for _, text in files:
            faq_pairs.extend(parse_faq_pairs(text))

    logger.info(f"FAQ 问答对：{len(faq_pairs)} 条")

    # case_docs 留空：本项目 data/ 中暂无带「场景限定」标注的问答，
    # 保留该入参是为了后续扩充（如「学校食堂」场景），不影响当前构建。
    builder = SFTDatasetBuilder(faq_pairs, case_docs=[])
    samples = builder.build()

    from core.dataset_builder import _stable_bucket
    bucket_limit = int(eval_ratio * 100)
    train = [s for s in samples if _stable_bucket(s['instruction']) >= bucket_limit]
    eval_set = [s for s in samples if _stable_bucket(s['instruction']) < bucket_limit]

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, 'sft_train.jsonl'), train)
    write_jsonl(os.path.join(out_dir, 'sft_eval.jsonl'), eval_set)

    logger.info(f"指令微调训练集 {len(train)} 条，评估集 {len(eval_set)} 条")
    return {'train_size': len(train), 'eval_size': len(eval_set)}


def write_dataset_readme(out_dir, summary):
    """写出数据集说明文件，交代来源、格式、用途与统计"""
    from core.dataset_builder import INTENT_LABELS, INTENT_LABEL_DOC

    lines = []
    lines.append("# 训练数据集说明")
    lines.append("")
    lines.append(f"> 由 `build_dataset.py` 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("> 所有数据均从 `data/` 知识库原文挖掘，**未经大模型生成**，可追溯、可复现。")
    lines.append("")
    lines.append("## 一、目录结构")
    lines.append("")
    lines.append("```")
    lines.append("data/dataset/")
    lines.append("├── intent/                    # 意图分类数据集（训练意图识别模型）")
    lines.append("│   ├── intent_train.jsonl")
    lines.append("│   ├── intent_eval.jsonl")
    lines.append("│   └── label_map.json         # 意图标签映射，训练端与推理端共用")
    lines.append("├── rerank/                    # 重排数据集（微调 BGE-Reranker）")
    lines.append("│   ├── rerank_train.jsonl")
    lines.append("│   └── rerank_eval.jsonl")
    lines.append("├── sft/                       # 指令微调数据集（LoRA 微调对话模型）")
    lines.append("│   ├── sft_train.jsonl")
    lines.append("│   └── sft_eval.jsonl")
    lines.append("└── README.md")
    lines.append("```")
    lines.append("")
    lines.append("## 二、各数据集格式")
    lines.append("")
    lines.append("### 1. 意图分类数据集")
    lines.append("")
    lines.append("```json")
    lines.append('{"text": "食品留样应该留多少克？", "label": 1, "label_name": "SOP 指导", "source": "auto_keyword"}')
    lines.append("```")
    lines.append("")
    lines.append("| 意图 ID | 键名 | 中文名 | 说明 |")
    lines.append("|---------|------|--------|------|")
    for idx, key, name in INTENT_LABELS:
        lines.append(f"| {idx} | `{key}` | {name} | {INTENT_LABEL_DOC.get(name, '')} |")
    lines.append("")
    lines.append("**重要——数据来源的三种性质：**")
    lines.append("")
    lines.append("- `source=auto_keyword`：由特征词规则自动标注的**弱标注**数据，来自知识库真实问句，质量受规则限制。")
    lines.append("- `source=seed`：人工编写的种子样本，用于覆盖规则命中不到的表达方式（尤其是「闲聊」类）。")
    lines.append("- `source=template`：由「主体词 × 疑问句式」机械组合扩写而成，规模大、句式分布比真实提问窄。")
    lines.append("")
    lines.append("**为什么需要模板扩写**：知识库中真实问句只有几十条，6 分类下每类不足 10 条，")
    lines.append("直接训练等于让模型背答案。模板扩写把样本量提升一个数量级，让训练具备可行性。")
    lines.append("")
    lines.append("**评估集净化**：模板样本句式分布比真实提问窄，若占满评估集会让分数虚高。")
    lines.append("因此切分时会控制模板样本在评估集中的占比，优先用真实样本填充评估集，")
    lines.append("同时保证训练集至少保留一半真实样本（避免「掏空训练集」只剩模板）。")
    lines.append("")
    lines.append("评估集（`intent_eval.jsonl`）按文本哈希独立切出，**绝不出现在训练集中**，")
    lines.append("且只应在**最终评估**时使用一次，不要拿它反复调超参。")
    lines.append("")
    lines.append("### 2. 重排数据集")
    lines.append("")
    lines.append("```json")
    lines.append('{"query": "冰箱冷藏温度标准是多少？", "positive": "根据 GB 31654-2021……", "negatives": ["……"]}')
    lines.append("```")
    lines.append("")
    lines.append("负样本从**同一知识库的其他文档**中采集（hard negative），主题相近但非答案，")
    lines.append("比随机负例更难，更能逼出模型的细粒度区分能力。")
    lines.append("")
    lines.append("### 3. 指令微调数据集")
    lines.append("")
    lines.append("```json")
    lines.append('{"instruction": "冰箱冷藏温度标准是多少？冷冻呢？", "input": "", "output": "根据 GB 31654-2021……", "system": "你是食品安全领域的专业助手……", "source": "faq"}')
    lines.append("```")
    lines.append("")
    lines.append("`output` 全部摘自知识库原文。**不要用大模型改写 output**——")
    lines.append("拿模型生成的答案去训练模型，会把上一代的幻觉固化成「标准答案」。")
    lines.append("")
    lines.append("## 三、本次构建统计")
    lines.append("")
    lines.append("| 数据集 | 训练集 | 评估集 |")
    lines.append("|--------|--------|--------|")
    for key, title in (('intent', '意图分类'), ('rerank', '重排'), ('sft', '指令微调')):
        item = summary.get(key, {})
        lines.append(f"| {title} | {item.get('train_size', 0)} | {item.get('eval_size', 0)} |")
    lines.append("")
    if 'intent' in summary:
        lines.append("### 意图分类类别分布")
        lines.append("")
        lines.append("| 意图 | 训练集 | 评估集 |")
        lines.append("|------|--------|--------|")
        train_stats = summary['intent'].get('train_stats', {})
        eval_stats = summary['intent'].get('eval_stats', {})
        for _, _, name in INTENT_LABELS:
            lines.append(f"| {name} | {train_stats.get(name, 0)} | {eval_stats.get(name, 0)} |")
        lines.append("")
        lines.append("> 若某个类别为 0，说明知识库中缺少该类问句。")
        lines.append("> 补齐方式：往 `data/faq_data/` 增补对应主题的问答对，再重新运行 `python build_dataset.py`。")
        lines.append("")
    lines.append("## 四、后续如何使用")
    lines.append("")
    lines.append("### 1. 意图分类（本项目已实现完整训练链路）")
    lines.append("")
    lines.append("```bash")
    lines.append("python train_intent.py            # 训练（默认 10 轮，基座用本地 models/bge-m3）")
    lines.append("python train_intent.py --dry-run  # 只体检数据，不训练")
    lines.append("```")
    lines.append("")
    lines.append("训练产物落到 `models/intent-classifier/`，`core/intent_classifier.py` 会自动加载并接管策略路由；")
    lines.append("模型不存在时系统回退到规则路由，功能不受影响。")
    lines.append("")
    lines.append("训练集 / 验证集 / 测试集三者职责不同，**不要混用**：")
    lines.append("")
    lines.append("| 集合 | 来源 | 用途 | 使用次数 |")
    lines.append("|------|------|------|----------|")
    lines.append("| 训练集 | `intent_train.jsonl` 内切 80% | 更新参数 | 每轮 |")
    lines.append("| 验证集 | 从**训练集内部**再切 20% | 早停、选最优 checkpoint | 每轮 |")
    lines.append("| 固定测试集 | `intent_eval.jsonl` | 最终评估真实泛化能力 | **全程 1 次** |")
    lines.append("")
    lines.append("### 2. 重排微调（仅提供数据，未提供训练脚本）")
    lines.append("")
    lines.append("用 `rerank/*.jsonl` 以 `(query, positive)=1 / (query, negative)=0` 为标签微调 CrossEncoder。")
    lines.append("**注意**：当前正样本数量偏少，微调 bge-reranker-large 会过拟合，")
    lines.append("建议先把数据扩到 500+ 组再考虑训练。")
    lines.append("")
    lines.append("### 3. 指令微调（仅提供数据，未提供训练脚本）")
    lines.append("")
    lines.append("用 `sft/*.jsonl` 做 LoRA 微调，参数可参考场景文档（rank=64, alpha=128, 3 epoch）。")
    lines.append("**注意**：SFT 需要 1k+ 条高质量人工/强模型标注数据，且本机显存不足以微调 7B 模型；")
    lines.append("本系统走 RAG 路线（检索 → 拼上下文 → DeepSeek 生成），无需微调本地对话模型。")
    lines.append("")

    path = os.path.join(out_dir, 'README.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logger.info(f"已写出数据集说明 -> {path}")


def main():
    parser = argparse.ArgumentParser(description='食品安全知识库训练数据集构建工具')
    parser.add_argument('--only', type=str, default='all',
                        choices=['all', 'intent', 'rerank', 'sft'],
                        help='只构建指定数据集（默认全部）')
    parser.add_argument('--eval-ratio', type=float, default=0.2,
                        help='评估集占比（默认 0.2）')
    parser.add_argument('--no-augment', action='store_true',
                        help='关闭意图数据的模板扩写（仅用知识库真实问句 + 种子样本）')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='输出目录（默认 data/dataset）')
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(config.DATA_DIR, 'dataset')

    print("=" * 60)
    print("食安通 · 训练数据集构建")
    print(f"数据来源：{config.DATA_DIR}")
    print(f"输出目录：{out_root}")
    print(f"评估集占比：{args.eval_ratio}")
    print("=" * 60)

    kb_files = collect_knowledge_files()
    if not kb_files:
        logger.error(f"未在 {config.DATA_DIR} 下找到任何 .md 知识文件")
        sys.exit(1)

    logger.info(f"扫描到知识库：{ {k: len(v) for k, v in kb_files.items()} }")

    summary = {}
    if args.only in ('all', 'intent'):
        summary['intent'] = build_intent_dataset(
            kb_files, os.path.join(out_root, 'intent'), args.eval_ratio,
            augment=not args.no_augment)
    if args.only in ('all', 'rerank'):
        summary['rerank'] = build_rerank_dataset(
            kb_files, os.path.join(out_root, 'rerank'), args.eval_ratio)
    if args.only in ('all', 'sft'):
        summary['sft'] = build_sft_dataset(
            kb_files, os.path.join(out_root, 'sft'), args.eval_ratio)

    write_dataset_readme(out_root, summary)

    print("\n" + "=" * 60)
    print("数据集构建完成")
    for key, title in (('intent', '意图分类'), ('rerank', '重排'), ('sft', '指令微调')):
        item = summary.get(key)
        if item:
            print(f"  {title}：训练集 {item['train_size']} 条 / 评估集 {item['eval_size']} 条")
    print(f"  输出目录：{out_root}")
    print("=" * 60)


if __name__ == '__main__':
    main()
