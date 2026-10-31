"""
    意图识别推理：core/intent_classifier.py

    对应需求基线的「模型预测」环节——加载训练好的意图分类模型，完成意图识别任务。

    两类预测器，对外暴露同一个 predict(query) -> 意图中文名 接口：
        1. ModelIntentClassifier —— 加载 models/intent-classifier/ 下的微调模型做真实预测。
        2. RuleIntentClassifier  —— 特征词规则兜底（即项目原有的策略路由逻辑）。

    加载策略（关键设计）：
        优先尝试加载模型；模型目录不存在、依赖缺失、加载失败时**自动降级为规则**，
        并在日志里写明原因。这样：
          - 数据集只是「准备好数据」时，系统照常运行，不会因为缺模型就跑不起来；
          - 训练完把模型丢进 models/intent-classifier/，重启即接管，无需改任何业务代码。
"""
import os

from base.config import config
from base.logger import logger
from core.dataset_builder import INTENT_LABELS

# 意图中文名集合，用于校验模型输出是否落在合法标签空间内
INTENT_NAMES = [name for _, _, name in INTENT_LABELS]

# 意图 -> 检索策略 的映射（对应需求基线「6 类意图驱动 4 种检索策略动态路由」）
#
# 说明每一处的取舍：
#   - 法规查询 / SOP 指导 / 营养咨询 / 供应商合规：问题通常单一明确，直接检索即可；
#   - 事故处置：问句往往包含「怎么处理 + 上报给谁 + 要保留什么证据」等多个子诉求，
#     拆成子问题分别检索能显著提高召回完整度，因此走 SubQuery；
#   - 闲聊：在 qa_system 层就被问候语短路拦掉了，这里映射为 direct 只是兜底。
#
# ⚠️ 重要工程判断：法规查询 / SOP 指导 / 营养咨询 / 供应商合规 这 4 类共用 direct 策略，
#    因此它们**彼此之间的混淆不产生任何路由差异**，业务代价为零。
#    实测测试集里 SOP 的 4 条误判全部落在法规查询(3)与供应商合规(1)，路由结果完全一致。
#    真正会产生代价的只有「事故处置」被误判出去（丢掉 subquery 拆分）或非事故类被误判进来
#    （白跑一次拆分，慢但不答错）——所以模型优化的优先级应该放在事故处置的边界上，
#    而不是去追 SOP 这一类的小数点。
INTENT_TO_STRATEGY = {
    "法规查询": "direct",
    "SOP 指导": "direct",
    "事故处置": "subquery",
    "营养咨询": "direct",
    "供应商合规": "direct",
    "闲聊": "direct",
}


class RuleIntentClassifier:
    """
    规则意图分类器：用特征词做匹配，零模型依赖。

    这是项目的**默认**分类器，也是模型缺失时的兜底。
    与旧版 select_strategy 的区别：旧版直接输出「检索策略」，现在输出「意图」，
    再由 INTENT_TO_STRATEGY 映射到策略——这样「意图」和「策略」两个概念解耦，
    将来调整策略不再需要动分类逻辑。
    """

    def __init__(self):
        # 延迟导入，避免模块级循环依赖（rag_system 会引用本模块）
        from core.dataset_builder import IntentDatasetBuilder
        self._keywords = IntentDatasetBuilder.INTENT_KEYWORDS

    def predict(self, query):
        """返回意图中文名；无法判断时返回 None"""
        query = (query or "").strip()
        if not query:
            return None
        for intent_name, keywords in self._keywords:
            if any(kw in query for kw in keywords):
                return intent_name
        return None

    @property
    def name(self):
        return "rule"


class ModelIntentClassifier:
    """模型意图分类器：加载微调好的分类模型做预测"""

    def __init__(self, model_path):
        # 延迟导入 torch/transformers：这两个包体积大，仅在真正需要时加载
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self._torch = torch
        logger.info(f"意图分类模型加载成功：{model_path}")

    # 低置信阈值：top1 概率低于此值时把样本落盘，作为下一轮补充标注的候选。
    # 0.60 是按实测分布定的——测试集错判样本的 top1 概率多在 0.28~0.65，
    # 而判对的样本大多在 0.80 以上；0.60 能捞到绝大部分错判，同时只占约 10% 的线上流量。
    LOW_CONF_THRESHOLD = 0.60

    def predict(self, query):
        """返回意图中文名；模型输出越界或为空时返回 None（由上层回退到规则）"""
        query = (query or "").strip()
        if not query:
            return None
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=64)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        label_id = int(probs.argmax().item())
        confidence = float(probs[label_id].item())

        # 只有分类器真正「拿不准」时才落盘，且不改变路由结果——记录难例不能影响回答
        if confidence < self.LOW_CONF_THRESHOLD:
            self._dump_hard_case(query, label_id, confidence, probs)

        if 0 <= label_id < len(INTENT_NAMES):
            return INTENT_NAMES[label_id]
        logger.warning(f"模型输出的标签 ID {label_id} 超出合法范围，忽略")
        return None

    def _dump_hard_case(self, query, label_id, confidence, probs):
        """
        把低置信样本追加到 logs/intent_hard_cases.jsonl。

        这是主动学习的**输入侧**闭环：模型拿不准的样本，正是最该补数据的地方。
        攒下来的难例按 core/dataset_builder.py 里的「标注互斥判据」人工确认后，
        直接并入下一轮训练集——比凭感觉批量造模板有效得多，也不会污染固定测试集。

        落盘失败一律吞掉：记录难例属于旁路能力，绝不能让日志写失败把问答拖挂。
        """
        try:
            import json
            from datetime import datetime

            os.makedirs(config.LOG_DIR, exist_ok=True)
            path = os.path.join(config.LOG_DIR, 'intent_hard_cases.jsonl')
            order = probs.argsort(descending=True).tolist()[:3]
            record = {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'text': query,
                'pred': INTENT_NAMES[label_id] if 0 <= label_id < len(INTENT_NAMES) else str(label_id),
                'confidence': round(confidence, 4),
                'top3': [
                    {'label': INTENT_NAMES[i], 'p': round(float(probs[i]), 4)}
                    for i in order if 0 <= i < len(INTENT_NAMES)
                ],
            }
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.debug(f"难例落盘失败（不影响问答）：{e}")

    @property
    def name(self):
        return "model"


def build_intent_classifier():
    """
    构建意图分类器：优先模型，失败则规则。
    :return: 具备 predict(query) / name 属性的分类器实例
    """
    model_path = config.INTENT_MODEL_PATH
    if not os.path.isdir(model_path):
        logger.info(f"未找到意图分类模型目录 {model_path}，使用规则分类器"
                    f"（运行 python build_dataset.py 准备数据，训练后将模型放入该目录即可自动接管）")
        return RuleIntentClassifier()

    try:
        return ModelIntentClassifier(model_path)
    except ImportError as e:
        logger.warning(f"加载意图模型所需依赖缺失（{e}），回退到规则分类器")
    except Exception as e:
        logger.warning(f"加载意图模型失败（{e}），回退到规则分类器")
    return RuleIntentClassifier()
