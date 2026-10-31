"""
    数据集构建：core/dataset_builder.py

    对应需求基线的「数据预处理 / 构建数据集」环节，为三个下游任务准备训练数据：

        1. IntentDatasetBuilder —— 意图分类数据集（6 类意图）
           用于训练「意图识别模型」，替代当前的规则化策略路由。
        2. RerankDatasetBuilder —— 重排数据集（query / positive / negative 三元组）
           用于微调 BGE-Reranker，提升本领域 Top-K 精度。
        3. SFTDatasetBuilder    —— 指令微调数据集（instruction / input / output）
           用于 LoRA 微调对话模型，格式与场景文档给出的示例一致。

    设计原则：
        - **不引入额外数据源**，全部从项目已有的 data/ 知识库中挖掘，保证可追溯。
        - **不使用 LLM 生成答案**，output 一律摘自知识库原文，避免「用幻觉数据训练模型」。
        - 意图样本的规则标注集按固定哈希切成「弱标注集」与「留出评估集」，
          留出集绝不出现在训练数据中，防止「用规则训练、又用规则评测」的自证式评估。
"""
import hashlib
import json
import random
import re
from collections import Counter

from base.config import config
from base.logger import logger

# 意图类别定义：(标签 ID, 英文名, 中文名)
# 顺序即标签 ID 顺序，改动会让已训练模型的标签错位，如需调整必须重新训练。
INTENT_LABELS = [
    (0, "regulation", "法规查询"),
    (1, "sop", "SOP 指导"),
    (2, "case", "事故处置"),
    (3, "nutrition", "营养咨询"),
    (4, "supplier", "供应商合规"),
    (5, "chitchat", "闲聊"),
]

# 中文名 -> 标签 ID 反查表（构建数据集时用）
LABEL_NAME_TO_ID = {name: idx for idx, _, name in INTENT_LABELS}
# 英文名 -> 标签 ID 反查表（训练脚本用）
LABEL_KEY_TO_ID = {key: idx for idx, key, _ in INTENT_LABELS}

# 意图标签说明（导出到 README，供标注人员与训练脚本对齐语义）
#
# ⚠️ 标注互斥判据（新增样本时按此裁决，否则同类问句会被标成不同标签，模型学到的就是噪声）
#    实测踩过的坑：「冰箱温度记录需要保存多久」被标成法规查询，而「食品留样应该留多少克、留多长时间」
#    被标成 SOP 指导——两句结构完全一样，标签却相反。这类矛盾样本会直接压低 SOP 类的 F1。
#
#    冲突时按以下顺序裁决，命中即停：
#      1. 是「已经发生」的负面事件（已发现/已发生/顾客说吃了/已经卖出去了）→ 事故处置
#      2. 主体是外部供方（供应商/物流商/包材商），且问资质、台账、协议、审核 → 供应商合规
#      3. 答案能被**一条可引用的条文**直接回答（是多少 / 几年 / 什么标准 / 有哪些义务）→ 法规查询
#      4. 答案是一串**可执行动作序列**（先做什么、再做什么、参数控多少）→ SOP 指导
#      5. 涉及营养素、膳食搭配、特殊人群配餐 → 营养咨询
#      6. 与业务无关的问候、身份询问、寒暄 → 闲聊
#
#    一句话区分法规 vs SOP：把问题换成「根据哪条规定」能答 → 法规；换成「按什么流程做」能答 → SOP。
INTENT_LABEL_DOC = {
    "法规查询": "询问国家标准、法规条款、限量值、法定期限等可引用条文能回答的问题。"
                "例：GB 2760 对苯甲酸钠的最大使用量是多少。"
                "边界：问「是多少/有效期几年/必须标注哪些内容」→ 本类；问「怎么做/什么步骤」→ SOP 指导",
    "SOP 指导": "询问具体操作做法、步骤、流程、控制点参数。"
                "例：冷链原料到货怎么验收。"
                "边界：答案是一串可执行动作时归本类；即使动作对象是供应商，只要问的是「怎么操作」也归本类",
    "事故处置": "顾客投诉、疑似食物中毒、异物、过敏等**已经发生**的突发事件的处置。"
                "例：顾客说吃了肚子疼怎么办。"
                "边界：句中含「已经/发现/发生」且诉求是「现在怎么办」→ 本类，优先级最高",
    "营养咨询": "膳食搭配、营养素、特殊人群配餐。例：学生餐怎么搭配更均衡",
    "供应商合规": "供应商资质审核、索证索票、供应商台账与协议。"
                  "例：供应商审核需要哪些资质材料。"
                  "边界：主体必须是**外部供方**；问「我方内部怎么操作」不算，归 SOP 指导",
    "闲聊": "问候、身份询问、与业务无关的寒暄。例：你好 / 你是谁",
}


def _stable_bucket(text, buckets=100):
    """
    把文本稳定地映射到 [0, buckets) 的桶号。

    用 md5 而不是内置 hash()：内置 hash 受 PYTHONHASHSEED 影响，
    同一份数据在不同进程里会分到不同桶，导致本地跑和 CI 跑出的划分不一致。
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


class IntentDatasetBuilder:
    """
    意图分类数据集构建器。

    数据来源分两条：
        A. 规则强标注集（弱标注）：用意图特征词命中 data/faq_data 中的真实问题，
           得到「query -> 意图」样本。这是**弱标注**——由规则产生，与线上路由同源。
        B. 种子样本（人工标注，hand seed）：手写的典型问句，用于覆盖规则覆盖不到的
           表达方式（尤其是「闲聊」类，规则集里几乎没有正样本）。

    泄漏控制：
        弱标注集内部按固定哈希做 8:2 切分，20% 作为评估集（eval），
        训练集则等于「弱标注集的 80% + 种子样本的 80%」。
        这样用训练集训出的模型在评估集上才能反映真实泛化，而不是复述规则。
    """

    # 每个意图的种子问句（人工编写，覆盖规则难以命中的口语化表达）
    # 扩充原则：每类尽量覆盖「书面问法 / 口语问法 / 省略主语问法」，让模型学到
    # 表达多样性，而不是记住几个固定句式。
    SEED_QUERIES = {
        "法规查询": [
            "GB 2760 对苯二甲酸钠在饮料中的最大使用量是多少",
            "食品添加剂的使用有哪些国家强制要求",
            "亚硝酸盐在肉制品中的残留限量标准是多少",
            "食品安全法规定食品生产经营者有哪些义务",
            "食品中铅的限量是多少",
            "菌落总数超标的判定依据是哪条国标",
            "预包装食品的标签必须标注哪些内容",
            "保质期的标注有什么强制性规定",
            "生产经营许可的有效期是几年",
            "食品召回管理办法对召回时限是怎么要求的",
            "禁止生产经营的食品有哪些",
            "食品添加剂的定义在法律上是怎么写的",
            "食品安全事故的分级标准是什么",
            "进销货台账要保存几年",
            "从业人员健康证的有效期是多久",
        ],
        "SOP 指导": [
            "餐具消毒要怎么做才算合格",
            "生熟分开具体要注意什么",
            "冷库进货验收的步骤是什么",
            "后厨人员的洗手消毒流程是怎样的",
            "食品留样具体要怎么做",
            "冷冻原料怎么解冻才合规",
            "备餐间的温度要控制在多少",
            "刀具砧板怎么清洗消毒",
            "更衣室的卫生要求有哪些",
            "冰箱应该多久除霜一次",
            "原料入库的验收流程是怎么走",
            "餐厨废弃物的处理步骤是什么",
            "烹饪后放多久必须处理掉",
            "紫外灯消毒怎么用才有效",
            "防虫防鼠的措施怎么做",
        ],
        "事故处置": [
            "有顾客投诉饭菜里吃到头发怎么办",
            "怀疑发生食物中毒应该先做什么",
            "客人吃完过敏了餐厅要怎么处理",
            "有员工举报后厨卫生问题该怎么调查",
            "顾客说吃了肚子疼怎么应对",
            "发现食材有异味已经卖出去了怎么办",
            "多人同时腹泻要怎么上报",
            "有顾客在店里晕倒应该先做什么",
            "监管抽检不合格要怎么处理",
            "被媒体曝光卫生问题怎么应对",
            "顾客要求十倍赔偿合理吗",
            "封存的问题食品要保留多久",
            "要不要主动向监管部门报告",
            "怎么安抚情绪激动的投诉顾客",
            "疑似异物混入的产品已发货怎么召回",
        ],
        "营养咨询": [
            "老人餐怎么搭配更符合膳食指南",
            "学生营养餐每天要吃多少蔬菜",
            "低盐餐应该控制在多少克盐以内",
            "补充蛋白质选哪些食材比较好",
            "糖尿病餐怎么设计主食",
            "减脂餐的热量应该控制在多少",
            "孕期膳食要注意补充什么",
            "幼儿园配餐怎么做到营养均衡",
            "术后流食应该怎么安排",
            "素食人群怎么保证蛋白质摄入",
            "每天应该摄入多少膳食纤维",
            "运动员餐的碳水比例是多少",
            "钙的每日推荐摄入量是多少",
            "儿童早餐怎么搭配比较好",
            "高血压人群的饮食要注意什么",
        ],
        "供应商合规": [
            "进货时供应商要提供哪些证件",
            "怎么审核一家新的原料供应商",
            "索证索票制度要求留存哪些材料",
            "供应商的冷链资质怎么验证",
            "肉类供应商需要哪些检验证明",
            "供应商的营业执照要每年核对吗",
            "进口原料的报关单和检疫证明要不要留",
            "供应商现场审核要看哪些项目",
            "供应商出了问题怎么追责",
            "建立供应商档案要包含哪些内容",
            "供应商评分表怎么设计",
            "供应商的运输车辆要符合什么条件",
            "食品包装材料供应商有哪些特殊要求",
            "怎么验证供应商的检验报告真伪",
            "供应商准入的审批流程是什么",
        ],
        "闲聊": [
            "你好",
            "早上好，在吗",
            "你是谁",
            "你叫什么名字",
            "今天天气不错啊",
            "谢谢你",
            "你能帮我做点别的事情吗",
            "再见",
            "喂",
            "在不在",
            "你能干什么",
            "你是机器人吗",
            "辛苦了",
            "好久不见",
            "打扰一下",
            "我随便问问",
            "嗯嗯",
            "好的明白了",
            "你是哪个公司做的",
            "你能听懂方言吗",
        ],
    }

    # 意图 -> 特征词（用于弱标注）。命中任一特征词即认为该 query 属于该意图。
    # 注意：特征词顺序即优先级，先匹配的意图胜出；把「事故处置」放在「法规查询」前面，
    #      因为「食物中毒怎么处理」同时含「食物中毒」（事故）与「规定」（法规），
    #      但用户真实意图是问处置流程。
    INTENT_KEYWORDS = [
        ("事故处置", ["中毒", "投诉", "异物", "过敏", "受伤", "肚子疼", "拉肚子", "呕吐",
                      "举报", "召回", "封存", "送医", "急救", "顾客吃", "顾客说"]),
        ("SOP 指导", ["怎么做", "怎么操作", "操作步骤", "流程", "步骤", "验收怎么", "怎么验收",
                      "消毒", "清洗", "解冻", "存放", "留样", "洗手", "更衣", "生熟分开",
                      "怎么处理", "如何操作", "怎么保存"]),
        ("供应商合规", ["供应商", "索证", "索票", "资质", "进货", "采购", "台账", "验收标准",
                        "原料验收", "合格证", "检验报告", "营业执照", "生产许可"]),
        ("营养咨询", ["营养", "膳食", "搭配", "热量", "蛋白质", "维生素", "菜谱", "配餐",
                      "盐", "糖", "脂肪", "钙", "膳食纤维", "特殊人群", "学生餐", "老人餐"]),
        ("法规查询", ["GB", "国标", "标准", "限量", "最大使用量", "规定", "法规", "条例",
                      "食品安全法", "要求是多少", "不得超过", "允许", "禁止", "条款",
                      "多少克", "多少毫克", "几天", "多久", "多少度"]),
    ]

    # 主体词：每个意图的核心业务对象（名词），用于模板扩写时替换主语
    INTENT_SUBJECTS = {
        "法规查询": ["食品添加剂", "亚硝酸盐", "防腐剂", "农药残留", "重金属", "菌落总数",
                     "大肠菌群", "食品标签", "保质期标注", "生产经营许可"],
        "SOP 指导": ["餐具", "后厨台面", "冷库", "操作台", "刀具砧板", "食品留样",
                     "原料解冻", "冰箱", "更衣室", "备餐间"],
        "事故处置": ["顾客投诉", "疑似食物中毒", "异物混入", "过敏反应", "员工受伤",
                     "批量腹泻", "顾客晕倒", "食品变质投诉", "媒体曝光", "监管抽检不合格"],
        "营养咨询": ["学生餐", "老人餐", "幼儿园配餐", "糖尿病餐", "低盐餐", "减脂餐",
                     "孕期膳食", "运动员餐", "术后流食", "素食餐"],
        "供应商合规": ["肉类供应商", "蔬菜供应商", "冷链物流商", "一次性餐具供应商",
                       "调味品供应商", "进口原料供应商", "中央厨房", "食品包装材料商",
                       "粮油供应商", "水产供应商"],
        "闲聊": ["你", "这个系统", "帮忙", "今天", "天气"],
    }

    # 疑问句式模板：{subject} 占位符会被 INTENT_SUBJECTS 替换
    # 同一主体 × 多个句式 = 一批语料，是低成本扩数据的主力。
    # 刻意覆盖多种表达（口语/书面/多问号），让模型学会「句式变化不影响意图」。
    INTENT_TEMPLATES = {
        "法规查询": [
            "{subject}的国家标准编号是多少",
            "{subject}在法规里是怎么规定的",
            "{subject}的限量标准是什么",
            "{subject}最大允许使用量是多少",
            "关于{subject}，食品安全法有什么要求",
            "{subject}不得超过多少",
            "{subject}有没有国标要求",
            "请问{subject}的相关条款是第几条",
            "{subject}的强制性标准有哪些",
            "{subject}要求是多少，超了会怎样",
        ],
        "SOP 指导": [
            "{subject}怎么操作才算规范",
            "{subject}的操作步骤是什么",
            "{subject}应该怎么清洗消毒",
            "{subject}的正确做法是怎样的",
            "{subject}的管理流程有哪些环节",
            "{subject}要怎么做才能达标",
            "{subject}日常应该怎么维护",
            "{subject}的具体操作要求是什么",
            "{subject}怎么处理比较合适",
            "{subject}的标准操作流程是什么",
        ],
        "事故处置": [
            "有{subject}应该怎么处理",
            "{subject}了第一时间要做什么",
            "{subject}发生后怎么上报",
            "遇到{subject}该怎么应对",
            "{subject}需要保留哪些证据",
            "{subject}应该找谁处理",
            "{subject}的应急处置流程是什么",
            "出现{subject}要怎么安抚顾客",
            "{subject}要不要送医",
            "{subject}了还能营业吗",
        ],
        "营养咨询": [
            "{subject}怎么搭配更营养",
            "{subject}每天应该吃多少",
            "{subject}的膳食建议是什么",
            "{subject}怎么配比更均衡",
            "{subject}适合吃哪些食材",
            "{subject}的热量应该控制在多少",
            "{subject}怎么安排更科学",
            "{subject}要注意补充什么营养",
            "{subject}的一日食谱怎么设计",
            "{subject}的营养标准是什么",
        ],
        "供应商合规": [
            "{subject}需要提供哪些资质",
            "{subject}的审核流程是什么",
            "{subject}怎么验收才合规",
            "{subject}要留存哪些索证材料",
            "{subject}的准入条件有哪些",
            "怎么审核{subject}",
            "{subject}需要哪些证件才算合格",
            "{subject}的台账应该怎么记",
            "{subject}不符合要求怎么办",
            "{subject}要签哪些协议",
        ],
        "闲聊": [
            "你好，我想问问{subject}",
            "{subject}今天怎么样",
            "在吗，{subject}有点问题想请教",
            "谢谢{subject}",
            "{subject}能帮我个忙吗",
            "早上好，{subject}那边情况如何",
            "{subject}是谁",
            "再见，{subject}",
        ],
    }

    def __init__(self, questions, eval_ratio=0.2, seed=42, augment=True):
        """
        :param questions: 从知识库中抽取的真实问句列表（只含 query 字符串）
        :param eval_ratio: 评估集占比（默认 0.2）
        :param seed: 划分随机种子，固定以保证可复现
        :param augment: 是否开启模板扩写。
            开启后样本量约提升一个数量级，否则 6 分类只有 30 条训练样本，
            模型学不到东西。扩写样本 source 标为 'template'，便于质量追溯与按需剔除。
        """
        self.questions = list(questions)
        self.eval_ratio = eval_ratio
        self.seed = seed
        self.augment = augment

    def _match_intent(self, query):
        """用特征词规则给 query 打意图标签，命中不到返回 None"""
        for intent_name, keywords in self.INTENT_KEYWORDS:
            if any(kw in query for kw in keywords):
                return intent_name
        return None

    def _auto_samples(self):
        """弱标注集：对知识库抽取的问句逐条做规则打标"""
        samples = []
        skipped = 0
        for query in self.questions:
            intent_name = self._match_intent(query)
            if intent_name is None:
                skipped += 1
                continue
            samples.append({
                "text": query,
                "label": LABEL_NAME_TO_ID[intent_name],
                "label_name": intent_name,
                "source": "auto_keyword",
            })
        logger.info(f"意图弱标注：命中 {len(samples)} 条，未命中 {skipped} 条")
        return samples

    def _seed_samples(self):
        """种子集：人工编写的典型问句"""
        samples = []
        for intent_name, queries in self.SEED_QUERIES.items():
            for query in queries:
                samples.append({
                    "text": query,
                    "label": LABEL_NAME_TO_ID[intent_name],
                    "label_name": intent_name,
                    "source": "seed",
                })
        logger.info(f"意图种子样本：{len(samples)} 条")
        return samples

    def _template_samples(self):
        """
        模板扩写集：主体词 × 疑问句式 组合生成。

        为什么需要它：知识库真实问句只有几十条，6 分类下每类不足 10 条，
        直接训练等于让模型背答案。模板扩写用「同义主体 × 多种句式」把样本量
        拉高一个数量级，且每条都带明确意图标签（构造即标注，无标注噪声）。

        局限（必须承认）：
            模板语料的句式和用词分布比真实用户窄，模型可能过拟合到模板措辞。
            因此 source 标记为 'template'，评估集切分时优先级最低——
            即「先用真实问句和种子撑住评估集，不足再用模板补」，保证评估集
            尽可能贴近真实分布。
        """
        samples = []
        for intent_name, templates in self.INTENT_TEMPLATES.items():
            subjects = self.INTENT_SUBJECTS.get(intent_name, [])
            for subject in subjects:
                for template in templates:
                    text = template.format(subject=subject)
                    samples.append({
                        "text": text,
                        "label": LABEL_NAME_TO_ID[intent_name],
                        "label_name": intent_name,
                        "source": "template",
                    })
        logger.info(f"意图模板扩写样本：{len(samples)} 条")
        return samples

    def build(self):
        """
        构建并返回 (train, eval) 两个列表。

        切分策略（关键改动）：
            先按 text 哈希初分（保证同一句话不跨集），再做**评估集净化**：
            控制模板样本在评估集中的占比不超过其自然比例，把超出的模板样本
            换回训练集、并用训练集的真实样本顶替（不超过训练集真实样本的一半，
            避免掏空训练集）。

            这样做的原因：模板样本是「主体词 × 句式」机械组合出来的，句式分布
            比真实提问窄。若评估集被模板样本占满，评估分数会虚高，反映不出对
            真实用户提问的泛化能力。评估集应尽量贴近线上分布。
            另：train/eval 仍严格互斥（同一句话只出现在一边）。
        """
        real_samples = self._auto_samples() + self._seed_samples()
        template_samples = self._template_samples() if self.augment else []

        # 去重：同一句话只保留一条（保留先出现的，即真实 > 种子 > 模板的优先级）
        dedup = {}
        for sample in real_samples + template_samples:
            dedup.setdefault(sample["text"], sample)
        all_samples = list(dedup.values())

        bucket_limit = int(self.eval_ratio * 100)

        # 按哈希初分。注意此时**不区分来源**——真实样本与模板样本都可能落到评估集，
        # 关键是后续要控制评估集里模板样本的占比，避免评估分数虚高。
        eval_set, train = [], []
        for sample in all_samples:
            if _stable_bucket(sample["text"]) < bucket_limit:
                eval_set.append(sample)
            else:
                train.append(sample)

        # 评估集净化：让评估集**优先由真实样本构成**。
        #
        # 目标构成：真实样本（种子 + 知识库问句）在其自身 20% 的规模上全部进评估集，
        #          剩余名额用模板样本补足；若真实样本不够填满评估集，才用模板兜底。
        #
        # 为什么这样更好：
        #   评估集衡量的是「模型对真实用户提问的泛化能力」。真实样本只有 100 条出头，
        #   把它们**尽可能**放进评估集（而不是按 84% 的模板比例稀释），评估结论才可信。
        #   同时必须给训练集留下足够真实样本——所以换入量上限设为真实样本总数的一半，
        #   保证训练集至少还有一半真实样本可学，不会「掏空训练集」。
        real_total = sum(1 for s in all_samples if s["source"] != "template")
        eval_template = [s for s in eval_set if s["source"] == "template"]
        real_in_train = [s for s in train if s["source"] != "template"]

        if eval_template and real_in_train:
            # 交换上限：真实样本总数的一半。
            # 理由：真实样本池很小时，全给评估集会让训练集退化成纯模板训练；
            #      留一半在训练集，两边都有真实分布。
            max_swap = min(len(eval_template), max(1, real_total // 2))
            if max_swap > 0:
                swap_out = sorted(eval_template, key=lambda s: _stable_bucket(s["text"]),
                                  reverse=True)[:max_swap]
                swap_in = sorted(real_in_train, key=lambda s: _stable_bucket(s["text"]))[:max_swap]
                for item in swap_out:
                    eval_set.remove(item)
                    train.append(item)
                for item in swap_in:
                    train.remove(item)
                    eval_set.append(item)
                logger.info(f"评估集净化：换出模板样本 {max_swap} 条，换入真实样本 {max_swap} 条"
                            f"（真实样本池共 {real_total} 条，保留至少一半在训练集）")

        rand = random.Random(self.seed)
        rand.shuffle(train)
        rand.shuffle(eval_set)

        logger.info(f"意图数据集构建完成：训练集 {len(train)} 条，评估集 {len(eval_set)} 条")
        return train, eval_set

    @staticmethod
    def stats(samples):
        """返回样本集的类别分布（用于数据均衡性检查）"""
        counter = Counter(sample["label_name"] for sample in samples)
        return {name: counter.get(name, 0) for _, _, name in INTENT_LABELS}

    @staticmethod
    def source_stats(samples):
        """返回样本集的来源分布（真实/种子/模板），用于检查评估集是否被模板样本污染"""
        counter = Counter(sample.get("source", "unknown") for sample in samples)
        return dict(counter)


class RerankDatasetBuilder:
    """
    重排数据集构建器：生成 (query, positive, negative) 三元组。

    正样本来源（都是「问句 ↔ 真实答案文本」的自然配对）：
        1. FAQ 的 Q-A 对：Q 作为 query，A 作为 positive。
        2. 文档标题为问句时：标题作为 query，其所在段落作为 positive。

    负样本（hard negative，比随机负例更难、更有训练价值）：
        从**同一知识库的其他文档**里取一段做负例——它们主题相近（都讲食品安全），
        但并非该问句的答案，模型必须学会细粒度区分。
        绝不从「同一条问答对内部」取反例，也绝不跨库乱取（那太容易，学不到东西）。
    """

    def __init__(self, faq_pairs, doc_sections, seed=42, negatives_per_query=1):
        """
        :param faq_pairs: [(question, answer), ...] 来自 FAQ 知识库
        :param doc_sections: [(title, section_text), ...] 来自其余知识库的标题-正文对
        :param negatives_per_query: 每个 query 采几条负例
        """
        self.faq_pairs = faq_pairs
        self.doc_sections = doc_sections
        self.seed = seed
        self.negatives_per_query = negatives_per_query

    def _positive_pool(self):
        """汇总所有正样本 (query, positive_text)"""
        pool = []
        for question, answer in self.faq_pairs:
            pool.append((question, answer))
        for title, text in self.doc_sections:
            pool.append((title, text))
        return pool

    def build(self):
        """构建三元组并返回列表"""
        pool = self._positive_pool()
        if not pool:
            logger.warning("重排数据集：没有可用正样本")
            return []

        # 负例候选池：所有正样本文本（去掉与当前正例同一条的那个）
        candidates = [text for _, text in pool]
        rand = random.Random(self.seed)

        triples = []
        for query, positive in pool:
            negatives = []
            # 最近邻式的 hard negative：从正例池里随机取样，但过滤掉和正例同一段的
            # （真实项目里这一步应改用向量检索召回 top-N 近邻，此处保持零额外依赖）
            shuffled = candidates[:]
            rand.shuffle(shuffled)
            for candidate in shuffled:
                if candidate == positive:
                    continue
                negatives.append(candidate)
                if len(negatives) >= self.negatives_per_query:
                    break
            if len(negatives) < self.negatives_per_query:
                continue
            triples.append({
                "query": query,
                "positive": positive,
                "negatives": negatives,
            })

        logger.info(f"重排数据集构建完成：{len(triples)} 条三元组")
        return triples


class SFTDatasetBuilder:
    """
    指令微调数据集构建器：生成 instruction / input / output 三段式样本。

    格式对齐场景文档给出的示例：
        {"instruction": "冰箱冷藏温度标准是多少？冷冻呢？", "input": "", "output": "根据 GB 31654-2021..."}

    重要约定：
        - output **必须摘自知识库原文**，绝不调用 LLM 生成。
          原因：用模型自己生成的答案去训练模型，会把幻觉固化成「标准答案」。
        - input 字段用于承载「场景限定」（如"学校食堂"），FAQ 无此信息时留空，
          保持字段存在但为空字符串，方便训练脚本统一处理。
    """

    def __init__(self, faq_pairs, case_docs, system_hint=None):
        """
        :param faq_pairs: [(question, answer), ...]
        :param case_docs: [(question, answer, scene), ...] 需要带场景限定的样本
        :param system_hint: 可选的系统提示前缀，写入每条样本以便训练时复现线上风格
        """
        self.faq_pairs = faq_pairs
        self.case_docs = case_docs
        self.system_hint = system_hint or "你是食品安全领域的专业助手，回答必须严谨、有依据，引用法规须注明标准编号与版本。"

    def build(self):
        """构建指令微调样本并返回列表"""
        samples = []
        for question, answer in self.faq_pairs:
            samples.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "system": self.system_hint,
                "source": "faq",
            })
        for question, answer, scene in self.case_docs:
            samples.append({
                "instruction": question,
                "input": scene,
                "output": answer,
                "system": self.system_hint,
                "source": "case_scene",
            })
        logger.info(f"指令微调数据集构建完成：{len(samples)} 条")
        return samples


# ============ 数据抽取辅助函数 ============

# FAQ 问答对解析：从 FAQ 文档里抽出 (Q, A) 列表
_FAQ_Q_PATTERN = re.compile(r"^Q[:：]\s*(.+)$")
_FAQ_A_PATTERN = re.compile(r"^A[:：]\s*(.+)$")
# Markdown 标题：用于从其余知识库抽出「标题-正文」对
_MD_TITLE_PATTERN = re.compile(r"^#{1,4}\s+(.+?)\s*$")
# 问句特征：句尾问号，或含疑问词
_QUESTION_PATTERN = re.compile(
    r"(？|\?)|(吗|呢|多少|多久|几天|怎么|如何|为什么|哪些|哪个|能不能|可不可以|是否)"
)


def is_question(text):
    """判断一段文本是否是问句（用于从知识库中挖掘意图训练样本）"""
    text = (text or "").strip()
    if not text or len(text) > 60:
        # 太长的多半是段落而不是问句，直接排除
        return False
    return bool(_QUESTION_PATTERN.search(text))


def parse_faq_pairs(faq_text):
    """
    从 FAQ Markdown 文本中解析出问答对。

    切分口径必须与 core/text_splitter.py 的 FAQ_SEPARATORS 保持一致：
    只按 Q 起始，A 的内容归属其上一条 Q，绝不在 A 处断开。
    """
    pairs = []
    current_question = None
    current_answer_lines = []

    for raw_line in faq_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        q_match = _FAQ_Q_PATTERN.match(line)
        a_match = _FAQ_A_PATTERN.match(line)
        if q_match:
            # 遇到新的 Q，先把上一条问答对收尾
            if current_question and current_answer_lines:
                pairs.append((current_question, " ".join(current_answer_lines)))
            current_question = q_match.group(1).strip()
            current_answer_lines = []
        elif a_match:
            current_answer_lines.append(a_match.group(1).strip())

    # 收尾最后一对
    if current_question and current_answer_lines:
        pairs.append((current_question, " ".join(current_answer_lines)))
    return pairs


def parse_doc_sections(doc_text):
    """
    从 Markdown 文档中解析出 (标题, 正文) 对，用于重排数据集的补充正样本。
    只取长度适中的正文段，太短的（如纯目录）跳过。
    """
    sections = []
    current_title = None
    current_lines = []

    def flush():
        if current_title and current_lines:
            body = " ".join(line for line in current_lines if line).strip()
            if len(body) >= 40:
                sections.append((current_title, body))

    for raw_line in doc_text.splitlines():
        line = raw_line.strip()
        title_match = _MD_TITLE_PATTERN.match(line)
        if title_match:
            flush()
            current_title = title_match.group(1).strip()
            current_lines = []
        elif line:
            current_lines.append(line)
    flush()
    return sections


def write_jsonl(path, records):
    """把记录列表写成 JSONL（每行一个 JSON 对象，UTF-8 不转义中文）"""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"已写出 {len(records)} 条记录 -> {path}")


def write_label_map(path):
    """把意图标签映射写成 JSON，供训练脚本与推理端共用，避免标签顺序错位"""
    label_map = {
        "labels": [
            {"id": idx, "key": key, "name": name, "description": INTENT_LABEL_DOC.get(name, "")}
            for idx, key, name in INTENT_LABELS
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info(f"已写出意图标签映射 -> {path}")
