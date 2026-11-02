"""
    领域注册表：core/domain.py

    职责：把「一套检索/拒答链路」与「某个具体领域的语料」解耦。
        链路（切分 → 召回 → 精排 → 阈值 → 拒答 → 引用）是领域无关的；
        领域相关的只有三件事：知识库类型清单、每类文档的切分边界、拒答阈值取值。

    设计原则：
        · 领域定义落在 domains/<id>/manifest.json，是数据不是代码
        · 未在 manifest 里覆盖的项，一律回落到原来的内置默认值——
          保证「不加 manifest 也能跑」，food_safety 的行为一字不变
        · 不 import 任何重依赖（langchain / pymilvus），装配期不该拉起重库

    为什么需要这层：
        改造前，6 类知识库的分隔符写死在 core/text_splitter.py 的 KB_SEPARATOR_MAP 里，
        kb_type 的中文名写死在 base/config.py 的 KB_TYPES 里。想让同一套链路去跑
        「劳动合同法规」，只能改源码。改造后换领域 = 加一个 manifest + 放一批语料。
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 项目根目录（core/ 的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS_DIR = os.path.join(ROOT, "domains")

# 领域状态：reference = 主领域；cross_validation = 用于证明「链路与领域解耦」的第二领域
DOMAIN_STATUS = ("reference", "cross_validation", "production")


class DomainError(ValueError):
    """领域定义非法。装配期抛，不允许带病启动。"""


@dataclass(frozen=True)
class KbTypeSpec:
    """单个知识库类型的领域内定义。"""

    id: str
    label: str
    file_prefix: str = ""
    access_level: str = "internal"
    department: str = "general"
    # 该类型的切分边界（正则分隔符）。为空表示沿用内置 KB_SEPARATOR_MAP
    separators: Optional[List[str]] = None
    # 该类型的子块大小覆盖。为空表示沿用 config.CHILD_CHUNK_SIZE
    child_chunk_size: Optional[int] = None


@dataclass
class DomainSpec:
    """一个领域的完整定义。"""

    domain_id: str
    display_name: str
    version: str
    status: str
    description: str
    corpus_dir: str
    kb_types: List[KbTypeSpec] = field(default_factory=list)
    eval_golden_set: str = ""
    score_threshold: Optional[float] = None
    reuse_note: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---------- 查询 ----------

    def kb_type_ids(self) -> List[str]:
        return [k.id for k in self.kb_types]

    def labels(self) -> Dict[str, str]:
        """{kb_type: 中文名}，可直接替换 base.config.KB_TYPES。"""
        return {k.id: k.label for k in self.kb_types}

    def get_kb(self, kb_type: str) -> Optional[KbTypeSpec]:
        for k in self.kb_types:
            if k.id == kb_type:
                return k
        return None

    def separators_for(self, kb_type: str) -> Optional[List[str]]:
        kb = self.get_kb(kb_type)
        return kb.separators if kb else None

    def child_chunk_size_for(self, kb_type: str) -> Optional[int]:
        kb = self.get_kb(kb_type)
        return kb.child_chunk_size if kb else None

    def abs_corpus_dir(self) -> str:
        return self.corpus_dir if os.path.isabs(self.corpus_dir) else os.path.join(ROOT, self.corpus_dir)

    # ---------- 校验 ----------

    def validate(self) -> None:
        if not self.domain_id:
            raise DomainError("领域缺少 domain_id")
        if self.status not in DOMAIN_STATUS:
            raise DomainError(
                f"领域 {self.domain_id} 的 status={self.status} 非法，允许：{DOMAIN_STATUS}")
        if not self.kb_types:
            raise DomainError(f"领域 {self.domain_id} 未定义任何 kb_type")
        ids = self.kb_type_ids()
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise DomainError(f"领域 {self.domain_id} 存在重复 kb_type：{sorted(dup)}")
        for k in self.kb_types:
            if not k.id or not k.label:
                raise DomainError(f"领域 {self.domain_id} 的 kb_type 缺少 id 或 label")


def _build(data: Dict[str, Any]) -> DomainSpec:
    ev = data.get("eval") or {}
    refusal = data.get("refusal") or {}
    return DomainSpec(
        domain_id=data.get("domain_id", ""),
        display_name=data.get("display_name", ""),
        version=data.get("version", ""),
        status=data.get("status", "cross_validation"),
        description=data.get("description", ""),
        corpus_dir=data.get("corpus_dir", ""),
        kb_types=[
            KbTypeSpec(
                id=k.get("id", ""),
                label=k.get("label", ""),
                file_prefix=k.get("file_prefix", ""),
                access_level=k.get("access_level", "internal"),
                department=k.get("department", "general"),
                separators=k.get("separators"),
                child_chunk_size=k.get("child_chunk_size"),
            )
            for k in data.get("kb_types", [])
        ],
        eval_golden_set=ev.get("golden_set", ""),
        score_threshold=refusal.get("score_threshold"),
        reuse_note=data.get("reuse_note", ""),
        raw=data,
    )


def list_domains(domains_dir: str = DOMAINS_DIR) -> Dict[str, DomainSpec]:
    """加载全部领域定义（不校验语料是否存在）。"""
    if not os.path.isdir(domains_dir):
        return {}
    out: Dict[str, DomainSpec] = {}
    for name in sorted(os.listdir(domains_dir)):
        path = os.path.join(domains_dir, name, "manifest.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                spec = _build(json.load(fh))
            spec.validate()
            out[spec.domain_id] = spec
        except (json.JSONDecodeError, DomainError) as exc:
            raise DomainError(f"领域定义非法（{path}）：{exc}") from exc
    return out


def get_domain(domain_id: str) -> Optional[DomainSpec]:
    return list_domains().get(domain_id)


def active_domain() -> Optional[DomainSpec]:
    """
    当前激活领域。

    取值顺序：环境变量 ACTIVE_DOMAIN → status=reference 的那个 → 第一个。
    这是「一套链路、多个领域」的唯一切换点。
    """
    domains = list_domains()
    if not domains:
        return None
    target = os.getenv("ACTIVE_DOMAIN", "").strip()
    if target and target in domains:
        return domains[target]
    for d in domains.values():
        if d.status == "reference":
            return d
    return next(iter(domains.values()))


def separators_for(kb_type: str, domain: Optional[DomainSpec] = None) -> Optional[List[str]]:
    """领域内声明的切分边界；未声明返回 None 表示「沿用内置默认」。"""
    d = domain or active_domain()
    if d is None:
        return None
    return d.separators_for(kb_type)


def child_chunk_size_for(kb_type: str, domain: Optional[DomainSpec] = None) -> Optional[int]:
    d = domain or active_domain()
    if d is None:
        return None
    return d.child_chunk_size_for(kb_type)


def domain_summary() -> Dict[str, Any]:
    """启动日志 / 自检用的摘要。"""
    domains = list_domains()
    active = active_domain()
    return {
        "active": active.domain_id if active else None,
        "count": len(domains),
        "domains": [
            {
                "domain_id": d.domain_id,
                "display_name": d.display_name,
                "status": d.status,
                "kb_types": d.kb_type_ids(),
                "corpus_dir": d.corpus_dir,
            }
            for d in domains.values()
        ],
    }
