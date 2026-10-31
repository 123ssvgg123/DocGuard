"""
    文档处理器：core/document_processor.py

    职责：把 data/ 下各知识库目录中的原始文档，加载 → 解析权限 → 结构化切分
         → 输出「子块 + 父块」列表。

    目录约定（目录名去掉 _data 后缀即知识库类型 kb_type）：
        data/regulation_data/  法规标准库
        data/sop_data/         SOP 规范库
        data/case_data/        事故处置案例库
        data/nutrition_data/   营养学知识库
        data/supplier_data/    供应商资质库
        data/faq_data/         FAQ 问答库

    三条链路共用同一套切分与权限逻辑，只是「喂进来的文件集合」不同：
        · process_documents(目录)        —— 单个知识库目录全量
        · process_all_knowledge_bases()  —— 全部知识库全量（出口做 ID 唯一性校验）
        · process_files(相对路径列表)     —— 增量更新只处理变化的文件（见 core/incremental.py）

    权限（需求 4.2 行级隔离）在**加载阶段**就写进元数据，理由：
        切分只是把文本切小，权限是「文档级」属性，切完再补容易漏；
        而且 split_documents 会把父文档的 metadata 原样复制给每个子块，
        在父文档上写一次即可自动覆盖全部子块与父块。
"""
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from base.config import config
from base.logger import logger
from core.access_control import parse_front_matter, resolve_doc_acl
from core.incremental import file_digest, rel_key
from core.loaders import (  # noqa: F401  （MarkdownLoader / document_loaders 为兼容旧引用而再导出）
    MarkdownLoader,
    PlainTextLoader,
    build_loader,
    iter_supported_files,
    loader_report,
)
from core.text_splitter import get_parent_splitter, get_child_splitter

# 说明：加载器注册表（原 document_loaders 字典）已抽到 core/loaders.py 的 LOADER_CHAIN，
#      并按「首选 → 轻量替代 → 零依赖兜底」组织成降级链，见该文件头部说明。


# ==================================================================
# 知识库类型与块 ID
# ==================================================================

def kb_type_from_dir(dir_name: str) -> str:
    """目录名去掉 `_data` 后缀即知识库类型：regulation_data → regulation"""
    return dir_name[:-5] if dir_name.endswith("_data") else dir_name


def _doc_key(kb_type, file_name):
    """
    生成「知识库 + 文件」级别的稳定前缀，用于构造全局唯一的块 ID。

    ⚠️ 为什么必须带前缀（这是一个真实踩过的坑）：
        process_all_knowledge_bases() 是**逐个目录**调用 process_documents() 的，
        而 process_documents 内部的 enumerate(documents) 序号 i 在**每个目录里都从 0 重新开始**。
        如果 ID 只用 f'doc_{i}_parent_{j}_child_{k}'：
            case_data      → doc_0_parent_0_child_0
            faq_data       → doc_0_parent_0_child_0   ← 撞车
            nutrition_data → doc_0_parent_0_child_0   ← 又撞
        6 个目录全从 doc_0 起算，222 个子块只有 93 个唯一 ID。
        写库用的是 upsert（按主键覆盖），后写入的会**静默覆盖**先写入的，
        最终集合里只剩 93 条 —— 日志却打印"写入 222"，
        表现为"检索不出来 / 答案不全"，且极难排查。

    前缀取 file_name 的 md5 前 12 位而不是原文：
        文件名含中文与空格，直接拼进 ID 会让 VARCHAR 主键变长且不好读；
        md5 前缀既短、又稳定（同一文件每次构建得到相同 ID，保证增量入库幂等）。
    """
    raw = f"{kb_type}/{file_name}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]


# ==================================================================
# 加载
# ==================================================================

def load_one_file(file_path: str, kb_type: str) -> Tuple[List[Document], Optional[str]]:
    """
    加载单个文件，并写入「来源 + 权限 + 内容哈希」元数据。

    :return: (Document 列表, 错误信息)。错误信息为 None 表示加载成功；
             非 None 表示该文件被跳过（调用方需要把跳过原因汇总打印出来）。
    """
    loader, desc = build_loader(file_path, encoding='utf-8')
    if loader is None:
        return [], desc

    file_name = os.path.basename(file_path)
    try:
        raw_docs = loader.load()
    except Exception as exc:  # noqa: BLE001
        return [], f"加载失败（{desc}）：{type(exc).__name__}: {exc}"

    docs: List[Document] = []
    for doc in raw_docs:
        text = doc.page_content or ""
        if not text.strip():
            continue
        # ① 解析文件头部权限声明块，并与知识库默认权限合并
        front_matter, body = parse_front_matter(text)
        acl = resolve_doc_acl(kb_type, front_matter)

        doc.page_content = body if body.strip() else text
        doc.metadata.update({
            'source': kb_type,                      # 知识库类型（沿用既有字段名）
            'file_path': file_path,
            'file_name': file_name,
            'timestamp': datetime.now().isoformat(),
            # ② 权限三元组（需求 4.2 的行级隔离依据）
            'access_level': acl['access_level'],
            'department': acl['department'],
            'visible_to': acl['visible_to'],
            # ③ 内容哈希：供审计与增量比对核对
            'content_hash': file_digest(file_path),
        })
        docs.append(doc)

    if not docs:
        return [], f"解析后内容为空（{desc}）"
    meta0 = docs[0].metadata
    logger.info(f"加载文档成功：{file_path}（知识库：{kb_type}，加载器：{desc}，"
                f"权限：{meta0.get('access_level')}/{meta0.get('department')}"
                f"{'，白名单=' + meta0['visible_to'] if meta0.get('visible_to') else ''}）")
    return docs, None


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    从指定知识库目录加载所有支持的文档（不切分）。

    返回：Document 列表，每条 metadata 含
        source(知识库类型) / file_path / file_name / timestamp /
        access_level / department / visible_to / content_hash
    """
    documents: List[Document] = []
    kb_type = kb_type_from_dir(os.path.basename(directory_path))
    skipped: List[str] = []

    for file_path, _ext in iter_supported_files(directory_path):
        docs, err = load_one_file(file_path, kb_type)
        if err:
            skipped.append(f"{os.path.basename(file_path)}：{err}")
            logger.warning(f"跳过文件 {file_path} —— {err}")
            continue
        documents.extend(docs)

    if skipped:
        logger.warning(f"目录 {directory_path} 共跳过 {len(skipped)} 个文件：{skipped}")
    return documents


# ==================================================================
# 切分
# ==================================================================

def split_loaded_document(doc: Document,
                          parent_splitter,
                          child_splitter) -> List[Document]:
    """
    对**单个已加载的父文档**做父子两级切分，返回子块列表。

    父块 metadata：id = {doc_key}_p{j} 且带 parent_content
    子块 metadata：id = {doc_key}_p{j}_c{k}、parent_id 指向父块、同样带 parent_content
    """
    kb_type = doc.metadata.get('source', 'unknown')
    doc_key = _doc_key(kb_type, doc.metadata.get('file_name', 'unknown'))

    child_chunks: List[Document] = []
    parent_docs = parent_splitter.split_documents([doc])
    for j, parent_doc in enumerate(parent_docs):
        parent_doc.metadata["id"] = f'{doc_key}_p{j}'
        parent_doc.metadata["parent_content"] = parent_doc.page_content
        child_docs = child_splitter.split_documents([parent_doc])
        for k, child_doc in enumerate(child_docs):
            child_doc.metadata["parent_id"] = parent_doc.metadata["id"]
            child_doc.metadata["parent_content"] = parent_doc.page_content
            child_doc.metadata["id"] = f'{doc_key}_p{j}_c{k}'
            child_chunks.append(child_doc)
    logger.info(f"处理完成：{doc.metadata.get('file_name')}，切出 {len(parent_docs)} 个父块")
    return child_chunks


def process_documents(directory_path,
                      parent_chunk_size=config.PARENT_CHUNK_SIZE,
                      child_chunk_size=config.CHILD_CHUNK_SIZE,
                      parent_chunk_overlap=config.PARENT_CHUNK_OVERLAP,
                      child_chunk_overlap=config.CHILD_CHUNK_OVERLAP):
    """
    加载目录下文档并做父子两级切分。

    返回：子块 Document 列表，每个子块 metadata 含：
        id             子块全局唯一 ID   {doc_key}_p{j}_c{k}
        parent_id      父块 ID           {doc_key}_p{j}
        parent_content 父块全文（命中后作为上下文送入提示词）
        source         知识库类型
        file_name / file_path / timestamp
        access_level / department / visible_to   权限三元组
    """
    documents = load_documents_from_directory(directory_path)
    if not documents:
        logger.warning(f"目录 {directory_path} 下没有加载到任何文档")
        return []

    kb_type = documents[0].metadata.get('source', 'regulation')
    parent_splitter = get_parent_splitter(kb_type)
    child_splitter = get_child_splitter(kb_type)
    logger.info(f"知识库 {kb_type} 使用分块策略：parent={parent_chunk_size}/{parent_chunk_overlap}, "
                f"child={child_chunk_size}/{child_chunk_overlap}")

    child_chunks: List[Document] = []
    for doc in documents:
        child_chunks.extend(split_loaded_document(doc, parent_splitter, child_splitter))

    logger.info(f"目录 {directory_path} 共生成子块 {len(child_chunks)} 个")
    return child_chunks


def process_files(rel_keys: List[str],
                  data_dir: Optional[str] = None) -> Dict[str, List[Document]]:
    """
    只处理指定的若干文件（增量更新用）。

    :param rel_keys: 相对 data/ 的路径键（core/incremental.py 的 rel_key 产出）
    :return: {相对键: 子块列表}；加载失败的文件不出现在结果里（日志给出原因）
    """
    data_dir = data_dir or config.DATA_DIR
    result: Dict[str, List[Document]] = {}
    # 切分器按知识库缓存，避免同一目录下反复构建
    splitters: Dict[str, Tuple] = {}

    for key in rel_keys:
        file_path = os.path.join(data_dir, key.replace('/', os.sep))
        if not os.path.isfile(file_path):
            logger.warning(f"[增量] 文件不存在，跳过：{file_path}")
            continue

        kb_type = kb_type_from_dir(os.path.dirname(file_path).split(os.sep)[-1])
        docs, err = load_one_file(file_path, kb_type)
        if err:
            logger.error(f"[增量] 处理失败 {key} —— {err}")
            continue

        if kb_type not in splitters:
            splitters[kb_type] = (get_parent_splitter(kb_type), get_child_splitter(kb_type))
        parent_splitter, child_splitter = splitters[kb_type]

        chunks: List[Document] = []
        for doc in docs:
            chunks.extend(split_loaded_document(doc, parent_splitter, child_splitter))
        result[key] = chunks
        logger.info(f"[增量] {key} → {len(chunks)} 个子块")

    return result


def process_all_knowledge_bases():
    """
    遍历 data/ 下所有 *_data 目录，完成全部知识库的加载与切分。
    返回：全部子块列表

    出口处做「ID 全局唯一」校验：一旦发现重复立即抛错。
    原因见 _doc_key 的注释——ID 撞车会让 upsert 静默覆盖，
    日志显示写入 N 条而库里只有 M 条（M < N），属于最难排查的一类问题。
    宁可在建库阶段直接失败，也不要带着残缺的向量库进入检索。
    """
    all_chunks: List[Document] = []
    if not os.path.isdir(config.DATA_DIR):
        logger.warning(f"数据目录不存在：{config.DATA_DIR}")
        return all_chunks

    for dir_name in sorted(os.listdir(config.DATA_DIR)):
        dir_path = os.path.join(config.DATA_DIR, dir_name)
        if not os.path.isdir(dir_path) or not dir_name.endswith('_data'):
            continue
        logger.info(f"===== 开始处理知识库目录：{dir_name} =====")
        all_chunks.extend(process_documents(dir_path))
    logger.info(f"全部知识库切分完成，共 {len(all_chunks)} 个子块")

    assert_unique_ids(all_chunks)
    return all_chunks


def assert_unique_ids(chunks: List[Document]) -> None:
    """子块 ID 全局唯一性校验（重复直接抛错，宁可建库失败也不要残缺向量库）"""
    ids = [c.metadata.get('id') for c in chunks]
    unique_ids = set(ids)
    if len(unique_ids) != len(ids):
        duplicates = [i for i in unique_ids if ids.count(i) > 1]
        sample = duplicates[:5]
        raise ValueError(
            f"子块 ID 存在重复：{len(ids)} 个子块仅有 {len(unique_ids)} 个唯一 ID，"
            f"重复 {len(ids) - len(unique_ids)} 个。示例重复 ID：{sample}。"
            f"写库用的是 upsert（按主键覆盖），重复 ID 会导致子块被静默丢弃，"
            f"请检查 _doc_key() 的前缀构造是否覆盖了所有知识库目录。"
        )
    logger.info(f"块 ID 唯一性校验通过：{len(unique_ids)} 个唯一 ID")


def describe_permissions(chunks: List[Document]) -> Dict[str, Dict[str, int]]:
    """统计「知识库 × 密级」与「知识库 × 部门」的分布，供建库收尾打印与验证脚本核对"""
    by_level: Dict[str, Dict[str, int]] = {}
    by_dept: Dict[str, Dict[str, int]] = {}
    for c in chunks:
        kb = c.metadata.get('source', 'unknown')
        level = c.metadata.get('access_level', 'unknown')
        dept = c.metadata.get('department', 'unknown')
        by_level.setdefault(kb, {})[level] = by_level.setdefault(kb, {}).get(level, 0) + 1
        by_dept.setdefault(kb, {})[dept] = by_dept.setdefault(kb, {}).get(dept, 0) + 1
    return {"by_access_level": by_level, "by_department": by_dept}


if __name__ == '__main__':
    # 单独调试：处理某一个知识库目录，并打印权限分布
    target = os.path.join(config.DATA_DIR, 'regulation_data')
    chunks = process_documents(target)
    print(f"子块数量：{len(chunks)}")
    if chunks:
        print("示例元数据：", chunks[0].metadata)
        print("示例内容：", chunks[0].page_content[:200])
        import json
        print("权限分布：", json.dumps(describe_permissions(chunks),
                                    ensure_ascii=False, indent=2))
    print("\n加载器可用性：", loader_report()['active'])
