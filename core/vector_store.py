"""
    文档向量化与存储：core/vector_store.py

    职责：
        1. 用 BGE-M3 生成稠密 + 稀疏向量
        2. 写入 Milvus（统一 Collection，通过 kb_type 字段区分 5 大知识库）
        3. 混合检索（稠密 + 稀疏，WeightedRanker 粗排）+ BGE-Reranker 精排
        4. 支持按知识库类型、法规编号等元数据过滤
"""
import hashlib
# 导入 time：count() 等待 Milvus 统计收敛时需要短暂休眠
import time
# 导入 unicodedata，用于把康熙部首等异体字归一化为标准汉字
import unicodedata

from base.config import config
from base.logger import logger
# 行级权限（需求 4.2）：过滤表达式拼装 + 命中结果的出口二次校验
from core.access_control import build_filter_expr, filter_docs_by_acl

# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序
from sentence_transformers import CrossEncoder


#: count() 等待 Milvus 统计收敛的参数。
#: 批量 upsert 之后行数统计会短暂偏小（实测写入 222 条时立即查询得到 192），
#: 这是统计延迟、不是数据丢失。不等待就会打出假的「子块被覆盖丢弃」告警。
COUNT_CONVERGE_TIMEOUT = 6.0      # 最长等待秒数
COUNT_CONVERGE_INTERVAL = 0.8     # 每次探测的间隔
COUNT_MAX_ATTEMPTS = 8            # 最多探测次数


def normalize_query(text: str) -> str:
    """
    查询文本归一化：
    1. NFKC 会把「康熙部首」(U+2F00-U+2FDF)、CJK 兼容区异体字、
       全角字符统一折叠成标准字符（如 ⾔->言, ⾛->走, Ａ->A）。
    2. 去除首尾空白。
    注意：NFKC 会影响 BM25 稀疏向量的分词结果，向量库入库时也应使用同一函数，
        否则「言」和「⾔」会分到不同 token 上，命中率下降。
    """
    if not text:
        return text
    return unicodedata.normalize("NFKC", str(text)).strip()


def _sparse_to_dict(row):
    """
    把 BGE-M3 输出的稀疏向量行转成 {index: value} 字典。

    注意：不同版本的 milvus-model 返回不同的稀疏矩阵类型，取列索引的属性名不同，
        必须三者兼容，否则会报 'coo_array' object has no attribute 'indices'：
        - scipy.sparse.coo_array / coo_matrix -> .col
        - scipy.sparse.csr_matrix            -> .indices
    """
    if hasattr(row, 'col'):          # coo_array / coo_matrix
        indices = row.col
    elif hasattr(row, 'indices'):    # csr_matrix
        indices = row.indices
    else:                            # 兜底：尝试任意可取到索引的常见属性
        indices = getattr(row, 'col', getattr(row, 'indices', None))
        if indices is None:
            raise TypeError(f"不支持的稀疏向量类型: {type(row)}")
    return {int(idx): float(value) for idx, value in zip(indices, row.data)}


class VectorStore:
    """封装向量存储与检索功能"""

    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=config.MILVUS_COLLECTION_NAME,
                 host=config.MILVUS_HOST,
                 port=config.MILVUS_PORT,
                 database=config.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database

        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        # 带 OOM 降级：显式配了 cuda 但显存不足时，自动退回 CPU 重载，
        # 避免整个服务因为「想用 GPU」而启动失败。
        self.reranker = None
        self._load_reranker(config.MODEL_DEVICE)

        # 初始化 BGE-M3 嵌入函数，关闭 FP16（FP32 更稳定、兼容性更好）
        # ⚠️ 参数名必须与 milvus_model.hybrid.bge_m3.BGEM3EmbeddingFunction 的签名严格一致：
        #      model_name（不是 model_name_or_path）
        #      use_fp16 （不是 use_f16）
        #    该类签名是 (model_name, batch_size, device, normalize_embeddings, use_fp16, ...)，
        #    末尾接受 **kwargs 并原样透传给 BGEM3FlagModel。写错名字不会报错，
        #    而是被 kwargs 吞掉 → 触发两个严重后果：
        #      1) model_name 回落到默认值 "BAAI/bge-m3"，忽略本地模型路径，
        #         离线环境下会尝试联网下载，或在下载/加载中途崩溃（实测 Segmentation fault）；
        #      2) 未知键 use_f16 被塞进 BGEM3FlagModel 构造参数，行为不可预期。
        self.embedding_function = self._load_embedding(config.MODEL_DEVICE)
        # 获取稠密向量的维度（BGE-M3 为 1024），用于建表
        dense_dim = self.embedding_function.dim["dense"]

        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f'http://{self.host}:{self.port}', db_name=self.database)
        # 创建或加载 Milvus 集合
        self._create_or_load_collection(dense_dim)

    # ============ 模型加载（含 OOM 自动降级） ============

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        """判断异常是否为 CUDA 显存不足（torch.OutOfMemoryError 或提示串）"""
        text = f"{type(exc).__name__}: {exc}".lower()
        return "outofmemory" in text or "out of memory" in text or "cuda error" in text

    def _load_reranker(self, device: str) -> None:
        """加载 Reranker；CUDA 显存不足时自动退回 CPU"""
        logger.info(f"加载 Reranker 模型：{config.RERANK_MODEL_PATH} (device={device})")
        try:
            self.reranker = CrossEncoder(config.RERANK_MODEL_PATH, device=device)
            return
        except Exception as exc:  # noqa: BLE001
            if device != "cpu" and self._is_oom(exc):
                logger.warning(f"[设备降级] Reranker 在 {device} 上显存不足，自动改为 CPU 重载：{exc}")
                self.reranker = CrossEncoder(config.RERANK_MODEL_PATH, device="cpu")
                return
            raise

    def _load_embedding(self, device: str):
        """加载 BGE-M3 嵌入函数；CUDA 显存不足时自动退回 CPU"""
        logger.info(f"加载 Embedding 模型：{config.EMBEDDING_MODEL_PATH} (device={device})")
        try:
            return BGEM3EmbeddingFunction(
                model_name=config.EMBEDDING_MODEL_PATH,
                use_fp16=False,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            if device != "cpu" and self._is_oom(exc):
                logger.warning(f"[设备降级] Embedding 在 {device} 上显存不足，自动改为 CPU 重载：{exc}")
                return BGEM3EmbeddingFunction(
                    model_name=config.EMBEDDING_MODEL_PATH,
                    use_fp16=False,
                    device="cpu",
                )
            raise

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self, dense_dim):
        """构造阶段调用：集合不存在则建，存在则加载到内存"""
        self._ensure_collection(dense_dim)
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    def _ensure_collection(self, dense_dim):
        """
        确保集合存在（不存在则按 schema 创建），但不负责 load。

        为什么要从 _create_or_load_collection 里拆出来：
            --rebuild 流程会先 drop_collection 再写数据，此时集合是**不存在**的。
            如果只有构造阶段才建表，删除后就再也没有机会重建，写库会直接失败。
            把它抽成独立方法后，add_documents 也能在写前自愈（集合被外部删掉也能恢复）。
        """
        if self.client.has_collection(self.collection_name):
            logger.info(f"已加载集合 {self.collection_name}")
            return

        # 创建集合 Schema，禁用自动 ID，启用动态字段（便于扩展元数据）
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        # 主键：子块 ID
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        # 子块文本
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        # 稠密向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
        # 稀疏向量
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 父块 ID
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
        # 父块内容（命中后作为上下文送入提示词）
        schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
        # 知识库类型：regulation / sop / case / nutrition / supplier / faq
        schema.add_field(field_name="kb_type", datatype=DataType.VARCHAR, max_length=50)
        # 来源文件名
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
        # 入库时间
        schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
        # ---- 行级权限三字段（需求 4.2）----
        # ⚠️ schema 是建集合时固定的：老集合没有这三个字段，检索时带权限的过滤表达式会直接
        #    报错。升级到本版本后必须执行一次 python build_kb.py --rebuild。
        # 密级：public / internal / restricted（由 core/access_control.py 按知识库默认表或
        #       文件声明块给出）
        schema.add_field(field_name="access_level", datatype=DataType.VARCHAR, max_length=20)
        # 归属部门：all / quality / kitchen / purchasing / management
        schema.add_field(field_name="department", datatype=DataType.VARCHAR, max_length=50)
        # 角色白名单（逗号分隔）：非空时优先于密级规则，用于 FAQ 等按需精细授权的文档
        schema.add_field(field_name="visible_to", datatype=DataType.VARCHAR, max_length=255)

        # 创建索引参数对象
        index_params = self.client.prepare_index_params()
        # 稠密向量索引：IVF_FLAT，内积
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128}
        )
        # 稀疏向量索引：稀疏倒排索引，内积
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            # 丢弃权值最小的 20% 的维度（只保留权值较大的 80%）
            params={"drop_ratio_build": 0.2}
        )
        # 创建集合
        self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                      index_params=index_params)
        logger.info(f"已创建集合 {self.collection_name}")

    # 定义方法，存储文档到向量数据库
    def add_documents(self, documents, batch_size=64):
        """批量向量化并写入 Milvus（分批以控制内存占用）"""
        if not documents:
            logger.error("没有加载到数据")
            return 0

        # 写前自愈：集合可能被 --rebuild 删除，也可能被外部清理掉
        self._ensure_collection(self.embedding_function.dim["dense"])
        self.client.load_collection(self.collection_name)

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            texts = [doc.page_content for doc in batch]
            # BGE-M3 生成嵌入【向量化】
            embeddings = self.embedding_function(texts)

            data = []
            for i, doc in enumerate(batch):
                data.append({
                    # 主键用子块 ID 的 md5，保证同 ID 重复入库时幂等覆盖
                    "id": hashlib.md5(doc.metadata['id'].encode('utf-8')).hexdigest(),
                    "text": doc.page_content,
                    "dense_vector": embeddings["dense"][i],
                    "sparse_vector": _sparse_to_dict(embeddings["sparse"][i]),
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "parent_content": doc.metadata.get("parent_content", doc.page_content),
                    "kb_type": doc.metadata.get("source", ""),
                    "file_name": doc.metadata.get("file_name", ""),
                    "timestamp": doc.metadata.get("timestamp", ""),
                    # 行级权限三字段（需求 4.2）。缺失时按最保守的 restricted 处理：
                    # 宁可让文档检索不到，也不要因为字段缺失把受限内容放开。
                    "access_level": doc.metadata.get("access_level", "restricted"),
                    "department": doc.metadata.get("department", "all"),
                    "visible_to": doc.metadata.get("visible_to", "")
                })

            if data:
                self.client.upsert(collection_name=self.collection_name, data=data)
                total += len(data)
                logger.info(f"存储 {len(data)} 个子块到向量数据库（累计 {total}/{len(documents)}）")
        return total

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, kb_filter=None, acl=None):
        """
        混合检索 + 重排：
            1. BGE-M3 生成 query 的稠密/稀疏向量
            2. Milvus 稠密 + 稀疏双路召回，WeightedRanker 粗排
            3. 子块命中去重出父块
            4. BGE-Reranker 对父块精排，返回 Top-M 父块
        :param kb_filter: 知识库类型过滤（str 或 list），None 表示全库检索
        :param acl: 调用方权限上下文（core.access_control.ACLContext）；
                    None 或 enabled=False 时不做权限过滤，行为与改造前一致

        权限过滤分两处（纵深防御，见 core/access_control.py 头部说明）：
            · 召回阶段：拼进 Milvus 标量表达式（access_level / department），决定「能否被检索到」
            · 出口阶段：对命中的父块再按 visible_to 白名单校验一次
        """
        # 归一化查询文本：修复康熙部首/兼容异体字（如 ⾔->言, ⾛->走）
        query = normalize_query(query)
        logger.info(f"hybrid_search_with_rerank 执行混合检索并重排序...query:{query}, k:{k}, "
                    f"kb_filter:{kb_filter}, acl:{acl.role if acl else 'off'}")

        # BGE-M3 生成 query 向量
        query_embeddings = self.embedding_function([query])
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = _sparse_to_dict(query_embeddings["sparse"][0])

        # 构建标量过滤表达式：知识库类型 + 行级权限（多个条件用 and 连接）
        filter_expr = build_filter_expr(kb_filter, acl)
        if filter_expr:
            logger.info(f"标量过滤表达式：{filter_expr}")

        # 稠密向量召回请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            # ⚠️ nprobe 与 nlist 的配合关系（召回率 vs 延迟）：
            #   索引用 IVF_FLAT + nlist=128（把向量聚成 128 个簇）。
            #   检索时 nprobe 决定实际扫描几个簇：
            #     nprobe=10  → 只扫 10/128 ≈ 8%，快但会漏（簇边界附近的正确答案召不回）
            #     nprobe=32  → 扫 32/128 = 25%，召回率明显提升，延迟增加可忽略（数据量小时几乎无感）
            #     nprobe=128 → 全扫描，等价暴力检索，最准但最慢
            #   经验值：nprobe ≈ nlist 的 1/4 ~ 1/3。本机 222 条数据下 32 是稳妥选择。
            param={"metric_type": "IP", "params": {"nprobe": config.NPROBE}},
            limit=k,
            expr=filter_expr
        )
        # 稀疏向量召回请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )
        # 指定稠密与稀疏向量的权重比
        ranker = WeightedRanker(1.0, 0.7)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,  # 粗排
            limit=k,
            output_fields=["id", "text", "parent_id", "parent_content", "kb_type", "file_name",
                           "timestamp", "access_level", "department", "visible_to"],
        )[0]

        # =========== 以上是混合检索 ==============
        if not results or len(results) == 0:
            logger.info("混合检索未召回任何结果")
            return []

        # 将命中结果转为 Document（子块）
        child_chunks = [self._doc_from_hit(hit['entity']) for hit in results]
        logger.info(f"去重前子块数量: {len(child_chunks)}")
        # 对父块去重（同一父块的多个子块命中只保留一次）
        parent_docs = self._get_unique_parent_docs(child_chunks)
        logger.info(f"去重后父块数量: {len(parent_docs)}")

        # 出口二次校验：Milvus 表达式管「密级 + 部门」，这里管「按文档的角色白名单」。
        # 之所以不能只靠表达式：visible_to 是「OR 一串角色」的语义，塞进标量表达式会很啰嗦；
        # 而且 L2 缓存里可能残留换权限之前写入的结果，出口必须再兜一道。
        if acl is not None and acl.enabled:
            before = len(parent_docs)
            parent_docs = filter_docs_by_acl(parent_docs, acl)
            if len(parent_docs) != before:
                logger.info(f"[权限] 出口校验剔除 {before - len(parent_docs)} 个父块")
        if not parent_docs:
            logger.info("权限过滤后无可用的父块，返回空结果")
            return []

        # 精排打分：给「命中的子块」打分，再按 parent_id 聚合出父块得分。
        #
        # 为什么不直接给父块全文打分（改造前的做法）：
        #   父块是 250~500 字的整段条款，里面往往只有一两句是答案，其余是相邻条款。
        #   拿全文喂 CrossEncoder 会被无关内容稀释相关性——实测 20 条评估集上，
        #   同一批召回结果按父块打分句级召回 57.2%，按子块打分 69.0%（+11.7pp）；
        #   更典型的是「进货查验记录保存多久」，父块里明明有「保质期满后六个月」，
        #   整段打分只有 0.0092（低于兜底线 0.3 直接误拒答），子块打分 0.912。
        #
        # 聚合规则：同一父块可能命中多个子块，取其中最高分作为该父块的得分。
        by_id = {}
        for chunk in child_chunks:
            by_id.setdefault(chunk.metadata.get("id") or chunk.page_content, chunk)
        unique_children = list(by_id.values())
        pairs = [[query, doc.page_content] for doc in unique_children]
        scores = self.reranker.predict(pairs) if pairs else []
        logger.info(f"子块打分完成，子块数量: {len(unique_children)}")

        best_of_parent = {}
        for doc, score in zip(unique_children, scores):
            pid = doc.metadata.get("parent_id") or doc.metadata.get("id")
            if pid not in best_of_parent or float(score) > best_of_parent[pid][1]:
                best_of_parent[pid] = (doc, float(score))

        # 还原成父块 Document（生成答案时用的是父块全文），并只保留通过权限校验的父块
        allowed = {d.metadata.get("parent_id"): d for d in parent_docs}
        ranked = []
        for pid, (_child, score) in sorted(best_of_parent.items(), key=lambda kv: -kv[1][1]):
            parent_doc = allowed.get(pid)
            if parent_doc is None:      # 被权限校验剔除，或父块内容缺失
                continue
            ranked.append((parent_doc, score))
        logger.info(f"重排完成（子块打分 → 父块聚合），父块数量: {len(ranked)}")

        # 取前 M 个，并把 rerank 分数写入元数据（供上层做阈值判断/拒答）
        top_docs = []
        for doc, score in ranked[:config.CANDIDATE_M]:
            doc.metadata["rerank_score"] = float(score)
            top_docs.append(doc)
        return top_docs

    # 定义私有方法，从 Milvus 查询结果转换成 Document 对象
    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit["text"],  # 子块内容
            metadata={
                "id": hit["id"],
                "parent_id": hit["parent_id"],
                "parent_content": hit["parent_content"],  # 父块内容，用于拼接提示词
                "source": hit.get("kb_type", ""),
                "file_name": hit.get("file_name", ""),
                "timestamp": hit.get("timestamp", ""),
                # 权限字段要一路带到父块，出口校验（visible_to）与来源展示都要用
                "access_level": hit.get("access_level", ""),
                "department": hit.get("department", ""),
                "visible_to": hit.get("visible_to", "")
            }
        )

    # 定义私有方法，从命中的子块中提取去重的父文档
    def _get_unique_parent_docs(self, child_chunks):
        parent_contents = set()
        parent_docs = []
        for chunk in child_chunks:
            content = chunk.metadata.get('parent_content', chunk.page_content)
            if content in parent_contents:
                continue
            parent_contents.add(content)
            # 父块 Document 保留来源信息，便于回答时标注引用来源
            parent_docs.append(Document(
                page_content=content,
                metadata={
                    "source": chunk.metadata.get("source", ""),
                    "file_name": chunk.metadata.get("file_name", ""),
                    "parent_id": chunk.metadata.get("parent_id", ""),
                    # 权限字段随父块保留，供出口校验
                    "access_level": chunk.metadata.get("access_level", ""),
                    "department": chunk.metadata.get("department", ""),
                    "visible_to": chunk.metadata.get("visible_to", ""),
                }
            ))
        return parent_docs

    def delete_by_ids(self, ids):
        """
        按主键批量删除向量（增量更新时清理「已变更 / 已删除」文档的旧块）。

        ⚠️ 为什么必须支持删除：Milvus 写入是 upsert（同主键覆盖）。
           若文档被编辑后块数变少，或文件被直接删掉，多出来的旧块不会被任何写操作覆盖，
           会永远留在库里被检索到——表现为「能检索到一份已经不存在的文档」，
           在法规库属于严重问题（需求 G4 的目标正是「引用现行有效版本」）。
           增量流程因此严格遵循「先按旧主键删除、再写入新块」。

        :param ids: Milvus 主键列表（即 md5(子块 ID)，见 add_documents）
        :return: 实际提交删除的主键数量
        """
        if not ids:
            return 0
        if not self.client.has_collection(self.collection_name):
            logger.warning(f"集合 {self.collection_name} 不存在，跳过删除")
            return 0
        ids = list(ids)
        deleted = 0
        # 单次 delete 的 id 列表不宜过大，按 1000 分批，失败不影响其它批次
        for start in range(0, len(ids), 1000):
            batch = ids[start:start + 1000]
            try:
                self.client.delete(collection_name=self.collection_name, ids=batch)
                deleted += len(batch)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"删除向量失败（批次 {start}~{start + len(batch)}）：{exc}")
        logger.info(f"已删除 {deleted} 个向量（增量更新清理旧块）")
        return deleted

    def delete_collection(self):
        """删除集合（重建知识库时使用）"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"已删除集合 {self.collection_name}")

    def count(self, expected: int = 0):
        """
        返回集合内向量数量（集合不存在时返回 0，不抛异常）。

        :param expected: 本次刚写入的条数。传入后会等待统计收敛——
            Milvus 在批量 upsert 之后行数统计会**短暂偏小**（实测重建后立即查询得到 192，
            而实际已写入 222），不等收敛就会打印假的「子块被覆盖丢弃」告警。
            传 0（默认）表示不等待，适合健康检查这类对响应时间更敏感的场景。

        ⚠️ 为什么要用 query(count(*)) 而不是 get_collection_stats：
            刚 create_collection + 批量 upsert 之后，Milvus 的段（segment）还在 flush 中，
            get_collection_stats().row_count 会**短暂返回 0**（实测重建后立即查询为 0，
            等若干秒才变正确）。这在建库脚本的收尾统计里会造成"写入 222 但显示 0"的
            假报警，也会让自检逻辑误判。query 走的是查询路径，读到的是当前真实可见行数。
            但 query 也有同样的瞬时延迟，所以这里做了「重试到非 0」的兜底。

        ⚠️ 另一个坑：output_fields=["count(*)"] 时**不能**同时传 limit，
            Milvus 会直接报 code=1100 "count entities with pagination is not allowed"。
            不传 limit 会返回单行 {"count(*)": N}，这才是正确的聚合查询写法。
        """
        if not self.client.has_collection(self.collection_name):
            return 0

        # 优先用查询聚合，拿到的是立即可见的真实行数。
        # 多次采样取最大值：统计只会「由小变大」地收敛，取最大最接近真值。
        best = 0
        deadline = time.time() + (COUNT_CONVERGE_TIMEOUT if expected else 0.0)
        attempt = 0
        while True:
            attempt += 1
            try:
                rows = self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=["count(*)"],
                )
                if rows:
                    value = rows[0].get("count(*)")
                    if value:
                        best = max(best, int(value))
            except Exception as e:
                logger.warning(f"聚合查询统计失败：{e}")
                break

            # 已经拿到期望值（或调用方不要求等待）就收工
            if best and (not expected or best >= expected):
                return best
            if time.time() >= deadline or attempt >= COUNT_MAX_ATTEMPTS:
                break
            time.sleep(COUNT_CONVERGE_INTERVAL)

        if best:
            logger.info(f"统计未达到本次写入条数（{best}/{expected}），"
                        f"已按最大采样值返回；若持续偏小请检查块 ID 是否重复")
            return best

        # 兜底：集合统计（可能滞后，仅在查询拿不到时使用）
        try:
            return self.client.get_collection_stats(self.collection_name).get('row_count', 0)
        except Exception as e:
            logger.error(f"统计集合数量失败: {e}")
            return 0


if __name__ == '__main__':
    vector_store = VectorStore()
    # 供调试：先建库再检索
    # from core.document_processor import process_all_knowledge_bases
    # vector_store.add_documents(process_all_knowledge_bases())
    print("集合内向量数量：", vector_store.count())
    docs = vector_store.hybrid_search_with_rerank(query="冷藏食品到货温度标准是多少", k=5, kb_filter='regulation')
    for d in docs:
        print(round(d.metadata.get('rerank_score', 0), 4), d.page_content[:80])
