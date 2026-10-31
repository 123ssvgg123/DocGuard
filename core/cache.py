"""
    Redis 三级缓存：core/cache.py

    职责：为 RAG 的三个「重复度高但计算昂贵」环节提供缓存——
        ① Embedding 向量编码（同一 query 反复编码，BGE-M3 单次约 50~200ms）
        ② 检索结果        （同一问题反复检索，含 Milvus + Rerank，约 200~800ms）
        ③ LLM 答案        （FAQ 类问题答案高度重复，约 1~5s）

    三级降级架构（保证「有 Redis 更快，无 Redis 也能跑」）：
        L1  进程内字典（默认，零依赖，单机可用，重启失效）
        L2  Redis（启用后接入，多进程/多副本共享）
        L3  无缓存（Redis 抖动时直接回源，绝不阻断问答）

    设计要点：
        · 缓存键用「内容哈希」而非原始文本，避免超长 query 撑爆 key
        · 检索结果缓存键包含 kb_filter 与策略，避免不同库/策略互相污染
        · 提供「知识库变更后清空」接口（rebuild 索引后必须调用）
"""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from base.config import config
from base.logger import logger

#: 进程内缓存：{key: (value, expire_at)}
_LOCAL_CACHE: Dict[str, Tuple[Any, float]] = {}
#: 进程内缓存条数上限（防止无界增长）
_LOCAL_MAX_SIZE = 5000


def content_hash(text: str, length: int = 16) -> str:
    """
    计算文本的内容哈希（用于缓存键）。

    说明：直接用原文做键有两个问题——
          ① 超长 query 会让 Redis key 变得巨大
          ② 特殊字符可能破坏 key 结构
          因此统一哈希后取前 N 位（默认 16 位，碰撞概率可忽略）。
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:length]


class RAGCache:
    """RAG 三级缓存（进程内 + 可选 Redis，自动降级）"""

    def __init__(self):
        self._redis = None
        self._redis_ready = False
        self._hits = 0
        self._misses = 0
        # 分项命中统计（便于判断哪个环节缓存收益最大）
        self._stat_by_kind: Dict[str, Dict[str, int]] = {}

    # ==================== 初始化 ====================

    def init(self) -> Dict[str, Any]:
        """初始化缓存后端（幂等）"""
        if not config.REDIS_ENABLED:
            logger.info("[缓存] Redis 未启用，RAG 缓存使用进程内模式（单机）")
            return {"backend": "local", "redis": False}

        try:
            import redis  # noqa: F401
        except ImportError:
            logger.warning("[缓存] 未安装 redis 依赖，降级为进程内缓存；"
                           "如需启用请执行：pip install redis")
            return {"backend": "local", "redis": False, "reason": "missing_dependency"}

        try:
            import redis
            self._redis = redis.Redis(
                host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB,
                password=config.REDIS_PASSWORD or None, decode_responses=True,
                socket_connect_timeout=2, socket_timeout=2,
            )
            self._redis.ping()
            self._redis_ready = True
            logger.info(f"[缓存] Redis 三级缓存已启用：{config.REDIS_HOST}:"
                        f"{config.REDIS_PORT}/db{config.REDIS_DB} "
                        f"prefix={config.REDIS_KEY_PREFIX}")
            return {"backend": "redis", "redis": True}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[缓存] Redis 连接失败（{exc}），降级为进程内缓存")
            self._redis = None
            self._redis_ready = False
            return {"backend": "local", "redis": False, "reason": str(exc)}

    # ==================== 键构造 ====================

    @staticmethod
    def _full_key(key: str) -> str:
        return f"{config.REDIS_KEY_PREFIX}{key}"

    @staticmethod
    def key_embedding(text: str) -> str:
        """Embedding 缓存键（与模型名绑定，换模型后自动失效）"""
        return f"emb:{config.EMBEDDING_MODEL}:{content_hash(text)}"

    @staticmethod
    def key_retrieval(query: str, kb_filter: str = "", strategy: str = "",
                      acl_key: str = "") -> str:
        """
        检索结果缓存键（与知识库 / 策略 / **权限**绑定，避免互相污染）。

        ⚠️ 为什么必须带权限维度（需求 4.2 第 ③ 条「缓存 key 按角色/部门前缀隔离」）：
            检索结果里可能含 restricted 文档。若缓存键不含角色，quality 角色检索一次之后，
            staff 角色就会直接命中这份含密结果 —— Milvus 的过滤只在回源时生效，
            管不到缓存，权限隔离会完全失效。
            acl_key 由 ACLContext.cache_key 生成，内含角色与可见范围哈希。
        """
        kb = kb_filter or "all"
        acl = acl_key or "acl-off"
        return f"ret:{kb}:{strategy or 'auto'}:{acl}:{content_hash(query)}"

    @staticmethod
    def key_answer(query: str, kb_filter: str = "", history_hash: str = "") -> str:
        """答案缓存键（含历史哈希：多轮对话历史不同则答案不同）"""
        kb = kb_filter or "all"
        suffix = f":{history_hash}" if history_hash else ""
        return f"ans:{kb}:{content_hash(query)}{suffix}"

    # ==================== 读写 ====================

    def get(self, key: str, kind: str = "generic") -> Optional[Any]:
        """读缓存（未命中返回 None）"""
        stat = self._stat_by_kind.setdefault(kind, {"hits": 0, "misses": 0})

        # L2：Redis
        if self._redis_ready:
            try:
                raw = self._redis.get(self._full_key(key))
                if raw is not None:
                    self._hits += 1
                    stat["hits"] += 1
                    return json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[缓存] Redis 读取失败，回退进程内: {exc}")

        # L1：进程内
        item = _LOCAL_CACHE.get(key)
        if item is None:
            self._misses += 1
            stat["misses"] += 1
            return None
        value, expire_at = item
        if expire_at and expire_at < time.time():
            _LOCAL_CACHE.pop(key, None)
            self._misses += 1
            stat["misses"] += 1
            return None
        self._hits += 1
        stat["hits"] += 1
        return value

    def set(self, key: str, value: Any, ttl: int) -> bool:
        """写缓存"""
        # L2：Redis
        if self._redis_ready:
            try:
                self._redis.setex(self._full_key(key), ttl,
                                  json.dumps(value, ensure_ascii=False, default=str))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[缓存] Redis 写入失败，仅写进程内: {exc}")

        # L1：进程内（始终写，保证 Redis 抖动时本地仍可命中）
        if len(_LOCAL_CACHE) >= _LOCAL_MAX_SIZE:
            now = time.time()
            for k in [k for k, (_, exp) in _LOCAL_CACHE.items() if exp and exp < now]:
                _LOCAL_CACHE.pop(k, None)
            if len(_LOCAL_CACHE) >= _LOCAL_MAX_SIZE:
                for k in list(_LOCAL_CACHE.keys())[: _LOCAL_MAX_SIZE // 10]:
                    _LOCAL_CACHE.pop(k, None)
        _LOCAL_CACHE[key] = (value, time.time() + ttl if ttl else 0)
        return True

    def delete(self, key: str) -> bool:
        if self._redis_ready:
            try:
                self._redis.delete(self._full_key(key))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[缓存] Redis 删除失败: {exc}")
        _LOCAL_CACHE.pop(key, None)
        return True

    def clear(self, category: str = "") -> int:
        """
        按类别清空缓存。

        :param category: "" 全清 / "emb" / "ret" / "ans"
        :return: 清理条数
        """
        count = 0
        prefix = f"{category}:" if category else ""
        keys = [k for k in _LOCAL_CACHE if k.startswith(prefix)]
        for k in keys:
            _LOCAL_CACHE.pop(k, None)
            count += 1
        if self._redis_ready:
            try:
                pattern = self._full_key(f"{prefix}*")
                cursor = 0
                while True:
                    cursor, batch = self._redis.scan(cursor=cursor, match=pattern, count=200)
                    if batch:
                        self._redis.delete(*batch)
                        count += len(batch)
                    if cursor == 0:
                        break
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[缓存] Redis 按类别清空失败: {exc}")
        logger.info(f"[缓存] 已清空缓存 category={category or 'ALL'}，共 {count} 条")
        return count

    # ==================== 统计 ====================

    def stats(self) -> Dict[str, Any]:
        """缓存统计（供 /health 与前端展示）"""
        total = self._hits + self._misses
        by_kind = {}
        for kind, s in self._stat_by_kind.items():
            t = s["hits"] + s["misses"]
            by_kind[kind] = {
                "hits": s["hits"], "misses": s["misses"],
                "hit_rate": round(s["hits"] / t, 4) if t else 0.0,
            }
        return {
            "backend": "redis" if self._redis_ready else "local",
            "redis_connected": self._redis_ready,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "local_entries": len(_LOCAL_CACHE),
            "by_kind": by_kind,
            "ttl": {
                "embedding": config.REDIS_EMBEDDING_TTL,
                "retrieval": config.REDIS_RETRIEVAL_TTL,
                "answer": config.REDIS_ANSWER_TTL,
            },
        }


# ==================== 全局单例 ====================

rag_cache = RAGCache()


def cached_retrieval(func):
    """
    装饰器：缓存检索函数的返回值（List[Document]）。

    注意：Document 对象不可 JSON 序列化，因此这里缓存的是
          「可序列化的精简结构」，取用时由调用方还原。
          为降低侵入性，本装饰器只缓存 `hybrid_search_with_rerank` 的结果快照。
    """
    def _wrapper(self, query, k=5, kb_filter=None, *args, **kwargs):
        key = rag_cache.key_retrieval(query, kb_filter or "", "direct")
        hit = rag_cache.get(key, kind="retrieval")
        if hit is not None:
            logger.info(f"[缓存] 检索结果命中缓存（query hash={content_hash(query)}）")
            return _restore_docs(hit)

        docs = func(self, query, k=k, kb_filter=kb_filter, *args, **kwargs)
        try:
            rag_cache.set(key, _snapshot_docs(docs), config.REDIS_RETRIEVAL_TTL)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[缓存] 检索结果写入缓存失败: {exc}")
        return docs

    return _wrapper


def _snapshot_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    """把 Document 列表转成可序列化快照"""
    out = []
    for doc in docs or []:
        out.append({
            "page_content": getattr(doc, "page_content", ""),
            "metadata": dict(getattr(doc, "metadata", {}) or {}),
        })
    return out


def _restore_docs(snapshot: List[Dict[str, Any]]) -> List[Any]:
    """把快照还原为 Document 对象"""
    try:
        from langchain_core.documents import Document
    except ImportError:  # 兼容旧版 langchain
        try:
            from langchain.schema import Document
        except ImportError:
            # 无 langchain 时用轻量替代对象（保证降级链路可用）
            class Document:  # type: ignore
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}
    return [Document(page_content=s.get("page_content", ""),
                     metadata=s.get("metadata", {})) for s in (snapshot or [])]


if __name__ == '__main__':
    print("=" * 60)
    print("RAG 三级缓存自检")
    print("=" * 60)
    print("Redis 开关：", config.REDIS_ENABLED)
    print("初始化：", rag_cache.init())

    # 1. Embedding 缓存
    k = rag_cache.key_embedding("食品添加剂的使用标准是什么")
    rag_cache.set(k, [0.1, 0.2, 0.3], config.REDIS_EMBEDDING_TTL)
    v = rag_cache.get(k, kind="embedding")
    assert v == [0.1, 0.2, 0.3], f"Embedding 缓存读写失败：{v}"
    print("Embedding 缓存校验通过")

    # 2. 检索结果缓存（走 Document 快照还原）
    docs = _restore_docs([{"page_content": "法规原文片段", "metadata": {"rerank_score": 0.82}}])
    k2 = rag_cache.key_retrieval("食品添加剂", "regulation", "direct")
    rag_cache.set(k2, _snapshot_docs(docs), config.REDIS_RETRIEVAL_TTL)
    back = _restore_docs(rag_cache.get(k2, kind="retrieval"))
    assert back[0].page_content == "法规原文片段", "检索结果还原失败"
    assert back[0].metadata["rerank_score"] == 0.82, "metadata 未保留"
    print("检索结果缓存校验通过（含 metadata 保留）")

    # 3. 答案缓存
    k3 = rag_cache.key_answer("添加剂限量是多少", "regulation")
    rag_cache.set(k3, "根据 GB 2760...", config.REDIS_ANSWER_TTL)
    assert rag_cache.get(k3, kind="answer") == "根据 GB 2760..."
    print("答案缓存校验通过")

    # 4. 键隔离：不同知识库不应互相命中
    k_other = rag_cache.key_retrieval("食品添加剂", "sop", "direct")
    assert rag_cache.get(k_other, kind="retrieval") is None, "不同知识库的检索缓存发生了污染"
    print("知识库隔离校验通过")

    # 5. 内容哈希稳定性
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
    print("内容哈希稳定性校验通过")

    print("\n统计：", json.dumps(rag_cache.stats(), ensure_ascii=False, indent=2))
    rag_cache.clear()
    print("\n自检通过。")
