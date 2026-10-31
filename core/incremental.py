"""
    知识库增量更新：core/incremental.py

    职责：实现需求「新文档 24h 内解析入库 / 法规每周定时增量爬取」中**可离线跑**的那一半——
         通过「清单文件（manifest）+ 文件内容哈希」判断哪些文档新增 / 变更 / 删除，
         只对变化的部分重新向量化，并清掉被替换或删除文档的旧向量。

    为什么需要它（旧实现的真实问题）：
        build_kb.py 原先只有「全量重建」与「按块 ID 覆盖」两种模式。
        前者每次都要重新向量化全部文档（实测 BGE-M3 全 CPU 编码几百个块要数分钟），
        后者虽然幂等，但**如果文档被删掉或改名，它的旧向量会永远留在库里**，
        表现为「知识库里能检索到一份已经不存在的文档」——这在法规库是严重问题
        （需求 G4 的目标正是「引用现行有效版本」）。

    清单文件长这样（放在 data/.kb_manifest.json，默认不进版本库）：
        {
          "version": 1,
          "updated_at": "2026-10-14T10:30:00",
          "files": {
            "regulation_data/GB2760-2014-食品添加剂使用标准.md": {
              "hash": "9f2c...",                # 文件内容 md5 前 16 位
              "chunks": 12,                      # 该文件切出的子块数
              "ids": ["a1b2...", "c3d4..."],     # 写入 Milvus 的主键（md5(块 ID)）
              "kb_type": "regulation",
              "file_name": "GB2760-2014-...md",
              "processed_at": "2026-10-14T10:30:00"
            }
          }
        }

    为什么清单里要存 Milvus 主键：
        Milvus 的 `id` 字段存的是 md5(块 ID)（见 core/vector_store.py.add_documents），
        删除时必须用这个主键。若重建时块数变了（文档被编辑），光靠「重新生成一遍 ID」
        未必能覆盖到旧块，所以要把上一次写入的主键原样记下来，先删后写。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from base.config import config
from base.logger import logger

#: 清单文件版本（结构变更时递增，便于自动失效旧清单）
MANIFEST_VERSION = 1


# ==================================================================
# 基础工具
# ==================================================================

def file_digest(path: str, length: int = 16) -> str:
    """计算文件内容的 md5 前 length 位。基于**字节**，与编码无关，稳定可比对。"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:length]


def milvus_pk(chunk_id: str) -> str:
    """把「子块 ID」转成 Milvus 主键，算法必须与 vector_store.add_documents 一致。"""
    return hashlib.md5(str(chunk_id).encode('utf-8')).hexdigest()


def rel_key(data_dir: str, file_path: str) -> str:
    """清单里的键：相对 data/ 的路径，统一用 / 分隔，保证跨平台一致。"""
    return os.path.relpath(file_path, data_dir).replace('\\', '/')


def manifest_path() -> str:
    return os.path.join(config.DATA_DIR, '.kb_manifest.json')


# ==================================================================
# 清单读写
# ==================================================================

def load_manifest(path: Optional[str] = None) -> Dict:
    """读取清单；不存在或损坏时返回空清单（不抛异常，保证首次构建可跑）"""
    path = path or manifest_path()
    empty = {"version": MANIFEST_VERSION, "updated_at": "", "files": {}}
    if not os.path.isfile(path):
        return empty
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[增量] 清单文件损坏（{exc}），按全新构建处理：{path}")
        return empty

    if int(data.get('version', 0)) != MANIFEST_VERSION:
        logger.warning(f"[增量] 清单版本不匹配（{data.get('version')} != {MANIFEST_VERSION}），"
                       f"按全新构建处理")
        return empty
    data.setdefault('files', {})
    return data


def save_manifest(manifest: Dict, path: Optional[str] = None) -> str:
    """原子写清单：先写临时文件再 rename，避免中途崩溃留下半个 JSON"""
    path = path or manifest_path()
    manifest['version'] = MANIFEST_VERSION
    manifest['updated_at'] = datetime.now().isoformat(timespec='seconds')
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    logger.info(f"[增量] 清单已更新：{path}（{len(manifest['files'])} 个文件）")
    return path


# ==================================================================
# 变更计划
# ==================================================================

@dataclass
class ChangePlan:
    """一次增量比对的结果"""

    added: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)

    @property
    def to_process(self) -> List[str]:
        """需要重新解析与向量化的文件（新增 + 变更）"""
        return sorted(self.added + self.changed)

    def summary(self) -> Dict[str, int]:
        return {
            "added": len(self.added),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "deleted": len(self.deleted),
        }

    def describe(self) -> str:
        s = self.summary()
        return (f"新增 {s['added']} / 变更 {s['changed']} / "
                f"未变 {s['unchanged']} / 删除 {s['deleted']}")


def scan_source_files(data_dir: Optional[str] = None) -> Dict[str, str]:
    """
    扫描 data/ 下所有 *_data 目录里**受支持**的文件，返回 {相对键: 内容 md5}。

    文件枚举口径与 core/loaders.py 完全一致（同一套扩展名注册表），
    避免「比对时算进去了、加载时又跳过」的不一致。
    """
    from core.loaders import iter_supported_files

    data_dir = data_dir or config.DATA_DIR
    result: Dict[str, str] = {}
    if not os.path.isdir(data_dir):
        return result

    for dir_name in sorted(os.listdir(data_dir)):
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.isdir(dir_path) or not dir_name.endswith('_data'):
            continue
        for file_path, _ext in iter_supported_files(dir_path):
            key = rel_key(data_dir, file_path)
            try:
                result[key] = file_digest(file_path)
            except OSError as exc:
                logger.warning(f"[增量] 读取失败，跳过 {file_path}：{exc}")
    return result


def plan_incremental(data_dir: Optional[str] = None,
                     manifest: Optional[Dict] = None,
                     current: Optional[Dict[str, str]] = None) -> ChangePlan:
    """
    比对「磁盘现状」与「清单」，产出变更计划。

    判定规则（只用内容哈希，不看 mtime）：
        · 磁盘有、清单无                    → added
        · 两边都有、哈希不同                 → changed
        · 两边都有、哈希相同                 → unchanged
        · 清单有、磁盘无（文件被删或改名）    → deleted

    为什么不用 mtime：mtime 在「复制文件 / git checkout / 解压」后都会变，
        会导致明明内容没改也要重新向量化；反过来有些工具改内容不更新 mtime。
        内容哈希没有这两个问题。

    :param current: 已扫描好的 {相对键: 哈希}；不传则内部扫描一次
                    （build_kb.py 会先扫一次复用，避免重复 IO）
    """
    data_dir = data_dir or config.DATA_DIR
    manifest = manifest if manifest is not None else load_manifest()
    known: Dict[str, Dict] = manifest.get('files', {})

    if current is None:
        current = scan_source_files(data_dir)
    plan = ChangePlan()
    for key, digest in current.items():
        old = known.get(key)
        if old is None:
            plan.added.append(key)
        elif old.get('hash') != digest:
            plan.changed.append(key)
        else:
            plan.unchanged.append(key)
    plan.deleted = sorted(k for k in known if k not in current)
    return plan


def stale_ids_for(plan: ChangePlan, manifest: Dict) -> List[str]:
    """
    收集需要先从 Milvus 删除的主键：**变更文件与删除文件的旧块**。

    注意顺序：必须先删后写。若文档被编辑后块数变少，不删的话旧块会成为
    「幽灵向量」永远留在库里并被检索到。
    """
    known: Dict[str, Dict] = manifest.get('files', {})
    ids: List[str] = []
    for key in plan.changed + plan.deleted:
        ids.extend(known.get(key, {}).get('ids', []) or [])
    # 去重并保持稳定顺序
    seen = set()
    out = []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def update_manifest_entry(manifest: Dict, key: str, digest: str, chunks: List,
                          kb_type: str = "", file_name: str = "") -> None:
    """把「某个文件本次处理结果」写回清单（chunks 为切分后的子块列表）"""
    manifest.setdefault('files', {})[key] = {
        "hash": digest,
        "chunks": len(chunks),
        "ids": [milvus_pk(c.metadata.get('id', '')) for c in chunks],
        "kb_type": kb_type,
        "file_name": file_name,
        "processed_at": datetime.now().isoformat(timespec='seconds'),
    }


def drop_manifest_entries(manifest: Dict, keys: List[str]) -> None:
    """删除已消失文件在清单里的记录"""
    for key in keys:
        manifest.get('files', {}).pop(key, None)


def describe_manifest(manifest: Dict) -> Dict:
    """清单概览（供 build_kb.py 打印与 /health 展示）"""
    files = manifest.get('files', {})
    return {
        "manifest_version": manifest.get('version'),
        "updated_at": manifest.get('updated_at', ''),
        "file_count": len(files),
        "chunk_count": sum(int(v.get('chunks', 0)) for v in files.values()),
        "by_kb_type": _count_by_kb(files),
    }


def _count_by_kb(files: Dict[str, Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in files.values():
        kb = v.get('kb_type') or 'unknown'
        out[kb] = out.get(kb, 0) + 1
    return out


if __name__ == '__main__':
    import json

    print("=" * 66)
    print("知识库增量比对自检（只读，不写库）")
    print("=" * 66)
    print(f"data 目录：{config.DATA_DIR}")
    print(f"清单文件 ：{manifest_path()}")
    mf = load_manifest()
    print("当前清单：", json.dumps(describe_manifest(mf), ensure_ascii=False))

    t0 = time.time()
    p = plan_incremental(manifest=mf)
    print(f"\n比对结果（耗时 {time.time() - t0:.2f}s）：{p.describe()}")
    for label, items in (("新增", p.added), ("变更", p.changed), ("删除", p.deleted)):
        for it in items[:10]:
            print(f"  [{label}] {it}")
    print(f"\n需先删除的旧主键数量：{len(stale_ids_for(p, mf))}")
