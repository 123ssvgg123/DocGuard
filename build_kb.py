"""
    知识库构建入口：build_kb.py

    用于把 data/ 下的原始文档导入向量库，完成「加载 → 解析权限 → 切分 → 向量化 → 入库」的离线流程。

    用法：
        python build_kb.py                      # 全量导入（按块 ID 覆盖同 ID 子块），并刷新增量清单
        python build_kb.py --incremental        # 增量导入：只处理新增/变更的文件，并删除已移除文件的旧向量
        python build_kb.py --incremental --dry-run   # 只看变更计划，不动库
        python build_kb.py --rebuild            # 先删除集合再全量重建（切分策略或权限规则变更后用）
        python build_kb.py --dir regulation_data     # 只导入指定知识库目录

    三种模式的关系与选择：
        ┌── 模式 ────────── 适用场景 ──────────────────────────────────────────────┐
        │ 全量（默认）       首次建库；或不确定改了什么，宁可全跑一遍                  │
        │ --incremental     日常更新（需求：新文档 24h 内入库 / 法规每周增量）        │
        │ --rebuild         改过切分策略、块 ID 规则、或权限元数据字段后，必须用它     │
        └─────────────────────────────────────────────────────────────────────────┘

    ⚠️ 关于 --incremental 与 --rebuild 的分工（实测结论）：
        增量只能识别「文件内容变了」，识别不出「切分规则变了」。
        如果你改了 core/text_splitter.py 的分隔符/块大小，旧块会残留成冗余向量，
        必须走 --rebuild 全量重建。

    ⚠️ 首次升级到「带权限元数据」的版本后，也必须执行一次 --rebuild：
        Milvus 的 schema 是建集合时固定的，新增 access_level / department / visible_to
        三个标量字段后，老集合里没有这些字段，检索时的权限过滤表达式会直接报错。
"""
import argparse
import os
import sys

from base.config import config
from base.logger import logger


def _print_plan(plan, manifest, hashes):
    """打印增量变更计划"""
    print("-" * 60)
    print(f"增量比对结果：{plan.describe()}")
    for label, items in (("新增", plan.added), ("变更", plan.changed),
                         ("未变", plan.unchanged), ("删除", plan.deleted)):
        for key in items[:15]:
            extra = ""
            if label == "未变":
                extra = f"  hash={hashes.get(key, '')[:8]}"
            print(f"  [{label}] {key}{extra}")
        if len(items) > 15:
            print(f"  [{label}] …… 其余 {len(items) - 15} 项省略")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='食品安全知识库构建工具')
    parser.add_argument('--rebuild', action='store_true', help='删除已有集合后全量重建')
    parser.add_argument('--incremental', action='store_true',
                        help='增量更新：只处理新增/变更的文件，并删除已移除文件的旧向量')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印变更计划与将删除的向量数，不写库（需配合 --incremental）')
    parser.add_argument('--dir', type=str, default=None, help='只导入指定知识库目录（如 regulation_data）')
    args = parser.parse_args()

    if args.rebuild and args.incremental:
        print("--rebuild 与 --incremental 互斥：前者清库重建，后者在现有库上增量维护。")
        sys.exit(2)

    # 延迟导入：避免仅查看帮助时也加载重型模型
    from core.document_processor import process_files, assert_unique_ids, describe_permissions
    from core.incremental import (load_manifest, save_manifest, plan_incremental,
                                  scan_source_files, stale_ids_for, update_manifest_entry,
                                  drop_manifest_entries, describe_manifest)
    from core.vector_store import VectorStore
    from core.cache import rag_cache
    import json

    data_dir = config.DATA_DIR

    # ---------- 1. 扫描磁盘 + 读清单 + 算变更计划 ----------
    hashes = scan_source_files(data_dir)
    if not hashes:
        logger.error(f"{data_dir} 下没有扫描到任何受支持的文件（支持的类型见 core/loaders.py）")
        sys.exit(1)

    # 清单：--rebuild 时视作全新（重建后旧记录全部失效）
    manifest = {"version": 1, "updated_at": "", "files": {}} if args.rebuild else load_manifest()
    plan = plan_incremental(data_dir=data_dir, manifest=manifest, current=hashes)

    if args.dir:
        prefix = args.dir.rstrip('/') + '/'
        keep = [k for k in hashes if k.startswith(prefix)]
        if not keep:
            logger.error(f"目录不存在或其中没有受支持的文件：{os.path.join(data_dir, args.dir)}")
            sys.exit(1)
        # 只导入目标目录：把其余文件全部标记为「未变」，不参与本次写入
        only = set(keep)
        plan.added = [k for k in plan.added if k in only] or []
        plan.changed = [k for k in plan.changed if k in only] or []
        if not plan.added and not plan.changed:
            plan.added = sorted(only)      # 单目录模式默认强制覆盖该目录
        plan.deleted = []

    print("\n" + "=" * 60)
    print("知识库构建")
    print(f"  data 目录 ：{data_dir}")
    print(f"  模式      ：{'全量重建(rebuild)' if args.rebuild else ('增量(incremental)' if args.incremental else '全量导入')}")
    if args.dir:
        print(f"  限定目录  ：{args.dir}")
    print("=" * 60)

    if args.incremental or args.dry_run:
        _print_plan(plan, manifest, hashes)

    # ---------- 2. 决定本次要处理的文件（纯计算，不需要向量库） ----------
    if args.incremental:
        keys = plan.to_process
    elif args.dir:
        keys = sorted(set(plan.added + plan.changed))
    else:
        keys = sorted(hashes)

    # ---------- 3. dry-run 提前返回 ----------
    # 顺序很重要：dry-run 只是「看一眼计划」，不该加载 BGE-M3 / Reranker（CPU 上约 7 秒），
    # 更不该去连 Milvus。所以这里在 VectorStore() 之前就返回。
    if args.dry_run:
        stale = stale_ids_for(plan, manifest)
        print(f"\n[dry-run] 将处理 {len(keys)} 个文件，"
              f"先删除 {len(stale)} 个旧向量，然后写入新块。未做任何修改。")
        return

    # ---------- 4. 初始化向量库 ----------
    vector_store = VectorStore()
    if args.rebuild:
        logger.info("执行全量重建：删除已有集合")
        # 只删集合、不重新 new VectorStore()：
        # 模型已加载在内存里，删掉集合后 add_documents 时会按同样的 schema 自动重建同名集合，
        # 重复构造会白白多花一次加载模型的时间（实测约 7 秒）。
        # 注意：删除后集合不存在，此阶段**不能**再调用 count()/任何查询。
        vector_store.delete_collection()

    if not keys:
        print("\n没有需要处理的文件（所有文档均未变更）。")
        if args.incremental:
            # 删除项仍需落地
            stale = stale_ids_for(plan, manifest)
            if stale:
                print(f"但有 {len(stale)} 个旧向量需要删除。")
            else:
                print("清单与磁盘一致，无需任何操作。")
        return

    # ---------- 4. 先删除「变更/删除」文件的旧向量（必须早于写入） ----------
    if args.incremental:
        stale = stale_ids_for(plan, manifest)
        if stale:
            removed = vector_store.delete_by_ids(stale)
            logger.info(f"[增量] 已删除失效向量 {removed} 条（覆盖 {len(plan.changed)} 个变更文件"
                        f"与 {len(plan.deleted)} 个已移除文件）")

    # ---------- 5. 加载 + 切分 ----------
    chunks_by_file = process_files(keys, data_dir=data_dir)
    if not chunks_by_file:
        logger.error("没有任何文件成功解析，请检查 data/ 目录与加载器依赖")
        sys.exit(1)

    processed_keys = sorted(chunks_by_file.keys())
    chunks = [c for key in processed_keys for c in chunks_by_file[key]]
    # 子块 ID 全局唯一性校验（撞 ID 会被 upsert 静默覆盖，必须建库阶段就失败）
    assert_unique_ids(chunks)

    # ---------- 6. 向量化并入库 ----------
    count = vector_store.add_documents(chunks)

    # ---------- 7. 更新清单 ----------
    for key in processed_keys:
        kb_type = chunks_by_file[key][0].metadata.get('source', '') if chunks_by_file[key] else ''
        file_name = os.path.basename(key)
        update_manifest_entry(manifest, key, hashes.get(key, ''), chunks_by_file[key],
                              kb_type=kb_type, file_name=file_name)
    if args.incremental:
        drop_manifest_entries(manifest, plan.deleted)
    manifest_path = save_manifest(manifest)

    # ---------- 8. 汇总 + 自检 ----------
    # 传入本次写入条数，让 count() 等待统计收敛：Milvus 批量 upsert 后行数统计会短暂偏小，
    # 不等待就会打印假的「子块被覆盖丢弃」告警（实测写入 222 时立即查询得到 192）。
    total = vector_store.count(expected=count)

    # 失效检索缓存：改了知识库内容后，旧的检索/答案缓存会指向已变更的文档，
    # 必须清掉，否则下次问答会命中过期结果（Embedding 缓存保留，向量模型未变）。
    cleared = rag_cache.clear("ret") + rag_cache.clear("ans")
    if cleared:
        logger.info(f"知识库变更，已清理检索/答案缓存 {cleared} 条")

    print("\n" + "=" * 60)
    print(f"知识库构建完成：本次处理 {len(processed_keys)} 个文件，写入 {count} 个子块")
    print(f"集合内向量总数：{total}")
    if total < count:
        # 正常情况下 total >= count（增量模式含历史块）。
        # total < count 说明有子块被覆盖丢掉了，最典型的原因是块 ID 重复。
        print("-" * 60)
        print(f"⚠️ 警告：集合内向量数（{total}）少于本次写入数（{count}），存在子块被覆盖丢弃。")
        print("   最常见原因：子块 ID 重复（写库为 upsert，同 ID 会互相覆盖）。")
        print("   请检查 core/document_processor.py 的 _doc_key() 前缀构造。")

    # 权限分布自检：一眼看出行级隔离有没有生效（需求 4.2）
    if config.ACCESS_ENABLED:
        print("-" * 60)
        print("权限分布（知识库 → 密级）：")
        dist = describe_permissions(chunks)['by_access_level']
        for kb, levels in sorted(dist.items()):
            detail = "，".join(f"{k}={v}" for k, v in sorted(levels.items()))
            print(f"  {kb:12s} {detail}")

    print("-" * 60)
    print(f"增量清单：{manifest_path}")
    print("  内容：" + json.dumps(describe_manifest(manifest), ensure_ascii=False))
    if not args.rebuild and not args.incremental:
        print("提示：本次是全量导入（按块 ID 覆盖同 ID 的子块）。")
        print("      日常更新建议改用 python build_kb.py --incremental（只处理变化的文件）。")
        print("      若你刚改过切分策略（core/text_splitter.py）或块 ID 生成规则，")
        print("      旧块会残留成为冗余向量，必须改用 python build_kb.py --rebuild 全量重建。")
    if args.rebuild:
        print("提示：--rebuild 已删除集合，本次写入的块构成全新向量库；增量清单也已重置。")
    print("=" * 60)


if __name__ == '__main__':
    main()
