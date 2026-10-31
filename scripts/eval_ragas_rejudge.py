#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_ragas_rejudge.py —— 离线重算 RAGAS 四项（只换裁判，复用已有的检索结果与答案）

为什么要单独一个脚本
--------------------
scripts/evaluate_ragas.py 走的是在线链路（RAGSystem → Milvus → 本地 BGE-M3/Reranker），
换一个裁判模型就要把 Milvus 和两个本地大模型全拉起来。而这次的需求只是「换裁判重打分」，
上下文和答案都是当天真实批次跑出来的，所以离线复算即可，且口径更干净：

  检索上下文  data/eval/results/retrieval_allkb_k5_m3.json   （线上配置 k=5 / m=3，20 条）
  系统答案    data/eval/results/eval_judge_judge.json          （同日线上批次，20 条）
  参考答案    取上面 retrieval 文件里的 ground_truth 字段
  裁判模型    由 --judge 指定

两个口径都会输出
----------------
  full     : 全量 20 条（含 2 条被阈值拒答的）
  answered : 仅 18 条实际作答的
拒答条目的 answer 是「知识库中暂无相关依据」，faithfulness / answer_relevancy 天然很低，
把它和正常作答混在一起看会低估生成质量，所以两个数都要有。

用法
----
  python scripts/eval_ragas_rejudge.py                       # 默认 qwen-max
  python scripts/eval_ragas_rejudge.py --judge qwen-plus --limit 5   # 先小样本试
  python scripts/eval_ragas_rejudge.py --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

RESULTS = os.path.join(BASE, "data", "eval", "results")
RETRIEVAL_FILE = os.path.join(RESULTS, "retrieval_allkb_k5_m3.json")
ANSWER_FILE = os.path.join(RESULTS, "eval_judge_judge.json")
OUT_DIR = RESULTS

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 运行策略：远程模型最小验证（见 base/run_policy.py）
from base.run_policy import RunPolicyError, run_policy  # noqa: E402


def load_env_file() -> dict:
    """读项目 .env，取不到就返回空字典（不抛错，方便 CI）"""
    path = os.path.join(BASE, ".env")
    env = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def resolve_api_key(which: str, env_file: dict) -> str:
    """裁判/embedding 的 key：环境变量优先，其次 .env 的 LLM_API_KEY"""
    if which == "dashscope":
        return os.environ.get("DASHSCOPE_API_KEY", "").strip()
    return (os.environ.get("JUDGE_API_KEY", "")
            or env_file.get("LLM_API_KEY", "")).strip()


def build_samples(limit: int = 0):
    """把两份结果文件按问题对齐成 RAGAS 需要的四元组"""
    with open(RETRIEVAL_FILE, encoding="utf-8") as f:
        retrieval = json.load(f)["per_sample"]
    with open(ANSWER_FILE, encoding="utf-8") as f:
        answers = {x["question"]: x for x in json.load(f)["per_sample"]}

    samples = []
    for r in retrieval:
        q = r["question"]
        a = answers.get(q, {})
        samples.append({
            "question": q,
            "answer": (a.get("answer") or "").strip(),
            "contexts": [c for c in (r.get("contexts") or []) if c and c.strip()],
            "ground_truth": (r.get("ground_truth") or "").strip(),
            "refused_by_threshold": bool(r.get("refused_by_threshold")),
            "top_score": r.get("top_score"),
        })
    if limit:
        samples = samples[:limit]
    return samples


def aggregate(rows: list, keys: list) -> dict:
    """对 per-sample 结果按指标求均值，跳过 None/NaN"""
    out = {"n": len(rows)}
    for k in keys:
        vals = [r.get(k) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="离线重算 RAGAS 四项（换裁判）")
    ap.add_argument("--judge", default="qwen-max", help="裁判模型（OpenAI 兼容）")
    ap.add_argument("--embed", default="text-embedding-v3", help="embedding 模型（answer_relevancy 需要）")
    ap.add_argument("--base-url", default=DASHSCOPE_BASE, help="OpenAI 兼容 base_url")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（默认按运行策略取最小验证条数）")
    ap.add_argument("--workers", type=int, default=2, help="并发数，限流时调小")
    ap.add_argument("--timeout", type=int, default=180, help="单次调用超时秒数")
    ap.add_argument("--allow-full", action="store_true",
                    help="允许全量 20 条（需同时设置 ALLOW_FULL_RUN=1；默认仅最小验证）")
    args = ap.parse_args()

    # ================================================================
    # 运行策略：裁判模型同样是远程 API（qwen-max / deepseek-chat），
    # RAGAS 四项指标每条样本要发起多次请求，全量 20 条很容易把额度跑空。
    # 因此默认只跑 SMOKE_MAX_SAMPLES 条做冒烟验证，全量必须双闸门同时打开。
    # ================================================================
    try:
        run_policy.require_full_permission(args.allow_full, stage="RAGAS 复算全量")
    except RunPolicyError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    run_policy.print_banner()
    args.limit = run_policy.clamp_samples(args.limit, stage="RAGAS 复算")

    env_file = load_env_file()
    key = resolve_api_key("dashscope" if "dashscope" in args.base_url else "other", env_file)
    if not key:
        print("[FAIL] 找不到 API key（DASHSCOPE_API_KEY）", file=sys.stderr)
        return 2

    samples = build_samples(args.limit)
    refused = sum(1 for s in samples if s["refused_by_threshold"])
    print(f"[INFO] 样本 {len(samples)} 条（其中阈值拒答 {refused} 条）| 裁判 {args.judge} | embedding {args.embed}")

    # 延迟导入，缺依赖时给出可操作的提示
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextRecall,
            LLMContextPrecisionWithReference,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import ragas
    except ImportError as exc:
        print(f"[FAIL] 依赖缺失：{exc}", file=sys.stderr)
        print("  安装：pip install --target D:/heima/.ragas_lib ragas langchain-openai datasets", file=sys.stderr)
        print("  运行前：set PYTHONPATH=D:/heima/.ragas_lib", file=sys.stderr)
        return 2

    print(f"[INFO] ragas {getattr(ragas, '__version__', '?')}")

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=args.judge, api_key=key, base_url=args.base_url,
        temperature=0.0, timeout=args.timeout, max_retries=3,
    ))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=args.embed, api_key=key, base_url=args.base_url,
        timeout=args.timeout, max_retries=3,
        check_embedding_ctx_length=False,
    ))

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    ds = Dataset.from_list([{
        "question": s["question"],
        "answer": s["answer"],
        "contexts": s["contexts"],
        "ground_truth": s["ground_truth"],
    } for s in samples])

    t0 = time.time()
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings,
                      raise_exceptions=False)
    cost_s = round(time.time() - t0, 1)
    print(f"[INFO] 打分完成，用时 {cost_s}s")

    df = result.to_pandas()
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rows = []
    for i, s in enumerate(samples):
        row = dict(s)
        for k in metric_keys:
            v = df.iloc[i].get(k) if i < len(df) else None
            row[k] = float(v) if isinstance(v, (int, float)) and v == v else None
        rows.append(row)

    answered = [r for r in rows if not r["refused_by_threshold"]]
    summary = {
        "full": aggregate(rows, metric_keys),
        "answered": aggregate(answered, metric_keys),
        "refused_count": len(rows) - len(answered),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine": f"ragas(official)/judge={args.judge}/embed={args.embed}",
        "judge_model": args.judge,
        "embedding_model": args.embed,
        "eval_source": {
            "contexts": os.path.relpath(RETRIEVAL_FILE, BASE),
            "answers": os.path.relpath(ANSWER_FILE, BASE),
            "note": "离线复算：检索与答案为同日真实批次，仅裁判模型为新跑",
        },
        "summary": summary,
        "elapsed_sec": cost_s,
        "per_sample": rows,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"ragas_rejudge_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUT_DIR, f"ragas_rejudge_{stamp}.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        cols = ["question", "refused_by_threshold", "top_score"] + metric_keys
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                '"%s"' % str(r.get(c, "")).replace('"', "'") for c in cols
            ) + "\n")

    print("\n=== 全量 %d 条 ===" % summary["full"]["n"])
    for k in metric_keys:
        print(f"  {k:20s} {summary['full'][k]}")
    print("\n=== 仅作答 %d 条（剔除 %d 条拒答）===" % (summary["answered"]["n"], summary["refused_count"]))
    for k in metric_keys:
        print(f"  {k:20s} {summary['answered'][k]}")
    print(f"\n报告：{json_path}")
    print(f"明细：{csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
