"""
检索复现差异对比 — 比较两个 run 的检索结果差异。

按 question_id 对齐，使用每个 run 自己的真实 trace_id 和 retrieval results。
输出 CSV 和 Markdown 诊断报告。

用法：
    from retrieval_diff import compare_runs
    report = compare_runs(old_run_id, new_run_id)
    print(report["markdown"])
"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path

from judge import get_gold_evidence, _normalize_text
from experiment import load_experiment_run


def _load_run_samples(run_id):
    """加载某个 run 的 processed samples，按 question_id 索引。

    Returns:
        dict: question_id -> sample dict (不含 observations)
    """
    proc_path = Path(__file__).parent / "data" / "processed" / "langfuse_samples.jsonl"
    by_qid = {}
    if not proc_path.exists():
        return by_qid

    with proc_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 匹配 run_id：直接字段或 user_id 嵌入
            obj_run_id = obj.get("run_id", "")
            if not obj_run_id:
                uid = obj.get("user_id", "")
                if uid.startswith("rag_eval:"):
                    parts = uid.split(":", 2)
                    if len(parts) == 3:
                        obj_run_id = parts[1]
            if obj_run_id != run_id:
                continue

            qid = obj.get("question_id", "")
            if not qid:
                continue

            obj.pop("observations", None)
            by_qid[qid] = obj

    return by_qid


def _load_run_judge_results(run_id, trace_ids):
    """加载某个 run 的 judge results，按 trace_id 索引。

    Args:
        run_id: run identifier
        trace_ids: set of real trace_ids belonging to this run

    Returns:
        dict: trace_id -> judge result dict
    """
    judged_path = Path(__file__).parent / "data" / "judged" / "eval_results.jsonl"
    by_tid = {}
    if not judged_path.exists():
        return by_tid

    with judged_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("trace_id", "")
            if obj.get("run_id") == run_id or tid in trace_ids:
                by_tid[tid] = obj

    return by_tid


def _find_evidence_rank(gold_norm, retrieval_results):
    """在 retrieval_results 中查找 gold_evidence 首次出现的 rank (1-based)。

    使用规范化文本匹配。返回 rank (1-10) 或 None。
    """
    if not gold_norm or not retrieval_results:
        return None
    for i, r in enumerate(retrieval_results[:10]):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        content_norm = _normalize_text(content)
        if gold_norm in content_norm:
            return i + 1
    return None


def _classify(old_rank, new_rank, old_judge, new_judge):
    """分类单个 question 的差异（用于逐题浏览）。

    Returns:
        str: evidence_lost | ranking_regression | judge_disagreement | unchanged
    """
    old_hit = old_rank is not None
    new_hit = new_rank is not None

    # evidence_lost: 旧有证据，新 Top10 无
    if old_hit and not new_hit:
        return "evidence_lost"

    # ranking_regression: 新有证据但 rank 下降
    if old_hit and new_hit and new_rank > old_rank:
        return "ranking_regression"

    # judge_disagreement: 检索结果有证据，但 Judge 结论不同
    old_t5 = old_judge.get("retrieval_top5_hit") if old_judge else None
    new_t5 = new_judge.get("retrieval_top5_hit") if new_judge else None
    if old_t5 is not None and new_t5 is not None and old_t5 != new_t5:
        return "judge_disagreement"

    return "unchanged"


def _cutoff_stats_for_pair(old_rank, new_rank, K):
    """判断单个 question 在 cutoff K 下的分类。

    Returns:
        str: loss | evidence_lost | ranking_drop | gain | neutral
    """
    old_in = old_rank is not None and old_rank <= K
    new_in = new_rank is not None and new_rank <= K

    if old_in and not new_in:
        # 旧在 TopK 内，新不在 TopK 内
        if new_rank is None:
            return "evidence_lost"  # 新 Top10 完全无证据
        else:
            return "ranking_drop"  # 新仍有证据但跌出 TopK
    elif not old_in and new_in:
        return "gain"
    else:
        return "neutral"


def compute_cutoff_stats(rows):
    """对所有对齐题目，按 cutoff=1,3,5 分别统计。

    Returns:
        dict: {1: {...}, 3: {...}, 5: {...}}，每个 cutoff 包含:
            loss, evidence_lost, ranking_drop, gain, neutral,
            old_hit_count, new_hit_count
    """
    stats = {}
    for K in (1, 3, 5):
        s = {"loss": 0, "evidence_lost": 0, "ranking_drop": 0,
             "gain": 0, "neutral": 0, "old_hit_count": 0, "new_hit_count": 0}
        for r in rows:
            old_rank = r.get("old_rank")
            new_rank = r.get("new_rank")
            if old_rank is not None and old_rank <= K:
                s["old_hit_count"] += 1
            if new_rank is not None and new_rank <= K:
                s["new_hit_count"] += 1
            cat = _cutoff_stats_for_pair(old_rank, new_rank, K)
            s[cat] += 1
        s["delta"] = s["new_hit_count"] - s["old_hit_count"]
        stats[K] = s
    return stats


def compare_runs(old_run_id, new_run_id):
    """比较两个 run 的检索结果差异。

    Args:
        old_run_id: 旧 run (baseline)
        new_run_id: 新 run (under test)

    Returns:
        dict with keys:
            rows: list of per-question comparison dicts
            summary: aggregation dict
            markdown: Markdown report string
            csv_string: CSV string
            old_config / new_config: config snapshots
    """
    # Load manifests
    old_manifest = load_experiment_run(old_run_id)
    new_manifest = load_experiment_run(new_run_id)
    if not old_manifest:
        raise ValueError(f"旧 run 不存在: {old_run_id}")
    if not new_manifest:
        raise ValueError(f"新 run 不存在: {new_run_id}")

    # Load samples by question_id
    old_samples = _load_run_samples(old_run_id)
    new_samples = _load_run_samples(new_run_id)

    # Collect trace_ids per run
    old_trace_ids = {s.get("trace_id") for s in old_samples.values() if s.get("trace_id")}
    new_trace_ids = {s.get("trace_id") for s in new_samples.values() if s.get("trace_id")}

    # Load judge results
    old_judge_map = _load_run_judge_results(old_run_id, old_trace_ids)
    new_judge_map = _load_run_judge_results(new_run_id, new_trace_ids)

    # Align by question_id
    all_qids = sorted(set(old_samples.keys()) | set(new_samples.keys()))

    rows = []
    counts = {"evidence_lost": 0, "ranking_regression": 0,
              "judge_disagreement": 0, "unchanged": 0}
    missing_old = 0
    missing_new = 0

    for qid in all_qids:
        old_s = old_samples.get(qid)
        new_s = new_samples.get(qid)

        if not old_s:
            missing_old += 1
            continue
        if not new_s:
            missing_new += 1
            continue

        # Gold evidence
        gold = get_gold_evidence(old_s) or get_gold_evidence(new_s)
        gold_norm = _normalize_text(gold) if gold else ""

        # Find evidence rank in each run's retrieval results
        old_rank = _find_evidence_rank(gold_norm, old_s.get("retrieval_results") or [])
        new_rank = _find_evidence_rank(gold_norm, new_s.get("retrieval_results") or [])

        # Judge results (via real trace_id)
        old_tid = old_s.get("trace_id", "")
        new_tid = new_s.get("trace_id", "")
        old_judge = old_judge_map.get(old_tid)
        new_judge = new_judge_map.get(new_tid)

        # Classify
        category = _classify(old_rank, new_rank, old_judge, new_judge)
        counts[category] += 1

        row = {
            "question_id": qid,
            "question": (old_s.get("question") or new_s.get("question") or "")[:80],
            "old_trace_id": old_tid,
            "new_trace_id": new_tid,
            "gold_evidence": gold[:100],
            "old_rank": old_rank,
            "new_rank": new_rank,
            "old_top1": old_judge.get("retrieval_top1_hit") if old_judge else None,
            "old_top5": old_judge.get("retrieval_top5_hit") if old_judge else None,
            "new_top1": new_judge.get("retrieval_top1_hit") if new_judge else None,
            "new_top5": new_judge.get("retrieval_top5_hit") if new_judge else None,
            "category": category,
        }
        rows.append(row)

    # Config snapshots
    old_cfg = old_manifest.get("config_snapshot") or {}
    new_cfg = new_manifest.get("config_snapshot") or {}

    # Summary
    total = len(rows)
    summary = {
        "total_questions": total,
        "missing_in_old": missing_old,
        "missing_in_new": missing_new,
        **counts,
        "old_top1_hit_rate": (
            sum(1 for r in rows if r["old_top1"]) / total if total else 0
        ),
        "new_top1_hit_rate": (
            sum(1 for r in rows if r["new_top1"]) / total if total else 0
        ),
        "old_top5_hit_rate": (
            sum(1 for r in rows if r["old_top5"]) / total if total else 0
        ),
        "new_top5_hit_rate": (
            sum(1 for r in rows if r["new_top5"]) / total if total else 0
        ),
    }
    summary["top1_delta"] = summary["new_top1_hit_rate"] - summary["old_top1_hit_rate"]
    summary["top5_delta"] = summary["new_top5_hit_rate"] - summary["old_top5_hit_rate"]

    # Per-cutoff statistics
    cutoff = compute_cutoff_stats(rows)
    summary["cutoff"] = cutoff

    # Primary diagnosis per cutoff
    cause_lines = []
    for K in (1, 3, 5):
        s = cutoff[K]
        if s["delta"] >= 0:
            cause_lines.append(f"Top{K}: 改善 {s['delta']:+d}（gain={s['gain']}）")
        else:
            if s["evidence_lost"] > s["ranking_drop"]:
                cause = f"evidence_lost（{s['evidence_lost']} 条）"
            elif s["ranking_drop"] > s["evidence_lost"]:
                cause = f"ranking_drop（{s['ranking_drop']} 条）"
            else:
                cause = f"evidence_lost={s['evidence_lost']} + ranking_drop={s['ranking_drop']}"
            cause_lines.append(f"Top{K}: 下降 {s['delta']}，主因 {cause}")
    summary["primary_cause_desc"] = "；".join(cause_lines)

    # Generate CSV
    csv_string = _build_csv(rows)

    # Generate Markdown
    markdown = _build_markdown(old_run_id, new_run_id, old_cfg, new_cfg,
                               old_manifest, new_manifest, rows, summary)

    return {
        "rows": rows,
        "summary": summary,
        "markdown": markdown,
        "csv_string": csv_string,
        "old_config": old_cfg,
        "new_config": new_cfg,
        "old_run_id": old_run_id,
        "new_run_id": new_run_id,
    }


def _build_csv(rows):
    """生成 CSV 字符串。"""
    if not rows:
        return ""
    fieldnames = [
        "question_id", "question", "old_trace_id", "new_trace_id",
        "gold_evidence", "old_rank", "new_rank",
        "old_top1", "old_top5", "new_top1", "new_top5", "category",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _build_markdown(old_run_id, new_run_id, old_cfg, new_cfg,
                    old_manifest, new_manifest, rows, summary):
    """生成 Markdown 诊断报告。"""
    lines = []
    lines.append("# 检索复现差异对比报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Run info
    lines.append("## 运行信息")
    lines.append("")
    lines.append(f"| 项目 | 旧 Run (baseline) | 新 Run (under test) |")
    lines.append(f"|------|-------------------|---------------------|")
    lines.append(f"| Run ID | `{old_run_id[:40]}...` | `{new_run_id[:40]}...` |")
    lines.append(f"| 配置名称 | {old_cfg.get('config_name', 'N/A')} | {new_cfg.get('config_name', 'N/A')} |")
    lines.append(f"| 知识库版本 | {old_cfg.get('knowledge_base_version', 'N/A')} | {new_cfg.get('knowledge_base_version', 'N/A')} |")
    lines.append(f"| 工作流版本 | {old_cfg.get('workflow_version', 'N/A')} | {new_cfg.get('workflow_version', 'N/A')} |")
    lines.append(f"| 检索模式 | {old_cfg.get('retrieval_mode', 'N/A')} | {new_cfg.get('retrieval_mode', 'N/A')} |")
    lines.append(f"| Top K | {old_cfg.get('top_k', 'N/A')} | {new_cfg.get('top_k', 'N/A')} |")
    lines.append(f"| Rerank | {old_cfg.get('rerank_model', 'N/A')} | {new_cfg.get('rerank_model', 'N/A')} |")
    lines.append(f"| Embedding | {old_cfg.get('embedding_model', 'N/A')} | {new_cfg.get('embedding_model', 'N/A')} |")
    lines.append(f"| 分块策略 | {old_cfg.get('chunk_strategy', 'N/A')} | {new_cfg.get('chunk_strategy', 'N/A')} |")
    lines.append(f"| 运行时间 | {(old_manifest.get('started_at') or '')[:19]} | {(new_manifest.get('started_at') or '')[:19]} |")
    qs_id = old_manifest.get("question_set_id") or new_manifest.get("question_set_id") or "N/A"
    qs_name = old_manifest.get("question_set_name") or new_manifest.get("question_set_name") or "N/A"
    lines.append(f"| 题集 ID | {qs_id} | — |")
    lines.append(f"| 题集名称 | {qs_name} | — |")
    lines.append("")

    # Summary
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 对齐题目数 | {summary['total_questions']} |")
    lines.append(f"| 旧 run 缺失 | {summary['missing_in_old']} |")
    lines.append(f"| 新 run 缺失 | {summary['missing_in_new']} |")
    lines.append(f"| evidence_lost (全局) | **{summary['evidence_lost']}** |")
    lines.append(f"| ranking_regression (全局) | **{summary['ranking_regression']}** |")
    lines.append(f"| judge_disagreement | **{summary['judge_disagreement']}** |")
    lines.append(f"| unchanged | {summary['unchanged']} |")
    lines.append("")

    # Per-cutoff table
    lines.append("### 按 TopK Cutoff 分解")
    lines.append("")
    lines.append("| Cutoff | 旧命中 | 新命中 | 变化 | loss | evidence_lost | ranking_drop | gain |")
    lines.append("|--------|--------|--------|------|------|---------------|--------------|------|")
    for K in (1, 3, 5):
        s = summary["cutoff"][K]
        lines.append(
            f"| Top{K} | {s['old_hit_count']} | {s['new_hit_count']} | "
            f"{s['delta']:+d} | {s['loss']} | {s['evidence_lost']} | "
            f"{s['ranking_drop']} | {s['gain']} |"
        )
    lines.append("")
    lines.append(f"**诊断**: {summary['primary_cause_desc']}")
    lines.append("")

    # Evidence lost details
    lost_rows = [r for r in rows if r["category"] == "evidence_lost"]
    if lost_rows:
        lines.append("## evidence_lost 详情")
        lines.append("")
        for r in lost_rows:
            lines.append(f"- **{r['question_id']}**: {r['question']}")
            lines.append(f"  - 旧 rank: {r['old_rank']}, 新 rank: 未找到")
            lines.append(f"  - 金标准: {r['gold_evidence'][:80]}")
        lines.append("")

    # Ranking regression details
    reg_rows = [r for r in rows if r["category"] == "ranking_regression"]
    if reg_rows:
        lines.append("## ranking_regression 详情")
        lines.append("")
        for r in reg_rows:
            lines.append(f"- **{r['question_id']}**: {r['question']}")
            lines.append(f"  - 旧 rank: {r['old_rank']} → 新 rank: {r['new_rank']}")
            lines.append(f"  - 金标准: {r['gold_evidence'][:80]}")
        lines.append("")

    # Judge disagreement details
    dis_rows = [r for r in rows if r["category"] == "judge_disagreement"]
    if dis_rows:
        lines.append("## judge_disagreement 详情")
        lines.append("")
        for r in dis_rows:
            lines.append(f"- **{r['question_id']}**: {r['question']}")
            lines.append(f"  - 旧 Top5: {r['old_top5']}, 新 Top5: {r['new_top5']}")
        lines.append("")

    return "\n".join(lines)
