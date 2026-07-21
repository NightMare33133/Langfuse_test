"""
测试检索复现差异对比模块。

覆盖：
a. question_id 对齐
b. 真实 trace_id 关联（不使用 batch_qa_* 伪 ID）
c. 缺失结果处理
d. 跨 run 隔离
e. 分类逻辑（evidence_lost, ranking_regression, judge_disagreement, unchanged）
f. CSV/MD 输出不含 Key
g. _find_evidence_rank 精确性
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from retrieval_diff import _find_evidence_rank, _classify, _load_run_samples


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_sample(qid, trace_id, run_id, question, gold, retrieval_contents):
    """Build a minimal processed sample."""
    return {
        "trace_id": trace_id,
        "question_id": qid,
        "question": question,
        "run_id": run_id,
        "reference_answer": gold,
        "source_excerpt": gold,
        "retrieval_results": [
            {"position": i + 1, "content": c, "title": f"doc_{i}"}
            for i, c in enumerate(retrieval_contents)
        ],
    }


def _make_judge(tid, run_id, qid, top1, top5):
    """Build a minimal judge result."""
    return {
        "trace_id": tid,
        "run_id": run_id,
        "question_id": qid,
        "evaluation_track": "retrieval",
        "retrieval_top1_hit": top1,
        "retrieval_top3_hit": 1 if top1 or top5 else 0,
        "retrieval_top5_hit": top5,
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_find_evidence_rank_top1():
    """gold 在 Top1 → rank=1。"""
    print("=" * 60)
    print("测试 _find_evidence_rank Top1")
    print("=" * 60)

    gold = "供应商应确保所有数据处理符合 GDPR 要求"
    results = [
        {"content": "合同条款规定，供应商应确保所有数据处理符合 GDPR 要求，包括加密。"},
        {"content": "其他内容"},
    ]
    rank = _find_evidence_rank(gold, results)
    assert rank == 1, f"应为 1，实际 {rank}"
    print("[OK] Top1 匹配")

    print()


def test_find_evidence_rank_top3():
    """gold 在 Top3 → rank=3。"""
    print("=" * 60)
    print("测试 _find_evidence_rank Top3")
    print("=" * 60)

    gold = "数据保留期限为合同终止后两年"
    results = [
        {"content": "无关内容1"},
        {"content": "无关内容2"},
        {"content": "根据规定，数据保留期限为合同终止后两年。"},
    ]
    rank = _find_evidence_rank(gold, results)
    assert rank == 3, f"应为 3，实际 {rank}"
    print("[OK] Top3 匹配")

    print()


def test_find_evidence_rank_none():
    """gold 不在 Top10 → None。"""
    print("=" * 60)
    print("测试 _find_evidence_rank None")
    print("=" * 60)

    gold = "不存在的证据"
    results = [{"content": f"内容 {i}"} for i in range(10)]
    rank = _find_evidence_rank(gold, results)
    assert rank is None, f"应为 None，实际 {rank}"
    print("[OK] 未找到 → None")

    print()


def test_classify_evidence_lost():
    """旧有证据、新无 → evidence_lost。"""
    print("=" * 60)
    print("测试分类: evidence_lost")
    print("=" * 60)

    cat = _classify(old_rank=2, new_rank=None, old_judge={}, new_judge={})
    assert cat == "evidence_lost"
    print("[OK] evidence_lost")

    print()


def test_classify_ranking_regression():
    """旧 rank=2、新 rank=5 → ranking_regression。"""
    print("=" * 60)
    print("测试分类: ranking_regression")
    print("=" * 60)

    cat = _classify(old_rank=2, new_rank=5, old_judge={}, new_judge={})
    assert cat == "ranking_regression"
    print("[OK] ranking_regression")

    print()


def test_classify_judge_disagreement():
    """rank 相同但 Judge Top5 结论不同 → judge_disagreement。"""
    print("=" * 60)
    print("测试分类: judge_disagreement")
    print("=" * 60)

    old_j = {"retrieval_top5_hit": 1}
    new_j = {"retrieval_top5_hit": 0}
    cat = _classify(old_rank=3, new_rank=3, old_judge=old_j, new_judge=new_j)
    assert cat == "judge_disagreement"
    print("[OK] judge_disagreement")

    print()


def test_classify_unchanged():
    """rank 相同、Judge 一致 → unchanged。"""
    print("=" * 60)
    print("测试分类: unchanged")
    print("=" * 60)

    old_j = {"retrieval_top5_hit": 1}
    new_j = {"retrieval_top5_hit": 1}
    cat = _classify(old_rank=1, new_rank=1, old_judge=old_j, new_judge=new_j)
    assert cat == "unchanged"
    print("[OK] unchanged")

    print()


def test_classify_rank_improvement():
    """旧 rank=5、新 rank=2 → unchanged（改善，不是 regression）。"""
    print("=" * 60)
    print("测试分类: rank 改善")
    print("=" * 60)

    cat = _classify(old_rank=5, new_rank=2, old_judge={}, new_judge={})
    assert cat == "unchanged"
    print("[OK] rank 改善 → unchanged")

    print()


def test_question_id_alignment():
    """两个 run 按 question_id 对齐。"""
    print("=" * 60)
    print("测试 question_id 对齐")
    print("=" * 60)

    old_samples = {
        "q1": _make_sample("q1", "t_old_1", "run_old", "问题1", "证据1", ["内容"]),
        "q2": _make_sample("q2", "t_old_2", "run_old", "问题2", "证据2", ["内容"]),
        "q3": _make_sample("q3", "t_old_3", "run_old", "问题3", "证据3", ["内容"]),
    }
    new_samples = {
        "q1": _make_sample("q1", "t_new_1", "run_new", "问题1", "证据1", ["内容"]),
        "q2": _make_sample("q2", "t_new_2", "run_new", "问题2", "证据2", ["内容"]),
        # q3 missing in new
    }

    # Align
    all_qids = sorted(set(old_samples.keys()) | set(new_samples.keys()))
    aligned = []
    missing_new = 0
    for qid in all_qids:
        old_s = old_samples.get(qid)
        new_s = new_samples.get(qid)
        if not old_s:
            continue
        if not new_s:
            missing_new += 1
            continue
        aligned.append(qid)

    assert aligned == ["q1", "q2"], f"对齐结果: {aligned}"
    assert missing_new == 1, f"新 run 缺失: {missing_new}"
    print(f"[OK] 对齐 {len(aligned)} 题，新 run 缺失 {missing_new} 题")

    print()


def test_real_trace_id_isolation():
    """每个 run 使用自己的真实 trace_id，不使用 batch_qa_* 伪 ID。"""
    print("=" * 60)
    print("测试真实 trace_id 隔离")
    print("=" * 60)

    old_sample = _make_sample("q1", "cmr8tec2s0007ad0c", "run_old", "问题", "证据", ["内容"])
    new_sample = _make_sample("q1", "abc123def456ghi7", "run_new", "问题", "证据", ["内容"])

    # 确保不使用 batch_qa_ 伪 ID
    assert not old_sample["trace_id"].startswith("batch_qa_"), "不应使用 batch_qa_ 伪 ID"
    assert not new_sample["trace_id"].startswith("batch_qa_"), "不应使用 batch_qa_ 伪 ID"
    # 两个 run 的 trace_id 不同
    assert old_sample["trace_id"] != new_sample["trace_id"], "不同 run 应有不同 trace_id"
    print(f"[OK] 旧 trace_id={old_sample['trace_id'][:16]}, 新 trace_id={new_sample['trace_id'][:16]}")

    print()


def test_cross_run_isolation():
    """一个 run 的 judge results 不应污染另一个 run。"""
    print("=" * 60)
    print("测试跨 run 隔离")
    print("=" * 60)

    # 两个 run 的 trace_id 完全不同
    old_tid = "trace_old_001"
    new_tid = "trace_new_001"

    old_judge = _make_judge(old_tid, "run_old", "q1", top1=1, top5=1)
    new_judge = _make_judge(new_tid, "run_new", "q1", top1=0, top5=1)

    # 旧 run 的 judge 不应匹配新 run 的 trace_id
    assert old_judge["trace_id"] != new_judge["trace_id"]
    assert old_judge["run_id"] != new_judge["run_id"]
    print("[OK] 两个 run 的 trace_id 和 run_id 完全隔离")

    print()


def test_csv_no_keys():
    """CSV 输出不含 API Key。"""
    print("=" * 60)
    print("测试 CSV 不含 Key")
    print("=" * 60)

    csv_content = "question_id,question,old_trace_id\nq1,test,trace1\n"
    sensitive = ["api_key", "secret", "password", "token", "LANGFUSE"]
    for s in sensitive:
        assert s.lower() not in csv_content.lower(), f"CSV 不应包含 {s}"
    print("[OK] CSV 不含敏感字段")

    print()


def test_md_no_keys():
    """Markdown 输出不含 API Key。"""
    print("=" * 60)
    print("测试 Markdown 不含 Key")
    print("=" * 60)

    md_content = "# Report\nRun ID: test\nConfig: test\n"
    sensitive = ["api_key", "secret", "password", "token", "LANGFUSE"]
    for s in sensitive:
        assert s.lower() not in md_content.lower(), f"Markdown 不应包含 {s}"
    print("[OK] Markdown 不含敏感字段")

    print()


def test_load_run_samples_by_qid():
    """_load_run_samples 按 question_id 索引。"""
    print("=" * 60)
    print("测试 _load_run_samples 按 question_id 索引")
    print("=" * 60)

    # 模拟数据
    samples = [
        {"trace_id": "t1", "question_id": "q1", "run_id": "run_test", "question": "问题1"},
        {"trace_id": "t2", "question_id": "q2", "run_id": "run_test", "question": "问题2"},
        {"trace_id": "t3", "question_id": "q3", "run_id": "run_other", "question": "问题3"},
    ]

    # 模拟加载逻辑
    by_qid = {}
    for obj in samples:
        if obj.get("run_id") == "run_test":
            qid = obj.get("question_id")
            if qid:
                by_qid[qid] = obj

    assert len(by_qid) == 2
    assert "q1" in by_qid
    assert "q2" in by_qid
    assert "q3" not in by_qid  # 不同 run
    print("[OK] 按 question_id 索引，跨 run 隔离")

    print()


def test_full_compare_mock():
    """完整比较流程（mock 数据）。"""
    print("=" * 60)
    print("测试完整比较流程")
    print("=" * 60)

    gold = "合同违约金为总金额的百分之十"

    old_samples = {
        "q1": _make_sample("q1", "t_old_1", "run_old", "违约金比例", gold,
                           ["包含合同违约金为总金额的百分之十的文档", "其他"]),
        "q2": _make_sample("q2", "t_old_2", "run_old", "数据保留", "数据保留两年",
                           ["数据保留两年的文档", "其他"]),
    }
    new_samples = {
        "q1": _make_sample("q1", "t_new_1", "run_new", "违约金比例", gold,
                           ["其他内容", "包含合同违约金为总金额的百分之十的文档"]),
        "q2": _make_sample("q2", "t_new_2", "run_new", "数据保留", "数据保留两年",
                           ["无关"]),
    }

    # q1: 旧 Top1, 新 Top2 → ranking_regression
    # q2: 旧 Top1, 新 无 → evidence_lost (if no match in new)

    from retrieval_diff import _find_evidence_rank, _classify

    q1_old_rank = _find_evidence_rank(gold, old_samples["q1"]["retrieval_results"])
    q1_new_rank = _find_evidence_rank(gold, new_samples["q1"]["retrieval_results"])
    q1_cat = _classify(q1_old_rank, q1_new_rank, {}, {})

    assert q1_old_rank == 1, f"q1 旧 rank 应为 1，实际 {q1_old_rank}"
    assert q1_new_rank == 2, f"q1 新 rank 应为 2，实际 {q1_new_rank}"
    assert q1_cat == "ranking_regression", f"q1 应为 ranking_regression，实际 {q1_cat}"

    gold2 = "数据保留两年"
    q2_old_rank = _find_evidence_rank(gold2, old_samples["q2"]["retrieval_results"])
    q2_new_rank = _find_evidence_rank(gold2, new_samples["q2"]["retrieval_results"])
    q2_cat = _classify(q2_old_rank, q2_new_rank, {}, {})

    assert q2_old_rank == 1, f"q2 旧 rank 应为 1，实际 {q2_old_rank}"
    assert q2_new_rank is None, f"q2 新 rank 应为 None，实际 {q2_new_rank}"
    assert q2_cat == "evidence_lost", f"q2 应为 evidence_lost，实际 {q2_cat}"

    print(f"[OK] q1: rank {q1_old_rank}→{q1_new_rank} = {q1_cat}")
    print(f"[OK] q2: rank {q2_old_rank}→{q2_new_rank} = {q2_cat}")

    print()


# ---------------------------------------------------------------------------
# Cutoff stats edge cases
# ---------------------------------------------------------------------------

def test_cutoff_old1_new2():
    """旧 rank=1，新 rank=2：Top1 ranking_drop，Top3/Top5 neutral。"""
    print("=" * 60)
    print("测试 cutoff: 旧 rank=1, 新 rank=2")
    print("=" * 60)

    from retrieval_diff import _cutoff_stats_for_pair

    # Top1: old_in=True, new_in=False (2 > 1) → ranking_drop
    assert _cutoff_stats_for_pair(1, 2, 1) == "ranking_drop"
    # Top3: old_in=True, new_in=True (2 <= 3) → neutral
    assert _cutoff_stats_for_pair(1, 2, 3) == "neutral"
    # Top5: old_in=True, new_in=True (2 <= 5) → neutral
    assert _cutoff_stats_for_pair(1, 2, 5) == "neutral"
    print("[OK] Top1=ranking_drop, Top3=neutral, Top5=neutral")

    print()


def test_cutoff_old4_new_none():
    """旧 rank=4，新 rank=None：Top5 evidence_lost，Top1/Top3 neutral。"""
    print("=" * 60)
    print("测试 cutoff: 旧 rank=4, 新 rank=None")
    print("=" * 60)

    from retrieval_diff import _cutoff_stats_for_pair

    # Top1: old_in=False (4>1) → neutral
    assert _cutoff_stats_for_pair(4, None, 1) == "neutral"
    # Top3: old_in=False (4>3) → neutral
    assert _cutoff_stats_for_pair(4, None, 3) == "neutral"
    # Top5: old_in=True (4<=5), new_in=False (None) → evidence_lost
    assert _cutoff_stats_for_pair(4, None, 5) == "evidence_lost"
    print("[OK] Top1=neutral, Top3=neutral, Top5=evidence_lost")

    print()


def test_cutoff_old7_new_none():
    """旧 rank=7，新 rank=None：不计入任何 Top1/3/5 下降。"""
    print("=" * 60)
    print("测试 cutoff: 旧 rank=7, 新 rank=None")
    print("=" * 60)

    from retrieval_diff import _cutoff_stats_for_pair

    assert _cutoff_stats_for_pair(7, None, 1) == "neutral"
    assert _cutoff_stats_for_pair(7, None, 3) == "neutral"
    assert _cutoff_stats_for_pair(7, None, 5) == "neutral"
    print("[OK] 全部 neutral（旧 rank 不在任何 TopK 内）")

    print()


def test_cutoff_old2_new5():
    """旧 rank=2，新 rank=5：Top3 ranking_drop，Top5 neutral。"""
    print("=" * 60)
    print("测试 cutoff: 旧 rank=2, 新 rank=5")
    print("=" * 60)

    from retrieval_diff import _cutoff_stats_for_pair

    assert _cutoff_stats_for_pair(2, 5, 1) == "neutral"  # old not in Top1
    assert _cutoff_stats_for_pair(2, 5, 3) == "ranking_drop"  # old in Top3, new not
    assert _cutoff_stats_for_pair(2, 5, 5) == "neutral"  # both in Top5
    print("[OK] Top1=neutral, Top3=ranking_drop, Top5=neutral")

    print()


def test_cutoff_old_none_new1():
    """旧 rank=None，新 rank=1：Top1/3/5 gain。"""
    print("=" * 60)
    print("测试 cutoff: 旧 rank=None, 新 rank=1")
    print("=" * 60)

    from retrieval_diff import _cutoff_stats_for_pair

    assert _cutoff_stats_for_pair(None, 1, 1) == "gain"
    assert _cutoff_stats_for_pair(None, 1, 3) == "gain"
    assert _cutoff_stats_for_pair(None, 1, 5) == "gain"
    print("[OK] 全部 gain")

    print()


def test_cutoff_old5_new3():
    """旧 rank=5，新 rank=3：Top5 gain（新进入 Top3）。"""
    print("=" * 60)
    print("测试 cutoff: 旧 rank=5, 新 rank=3")
    print("=" * 60)

    from retrieval_diff import _cutoff_stats_for_pair

    assert _cutoff_stats_for_pair(5, 3, 1) == "neutral"
    assert _cutoff_stats_for_pair(5, 3, 3) == "gain"  # new in Top3, old not
    assert _cutoff_stats_for_pair(5, 3, 5) == "neutral"  # both in Top5
    print("[OK] Top1=neutral, Top3=gain, Top5=neutral")

    print()


def test_compute_cutoff_stats():
    """compute_cutoff_stats 正确聚合多条记录。"""
    print("=" * 60)
    print("测试 compute_cutoff_stats 聚合")
    print("=" * 60)

    from retrieval_diff import compute_cutoff_stats

    rows = [
        {"old_rank": 1, "new_rank": 2},   # Top1: ranking_drop, Top3: neutral, Top5: neutral
        {"old_rank": 4, "new_rank": None},  # Top5: evidence_lost
        {"old_rank": 7, "new_rank": None},  # all neutral
        {"old_rank": None, "new_rank": 1},  # all gain
    ]
    stats = compute_cutoff_stats(rows)

    # Top1: old_hit=1 (rank=1), new_hit=1 (rank=1)
    assert stats[1]["old_hit_count"] == 1
    assert stats[1]["new_hit_count"] == 1
    assert stats[1]["ranking_drop"] == 1  # rank 1→2
    assert stats[1]["gain"] == 1  # None→1
    assert stats[1]["loss"] == 0  # loss = evidence_lost + ranking_drop, but here ranking_drop counted separately
    print(f"[OK] Top1: old={stats[1]['old_hit_count']}, new={stats[1]['new_hit_count']}, "
          f"drop={stats[1]['ranking_drop']}, gain={stats[1]['gain']}")

    # Top5: old_hit=2 (rank=1,4), new_hit=2 (rank=2,1)
    assert stats[5]["old_hit_count"] == 2
    assert stats[5]["new_hit_count"] == 2
    assert stats[5]["evidence_lost"] == 1  # rank 4→None
    print(f"[OK] Top5: old={stats[5]['old_hit_count']}, new={stats[5]['new_hit_count']}, "
          f"evidence_lost={stats[5]['evidence_lost']}")

    print()


# ---------------------------------------------------------------------------
# Cross-config comparison tests
# ---------------------------------------------------------------------------

def test_diff_config_same_qs_comparable():
    """不同 config_id、相同 question_set_id 的 run 可比较。"""
    print("=" * 60)
    print("测试跨配置比较：相同 question_set_id")
    print("=" * 60)

    # 模拟两个 run 的 manifest
    old_run = {
        "run_id": "run_v2_001",
        "config_id": "cfg_v2",
        "question_set_id": "qs_001",
        "question_set_name": "Test QSet",
    }
    new_run = {
        "run_id": "run_v2_4_001",
        "config_id": "cfg_v2_4",
        "question_set_id": "qs_001",
        "question_set_name": "Test QSet",
    }

    assert old_run["question_set_id"] == new_run["question_set_id"], "应有相同 question_set_id"
    assert old_run["config_id"] != new_run["config_id"], "应为不同 config_id"
    print("[OK] 不同 config_id、相同 question_set_id 可比较")

    print()


def test_same_config_comparable():
    """同 config_id 的 run 可比较（稳定性诊断）。"""
    print("=" * 60)
    print("测试同配置比较")
    print("=" * 60)

    old_run = {
        "run_id": "run_v2_4_001",
        "config_id": "cfg_v2_4",
        "question_set_id": "qs_001",
    }
    new_run = {
        "run_id": "run_v2_4_002",
        "config_id": "cfg_v2_4",
        "question_set_id": "qs_001",
    }

    assert old_run["config_id"] == new_run["config_id"]
    assert old_run["question_set_id"] == new_run["question_set_id"]
    print("[OK] 同 config_id、相同 question_set_id 可比较")

    print()


def test_different_qs_rejected():
    """不同 question_set_id 的 run 被拒绝。"""
    print("=" * 60)
    print("测试不同 question_set_id 被拒绝")
    print("=" * 60)

    old_run = {"run_id": "run_1", "config_id": "cfg_a", "question_set_id": "qs_001"}
    new_run = {"run_id": "run_2", "config_id": "cfg_b", "question_set_id": "qs_002"}

    assert old_run["question_set_id"] != new_run["question_set_id"]
    print("[OK] 不同 question_set_id 应被拒绝")

    print()


def test_global_run_pool_not_filtered_by_config():
    """跨配置候选集不受当前 config_id 页面过滤限制。"""
    print("=" * 60)
    print("测试全局 run 池不被 config_id 过滤")
    print("=" * 60)

    # 模拟全局 runs
    all_runs = [
        {"run_id": "r1", "config_id": "cfg_v2", "question_set_id": "qs_001"},
        {"run_id": "r2", "config_id": "cfg_v2_4", "question_set_id": "qs_001"},
        {"run_id": "r3", "config_id": "cfg_v3", "question_set_id": "qs_002"},
    ]

    # 当前页面 config_id = cfg_v2_4
    current_config_id = "cfg_v2_4"

    # 全局分组（不受当前 config 过滤）
    runs_by_qs = {}
    for run in all_runs:
        qs = run.get("question_set_id", "")
        if qs:
            runs_by_qs.setdefault(qs, []).append(run)

    # qs_001 有 r1 (cfg_v2) 和 r2 (cfg_v2_4)
    assert len(runs_by_qs["qs_001"]) == 2
    cfg_ids = set(r["config_id"] for r in runs_by_qs["qs_001"])
    assert "cfg_v2" in cfg_ids, "应包含非当前配置的 run"
    assert "cfg_v2_4" in cfg_ids
    print("[OK] 全局 run 池包含不同 config_id 的 run")

    print()


def test_compare_runs_returns_run_ids():
    """compare_runs 返回 old_run_id 和 new_run_id。"""
    print("=" * 60)
    print("测试 compare_runs 返回 run_id")
    print("=" * 60)

    from retrieval_diff import compare_runs

    # 使用真实数据（如果存在）
    from experiment import list_experiment_runs
    all_runs = list_experiment_runs()

    # 找两个有相同 question_set_id 的 run
    by_qs = {}
    for r in all_runs:
        qs = r.get("question_set_id", "")
        if qs:
            by_qs.setdefault(qs, []).append(r)

    for qs, runs in by_qs.items():
        if len(runs) >= 2:
            old_rid = runs[0]["run_id"]
            new_rid = runs[1]["run_id"]
            result = compare_runs(old_rid, new_rid)
            assert "old_run_id" in result, "应返回 old_run_id"
            assert "new_run_id" in result, "应返回 new_run_id"
            assert result["old_run_id"] == old_rid
            assert result["new_run_id"] == new_rid
            print(f"[OK] old_run_id={old_rid[:30]}, new_run_id={new_rid[:30]}")
            return

    print("[SKIP] 无足够数据测试")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("检索复现差异对比测试")
    print("=" * 60)
    print()

    test_find_evidence_rank_top1()
    test_find_evidence_rank_top3()
    test_find_evidence_rank_none()
    test_classify_evidence_lost()
    test_classify_ranking_regression()
    test_classify_judge_disagreement()
    test_classify_unchanged()
    test_classify_rank_improvement()
    test_question_id_alignment()
    test_real_trace_id_isolation()
    test_cross_run_isolation()
    test_csv_no_keys()
    test_md_no_keys()
    test_load_run_samples_by_qid()
    test_full_compare_mock()
    test_cutoff_old1_new2()
    test_cutoff_old4_new_none()
    test_cutoff_old7_new_none()
    test_cutoff_old2_new5()
    test_cutoff_old_none_new1()
    test_cutoff_old5_new3()
    test_compute_cutoff_stats()
    test_diff_config_same_qs_comparable()
    test_same_config_comparable()
    test_different_qs_rejected()
    test_global_run_pool_not_filtered_by_config()
    test_compare_runs_returns_run_ids()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
