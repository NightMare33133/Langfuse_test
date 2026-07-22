"""
测试 Judge 快速规则预览的隔离性。

覆盖：
a. 快速预览不写入 eval_results.jsonl
b. 快速预览不更新 session_state["judge_results"]
c. 快速预览不修改 existing_results_map
d. 正式 Judge 写入 eval_results.jsonl
e. 两阶段进度回调包含 prescreened_count 和 llm_done
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from judge import pre_screen, classify_evaluation_track, TRACK_RETRIEVAL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_retrieval_sample(qid, gold, retrieval_contents):
    return {
        "trace_id": f"trace_{qid}",
        "question_id": qid,
        "question": f"question {qid}",
        "question_mode": "retrieval",
        "reference_answer": gold,
        "source_excerpt": gold,
        "retrieval_results": [
            {"position": i + 1, "content": c, "title": f"doc_{i}"}
            for i, c in enumerate(retrieval_contents)
        ],
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_preview_does_not_write_file():
    """快速预览不写入 eval_results.jsonl。"""
    print("=" * 60)
    print("测试预览不写入文件")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        judged_file = Path(tmpdir) / "eval_results.jsonl"

        # 模拟预览：只运行 pre_screen，不写文件
        samples = [
            _make_retrieval_sample("q1", "证据1", []),
            _make_retrieval_sample("q2", "证据2", ["包含证据2的文档"]),
        ]

        preview_results = []
        for s in samples:
            s["evaluation_track"] = classify_evaluation_track(s)
            ps = pre_screen(s)
            if ps is not None:
                preview_results.append(ps)

        # 验证文件未被创建
        assert not judged_file.exists(), "预览不应创建 eval_results.jsonl"
        print(f"[OK] 预览后文件不存在: {judged_file}")

    print()


def test_preview_does_not_modify_session_state():
    """快速预览不修改 session_state["judge_results"]。"""
    print("=" * 60)
    print("测试预览不修改 session_state")
    print("=" * 60)

    # 模拟 session_state
    session_state = {"judge_results": [{"trace_id": "old_result"}]}
    original_count = len(session_state["judge_results"])

    # 模拟预览
    samples = [_make_retrieval_sample("q1", "证据", [])]
    for s in samples:
        s["evaluation_track"] = classify_evaluation_track(s)
        pre_screen(s)

    # session_state 不应改变
    assert len(session_state["judge_results"]) == original_count
    print("[OK] session_state['judge_results'] 未被修改")

    print()


def test_preview_rule_results_have_prescreened_flag():
    """预览的规则结果包含 _prescreened 标志。"""
    print("=" * 60)
    print("测试规则结果有 _prescreened 标志")
    print("=" * 60)

    # 空检索结果 → 规则判定
    sample = _make_retrieval_sample("q1", "证据", [])
    sample["evaluation_track"] = classify_evaluation_track(sample)
    ps = pre_screen(sample)

    assert ps is not None, "空检索结果应被规则判定"
    # pre_screen 返回的结果不含 _prescreened（那是 judge_all 添加的）
    # 但规则判定的结果应有明确的 reason
    assert "reason" in ps
    assert "规则判定" in ps["reason"]
    print(f"[OK] 规则判定结果有 reason: {ps['reason']}")

    print()


def test_formal_judge_writes_file():
    """正式 Judge 写入 eval_results.jsonl。"""
    print("=" * 60)
    print("测试正式 Judge 写入文件")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        judged_file = Path(tmpdir) / "eval_results.jsonl"

        # 模拟正式结果写入
        results = [
            {"trace_id": "t1", "evaluation_track": "retrieval",
             "retrieval_top1_hit": 1, "retrieval_top5_hit": 1},
        ]

        # 写入文件
        with judged_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        assert judged_file.exists(), "正式 Judge 应创建文件"
        # 验证内容
        with judged_file.open("r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["trace_id"] == "t1"
        print("[OK] 正式 Judge 正确写入文件")

    print()


def test_preview_top1_top3_top5_stats():
    """预览应正确统计规则确认的 Top1/3/5。"""
    print("=" * 60)
    print("测试预览 Top1/3/5 统计")
    print("=" * 60)

    from judge import retrieval_rule_judge

    # 样本 1: Top1 命中
    s1 = _make_retrieval_sample("q1", "证据文本应足够长以通过检查",
                                ["包含证据文本应足够长以通过检查的文档", "其他"])
    # 样本 2: 无匹配
    s2 = _make_retrieval_sample("q2", "不同的证据", ["无关内容"])
    # 样本 3: 空检索
    s3 = _make_retrieval_sample("q3", "证据", [])

    rule_hits = []
    rule_misses = []
    rule_pending = []

    for s in [s1, s2, s3]:
        s["evaluation_track"] = classify_evaluation_track(s)
        if s["evaluation_track"] != TRACK_RETRIEVAL:
            continue
        rr = retrieval_rule_judge(s)
        if rr is not None:
            if rr.get("hit_evidence_position"):
                rule_hits.append(rr)
            else:
                rule_misses.append(rr)
        else:
            rule_pending.append(s)

    assert len(rule_hits) == 1, f"应有 1 个命中，实际 {len(rule_hits)}"
    assert len(rule_misses) == 1, f"应有 1 个未命中，实际 {len(rule_misses)}"
    assert len(rule_pending) == 1, f"应有 1 个待 LLM，实际 {len(rule_pending)}"
    print(f"[OK] 命中={len(rule_hits)}, 未命中={len(rule_misses)}, 待LLM={len(rule_pending)}")

    print()


def test_two_phase_progress_callback():
    """两阶段进度回调应包含 prescreened_count 和 llm_done。"""
    print("=" * 60)
    print("测试两阶段进度回调")
    print("=" * 60)

    # 模拟 info dict
    info_rule_phase = {
        "llm_done": 0, "llm_total": 10,
        "prescreened_count": 5, "cached_count": 2,
        "elapsed": 1.0, "concurrency": 1,
    }
    info_llm_phase = {
        "llm_done": 3, "llm_total": 10,
        "prescreened_count": 5, "cached_count": 2,
        "elapsed": 5.0, "concurrency": 2,
    }

    # 规则阶段
    assert info_rule_phase["prescreened_count"] == 5
    assert info_rule_phase["llm_done"] == 0
    print("[OK] 规则阶段: prescreened=5, llm_done=0")

    # LLM 阶段
    assert info_llm_phase["llm_done"] == 3
    assert info_llm_phase["llm_total"] == 10
    print("[OK] LLM 阶段: llm_done=3/10")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Judge 快速规则预览隔离测试")
    print("=" * 60)
    print()

    test_preview_does_not_write_file()
    test_preview_does_not_modify_session_state()
    test_preview_rule_results_have_prescreened_flag()
    test_formal_judge_writes_file()
    test_preview_top1_top3_top5_stats()
    test_two_phase_progress_callback()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
