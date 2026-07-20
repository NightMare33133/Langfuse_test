"""
build_judge_plan 测试。

验证：
1. 首条原始样本已成功、第二条待评时，快速测试必须选择第二条
2. 所有样本已成功时，快速测试和增量评测都不应发起 LLM
3. 新样本 74 条、历史成功 181 条时，增量计划恰好选中 74 条
4. 样本准备更新后，计划数量随新 samples 更新，不残留旧 max_samples
5. 轨道筛选后，快速/增量/失败重试均只作用于筛选范围
6. UI 预览的样本集合、预计 LLM 调用数与实际执行集合完全一致
7. 快速测试跳过规则预筛样本
8. 失败重试仅选中 error 样本
9. 强制全部包含所有样本
10. 内容去重正确计入 llm_count
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from app import build_judge_plan


# ====== 辅助函数 ======

def _make_sample(idx, question_mode="qa", has_retrieval=True, has_answer=True,
                 has_reference=False, has_source=False):
    """创建一个可评测的样本。"""
    s = {
        "trace_id": f"trace_{idx:04d}",
        "question": f"测试问题 {idx}",
        "question_mode": question_mode,
        "retrieval_query": f"查询 {idx}",
        "retrieval_results": (
            [{"position": 1, "score": 0.9, "content": f"结果内容 {idx}",
              "document_name": "doc"}]
            if has_retrieval else []
        ),
        "final_answer": f"测试回答 {idx}" if has_answer else "",
    }
    if has_reference:
        s["reference_answer"] = f"参考答案 {idx}"
    if has_source:
        s["source_excerpt"] = f"来源摘录 {idx}"
    return s


def _make_prescreen_sample(idx):
    """创建一个会被规则预筛的样本（无检索结果、无回答）。"""
    return {
        "trace_id": f"trace_{idx:04d}",
        "question": f"预筛问题 {idx}",
        "question_mode": "qa",
        "retrieval_query": "",
        "retrieval_results": [],
        "final_answer": "",
    }


def _success_result(trace_id):
    """模拟成功的评测结果。"""
    return {"trace_id": trace_id, "retrieval_top1_hit": 1, "reason": "ok"}


def _error_result(trace_id):
    """模拟失败的评测结果。"""
    return {"trace_id": trace_id, "error": "LLM timeout"}


# ====== 测试用例 ======

def test_quick_test_first_pending_when_first_is_successful():
    """首条已成功、第二条待评时，快速测试选择第二条。"""
    samples = [_make_sample(0), _make_sample(1)]
    existing = {"trace_0000": _success_result("trace_0000")}

    plan = build_judge_plan(samples, existing, "quick_test")

    assert len(plan["samples"]) == 1
    assert plan["samples"][0]["trace_id"] == "trace_0001"
    assert plan["selected_sample_preview"] is not None
    assert plan["selected_sample_preview"]["trace_id_suffix"] == "race_0001"[-8:]
    assert plan["new_count"] == 1
    assert plan["retry_count"] == 0


def test_quick_test_and_incremental_empty_when_all_successful():
    """所有样本已成功时，快速测试和增量都不应发起 LLM。"""
    samples = [_make_sample(0), _make_sample(1), _make_sample(2)]
    existing = {
        f"trace_{i:04d}": _success_result(f"trace_{i:04d}")
        for i in range(3)
    }

    for mode in ("quick_test", "incremental"):
        plan = build_judge_plan(samples, existing, mode)
        assert plan["samples"] == [], f"mode={mode} should return empty"
        assert plan["llm_count"] == 0
        assert plan["success_count"] == 3


def test_incremental_74_new_181_successful():
    """新样本 74 条、历史成功 181 条时，增量恰好选中 74 条。"""
    samples = [_make_sample(i) for i in range(255)]
    existing = {
        f"trace_{i:04d}": _success_result(f"trace_{i:04d}")
        for i in range(181)
    }

    plan = build_judge_plan(samples, existing, "incremental")

    assert len(plan["samples"]) == 74
    assert plan["new_count"] == 74
    assert plan["retry_count"] == 0
    assert plan["success_count"] == 181
    assert plan["total_filtered"] == 255


def test_plan_counts_update_after_sample_update():
    """样本列表变化后，计划数量随之更新，无残留旧值。"""
    samples_v1 = [_make_sample(i) for i in range(10)]
    existing = {
        f"trace_{i:04d}": _success_result(f"trace_{i:04d}")
        for i in range(8)
    }

    plan_v1 = build_judge_plan(samples_v1, existing, "incremental")
    assert len(plan_v1["samples"]) == 2
    assert plan_v1["total_filtered"] == 10

    # 模拟样本更新：新增 5 条
    samples_v2 = samples_v1 + [_make_sample(i) for i in range(10, 15)]
    plan_v2 = build_judge_plan(samples_v2, existing, "incremental")
    assert len(plan_v2["samples"]) == 7  # 2 old pending + 5 new
    assert plan_v2["total_filtered"] == 15
    assert plan_v2["success_count"] == 8


def test_track_filter_restricts_modes():
    """轨道筛选后，所有模式均只作用于筛选范围。"""
    # 5 retrieval + 5 grounded_qa
    retrieval_samples = [_make_sample(i, question_mode="retrieval", has_source=True)
                         for i in range(5)]
    qa_samples = [_make_sample(i + 100, question_mode="qa", has_reference=False)
                  for i in range(5)]
    all_samples = retrieval_samples + qa_samples

    # 前 3 条 retrieval 已成功
    existing = {
        f"trace_{i:04d}": _success_result(f"trace_{i:04d}")
        for i in range(3)
    }

    # 筛选 retrieval 轨道
    filtered = [s for s in all_samples if s["question_mode"] == "retrieval"]

    plan = build_judge_plan(filtered, existing, "incremental")
    assert len(plan["samples"]) == 2  # retrieval 3,4 pending
    assert plan["total_filtered"] == 5

    plan_quick = build_judge_plan(filtered, existing, "quick_test")
    assert len(plan_quick["samples"]) == 1
    assert plan_quick["samples"][0]["trace_id"] == "trace_0003"

    # 失败重试：在 retrieval 范围内加一个 error
    existing["trace_0004"] = _error_result("trace_0004")
    plan_retry = build_judge_plan(filtered, existing, "retry_failed")
    assert len(plan_retry["samples"]) == 1
    assert plan_retry["samples"][0]["trace_id"] == "trace_0004"


def test_ui_preview_matches_execution():
    """两次调用同一输入，结果完全一致（确定性）。"""
    samples = [_make_sample(i) for i in range(20)]
    existing = {
        f"trace_{i:04d}": _success_result(f"trace_{i:04d}")
        for i in range(10)
    }

    plan_a = build_judge_plan(samples, existing, "incremental")
    plan_b = build_judge_plan(samples, existing, "incremental")

    assert [s["trace_id"] for s in plan_a["samples"]] == \
           [s["trace_id"] for s in plan_b["samples"]]
    assert plan_a["llm_count"] == plan_b["llm_count"]
    assert plan_a["new_count"] == plan_b["new_count"]
    assert plan_a["retry_count"] == plan_b["retry_count"]


def test_quick_test_skips_prescreened_samples():
    """快速测试跳过规则预筛样本，只选需要 LLM 的。"""
    # 全部是会被预筛的样本
    prescreen_samples = [_make_prescreen_sample(i) for i in range(5)]
    plan = build_judge_plan(prescreen_samples, {}, "quick_test")
    assert plan["samples"] == []
    assert plan["llm_count"] == 0

    # 混合：前 3 条预筛，第 4 条需要 LLM
    mixed = [_make_prescreen_sample(i) for i in range(3)] + [_make_sample(3)]
    plan = build_judge_plan(mixed, {}, "quick_test")
    assert len(plan["samples"]) == 1
    assert plan["samples"][0]["trace_id"] == "trace_0003"


def test_retry_failed_only_errors():
    """失败重试仅选中 error 样本。"""
    samples = [_make_sample(i) for i in range(5)]
    existing = {
        "trace_0000": _success_result("trace_0000"),
        "trace_0001": _error_result("trace_0001"),
        "trace_0002": _error_result("trace_0002"),
        # trace_0003: never judged
        # trace_0004: never judged
    }

    plan = build_judge_plan(samples, existing, "retry_failed")
    assert len(plan["samples"]) == 2
    ids = {s["trace_id"] for s in plan["samples"]}
    assert ids == {"trace_0001", "trace_0002"}
    assert plan["retry_count"] == 2
    assert plan["new_count"] == 0


def test_force_all_includes_everything():
    """强制全部包含所有样本，无论历史状态。"""
    samples = [_make_sample(i) for i in range(10)]
    existing = {
        f"trace_{i:04d}": _success_result(f"trace_{i:04d}")
        for i in range(7)
    }

    plan = build_judge_plan(samples, existing, "force_all")
    assert len(plan["samples"]) == 10
    assert plan["success_count"] == 7
    assert plan["new_count"] == 3
    assert plan["retry_count"] == 0
    assert plan["total_filtered"] == 10


def test_content_dedup_in_llm_count():
    """内容相同样本去重后，llm_count 减少。"""
    # 两个样本内容完全相同但 trace_id 不同
    s1 = _make_sample(0)
    s2 = _make_sample(1)
    # 使内容完全相同
    s2["question"] = s1["question"]
    s2["retrieval_query"] = s1["retrieval_query"]
    s2["final_answer"] = s1["final_answer"]
    s2["retrieval_results"] = s1["retrieval_results"]

    plan = build_judge_plan([s1, s2], {}, "incremental")
    assert len(plan["samples"]) == 2  # 两条都选中
    assert plan["llm_count"] == 1  # 但只需 1 次 LLM 调用（去重）


def test_incremental_includes_error_retries():
    """增量模式包含失败重试。"""
    samples = [_make_sample(i) for i in range(5)]
    existing = {
        "trace_0000": _success_result("trace_0000"),
        "trace_0001": _error_result("trace_0001"),
    }

    plan = build_judge_plan(samples, existing, "incremental")
    assert len(plan["samples"]) == 4  # 1 error + 3 new
    assert plan["new_count"] == 3
    assert plan["retry_count"] == 1
    assert plan["success_count"] == 1


def test_quick_test_selects_error_retry_as_fallback():
    """快速测试：如果没有新样本但有失败样本，选择失败样本重试。"""
    samples = [_make_sample(0), _make_sample(1)]
    existing = {
        "trace_0000": _success_result("trace_0000"),
        "trace_0001": _error_result("trace_0001"),
    }

    plan = build_judge_plan(samples, existing, "quick_test")
    assert len(plan["samples"]) == 1
    assert plan["samples"][0]["trace_id"] == "trace_0001"
    assert plan["retry_count"] == 1
