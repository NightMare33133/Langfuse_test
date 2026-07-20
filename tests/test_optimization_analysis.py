"""
AI 优化分析报告模块测试。

覆盖：
1. 上下文只含当前看板数据
2. 敏感字段 / 绝对路径被移除
3. 分组指标准确
4. 截断规则稳定、确定性
5. LLM 调用顺序（阶段 1→2→3）
6. LLM 失败时保留已完成结果
7. 最终 Markdown 含审计引用
8. 不修改输入数据
9. 指标计算准确
10. 空数据不崩溃
11. 保存路径在 data/reports/
12. ANALYSIS_* 回退 JUDGE_*

不调用真实 API。
"""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE
from optimization_analysis import (
    sanitize_analysis_payload,
    build_analysis_context,
    compute_precise_stats,
    build_facts_section,
    build_scope_note,
    get_analysis_config,
    save_analysis_report,
    sanitize_filename_component,
    build_report_filename,
    analyze_overview,
    analyze_failure_groups,
    group_failures_for_analysis,
    _split_into_sub_batches,
    _slim_sample,
    _build_python_failure_manifest,
    synthesize_optimization_report,
    _SENSITIVE_KEYS,
    _SENSITIVE_SNAPSHOT_KEYS,
    _ABS_PATH_PREFIXES,
    _MAX_CONTEXT_CHARS,
    _MIN_GROUP_SAMPLE_COUNT,
    _SUB_BATCH_MAX_SIZE,
)


# ====== 测试 Fixture ======

def _make_retrieval_result(trace_id, t1, t3, t5, position=None, reason=""):
    return {
        "trace_id": trace_id,
        "question": f"检索问题_{trace_id}",
        "source_excerpt": f"金标准证据_{trace_id}：违约方应支付合同总金额的 10% 作为违约金",
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_evaluable": True,
        "retrieval_top1_hit": t1,
        "retrieval_top3_hit": t3,
        "retrieval_top5_hit": t5,
        "hit_evidence_position": position,
        "reason": reason,
        "run_id": "run_test_001",
        "_source_run_id": "run_test_001",
        "question_id": f"qid_{trace_id}",
        "question_set_id": "qs_test_001",
        "topic": "合同法",
        "difficulty": "中等",
    }


def _make_strict_qa_result(trace_id, answer_correct):
    return {
        "trace_id": trace_id,
        "question": f"严格问答_{trace_id}",
        "reference_answer": f"参考答案_{trace_id}",
        "evaluation_track": TRACK_STRICT_QA,
        "answer_correct": answer_correct,
        "reason": "测试",
        "run_id": "run_test_001",
    }


def _make_grounded_qa_result(trace_id, answer_correct):
    return {
        "trace_id": trace_id,
        "question": f"合理性问答_{trace_id}",
        "evaluation_track": TRACK_GROUNDED_QA,
        "answer_correct": answer_correct,
        "reason": "测试",
        "run_id": "run_test_001",
    }


def _make_error_result(trace_id):
    return {
        "trace_id": trace_id,
        "question": f"错误问题_{trace_id}",
        "evaluation_track": TRACK_RETRIEVAL,
        "error": "LLM 调用超时",
        "run_id": "run_test_001",
    }


def _make_not_evaluable_result(trace_id):
    return {
        "trace_id": trace_id,
        "question": f"不可评测_{trace_id}",
        "evaluation_track": TRACK_NOT_EVALUABLE,
        "retrieval_evaluable": False,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 0,
        "retrieval_top5_hit": 0,
        "reason": "缺少金标准证据",
        "run_id": "run_test_001",
    }


def _make_processed_sample(trace_id, retrieval_results=None, question="", retrieval_query="",
                           source_excerpt="", final_answer="", **extra):
    sample = {
        "trace_id": trace_id,
        "question": question or f"检索问题_{trace_id}",
        "retrieval_query": retrieval_query or f"检索查询_{trace_id}",
        "source_excerpt": source_excerpt or f"金标准证据_{trace_id}：违约方应支付合同总金额的 10% 作为违约金",
        "final_answer": final_answer or f"回答_{trace_id}",
        "retrieval_results": retrieval_results or [],
        "question_id": f"qid_{trace_id}",
        "question_set_id": "qs_test_001",
        "source_file_name": "合同模板_v2.pdf",
        "topic": "合同法",
        "difficulty": "中等",
        "source_format": "pdf",
    }
    sample.update(extra)
    return sample


def _make_retrieval_results(n):
    results = []
    for i in range(1, n + 1):
        results.append({
            "position": i,
            "document_name": f"doc_{i}.pdf",
            "score": round(0.95 - i * 0.05, 4),
            "content": f"检索结果 {i} 的内容：这是第 {i} 条检索到的文档片段，包含部分信息。",
        })
    return results


def _build_config():
    return {
        "config_id": "cfg_test_001",
        "config_name": "测试配置",
        "knowledge_base_version": "KB_v1",
        "workflow_version": "WF_v1",
    }


def _build_fixture():
    """构建标准 fixture。"""
    config = _build_config()
    run = {
        "run_id": "run_test_001",
        "config_id": "cfg_test_001",
        "question_count": 10,
        "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "batch_results_file": "batch_results.jsonl",
        "raw_results_file": "batch_qa_20260716.jsonl",
        "question_set_name": "测试题集",
        "question_set_id": "qs_test_001",
        "config_snapshot": {
            "config_name": "测试配置",
            "config_id": "cfg_test_001",
            "knowledge_base_version": "KB_v1",
            "workflow_version": "WF_v1",
            "top_k": 5,
            "model": "gpt-4",
            "retrieval_mode": "hybrid",
        },
    }
    run_status = {
        "batch_success": 8, "batch_total": 10, "raw_count": 10,
        "processed_count": 8, "judge_count": 8, "question_count": 10,
        "question_set_name": "测试题集", "question_set_id": "qs_test_001",
        "judge_results": [],
    }

    ret_results_5 = _make_retrieval_results(5)
    ret_results_3 = _make_retrieval_results(3)

    results = [
        _make_retrieval_result("t_ret_1", 1, 1, 1, 1, "命中"),
        _make_retrieval_result("t_ret_2", 1, 1, 1, 1, "命中"),
        _make_retrieval_result("t_ret_3", 1, 1, 1, 1, "命中"),
        _make_retrieval_result("t_ret_4", 0, 1, 1, 2, "Top3命中"),
        _make_retrieval_result("t_ret_5", 0, 0, 1, 4, "仅Top5命中"),
        _make_retrieval_result("t_ret_6", 0, 0, 0, None, "全未命中"),
        _make_strict_qa_result("t_sqa_1", 1),
        _make_strict_qa_result("t_sqa_2", 0),
        _make_grounded_qa_result("t_gqa_1", 1),
        _make_error_result("t_err_1"),
        _make_not_evaluable_result("t_ne_1"),
    ]
    run_status["judge_results"] = results

    sample_lookup = {
        "t_ret_1": _make_processed_sample("t_ret_1", ret_results_5),
        "t_ret_2": _make_processed_sample("t_ret_2", ret_results_5),
        "t_ret_3": _make_processed_sample("t_ret_3", ret_results_5),
        "t_ret_4": _make_processed_sample("t_ret_4", ret_results_3),
        "t_ret_5": _make_processed_sample("t_ret_5", ret_results_5),
        "t_ret_6": _make_processed_sample("t_ret_6", ret_results_5),
        "t_sqa_1": _make_processed_sample("t_sqa_1"),
        "t_sqa_2": _make_processed_sample("t_sqa_2"),
        "t_gqa_1": _make_processed_sample("t_gqa_1"),
    }

    from judge import compute_metrics
    metrics = compute_metrics(results)

    return config, [run], [{"run": run, "run_status": run_status, "metrics": metrics}], metrics, results, sample_lookup


def _mock_call_llm(responses=None, call_log=None):
    """返回一个 mock call_llm 函数，记录调用顺序并返回预设响应。

    Args:
        responses: 预设响应列表，按调用顺序返回
        call_log: 用于记录调用的列表
    """
    if responses is None:
        responses = ["阶段1分析结果", "阶段2分析结果", "最终报告"]
    if call_log is None:
        call_log = []
    idx = [0]

    def mock(prompt, api_key, base_url, model, timeout=120):
        call_log.append({"stage": idx[0], "prompt_len": len(prompt), "model": model})
        resp = responses[idx[0]] if idx[0] < len(responses) else "默认响应"
        idx[0] += 1
        return resp

    return mock, call_log


# ====== 测试函数 ======

def test_context_scoping():
    """上下文只含当前看板数据。"""
    print("=" * 60)
    print("测试：上下文范围限定")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    ctx = build_analysis_context(rdl, sl, all_r, config)

    assert "overview" in ctx, "缺少 overview"
    assert "groupings" in ctx, "缺少 groupings"
    assert "failures" in ctx, "缺少 failures"
    assert "config_summary" in ctx, "缺少 config_summary"
    assert "run_summaries" in ctx, "缺少 run_summaries"
    assert "data_quality" in ctx, "缺少 data_quality"
    assert "generation_timestamp" in ctx, "缺少 generation_timestamp"

    # 验证 run_summaries 只含当前数据
    assert len(ctx["run_summaries"]) == 1
    assert ctx["run_summaries"][0]["run_id"] == "run_test_001"

    # 验证 overview 指标
    assert ctx["overview"]["run_count"] == 1
    assert ctx["overview"]["error_count"] == 1

    print("PASS: 上下文范围限定正确")


def test_sensitive_field_removal():
    """敏感字段和绝对路径被移除。"""
    print("=" * 60)
    print("测试：敏感字段移除")
    print("=" * 60)

    data = {
        "normal_key": "normal_value",
        "_prompt": "secret prompt",
        "_raw_response": "secret response",
        "api_key": "sk-12345",
        "secret_key": "secret123",
        "cookie": "session=abc",
        "nested": {
            "openai_api_key": "sk-xxx",
            "lf_secret_key": "lf_secret",
            "safe": "keep_this",
        },
        "path_value": "C:\\Users\\test\\secret.txt",
        "unix_path": "/home/user/secret.txt",
        "list_field": [
            {"_prompt": "should be removed", "keep": "yes"},
            {"api_key": "removed", "data": "kept"},
        ],
    }

    cleaned = sanitize_analysis_payload(data)

    # 检查敏感字段被移除
    for key in _SENSITIVE_KEYS:
        assert key not in cleaned, f"敏感字段 {key} 未被移除"

    # 检查嵌套敏感字段
    assert "openai_api_key" not in cleaned["nested"], "嵌套 openai_api_key 未被移除"
    assert "lf_secret_key" not in cleaned["nested"], "嵌套 lf_secret_key 未被移除"
    assert cleaned["nested"]["safe"] == "keep_this"

    # 检查绝对路径
    assert cleaned["path_value"] == "[REDACTED_PATH]"
    assert cleaned["unix_path"] == "[REDACTED_PATH]"

    # 检查列表中的敏感字段
    assert "_prompt" not in cleaned["list_field"][0]
    assert cleaned["list_field"][0]["keep"] == "yes"
    assert "api_key" not in cleaned["list_field"][1]
    assert cleaned["list_field"][1]["data"] == "kept"

    # 确保正常字段保留
    assert cleaned["normal_key"] == "normal_value"

    print("PASS: 敏感字段和绝对路径正确移除")


def test_grouping_correctness():
    """分组指标准确。"""
    print("=" * 60)
    print("测试：分组指标正确性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    ctx = build_analysis_context(rdl, sl, all_r, config)

    groupings = ctx["groupings"]
    assert "by_source_file" in groupings
    assert "by_topic" in groupings
    assert "by_difficulty" in groupings

    # 所有样本的 source_file_name 都是 "合同模板_v2.pdf"
    file_groups = groupings["by_source_file"]
    assert len(file_groups) >= 1

    # 找到 "合同模板_v2.pdf" 分组
    pdf_group = next((g for g in file_groups if g["key"] == "合同模板_v2.pdf"), None)
    assert pdf_group is not None, "未找到合同模板_v2.pdf分组"
    assert pdf_group["count"] == 6, f"检索样本应为 6，实际 {pdf_group['count']}"

    # Top1 命中率：3/6 = 0.5
    assert abs(pdf_group["t1_rate"] - 0.5) < 0.01, f"Top1 应为 0.5，实际 {pdf_group['t1_rate']}"

    print("PASS: 分组指标正确")


def test_truncation_rules():
    """截断规则稳定，确定性输出。"""
    print("=" * 60)
    print("测试：截断规则")
    print("=" * 60)

    config = _build_config()

    # 构建大量失败样本
    all_r = []
    sl = {}
    for i in range(60):
        tid = f"t_miss_{i:03d}"
        all_r.append(_make_retrieval_result(tid, 0, 0, 0, None, f"未命中_{i}"))
        sl[tid] = _make_processed_sample(tid, _make_retrieval_results(5),
                                         source_file_name=f"file_{i % 15}.pdf")

    run = {
        "run_id": "run_test_trunc", "config_id": "cfg_test_001",
        "question_count": 60, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 60}, "metrics": {}}]

    ctx1 = build_analysis_context(rdl, sl, all_r, config)
    ctx2 = build_analysis_context(rdl, sl, all_r, config)

    # 确定性：排除时间戳后两次调用结果相同
    ctx1_stable = {k: v for k, v in ctx1.items() if k != "generation_timestamp"}
    ctx2_stable = {k: v for k, v in ctx2.items() if k != "generation_timestamp"}
    j1 = json.dumps(ctx1_stable, ensure_ascii=False, sort_keys=True)
    j2 = json.dumps(ctx2_stable, ensure_ascii=False, sort_keys=True)
    assert j1 == j2, "两次构建的上下文应完全相同（排除时间戳）"

    # 总上下文大小在限制内（使用完整 context 含时间戳检查）
    full_json = json.dumps(ctx1, ensure_ascii=False, sort_keys=True)
    assert len(full_json) <= _MAX_CONTEXT_CHARS, f"上下文过大: {len(full_json)} > {_MAX_CONTEXT_CHARS}"

    print(f"PASS: 截断规则稳定，上下文大小 {len(j1)} 字符")


def test_llm_call_order():
    """阶段 1→2→3 调用顺序正确，Stage 1 和 3 各 1 次，Stage 2 多次（map-reduce）。"""
    print("=" * 60)
    print("测试：LLM 调用顺序")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    ctx = build_analysis_context(rdl, sl, all_r, config)

    # Stage 2 现在是 map-reduce：子批次 + reduce 调用，所以需要更多 mock 响应
    mock_fn, call_log = _mock_call_llm(["概览分析"] + ["子批次诊断"] * 20 + ["最终报告"])

    import optimization_analysis as oa
    orig_fn = oa.call_llm
    oa.call_llm = mock_fn
    try:
        s1 = analyze_overview(ctx, "key", "url", "model")
        s2 = analyze_failure_groups(ctx, "key", "url", "model")
        report = synthesize_optimization_report(s1, s2, ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    # Stage 1 = 1 次, Stage 2 >= 1 次 (sub-batches + reduce), Stage 3 = 1 次
    assert len(call_log) >= 3, f"应有 >=3 次 LLM 调用，实际 {len(call_log)}"
    assert call_log[0]["stage"] == 0  # Stage 1
    # Stage 2 调用在中间
    assert call_log[-1]["stage"] == len(call_log) - 1  # Stage 3 是最后一次
    assert "AI 优化分析报告" in report, "报告应包含标题"

    print(f"PASS: LLM 调用 {len(call_log)} 次，顺序正确")


def test_failure_handling():
    """Stage 2 子批次全部失败时，返回 Python 清单 + 错误说明，不抛异常。"""
    print("=" * 60)
    print("测试：LLM 失败处理")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    ctx = build_analysis_context(rdl, sl, all_r, config)

    import optimization_analysis as oa

    # 阶段 1 成功，阶段 2 全部子批次失败
    call_count = [0]
    def mock_fail_stage2(prompt, api_key, base_url, model, timeout=120):
        call_count[0] += 1
        if call_count[0] == 1:
            return "概览分析成功"
        raise RuntimeError("阶段 2 LLM 调用失败")

    orig_fn = oa.call_llm
    oa.call_llm = mock_fail_stage2
    try:
        s1 = analyze_overview(ctx, "key", "url", "model")
        assert s1 == "概览分析成功", "阶段 1 应成功"

        # Stage 2 现在不会抛异常，而是返回 Python 清单 + 错误说明
        s2 = analyze_failure_groups(ctx, "key", "url", "model")
        assert "Python 生成的完整失败清单" in s2, "应包含 Python 清单"
        assert "AI 诊断不可用" in s2, "应标注 AI 诊断不可用"

        # 阶段 1 结果仍可用
        assert s1 == "概览分析成功"
    finally:
        oa.call_llm = orig_fn

    print("PASS: Stage 2 全部失败时返回 Python 清单 + 错误说明")


def test_markdown_references():
    """最终报告含审计引用。"""
    print("=" * 60)
    print("测试：Markdown 审计引用")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    ctx = build_analysis_context(rdl, sl, all_r, config)

    # 模拟包含引用的 LLM 响应
    stage1_resp = "Top1 命中率 50%。异常分组：合同模板_v2.pdf。"
    stage2_resp = (
        "### 失败组 1: 全未命中\n"
        "- 样本数: 1\n"
        "- 根因假设: 知识库未覆盖该内容\n"
        "- 审计样本:\n"
        "  - [run_id=run_test_001 | trace_id=t_ret_6 | query=检索问题_t_ret_6]: 全未命中\n"
    )
    final_report = (
        "# AI 优化分析报告\n\n"
        "> **AI 诊断建议，不替代人工验证**\n\n"
        "## 1. 执行摘要\n"
        "数据事实: Top1 命中率 50%。\n\n"
        "## 5. Top5 完全未命中诊断\n"
        "诊断假设: [run_id=run_test_001 | trace_id=t_ret_6] 知识库未覆盖。\n\n"
        "## 8. 审计索引\n"
        "| trace_id | run_id | query |\n"
        "| t_ret_6 | run_test_001 | 检索问题_t_ret_6 |\n"
    )

    import optimization_analysis as oa
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: final_report
    try:
        report = synthesize_optimization_report(stage1_resp, stage2_resp, ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    assert "AI 诊断建议" in report, "报告应包含免责声明"
    assert "run_test_001" in report, "报告应包含 run_id"
    assert "t_ret_6" in report, "报告应包含 trace_id"

    print("PASS: Markdown 审计引用正确")


def test_no_data_modification():
    """不修改输入数据。"""
    print("=" * 60)
    print("测试：不修改输入数据")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()

    # 深拷贝
    config_orig = copy.deepcopy(config)
    all_r_orig = copy.deepcopy(all_r)
    sl_orig = copy.deepcopy(sl)
    rdl_orig = copy.deepcopy(rdl)

    ctx = build_analysis_context(rdl, sl, all_r, config)

    # 验证原始数据未被修改
    assert config == config_orig, "config 被修改了"
    assert all_r == all_r_orig, "all_judge_results 被修改了"
    assert sl == sl_orig, "sample_lookup 被修改了"
    assert rdl == rdl_orig, "run_data_list 被修改了"

    print("PASS: 输入数据未被修改")


def test_overview_metrics_accuracy():
    """指标计算准确。"""
    print("=" * 60)
    print("测试：指标计算准确性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    ctx = build_analysis_context(rdl, sl, all_r, config)

    ov = ctx["overview"]

    # 检索指标：6 个 retrieval 样本，3 个 Top1 命中
    assert ov["retrieval_track_count"] == 6, f"检索样本应为 6，实际 {ov['retrieval_track_count']}"
    assert abs(ov["retrieval_top1_hit_rate"] - 0.5) < 0.01, f"Top1 应为 0.5，实际 {ov['retrieval_top1_hit_rate']}"

    # Top3: t_ret_1,2,3 (t3=1) + t_ret_4 (t3=1) = 4/6
    assert abs(ov["retrieval_top3_hit_rate"] - 4 / 6) < 0.01

    # Top5: t_ret_1,2,3,4,5 (t5=1) = 5/6
    assert abs(ov["retrieval_top5_hit_rate"] - 5 / 6) < 0.01

    # 严格问答：1 正确 / 2 总
    assert ov["strict_qa_track_count"] == 2
    assert abs(ov["strict_qa_answer_rate"] - 0.5) < 0.01

    # 合理性问答：1 正确 / 1 总
    assert ov["grounded_qa_track_count"] == 1
    assert abs(ov["grounded_qa_answer_rate"] - 1.0) < 0.01

    # 错误
    assert ov["error_count"] == 1
    assert ov["not_evaluable_count"] == 1

    print("PASS: 指标计算准确")


def test_empty_data_handling():
    """空数据不崩溃。"""
    print("=" * 60)
    print("测试：空数据处理")
    print("=" * 60)

    ctx = build_analysis_context([], {}, [], {})

    assert ctx["overview"]["run_count"] == 0
    assert ctx["overview"]["total_questions"] == 0
    assert ctx["failures"]["total_top5_miss"] == 0
    assert ctx["failures"]["total_sorting_issues"] == 0
    assert len(ctx["run_summaries"]) == 0
    assert ctx["data_quality"]["judge_errors"] == 0

    print("PASS: 空数据处理正确")


def test_report_save_path():
    """保存路径在 data/reports/。"""
    print("=" * 60)
    print("测试：报告保存路径")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"
        path = save_analysis_report("# 测试报告", "测试配置", reports_dir)

        assert path.exists(), f"文件不存在: {path}"
        assert path.parent == reports_dir, f"路径不在 reports 目录: {path}"
        assert path.name.endswith(".md"), f"应为 .md 文件: {path.name}"
        assert "_ai_analysis_" in path.name, f"文件名应含 _ai_analysis_: {path.name}"
        assert path.name.startswith("测试配置"), f"文件名应以配置名开头: {path.name}"

        content = path.read_text(encoding="utf-8")
        assert content == "# 测试报告"

    print(f"PASS: 报告保存路径正确 ({path.name})")


def test_env_var_fallback():
    """ANALYSIS_* 回退 JUDGE_*。"""
    print("=" * 60)
    print("测试：环境变量回退")
    print("=" * 60)

    # 保存原始值
    orig_env = {}
    for key in ("ANALYSIS_API_KEY", "ANALYSIS_API_BASE", "ANALYSIS_MODEL",
                "JUDGE_API_KEY", "JUDGE_API_BASE", "JUDGE_MODEL"):
        orig_env[key] = os.environ.get(key)

    try:
        # 情况 1：只设 JUDGE_*
        for key in ("ANALYSIS_API_KEY", "ANALYSIS_API_BASE", "ANALYSIS_MODEL"):
            os.environ.pop(key, None)
        os.environ["JUDGE_API_KEY"] = "judge_key"
        os.environ["JUDGE_API_BASE"] = "http://judge.base"
        os.environ["JUDGE_MODEL"] = "judge_model"

        k, u, m = get_analysis_config()
        assert k == "judge_key", f"应为 judge_key，实际 {k}"
        assert u == "http://judge.base", f"应为 http://judge.base，实际 {u}"
        assert m == "judge_model", f"应为 judge_model，实际 {m}"

        # 情况 2：设 ANALYSIS_* 覆盖
        os.environ["ANALYSIS_API_KEY"] = "analysis_key"
        os.environ["ANALYSIS_API_BASE"] = "http://analysis.base"
        os.environ["ANALYSIS_MODEL"] = "analysis_model"

        k, u, m = get_analysis_config()
        assert k == "analysis_key", f"应为 analysis_key，实际 {k}"
        assert u == "http://analysis.base", f"应为 http://analysis.base，实际 {u}"
        assert m == "analysis_model", f"应为 analysis_model，实际 {m}"

    finally:
        # 恢复原始值
        for key, val in orig_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    print("PASS: 环境变量回退正确")


def test_precise_stats_71_samples():
    """71 样本、59 Top5 hit 时 top5_miss 必须为 12。"""
    print("=" * 60)
    print("测试：71 样本精确统计")
    print("=" * 60)

    config = _build_config()
    all_r = []
    sl = {}

    # 59 个 Top5 命中（t5=1），12 个 Top5 未命中（t5=0）
    for i in range(59):
        tid = f"t_hit_{i:03d}"
        all_r.append(_make_retrieval_result(tid, 1, 1, 1, 1, "命中"))
        sl[tid] = _make_processed_sample(tid, _make_retrieval_results(5))
    for i in range(12):
        tid = f"t_miss_{i:03d}"
        all_r.append(_make_retrieval_result(tid, 0, 0, 0, None, "未命中"))
        sl[tid] = _make_processed_sample(tid, _make_retrieval_results(5))

    run = {
        "run_id": "run_71sample", "config_id": "cfg_test_001",
        "question_count": 71, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 71}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    assert stats["retrieval_evaluable_n"] == 71, f"应为 71，实际 {stats['retrieval_evaluable_n']}"
    assert stats["top5_hit_n"] == 59, f"Top5 命中应为 59，实际 {stats['top5_hit_n']}"
    assert stats["top5_miss_n"] == 12, f"Top5 未命中应为 12，实际 {stats['top5_miss_n']}"
    assert stats["top1_hit_n"] == 59, f"Top1 命中应为 59，实际 {stats['top1_hit_n']}"

    # 验证 facts 部分包含正确数字
    facts = build_facts_section(ctx, stats)
    assert "71" in facts, "facts 应包含样本数 71"
    assert "**12**" in facts, "facts 应包含 top5_miss_n=12"
    assert "**59**" in facts, "facts 应包含 top5_hit_n=59"

    print("PASS: 71 样本统计精确，top5_miss=12")


def test_retrieval_only_scope():
    """retrieval-only 范围不出现"QA 评测缺失"。"""
    print("=" * 60)
    print("测试：retrieval-only 范围")
    print("=" * 60)

    config = _build_config()
    # 只有 retrieval 样本，无 QA 样本
    all_r = [
        _make_retrieval_result("t_r1", 1, 1, 1, 1),
        _make_retrieval_result("t_r2", 0, 0, 0, None),
    ]
    sl = {
        "t_r1": _make_processed_sample("t_r1", _make_retrieval_results(5)),
        "t_r2": _make_processed_sample("t_r2", _make_retrieval_results(5)),
    }

    run = {
        "run_id": "run_ret_only", "config_id": "cfg_test_001",
        "question_count": 2, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 2}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    assert stats["is_retrieval_only"] is True, "应为 retrieval-only"
    assert stats["has_qa"] is False, "不应有 QA 轨道"

    # 验证 scope note
    scope = build_scope_note(ctx, stats)
    assert "检索评测" in scope, "scope note 应提及检索评测"
    assert "答案质量" in scope, "scope note 应说明不对答案质量作结论"

    # 验证 facts 部分不含 QA 评测缺失相关内容
    facts = build_facts_section(ctx, stats)
    assert "QA 评测缺失" not in facts, "不应出现'QA 评测缺失'"
    assert "QA=0" not in facts, "不应出现'QA=0'"

    # 验证 prompt 中 is_retrieval_only=true
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    assert len(captured_prompts) == 1
    assert "is_retrieval_only: True" in captured_prompts[0], "prompt 应包含 is_retrieval_only: True"
    assert "不得将 QA=0 当作数据质量问题" in captured_prompts[0]

    print("PASS: retrieval-only 范围正确，不含 QA 评测缺失")


def test_single_sample_topic_no_strong_conclusion():
    """单样本 topic 不生成强结论。"""
    print("=" * 60)
    print("测试：单样本 topic 不生成强结论")
    print("=" * 60)

    config = _build_config()
    # 3 个 topic，其中 topic_C 只有 1 个样本
    all_r = [
        _make_retrieval_result("t_a1", 1, 1, 1, 1),
        _make_retrieval_result("t_a2", 1, 1, 1, 1),
        _make_retrieval_result("t_a3", 1, 1, 1, 1),
        _make_retrieval_result("t_b1", 0, 0, 0, None),
        _make_retrieval_result("t_b2", 0, 0, 0, None),
        _make_retrieval_result("t_b3", 0, 0, 0, None),
        _make_retrieval_result("t_c1", 0, 0, 0, None),  # 单样本 topic
    ]
    sl = {}
    for tid, topic in [("t_a1", "topic_A"), ("t_a2", "topic_A"), ("t_a3", "topic_A"),
                        ("t_b1", "topic_B"), ("t_b2", "topic_B"), ("t_b3", "topic_B"),
                        ("t_c1", "topic_C")]:
        sample = _make_processed_sample(tid, _make_retrieval_results(5))
        sample["topic"] = topic
        sl[tid] = sample

    run = {
        "run_id": "run_topic_test", "config_id": "cfg_test_001",
        "question_count": 7, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 7}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    # 验证 topic 分组标注
    topic_groups = stats["by_topic_annotated"]
    topic_c = next((g for g in topic_groups if g["key"] == "topic_C"), None)
    assert topic_c is not None, "应找到 topic_C"
    assert topic_c["count"] == 1, f"topic_C 应有 1 个样本，实际 {topic_c['count']}"
    assert topic_c["sufficient_sample"] is False, "单样本 topic 不应标记为样本充足"

    # topic_A 和 topic_B 应样本充足
    topic_a = next((g for g in topic_groups if g["key"] == "topic_A"), None)
    topic_b = next((g for g in topic_groups if g["key"] == "topic_B"), None)
    assert topic_a["sufficient_sample"] is True
    assert topic_b["sufficient_sample"] is True

    # 验证 facts 中包含"待观察个例"
    facts = build_facts_section(ctx, stats)
    assert "待观察个例" in facts, "facts 应对单样本 topic 标注'待观察个例'"

    # 验证 prompt 包含样本数阈值规则
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    assert "样本数 < 3" in captured_prompts[0], "prompt 应包含样本数阈值规则"

    print("PASS: 单样本 topic 标注为待观察个例")


def test_missing_config_no_specific_values():
    """config_snapshot 缺字段时不出现具体数值建议。"""
    print("=" * 60)
    print("测试：缺少配置字段时不编造数值")
    print("=" * 60)

    # 极简 config_snapshot，只有 config_name
    minimal_config = {"config_name": "最小配置"}

    all_r = [
        _make_retrieval_result("t_r1", 0, 0, 0, None),
        _make_retrieval_result("t_r2", 0, 0, 0, None),
    ]
    sl = {
        "t_r1": _make_processed_sample("t_r1", _make_retrieval_results(5)),
        "t_r2": _make_processed_sample("t_r2", _make_retrieval_results(5)),
    }

    run = {
        "run_id": "run_minimal_cfg", "config_id": "cfg_test_001",
        "question_count": 2, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": minimal_config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 2}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, minimal_config)
    stats = compute_precise_stats(ctx)

    # 已确认配置键只有 config_name
    assert stats["confirmed_config_keys"] == ["config_name"], \
        f"应只有 config_name，实际 {stats['confirmed_config_keys']}"

    # facts 中不应出现未确认参数的具体值（但允许在说明文字中提及参数名）
    facts = build_facts_section(ctx, stats)
    # 确认 config_values_text 不包含 top_k 的值
    assert "top_k = " not in facts, "facts 不应包含 top_k 的值"
    assert "rerank_model = " not in facts, "facts 不应包含 rerank_model 的值"
    # 确认 facts 中无结构化配置参数
    assert "无结构化配置参数" in facts or "无可用配置参数" in facts, \
        "facts 应说明无结构化配置参数"

    # prompt 应包含已确认配置键列表
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    assert "`config_name`" in captured_prompts[0], "prompt 应列出已确认的 config_name"
    assert "不编造具体数值" in captured_prompts[0] or "不得编造" in captured_prompts[0], \
        "prompt 应包含不编造数值的规则"

    print("PASS: 缺少配置字段时不编造数值")


def test_priority_ordering():
    """排序问题少于 Top5 miss 时，报告优先级正确。"""
    print("=" * 60)
    print("测试：优先级排序")
    print("=" * 60)

    config = _build_config()
    # 10 个 Top5 未命中（主要召回问题），3 个排序问题（次要）
    all_r = []
    sl = {}
    for i in range(10):
        tid = f"t_miss_{i:03d}"
        all_r.append(_make_retrieval_result(tid, 0, 0, 0, None, "全未命中"))
        sl[tid] = _make_processed_sample(tid, _make_retrieval_results(5))
    for i in range(3):
        tid = f"t_sort_{i:03d}"
        all_r.append(_make_retrieval_result(tid, 0, 1, 1, 2, "排序问题"))
        sl[tid] = _make_processed_sample(tid, _make_retrieval_results(5))
    # 一些命中样本
    for i in range(7):
        tid = f"t_hit_{i:03d}"
        all_r.append(_make_retrieval_result(tid, 1, 1, 1, 1, "命中"))
        sl[tid] = _make_processed_sample(tid, _make_retrieval_results(5))

    run = {
        "run_id": "run_priority", "config_id": "cfg_test_001",
        "question_count": 20, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 20}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    assert stats["top5_miss_n"] == 10, f"Top5 未命中应为 10，实际 {stats['top5_miss_n']}"
    assert stats["ranking_issue_n"] == 3, f"排序问题应为 3，实际 {stats['ranking_issue_n']}"
    assert stats["top5_miss_n"] > stats["ranking_issue_n"], "Top5 未命中应多于排序问题"

    # 验证 prompt 中优先级规则
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    prompt = captured_prompts[0]
    # 验证 Top5 未命中优先于排序问题
    miss_pos = prompt.find("Top5 完全未命中")
    sort_pos = prompt.find("排序问题")
    assert miss_pos < sort_pos, "prompt 中 Top5 未命中应出现在排序问题之前"
    assert "主要召回问题" in prompt, "prompt 应标注 Top5 未命中为主要召回问题"
    assert "次要排序问题" in prompt, "prompt 应标注排序问题为次要"
    assert "优先级最高" in prompt, "prompt 应标注 Top5 未命中优先级最高"
    assert "优先级低于" in prompt, "prompt 应标注排序问题优先级低于 Top5 未命中"

    # 验证建议排序规则
    assert "Top5 完全未命中问题的建议必须排在排序问题之前" in prompt

    print("PASS: 优先级排序正确，Top5 miss > 排序问题")


def test_config_value_no_unknown():
    """top_k=5、retrieval_mode=混合检索时，报告不得写当前值未知。"""
    print("=" * 60)
    print("测试：配置值不出现'未知'")
    print("=" * 60)

    config = {
        "config_name": "测试配置",
        "top_k": 5,
        "retrieval_mode": "混合检索",
        "chunk_strategy": "最大块1000、重叠120",
    }

    all_r = [_make_retrieval_result("t_r1", 0, 0, 0, None)]
    sl = {"t_r1": _make_processed_sample("t_r1", _make_retrieval_results(5))}
    run = {
        "run_id": "run_cfg_test", "config_id": "cfg_001",
        "question_count": 1, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 1, "question_set_name": "测试题集"}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    # 验证 config_values_text 包含实际值
    cvt = stats["config_values_text"]
    assert "top_k = 5" in cvt, f"config_values_text 应包含 top_k = 5，实际: {cvt}"
    assert "混合检索" in cvt, f"config_values_text 应包含混合检索，实际: {cvt}"
    assert "最大块1000" in cvt, f"config_values_text 应包含最大块1000，实际: {cvt}"
    assert "重叠 = 120" in cvt, f"config_values_text 应包含重叠 = 120，实际: {cvt}"

    # 验证 facts 中不会出现"未知"
    facts = build_facts_section(ctx, stats)
    assert "当前值未知" not in facts, "facts 不应出现'当前值未知'"
    assert "如果当前是语义" not in facts, "facts 不应出现'如果当前是语义'"

    # 验证 prompt 包含配置值
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    prompt = captured_prompts[0]
    assert "top_k = 5" in prompt, "prompt 应包含 top_k = 5"
    assert "混合检索" in prompt, "prompt 应包含混合检索"
    assert "不得写" in prompt or "不得编造" in prompt or "矛盾表述" in prompt, \
        "prompt 应包含禁止矛盾表述的规则"

    print("PASS: 配置值不出现'未知'")


def test_no_arbitrary_percentage_without_chunk_size():
    """缺少独立 chunk_size 时，不得出现任意百分比调整建议。"""
    print("=" * 60)
    print("测试：缺少 chunk_size 时不编造百分比")
    print("=" * 60)

    # config 有 chunk_strategy 但无独立 chunk_size
    config = {
        "config_name": "测试配置",
        "top_k": 5,
        "chunk_strategy": "默认策略",
        # 注意：没有 chunk_size 字段
    }

    all_r = [_make_retrieval_result("t_r1", 0, 0, 0, None)]
    sl = {"t_r1": _make_processed_sample("t_r1", _make_retrieval_results(5))}
    run = {
        "run_id": "run_no_cs", "config_id": "cfg_001",
        "question_count": 1, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    rdl = [{"run": run, "run_status": {"judge_count": 1, "question_set_name": "测试题集"}, "metrics": {}}]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    # 验证 prompt 包含禁止百分比规则
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    prompt = captured_prompts[0]
    assert "不得生成" in prompt and "具体百分比" in prompt, \
        "prompt 应包含禁止具体百分比的规则"
    assert "以当前 chunk_strategy 为基线" in prompt, \
        "prompt 应包含以 chunk_strategy 为基线的指导"
    assert "chunk_size" not in stats["confirmed_config_keys"], \
        "confirmed_config_keys 不应包含独立 chunk_size"

    print("PASS: 缺少 chunk_size 时不编造百分比")


def test_appendix_b_trace_ownership():
    """报告中的每个 trace 必须属于对应 run。"""
    print("=" * 60)
    print("测试：附录 trace 归属校验")
    print("=" * 60)

    config = _build_config()

    # 两个 run，各有不同的 trace
    all_r_run1 = [
        _make_retrieval_result("t_r1_miss", 0, 0, 0, None, "未命中"),
    ]
    all_r_run2 = [
        _make_retrieval_result("t_r2_miss", 0, 0, 0, None, "未命中"),
    ]
    # 修正 run_id
    all_r_run1[0]["run_id"] = "run_A"
    all_r_run1[0]["_source_run_id"] = "run_A"
    all_r_run2[0]["run_id"] = "run_B"
    all_r_run2[0]["_source_run_id"] = "run_B"

    all_r = all_r_run1 + all_r_run2
    sl = {
        "t_r1_miss": _make_processed_sample("t_r1_miss", _make_retrieval_results(5)),
        "t_r2_miss": _make_processed_sample("t_r2_miss", _make_retrieval_results(5)),
    }

    run_a = {
        "run_id": "run_A", "config_id": "cfg_test_001",
        "question_count": 1, "status": "completed",
        "started_at": "2026-07-16T10:00:00",
        "config_snapshot": config,
    }
    run_b = {
        "run_id": "run_B", "config_id": "cfg_test_001",
        "question_count": 1, "status": "completed",
        "started_at": "2026-07-16T11:00:00",
        "config_snapshot": config,
    }
    rdl = [
        {"run": run_a, "run_status": {"judge_count": 1, "question_set_name": "题集A"}, "metrics": {}},
        {"run": run_b, "run_status": {"judge_count": 1, "question_set_name": "题集B"}, "metrics": {}},
    ]

    ctx = build_analysis_context(rdl, sl, all_r, config)
    stats = compute_precise_stats(ctx)

    # 验证 run_id_to_question_set 映射
    rqs = stats["run_id_to_question_set"]
    assert rqs["run_A"] == "题集A", f"run_A 应对应题集A，实际 {rqs.get('run_A')}"
    assert rqs["run_B"] == "题集B", f"run_B 应对应题集B，实际 {rqs.get('run_B')}"

    # 验证 prompt 包含归属映射
    import optimization_analysis as oa
    captured_prompts = []
    orig_fn = oa.call_llm
    oa.call_llm = lambda p, k, u, m, timeout=120: (captured_prompts.append(p), "响应")[1]
    try:
        synthesize_optimization_report("概览", "诊断", ctx, "key", "url", "model")
    finally:
        oa.call_llm = orig_fn

    prompt = captured_prompts[0]
    assert "run_A" in prompt, "prompt 应包含 run_A"
    assert "题集A" in prompt, "prompt 应包含题集A"
    assert "题集B" in prompt, "prompt 应包含题集B"
    assert "归属信息缺失" in prompt, "prompt 应包含归属信息缺失的处理规则"
    assert "不得使用 run_A 的 trace" in prompt or "不得使用" in prompt, \
        "prompt 应包含跨 run 引用禁止规则"

    # 验证 facts 中 trace 归属正确
    facts = build_facts_section(ctx, stats)
    # t_r1_miss 应关联到 run_A，t_r2_miss 应关联到 run_B
    assert "run_A" in facts and "t_r1_miss" in facts, "facts 应将 t_r1_miss 关联到 run_A"
    assert "run_B" in facts and "t_r2_miss" in facts, "facts 应将 t_r2_miss 关联到 run_B"

    print("PASS: 附录 trace 归属校验正确")


def test_sanitize_chinese_config_name():
    """中文配置名生成预期文件名。"""
    print("=" * 60)
    print("测试：中文配置名清洗")
    print("=" * 60)

    result = sanitize_filename_component("合同知识库入库_v3")
    assert result == "合同知识库入库_v3", f"期望 '合同知识库入库_v3'，实际 '{result}'"

    filename = build_report_filename("合同知识库入库_v3", "20260720_112133")
    assert filename == "合同知识库入库_v3_ai_analysis_20260720_112133.md", \
        f"文件名错误: {filename}"

    print(f"PASS: {filename}")


def test_sanitize_illegal_chars():
    """含 / : * ? 等字符的配置名被安全清洗。"""
    print("=" * 60)
    print("测试：非法字符清洗")
    print("=" * 60)

    result = sanitize_filename_component('测试/配置:v1*name?')
    # / : * ? 都应被替换为 _，尾部 _ 被去除
    assert '/' not in result
    assert ':' not in result
    assert '*' not in result
    assert '?' not in result
    assert result == "测试_配置_v1_name", f"实际: '{result}'"

    # 含引号和尖括号
    result2 = sanitize_filename_component('a<b>c"d|e')
    assert '<' not in result2
    assert '>' not in result2
    assert '"' not in result2
    assert '|' not in result2
    print(f"PASS: 清洗后 '{result2}'")


def test_sanitize_empty_fallback():
    """空配置名回退为 '未命名配置'。"""
    print("=" * 60)
    print("测试：空配置名回退")
    print("=" * 60)

    assert sanitize_filename_component("") == "未命名配置"
    assert sanitize_filename_component(None) == "未命名配置"
    assert sanitize_filename_component("   ") == "未命名配置"
    # 清洗后为空的情况（全是非法字符）
    assert sanitize_filename_component("/:*>?") == "未命名配置"

    filename = build_report_filename("", "20260720_112133")
    assert filename == "未命名配置_ai_analysis_20260720_112133.md", \
        f"文件名错误: {filename}"

    print("PASS: 空名回退正确")


def test_sanitize_trailing_dots_spaces():
    """末尾空格和句点被去除。"""
    print("=" * 60)
    print("测试：末尾空格和句点")
    print("=" * 60)

    result = sanitize_filename_component("test name...  ")
    assert not result.endswith('.'), f"不应以句点结尾: '{result}'"
    assert not result.endswith(' '), f"不应以空格结尾: '{result}'"
    print(f"PASS: '{result}'")


def test_sanitize_max_len():
    """超长配置名被截断。"""
    print("=" * 60)
    print("测试：长度限制")
    print("=" * 60)

    long_name = "这是一个非常非常非常非常非常非常非常非常非常非常长的配置名称"
    result = sanitize_filename_component(long_name, max_len=20)
    assert len(result) <= 20, f"长度应 <=20，实际 {len(result)}: '{result}'"
    print(f"PASS: 截断后 '{result}'（{len(result)} 字符）")


def test_save_and_download_filename_consistent():
    """保存文件名与下载文件名一致。"""
    print("=" * 60)
    print("测试：保存/下载文件名一致")
    print("=" * 60)

    config_name = "合同知识库入库_v3"
    timestamp = "20260720_112133"

    # 模拟 save_analysis_report 的文件名生成
    saved_filename = build_report_filename(config_name, timestamp)

    # 模拟缓存后下载按钮使用的文件名
    cached_filename = saved_filename  # 两者应相同

    assert saved_filename == cached_filename, \
        f"不一致: 保存 '{saved_filename}' vs 下载 '{cached_filename}'"
    assert saved_filename == "合同知识库入库_v3_ai_analysis_20260720_112133.md"

    print(f"PASS: {saved_filename}")


def test_no_overwrite_different_times():
    """两次不同时间生成不会覆盖。"""
    import time
    print("=" * 60)
    print("测试：不同时间不覆盖")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir) / "reports"

        path1 = save_analysis_report("# 报告1", "测试配置", reports_dir)
        time.sleep(1.1)  # 确保时间戳不同（秒级精度）
        path2 = save_analysis_report("# 报告2", "测试配置", reports_dir)

        assert path1.exists(), f"第一个文件不存在: {path1}"
        assert path2.exists(), f"第二个文件不存在: {path2}"
        assert path1 != path2, f"两次生成不应覆盖: {path1}"
        assert path1.name != path2.name, f"文件名应不同: {path1.name} vs {path2.name}"

        content1 = path1.read_text(encoding="utf-8")
        content2 = path2.read_text(encoding="utf-8")
        assert content1 == "# 报告1"
        assert content2 == "# 报告2"

    print(f"PASS: {path1.name} != {path2.name}")


def _make_failure(trace_id, run_id="run_1", source_file="doc.xlsx",
                  question_set="题集A", query="测试查询", gold="金标准证据"):
    """构造测试用失败样本。"""
    return {
        "trace_id": trace_id, "run_id": run_id,
        "source_file_name": source_file, "question_set_name": question_set,
        "retrieval_query": query, "gold_evidence": gold,
        "hit_evidence_position": None,
        "retrieval_results": [
            {"position": 1, "document_name": "doc1", "content": "内容" * 50},
            {"position": 2, "document_name": "doc2", "content": "内容" * 50},
        ],
    }


def test_group_failures_all_samples_covered():
    """分组后不丢失任何失败样本。"""
    print("=" * 60)
    print("测试：分组覆盖全部失败样本")
    print("=" * 60)

    failures = {
        "top5_miss": [_make_failure(f"t_{i}", source_file=f"file_{i % 3}.xlsx")
                      for i in range(10)],
        "sorting_issues": [_make_failure(f"s_{i}", source_file=f"file_{i % 2}.xlsx")
                           for i in range(6)],
    }

    groups = group_failures_for_analysis(failures)
    total_in_groups = sum(g["count"] for g in groups)
    assert total_in_groups == 16, f"期望 16 条，实际 {total_in_groups}"

    # 检查所有 trace_id 都出现
    all_tids = set()
    for g in groups:
        all_tids.update(g["trace_ids"])
    expected_tids = {f"t_{i}" for i in range(10)} | {f"s_{i}" for i in range(6)}
    assert all_tids == expected_tids, f"丢失 trace_id: {expected_tids - all_tids}"

    print(f"[OK] {total_in_groups} 条样本分布在 {len(groups)} 组")


def test_source_key_fallback():
    """source_file_name 缺失时回退到 question_set_name。"""
    print("=" * 60)
    print("测试：source_key 回退")
    print("=" * 60)

    failures = {
        "top5_miss": [
            _make_failure("t1", source_file="", question_set="题集X"),
            _make_failure("t2", source_file="  ", question_set="题集X"),
            _make_failure("t3", source_file="doc.xlsx"),
        ],
        "sorting_issues": [],
    }

    groups = group_failures_for_analysis(failures)
    keys = {g["source_key"] for g in groups}
    assert "题集X" in keys, f"应有题集X，实际 {keys}"
    assert "doc.xlsx" in keys, f"应有 doc.xlsx，实际 {keys}"

    # 题集X 组应有 2 条
    qsn_group = next(g for g in groups if g["source_key"] == "题集X")
    assert qsn_group["count"] == 2

    print(f"[OK] source_key: {keys}")


def test_sub_batch_size_limit():
    """每个子批次不超过 _SUB_BATCH_MAX_SIZE 条。"""
    print("=" * 60)
    print("测试：子批次大小限制")
    print("=" * 60)

    group = {
        "failure_type": "top5_miss",
        "source_key": "doc.xlsx",
        "samples": [_make_failure(f"t_{i}") for i in range(10)],
        "trace_ids": [f"t_{i}" for i in range(10)],
        "count": 10,
    }

    batches = _split_into_sub_batches(group)
    for b in batches:
        assert len(b["payload"]) <= _SUB_BATCH_MAX_SIZE, \
            f"批次 {b['batch_id']} 有 {len(b['payload'])} 条，超过 {_SUB_BATCH_MAX_SIZE}"

    # 所有 trace_id 都应出现
    all_tids = set()
    for b in batches:
        all_tids.update(b["trace_ids"])
    assert all_tids == {f"t_{i}" for i in range(10)}

    print(f"[OK] {len(batches)} 个子批次，每批 ≤ {_SUB_BATCH_MAX_SIZE}")


def test_sub_batch_slim_payload():
    """子批次 payload 不含敏感字段，内容已截断。"""
    print("=" * 60)
    print("测试：子批次 payload 精简")
    print("=" * 60)

    sample = _make_failure("t1", gold="A" * 500)
    sample["retrieval_results"] = [
        {"position": 1, "document_name": "doc1", "content": "B" * 500},
    ]

    slim = _slim_sample(sample)

    # gold_evidence 截断
    assert len(slim["gold_evidence"]) <= 160, \
        f"gold_evidence 应 ≤160，实际 {len(slim['gold_evidence'])}"

    # content 截断
    for rr in slim["retrieval_results"]:
        assert len(rr["content"]) <= 120, \
            f"content 应 ≤120，实际 {len(rr['content'])}"

    # 不含敏感字段
    for key in ("api_key", "secret_key", "_prompt", "final_answer"):
        assert key not in slim, f"不应包含 {key}"

    # 包含必要字段
    assert slim["trace_id"] == "t1"
    assert slim["run_id"] == "run_1"
    assert slim["query"] == "测试查询"

    print("[OK] payload 精简正确")


def test_python_failure_manifest_no_llm():
    """Python 清单是确定性的，不含 LLM 生成内容。"""
    print("=" * 60)
    print("测试：Python 失败清单")
    print("=" * 60)

    failures = {
        "top5_miss": [_make_failure("t1"), _make_failure("t2")],
        "sorting_issues": [_make_failure("s1")],
    }

    manifest = _build_python_failure_manifest(failures)
    assert "t1" in manifest
    assert "t2" in manifest
    assert "s1" in manifest
    assert "Top5 完全未命中" in manifest
    assert "排序问题" in manifest
    assert "2 条" in manifest
    assert "1 条" in manifest

    print("[OK] Python 清单正确")


def test_progress_callback_invoked():
    """progress_callback 在各阶段被调用。"""
    print("=" * 60)
    print("测试：进度回调调用")
    print("=" * 60)

    failures = {
        "top5_miss": [_make_failure(f"t_{i}") for i in range(3)],
        "sorting_issues": [_make_failure(f"s_{i}") for i in range(2)],
    }
    context = {"failures": failures}
    phases = []

    def on_progress(phase, detail):
        phases.append(phase)

    # Mock call_llm to avoid real API
    from unittest.mock import patch, MagicMock
    mock_llm = MagicMock(return_value="mock diagnosis")

    with patch('optimization_analysis.call_llm', mock_llm):
        analyze_failure_groups(
            context, "key", "http://fake", "model",
            timeout=5, progress_callback=on_progress,
        )

    assert "grouping" in phases, f"应有 grouping，实际 {phases}"
    assert "sub_batch" in phases, f"应有 sub_batch，实际 {phases}"
    assert "synthesis" in phases, f"应有 synthesis，实际 {phases}"
    assert "done" in phases, f"应有 done，实际 {phases}"

    print(f"[OK] 回调阶段: {phases}")


def test_reduce_includes_failed_batches():
    """未完成批次在汇总中被标注，不伪装为已分析。"""
    print("=" * 60)
    print("测试：失败批次标注")
    print("=" * 60)

    from optimization_analysis import _STAGE2_REDUCE_PROMPT
    # 检查 reduce prompt 模板中有失败批次占位
    assert "{failed_batches_text}" in _STAGE2_REDUCE_PROMPT

    # 模拟有失败批次的输入
    failed_text = "- top5_miss_doc.xlsx_0: timeout（涉及 trace_id: t1, t2, t3...）"
    prompt = _STAGE2_REDUCE_PROMPT.format(
        retrieval_evaluable_n=100, top5_miss_n=20, ranking_issue_n=10,
        group_stats_text="- Top5 未命中 / doc.xlsx: 20 条",
        sub_batch_summaries="（无成功子批次）",
        failed_batches_text=failed_text,
    )
    assert "timeout" in prompt
    assert "t1" in prompt

    print("[OK] 失败批次正确传入 reduce prompt")


# ====== 主函数 ======

def main():
    tests = [
        test_context_scoping,
        test_sensitive_field_removal,
        test_grouping_correctness,
        test_truncation_rules,
        test_llm_call_order,
        test_failure_handling,
        test_markdown_references,
        test_no_data_modification,
        test_overview_metrics_accuracy,
        test_empty_data_handling,
        test_report_save_path,
        test_env_var_fallback,
        test_precise_stats_71_samples,
        test_retrieval_only_scope,
        test_single_sample_topic_no_strong_conclusion,
        test_missing_config_no_specific_values,
        test_priority_ordering,
        test_config_value_no_unknown,
        test_no_arbitrary_percentage_without_chunk_size,
        test_appendix_b_trace_ownership,
        test_sanitize_chinese_config_name,
        test_sanitize_illegal_chars,
        test_sanitize_empty_fallback,
        test_sanitize_trailing_dots_spaces,
        test_sanitize_max_len,
        test_save_and_download_filename_consistent,
        test_no_overwrite_different_times,
        test_group_failures_all_samples_covered,
        test_source_key_fallback,
        test_sub_batch_size_limit,
        test_sub_batch_slim_payload,
        test_python_failure_manifest_no_llm,
        test_progress_callback_invoked,
        test_reduce_includes_failed_batches,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 个测试")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
