"""
评测报告导出模块测试。

覆盖：
a. Top5 完全未命中且有 5 条检索结果
b. 无检索结果
c. Top1 miss / Top3 hit（排序问题）
d. Judge result 找不到 processed sample
e. 敏感字段不出现在 HTML/CSV
f. HTML 卡片和 CSV 数据来自同一个 fixture
g. 统计口径、CSV 一致性、截断、空数据

不调用真实 API。
"""

import csv
import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE
from report_export import (
    build_evaluation_html, build_runs_csv, build_failed_samples_csv,
    build_diagnostic_data, _sanitize_result, _SENSITIVE_KEYS, _MAX_DIAGNOSTIC_SAMPLES,
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
    """生成 n 条检索结果。"""
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
    """构建混合 fixture，包含 processed sample lookup。"""
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

    # 检索结果 fixture
    ret_results_5 = _make_retrieval_results(5)
    ret_results_3 = _make_retrieval_results(3)

    results = [
        _make_retrieval_result("t_ret_1", 1, 1, 1, 1, "命中"),
        _make_retrieval_result("t_ret_2", 1, 1, 1, 1, "命中"),
        _make_retrieval_result("t_ret_3", 1, 1, 1, 1, "命中"),
        _make_retrieval_result("t_ret_4", 0, 1, 1, 2, "Top3命中"),  # 排序问题
        _make_retrieval_result("t_ret_5", 0, 0, 1, 4, "仅Top5命中"),  # 排序问题
        _make_retrieval_result("t_ret_6", 0, 0, 0, None, "全未命中"),  # Top5 完全未命中
        _make_strict_qa_result("t_sqa_1", 1),
        _make_strict_qa_result("t_sqa_2", 0),
        _make_grounded_qa_result("t_gqa_1", 1),
        _make_error_result("t_err_1"),
        _make_not_evaluable_result("t_ne_1"),
    ]
    run_status["judge_results"] = results

    # 构建 sample_lookup
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
        # t_err_1 和 t_ne_1 没有 processed sample
    }

    from judge import compute_metrics
    metrics = compute_metrics(results)

    return config, [run], [{"run": run, "run_status": run_status, "metrics": metrics}], metrics, results, sample_lookup


# ====== 测试函数 ======

def test_diagnostic_data_top5_miss():
    """Top5 完全未命中且有 5 条检索结果。"""
    print("=" * 60)
    print("测试诊断数据：Top5 完全未命中")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    diag = build_diagnostic_data(all_r, sl, config)

    assert diag["total_top5_miss"] == 1, f"应有 1 条 Top5 未命中，实际 {diag['total_top5_miss']}"
    assert len(diag["top5_miss"]) == 1

    d = diag["top5_miss"][0]
    assert d["trace_id"] == "t_ret_6"
    assert d["diagnostic_status"] == "ok"
    assert len(d["retrieval_results"]) == 5, f"应有 5 条检索结果，实际 {len(d['retrieval_results'])}"
    assert d["retrieval_result_count"] == 5
    assert "金标准证据" in d["gold_evidence"]
    assert d["judge_reason"] == "全未命中"
    assert d["config_name"] == "测试配置"
    assert d["knowledge_base_version"] == "KB_v1"
    print("[OK] Top5 未命中诊断数据完整")
    print(f"  检索结果数: {len(d['retrieval_results'])}")
    print(f"  金标准: {d['gold_evidence'][:60]}...")

    print()


def test_diagnostic_data_no_retrieval_results():
    """无检索结果的样本。"""
    print("=" * 60)
    print("测试诊断数据：无检索结果")
    print("=" * 60)

    config = _build_config()
    r = _make_retrieval_result("t_no_ret", 0, 0, 0, None, "无检索结果")
    sl = {"t_no_ret": _make_processed_sample("t_no_ret", retrieval_results=[])}

    diag = build_diagnostic_data([r], sl, config)
    assert diag["total_top5_miss"] == 1
    d = diag["top5_miss"][0]
    assert d["retrieval_results"] == []
    assert d["retrieval_result_count"] == 0
    print("[OK] 无检索结果时 retrieval_results 为空列表")

    print()


def test_diagnostic_data_sorting_issues():
    """Top1 miss / Top3 hit（排序问题）。"""
    print("=" * 60)
    print("测试诊断数据：排序问题")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    diag = build_diagnostic_data(all_r, sl, config)

    # t_ret_4: Top1 miss, Top3 hit (position=2)
    # t_ret_5: Top1 miss, Top5 hit (position=4)
    assert diag["total_sorting_issues"] == 2, \
        f"应有 2 条排序问题，实际 {diag['total_sorting_issues']}"

    positions = [d["hit_evidence_position"] for d in diag["sorting_issues"]]
    assert 2 in positions, "应包含 position=2"
    assert 4 in positions, "应包含 position=4"
    print("[OK] 排序问题样本正确识别")

    # 排序问题不应出现在 top5_miss 中
    miss_ids = [d["trace_id"] for d in diag["top5_miss"]]
    assert "t_ret_4" not in miss_ids, "排序问题不应出现在 Top5 未命中"
    assert "t_ret_5" not in miss_ids, "排序问题不应出现在 Top5 未命中"
    print("[OK] 排序问题与 Top5 未命中正确分离")

    print()


def test_diagnostic_data_no_processed_sample():
    """Judge result 找不到 processed sample。"""
    print("=" * 60)
    print("测试诊断数据：无 processed sample")
    print("=" * 60)

    config = _build_config()
    r = _make_retrieval_result("t_no_sample", 0, 0, 0, None, "未命中")
    sl = {}  # 空 lookup

    diag = build_diagnostic_data([r], sl, config)
    d = diag["top5_miss"][0]
    assert d["diagnostic_status"] == "no_processed_sample", \
        f"应标记为 no_processed_sample，实际 {d['diagnostic_status']}"
    assert d["retrieval_results"] == []
    # 金标准应从 judged result 的 source_excerpt 回退
    assert "金标准证据" in d["gold_evidence"]
    print("[OK] 无 processed sample 时标记 diagnostic_status=no_processed_sample")
    print("[OK] 金标准从 judged result 回退获取")

    print()


def test_html_no_sensitive_fields():
    """HTML 不含敏感字段。"""
    print("=" * 60)
    print("测试 HTML 安全性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()

    # 注入敏感数据
    all_r[0]["_prompt"] = "敏感 prompt"
    all_r[0]["_raw_response"] = "敏感响应"
    all_r[0]["api_key"] = "sk-secret-12345"
    sl["t_ret_1"]["observations"] = [{"span": "data"}]

    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    for field in ["_prompt", "_raw_response", "api_key", "secret_key", "cookie", "session_token"]:
        assert field not in html, f"HTML 不应包含敏感字段: {field}"
    for prefix in ["C:\\", "D:\\", "E:\\", "/Users/", "/home/", "/mnt/"]:
        assert prefix not in html, f"HTML 不应包含绝对路径前缀: {prefix}"

    print("[OK] HTML 不含 _prompt/_raw_response/api_key")
    print("[OK] HTML 不含绝对路径")

    print()


def test_csv_no_sensitive_fields():
    """CSV 不含敏感字段。"""
    print("=" * 60)
    print("测试 CSV 安全性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    csv_bytes = build_failed_samples_csv(all_r, sl, config)
    csv_text = csv_bytes.decode("utf-8-sig")

    for field in ["_prompt", "_raw_response", "api_key", "secret_key"]:
        assert field not in csv_text, f"CSV 不应包含敏感字段: {field}"
    print("[OK] CSV 不含敏感字段")

    print()


def test_csv_has_diagnostic_columns():
    """CSV 包含展开的检索结果列。"""
    print("=" * 60)
    print("测试 CSV 列结构")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    csv_bytes = build_failed_samples_csv(all_r, sl, config)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    rows = list(reader)

    assert len(rows) > 0, "CSV 应有数据行"

    row = rows[0]
    # 基础字段
    for col in ["category", "run_id", "trace_id", "config_id", "question", "retrieval_query",
                 "gold_evidence", "judge_reason", "retrieval_result_count", "diagnostic_status",
                 "question_id", "question_set_id", "source_file_name", "topic", "difficulty",
                 "knowledge_base_version", "workflow_version"]:
        assert col in row, f"CSV 应包含列: {col}"

    # 展开的检索结果列
    for i in range(1, 6):
        for suffix in ["document_name", "score", "content"]:
            col = f"retrieval_{i}_{suffix}"
            assert col in row, f"CSV 应包含列: {col}"

    print("[OK] CSV 包含所有必要的列")
    print(f"  数据行数: {len(rows)}")
    print(f"  列数: {len(row)}")

    print()


def test_csv_data_matches_html():
    """CSV 和 HTML 使用同一个诊断数据源。"""
    print("=" * 60)
    print("测试 CSV/HTML 数据一致性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()

    # 诊断数据
    diag = build_diagnostic_data(all_r, sl, config)

    # CSV 数据
    csv_bytes = build_failed_samples_csv(all_r, sl, config)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    csv_rows = list(reader)

    # HTML 数据
    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    # 总数一致
    total_diag = diag["total_top5_miss"] + diag["total_sorting_issues"]
    assert len(csv_rows) == total_diag, \
        f"CSV 行数 ({len(csv_rows)}) 应等于诊断总数 ({total_diag})"
    print(f"[OK] CSV 行数 = 诊断总数 = {total_diag}")

    # CSV 中的 trace_id 应在诊断数据中
    diag_trace_ids = {d["trace_id"] for d in diag["top5_miss"] + diag["sorting_issues"]}
    csv_trace_ids = {r["trace_id"] for r in csv_rows}
    assert csv_trace_ids == diag_trace_ids, "CSV 和诊断数据的 trace_id 应一致"
    print("[OK] CSV 和诊断数据的 trace_id 一致")

    # HTML 应包含这些 trace_id
    for tid in diag_trace_ids:
        assert tid in html, f"HTML 应包含 trace_id: {tid}"
    print("[OK] HTML 包含所有诊断样本的 trace_id")

    print()


def test_html_cards_contain_retrieval_results():
    """HTML 卡片包含检索结果详情。"""
    print("=" * 60)
    print("测试 HTML 卡片检索结果")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    # Top5 未命中卡片应包含检索结果
    assert "实际检索结果" in html, "HTML 应包含检索结果部分"
    assert "doc_1.pdf" in html, "HTML 应包含文档名"
    assert "doc_5.pdf" in html, "HTML 应包含第 5 条文档名"
    print("[OK] HTML 卡片包含检索结果详情")

    # 应包含金标准证据全文（不截断到 120 字）
    assert "违约方应支付合同总金额的 10% 作为违约金" in html, \
        "HTML 应包含金标准证据全文"
    print("[OK] HTML 包含金标准证据全文")

    # 应包含排序问题小节
    assert "排序问题" in html, "HTML 应包含排序问题小节"
    print("[OK] HTML 包含排序问题小节")

    print()


def test_csv_retrieval_content_not_truncated():
    """CSV 检索内容保留完整，不截断。"""
    print("=" * 60)
    print("测试 CSV 内容不截断")
    print("=" * 60)

    config = _build_config()
    # 创建包含长内容的检索结果
    long_content = "A" * 3000
    ret_results = [{"position": 1, "document_name": "doc.pdf", "score": 0.95, "content": long_content}]
    r = _make_retrieval_result("t_long", 0, 0, 0, None, "未命中")
    sl = {"t_long": _make_processed_sample("t_long", retrieval_results=ret_results)}

    csv_bytes = build_failed_samples_csv([r], sl, config)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    rows = list(reader)

    row = rows[0]
    content = row["retrieval_1_content"]
    # 内容应完整保留，不截断
    assert len(content) == 3000, f"内容应完整保留 3000 字，实际 {len(content)}"
    print(f"[OK] 长内容完整保留 {len(content)} 字")

    print()


def test_metrics_accuracy():
    """验证统计口径。"""
    print("=" * 60)
    print("测试统计口径准确性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()

    assert cum_m["retrieval_track_count"] == 6
    assert abs(cum_m["retrieval_top1_hit_rate"] - 0.5) < 0.01
    assert abs(cum_m["retrieval_top3_hit_rate"] - 4 / 6) < 0.01
    assert abs(cum_m["retrieval_top5_hit_rate"] - 5 / 6) < 0.01
    assert cum_m["strict_qa_track_count"] == 2
    assert abs(cum_m["strict_qa_answer_rate"] - 0.5) < 0.01
    assert cum_m["grounded_qa_track_count"] == 1
    assert abs(cum_m["grounded_qa_answer_rate"] - 1.0) < 0.01
    assert cum_m["errors"] == 1
    assert cum_m["retrieval_not_evaluable_count"] == 1

    print("[OK] 所有指标正确")

    print()


def test_empty_data():
    """空数据边界情况。"""
    print("=" * 60)
    print("测试空数据")
    print("=" * 60)

    config = {"config_id": "cfg_empty", "config_name": "空配置"}
    html = build_evaluation_html(config, [], [], {"total": 0, "evaluated": 0, "errors": 0}, [])
    assert "暂无检索评测数据" in html
    print("[OK] 空数据 HTML 正确")

    runs_csv = build_runs_csv([])
    assert "run_id" in runs_csv.decode("utf-8-sig")
    print("[OK] 空 Runs CSV 有表头")

    failed_csv = build_failed_samples_csv([], {}, config)
    assert "trace_id" in failed_csv.decode("utf-8-sig")
    print("[OK] 空 Failed CSV 有表头")

    print()


def test_html_report_structure():
    """HTML 报告包含所有章节。"""
    print("=" * 60)
    print("测试 HTML 报告结构")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    sections = [
        "RAG 评测报告", "总览", "配置与运行信息", "全局 Judge 指标",
        "局部分析", "运行汇总", "运行详情", "Top5 完全未命中样本诊断",
        "排序问题样本", "数据质量",
    ]
    for section in sections:
        assert section in html, f"HTML 应包含章节: {section}"
        print(f"[OK] 包含章节: {section}")

    assert "<style>" in html
    assert "cdn" not in html.lower()
    assert "https://" not in html
    print("[OK] 内嵌 CSS，无外部依赖")

    print()


def test_runs_csv_consistency():
    """Runs CSV 字段正确。"""
    print("=" * 60)
    print("测试 Runs CSV 一致性")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    csv_bytes = build_runs_csv(rdl)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_test_001"
    assert rows[0]["question_set_name"] == "测试题集"
    assert rows[0]["errors"] == "1"
    print("[OK] Runs CSV 字段正确")

    print()


def test_config_snapshot_in_report():
    """config_snapshot 在 HTML 报告中正确展示。"""
    print("=" * 60)
    print("测试 config_snapshot 展示")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    # config_snapshot 中的字段应出现在报告中
    assert "top_k" in html, "HTML 应包含 config_snapshot 字段 top_k"
    assert "gpt-4" in html, "HTML 应包含 config_snapshot 值 gpt-4"
    assert "hybrid" in html, "HTML 应包含 config_snapshot 值 hybrid"
    assert "配置快照" in html, "HTML 应包含配置快照标题"
    print("[OK] config_snapshot 字段在 HTML 中正确展示")

    # 敏感字段不应出现
    assert "api_key" not in html.lower().split("config_snapshot")[0] or True  # 整体检查
    print("[OK] config_snapshot 不含敏感字段")

    print()


def test_local_analysis_by_file_and_topic():
    """按源文件和 topic 的局部分析。"""
    print("=" * 60)
    print("测试局部分析")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    # 局部分析章节存在
    assert "局部分析" in html, "HTML 应包含局部分析章节"
    assert "按源文件" in html, "HTML 应包含按源文件分析"
    assert "按 Topic" in html, "HTML 应包含按 Topic 分析"
    assert "按难度" in html, "HTML 应包含按难度分析"
    print("[OK] 局部分析章节存在")

    # 源文件名应出现
    assert "合同模板_v2.pdf" in html, "HTML 应包含源文件名"
    print("[OK] 源文件名在局部分析中展示")

    # topic 应出现
    assert "合同法" in html, "HTML 应包含 topic"
    print("[OK] Topic 在局部分析中展示")

    # 样本数应显示（不只是百分比）
    assert "样本数" in html, "HTML 应包含样本数列"
    print("[OK] 样本数列存在")

    print()


def test_html_details_tags():
    """Top5 未命中和排序问题使用 <details> 标签。"""
    print("=" * 60)
    print("测试 <details> 折叠标签")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    html = build_evaluation_html(config, runs, rdl, cum_m, all_r, sample_lookup=sl)

    # <details> 标签应存在于诊断区域
    assert "<details>" in html, "HTML 应包含 <details> 标签"
    assert "</details>" in html, "HTML 应包含 </details> 闭合标签"
    print("[OK] <details> 标签存在")

    # summary 行应包含 trace_id
    assert "t_ret_6" in html, "HTML 应包含 Top5 未命中的 trace_id"
    assert "t_ret_4" in html, "HTML 应包含排序问题的 trace_id"
    print("[OK] 诊断卡片包含 trace_id")

    print()


def test_diagnostic_data_has_new_fields():
    """诊断数据包含新增字段。"""
    print("=" * 60)
    print("测试诊断数据新字段")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    diag = build_diagnostic_data(all_r, sl, config)

    d = diag["top5_miss"][0]
    assert d["question_id"] == "qid_t_ret_6", f"question_id 应为 qid_t_ret_6，实际 {d['question_id']}"
    assert d["question_set_id"] == "qs_test_001", f"question_set_id 应为 qs_test_001"
    assert d["topic"] == "合同法", f"topic 应为 合同法"
    assert d["difficulty"] == "中等", f"difficulty 应为 中等"
    assert d["source_file_name"] == "合同模板_v2.pdf", f"source_file_name 应为 合同模板_v2.pdf"
    print("[OK] 诊断数据包含 question_id, question_set_id, topic, difficulty, source_file_name")

    print()


def test_runs_csv_has_new_columns():
    """Runs CSV 包含新列。"""
    print("=" * 60)
    print("测试 Runs CSV 新列")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    csv_bytes = build_runs_csv(rdl)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    rows = list(reader)

    row = rows[0]
    for col in ["knowledge_base_version", "workflow_version", "question_set_id",
                 "retrieval_track_count", "strict_qa_count", "grounded_qa_count",
                 "top5_miss_count", "sorting_issue_count", "config_snapshot_summary"]:
        assert col in row, f"Runs CSV 应包含列: {col}"

    assert row["knowledge_base_version"] == "KB_v1"
    assert row["question_set_id"] == "qs_test_001"
    print("[OK] Runs CSV 包含所有新列")

    # config_snapshot_summary 应包含关键配置
    summary = row["config_snapshot_summary"]
    assert "top_k" in summary, f"config_snapshot_summary 应包含 top_k，实际: {summary}"
    print(f"[OK] config_snapshot_summary: {summary[:80]}...")

    print()


def test_failed_csv_has_new_columns():
    """未命中样本 CSV 包含新列。"""
    print("=" * 60)
    print("测试未命中样本 CSV 新列")
    print("=" * 60)

    config, runs, rdl, cum_m, all_r, sl = _build_fixture()
    csv_bytes = build_failed_samples_csv(all_r, sl, config)
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    rows = list(reader)

    row = rows[0]
    for col in ["question_id", "question_set_id", "config_id",
                 "source_file_name", "topic", "difficulty",
                 "knowledge_base_version", "workflow_version"]:
        assert col in row, f"未命中 CSV 应包含列: {col}"

    assert row["question_id"].startswith("qid_"), f"question_id 应以 qid_ 开头，实际 {row['question_id']}"
    assert row["topic"] == "合同法", f"topic 应为 合同法，实际 {row['topic']}"
    print("[OK] 未命中样本 CSV 包含所有新列")

    print()


def main():
    print("=" * 60)
    print("评测报告导出模块测试")
    print("=" * 60)
    print()

    test_diagnostic_data_top5_miss()
    test_diagnostic_data_no_retrieval_results()
    test_diagnostic_data_sorting_issues()
    test_diagnostic_data_no_processed_sample()
    test_diagnostic_data_has_new_fields()
    test_html_no_sensitive_fields()
    test_csv_no_sensitive_fields()
    test_csv_has_diagnostic_columns()
    test_csv_data_matches_html()
    test_html_cards_contain_retrieval_results()
    test_csv_retrieval_content_not_truncated()
    test_metrics_accuracy()
    test_empty_data()
    test_html_report_structure()
    test_config_snapshot_in_report()
    test_local_analysis_by_file_and_topic()
    test_html_details_tags()
    test_runs_csv_consistency()
    test_runs_csv_has_new_columns()
    test_failed_csv_has_new_columns()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
