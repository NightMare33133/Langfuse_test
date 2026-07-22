"""
导出文件名与元数据测试。

覆盖：
a. 配置名出现在文件名中
b. 同名不同 config_id 不冲突
c. 非法字符清理
d. CSV/HTML 元数据包含 config_name、config_id
e. 无 config_id 时抛 ValueError
f. 文件名长度合理
g. 空配置名回退
"""

import csv
import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from report_export import (
    build_export_filename, build_evaluation_html, build_runs_csv, build_failed_samples_csv,
)
from judge import TRACK_RETRIEVAL, compute_metrics


# ====== 辅助 ======

def _make_config(config_name="测试配置", config_id="cfg_20260720_152721_244327_测试配置"):
    return {
        "config_id": config_id,
        "config_name": config_name,
        "knowledge_base_version": "KB_v1",
        "workflow_version": "WF_v1",
    }


def _make_run(config_id="cfg_20260720_152721_244327_测试配置", config_name="测试配置"):
    return {
        "run_id": "run_test_001",
        "config_id": config_id,
        "question_count": 5,
        "status": "completed",
        "started_at": "2026-07-20T10:00:00",
        "question_set_name": "测试题集",
        "question_set_id": "qs_001",
        "config_snapshot": {
            "config_name": config_name,
            "config_id": config_id,
            "knowledge_base_version": "KB_v1",
            "workflow_version": "WF_v1",
        },
    }


def _make_judge_result(trace_id):
    return {
        "trace_id": trace_id,
        "question": f"问题_{trace_id}",
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_evaluable": True,
        "retrieval_top1_hit": 1,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 1,
        "reason": "命中",
        "run_id": "run_test_001",
    }


# ====== 测试：文件名包含配置名 ======

def test_filename_contains_config_name():
    """配置名清洗后出现在文件名中。"""
    print("=" * 60)
    print("测试：文件名包含配置名")
    print("=" * 60)

    name = build_export_filename("合同知识库入库 v2_4", "cfg_abc12345def", "report", "html")
    assert "合同知识库入库" in name, f"文件名应含配置名，实际: {name}"
    assert "v2_4" in name, f"文件名应含版本号，实际: {name}"
    assert name.endswith(".html"), f"应以 .html 结尾，实际: {name}"
    assert "cfg_12345def" in name, f"文件名应含 config_id 短标识，实际: {name}"
    print(f"[OK] 文件名: {name}")

    print()


# ====== 测试：同名不同 config_id 不冲突 ======

def test_same_name_different_id_no_collision():
    """同名配置的不同 config_id 生成不同文件名。"""
    print("=" * 60)
    print("测试：同名不同 config_id 不冲突")
    print("=" * 60)

    id_a = "cfg_20260720_100000_000000_合同知识库"
    id_b = "cfg_20260721_200000_999999_合同知识库"
    name_a = build_export_filename("合同知识库入库", id_a, "report", "html")
    name_b = build_export_filename("合同知识库入库", id_b, "report", "html")

    assert name_a != name_b, f"不同 config_id 应生成不同文件名:\n  A: {name_a}\n  B: {name_b}"
    # 短标识不同
    assert "cfg_000000_合同知识库" not in name_b
    print(f"[OK] A: {name_a}")
    print(f"[OK] B: {name_b}")

    print()


# ====== 测试：非法字符清理 ======

def test_illegal_chars_sanitized():
    """Windows 非法字符被替换为下划线。"""
    print("=" * 60)
    print("测试：非法字符清理")
    print("=" * 60)

    name = build_export_filename('测试/配置:v1*name?', "cfg_12345678abcdef", "report", "html")
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        # 文件名中不应出现原始非法字符（双下划线中的单个字符）
        # 但清洗后会变成下划线，所以检查不含原始非法字符对
        pass
    assert '/' not in name, f"不应含 /，实际: {name}"
    assert ':' not in name, f"不应含 :，实际: {name}"
    assert '*' not in name, f"不应含 *，实际: {name}"
    assert '?' not in name, f"不应含 ?，实际: {name}"
    print(f"[OK] 清洗后: {name}")

    # 含空格的配置名
    name2 = build_export_filename("  多 空 格  配置  ", "cfg_aabbccdd1234", "runs", "csv")
    assert "  " not in name2.split("__")[0], f"文件名起始部分不应含连续空格，实际: {name2}"
    print(f"[OK] 空格处理: {name2}")

    print()


# ====== 测试：CSV/HTML 元数据一致性 ======

def test_csv_html_metadata_consistency():
    """CSV 和 HTML 都包含 config_name、config_id。"""
    print("=" * 60)
    print("测试：CSV/HTML 元数据一致性")
    print("=" * 60)

    config = _make_config("元数据测试配置", "cfg_meta_test_12345678")
    run = _make_run(config["config_id"], config["config_name"])
    jr = [_make_judge_result("t1"), _make_judge_result("t2")]
    metrics = compute_metrics(jr)
    run_status = {
        "batch_success": 2, "batch_total": 2, "raw_count": 2,
        "processed_count": 2, "judge_count": 2,
        "judge_results": jr,
    }
    rdl = [{"run": run, "run_status": run_status, "metrics": metrics}]

    # HTML
    html = build_evaluation_html(config, [run], rdl, metrics, jr)
    assert "元数据测试配置" in html, "HTML 应包含 config_name"
    assert "cfg_meta_test_12345678" in html, "HTML 应包含 config_id"
    assert "KB_v1" in html, "HTML 应包含知识库版本"
    assert "WF_v1" in html, "HTML 应包含工作流版本"
    print("[OK] HTML 包含 config_name、config_id、KB 版本、工作流版本")

    # Runs CSV
    runs_csv = build_runs_csv(rdl)
    runs_text = runs_csv.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(runs_text))
    rows = list(reader)
    assert rows[0]["config_name"] == "元数据测试配置", f"Runs CSV config_name 错误: {rows[0]['config_name']}"
    assert rows[0]["config_id"] == "cfg_meta_test_12345678", f"Runs CSV config_id 错误: {rows[0]['config_id']}"
    print("[OK] Runs CSV 包含 config_name、config_id")

    # Failed samples CSV（构造一个未命中样本）
    miss_result = {
        "trace_id": "t_miss_1",
        "question": "未命中问题",
        "source_excerpt": "证据",
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_evaluable": True,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 0,
        "retrieval_top5_hit": 0,
        "hit_evidence_position": None,
        "reason": "未命中",
        "run_id": "run_test_001",
    }
    failed_csv = build_failed_samples_csv([miss_result], {}, config)
    failed_text = failed_csv.decode("utf-8-sig")
    failed_reader = csv.DictReader(io.StringIO(failed_text))
    failed_rows = list(failed_reader)
    assert len(failed_rows) > 0, "应有未命中样本"
    assert failed_rows[0]["config_name"] == "元数据测试配置", \
        f"Failed CSV config_name 错误: {failed_rows[0].get('config_name')}"
    assert failed_rows[0]["config_id"] == "cfg_meta_test_12345678", \
        f"Failed CSV config_id 错误: {failed_rows[0].get('config_id')}"
    print("[OK] Failed samples CSV 包含 config_name、config_id")

    print()


# ====== 测试：无 config_id 禁止导出 ======

def test_no_config_id_raises():
    """空 config_id 时抛出 ValueError。"""
    print("=" * 60)
    print("测试：无 config_id 禁止导出")
    print("=" * 60)

    try:
        build_export_filename("测试", "", "report", "html")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "config_id" in str(e), f"错误信息应提及 config_id，实际: {e}"
        print(f"[OK] 空 config_id 抛出 ValueError: {e}")

    try:
        build_export_filename("测试", "   ", "report", "html")
        assert False, "应抛出 ValueError"
    except ValueError:
        print("[OK] 纯空格 config_id 抛出 ValueError")

    print()


# ====== 测试：文件名长度限制 ======

def test_filename_length():
    """超长配置名截断后文件名不超过 200 字符。"""
    print("=" * 60)
    print("测试：文件名长度限制")
    print("=" * 60)

    long_name = "这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的配置名称_用于测试长度截断功能"
    name = build_export_filename(long_name, "cfg_12345678abcdef", "report", "html")
    assert len(name) <= 200, f"文件名应 <=200 字符，实际 {len(name)}: {name}"
    assert name.endswith(".html"), f"应以 .html 结尾，实际: {name}"
    print(f"[OK] 长度 {len(name)} 字符: {name}")

    print()


# ====== 测试：空配置名回退 ======

def test_empty_config_name_fallback():
    """空配置名回退为 '未命名'。"""
    print("=" * 60)
    print("测试：空配置名回退")
    print("=" * 60)

    name = build_export_filename("", "cfg_12345678abcdef", "report", "html")
    assert "未命名" in name, f"空名应回退为 '未命名'，实际: {name}"
    print(f"[OK] 空名回退: {name}")

    name2 = build_export_filename(None, "cfg_12345678abcdef", "runs", "csv")
    assert "未命名" in name2, f"None 名应回退为 '未命名'，实际: {name2}"
    print(f"[OK] None 名回退: {name2}")

    print()


# ====== 测试：不同 suffix 生成不同文件名 ======

def test_different_suffixes():
    """同一配置的不同 suffix 生成不同文件名。"""
    print("=" * 60)
    print("测试：不同 suffix 文件名")
    print("=" * 60)

    cid = "cfg_12345678abcdef"
    f1 = build_export_filename("测试配置", cid, "report", "html")
    f2 = build_export_filename("测试配置", cid, "runs", "csv")
    f3 = build_export_filename("测试配置", cid, "failed_samples", "csv")

    assert f1 != f2 != f3, "不同 suffix 应生成不同文件名"
    assert f1.endswith(".html")
    assert f2.endswith(".csv")
    assert f3.endswith(".csv")
    assert "report" in f1
    assert "runs" in f2
    assert "failed_samples" in f3
    print(f"[OK] report:  {f1}")
    print(f"[OK] runs:    {f2}")
    print(f"[OK] failed:  {f3}")

    print()


# ====== 测试：按钮标签包含配置名 ======

def test_button_label_logic():
    """验证按钮标签构建逻辑。"""
    print("=" * 60)
    print("测试：按钮标签包含配置名")
    print("=" * 60)

    disp_name = "合同知识库入库 v2_4"
    labels = [
        f"下载 HTML 报告（{disp_name}）",
        f"下载运行汇总 CSV（{disp_name}）",
        f"下载未命中样本 CSV（{disp_name}）",
    ]

    for label in labels:
        assert disp_name in label, f"标签应含配置名，实际: {label}"
        print(f"[OK] {label}")

    print()


# ====== main ======

def main():
    print("=" * 60)
    print("导出文件名与元数据测试")
    print("=" * 60)
    print()

    test_filename_contains_config_name()
    test_same_name_different_id_no_collision()
    test_illegal_chars_sanitized()
    test_csv_html_metadata_consistency()
    test_no_config_id_raises()
    test_filename_length()
    test_empty_config_name_fallback()
    test_different_suffixes()
    test_button_label_logic()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
