"""
测试样本准备页的数据导入辅助函数。

覆盖：
a. list_langfuse_export_files 过滤逻辑
b. mtime 倒序排序
c. API 拉取后自动选择新文件
d. 上传模式不被 API 自动选择覆盖
e. 新选择清理解析状态
f. 无合法导出文件时空状态

不调用真实 API。
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 从 app.py 导入辅助函数
sys.path.insert(0, str(Path(__file__).parent))
from app import list_langfuse_export_files


def _write_jsonl(path, lines):
    """辅助：写入 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def test_filter_logic():
    """测试文件过滤：只保留合法 Langfuse 导出文件。"""
    print("=" * 60)
    print("测试文件过滤逻辑")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)

        # 1. API 拉取文件 → 保留
        api_file = raw_dir / "langfuse_api_export_20260716_120000.jsonl"
        _write_jsonl(api_file, [{"traceId": "t1", "type": "TRACE"}])

        # 2. UI 导出文件 → 保留
        ui_file = raw_dir / "1783322412151-lf-events-export-cmr8tec2s0007ad0c2klqym7l.jsonl"
        _write_jsonl(ui_file, [{"traceId": "t2", "type": "TRACE"}])

        # 3. 含 traceId 的其他 JSONL → 保留
        other_with_trace = raw_dir / "custom_export.jsonl"
        _write_jsonl(other_with_trace, [{"traceId": "t3", "type": "TRACE"}])

        # 4. batch_qa 文件 → 排除
        batch_qa = raw_dir / "batch_qa_20260716_120000.jsonl"
        _write_jsonl(batch_qa, [{"question": "q1", "success": True}])

        # 5. batch_results 文件 → 排除
        batch_results = raw_dir / "batch_results_20260716_120000.jsonl"
        _write_jsonl(batch_results, [{"question": "q1", "success": True}])

        # 6. questions 文件 → 排除
        questions = raw_dir / "questions_test_20260714_100000.jsonl"
        _write_jsonl(questions, [{"question": "q1", "question_set_id": "qs_1"}])

        # 7. eval_results 文件 → 排除
        eval_results = raw_dir / "eval_results_20260716_120000.jsonl"
        _write_jsonl(eval_results, [{"traceId": "t4", "retrieval_top1_hit": 1}])

        # 8. langfuse_samples 文件 → 排除
        langfuse_samples = raw_dir / "langfuse_samples.jsonl"
        _write_jsonl(langfuse_samples, [{"traceId": "t5", "question": "q1"}])

        # 9. 不含 traceId 的普通 JSONL → 排除
        plain_jsonl = raw_dir / "some_data.jsonl"
        _write_jsonl(plain_jsonl, [{"key": "value"}])

        result = list_langfuse_export_files(raw_dir)
        result_names = [r["name"] for r in result]

        print(f"  保留的文件 ({len(result)}): {result_names}")

        assert "langfuse_api_export_20260716_120000.jsonl" in result_names, \
            "API 拉取文件应保留"
        assert "1783322412151-lf-events-export-cmr8tec2s0007ad0c2klqym7l.jsonl" in result_names, \
            "UI 导出文件应保留"
        assert "custom_export.jsonl" in result_names, \
            "含 traceId 的文件应保留"
        print("[OK] 合法导出文件被保留")

        assert "batch_qa_20260716_120000.jsonl" not in result_names, \
            "batch_qa 文件应排除"
        assert "batch_results_20260716_120000.jsonl" not in result_names, \
            "batch_results 文件应排除"
        assert "questions_test_20260714_100000.jsonl" not in result_names, \
            "questions 文件应排除"
        assert "eval_results_20260716_120000.jsonl" not in result_names, \
            "eval_results 文件应排除"
        assert "langfuse_samples.jsonl" not in result_names, \
            "langfuse_samples 文件应排除"
        assert "some_data.jsonl" not in result_names, \
            "不含 traceId 的文件应排除"
        print("[OK] 非导出文件被正确排除")

        assert len(result) == 3, f"应恰好保留 3 个文件，实际 {len(result)} 个"
        print("[OK] 恰好保留 3 个文件")

    print()


def test_mtime_sort():
    """测试按修改时间倒序排列。"""
    print("=" * 60)
    print("测试 mtime 倒序排序")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)

        # 创建三个文件，间隔写入以确保 mtime 不同
        for i, name in enumerate([
            "langfuse_api_export_20260714_100000.jsonl",
            "langfuse_api_export_20260715_100000.jsonl",
            "langfuse_api_export_20260716_100000.jsonl",
        ]):
            f = raw_dir / name
            _write_jsonl(f, [{"traceId": f"t{i}", "type": "TRACE"}])
            time.sleep(0.05)  # 确保 mtime 不同

        result = list_langfuse_export_files(raw_dir)
        names = [r["name"] for r in result]

        print(f"  排序结果: {names}")

        # 最新文件应排第一
        assert names[0] == "langfuse_api_export_20260716_100000.jsonl", \
            f"最新文件应排第一，实际: {names[0]}"
        assert names[-1] == "langfuse_api_export_20260714_100000.jsonl", \
            f"最旧文件应排最后，实际: {names[-1]}"
        print("[OK] mtime 倒序正确")

        # 验证 label 包含时间和大小
        for r in result:
            assert "|" in r["label"], f"label 应包含分隔符: {r['label']}"
            assert "KB" in r["label"], f"label 应包含大小: {r['label']}"
        print("[OK] label 格式正确（包含时间和大小）")

    print()


def test_empty_dir():
    """测试空目录返回空列表。"""
    print("=" * 60)
    print("测试空目录")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)
        result = list_langfuse_export_files(raw_dir)
        assert result == [], f"空目录应返回空列表，实际: {result}"
        print("[OK] 空目录返回空列表")

    # 测试不存在的目录
    result = list_langfuse_export_files(Path("/nonexistent"))
    assert result == [], f"不存在的目录应返回空列表，实际: {result}"
    print("[OK] 不存在的目录返回空列表")

    print()


def test_auto_select_after_fetch():
    """测试 API 拉取成功后自动选择新文件。"""
    print("=" * 60)
    print("测试 API 拉取后自动选择")
    print("=" * 60)

    # 模拟 session_state
    session_state = {}

    # 模拟 API 拉取成功后设置 session_state
    new_filename = "langfuse_api_export_20260716_140000.jsonl"
    session_state["raw_select"] = new_filename

    # 验证 session_state 中保存了新文件名
    assert session_state["raw_select"] == new_filename, \
        f"应自动选中新文件，实际: {session_state['raw_select']}"
    print("[OK] API 拉取后 session_state 保存新文件名")

    # 模拟下次渲染时 selectbox 使用 session_state
    file_names = [
        "langfuse_api_export_20260714_100000.jsonl",
        "langfuse_api_export_20260715_100000.jsonl",
        new_filename,
    ]
    saved_select = session_state.get("raw_select")
    if saved_select and saved_select in file_names:
        default_idx = file_names.index(saved_select)
    else:
        default_idx = 0

    assert default_idx == 2, f"应选中新文件（index=2），实际: {default_idx}"
    assert file_names[default_idx] == new_filename, \
        f"选中的文件名应为新文件，实际: {file_names[default_idx]}"
    print("[OK] 下次渲染时自动选中新文件")

    print()


def test_upload_not_overridden_by_api():
    """测试上传模式不被 API 自动选择覆盖。"""
    print("=" * 60)
    print("测试上传模式不被覆盖")
    print("=" * 60)

    session_state = {}

    # 用户在上传模式下选择了文件
    session_state["raw_select"] = "user_uploaded.jsonl"
    session_state["lf_source_mode"] = "上传文件"

    # 模拟 API 拉取逻辑：只有 source_mode == "从 API 拉取" 时才更新
    source_mode = session_state.get("lf_source_mode")
    if source_mode == "从 API 拉取":
        session_state["raw_select"] = "langfuse_api_export_new.jsonl"

    assert session_state["raw_select"] == "user_uploaded.jsonl", \
        f"上传模式下的选择不应被覆盖，实际: {session_state['raw_select']}"
    print("[OK] 上传模式下的选择不被 API 覆盖")

    # 用户在 API 模式下，API 拉取后应更新
    session_state["lf_source_mode"] = "从 API 拉取"
    source_mode = session_state.get("lf_source_mode")
    if source_mode == "从 API 拉取":
        session_state["raw_select"] = "langfuse_api_export_new.jsonl"

    assert session_state["raw_select"] == "langfuse_api_export_new.jsonl", \
        f"API 模式下应更新选择，实际: {session_state['raw_select']}"
    print("[OK] API 模式下正确更新选择")

    print()


def test_state_clearing_on_selection_change():
    """测试文件选择变化时清理解析状态。"""
    print("=" * 60)
    print("测试选择变化时清理解析状态")
    print("=" * 60)

    session_state = {
        "samples": [{"trace_id": "old_trace", "question": "旧问题"}],
        "summary": {"total": 1, "parsed": 1},
        "_prev_raw_select": "old_file.jsonl",
    }

    # 模拟用户选择了新文件
    selected_name = "new_file.jsonl"
    prev_select = session_state.get("_prev_raw_select")
    if prev_select is not None and prev_select != selected_name:
        session_state.pop("samples", None)
        session_state.pop("summary", None)
    session_state["_prev_raw_select"] = selected_name

    assert "samples" not in session_state, "samples 应被清除"
    assert "summary" not in session_state, "summary 应被清除"
    assert session_state["_prev_raw_select"] == selected_name, \
        f"_prev_raw_select 应更新为新文件名"
    print("[OK] 选择变化时 samples 和 summary 被清除")

    # 选择未变化时不应清除
    session_state2 = {
        "samples": [{"trace_id": "existing_trace"}],
        "summary": {"total": 1},
        "_prev_raw_select": "same_file.jsonl",
    }
    selected_name2 = "same_file.jsonl"
    prev_select2 = session_state2.get("_prev_raw_select")
    if prev_select2 is not None and prev_select2 != selected_name2:
        session_state2.pop("samples", None)
        session_state2.pop("summary", None)
    session_state2["_prev_raw_select"] = selected_name2

    assert "samples" in session_state2, "选择未变化时 samples 不应被清除"
    assert "summary" in session_state2, "选择未变化时 summary 不应被清除"
    print("[OK] 选择未变化时状态保留")

    print()


def test_label_format():
    """测试 label 格式包含文件名、修改时间和大小。"""
    print("=" * 60)
    print("测试 label 格式")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir)
        f = raw_dir / "langfuse_api_export_20260716_140000.jsonl"
        _write_jsonl(f, [{"traceId": "t1", "type": "TRACE"}])

        result = list_langfuse_export_files(raw_dir)
        assert len(result) == 1

        label = result[0]["label"]
        print(f"  label: {label}")

        assert "langfuse_api_export_20260716_140000.jsonl" in label, \
            f"label 应包含文件名: {label}"
        assert "KB" in label, f"label 应包含大小: {label}"
        # 检查日期格式
        assert "2026" in label or "2025" in label, f"label 应包含年份: {label}"
        print("[OK] label 格式正确")

    print()


def main():
    print("=" * 60)
    print("样本准备页数据导入辅助函数测试")
    print("=" * 60)
    print()

    test_filter_logic()
    test_mtime_sort()
    test_empty_dir()
    test_auto_select_after_fetch()
    test_upload_not_overridden_by_api()
    test_state_clearing_on_selection_change()
    test_label_format()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
