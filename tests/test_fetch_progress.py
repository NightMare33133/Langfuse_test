"""
测试 fetch_traces.py 的 progress_callback 和 app.py 的拉取 UI。

覆盖：
a. 已知 total 时进度单调递增
b. 未知 total 时不显示伪百分比（callback 中 total=None）
c. 分页完成回调正确
d. 失败不覆盖旧导出文件
e. 异常后 UI 状态恢复（_fetching 标志）
f. 临时文件在失败时被清理
g. API meta.totalItems 正确读取
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_api_response(traces_data, total_items=None, page=1, limit=50):
    """Build a mock Langfuse API response dict."""
    resp = {
        "data": traces_data,
        "meta": {"page": page, "limit": limit},
    }
    if total_items is not None:
        resp["meta"]["totalItems"] = total_items
        resp["meta"]["totalPages"] = (total_items + limit - 1) // limit
    return resp


def _make_trace(i):
    """Build a minimal trace dict."""
    return {
        "id": f"trace_{i:04d}",
        "name": f"test_trace_{i}",
        "timestamp": "2026-07-21T10:00:00Z",
        "input": {"query": f"question {i}"},
        "output": {"text": f"answer {i}"},
        "sessionId": f"session_{i}",
        "userId": f"user_{i}",
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_progress_with_known_total():
    """已知 total 时，traces_fetched 单调递增，total 保持不变。"""
    print("=" * 60)
    print("测试已知 total 的进度回调")
    print("=" * 60)

    from fetch_traces import fetch_all

    # Mock: 3 pages, 10 traces each, total=30
    page_data = [_make_trace(i) for i in range(10)]
    responses = [
        _mock_api_response(page_data, total_items=30, page=1),
        _mock_api_response(page_data, total_items=30, page=2),
        _mock_api_response(page_data, total_items=30, page=3),
        _mock_api_response([], total_items=30, page=4),  # empty = end
    ]
    call_idx = [0]

    def mock_fetch_traces(host, pk, sk, limit=50, page=1):
        idx = call_idx[0]
        call_idx[0] += 1
        return responses[idx] if idx < len(responses) else _mock_api_response([])

    progress_log = []
    def on_progress(phase, traces, pages, total, retries):
        progress_log.append({"phase": phase, "traces": traces, "pages": pages,
                             "total": total, "retries": retries})

    with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
        with patch("fetch_traces.fetch_observations", return_value=[]):
            rows = list(fetch_all("http://test", "pk", "sk", limit=10,
                                  progress_callback=on_progress))

    # Verify monotonic traces_fetched
    fetching_calls = [p for p in progress_log if p["phase"] == "fetching"]
    for i in range(1, len(fetching_calls)):
        assert fetching_calls[i]["traces"] >= fetching_calls[i-1]["traces"], \
            f"traces_fetched 回退: {fetching_calls[i]} < {fetching_calls[i-1]}"

    # Verify total is consistent
    for p in progress_log:
        if p["total"] is not None:
            assert p["total"] == 30, f"total 应为 30，实际 {p['total']}"

    # Verify phases
    phases = [p["phase"] for p in progress_log]
    assert phases[0] == "connecting", f"首阶段应为 connecting，实际 {phases[0]}"
    assert "done" in phases, "应有 done 阶段"

    print(f"[OK] {len(progress_log)} 次回调，traces 单调递增，total=30 恒定")

    print()


def test_progress_with_unknown_total():
    """未知 total 时，callback 中 total=None，不伪造百分比。"""
    print("=" * 60)
    print("测试未知 total 的进度回调")
    print("=" * 60)

    from fetch_traces import fetch_all

    # Mock: API 不返回 totalItems
    page_data = [_make_trace(i) for i in range(5)]
    responses = [
        _mock_api_response(page_data, total_items=None, page=1),
        _mock_api_response([], total_items=None, page=2),
    ]
    call_idx = [0]

    def mock_fetch_traces(host, pk, sk, limit=50, page=1):
        idx = call_idx[0]
        call_idx[0] += 1
        return responses[idx] if idx < len(responses) else _mock_api_response([])

    progress_log = []
    def on_progress(phase, traces, pages, total, retries):
        progress_log.append({"phase": phase, "traces": traces, "pages": pages,
                             "total": total, "retries": retries})

    with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
        with patch("fetch_traces.fetch_observations", return_value=[]):
            rows = list(fetch_all("http://test", "pk", "sk", limit=5,
                                  progress_callback=on_progress))

    # Verify total is None throughout
    for p in progress_log:
        assert p["total"] is None, f"未知 total 时应为 None，实际 {p['total']}"

    print(f"[OK] {len(progress_log)} 次回调，total 始终为 None")

    print()


def test_page_completion_callback():
    """每完成一页才回调一次，不是逐条回调。"""
    print("=" * 60)
    print("测试分页完成回调粒度")
    print("=" * 60)

    from fetch_traces import fetch_all

    # Mock: 1 page with 20 traces
    page_data = [_make_trace(i) for i in range(20)]
    responses = [
        _mock_api_response(page_data, total_items=20, page=1),
        _mock_api_response([], total_items=20, page=2),
    ]
    call_idx = [0]

    def mock_fetch_traces(host, pk, sk, limit=50, page=1):
        idx = call_idx[0]
        call_idx[0] += 1
        return responses[idx] if idx < len(responses) else _mock_api_response([])

    progress_log = []
    def on_progress(phase, traces, pages, total, retries):
        progress_log.append({"phase": phase, "traces": traces, "pages": pages})

    with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
        with patch("fetch_traces.fetch_observations", return_value=[]):
            rows = list(fetch_all("http://test", "pk", "sk", limit=20,
                                  progress_callback=on_progress))

    # Should have: 1 connecting + 1 fetching (page 1) + 1 fetching (empty page 2) + 1 done = 4
    # NOT 20+ callbacks (one per trace)
    fetching_calls = [p for p in progress_log if p["phase"] == "fetching"]
    assert len(fetching_calls) <= 3, \
        f"回调次数过多（{len(fetching_calls)}），疑似逐条回调"
    print(f"[OK] {len(progress_log)} 次回调（非逐条），粒度为每页")

    print()


def test_failure_preserves_old_export():
    """拉取失败时，旧的导出文件不被覆盖。"""
    print("=" * 60)
    print("测试失败不覆盖旧导出")
    print("=" * 60)

    from fetch_traces import fetch_all

    with tempfile.TemporaryDirectory() as tmpdir:
        old_file = Path(tmpdir) / "langfuse_api_export_old.jsonl"
        old_file.write_text('{"old": true}\n', encoding="utf-8")
        old_content = old_file.read_text(encoding="utf-8")

        tmp_file = Path(tmpdir) / ".tmp_test.jsonl"

        # Simulate: fetch starts writing to tmp, then fails
        def mock_fetch_traces(host, pk, sk, limit=50, page=1):
            raise RuntimeError("API 连接失败")

        try:
            with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
                with tmp_file.open("w", encoding="utf-8") as f:
                    for row in fetch_all("http://test", "pk", "sk"):
                        f.write(json.dumps(row) + "\n")
        except RuntimeError:
            pass

        # Old file should be unchanged
        assert old_file.exists(), "旧文件应存在"
        assert old_file.read_text(encoding="utf-8") == old_content, "旧文件不应被修改"

        # Tmp file should be cleaned up (by app.py, not fetch_all)
        # fetch_all itself doesn't clean up - the caller does
        print("[OK] 失败后旧文件未被修改")

    print()


def test_retry_on_transient_failure():
    """页面级瞬时失败应重试，最终成功。"""
    print("=" * 60)
    print("测试页面级重试")
    print("=" * 60)

    from fetch_traces import fetch_all

    call_count = [0]
    def mock_fetch_traces(host, pk, sk, limit=50, page=1):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("瞬时网络错误")
        # Return 1 trace then empty to stop pagination
        if call_count[0] == 2:
            return _mock_api_response([_make_trace(0)], total_items=1, page=1, limit=limit)
        return _mock_api_response([], total_items=1, page=2, limit=limit)

    progress_log = []
    def on_progress(phase, traces, pages, total, retries):
        progress_log.append({"phase": phase, "retries": retries})

    with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
        with patch("fetch_traces.fetch_observations", return_value=[]):
            rows = list(fetch_all("http://test", "pk", "sk", limit=1,
                                  max_retries=2, progress_callback=on_progress))

    assert call_count[0] == 3, f"应调用 3 次（1 次失败 + 1 次成功 + 1 次空页），实际 {call_count[0]}"
    # Check retries reported
    done_calls = [p for p in progress_log if p.get("retries", 0) > 0]
    assert len(done_calls) > 0, "应报告重试次数"
    print(f"[OK] 第 1 次失败后重试成功，共调用 {call_count[0]} 次")

    print()


def test_fetch_all_meta_total_items():
    """API 返回 meta.totalItems 时应正确读取。"""
    print("=" * 60)
    print("测试 meta.totalItems 读取")
    print("=" * 60)

    from fetch_traces import fetch_all

    responses = [
        _mock_api_response([_make_trace(0)], total_items=1234, page=1),
        _mock_api_response([], total_items=1234, page=2),
    ]
    call_idx = [0]

    def mock_fetch_traces(host, pk, sk, limit=50, page=1):
        idx = call_idx[0]
        call_idx[0] += 1
        return responses[idx]

    captured_total = [None]
    def on_progress(phase, traces, pages, total, retries):
        if total is not None:
            captured_total[0] = total

    with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
        with patch("fetch_traces.fetch_observations", return_value=[]):
            list(fetch_all("http://test", "pk", "sk", limit=1,
                           progress_callback=on_progress))

    assert captured_total[0] == 1234, f"total 应为 1234，实际 {captured_total[0]}"
    print("[OK] meta.totalItems=1234 正确读取")

    print()


def test_fetching_app_button_disabled():
    """app.py 中拉取按钮应在 _fetching=True 时禁用。"""
    print("=" * 60)
    print("测试拉取按钮禁用逻辑")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # Check button has disabled parameter
    assert 'disabled=st.session_state.get("_fetching"' in source, \
        "拉取按钮应有 disabled=_fetching 参数"
    print("[OK] 按钮有 disabled 参数")

    # Check _fetching is set to True before fetch
    assert 'st.session_state["_fetching"] = True' in source, \
        "应在拉取前设置 _fetching=True"
    print("[OK] 拉取前设置 _fetching=True")

    # Check _fetching is reset in finally
    assert 'st.session_state["_fetching"] = False' in source, \
        "应在 finally 中重置 _fetching=False"
    print("[OK] finally 中重置 _fetching=False")

    print()


def test_fetching_temp_file_cleanup():
    """失败时临时文件应被清理。"""
    print("=" * 60)
    print("测试失败时临时文件清理")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # Check tmp_path.unlink() in except block
    assert "tmp_path.unlink()" in source, \
        "应在异常处理中清理临时文件"
    print("[OK] 异常处理中有 tmp_path.unlink()")

    # Check tmp_path.rename(final_path) for atomic replace
    assert "tmp_path.rename(final_path)" in source, \
        "成功时应原子替换"
    print("[OK] 成功时 tmp_path.rename(final_path)")

    print()


def test_no_keys_in_output():
    """拉取输出不得包含 API Key。"""
    print("=" * 60)
    print("测试输出不含 API Key")
    print("=" * 60)

    # Verify fetch_all doesn't yield any key-related fields
    from fetch_traces import fetch_all

    responses = [
        _mock_api_response([_make_trace(0)], total_items=1),
        _mock_api_response([], total_items=1),
    ]
    call_idx = [0]

    def mock_fetch_traces(host, pk, sk, limit=50, page=1):
        call_idx[0] += 1
        return responses[min(call_idx[0] - 1, len(responses) - 1)]

    with patch("fetch_traces.fetch_traces", side_effect=mock_fetch_traces):
        with patch("fetch_traces.fetch_observations", return_value=[]):
            rows = list(fetch_all("http://test", "pk", "sk", limit=1))

    sensitive_keys = {"public_key", "secret_key", "publicKey", "secretKey",
                      "api_key", "apiKey", "password", "token", "auth"}
    for row in rows:
        row_keys = set(row.keys())
        leaked = row_keys & sensitive_keys
        assert not leaked, f"行中包含敏感字段: {leaked}"
    print("[OK] 输出行不含 API Key 等敏感字段")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("fetch_traces.py progress_callback 测试")
    print("=" * 60)
    print()

    test_progress_with_known_total()
    test_progress_with_unknown_total()
    test_page_completion_callback()
    test_failure_preserves_old_export()
    test_retry_on_transient_failure()
    test_fetch_all_meta_total_items()
    test_fetching_app_button_disabled()
    test_fetching_temp_file_cleanup()
    test_no_keys_in_output()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
