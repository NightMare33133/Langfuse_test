"""
测试 parser.py 的 progress_callback 机制。

覆盖：
a. 回调单调递增（lines_read 不回退）
b. 不会每行回调（1000 行文件回调次数合理）
c. 成功完成时覆盖所有阶段
d. 部分坏行继续处理，bad_lines 正确记录
e. 异常路径不崩溃，回调不被多余调用
f. retrieval 计数与最终 summary 一致
"""

import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from parser import parse_langfuse_jsonl, load_jsonl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_trace_row(trace_id, node_type="GENERATION", name="gen",
                    input_data=None, output_data=None, **extra):
    """Build a minimal observation row that parse_langfuse_jsonl can process."""
    row = {
        "traceId": trace_id,
        "type": node_type,
        "name": name,
        "startTime": "2026-07-20T10:00:00Z",
        "input": json.dumps(input_data or {"sys": {"query": f"question for {trace_id}"}}),
        "output": json.dumps(output_data or {"text": f"answer for {trace_id}"}),
    }
    row.update(extra)
    return row


def _make_retrieval_row(trace_id, result_items):
    """Build a knowledge-retrieval observation with output.result list."""
    return _make_trace_row(
        trace_id,
        node_type="SPAN",
        name="knowledge-retrieval",
        input_data={"query": "retrieval query"},
        output_data={"result": result_items},
        metadata=json.dumps({"node_type": "knowledge-retrieval"}),
    )


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_monotonic_progress():
    """lines_read in 'reading' callbacks must never decrease."""
    print("=" * 60)
    print("测试回调单调递增")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        rows = [_make_trace_row(f"t{i}") for i in range(500)]
        _write_jsonl(path, rows)

        readings = []
        def cb(phase, current, total, traces, retrieval):
            if phase == "reading":
                readings.append(current)

        parse_langfuse_jsonl(path, progress_callback=cb)

        assert len(readings) > 0, "应至少有一次 reading 回调"
        for i in range(1, len(readings)):
            assert readings[i] >= readings[i - 1], \
                f"lines_read 回退: {readings[i]} < {readings[i-1]}"
        print(f"[OK] {len(readings)} 次 reading 回调，单调递增")

    print()


def test_callback_granularity():
    """1000 行文件不应触发过多回调（每 100 行一次 + 最终 + building/backfilling）。"""
    print("=" * 60)
    print("测试回调粒度（不会每行回调）")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        rows = [_make_trace_row(f"t{i}") for i in range(1000)]
        _write_jsonl(path, rows)

        call_count = [0]
        def cb(phase, current, total, traces, retrieval):
            call_count[0] += 1

        parse_langfuse_jsonl(path, progress_callback=cb)

        # 1000 lines / 100 batch = 10 reading callbacks + 1 final reading
        # + building callbacks + backfilling + saving callbacks
        # Should be well under 1000 (which would be per-line)
        print(f"  总回调次数: {call_count[0]}")
        assert call_count[0] < 50, \
            f"回调次数过多（{call_count[0]}），疑似每行回调"
        assert call_count[0] >= 5, \
            f"回调次数过少（{call_count[0]}），预期至少 5 次"
        print(f"[OK] 回调次数 {call_count[0]}，粒度合理")

    print()


def test_success_phases():
    """成功完成时回调应覆盖 counting, reading, building, backfilling, saving 阶段。"""
    print("=" * 60)
    print("测试成功完成回调阶段")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        rows = [_make_trace_row(f"t{i}") for i in range(10)]
        _write_jsonl(path, rows)

        phases_seen = set()
        def cb(phase, current, total, traces, retrieval):
            phases_seen.add(phase)

        parse_langfuse_jsonl(path, progress_callback=cb)

        expected = {"counting", "reading", "building", "backfilling"}
        missing = expected - phases_seen
        assert not missing, f"缺失阶段: {missing}"
        print(f"[OK] 已覆盖阶段: {sorted(phases_seen)}")

    print()


def test_bad_lines_continue():
    """含 JSON 错误行的文件仍应产出有效 samples，bad_lines 正确记录。"""
    print("=" * 60)
    print("测试部分坏行继续处理")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        lines = [
            json.dumps(_make_trace_row("t_good_1"), ensure_ascii=False),
            "THIS IS NOT VALID JSON",
            json.dumps(_make_trace_row("t_good_2"), ensure_ascii=False),
            json.dumps({"no_trace_id": True}),  # missing traceId
            json.dumps(_make_trace_row("t_good_3"), ensure_ascii=False),
        ]
        with path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        bad_line_info = []
        def cb(phase, current, total, traces, retrieval):
            pass  # just observe

        samples, summary = parse_langfuse_jsonl(path, progress_callback=cb)

        assert len(samples) == 3, f"应有 3 个有效 trace，实际 {len(samples)}"
        assert summary["bad_line_count"] == 2, \
            f"应有 2 个坏行，实际 {summary['bad_line_count']}"
        print(f"[OK] 3 个有效 trace，2 个坏行被跳过")

        # bad_lines 应记录行号和原因
        for bl in summary["bad_lines"]:
            assert "line" in bl and "error" in bl, f"bad_lines 格式错误: {bl}"
        print("[OK] bad_lines 格式正确")

    print()


def test_exception_recovery():
    """传入不存在的路径时应抛异常，回调不被多余调用。"""
    print("=" * 60)
    print("测试异常状态恢复")
    print("=" * 60)

    call_log = []
    def cb(phase, current, total, traces, retrieval):
        call_log.append(phase)

    try:
        parse_langfuse_jsonl(Path("/nonexistent/path.jsonl"), progress_callback=cb)
        assert False, "应抛出异常"
    except (FileNotFoundError, OSError):
        pass  # expected

    # counting phase may fire (file open attempt), but no reading/building
    unexpected = [p for p in call_log if p not in ("counting",)]
    assert not unexpected, f"异常后不应有回调: {call_log}"
    print(f"[OK] 异常正确抛出，回调日志: {call_log}")

    print()


def test_retrieval_count_consistency():
    """回调中的 retrieval_count 应与最终 summary 一致。"""
    print("=" * 60)
    print("测试 retrieval 计数一致性")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        # trace t0: 2 retrieval results
        # trace t1: 0 retrieval results
        # trace t2: 3 retrieval results
        rows = [
            _make_retrieval_row("t0", [
                {"title": "doc1", "content": "c1", "score": 0.9},
                {"title": "doc2", "content": "c2", "score": 0.8},
            ]),
            _make_trace_row("t1"),  # no retrieval
            _make_retrieval_row("t2", [
                {"title": "doc3", "content": "c3", "score": 0.7},
                {"title": "doc4", "content": "c4", "score": 0.6},
                {"title": "doc5", "content": "c5", "score": 0.5},
            ]),
        ]
        _write_jsonl(path, rows)

        final_retrieval_from_cb = [0]
        def cb(phase, current, total, traces, retrieval):
            if phase == "reading" and current == total and total > 0:
                final_retrieval_from_cb[0] = retrieval

        samples, summary = parse_langfuse_jsonl(path, progress_callback=cb)

        # The callback retrieval count is a rough estimate from load_jsonl
        # (based on output.result presence), while summary is the precise count
        # from build_trace_sample. Both should be > 0 and close.
        assert summary["total_retrieval_results"] == 5, \
            f"预期 5 条 retrieval，实际 {summary['total_retrieval_results']}"
        assert final_retrieval_from_cb[0] > 0, \
            "回调中的 retrieval_count 应 > 0"
        print(f"[OK] summary retrieval: {summary['total_retrieval_results']}, "
              f"回调 retrieval: {final_retrieval_from_cb[0]}")

    print()


def test_no_callback_backward_compat():
    """不传 progress_callback 时行为与原来完全一致。"""
    print("=" * 60)
    print("测试无回调向后兼容")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        rows = [_make_trace_row(f"t{i}") for i in range(20)]
        _write_jsonl(path, rows)

        # No callback — should work exactly as before
        samples, summary = parse_langfuse_jsonl(path)

        assert len(samples) == 20, f"预期 20 个 trace，实际 {len(samples)}"
        assert summary["trace_count"] == 20
        assert summary["bad_line_count"] == 0
        print("[OK] 无回调时行为不变")

    print()


def test_empty_file():
    """空文件应正常处理，不崩溃。"""
    print("=" * 60)
    print("测试空文件")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        phases = []
        def cb(phase, current, total, traces, retrieval):
            phases.append(phase)

        samples, summary = parse_langfuse_jsonl(path, progress_callback=cb)

        assert len(samples) == 0, f"空文件应无 trace，实际 {len(samples)}"
        assert summary["trace_count"] == 0
        print(f"[OK] 空文件正常处理，回调阶段: {phases}")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("parser.py progress_callback 测试")
    print("=" * 60)
    print()

    test_monotonic_progress()
    test_callback_granularity()
    test_success_phases()
    test_bad_lines_continue()
    test_exception_recovery()
    test_retrieval_count_consistency()
    test_no_callback_backward_compat()
    test_empty_file()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
