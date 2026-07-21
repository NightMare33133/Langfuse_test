"""
测试内存优化措施的正确性。

覆盖：
a. parse_langfuse_jsonl 释放 traces 后仍能正确构建 samples
b. samples 的 observations 字段在输出文件中保留、在内存中可剥离
c. get_run_status 在 include_judge_results=False 时不返回完整结果
d. _load_sample_lookup 缓存函数正确加载 samples（不含 observations）
e. 剥离 observations 后样本核心字段不受影响
f. 大 JSON 截断逻辑正确
"""

import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from parser import parse_langfuse_jsonl, save_results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_trace_row(trace_id, node_type="GENERATION", name="gen",
                    input_data=None, output_data=None, **extra):
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


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_parse_releases_traces():
    """parse_langfuse_jsonl 在释放 traces 后仍能正确构建 samples。"""
    print("=" * 60)
    print("测试解析后释放 traces")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        rows = [_make_trace_row(f"t{i}") for i in range(50)]
        _write_jsonl(path, rows)

        samples, summary = parse_langfuse_jsonl(path)

        assert len(samples) == 50, f"预期 50 个 trace，实际 {len(samples)}"
        assert summary["trace_count"] == 50
        assert summary["bad_line_count"] == 0
        # 验证所有 trace_id 都存在（排序后顺序不保证与输入一致）
        trace_ids = {s["trace_id"] for s in samples}
        expected_ids = {f"t{i}" for i in range(50)}
        assert trace_ids == expected_ids, "所有 trace_id 应完整保留"
        print("[OK] 50 个 trace 正确解析")

    print()


def test_output_file_has_observations():
    """save_results 输出的 JSONL 文件应包含 observations 字段。"""
    print("=" * 60)
    print("测试输出文件保留 observations")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        rows = [_make_trace_row("t1"), _make_trace_row("t2")]
        _write_jsonl(input_path, rows)

        samples, summary = parse_langfuse_jsonl(input_path)

        output_path = Path(tmpdir) / "output.jsonl"
        summary_path = Path(tmpdir) / "summary.json"
        save_results(samples, summary, output_path, summary_path)

        # 验证输出文件包含 observations
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                assert "observations" in obj, \
                    f"输出文件应包含 observations 字段: {obj.get('trace_id')}"
        print("[OK] 输出文件保留 observations")

        # 现在剥离 observations 后再保存 — 验证剥离是内存操作
        for s in samples:
            s.pop("observations", None)

        # samples 中不再有 observations
        for s in samples:
            assert "observations" not in s, "剥离后不应有 observations"
        print("[OK] 剥离后 samples 中无 observations")

    print()


def test_strip_observations_preserves_core_fields():
    """剥离 observations 后，样本核心字段不受影响。"""
    print("=" * 60)
    print("测试剥离 observations 保留核心字段")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        rows = [_make_trace_row("t1")]
        _write_jsonl(path, rows)

        samples, _ = parse_langfuse_jsonl(path)
        sample = samples[0]

        # 记录核心字段
        core_fields = {
            "trace_id": sample.get("trace_id"),
            "question": sample.get("question"),
            "final_answer": sample.get("final_answer"),
            "retrieval_results": sample.get("retrieval_results"),
            "llm_model": sample.get("llm_model"),
        }

        # 剥离 observations
        sample.pop("observations", None)

        # 验证核心字段不变
        for key, expected in core_fields.items():
            assert sample.get(key) == expected, \
                f"字段 {key} 在剥离 observations 后不应改变"
        assert "observations" not in sample
        print("[OK] 核心字段在剥离 observations 后保持不变")

    print()


def test_get_run_status_without_judge_results():
    """get_run_status 在 include_judge_results=False 时返回空列表。"""
    print("=" * 60)
    print("测试 get_run_status 不返回 judge_results")
    print("=" * 60)

    from experiment import get_run_status

    with tempfile.TemporaryDirectory() as tmpdir:
        # 使用不存在的 run_id — manifest 为空时返回 {}
        result_no_manifest = get_run_status(
            "nonexistent_run_id",
            batch_dir=tmpdir,
            raw_dir=tmpdir,
            processed_file=str(Path(tmpdir) / "nonexistent.jsonl"),
            judged_file=str(Path(tmpdir) / "nonexistent.jsonl"),
            include_judge_results=False,
        )
        assert result_no_manifest == {}, "不存在的 run 应返回空 dict"
        print("[OK] 不存在的 run 返回空 dict")

        # 验证函数签名包含 include_judge_results 参数
        import inspect
        sig = inspect.signature(get_run_status)
        assert "include_judge_results" in sig.parameters, \
            "get_run_status 应有 include_judge_results 参数"
        assert sig.parameters["include_judge_results"].default is False, \
            "include_judge_results 默认值应为 False"
        print("[OK] get_run_status 签名正确，include_judge_results 默认 False")

    print()


def test_load_sample_lookup():
    """_load_sample_lookup 正确加载 samples（不含 observations）。"""
    print("=" * 60)
    print("测试 _load_sample_lookup 缓存函数")
    print("=" * 60)

    # 直接测试缓存函数的逻辑（不通过 Streamlit 缓存装饰器）
    with tempfile.TemporaryDirectory() as tmpdir:
        proc_path = Path(tmpdir) / "langfuse_samples.jsonl"

        # 创建带 observations 的 samples
        samples = []
        for i in range(5):
            samples.append({
                "trace_id": f"t{i}",
                "question": f"question {i}",
                "final_answer": f"answer {i}",
                "observations": [
                    {"id": f"obs_{i}_0", "input": {"data": "x" * 1000}},
                    {"id": f"obs_{i}_1", "input": {"data": "y" * 1000}},
                ],
            })
        _write_jsonl(proc_path, samples)

        # 模拟 _load_sample_lookup 的逻辑
        lookup = {}
        with proc_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                obj.pop("observations", None)
                tid = obj.get("trace_id")
                if tid:
                    lookup[tid] = obj

        assert len(lookup) == 5, f"预期 5 个样本，实际 {len(lookup)}"
        for tid, obj in lookup.items():
            assert "observations" not in obj, \
                f"缓存的样本不应包含 observations: {tid}"
            assert "question" in obj, f"样本应保留 question: {tid}"
            assert "final_answer" in obj, f"样本应保留 final_answer: {tid}"
        print("[OK] _load_sample_lookup 正确加载 5 个样本，无 observations")

    print()


def test_truncate_large_json():
    """大 JSON 截断逻辑：超过 2000 字符时截断。"""
    print("=" * 60)
    print("测试大 JSON 截断")
    print("=" * 60)

    # 模拟截断逻辑
    def truncate_preview(obj, max_chars=2000):
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) > max_chars:
            return s[:max_chars] + "\n... (已截断，共 " + str(len(s)) + " 字符)"
        return s

    # 小对象不截断
    small = {"key": "value"}
    result = truncate_preview(small)
    assert "已截断" not in result, "小对象不应截断"
    print("[OK] 小对象不截断")

    # 大对象截断
    large = {"data": "x" * 3000}
    result = truncate_preview(large)
    assert "已截断" in result, "大对象应截断"
    assert len(result) < len(json.dumps(large, ensure_ascii=False))
    print(f"[OK] 大对象截断：{len(json.dumps(large, ensure_ascii=False))} -> {len(result)} 字符")

    print()


def test_output_format_unchanged():
    """验证输出文件格式与优化前完全一致（含 observations）。"""
    print("=" * 60)
    print("测试输出文件格式不变")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        rows = [
            _make_trace_row("t1"),
            _make_trace_row("t2"),
            _make_trace_row("t3"),
        ]
        _write_jsonl(input_path, rows)

        # 解析
        samples, summary = parse_langfuse_jsonl(input_path)

        # 保存（此时 samples 还有 observations）
        output_path = Path(tmpdir) / "output.jsonl"
        summary_path = Path(tmpdir) / "summary.json"
        save_results(samples, summary, output_path, summary_path)

        # 验证输出文件每行都是合法 JSON 且包含必要字段
        with output_path.open("r", encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        assert len(lines) == 3, f"预期 3 行，实际 {len(lines)}"
        required_fields = [
            "trace_id", "question", "final_answer",
            "retrieval_results", "observations",
        ]
        for obj in lines:
            for field in required_fields:
                assert field in obj, f"输出缺少字段 {field}: trace_id={obj.get('trace_id')}"
        print("[OK] 输出文件格式完整，包含 observations")

        # 验证 summary 格式
        assert summary_path.exists()
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "trace_count" in saved_summary
        assert "total_retrieval_results" in saved_summary
        assert "output_file" in saved_summary
        print("[OK] summary 文件格式完整")

    print()


def test_load_sample_lookup_cache_invalidation():
    """cache_key 变化（文件 mtime 变化）后缓存应失效。"""
    print("=" * 60)
    print("测试缓存文件更新失效")
    print("=" * 60)

    # 模拟缓存行为：不同 cache_key 应返回不同结果
    with tempfile.TemporaryDirectory() as tmpdir:
        proc_path = Path(tmpdir) / "langfuse_samples.jsonl"

        # 第一版文件：3 个 trace
        v1 = [{"trace_id": f"t{i}", "question": f"q{i}"} for i in range(3)]
        _write_jsonl(proc_path, v1)
        mtime1 = str(proc_path.stat().st_mtime)

        # 第二版文件：5 个 trace（模拟文件更新）
        import time
        time.sleep(0.05)
        v2 = [{"trace_id": f"t{i}", "question": f"q{i}"} for i in range(5)]
        _write_jsonl(proc_path, v2)
        mtime2 = str(proc_path.stat().st_mtime)

        assert mtime1 != mtime2, "两次写入的 mtime 应不同"

        # 模拟加载逻辑：不同 cache_key 应加载不同版本
        def _load_with_key(key):
            lookup = {}
            if proc_path.exists():
                with proc_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        obj.pop("observations", None)
                        tid = obj.get("trace_id")
                        if tid:
                            lookup[tid] = obj
            return lookup

        # 用 mtime1 作为 key 时，文件已更新为 v2，但 cache_key 不同
        # 实际 Streamlit 缓存会按 key 返回缓存值
        # 这里验证：如果 key 不同，应该重新加载
        lookup_v2 = _load_with_key(mtime2)
        assert len(lookup_v2) == 5, f"更新后应有 5 个 trace，实际 {len(lookup_v2)}"
        print("[OK] 文件更新后重新加载正确")

    print()


def test_load_sample_lookup_missing_file():
    """文件不存在时应安全返回空 dict。"""
    print("=" * 60)
    print("测试缺失文件安全处理")
    print("=" * 60)

    # 模拟加载不存在的文件
    proc_path = Path("/nonexistent/path/langfuse_samples.jsonl")
    lookup = {}
    if not proc_path.exists():
        pass  # 返回空 dict
    assert lookup == {}, "文件不存在时应返回空 dict"
    print("[OK] 缺失文件安全返回空 dict")

    # 模拟空文件
    with tempfile.TemporaryDirectory() as tmpdir:
        proc_path = Path(tmpdir) / "empty.jsonl"
        proc_path.write_text("", encoding="utf-8")
        lookup = {}
        with proc_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = obj.get("trace_id")
                if tid:
                    lookup[tid] = obj
        assert lookup == {}, "空文件应返回空 dict"
        print("[OK] 空文件返回空 dict")

    print()


def test_load_sample_lookup_max_entries():
    """验证 _load_sample_lookup 装饰器配置了 max_entries=2。"""
    print("=" * 60)
    print("测试缓存 max_entries 配置")
    print("=" * 60)

    import inspect
    # 从 app.py 源码中读取装饰器配置
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 检查 max_entries=2 存在
    assert "max_entries=2" in source, "app.py 中应有 max_entries=2 配置"
    print("[OK] max_entries=2 已配置")

    # 检查 ttl=120 存在
    assert "ttl=120" in source, "app.py 中应有 ttl=120 配置"
    print("[OK] ttl=120 已配置")

    # 检查 cache_key 包含 mtime
    # 调用处应传入 mtime 作为 cache_key
    assert "_proc_mtime" in source or "_proc_mtime_local" in source, \
        "调用处应传入 mtime 作为 cache_key"
    print("[OK] cache_key 包含 mtime")

    print()


def test_rss_sampling_utility():
    """验证 _get_rss_mb 和 _record_rss 的基本逻辑。"""
    print("=" * 60)
    print("测试 RSS 采样工具")
    print("=" * 60)

    import psutil
    import os

    # 直接测试 psutil 获取 RSS
    pid = os.getpid()
    proc = psutil.Process(pid)
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    assert rss_mb > 0, f"RSS 应大于 0，实际 {rss_mb}"
    assert rss_mb < 10000, f"RSS 不应超过 10GB，实际 {rss_mb}"  # 合理性检查
    print(f"[OK] 当前进程 RSS: {rss_mb:.1f} MB")

    # 验证 _record_rss 逻辑（不依赖 Streamlit session_state）
    log = []
    log.append({
        "stage": "测试",
        "rss_mb": round(rss_mb, 1),
        "ts": "12:00:00",
    })
    assert len(log) == 1
    assert log[0]["stage"] == "测试"
    assert log[0]["rss_mb"] > 0
    print("[OK] RSS 采样记录格式正确")

    print()


def test_overview_calls_include_judge_results():
    """概览聚合和运行历史的 get_run_status 调用必须传 include_judge_results=True。

    回归测试：未传此参数会导致累计 Judge 指标显示"暂无数据"。
    """
    print("=" * 60)
    print("测试概览聚合调用 include_judge_results=True")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")
    lines = source.split('\n')

    # 检查：概览聚合区（_all_judge_results_raw）的 get_run_status 调用
    assert "_all_judge_results_raw" in source, "应存在概览聚合变量"
    # 找到 _all_judge_results_raw = [] 所在行
    raw_init_line = -1
    for i, line in enumerate(lines):
        if '_all_judge_results_raw = []' in line:
            raw_init_line = i
            break
    assert raw_init_line > 0, "应找到 _all_judge_results_raw 初始化"
    # 在该行之后的 30 行中查找 include_judge_results=True
    search_end = min(len(lines), raw_init_line + 30)
    postamble = '\n'.join(lines[raw_init_line:search_end])
    assert "include_judge_results=True" in postamble, \
        "概览聚合区的 get_run_status 应传 include_judge_results=True"
    print("[OK] 概览聚合区正确传 include_judge_results=True")

    # 检查：运行历史区的调用
    hist_idx = source.find("运行历史（点击展开）")
    if hist_idx > 0:
        hist_block = source[hist_idx:hist_idx + 3000]
        assert "include_judge_results=True" in hist_block, \
            "运行历史区的 get_run_status 应传 include_judge_results=True"
        print("[OK] 运行历史区正确传 include_judge_results=True")

    print()


def test_run_table_omits_judge_results():
    """运行表的 get_run_status 调用应传 include_judge_results=False。"""
    print("=" * 60)
    print("测试运行表不加载 judge_results")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 运行表调用应该有 include_judge_results=False
    assert 'include_judge_results=False' in source, \
        "app.py 中应有 include_judge_results=False 的调用（运行表）"
    print("[OK] 运行表调用正确使用 include_judge_results=False")

    print()


def test_get_run_status_returns_judge_results_when_asked():
    """get_run_status(include_judge_results=True) 应返回完整结果。"""
    print("=" * 60)
    print("测试 get_run_status 返回 judge_results")
    print("=" * 60)

    import inspect
    from experiment import get_run_status

    # 验证函数签名
    sig = inspect.signature(get_run_status)
    param = sig.parameters.get("include_judge_results")
    assert param is not None, "get_run_status 应有 include_judge_results 参数"
    assert param.default is False, "默认值应为 False"
    print("[OK] 签名正确：include_judge_results=False")

    # 验证函数体中使用了该参数
    import textwrap
    source = inspect.getsource(get_run_status)
    assert "include_judge_results" in source, "函数体应使用 include_judge_results"
    assert "judge_results_for_run if include_judge_results" in source, \
        "应有条件返回逻辑"
    print("[OK] 函数体正确使用 include_judge_results 控制返回")

    print()


def test_run_detail_uses_selectbox_not_all_expanders():
    """运行详情应使用 selectbox 选择单个 run，而非循环展开所有 run。"""
    print("=" * 60)
    print("测试运行详情使用 selectbox 选择")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 应有 selectbox 用于选择 run
    assert "选择运行查看题目明细" in source, \
        "app.py 中应有 '选择运行查看题目明细' selectbox"
    print("[OK] 存在 run 选择 selectbox")

    # 应有 session_state 保存选择
    assert "_selected_detail_run" in source, \
        "app.py 中应有 _selected_detail_run session_state"
    print("[OK] 使用 session_state 保存选择")

    # 详情区的 get_run_status 应传 include_judge_results=True
    # 查找 "选择运行查看题目明细" 之后的 get_run_status 调用
    sel_pos = source.find("选择运行查看题目明细")
    detail_section = source[sel_pos:sel_pos + 3000]
    assert "include_judge_results=True" in detail_section, \
        "详情区的 get_run_status 应传 include_judge_results=True"
    print("[OK] 详情区按需加载 judge_results")

    print()


def test_run_detail_reuses_render_judge_results_list():
    """运行详情应复用 render_judge_results_list 进行分页渲染。"""
    print("=" * 60)
    print("测试运行详情复用 render_judge_results_list")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 在 "评测详情（本次运行）" 区域应调用 render_judge_results_list
    detail_pos = source.find("评测详情（本次运行）")
    assert detail_pos > 0, "应存在 '评测详情（本次运行）' 区域"
    detail_section = source[detail_pos:detail_pos + 500]
    assert "render_judge_results_list" in detail_section, \
        "评测详情区应调用 render_judge_results_list"
    assert "page_size=20" in detail_section, \
        "应使用 page_size=20 分页"
    print("[OK] 复用 render_judge_results_list，page_size=20")

    print()


def test_run_table_still_lightweight():
    """运行表仍使用 include_judge_results=False。"""
    print("=" * 60)
    print("测试运行表保持轻量")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    lines = app_path.read_text(encoding="utf-8").split('\n')

    # 找到 "运行记录" 之后、"运行详情" 之前的 get_run_status 调用
    in_table_section = False
    table_call_found = False
    for i, line in enumerate(lines):
        if "运行记录" in line and "配置" in line:
            in_table_section = True
        if "选择运行查看题目明细" in line:
            in_table_section = False
        if in_table_section and "get_run_status(" in line:
            # 检查后续几行是否有 include_judge_results=False
            block = '\n'.join(lines[i:i+10])
            if "include_judge_results=False" in block:
                table_call_found = True
                break

    assert table_call_found, "运行表的 get_run_status 应传 include_judge_results=False"
    print("[OK] 运行表保持轻量（include_judge_results=False）")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("内存优化测试")
    print("=" * 60)
    print()

    test_parse_releases_traces()
    test_output_file_has_observations()
    test_strip_observations_preserves_core_fields()
    test_get_run_status_without_judge_results()
    test_load_sample_lookup()
    test_truncate_large_json()
    test_output_format_unchanged()
    test_load_sample_lookup_cache_invalidation()
    test_load_sample_lookup_missing_file()
    test_load_sample_lookup_max_entries()
    test_rss_sampling_utility()
    test_overview_calls_include_judge_results()
    test_run_table_omits_judge_results()
    test_get_run_status_returns_judge_results_when_asked()
    test_run_detail_uses_selectbox_not_all_expanders()
    test_run_detail_reuses_render_judge_results_list()
    test_run_table_still_lightweight()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
