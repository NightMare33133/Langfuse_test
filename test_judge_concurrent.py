"""
Judge 并发评测测试。

验证：
1. max_workers=1 与串行行为一致
2. 多线程下结果保持原始顺序
3. 单条失败不影响其他 futures
4. 预筛样本不提交 executor
5. 最大同时运行数不超过设置值
6. judge_llm_call_count / 成功 / 失败 / 跳过数正确

不调用真实 API。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import threading
import time
from unittest.mock import patch

from judge import (
    judge_all, _retry_call_llm, call_llm,
    TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE,
)


# ====== 辅助函数 ======

_VALID_GROUNDED_QA_RESPONSE = '{"answer_correct": 1, "reason": "测试原因"}'
_VALID_RETRIEVAL_RESPONSE = (
    '{"retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, '
    '"hit_evidence_position": 1, "reason": "测试原因"}'
)


def _make_grounded_qa_sample(idx):
    """创建一个 grounded_qa 样本（需要 LLM 调用）。"""
    return {
        "trace_id": f"trace_{idx}",
        "question": f"测试问题 {idx}",
        "question_mode": "qa",
        "retrieval_query": f"查询 {idx}",
        "retrieval_results": [
            {"position": 1, "score": 0.9, "content": f"结果内容 {idx}", "document_name": "doc"},
        ],
        "final_answer": f"测试回答 {idx}",
    }


def _make_retrieval_sample(idx):
    """创建一个 retrieval 样本（需要 LLM 调用）。"""
    return {
        "trace_id": f"trace_{idx}",
        "question": f"检索问题 {idx}",
        "question_mode": "retrieval",
        "retrieval_query": f"查询 {idx}",
        "retrieval_results": [
            {"position": 1, "score": 0.9, "content": f"结果内容 {idx}", "document_name": "doc"},
        ],
        "final_answer": f"回答 {idx}",
        "source_excerpt": f"来源摘录 {idx}",
    }


def _make_prescreen_sample(idx):
    """创建一个会被规则预筛的样本（无检索结果）。"""
    return {
        "trace_id": f"trace_{idx}",
        "question": f"预筛问题 {idx}",
        "question_mode": "qa",
        "retrieval_query": "",
        "retrieval_results": [],
        "final_answer": "",
    }


def _mock_call_llm_factory(response_text=_VALID_GROUNDED_QA_RESPONSE, delay=0):
    """创建一个 mock call_llm，支持可控延迟。"""
    call_count = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            call_count[0] += 1
        if delay > 0:
            time.sleep(delay)
        return response_text

    return mock_call_llm, call_count


def _mock_call_llm_with_errors(response_map, default_response=_VALID_GROUNDED_QA_RESPONSE):
    """创建一个 mock call_llm，根据调用次数返回不同结果。

    response_map: {调用序号(从1开始): response_text 或 Exception}
    """
    call_count = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            call_count[0] += 1
            current = call_count[0]
        action = response_map.get(current, default_response)
        if isinstance(action, Exception):
            raise action
        return action

    return mock_call_llm, call_count


# ====== 测试函数 ======

def test_concurrency_1_matches_serial():
    """max_workers=1 时结果与串行逻辑一致。"""
    print("=" * 60)
    print("测试：max_workers=1 与串行一致")
    print("=" * 60)

    samples = [
        _make_prescreen_sample(0),       # 预筛
        _make_grounded_qa_sample(1),     # LLM
        _make_grounded_qa_sample(2),     # LLM
        _make_prescreen_sample(3),       # 预筛
        _make_grounded_qa_sample(4),     # LLM
    ]

    mock_llm, call_count = _mock_call_llm_factory()

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 max_workers=1))

    assert len(results) == 5, f"期望 5 条结果，实际 {len(results)}"
    for i, r in enumerate(results):
        assert r["trace_id"] == f"trace_{i}", \
            f"结果 {i} trace_id 不对: {r['trace_id']}"

    assert results[0].get("_prescreened"), "样本 0 应为预筛"
    assert results[3].get("_prescreened"), "样本 3 应为预筛"
    assert not results[1].get("_prescreened"), "样本 1 不应为预筛"
    assert call_count[0] == 3, f"期望 3 次 LLM 调用，实际 {call_count[0]}"

    print("[OK] max_workers=1 结果顺序和标记正确")
    print(f"[OK] LLM 调用次数: {call_count[0]}")


def test_order_preserved_with_concurrency():
    """多线程乱序完成时，结果仍按输入顺序。"""
    print("=" * 60)
    print("测试：并发下结果顺序保持")
    print("=" * 60)

    n = 10
    samples = [_make_grounded_qa_sample(i) for i in range(n)]

    # 不同延迟模拟乱序完成
    delays = [0.08, 0.02, 0.06, 0.01, 0.07, 0.03, 0.05, 0.09, 0.04, 0.06]
    call_idx = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            idx = call_idx[0]
            call_idx[0] += 1
        time.sleep(delays[idx % len(delays)])
        return _VALID_GROUNDED_QA_RESPONSE

    progress_dones = []
    def on_progress(done, total, result, info):
        progress_dones.append(done)

    with patch('judge.call_llm', side_effect=mock_call_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=4))

    assert len(results) == n, f"期望 {n} 条结果，实际 {len(results)}"
    for i, r in enumerate(results):
        assert r["trace_id"] == f"trace_{i}", \
            f"位置 {i} 期望 trace_{i}，实际 {r['trace_id']}"

    assert progress_dones == list(range(1, n + 1)), \
        f"progress 回调顺序不对: {progress_dones}"

    print(f"[OK] {n} 条结果全部按原始顺序排列")
    print(f"[OK] progress 回调顺序: {progress_dones}")


def test_single_error_does_not_abort_others():
    """一个 worker 抛错不影响其他样本。"""
    print("=" * 60)
    print("测试：单条失败不影响其他样本")
    print("=" * 60)

    samples = [_make_grounded_qa_sample(i) for i in range(4)]

    error_resp = RuntimeError("HTTP 401 | 测试错误")
    # 第 3 次调用失败（样本索引 2），其余成功
    response_map = {3: error_resp}
    mock_llm, call_count = _mock_call_llm_with_errors(response_map)

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 max_workers=2))

    assert len(results) == 4, f"期望 4 条结果，实际 {len(results)}"

    # 顺序不变
    for i, r in enumerate(results):
        assert r["trace_id"] == f"trace_{i}"

    # 找到出错的结果
    errors = [r for r in results if "error" in r]
    successes = [r for r in results if "error" not in r]
    assert len(errors) == 1, f"期望 1 条错误，实际 {len(errors)}"
    assert len(successes) == 3, f"期望 3 条成功，实际 {len(successes)}"
    assert "401" in errors[0]["error"], f"错误信息应包含 401: {errors[0]['error']}"

    print(f"[OK] 1 条错误，3 条成功，顺序正确")


def test_prescreen_skips_executor():
    """预筛样本不提交 executor，只计入 immediate_results。"""
    print("=" * 60)
    print("测试：预筛样本不提交 executor")
    print("=" * 60)

    # 3 个预筛 + 2 个需要 LLM
    samples = [
        _make_prescreen_sample(0),
        _make_prescreen_sample(1),
        _make_grounded_qa_sample(2),
        _make_prescreen_sample(3),
        _make_grounded_qa_sample(4),
    ]

    submitted_to_executor = []
    original_submit = None

    class TrackingExecutor:
        """包装 ThreadPoolExecutor，记录提交的样本索引。"""
        def __init__(self, max_workers=None):
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        def submit(self, fn, *args):
            # fn 是 _worker(idx, sample)，第一个参数是 idx
            submitted_to_executor.append(args[0])
            return self._executor.submit(fn, *args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._executor.shutdown(wait=True)

    mock_llm, call_count = _mock_call_llm_factory()

    with patch('judge.call_llm', side_effect=mock_llm), \
         patch('judge.ThreadPoolExecutor', TrackingExecutor):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 max_workers=4))

    assert len(results) == 5
    assert call_count[0] == 2, f"期望 2 次 LLM 调用，实际 {call_count[0]}"

    # 只有样本 2 和 4 被提交到 executor
    assert submitted_to_executor == [2, 4], \
        f"提交到 executor 的索引: {submitted_to_executor}"

    # 预筛样本有标记
    for i in (0, 1, 3):
        assert results[i].get("_prescreened"), f"样本 {i} 应为预筛"
    for i in (2, 4):
        assert not results[i].get("_prescreened"), f"样本 {i} 不应为预筛"

    print(f"[OK] 仅 {submitted_to_executor} 提交到 executor")
    print(f"[OK] LLM 调用次数: {call_count[0]}")


def test_max_concurrency_respected():
    """最大同时运行数不超过设置值。"""
    print("=" * 60)
    print("测试：最大并发数不超过设置值")
    print("=" * 60)

    n = 8
    samples = [_make_grounded_qa_sample(i) for i in range(n)]
    max_workers = 3

    concurrent_count = [0]
    max_concurrent = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            concurrent_count[0] += 1
            if concurrent_count[0] > max_concurrent[0]:
                max_concurrent[0] = concurrent_count[0]
        time.sleep(0.05)  # 模拟 I/O 延迟
        with lock:
            concurrent_count[0] -= 1
        return _VALID_GROUNDED_QA_RESPONSE

    with patch('judge.call_llm', side_effect=mock_call_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 max_workers=max_workers))

    assert len(results) == n
    assert max_concurrent[0] <= max_workers, \
        f"最大并发 {max_concurrent[0]} 超过设置值 {max_workers}"

    print(f"[OK] 最大同时运行数: {max_concurrent[0]}（限制: {max_workers}）")


def test_manifest_counts_correct():
    """LLM 调用数、成功/失败/跳过数正确。"""
    print("=" * 60)
    print("测试：调用计数正确")
    print("=" * 60)

    # 2 预筛 + 4 LLM（其中 1 个失败）+ 1 内容复用（与第 1 个 LLM 样本内容相同）
    samples = [
        _make_prescreen_sample(0),       # 预筛
        _make_prescreen_sample(1),       # 预筛
        _make_grounded_qa_sample(2),     # LLM 成功（首个 LLM）
        _make_grounded_qa_sample(3),     # LLM 成功
        _make_grounded_qa_sample(4),     # LLM 失败
        _make_grounded_qa_sample(5),     # LLM 成功
    ]
    # 让样本 3 与样本 2 内容完全相同（trace_id 不同）
    # 在 max_workers=1 串行模式下，样本 3 会命中内容缓存
    samples[3]["question"] = samples[2]["question"]
    samples[3]["retrieval_query"] = samples[2]["retrieval_query"]
    samples[3]["retrieval_results"] = samples[2]["retrieval_results"]
    samples[3]["final_answer"] = samples[2]["final_answer"]

    error_resp = RuntimeError("HTTP 503 | 测试服务不可用")
    # 样本 4（第 3 次 LLM 调用）失败，其余成功
    response_map = {3: error_resp}
    mock_llm, call_count = _mock_call_llm_with_errors(response_map)

    # 使用 max_workers=1 串行模式，确保内容去重正常工作
    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 max_workers=1))

    assert len(results) == 6, f"期望 6 条结果，实际 {len(results)}"

    # 统计
    prescreened = sum(1 for r in results if r.get("_prescreened"))
    cached = sum(1 for r in results if r.get("_content_cached"))
    errors = sum(1 for r in results if "error" in r)

    assert prescreened == 2, f"期望 2 条预筛，实际 {prescreened}"
    assert cached == 1, f"期望 1 条内容复用，实际 {cached}"
    assert call_count[0] == 3, f"期望 3 次 LLM 调用，实际 {call_count[0]}"
    assert errors == 1, f"期望 1 条错误，实际 {errors}"

    print(f"[OK] 预筛: {prescreened}, 内容复用: {cached}")
    print(f"[OK] LLM 调用: {call_count[0]}, 失败: {errors}")


def test_retry_on_429():
    """429 错误自动重试成功。"""
    print("=" * 60)
    print("测试：429 自动重试")
    print("=" * 60)

    call_count = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            call_count[0] += 1
            current = call_count[0]
        if current <= 2:
            err = RuntimeError("HTTP 429 | Too Many Requests")
            err.status_code = 429
            err.retry_after = None
            raise err
        return '{"answer_correct": 1, "reason": "ok"}'

    with patch('judge.call_llm', side_effect=mock_call_llm):
        result = _retry_call_llm("test prompt", "key", "http://fake", "model",
                                 timeout=10, max_retries=3, base_delay=0.01)

    assert call_count[0] == 3, f"期望 3 次调用，实际 {call_count[0]}"
    assert "answer_correct" in result
    print(f"[OK] 429 重试 {call_count[0]} 次后成功")


def test_retry_exhaustion_raises():
    """重试耗尽后正确抛出异常。"""
    print("=" * 60)
    print("测试：重试耗尽抛异常")
    print("=" * 60)

    call_count = [0]

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        call_count[0] += 1
        err = RuntimeError("HTTP 503 | Service Unavailable")
        err.status_code = 503
        err.retry_after = None
        raise err

    with patch('judge.call_llm', side_effect=mock_call_llm):
        try:
            _retry_call_llm("test prompt", "key", "http://fake", "model",
                            timeout=10, max_retries=2, base_delay=0.01)
            assert False, "应该抛出异常"
        except RuntimeError as e:
            assert "503" in str(e)

    assert call_count[0] == 3, f"期望 3 次调用（1+2重试），实际 {call_count[0]}"
    print(f"[OK] 重试耗尽 {call_count[0]} 次后正确抛出异常")


def test_content_cache_dedup_with_concurrency():
    """串行模式下内容去重有效；并发模式下去重仅跨 Phase 1/2 边界生效。"""
    print("=" * 60)
    print("测试：内容去重（串行模式）")
    print("=" * 60)

    # 样本 0,1,2 内容不同，样本 3,4,5 分别与 0,1,2 内容相同
    samples = []
    for i in range(3):
        s = _make_grounded_qa_sample(i)
        samples.append(s)
    for i in range(3):
        s = _make_grounded_qa_sample(i + 3)
        s["question"] = samples[i]["question"]
        s["retrieval_query"] = samples[i]["retrieval_query"]
        s["retrieval_results"] = samples[i]["retrieval_results"]
        s["final_answer"] = samples[i]["final_answer"]
        samples.append(s)

    # 串行模式：Phase 2 处理样本 2 时缓存结果，样本 5（同内容）命中缓存
    mock_llm, call_count = _mock_call_llm_factory()

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 max_workers=1))

    assert len(results) == 6
    assert call_count[0] == 3, f"期望 3 次 LLM 调用，实际 {call_count[0]}"

    cached = [r for r in results if r.get("_content_cached")]
    assert len(cached) == 3, f"期望 3 条内容复用，实际 {len(cached)}"

    print(f"[OK] LLM 调用: {call_count[0]}, 内容复用: {len(cached)}")

    print()
    print("=" * 60)
    print("测试：内容去重（并发模式）")
    print("=" * 60)

    # 并发模式：Phase 1 中所有样本都进入 llm_queue（缓存为空），
    # Phase 2 中先完成的样本填充缓存，但后完成的同内容样本已在 executor 中。
    # 因此并发模式下同一 Phase 内的去重不生效，这是预期行为。
    mock_llm2, call_count2 = _mock_call_llm_factory()

    with patch('judge.call_llm', side_effect=mock_llm2):
        results2 = list(judge_all(samples, "key", "http://fake", "model",
                                  max_workers=4))

    assert len(results2) == 6
    # 并发模式下 6 个样本都进了 llm_queue，但部分可能因缓存而跳过
    assert call_count2[0] <= 6, f"LLM 调用不应超过 6，实际 {call_count2[0]}"

    print(f"[OK] 并发模式 LLM 调用: {call_count2[0]}（≤6）")


def test_progress_fires_during_as_completed():
    """并发场景中，第一个 future 完成时 progress_callback 已被调用，不能等待全部完成。"""
    print("=" * 60)
    print("测试：进度回调在 as_completed 中实时触发")
    print("=" * 60)

    n = 6
    samples = [_make_grounded_qa_sample(i) for i in range(n)]

    # 不同延迟：第 1 条很快，其余较慢
    delays = [0.01, 0.15, 0.15, 0.15, 0.15, 0.15]
    call_idx = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            idx = call_idx[0]
            call_idx[0] += 1
        time.sleep(delays[idx % len(delays)])
        return _VALID_GROUNDED_QA_RESPONSE

    progress_times = []
    start_time = [None]

    def on_progress(done, total, result, info):
        if start_time[0] is None:
            start_time[0] = time.monotonic()
        progress_times.append((time.monotonic() - start_time[0], done))

    with patch('judge.call_llm', side_effect=mock_call_llm):
        start_time[0] = time.monotonic()
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=4))

    assert len(results) == n
    # 第一次回调应在很短时间内发生（第 1 条完成时）
    assert len(progress_times) > 0, "应该有进度回调"
    assert progress_times[0][0] < 0.1, \
        f"第一次回调应在 0.1s 内，实际 {progress_times[0][0]:.3f}s"
    # 回调次数应等于样本数
    assert len(progress_times) == n, \
        f"回调次数应为 {n}，实际 {len(progress_times)}"

    print(f"[OK] 第一次回调在 {progress_times[0][0]:.3f}s，共 {len(progress_times)} 次回调")


def test_progress_callback_count_equals_total():
    """progress callback 次数恰好等于样本数。"""
    print("=" * 60)
    print("测试：回调次数 = 样本数")
    print("=" * 60)

    samples = [
        _make_prescreen_sample(0),
        _make_grounded_qa_sample(1),
        _make_grounded_qa_sample(2),
        _make_prescreen_sample(3),
        _make_grounded_qa_sample(4),
    ]

    mock_llm, _ = _mock_call_llm_factory()
    callback_count = [0]

    def on_progress(done, total, result, info):
        callback_count[0] += 1

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=2))

    assert len(results) == 5
    assert callback_count[0] == 5, \
        f"回调次数应为 5，实际 {callback_count[0]}"

    print(f"[OK] 回调次数: {callback_count[0]}，样本数: 5")


def test_mixed_prescreen_llm_progress_continuous():
    """规则预筛和 LLM 混合时，进度从 1 到 total 连续增长。"""
    print("=" * 60)
    print("测试：混合场景进度连续")
    print("=" * 60)

    samples = [
        _make_prescreen_sample(0),       # 预筛
        _make_grounded_qa_sample(1),     # LLM
        _make_prescreen_sample(2),       # 预筛
        _make_grounded_qa_sample(3),     # LLM
        _make_prescreen_sample(4),       # 预筛
    ]

    mock_llm, _ = _mock_call_llm_factory()
    progress_dones = []

    def on_progress(done, total, result, info):
        progress_dones.append(done)

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=2))

    assert len(results) == 5
    # 预筛样本在 Phase 1 完成，LLM 在 Phase 2 完成
    # 回调顺序应为 1,2,3,4,5（连续递增）
    assert progress_dones == [1, 2, 3, 4, 5], \
        f"进度应连续递增: {progress_dones}"

    print(f"[OK] 进度序列: {progress_dones}")


def test_eta_calculating_before_two_llm():
    """前 2 条 LLM 完成前 ETA 为'计算中'。"""
    print("=" * 60)
    print("测试：ETA 前 2 条为'计算中'")
    print("=" * 60)

    samples = [_make_grounded_qa_sample(i) for i in range(4)]
    mock_llm, _ = _mock_call_llm_factory(delay=0.02)

    eta_texts = []
    llm_dones = []

    def on_progress(done, total, result, info):
        llm_dones.append(info["llm_done"])
        eta_texts.append(info["eta_text"])

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=1))

    assert len(results) == 4
    # 前 2 条 LLM 完成时 eta_text 应为 "计算中"
    for i, (ld, eta) in enumerate(zip(llm_dones, eta_texts)):
        if ld < 2:
            assert eta == "计算中", \
                f"第 {i+1} 条完成时 llm_done={ld}，ETA 应为'计算中'，实际'{eta}'"

    # 第 3 条完成后 ETA 应有具体值
    assert llm_dones[-1] >= 3
    assert eta_texts[-1] != "计算中", \
        f"最后一条 ETA 不应为'计算中'，实际'{eta_texts[-1]}'"

    print(f"[OK] ETA 序列: {eta_texts}")


def test_prescreen_not_in_llm_throughput():
    """规则预筛不进入 LLM 平均耗时和 ETA 吞吐计算。"""
    print("=" * 60)
    print("测试：预筛不影响 LLM 吞吐计算")
    print("=" * 60)

    # 5 个预筛 + 3 个 LLM
    samples = [_make_prescreen_sample(i) for i in range(5)] + \
              [_make_grounded_qa_sample(i + 5) for i in range(3)]

    mock_llm, _ = _mock_call_llm_factory(delay=0.02)

    infos = []
    def on_progress(done, total, result, info):
        infos.append(dict(info))

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=1))

    assert len(results) == 8

    # 最后的 info 应显示 llm_done=3（不是 8）
    last_info = infos[-1]
    assert last_info["llm_done"] == 3, \
        f"llm_done 应为 3，实际 {last_info['llm_done']}"
    assert last_info["prescreened_count"] == 5, \
        f"prescreened_count 应为 5，实际 {last_info['prescreened_count']}"

    # 吞吐应基于 LLM 耗时，不是总耗时
    # 如果预筛耗时被计入，throughput 会虚高
    assert last_info["throughput"] > 0, "吞吐应大于 0"

    print(f"[OK] llm_done={last_info['llm_done']}, "
          f"prescreened={last_info['prescreened_count']}, "
          f"throughput={last_info['throughput']:.2f}")


def test_info_dict_has_all_fields():
    """info dict 包含所有必需字段。"""
    print("=" * 60)
    print("测试：info dict 字段完整")
    print("=" * 60)

    samples = [_make_grounded_qa_sample(0)]
    mock_llm, _ = _mock_call_llm_factory()

    infos = []
    def on_progress(done, total, result, info):
        infos.append(dict(info))

    with patch('judge.call_llm', side_effect=mock_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=1))

    assert len(results) == 1
    assert len(infos) == 1

    required_keys = {"llm_done", "llm_total", "elapsed", "eta_text",
                     "throughput", "prescreened_count", "cached_count",
                     "concurrency"}
    missing = required_keys - set(infos[0].keys())
    assert not missing, f"info dict 缺少字段: {missing}"

    assert infos[0]["llm_total"] == 1
    assert infos[0]["concurrency"] == 1
    assert infos[0]["elapsed"] >= 0

    print(f"[OK] info dict 字段完整: {sorted(infos[0].keys())}")


def test_yield_order_preserved_with_out_of_order_futures():
    """future 完成顺序乱序时，最终 yield 顺序仍与输入顺序一致。"""
    print("=" * 60)
    print("测试：乱序 future → 顺序 yield")
    print("=" * 60)

    n = 8
    samples = [_make_grounded_qa_sample(i) for i in range(n)]

    # 逆序延迟：第 1 条最慢，最后 1 条最快
    delays = [0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
    call_idx = [0]
    lock = threading.Lock()

    def mock_call_llm(prompt, api_key, base_url, model, timeout=30):
        with lock:
            idx = call_idx[0]
            call_idx[0] += 1
        time.sleep(delays[idx % len(delays)])
        return _VALID_GROUNDED_QA_RESPONSE

    progress_dones = []
    def on_progress(done, total, result, info):
        progress_dones.append(done)

    with patch('judge.call_llm', side_effect=mock_call_llm):
        results = list(judge_all(samples, "key", "http://fake", "model",
                                 progress_callback=on_progress,
                                 max_workers=4))

    assert len(results) == n
    # yield 顺序必须与输入一致
    for i, r in enumerate(results):
        assert r["trace_id"] == f"trace_{i}", \
            f"位置 {i} 期望 trace_{i}，实际 {r['trace_id']}"

    # 回调次数等于样本数
    assert len(progress_dones) == n

    print(f"[OK] {n} 条结果按原始顺序 yield，回调 {len(progress_dones)} 次")


# ====== 主入口 ======

def main():
    tests = [
        test_concurrency_1_matches_serial,
        test_order_preserved_with_concurrency,
        test_single_error_does_not_abort_others,
        test_prescreen_skips_executor,
        test_max_concurrency_respected,
        test_manifest_counts_correct,
        test_retry_on_429,
        test_retry_exhaustion_raises,
        test_content_cache_dedup_with_concurrency,
        test_progress_fires_during_as_completed,
        test_progress_callback_count_equals_total,
        test_mixed_prescreen_llm_progress_continuous,
        test_eta_calculating_before_two_llm,
        test_prescreen_not_in_llm_throughput,
        test_info_dict_has_all_fields,
        test_yield_order_preserved_with_out_of_order_futures,
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
