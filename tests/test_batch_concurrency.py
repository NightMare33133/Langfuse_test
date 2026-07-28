"""批量提问并发模式测试。

验证：
1. max_workers=1 保持串行行为（与旧行为完全一致）
2. max_workers>1 时多任务重叠执行，实际并发不超过设定值
3. 某题失败不影响其他题，全部结果都会返回
4. 最终结果按原始题目索引排序（_original_index 字段正确）
"""

import json
import time
import threading
from unittest.mock import patch, MagicMock

import pytest

from batch_query import run_batch_query, normalize_questions


# ── 测试辅助 ──────────────────────────────────────────────────

def _make_questions(n):
    """生成 n 个标准问题 dict。"""
    return [{"question": f"问题 {i}", "question_id": f"q_{i}"} for i in range(n)]


def _mock_dify_ok(delay=0.05):
    """返回一个 mock call_dify_query，模拟正常响应并可控延迟。"""
    def _fake(question, api_key, base_url, timeout=60, user="batch-query"):
        time.sleep(delay)
        return {
            "answer": f"回答: {question}",
            "conversation_id": "conv_123",
            "message_id": "msg_456",
            "retriever_resources": [],
            "raw_response": {"answer": f"回答: {question}"},
        }
    return _fake


def _mock_dify_with_concurrency_tracker(delay=0.1):
    """返回 mock call_dify_query 并追踪并发数。"""
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def _fake(question, api_key, base_url, timeout=60, user="batch-query"):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(delay)
        with lock:
            active["count"] -= 1
        return {
            "answer": f"回答: {question}",
            "conversation_id": "conv_123",
            "message_id": "msg_456",
            "retriever_resources": [],
            "raw_response": {"answer": f"回答: {question}"},
        }
    return _fake, active


def _mock_dify_mixed(fail_indices=None, delay=0.05):
    """返回 mock call_dify_query，对指定索引抛出异常。"""
    fail_set = set(fail_indices or [])
    call_count = {"n": 0}

    def _fake(question, api_key, base_url, timeout=60, user="batch-query"):
        idx = call_count["n"]
        call_count["n"] += 1
        time.sleep(delay)
        if idx in fail_set:
            raise RuntimeError(f"模拟第 {idx} 题失败")
        return {
            "answer": f"回答: {question}",
            "conversation_id": "conv_123",
            "message_id": "msg_456",
            "retriever_resources": [],
            "raw_response": {"answer": f"回答: {question}"},
        }
    return _fake


# ── 测试用例 ──────────────────────────────────────────────────

class TestSerialMode:
    """max_workers=1 时保持原有串行行为。"""

    def test_serial_yields_in_order(self):
        """串行模式按 0,1,2,... 顺序 yield。"""
        questions = _make_questions(5)
        mock_fn = _mock_dify_ok(delay=0.01)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            indices = []
            for idx, total, result in run_batch_query(
                questions, "fake_key", "http://fake", max_workers=1
            ):
                indices.append(idx)
                assert total == 5

        assert indices == [0, 1, 2, 3, 4]

    def test_serial_preserves_result_structure(self):
        """串行模式返回的 result dict 包含所有必需字段。"""
        questions = _make_questions(1)
        mock_fn = _mock_dify_ok()

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=1
            ))

        assert len(results) == 1
        idx, total, result = results[0]
        assert idx == 0
        assert total == 1
        assert result["success"] is True
        assert "question" in result
        assert "sample" in result
        assert "raw_response" in result
        assert "_original_index" in result
        assert result["_original_index"] == 0

    def test_serial_delay_respected(self):
        """串行模式下 delay 参数生效（请求间有间隔）。"""
        questions = _make_questions(3)
        mock_fn = _mock_dify_ok(delay=0.01)
        delay = 0.1

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            t0 = time.monotonic()
            list(run_batch_query(
                questions, "fake_key", "http://fake",
                delay=delay, max_workers=1
            ))
            elapsed = time.monotonic() - t0

        # 3 题，2 个间隔 => 至少 2*delay 的间隔时间
        assert elapsed >= delay * 1.5, f"串行延迟不足: {elapsed:.3f}s < {delay * 1.5:.3f}s"

    def test_serial_failure_continues(self):
        """串行模式下单题失败不影响后续题目。"""
        questions = _make_questions(3)
        mock_fn = _mock_dify_mixed(fail_indices=[1], delay=0.01)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=1
            ))

        assert len(results) == 3
        assert results[0][2]["success"] is True
        assert results[1][2]["success"] is False
        assert results[2][2]["success"] is True


class TestConcurrentMode:
    """max_workers>1 时使用线程池并发。"""

    def test_concurrent_all_results_returned(self):
        """并发模式下所有题目结果都会返回。"""
        questions = _make_questions(6)
        mock_fn = _mock_dify_ok(delay=0.02)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=3
            ))

        assert len(results) == 6
        returned_indices = {r[0] for r in results}
        assert returned_indices == {0, 1, 2, 3, 4, 5}

    def test_concurrent_max_workers_respected(self):
        """并发模式下实际并发数不超过 max_workers。"""
        questions = _make_questions(8)
        mock_fn, active = _mock_dify_with_concurrency_tracker(delay=0.15)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=3
            ))

        assert active["max"] <= 3, f"实际最大并发 {active['max']} > 设定值 3"

    def test_concurrent_faster_than_serial(self):
        """并发模式总耗时明显短于串行模式。"""
        questions = _make_questions(6)
        delay = 0.15

        # 串行
        mock_serial = _mock_dify_ok(delay=delay)
        with patch("batch_query.call_dify_query", side_effect=mock_serial):
            t0 = time.monotonic()
            list(run_batch_query(
                questions, "fake_key", "http://fake",
                delay=0, max_workers=1
            ))
            serial_time = time.monotonic() - t0

        # 并发
        mock_concurrent = _mock_dify_ok(delay=delay)
        with patch("batch_query.call_dify_query", side_effect=mock_concurrent):
            t0 = time.monotonic()
            list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=3
            ))
            concurrent_time = time.monotonic() - t0

        assert concurrent_time < serial_time * 0.7, (
            f"并发 ({concurrent_time:.2f}s) 应快于串行 ({serial_time:.2f}s)"
        )

    def test_concurrent_failure_isolation(self):
        """并发模式下某题失败不影响其他题。"""
        questions = _make_questions(5)
        # 让第 0、3 题失败
        mock_fn = _mock_dify_mixed(fail_indices=[0, 3], delay=0.02)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=3
            ))

        assert len(results) == 5
        success_count = sum(1 for _, _, r in results if r["success"])
        fail_count = sum(1 for _, _, r in results if not r["success"])
        assert success_count == 3
        assert fail_count == 2

        # 失败的题目包含 error 字段
        for _, _, r in results:
            if not r["success"]:
                assert "error" in r

    def test_concurrent_results_have_original_index(self):
        """并发模式下每条结果都携带正确的 _original_index。"""
        questions = _make_questions(5)
        mock_fn = _mock_dify_ok(delay=0.02)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=3
            ))

        for idx, total, result in results:
            assert result["_original_index"] == idx
            assert 0 <= idx < 5

    def test_concurrent_delay_ignored(self):
        """并发模式下 delay 参数不生效（总耗时不受 delay 影响）。"""
        questions = _make_questions(4)
        mock_fn = _mock_dify_ok(delay=0.02)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            t0 = time.monotonic()
            list(run_batch_query(
                questions, "fake_key", "http://fake",
                delay=5.0,  # 很大的 delay
                max_workers=2,
            ))
            elapsed = time.monotonic() - t0

        # 如果 delay 生效，4 题至少需要 3*5=15 秒
        assert elapsed < 5.0, f"并发模式不应受 delay 影响: {elapsed:.2f}s"


class TestResultOrdering:
    """验证结果按原始索引排序。"""

    def test_sort_by_original_index(self):
        """按 _original_index 排序后与原始题目顺序一致。"""
        questions = _make_questions(8)
        mock_fn = _mock_dify_ok(delay=0.02)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=4
            ))

        # 排序前可能乱序（完成顺序）
        sorted_results = sorted(results, key=lambda r: r[2]["_original_index"])
        sorted_indices = [r[0] for r in sorted_results]
        assert sorted_indices == list(range(8))

    def test_sort_preserves_all_data(self):
        """排序后每条结果的数据完整性不变。"""
        questions = _make_questions(4)
        mock_fn = _mock_dify_ok(delay=0.02)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=2
            ))

        sorted_results = sorted(results, key=lambda r: r[2]["_original_index"])
        for i, (idx, total, result) in enumerate(sorted_results):
            assert idx == i
            assert total == 4
            assert result["success"] is True
            assert "sample" in result


class TestEdgeCases:
    """边界情况。"""

    def test_empty_questions(self):
        """空问题列表不报错，不 yield。"""
        results = list(run_batch_query(
            [], "fake_key", "http://fake", max_workers=3
        ))
        assert results == []

    def test_single_question_concurrent(self):
        """单题 + max_workers>1 正常工作。"""
        questions = _make_questions(1)
        mock_fn = _mock_dify_ok(delay=0.01)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=3
            ))

        assert len(results) == 1
        assert results[0][2]["success"] is True

    def test_max_workers_clamped_to_8(self):
        """max_workers 超过 8 时被钳制为 8（不报错）。"""
        questions = _make_questions(3)
        mock_fn, active = _mock_dify_with_concurrency_tracker(delay=0.05)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=100
            ))

        assert len(results) == 3
        assert active["max"] <= 8

    def test_all_fail(self):
        """全部失败时仍返回所有结果。"""
        questions = _make_questions(4)
        mock_fn = _mock_dify_mixed(fail_indices=[0, 1, 2, 3], delay=0.01)

        with patch("batch_query.call_dify_query", side_effect=mock_fn):
            results = list(run_batch_query(
                questions, "fake_key", "http://fake", max_workers=2
            ))

        assert len(results) == 4
        assert all(not r[2]["success"] for r in results)

    def test_concurrent_user_field_correct(self):
        """并发模式下每题的 Dify user 字段包含正确的 run_id 和 question_id。"""
        questions = _make_questions(3)
        captured_users = []
        lock = threading.Lock()

        def _capture(question, api_key, base_url, timeout=60, user="batch-query"):
            with lock:
                captured_users.append(user)
            return {
                "answer": "ok",
                "conversation_id": "c",
                "message_id": "m",
                "retriever_resources": [],
                "raw_response": {},
            }

        with patch("batch_query.call_dify_query", side_effect=_capture):
            list(run_batch_query(
                questions, "fake_key", "http://fake",
                run_id="run_abc", question_ids=["q_0", "q_1", "q_2"],
                max_workers=2,
            ))

        assert len(captured_users) == 3
        assert "rag_eval:run_abc:q_0" in captured_users
        assert "rag_eval:run_abc:q_1" in captured_users
        assert "rag_eval:run_abc:q_2" in captured_users
