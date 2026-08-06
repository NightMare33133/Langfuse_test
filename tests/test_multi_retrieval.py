"""
多检索支持测试。

覆盖：
a. 一个 trace 含 3 次检索时，retrieval_calls 包含 3 条记录
b. 三个 query 都被保留
c. 每次检索的结果不互相覆盖
d. 旧版只有一次检索的 trace 仍能正常展示
e. 缺失 parent_observation_id 时仍按 trace_id 和时间降级排序
f. Judge 多检索命中统计
g. 兼容字段 retrieval_query/retrieval_results 正确投影

不调用真实 API。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from parser import build_trace_sample, normalize_observation_row
from judge import _judge_chunk_exact


# ====== 辅助函数 ======

def _make_obs(obs_id, trace_id, obs_type="SPAN", start_time=None, end_time=None,
              is_root=False, name=None, node_type=None, input_data=None, output_data=None,
              parent_obs_id=None):
    """创建一个 observation 行。"""
    import json
    metadata = {"node_type": node_type} if node_type else None
    obs = {
        "id": obs_id,
        "traceId": trace_id,
        "type": obs_type,
        "name": name,
        "startTime": start_time,
        "endTime": end_time,
        "input": json.dumps(input_data) if input_data else None,
        "output": json.dumps(output_data) if output_data else None,
        "metadata": json.dumps(metadata) if metadata else None,
        "sessionId": None,
        "userId": None,
        "traceName": None,
        "providedModelName": None,
        "parentObservationId": parent_obs_id,
    }
    if is_root:
        obs["rawType"] = "TRACE"
        obs["isTraceRoot"] = True
    else:
        obs["rawType"] = obs_type
        obs["isTraceRoot"] = False
    return obs


def _make_retrieval_obs(obs_id, trace_id, query, results, start_time, end_time, parent_obs_id=None):
    """创建一个知识检索 observation。"""
    return _make_obs(
        obs_id, trace_id,
        name="knowledge-retrieval",
        node_type="knowledge-retrieval",
        start_time=start_time,
        end_time=end_time,
        input_data={"query": query},
        output_data={"result": results},
        parent_obs_id=parent_obs_id,
    )


def _make_retrieval_result(segment_id, content="test content", score=0.9, position=1):
    """创建一个检索结果。"""
    return {
        "title": f"Title {segment_id}",
        "content": content,
        "metadata": {
            "position": position,
            "score": score,
            "document_name": f"doc_{segment_id}.pdf",
            "segment_id": segment_id,
            "chunk_id": f"chunk_{segment_id}",
        },
    }


# ====== 测试函数 ======

def test_multi_retrieval_calls_preserved():
    """一个 trace 含 3 次检索时，retrieval_calls 包含 3 条记录。"""
    trace_id = "trace_multi_3"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "query 1",
                           [_make_retrieval_result("seg_1")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
        _make_retrieval_obs("obs_r2", trace_id, "query 2",
                           [_make_retrieval_result("seg_2")],
                           "2026-08-05T10:00:02Z", "2026-08-05T10:00:03Z"),
        _make_retrieval_obs("obs_r3", trace_id, "query 3",
                           [_make_retrieval_result("seg_3")],
                           "2026-08-05T10:00:04Z", "2026-08-05T10:00:05Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_call_count"] == 3, f"Expected 3, got {sample['retrieval_call_count']}"
    assert len(sample["retrieval_calls"]) == 3
    print("[OK] multi-retrieval: 3 calls preserved")


def test_multi_retrieval_queries_preserved():
    """三个 query 都被保留。"""
    trace_id = "trace_multi_query"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "什么是违约金",
                           [_make_retrieval_result("seg_1")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
        _make_retrieval_obs("obs_r2", trace_id, "合同解除条件",
                           [_make_retrieval_result("seg_2")],
                           "2026-08-05T10:00:02Z", "2026-08-05T10:00:03Z"),
        _make_retrieval_obs("obs_r3", trace_id, "赔偿标准",
                           [_make_retrieval_result("seg_3")],
                           "2026-08-05T10:00:04Z", "2026-08-05T10:00:05Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    queries = [c["query"] for c in sample["retrieval_calls"]]
    assert queries == ["什么是违约金", "合同解除条件", "赔偿标准"], f"Got {queries}"
    print("[OK] multi-retrieval: all 3 queries preserved")


def test_multi_retrieval_results_not_overwritten():
    """每次检索的结果不互相覆盖。"""
    trace_id = "trace_no_overwrite"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "q1",
                           [_make_retrieval_result("seg_A"), _make_retrieval_result("seg_B")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
        _make_retrieval_obs("obs_r2", trace_id, "q2",
                           [_make_retrieval_result("seg_C"), _make_retrieval_result("seg_D")],
                           "2026-08-05T10:00:02Z", "2026-08-05T10:00:03Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    call1_segs = [r["segment_id"] for r in sample["retrieval_calls"][0]["results"]]
    call2_segs = [r["segment_id"] for r in sample["retrieval_calls"][1]["results"]]
    assert call1_segs == ["seg_A", "seg_B"], f"Call 1: {call1_segs}"
    assert call2_segs == ["seg_C", "seg_D"], f"Call 2: {call2_segs}"
    # 确保没有交叉污染
    assert "seg_C" not in call1_segs
    assert "seg_A" not in call2_segs
    print("[OK] multi-retrieval: results not overwritten")


def test_single_retrieval_backward_compat():
    """旧版只有一次检索的 trace 仍能正常展示。"""
    trace_id = "trace_single"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "单次查询",
                           [_make_retrieval_result("seg_1"), _make_retrieval_result("seg_2")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    # 兼容字段正常
    assert sample["retrieval_query"] == "单次查询"
    assert len(sample["retrieval_results"]) == 2
    assert sample["retrieval_results"][0]["segment_id"] == "seg_1"

    # 新字段也正常
    assert sample["retrieval_call_count"] == 1
    assert len(sample["retrieval_calls"]) == 1
    assert sample["retrieval_calls"][0]["query"] == "单次查询"
    print("[OK] single-retrieval backward compat")


def test_compat_fields_project_from_last_call():
    """兼容字段 retrieval_query/retrieval_results 来自最后一次 retrieval call。"""
    trace_id = "trace_compat"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "first query",
                           [_make_retrieval_result("seg_first")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
        _make_retrieval_obs("obs_r2", trace_id, "last query",
                           [_make_retrieval_result("seg_last")],
                           "2026-08-05T10:00:02Z", "2026-08-05T10:00:03Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_query"] == "last query"
    assert len(sample["retrieval_results"]) == 1
    assert sample["retrieval_results"][0]["segment_id"] == "seg_last"
    print("[OK] compat fields from last call")


def test_no_retrieval_trace():
    """无检索的 trace 正常处理。"""
    trace_id = "trace_no_ret"
    obs = [
        _make_obs("obs_root", trace_id, "TRACE", is_root=True,
                  name="message", start_time="2026-08-05T10:00:00Z",
                  input_data={"sys.query": "hello"}),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_call_count"] == 0
    assert sample["retrieval_calls"] == []
    assert sample["retrieval_query"] is None
    assert sample["retrieval_results"] == []
    print("[OK] no retrieval trace")


def test_judge_multi_retrieval_hit():
    """Judge 多检索命中统计。"""
    sample = {
        "trace_id": "trace_judge_multi",
        "expected_segment_id": "seg_target",
        "expected_content_hash": "",
        "retrieval_calls": [
            {"order": 1, "observation_id": "obs1", "query": "q1", "results": [
                {"segment_id": "seg_other", "content": "other"},
            ]},
            {"order": 2, "observation_id": "obs2", "query": "q2", "results": [
                {"segment_id": "seg_target", "content": "target content"},
            ]},
        ],
    }
    result = _judge_chunk_exact(sample)

    assert result["retrieval_evaluable"] is True
    assert result["retrieval_top1_hit"] == 1
    assert result["hit_evidence_position"] == 1
    assert result["retrieval_call_count"] == 2
    assert result["subquery_hit_count"] == 1
    assert result["per_subquery_hit"] is True
    assert result["trace_level_coverage"] is True
    # 第一个 call 未命中，第二个 call 命中
    assert result["per_call_hits"][0]["hit_position"] is None
    assert result["per_call_hits"][1]["hit_position"] == 1
    print("[OK] judge multi-retrieval hit")


def test_judge_multi_retrieval_miss():
    """Judge 多检索全部未命中。"""
    sample = {
        "trace_id": "trace_judge_miss",
        "expected_segment_id": "seg_target",
        "expected_content_hash": "",
        "retrieval_calls": [
            {"order": 1, "observation_id": "obs1", "query": "q1", "results": [
                {"segment_id": "seg_other1", "content": "other1"},
            ]},
            {"order": 2, "observation_id": "obs2", "query": "q2", "results": [
                {"segment_id": "seg_other2", "content": "other2"},
            ]},
        ],
    }
    result = _judge_chunk_exact(sample)

    assert result["retrieval_evaluable"] is True
    assert result["retrieval_top1_hit"] == 0
    assert result["hit_evidence_position"] is None
    assert result["subquery_hit_count"] == 0
    assert result["per_subquery_hit"] is False
    assert result["trace_level_coverage"] is False
    print("[OK] judge multi-retrieval miss")


def test_judge_single_retrieval_backward_compat():
    """Judge 单检索模式向后兼容。"""
    sample = {
        "trace_id": "trace_judge_single",
        "expected_segment_id": "seg_target",
        "expected_content_hash": "",
        "retrieval_results": [
            {"segment_id": "seg_other", "content": "other"},
            {"segment_id": "seg_target", "content": "target"},
        ],
    }
    result = _judge_chunk_exact(sample)

    assert result["retrieval_evaluable"] is True
    assert result["retrieval_top1_hit"] == 0  # 第一个不是
    assert result["retrieval_top3_hit"] == 1  # 第二个是
    assert result["hit_evidence_position"] == 2
    # 单检索模式不应有 per_call_hits
    assert "per_call_hits" not in result
    print("[OK] judge single-retrieval backward compat")


def test_latency_calculated():
    """延迟正确计算。"""
    trace_id = "trace_latency"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "q1",
                           [_make_retrieval_result("seg_1")],
                           "2026-08-05T10:00:00.000Z", "2026-08-05T10:00:00.500Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_calls"][0]["latency_ms"] == 500, \
        f"Expected 500ms, got {sample['retrieval_calls'][0]['latency_ms']}"
    print("[OK] latency calculated")


def test_observation_ids_preserved():
    """observation_id 正确保留。"""
    trace_id = "trace_obs_id"
    obs = [
        _make_retrieval_obs("obs_abc123", trace_id, "q1",
                           [_make_retrieval_result("seg_1")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_calls"][0]["observation_id"] == "obs_abc123"
    print("[OK] observation_id preserved")


def test_order_preserved_by_time():
    """retrieval_calls 按时间排序。"""
    trace_id = "trace_order"
    obs = [
        _make_retrieval_obs("obs_late", trace_id, "late query",
                           [_make_retrieval_result("seg_late")],
                           "2026-08-05T10:00:10Z", "2026-08-05T10:00:11Z"),
        _make_retrieval_obs("obs_early", trace_id, "early query",
                           [_make_retrieval_result("seg_early")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_calls"][0]["query"] == "early query"
    assert sample["retrieval_calls"][1]["query"] == "late query"
    assert sample["retrieval_calls"][0]["order"] == 1
    assert sample["retrieval_calls"][1]["order"] == 2
    print("[OK] order preserved by time")


def test_end_time_preserved():
    """end_time 正确保留。"""
    trace_id = "trace_end_time"
    obs = [
        _make_retrieval_obs("obs_r1", trace_id, "q1",
                           [_make_retrieval_result("seg_1")],
                           "2026-08-05T10:00:00Z", "2026-08-05T10:00:01Z"),
    ]
    normalized = [normalize_observation_row(o) for o in obs]
    sample = build_trace_sample(trace_id, normalized)

    assert sample["retrieval_calls"][0]["start_time"] == "2026-08-05T10:00:00Z"
    assert sample["retrieval_calls"][0]["end_time"] == "2026-08-05T10:00:01Z"
    print("[OK] end_time preserved")


# ====== 主函数 ======

def main():
    print("=" * 60)
    print("多检索支持测试")
    print("=" * 60)
    print()

    # parser 测试
    test_multi_retrieval_calls_preserved()
    test_multi_retrieval_queries_preserved()
    test_multi_retrieval_results_not_overwritten()
    test_single_retrieval_backward_compat()
    test_compat_fields_project_from_last_call()
    test_no_retrieval_trace()
    test_latency_calculated()
    test_observation_ids_preserved()
    test_order_preserved_by_time()
    test_end_time_preserved()

    # judge 测试
    test_judge_multi_retrieval_hit()
    test_judge_multi_retrieval_miss()
    test_judge_single_retrieval_backward_compat()

    print()
    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
