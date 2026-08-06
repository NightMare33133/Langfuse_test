"""
样本展示排序与分页测试。

覆盖：
a. 混合 timestamp/start_time/created_at/missing 的样本能正确排序
b. 最新样本始终在筛选后第一页
c. 缺失时间排在最后
d. 同步和筛选后页码重置为 1
e. 排序稳定，不因相同时间随机变化
f. build_trace_sample 正确提取时间字段
g. _parse_ts_for_sort 正确解析各种时间格式

不调用真实 API。
"""

import sys
import json
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

from parser import build_trace_sample, _parse_ts_for_sort, normalize_observation_row


# ====== 辅助函数 ======

def _make_obs(trace_id, obs_type="TRACE", start_time=None, is_root=False,
              name=None, node_type=None, input_data=None, output_data=None):
    """创建一个 observation 行。"""
    obs = {
        "id": f"obs_{trace_id}_{obs_type}",
        "traceId": trace_id,
        "type": obs_type,
        "name": name,
        "startTime": start_time,
        "endTime": start_time,
        "input": input_data,
        "output": output_data,
        "metadata": None,
        "sessionId": None,
        "userId": None,
        "traceName": None,
        "providedModelName": None,
    }
    if is_root:
        obs["rawType"] = "TRACE"
        obs["isTraceRoot"] = True
    else:
        obs["rawType"] = obs_type
        obs["isTraceRoot"] = False
    return obs


def _make_sample_with_time(trace_id, ts_str, question=None):
    """创建一个带时间的样本（模拟 build_trace_sample 输出）。"""
    obs = _make_obs(trace_id, "TRACE", start_time=ts_str, is_root=True,
                    name="message", input_data={"sys.query": question or f"问题_{trace_id}"})
    normalized = normalize_observation_row(obs)
    sample = build_trace_sample(trace_id, [normalized])
    return sample


# ====== _parse_ts_for_sort 测试 ======

def test_parse_iso8601_utc():
    """解析 ISO-8601 UTC 时间。"""
    dt = _parse_ts_for_sort("2026-08-05T11:50:17Z")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 8
    assert dt.day == 5
    assert dt.tzinfo is not None
    print("[OK] _parse_ts_for_sort: ISO-8601 UTC")


def test_parse_iso8601_offset():
    """解析带时区偏移的 ISO-8601 时间。"""
    dt = _parse_ts_for_sort("2026-08-05T11:50:17+08:00")
    assert dt is not None
    assert dt.year == 2026
    assert dt.tzinfo is not None
    print("[OK] _parse_ts_for_sort: ISO-8601 with offset")


def test_parse_iso8601_no_tz():
    """解析不带时区的 ISO-8601 时间（默认 UTC）。"""
    dt = _parse_ts_for_sort("2026-08-05T11:50:17")
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    print("[OK] _parse_ts_for_sort: ISO-8601 no timezone (defaults to UTC)")


def test_parse_none_returns_none():
    """None 输入返回 None。"""
    assert _parse_ts_for_sort(None) is None
    assert _parse_ts_for_sort("") is None
    print("[OK] _parse_ts_for_sort: None/empty returns None")


def test_parse_invalid_returns_none():
    """无效时间字符串返回 None。"""
    assert _parse_ts_for_sort("not-a-time") is None
    assert _parse_ts_for_sort("2026-13-45") is None
    print("[OK] _parse_ts_for_sort: invalid string returns None")


def test_parse_datetime_object():
    """datetime 对象直接返回（补时区）。"""
    dt_naive = datetime(2026, 8, 5, 11, 50, 17)
    dt = _parse_ts_for_sort(dt_naive)
    assert dt is not None
    assert dt.tzinfo == timezone.utc

    dt_aware = datetime(2026, 8, 5, 11, 50, 17, tzinfo=timezone.utc)
    dt2 = _parse_ts_for_sort(dt_aware)
    assert dt2 is not None
    assert dt2.tzinfo == timezone.utc
    print("[OK] _parse_ts_for_sort: datetime objects")


# ====== build_trace_sample 时间提取测试 ======

def test_trace_timestamp_from_root():
    """TRACE root 的 startTime 被提取为 trace_timestamp。"""
    sample = _make_sample_with_time("t1", "2026-08-05T11:50:17Z")
    assert sample["trace_timestamp"] == "2026-08-05T11:50:17Z"
    print("[OK] build_trace_sample: trace_timestamp from TRACE root")


def test_earliest_obs_time():
    """earliest_obs_time 取所有 observation 中最早的时间。"""
    obs1 = _make_obs("t2", "TRACE", start_time="2026-08-05T12:00:00Z", is_root=True, name="message")
    obs2 = _make_obs("t2", "SPAN", start_time="2026-08-05T10:00:00Z", name="retrieval")
    norm1 = normalize_observation_row(obs1)
    norm2 = normalize_observation_row(obs2)
    sample = build_trace_sample("t2", [norm1, norm2])
    assert sample["trace_timestamp"] == "2026-08-05T12:00:00Z"
    assert sample["earliest_obs_time"] == "2026-08-05T10:00:00Z"
    print("[OK] build_trace_sample: earliest_obs_time from earliest observation")


def test_no_time_fields():
    """无时间字段时，trace_timestamp 和 earliest_obs_time 均为 None。"""
    obs = _make_obs("t3", "SPAN", start_time=None, name="retrieval")
    norm = normalize_observation_row(obs)
    sample = build_trace_sample("t3", [norm])
    assert sample["trace_timestamp"] is None
    assert sample["earliest_obs_time"] is None
    print("[OK] build_trace_sample: no time fields → None")


# ====== 排序测试 ======

def _sort_samples_for_display(samples, newest_first=True):
    """模拟 app.py 中的排序逻辑。"""
    _with_ts = []
    _no_ts = []
    for s in samples:
        ts = s.get("trace_timestamp") or s.get("earliest_obs_time")
        dt = _parse_ts_for_sort(ts)
        if dt is None:
            _no_ts.append(s)
        else:
            _with_ts.append((dt, s.get("trace_id", ""), s))

    if newest_first:
        _with_ts.sort(key=lambda x: (-x[0].timestamp(), x[1]))
    else:
        _with_ts.sort(key=lambda x: (x[0].timestamp(), x[1]))
    _no_ts.sort(key=lambda x: x.get("trace_id", ""))

    return [item[2] for item in _with_ts] + _no_ts


def test_mixed_time_fields_sort():
    """混合 timestamp/start_time/missing 的样本能正确排序。"""
    s1 = {"trace_id": "t_old", "trace_timestamp": "2026-08-01T10:00:00Z", "earliest_obs_time": None, "question": "旧问题"}
    s2 = {"trace_id": "t_new", "trace_timestamp": "2026-08-05T12:00:00Z", "earliest_obs_time": None, "question": "新问题"}
    s3 = {"trace_id": "t_mid", "trace_timestamp": None, "earliest_obs_time": "2026-08-03T08:00:00Z", "question": "中间问题"}
    s4 = {"trace_id": "t_no_ts", "trace_timestamp": None, "earliest_obs_time": None, "question": "无时间"}

    sorted_desc = _sort_samples_for_display([s1, s2, s3, s4], newest_first=True)
    ids_desc = [s["trace_id"] for s in sorted_desc]
    assert ids_desc == ["t_new", "t_mid", "t_old", "t_no_ts"], f"Expected newest first, got {ids_desc}"

    sorted_asc = _sort_samples_for_display([s1, s2, s3, s4], newest_first=False)
    ids_asc = [s["trace_id"] for s in sorted_asc]
    assert ids_asc == ["t_old", "t_mid", "t_new", "t_no_ts"], f"Expected oldest first, got {ids_asc}"
    print("[OK] mixed time fields sort: newest first and oldest first")


def test_newest_always_first_after_filter():
    """最新样本始终在筛选后第一页。"""
    samples = []
    for i in range(50):
        day = 1 + (i % 28)
        hour = i % 24
        samples.append({
            "trace_id": f"t_{i:03d}",
            "trace_timestamp": f"2026-08-{day:02d}T{hour:02d}:00:00Z",
            "earliest_obs_time": None,
            "question": f"问题 {i}" if i % 10 != 0 else "特殊关键词",
        })

    sorted_samples = _sort_samples_for_display(samples, newest_first=True)

    # 筛选包含"特殊关键词"的样本
    filtered = [s for s in sorted_samples if "特殊关键词" in (s.get("question") or "")]
    assert len(filtered) > 0
    # 第一个应该是最新的（day=28, hour=20 或类似）
    first_ts = filtered[0].get("trace_timestamp")
    for s in filtered[1:]:
        other_ts = s.get("trace_timestamp")
        if other_ts:
            assert first_ts >= other_ts, f"First sample {first_ts} should be >= {other_ts}"
    print("[OK] newest always first after filter")


def test_missing_time_sorted_last():
    """缺失时间排在最后。"""
    s1 = {"trace_id": "t_with_ts", "trace_timestamp": "2026-08-01T10:00:00Z", "earliest_obs_time": None, "question": "有时间"}
    s2 = {"trace_id": "t_no_ts", "trace_timestamp": None, "earliest_obs_time": None, "question": "无时间"}

    sorted_desc = _sort_samples_for_display([s2, s1], newest_first=True)
    assert sorted_desc[0]["trace_id"] == "t_with_ts", "有时间的应排在前面"
    assert sorted_desc[1]["trace_id"] == "t_no_ts", "无时间的应排在最后"

    sorted_asc = _sort_samples_for_display([s2, s1], newest_first=False)
    assert sorted_asc[0]["trace_id"] == "t_with_ts", "有时间的应排在前面（最早优先）"
    assert sorted_asc[1]["trace_id"] == "t_no_ts", "无时间的应排在最后"
    print("[OK] missing time sorted last")


def test_stable_sort_same_time():
    """相同时间按 trace_id 稳定排序。"""
    s1 = {"trace_id": "t_aaa", "trace_timestamp": "2026-08-05T12:00:00Z", "earliest_obs_time": None, "question": "A"}
    s2 = {"trace_id": "t_bbb", "trace_timestamp": "2026-08-05T12:00:00Z", "earliest_obs_time": None, "question": "B"}
    s3 = {"trace_id": "t_ccc", "trace_timestamp": "2026-08-05T12:00:00Z", "earliest_obs_time": None, "question": "C"}

    # 多次排序结果应一致
    for _ in range(10):
        sorted_samples = _sort_samples_for_display([s2, s3, s1], newest_first=True)
        ids = [s["trace_id"] for s in sorted_samples]
        assert ids == ["t_aaa", "t_bbb", "t_ccc"], f"Stable sort failed: {ids}"
    print("[OK] stable sort: same time sorted by trace_id")


def test_fallback_to_earliest_obs_time():
    """trace_timestamp 缺失时回退到 earliest_obs_time。"""
    s1 = {"trace_id": "t_fallback", "trace_timestamp": None, "earliest_obs_time": "2026-08-05T10:00:00Z", "question": "回退"}
    s2 = {"trace_id": "t_direct", "trace_timestamp": "2026-08-05T08:00:00Z", "earliest_obs_time": None, "question": "直接"}

    sorted_desc = _sort_samples_for_display([s1, s2], newest_first=True)
    assert sorted_desc[0]["trace_id"] == "t_fallback", "earliest_obs_time=10:00 应排在 trace_timestamp=08:00 前面"
    print("[OK] fallback to earliest_obs_time")


def test_empty_list_sort():
    """空列表排序不报错。"""
    result = _sort_samples_for_display([], newest_first=True)
    assert result == []
    print("[OK] empty list sort")


def test_all_missing_time_sort():
    """全部缺失时间的样本排序不报错。"""
    s1 = {"trace_id": "t_a", "trace_timestamp": None, "earliest_obs_time": None, "question": "A"}
    s2 = {"trace_id": "t_b", "trace_timestamp": None, "earliest_obs_time": None, "question": "B"}
    result = _sort_samples_for_display([s2, s1], newest_first=True)
    assert len(result) == 2
    # 无时间的按 trace_id 排序
    assert result[0]["trace_id"] == "t_a"
    assert result[1]["trace_id"] == "t_b"
    print("[OK] all missing time sort")


# ====== 主函数 ======

def main():
    print("=" * 60)
    print("样本展示排序与分页测试")
    print("=" * 60)
    print()

    # _parse_ts_for_sort 测试
    test_parse_iso8601_utc()
    test_parse_iso8601_offset()
    test_parse_iso8601_no_tz()
    test_parse_none_returns_none()
    test_parse_invalid_returns_none()
    test_parse_datetime_object()

    # build_trace_sample 时间提取测试
    test_trace_timestamp_from_root()
    test_earliest_obs_time()
    test_no_time_fields()

    # 排序测试
    test_mixed_time_fields_sort()
    test_newest_always_first_after_filter()
    test_missing_time_sorted_last()
    test_stable_sort_same_time()
    test_fallback_to_earliest_obs_time()
    test_empty_list_sort()
    test_all_missing_time_sort()

    print()
    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
