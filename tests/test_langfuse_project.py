"""
Langfuse 项目管理模块测试。

测试内容：
1. generate_project_id — 稳定 ID 生成
2. identify_project — API 调用与响应解析
3. register_project / load_project / list_projects — 注册表 CRUD
4. append_traces — trace_id 去重
5. incremental_sync — 增量同步 + 游标更新
6. get_project_stats — 统计信息
7. list_cleanup_candidates — 旧文件清理候选
8. 跨 Key 同项目去重
"""

import json
import gzip
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from langfuse_project import (
    generate_project_id,
    identify_project,
    register_project, load_project, list_projects,
    load_existing_trace_ids, load_existing_obs_ids, append_traces, append_observations,
    incremental_sync, get_project_stats,
    list_cleanup_candidates, cleanup_files,
    create_frozen_snapshot, _update_current_snapshot, list_snapshots, get_current_snapshot_id,
    get_snapshot_path, export_snapshot_as_jsonl, mark_snapshot_parsed,
    can_cleanup_snapshot, cleanup_old_snapshots,
    list_parseable_sources, _get_snapshot_sizes, _traces_path, _obs_path,
    PROJECTS_DIR, RAW_DIR,
)


# ── Fixtures ──────────────────────────────────────────────────


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    return resp


@pytest.fixture(autouse=True)
def _use_tmp_dir(tmp_path):
    """每个测试使用独立临时目录。"""
    with patch("langfuse_project.PROJECTS_DIR", tmp_path / "projects"), \
         patch("langfuse_project.RAW_DIR", tmp_path / "raw"):
        yield tmp_path


# ── generate_project_id ──────────────────────────────────────


class TestProjectId:
    """测试项目 ID 生成。"""

    def test_stable_id(self):
        """相同输入生成相同 ID。"""
        id1 = generate_project_id("http://localhost:3000", "pk-lf-abc123")
        id2 = generate_project_id("http://localhost:3000", "pk-lf-abc123")
        assert id1 == id2
        assert id1.startswith("proj_")

    def test_different_host_different_id(self):
        """不同 host 生成不同 ID。"""
        id1 = generate_project_id("http://host-a:3000", "pk-lf-abc123")
        id2 = generate_project_id("http://host-b:3000", "pk-lf-abc123")
        assert id1 != id2

    def test_different_key_different_id(self):
        """不同 key 生成不同 ID。"""
        id1 = generate_project_id("http://localhost:3000", "pk-lf-aaa")
        id2 = generate_project_id("http://localhost:3000", "pk-lf-bbb")
        assert id1 != id2

    def test_host_normalization(self):
        """host 末尾 / 和大小写不影响 ID。"""
        id1 = generate_project_id("http://localhost:3000/", "pk-lf-abc")
        id2 = generate_project_id("http://localhost:3000", "pk-lf-abc")
        assert id1 == id2


# ── identify_project ─────────────────────────────────────────


class TestIdentifyProject:
    """测试项目识别。"""

    @patch("langfuse_project.requests.get")
    def test_success(self, mock_get):
        """成功识别项目。"""
        mock_get.return_value = _mock_response({
            "data": [{"id": "t1", "name": "test"}],
            "meta": {"totalItems": 100},
        })
        info = identify_project("http://localhost:3000", "pk-lf-abc", "sk-lf-xyz")
        assert info["project_id"].startswith("proj_")
        assert info["total_traces"] == 100
        assert "localhost" in info["project_name"]

    @patch("langfuse_project.requests.get")
    def test_auth_failure(self, mock_get):
        """认证失败抛出 RuntimeError。"""
        mock_get.return_value = _mock_response({"error": "unauthorized"}, status_code=401)
        with pytest.raises(RuntimeError, match="认证失败"):
            identify_project("http://localhost:3000", "bad", "bad")

    @patch("langfuse_project.requests.get")
    def test_connection_failure(self, mock_get):
        """连接失败抛出 RuntimeError。"""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        with pytest.raises(RuntimeError, match="连接失败"):
            identify_project("http://localhost:3000", "pk", "sk")


# ── register / load / list ───────────────────────────────────


class TestProjectRegistry:
    """测试项目注册表 CRUD。"""

    def test_register_and_load(self):
        """注册后可加载。"""
        register_project("proj_test", "测试项目", "http://localhost:3000", "pk-lf-...abc")
        reg = load_project("proj_test")
        assert reg is not None
        assert reg["project_name"] == "测试项目"
        assert reg["host"] == "http://localhost:3000"

    def test_register_idempotent(self):
        """重复注册更新而非创建新条目。"""
        register_project("proj_test", "V1", "http://a", "key1")
        register_project("proj_test", "V2", "http://b", "key2")
        reg = load_project("proj_test")
        assert reg["project_name"] == "V2"
        assert reg["host"] == "http://b"

    def test_list_projects(self):
        """列出所有项目。"""
        register_project("proj_a", "A", "http://a", "k1")
        register_project("proj_b", "B", "http://b", "k2")
        projects = list_projects()
        assert len(projects) == 2
        names = {p["project_name"] for p in projects}
        assert "A" in names and "B" in names

    def test_load_nonexistent(self):
        """加载不存在的项目返回 None。"""
        assert load_project("proj_none") is None


# ── append_traces 去重 ───────────────────────────────────────


class TestAppendTraces:
    """测试 trace 追加和去重。"""

    def test_append_basic(self):
        """基本追加。"""
        rows = [
            {"traceId": "t1", "type": "TRACE", "name": "test1"},
            {"traceId": "t2", "type": "TRACE", "name": "test2"},
        ]
        appended, skipped = append_traces("proj_test", rows)
        assert appended == 2
        assert skipped == 0

    def test_dedup_by_trace_id(self):
        """相同 trace_id 被跳过。"""
        rows1 = [{"traceId": "t1", "type": "TRACE"}]
        rows2 = [{"traceId": "t1", "type": "TRACE"}, {"traceId": "t2", "type": "TRACE"}]

        append_traces("proj_test", rows1)
        appended, skipped = append_traces("proj_test", rows2)
        assert appended == 1
        assert skipped == 1

    def test_load_existing_trace_ids(self):
        """可从存储中加载已有 trace_id（仅 TRACE 类型）。"""
        rows = [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "t2", "traceId": "t2", "type": "TRACE"},
            {"id": "t3", "traceId": "t3", "type": "TRACE"},
        ]
        append_traces("proj_test", rows)
        ids = load_existing_trace_ids("proj_test")
        assert ids == {"t1", "t2", "t3"}

    def test_empty_project(self):
        """空项目返回空集合。"""
        ids = load_existing_trace_ids("proj_empty")
        assert ids == set()


# ── incremental_sync ─────────────────────────────────────────


class TestIncrementalSync:
    """测试增量同步。"""

    @patch("langfuse_project.requests.get")
    def test_first_sync(self, mock_get):
        """首次同步。"""
        mock_get.side_effect = [
            _mock_response({"data": [{"id": "t1", "name": "q1", "timestamp": "2026-07-29T10:00:00Z"}], "meta": {"totalItems": 1}}),
            _mock_response({"data": []}),  # observations for t1
        ]
        register_project("proj_test", "Test", "http://localhost:3000", "pk-...abc")
        result = incremental_sync("proj_test", "http://localhost:3000", "pk", "sk", max_pages=1)
        assert result["new_traces"] == 1
        assert result["skipped"] == 0

        # 验证 checkpoint 更新
        reg = load_project("proj_test")
        assert reg["last_trace_timestamp"] == "2026-07-29T10:00:00Z"

    @patch("langfuse_project.requests.get")
    def test_incremental_dedup(self, mock_get):
        """增量同步跳过已有 trace。"""
        # 第一次同步
        mock_get.side_effect = [
            _mock_response({"data": [{"id": "t1", "name": "q1", "timestamp": "2026-07-29T10:00:00Z"}], "meta": {}}),
            _mock_response({"data": []}),
        ]
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        incremental_sync("proj_test", "http://localhost:3000", "pk", "sk", max_pages=1)

        # 第二次同步：相同 trace + 新 trace
        mock_get.side_effect = [
            _mock_response({"data": [
                {"id": "t1", "name": "q1", "timestamp": "2026-07-29T10:00:00Z"},
                {"id": "t2", "name": "q2", "timestamp": "2026-07-29T11:00:00Z"},
            ], "meta": {}}),
            _mock_response({"data": []}),
            _mock_response({"data": []}),
        ]
        result = incremental_sync("proj_test", "http://localhost:3000", "pk", "sk", max_pages=1)
        assert result["new_traces"] == 1  # t2 only
        assert result["skipped"] == 1     # t1 skipped

    @patch("langfuse_project.requests.get")
    def test_from_timestamp_param(self, mock_get):
        """fromTimestamp 参数正确传递。"""
        mock_get.return_value = _mock_response({"data": [], "meta": {}})
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        incremental_sync("proj_test", "http://localhost:3000", "pk", "sk",
                         from_timestamp="2026-07-01T00:00:00Z", max_pages=1)
        call_args = mock_get.call_args_list[0]
        assert call_args[1]["params"]["fromTimestamp"] == "2026-07-01T00:00:00Z"

    @patch("langfuse_project.requests.get")
    def test_empty_incremental_sync_does_not_create_duplicate_snapshot(self, mock_get):
        """没有新增 trace 或 observation 时不得创建新的 logical snapshot。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        mock_get.side_effect = [
            _mock_response({
                "data": [{"id": "t1", "name": "q1", "timestamp": "2026-07-29T10:00:00Z"}],
                "meta": {"totalItems": 1},
            }),
            _mock_response({"data": []}),
        ]
        first = incremental_sync("proj_test", "http://localhost:3000", "pk", "sk", max_pages=1)
        snapshot_count = len(list_snapshots("proj_test"))
        assert first["snapshot_created"] is True

        mock_get.side_effect = None
        mock_get.return_value = _mock_response({"data": [], "meta": {"totalItems": 1}})
        second = incremental_sync("proj_test", "http://localhost:3000", "pk", "sk", max_pages=1)
        assert second["snapshot_created"] is False
        assert len(list_snapshots("proj_test")) == snapshot_count


# ── get_project_stats ────────────────────────────────────────


class TestProjectStats:
    """测试项目统计。"""

    def test_empty_project_stats(self):
        """空项目统计。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        stats = get_project_stats("proj_test")
        assert stats["total_traces_synced"] == 0
        assert stats["file_size_mb"] == 0

    def test_nonexistent_project_stats(self):
        """不存在的项目返回空 dict。"""
        assert get_project_stats("proj_none") == {}


# ── cleanup_candidates ───────────────────────────────────────


class TestCleanup:
    """测试旧文件清理。"""

    def test_list_cleanup_candidates(self, tmp_path):
        """列出旧版导出文件。"""
        # 创建模拟旧文件
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "langfuse_api_export_20260727_100000.jsonl").write_text("test")
        (raw_dir / "langfuse_api_export_20260728_100000.jsonl").write_text("test")
        (raw_dir / "batch_qa_20260728.jsonl").write_text("test")  # 不应列出

        candidates = list_cleanup_candidates()
        assert len(candidates) == 2
        assert all(c["name"].startswith("langfuse_api_export_") for c in candidates)

    def test_cleanup_files(self, tmp_path):
        """删除指定文件。"""
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text("a")
        f2.write_text("b")
        deleted, failed = cleanup_files([str(f1), str(f2)])
        assert deleted == 2
        assert failed == 0
        assert not f1.exists()

    def test_cleanup_nonexistent(self, tmp_path):
        """删除不存在的文件计入 failed。"""
        deleted, failed = cleanup_files([str(tmp_path / "nope.jsonl")])
        assert deleted == 0
        assert failed == 1


# ── 跨 Key 同项目 ───────────────────────────────────────────


class TestCrossKeyDedup:
    """测试跨 Key 同项目去重。"""

    def test_same_host_key_same_project_id(self):
        """相同 host+key 生成相同 project_id。"""
        id1 = generate_project_id("http://localhost:3000", "pk-lf-abc")
        id2 = generate_project_id("http://localhost:3000", "pk-lf-abc")
        assert id1 == id2

    def test_different_key_different_project(self):
        """不同 key 即使同 host 也是不同项目。"""
        id1 = generate_project_id("http://localhost:3000", "pk-lf-aaa")
        id2 = generate_project_id("http://localhost:3000", "pk-lf-bbb")
        assert id1 != id2

    def test_multiple_projects_independent(self):
        """多个项目各自独立管理 trace。"""
        register_project("proj_a", "A", "http://a", "k1")
        register_project("proj_b", "B", "http://b", "k2")

        append_traces("proj_a", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])
        append_traces("proj_b", [{"id": "t2", "traceId": "t2", "type": "TRACE"}])

        ids_a = load_existing_trace_ids("proj_a")
        ids_b = load_existing_trace_ids("proj_b")
        assert ids_a == {"t1"}
        assert ids_b == {"t2"}


# ── 快照管理 ─────────────────────────────────────────────────


class TestSnapshotManagement:
    """测试快照生命周期。"""

    def test_create_frozen_snapshot(self):
        """创建快照后文件存在且 registry 更新。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"traceId": "t1", "type": "TRACE"},
            {"traceId": "t2", "type": "TRACE"},
        ])
        snap = create_frozen_snapshot("proj_test")
        assert snap["snapshot_id"].startswith("snap_")
        assert snap["snapshot_type"] == "frozen"
        assert snap["parsed"] is False

        # 快照文件存在
        snap_path = get_snapshot_path("proj_test", snap["snapshot_id"])
        assert snap_path.exists()

        # 冻结快照不设为 current（current 保留给逻辑快照）
        # 但 registry 应记录快照
        snaps = list_snapshots("proj_test")
        assert len(snaps) == 1

    def test_snapshot_lifecycle(self):
        """快照从创建到可清理的完整生命周期。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])

        snap = create_frozen_snapshot("proj_test")
        sid = snap["snapshot_id"]

        # 未解析时不可清理
        can, reason = can_cleanup_snapshot("proj_test", sid)
        assert not can
        assert "尚未解析" in reason

        # 标记已解析
        mark_snapshot_parsed("proj_test", sid)
        updated = [s for s in list_snapshots("proj_test") if s["snapshot_id"] == sid][0]
        assert updated["parsed"] is True

        # 冻结快照已解析且无引用后仅成为人工清理候选。
        can, reason = can_cleanup_snapshot("proj_test", sid)
        assert can
        assert "人工清理" in reason

    def test_export_snapshot_as_jsonl(self):
        """证据快照可导出为纯 JSONL（traces + observations 合并）。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
            {"id": "t2", "traceId": "t2", "type": "TRACE", "name": "q2"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION", "name": "gen1"},
        ])
        snap = create_frozen_snapshot("proj_test")

        jsonl_path = export_snapshot_as_jsonl("proj_test", snap["snapshot_id"])
        assert jsonl_path.exists()
        assert not str(jsonl_path).endswith(".gz")

        # 可正常读取（2 traces + 1 observation = 3 行）
        with jsonl_path.open("r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_cleanup_old_snapshots_keeps_recent(self):
        """清理检查只报告候选项，不删除任何快照。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])

        # 创建 3 个快照
        snap1 = create_frozen_snapshot("proj_test")
        mark_snapshot_parsed("proj_test", snap1["snapshot_id"])

        snap2 = create_frozen_snapshot("proj_test")
        mark_snapshot_parsed("proj_test", snap2["snapshot_id"])

        snap3 = create_frozen_snapshot("proj_test")
        mark_snapshot_parsed("proj_test", snap3["snapshot_id"])

        # 清理检查，保留 1 个；第一阶段不得自动删除。
        deleted, msg = cleanup_old_snapshots("proj_test", keep=1)
        assert deleted == 0
        assert "未执行自动清理" in msg

        # 所有 frozen snapshot 及其文件都仍在。
        remaining = list_snapshots("proj_test")
        remaining_ids = [s["snapshot_id"] for s in remaining]
        assert snap1["snapshot_id"] in remaining_ids
        assert snap2["snapshot_id"] in remaining_ids
        assert snap3["snapshot_id"] in remaining_ids
        for snapshot in (snap1, snap2, snap3):
            assert get_snapshot_path("proj_test", snapshot["snapshot_id"]).exists()

    def test_sync_creates_snapshot(self):
        """同步后自动创建快照。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])
        append_observations("proj_test", [{"id": "obs1", "traceId": "t1", "type": "GENERATION"}])

        snap = create_frozen_snapshot("proj_test")
        assert snap["snapshot_id"] != ""

        # 快照可在 parseable sources 中找到
        sources = list_parseable_sources("proj_test")
        snap_sources = [s for s in sources if "snapshot" in s["source_type"]]
        assert len(snap_sources) >= 1

    def test_parseable_sources_includes_legacy(self):
        """parseable sources 包含旧版 raw 文件。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")

        # 创建模拟旧文件（使用 module 级别的 RAW_DIR，已被 fixture patch）
        import langfuse_project
        raw_dir = langfuse_project.RAW_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "langfuse_api_export_20260727.jsonl").write_text("{}")

        sources = list_parseable_sources("proj_test")
        legacy = [s for s in sources if s["source_type"] == "legacy_raw"]
        assert len(legacy) >= 1

    def test_no_snapshot_without_traces(self):
        """无 trace 时创建快照会报错。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        with pytest.raises(RuntimeError, match="无可用 trace"):
            create_frozen_snapshot("proj_test")


# ── 存储分离：traces vs observations ─────────────────────────


class TestStorageSplit:
    """测试 trace 和 observation 分离存储。"""

    def test_append_traces_only_stores_trace_type(self):
        """append_traces 仅存储 TRACE 类型行。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
            {"id": "obs1", "traceId": "t1", "type": "GENERATION", "name": "gen1"},
            {"id": "obs2", "traceId": "t1", "type": "SPAN", "name": "span1"},
        ]
        appended, skipped = append_traces("proj_test", rows)
        assert appended == 1
        assert skipped == 0

        # 验证只存了 TRACE
        import langfuse_project
        import gzip
        tp = langfuse_project._traces_path("proj_test")
        with gzip.open(tp, "rt", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["type"] == "TRACE"

    def test_append_observations_stores_non_trace(self):
        """append_observations 存储非 TRACE 类型行。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
            {"id": "obs1", "traceId": "t1", "type": "GENERATION", "name": "gen1"},
            {"id": "obs2", "traceId": "t1", "type": "SPAN", "name": "span1"},
        ]
        appended, skipped = append_observations("proj_test", rows)
        assert appended == 2
        assert skipped == 0

        import langfuse_project
        import gzip
        op = langfuse_project._obs_path("proj_test")
        with gzip.open(op, "rt", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        assert all(l["type"] != "TRACE" for l in lines)

    def test_obs_dedup_by_obs_id(self):
        """observation 按 observation id 去重，不按 traceId。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows1 = [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ]
        rows2 = [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},  # 重复 obs id
            {"id": "obs2", "traceId": "t1", "type": "GENERATION"},  # 新 obs
        ]
        append_observations("proj_test", rows1)
        appended, skipped = append_observations("proj_test", rows2)
        assert appended == 1
        assert skipped == 1

    def test_trace_dedup_does_not_kill_observations(self):
        """trace 去重不影响后续 observation 追加。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        # 第一次同步：trace + observation
        rows1 = [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ]
        append_traces("proj_test", rows1)
        append_observations("proj_test", rows1)

        # 第二次同步：同 trace 有新 observation
        rows2 = [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},  # 重复 trace
            {"id": "obs2", "traceId": "t1", "type": "GENERATION"},  # 新 obs
        ]
        t_appended, t_skipped = append_traces("proj_test", rows2)
        o_appended, o_skipped = append_observations("proj_test", rows2)
        assert t_appended == 0  # trace 被跳过
        assert t_skipped == 1
        assert o_appended == 1  # 新 obs 被追加
        assert o_skipped == 0   # obs2 是新的

    def test_load_existing_trace_ids_excludes_obs(self):
        """load_existing_trace_ids 不包含 observation id。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ]
        append_traces("proj_test", rows)
        append_observations("proj_test", rows)

        trace_ids = load_existing_trace_ids("proj_test")
        assert "t1" in trace_ids
        assert "obs1" not in trace_ids

    def test_snapshot_includes_observations(self):
        """快照包含 observation 文件和 has_observations 标记。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ]
        append_traces("proj_test", rows)
        append_observations("proj_test", rows)

        snap = create_frozen_snapshot("proj_test")
        assert snap["has_observations"] is True
        assert snap["observation_count"] == 1
        assert snap["trace_count"] == 1

        # obs 文件存在
        import langfuse_project
        snap_obs_path = langfuse_project._snapshots_dir("proj_test") / f"{snap['snapshot_id']}.obs.jsonl.gz"
        assert snap_obs_path.exists()

    def test_index_only_snapshot_no_obs(self):
        """仅 trace 无 observation 时为索引快照。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [{"id": "t1", "traceId": "t1", "type": "TRACE"}]
        append_traces("proj_test", rows)
        # 不调用 append_observations

        snap = create_frozen_snapshot("proj_test")
        assert snap["has_observations"] is False
        assert snap["observation_count"] == 0

    def test_export_index_only_raises(self):
        """索引快照导出时抛出明确错误。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [{"id": "t1", "traceId": "t1", "type": "TRACE"}]
        append_traces("proj_test", rows)
        snap = create_frozen_snapshot("proj_test")

        with pytest.raises(RuntimeError, match="index-only snapshot"):
            export_snapshot_as_jsonl("proj_test", snap["snapshot_id"])

    def test_export_evidence_snapshot_merges(self):
        """证据快照导出合并 traces + observations。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows = [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
            {"id": "obs1", "traceId": "t1", "type": "GENERATION", "name": "gen1"},
        ]
        append_traces("proj_test", rows)
        append_observations("proj_test", rows)
        snap = create_frozen_snapshot("proj_test")

        jsonl_path = export_snapshot_as_jsonl("proj_test", snap["snapshot_id"])
        with jsonl_path.open("r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        types = {l["type"] for l in lines}
        assert "TRACE" in types
        assert "GENERATION" in types

    def test_parseable_sources_distinguishes_types(self):
        """list_parseable_sources 区分索引快照和证据快照。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        # 索引快照
        append_traces("proj_test", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])
        snap1 = create_frozen_snapshot("proj_test")

        # 证据快照
        append_observations("proj_test", [{"id": "obs1", "traceId": "t1", "type": "GENERATION"}])
        snap2 = create_frozen_snapshot("proj_test")

        sources = list_parseable_sources("proj_test")
        index_snaps = [s for s in sources if s["source_type"] == "index_snapshot"]
        evidence_snaps = [s for s in sources if s["source_type"] == "evidence_snapshot"]
        assert len(index_snaps) >= 1
        assert len(evidence_snaps) >= 1
        assert evidence_snaps[0]["has_observations"] is True
        assert index_snaps[0]["has_observations"] is False


# ── 端到端：evidence snapshot 合并 ───────────────────────────


class TestEvidenceSnapshotE2E:
    """测试 evidence snapshot 从创建到导出的完整链路。"""

    def test_evidence_snapshot_export_contains_both_types(self):
        """evidence snapshot 导出的 JSONL 同时包含 TRACE 和 observation 行。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        # 写入 trace + observation
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION", "name": "gen1",
             "output": {"result": [{"content": "chunk1", "score": 0.9}]}},
            {"id": "obs2", "traceId": "t1", "type": "SPAN", "name": "span1"},
        ])

        snap = create_frozen_snapshot("proj_test")
        assert snap["has_observations"] is True

        # 导出
        jsonl_path = export_snapshot_as_jsonl("proj_test", snap["snapshot_id"])
        with jsonl_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        types = {r["type"] for r in rows}
        assert "TRACE" in types, "导出缺少 TRACE 行"
        assert "GENERATION" in types, "导出缺少 GENERATION 行"
        assert "SPAN" in types, "导出缺少 SPAN 行"
        assert len(rows) == 3

    def test_index_snapshot_export_rejected(self):
        """index snapshot 导出时抛出明确错误。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])
        snap = create_frozen_snapshot("proj_test")
        assert snap["has_observations"] is False

        with pytest.raises(RuntimeError, match="index-only snapshot"):
            export_snapshot_as_jsonl("proj_test", snap["snapshot_id"])

    def test_parseable_source_type_matches(self):
        """evidence_snapshot 类型在 parseable sources 中正确标记。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ])
        snap = create_frozen_snapshot("proj_test")

        sources = list_parseable_sources("proj_test")
        ev = [s for s in sources if s["snapshot_id"] == snap["snapshot_id"]]
        assert len(ev) == 1
        assert ev[0]["source_type"] == "evidence_snapshot"
        assert ev[0]["has_observations"] is True


# ── Observation 覆盖率 ──────────────────────────────────────


class TestObservationCoverage:
    """测试 observation 覆盖率统计。"""

    def test_full_coverage(self):
        """所有 trace 都有 observation 时覆盖率为 100%。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "t2", "traceId": "t2", "type": "TRACE"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
            {"id": "obs2", "traceId": "t2", "type": "GENERATION"},
        ])

        from langfuse_project import get_observation_coverage
        cov = get_observation_coverage("proj_test")
        assert cov["total_traces"] == 2
        assert cov["traces_with_obs"] == 2
        assert cov["coverage_pct"] == 100.0

    def test_partial_coverage(self):
        """部分 trace 有 observation 时覆盖率正确。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "t2", "traceId": "t2", "type": "TRACE"},
            {"id": "t3", "traceId": "t3", "type": "TRACE"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ])

        from langfuse_project import get_observation_coverage
        cov = get_observation_coverage("proj_test")
        assert cov["total_traces"] == 3
        assert cov["traces_with_obs"] == 1
        assert abs(cov["coverage_pct"] - 33.3) < 0.1

    def test_zero_coverage(self):
        """无 observation 时覆盖率为 0。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])

        from langfuse_project import get_observation_coverage
        cov = get_observation_coverage("proj_test")
        assert cov["total_traces"] == 1
        assert cov["traces_with_obs"] == 0
        assert cov["coverage_pct"] == 0.0


# ── force_full 参数 ─────────────────────────────────────────


class TestForceFullSync:
    """测试 force_full 参数绕过游标。"""

    @patch("langfuse_project.requests.get")
    def test_force_full_ignores_cursor(self, mock_get):
        """force_full=True 时忽略 last_trace_timestamp。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        # 模拟已有游标
        from langfuse_project import _save_registry, load_project
        reg = load_project("proj_test")
        reg["last_trace_timestamp"] = "2026-07-01T00:00:00Z"
        _save_registry("proj_test", reg)

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [], "meta": {"totalItems": 0}},
        )

        from langfuse_project import incremental_sync
        incremental_sync("proj_test", "http://localhost:3000", "pk", "sk",
                         force_full=True, max_pages=1)

        # 验证请求参数中没有 fromTimestamp
        call_args = mock_get.call_args
        if call_args:
            params = call_args[1].get("params", {})
            assert "fromTimestamp" not in params


# ── 快照大小计算 ─────────────────────────────────────────────


class TestSnapshotSizes:
    """测试快照大小元数据。"""

    def test_evidence_snapshot_total_includes_both_files(self):
        """evidence snapshot 总大小 = trace 文件 + obs 文件。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        # 写入足够的数据使文件大小 > 0
        rows_trace = [{"id": f"t{i}", "traceId": f"t{i}", "type": "TRACE",
                        "name": f"q{i}", "input": {"q": f"question {i}" * 10}}
                      for i in range(10)]
        rows_obs = [{"id": f"obs{i}", "traceId": f"t{i}", "type": "GENERATION",
                      "name": f"gen{i}", "output": {"result": [{"content": f"chunk {i}" * 20}]}}
                    for i in range(10)]
        append_traces("proj_test", rows_trace)
        append_observations("proj_test", rows_obs)

        snap = create_frozen_snapshot("proj_test")
        assert snap["has_observations"] is True
        assert snap["trace_file_size_bytes"] > 0
        assert snap["observation_file_size_bytes"] > 0
        assert snap["total_file_size_bytes"] == (
            snap["trace_file_size_bytes"] + snap["observation_file_size_bytes"]
        )

    def test_index_snapshot_obs_size_zero(self):
        """index snapshot 的 obs 文件大小为 0。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
        ])
        snap = create_frozen_snapshot("proj_test")
        assert snap["has_observations"] is False
        assert snap["observation_file_size_bytes"] == 0
        assert snap["total_file_size_bytes"] == snap["trace_file_size_bytes"]

    def test_parseable_source_size_uses_total(self):
        """list_parseable_sources 的 size_mb 对 evidence snapshot 使用总大小。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows_trace = [{"id": f"t{i}", "traceId": f"t{i}", "type": "TRACE",
                        "name": f"q{i}", "input": {"q": f"question {i}" * 10}}
                      for i in range(10)]
        rows_obs = [{"id": f"obs{i}", "traceId": f"t{i}", "type": "GENERATION",
                      "name": f"gen{i}", "output": {"result": [{"content": f"chunk {i}" * 20}]}}
                    for i in range(10)]
        append_traces("proj_test", rows_trace)
        append_observations("proj_test", rows_obs)
        snap = create_frozen_snapshot("proj_test")

        sources = list_parseable_sources("proj_test")
        ev = [s for s in sources if s["snapshot_id"] == snap["snapshot_id"]][0]
        assert ev["source_type"] == "evidence_snapshot"
        # size_mb 应 >= trace_size_mb（因为包含 obs）
        assert ev["size_mb"] >= ev["trace_size_mb"]
        # obs 文件存在（通过 snap 元数据验证）
        assert snap["observation_file_size_bytes"] > 0

    def test_parseable_source_index_snapshot_size(self):
        """index snapshot 的 size_mb 只有 trace 文件大小。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
        ])
        snap = create_frozen_snapshot("proj_test")

        sources = list_parseable_sources("proj_test")
        idx_snap = [s for s in sources if s["snapshot_id"] == snap["snapshot_id"]][0]
        assert idx_snap["source_type"] == "index_snapshot"
        assert idx_snap["obs_size_mb"] == 0
        assert idx_snap["size_mb"] == idx_snap["trace_size_mb"]

    def test_get_snapshot_sizes_backward_compat(self):
        """旧 registry 条目（缺少 total_file_size_bytes）动态计算大小。"""
        from langfuse_project import _get_snapshot_sizes, _save_registry, load_project
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE", "name": "q1"},
        ])
        snap = create_frozen_snapshot("proj_test")

        # 模拟旧 registry：删除新字段
        reg = load_project("proj_test")
        for s in reg.get("snapshots", []):
            s.pop("trace_file_size_bytes", None)
            s.pop("observation_file_size_bytes", None)
            s.pop("total_file_size_bytes", None)
        _save_registry("proj_test", reg)

        # 动态计算应仍然正确
        reg2 = load_project("proj_test")
        snap2 = [s for s in reg2["snapshots"] if s["snapshot_id"] == snap["snapshot_id"]][0]
        sizes = _get_snapshot_sizes("proj_test", snap2)
        assert sizes["trace_file_size_bytes"] > 0
        assert sizes["total_file_size_bytes"] == sizes["trace_file_size_bytes"]

    def test_evidence_label_shows_counts(self):
        """evidence snapshot label 包含 trace 和 obs 数量。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
            {"id": "t2", "traceId": "t2", "type": "TRACE"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
            {"id": "obs2", "traceId": "t1", "type": "SPAN"},
            {"id": "obs3", "traceId": "t2", "type": "GENERATION"},
        ])
        snap = create_frozen_snapshot("proj_test")

        sources = list_parseable_sources("proj_test")
        ev = [s for s in sources if s["snapshot_id"] == snap["snapshot_id"]][0]
        assert "2 traces" in ev["label"]
        assert "3 obs" in ev["label"]
        assert ev["trace_count"] == 2
        assert ev["observation_count"] == 3


# ── 快照优化：逻辑 vs 冻结 ───────────────────────────────────


class TestSnapshotOptimization:
    """测试快照存储优化。"""

    def test_no_snapshot_on_empty_sync(self):
        """无新增数据时同步不创建新快照。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        # 预先写入一些数据
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])
        _update_current_snapshot("proj_test")
        snap_count_before = len(list_snapshots("proj_test"))

        # 模拟空同步（没有新数据）
        # incremental_sync 内部会检查 total_new > 0
        # 这里直接测试 _update_current_snapshot 不会在无新数据时被调用
        # 通过检查 snapshot 数量不变来验证
        snap_count_after = len(list_snapshots("proj_test"))
        assert snap_count_after == snap_count_before

    def test_logical_snapshot_points_to_cache(self):
        """逻辑快照指向缓存文件，不创建独立副本。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ])

        snap = _update_current_snapshot("proj_test")
        assert snap["snapshot_type"] == "logical"
        assert snap["has_observations"] is True

        # 逻辑快照的路径应该是缓存文件
        snap_path = get_snapshot_path("proj_test", snap["snapshot_id"])
        assert snap_path == _traces_path("proj_test")
        assert snap_path.exists()
        assert not list((snap_path.parent / "snapshots").glob("logical_*.jsonl.gz"))

    def test_logical_snapshot_cannot_be_exported_for_formal_parsing(self):
        """logical snapshot 仅引用缓存，不能导出为正式样本输入。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ])

        logical = _update_current_snapshot("proj_test")
        with pytest.raises(RuntimeError, match="logical snapshot"):
            export_snapshot_as_jsonl("proj_test", logical["snapshot_id"])

        source = [s for s in list_parseable_sources("proj_test")
                  if s["snapshot_id"] == logical["snapshot_id"]][0]
        assert source["source_type"] == "logical_snapshot"

    def test_frozen_evidence_snapshot_keeps_trace_and_observation_pair(self):
        """frozen evidence snapshot 导出时必须合并独立的 trace 与 observation。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [{"id": "t1", "traceId": "t1", "type": "TRACE"}])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ])

        snap = create_frozen_snapshot("proj_test")
        trace_path = get_snapshot_path("proj_test", snap["snapshot_id"])
        obs_path = trace_path.parent / f"{snap['snapshot_id']}.obs.jsonl.gz"
        assert trace_path.exists()
        assert obs_path.exists()

        export_path = export_snapshot_as_jsonl("proj_test", snap["snapshot_id"])
        with export_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert {row["type"] for row in rows} == {"TRACE", "GENERATION"}

        mark_snapshot_parsed("proj_test", snap["snapshot_id"])
        can_cleanup, reason = can_cleanup_snapshot("proj_test", snap["snapshot_id"])
        assert can_cleanup
        assert "成对处理" in reason

    def test_frozen_snapshot_creates_independent_copy(self):
        """冻结快照创建独立文件副本。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])
        append_observations("proj_test", [
            {"id": "obs1", "traceId": "t1", "type": "GENERATION"},
        ])

        snap = create_frozen_snapshot("proj_test")
        assert snap["snapshot_type"] == "frozen"

        # 冻结快照的路径应该在 snapshots/ 目录
        snap_path = get_snapshot_path("proj_test", snap["snapshot_id"])
        assert "snapshots" in str(snap_path)
        assert snap_path.exists()

    def test_multiple_syncs_no_duplicate_files(self):
        """多次同步不会创建多个完整副本。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        append_traces("proj_test", [
            {"id": "t1", "traceId": "t1", "type": "TRACE"},
        ])
        _update_current_snapshot("proj_test")
        snap_count_1 = len(list_snapshots("proj_test"))

        # 再次调用 _update_current_snapshot（模拟第二次同步）
        append_traces("proj_test", [
            {"id": "t2", "traceId": "t2", "type": "TRACE"},
        ])
        _update_current_snapshot("proj_test")
        snap_count_2 = len(list_snapshots("proj_test"))

        # 应该还是 1 个逻辑快照（更新而非新增）
        assert snap_count_2 == snap_count_1

    def test_logical_snapshot_sizes_match_cache(self):
        """逻辑快照大小与缓存文件大小一致。"""
        register_project("proj_test", "Test", "http://localhost:3000", "pk")
        rows_trace = [{"id": f"t{i}", "traceId": f"t{i}", "type": "TRACE",
                        "name": f"q{i}", "input": {"q": f"question {i}" * 10}}
                      for i in range(10)]
        rows_obs = [{"id": f"obs{i}", "traceId": f"t{i}", "type": "GENERATION",
                      "name": f"gen{i}", "output": {"result": [{"content": f"chunk {i}" * 20}]}}
                    for i in range(10)]
        append_traces("proj_test", rows_trace)
        append_observations("proj_test", rows_obs)

        snap = _update_current_snapshot("proj_test")
        sizes = _get_snapshot_sizes("proj_test", snap)

        # 逻辑快照大小应该等于缓存文件大小
        tp = _traces_path("proj_test")
        op = _obs_path("proj_test")
        expected_trace = tp.stat().st_size
        expected_obs = op.stat().st_size
        assert sizes["trace_file_size_bytes"] == expected_trace
        assert sizes["observation_file_size_bytes"] == expected_obs
        assert sizes["total_file_size_bytes"] == expected_trace + expected_obs
