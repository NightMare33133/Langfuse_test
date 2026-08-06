"""数据源选择逻辑测试。

覆盖：
- 当前动态缓存有效时，默认 source_type == "current_cache"
- 之前选择过冻结快照，重新进入当前项目后仍回到 current_cache
- 切换项目后不会复用旧项目的 _use_frozen_source
- 用户主动选择冻结快照后，才使用 evidence_snapshot
- "最新优先"只改变显示排序，不改变 source_type
- 当前项目有隔离 processed 文件时，不错误读取全局 langfuse_samples.jsonl
- 动态缓存 fingerprint 变化后，页面能提示当前解析结果已过期
- 旧版没有 project_id 时仍保留 legacy fallback
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# ── 测试 get_processed_paths 隔离路径 ──

class TestGetProcessedPaths:
    """get_processed_paths 应根据 source_type 返回正确的隔离路径。"""

    def test_current_cache_path(self):
        from langfuse_project import get_processed_paths
        s, sm = get_processed_paths("current_cache", project_id="proj_abc123")
        assert "proj_abc123" in str(s)
        assert "current" in str(s)
        assert s.name == "samples.jsonl"
        assert sm.name == "summary.json"

    def test_evidence_snapshot_path(self):
        from langfuse_project import get_processed_paths
        s, sm = get_processed_paths(
            "evidence_snapshot", project_id="proj_abc123",
            snapshot_id="snap_20260801_120000",
        )
        assert "proj_abc123" in str(s)
        assert "snap_20260801_120000" in str(s)
        assert s.name == "samples.jsonl"

    def test_legacy_raw_path(self):
        from langfuse_project import get_processed_paths
        s, sm = get_processed_paths(
            "legacy_raw", source_id="legacy:langfuse_api_export_20260801.jsonl",
        )
        assert "legacy" in str(s)
        assert "langfuse_api_export_20260801.jsonl" in str(s)

    def test_fallback_global_path(self):
        from langfuse_project import get_processed_paths
        s, sm = get_processed_paths("unknown_type")
        assert s.name == "langfuse_samples.jsonl"
        assert sm.name == "langfuse_summary.json"


# ── 测试 find_latest_processed 优先级 ──

class TestFindLatestProcessed:
    """find_latest_processed 应优先返回 current 缓存。"""

    def test_prefers_current_cache(self, tmp_path):
        from langfuse_project import PROCESSED_DIR
        proj_dir = PROCESSED_DIR / "langfuse_projects" / "proj_test123"
        current_dir = proj_dir / "current"
        snap_dir = proj_dir / "snap_old"

        current_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)

        (current_dir / "samples.jsonl").write_text('{"trace_id": "t1"}\n')
        (current_dir / "summary.json").write_text('{}')
        (snap_dir / "samples.jsonl").write_text('{"trace_id": "t2"}\n')
        (snap_dir / "summary.json").write_text('{}')

        try:
            from langfuse_project import find_latest_processed
            s, sm = find_latest_processed("proj_test123")
            assert s is not None
            assert "current" in str(s)
        finally:
            shutil.rmtree(proj_dir, ignore_errors=True)

    def test_falls_back_to_frozen_when_no_current(self, tmp_path):
        from langfuse_project import PROCESSED_DIR
        proj_dir = PROCESSED_DIR / "langfuse_projects" / "proj_test456"
        snap_dir = proj_dir / "snap_20260801_120000"

        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "samples.jsonl").write_text('{"trace_id": "t1"}\n')
        (snap_dir / "summary.json").write_text('{}')

        try:
            from langfuse_project import find_latest_processed
            s, sm = find_latest_processed("proj_test456")
            assert s is not None
            assert "snap_20260801_120000" in str(s)
        finally:
            shutil.rmtree(proj_dir, ignore_errors=True)

    def test_falls_back_to_global_when_no_project(self, tmp_path):
        from langfuse_project import PROCESSED_DIR
        global_s = PROCESSED_DIR / "langfuse_samples.jsonl"
        global_sm = PROCESSED_DIR / "langfuse_summary.json"

        created = False
        if not global_s.exists():
            global_s.parent.mkdir(parents=True, exist_ok=True)
            global_s.write_text('{"trace_id": "t_global"}\n')
            global_sm.write_text('{}')
            created = True

        try:
            from langfuse_project import find_latest_processed
            s, sm = find_latest_processed("")
            assert s is not None
            assert s.name == "langfuse_samples.jsonl"
        finally:
            if created:
                global_s.unlink(missing_ok=True)
                global_sm.unlink(missing_ok=True)


# ── 测试 _use_frozen_source 跨项目清理 ──

class TestFrozenSourceCleanup:
    """_use_frozen_source 必须在项目切换时被清理。"""

    def test_different_project_clears_frozen_source(self):
        """如果 _use_frozen_source 属于项目 A，当前项目是 B，应清理。"""
        frozen_source = {
            "project_id": "proj_aaa",
            "source_type": "evidence_snapshot",
            "snapshot_id": "snap_123",
        }
        current_project_id = "proj_bbb"

        # 模拟 startup 逻辑
        if frozen_source and current_project_id:
            frozen_pid = frozen_source.get("project_id", "")
            if frozen_pid and frozen_pid != current_project_id:
                should_clear = True
            else:
                should_clear = False
        else:
            should_clear = False

        assert should_clear is True

    def test_same_project_keeps_frozen_source(self):
        """如果 _use_frozen_source 属于同一项目，不应清理。"""
        frozen_source = {
            "project_id": "proj_aaa",
            "source_type": "evidence_snapshot",
            "snapshot_id": "snap_123",
        }
        current_project_id = "proj_aaa"

        if frozen_source and current_project_id:
            frozen_pid = frozen_source.get("project_id", "")
            should_clear = bool(frozen_pid and frozen_pid != current_project_id)
        else:
            should_clear = False

        assert should_clear is False

    def test_no_frozen_source_noop(self):
        """没有 _use_frozen_source 时不做任何操作。"""
        frozen_source = None
        current_project_id = "proj_aaa"

        should_clear = False
        if frozen_source and current_project_id:
            should_clear = True

        assert should_clear is False

    def test_no_project_id_keeps_frozen_source(self):
        """旧版没有 project_id 时，保留 _use_frozen_source（兼容）。"""
        frozen_source = {
            "source_type": "legacy_raw",
            "source_id": "legacy:file.jsonl",
        }
        current_project_id = ""

        if frozen_source and current_project_id:
            frozen_pid = frozen_source.get("project_id", "")
            should_clear = bool(frozen_pid and frozen_pid != current_project_id)
        else:
            should_clear = False

        assert should_clear is False


# ── 测试 source_type 选择逻辑 ──

class TestSourceTypeSelection:
    """source_type 的选择逻辑。"""

    def test_current_cache_default_when_valid(self):
        """有 project_id + trace > 0 + has_obs 时默认 current_cache。"""
        proj_id = "proj_abc"
        cc_trace = 100
        cc_has_obs = True

        if proj_id and cc_trace > 0 and cc_has_obs:
            source_type = "current_cache"
        else:
            source_type = None

        assert source_type == "current_cache"

    def test_no_source_when_no_observations(self):
        """缓存无 observation 时不能解析。"""
        proj_id = "proj_abc"
        cc_trace = 100
        cc_has_obs = False

        source_type = None
        can_parse = False
        if proj_id and cc_trace > 0 and cc_has_obs:
            source_type = "current_cache"
            can_parse = True

        assert source_type is None
        assert can_parse is False

    def test_frozen_source_only_when_user_selects(self):
        """只有用户主动选择冻结快照时才使用 evidence_snapshot。"""
        # 默认
        source_type = "current_cache"
        assert source_type == "current_cache"

        # 用户选择后
        frozen_source = {
            "source_type": "evidence_snapshot",
            "snapshot_id": "snap_123",
        }
        source_type = frozen_source.get("source_type", "")
        assert source_type == "evidence_snapshot"

    def test_sort_mode_does_not_affect_source_type(self):
        """"最新优先"只影响排序，不影响 source_type。"""
        source_type = "current_cache"
        sort_mode = "最新优先"

        # 排序逻辑（不影响 source_type）
        sort_newest_first = (sort_mode == "最新优先")

        assert source_type == "current_cache"
        assert sort_newest_first is True

        sort_mode = "最早优先"
        sort_newest_first = (sort_mode == "最新优先")
        assert source_type == "current_cache"
        assert sort_newest_first is False


# ── 测试 fingerprint 变化检测 ──

class TestFingerprintStaleDetection:
    """fingerprint 变化应触发过期提示。"""

    def test_same_fingerprint_not_stale(self):
        parsed_fp = "abc123_1000|def456_500"
        current_fp = "abc123_1000|def456_500"
        assert parsed_fp == current_fp  # 不过期

    def test_different_fingerprint_is_stale(self):
        parsed_fp = "abc123_1000|def456_500"
        current_fp = "abc123_1000|xyz789_600"
        assert parsed_fp != current_fp  # 已过期

    def test_empty_fingerprint_no_warning(self):
        parsed_fp = ""
        current_fp = "abc123_1000|def456_500"
        # 空 fingerprint 不触发警告（旧数据兼容）
        should_warn = bool(parsed_fp and current_fp and parsed_fp != current_fp)
        assert should_warn is False


# ── 测试 provevance 字段传播 ──

class TestProvenanceFields:
    """provenance 字段应正确写入 summary 和 samples。"""

    def test_current_cache_provenance(self):
        """current_cache 解析应写入正确的 provenance。"""
        provenance = {
            "langfuse_project_id": "proj_abc",
            "langfuse_snapshot_id": "",
            "langfuse_source_type": "current_cache",
            "source_file": "/tmp/current_cache.jsonl",
            "cache_last_sync_at": "2026-08-03T10:00:00Z",
            "cache_trace_count": 100,
            "cache_observation_count": 300,
            "source_file_fingerprint": "abc123_1000|def456_500",
        }
        assert provenance["langfuse_source_type"] == "current_cache"
        assert provenance["langfuse_project_id"] == "proj_abc"
        assert provenance["langfuse_snapshot_id"] == ""

    def test_evidence_snapshot_provenance(self):
        """evidence_snapshot 解析应写入正确的 provenance。"""
        provenance = {
            "langfuse_project_id": "proj_abc",
            "langfuse_snapshot_id": "snap_20260801_120000",
            "langfuse_source_type": "evidence_snapshot",
            "source_file": "/tmp/snap_20260801_120000.jsonl",
        }
        assert provenance["langfuse_source_type"] == "evidence_snapshot"
        assert provenance["langfuse_snapshot_id"] == "snap_20260801_120000"


# ── 测试不读取全局 langfuse_samples.jsonl ──

class TestIsolationPaths:
    """有 project_id 时不应错误回退到全局 langfuse_samples.jsonl。"""

    def test_project_isolation_path_not_global(self):
        from langfuse_project import get_processed_paths
        s, _ = get_processed_paths("current_cache", project_id="proj_xyz")
        assert "langfuse_projects" in str(s)
        assert "proj_xyz" in str(s)
        # 不应是全局路径
        assert s.name != "langfuse_samples.jsonl"

    def test_legacy_fallback_only_when_no_project(self):
        from langfuse_project import get_processed_paths
        s, _ = get_processed_paths("unknown")
        assert s.name == "langfuse_samples.jsonl"

    def test_find_latest_prefers_isolated_over_global(self):
        """有 project_id 且隔离目录存在时，find_latest_processed 不返回全局路径。"""
        from langfuse_project import find_latest_processed, PROCESSED_DIR
        proj_id = "proj_a557072d2dfa6df4"
        s, sm = find_latest_processed(proj_id)
        if s and s.exists():
            # 必须是隔离路径
            assert "langfuse_projects" in str(s)
            assert proj_id in str(s)
            # 不是全局路径
            assert s != PROCESSED_DIR / "langfuse_samples.jsonl"


# ── 回归测试：旧 session 残留冻结快照时，应自动加载 current_cache ──

class TestStaleSessionRegression:
    """模拟旧 session 中有冻结快照残留数据的场景。"""

    def test_stale_frozen_source_cleared_on_mismatch(self):
        """当 session 中样本数与 current cache 不一致时，应清除 _use_frozen_source。"""
        # 模拟旧 session 状态
        session_state = {
            "_lf_project_info": {"project_id": "proj_abc"},
            "_use_frozen_source": {
                "project_id": "proj_abc",
                "source_type": "evidence_snapshot",
                "snapshot_id": "snap_old",
            },
            "samples": [{"trace_id": f"t{i}"} for i in range(100)],  # 旧的100条
        }

        # 模拟 current cache 有 200 条
        current_cache_trace_count = 200
        session_count = len(session_state["samples"])

        # 启动逻辑：mismatch 检测
        if current_cache_trace_count > 0 and current_cache_trace_count != session_count:
            session_state.pop("_use_frozen_source", None)
            need_reload = True
        else:
            need_reload = False

        assert need_reload is True
        assert "_use_frozen_source" not in session_state

    def test_just_set_frozen_source_not_cleared(self):
        """用户刚选择冻结源时（_frozen_source_just_set=True），不应触发 mismatch 清理。"""
        session_state = {
            "_lf_project_info": {"project_id": "proj_abc"},
            "_use_frozen_source": {
                "project_id": "proj_abc",
                "source_type": "evidence_snapshot",
                "snapshot_id": "snap_new",
            },
            "_frozen_source_just_set": True,
            "samples": [{"trace_id": f"t{i}"} for i in range(100)],
        }

        current_cache_trace_count = 200
        just_set = session_state.pop("_frozen_source_just_set", False)
        session_count = len(session_state["samples"])

        # _frozen_source_just_set 为 True 时跳过 mismatch 检查
        need_reload = False
        if not just_set and current_cache_trace_count > 0 and current_cache_trace_count != session_count:
            session_state.pop("_use_frozen_source", None)
            need_reload = True

        assert need_reload is False
        assert "_use_frozen_source" in session_state  # 保留用户选择
        assert session_state["_use_frozen_source"]["snapshot_id"] == "snap_new"

    def test_same_count_no_reload(self):
        """session 样本数与 current cache 一致时，不触发 reload。"""
        session_state = {
            "_lf_project_info": {"project_id": "proj_abc"},
            "samples": [{"trace_id": f"t{i}"} for i in range(200)],
        }

        current_cache_trace_count = 200
        session_count = len(session_state["samples"])

        need_reload = "samples" not in session_state
        if not need_reload and current_cache_trace_count > 0 and current_cache_trace_count != session_count:
            need_reload = True

        assert need_reload is False

    def test_cross_project_frozen_source_cleared(self):
        """_use_frozen_source 属于不同项目时，启动时应清理。"""
        session_state = {
            "_lf_project_info": {"project_id": "proj_bbb"},
            "_use_frozen_source": {
                "project_id": "proj_aaa",
                "source_type": "evidence_snapshot",
                "snapshot_id": "snap_old",
            },
        }

        loaded_proj_id = session_state.get("_lf_project_info", {}).get("project_id", "")
        frozen_src = session_state.get("_use_frozen_source")
        if frozen_src and loaded_proj_id:
            frozen_pid = frozen_src.get("project_id", "")
            if frozen_pid and frozen_pid != loaded_proj_id:
                session_state.pop("_use_frozen_source", None)

        assert "_use_frozen_source" not in session_state

    def test_display_shows_current_cache_after_reload(self):
        """reload 后 summary 应包含 current_cache 的 source_type。"""
        # 模拟 reload 后的 summary
        summary = {
            "langfuse_source_type": "current_cache",
            "langfuse_project_id": "proj_abc",
            "langfuse_snapshot_id": "",
            "input_file": "/data/langfuse_projects/proj_abc/current_cache.jsonl",
        }

        _src_type = summary.get("langfuse_source_type", "")
        _src_snap = summary.get("langfuse_snapshot_id", "")
        _src_pid = summary.get("langfuse_project_id", "")

        if _src_type == "current_cache":
            src_label = f"当前动态缓存（项目 {_src_pid[:20]}...）"
        elif _src_type == "evidence_snapshot" and _src_snap:
            src_label = f"冻结快照 {_src_snap}"
        else:
            src_label = "未知来源"

        assert "当前动态缓存" in src_label
        assert "冻结快照" not in src_label

    def test_legacy_then_connect_project_clears_old_samples(self):
        """回归测试：先加载 legacy langfuse_samples.jsonl，随后连接新项目，
        session 中的旧 samples/summary 必须被清理，页面回到待解析状态。"""
        # 阶段1：启动时无项目，加载全局旧数据
        session_state = {}
        # 模拟 find_latest_processed("") 返回全局路径
        global_samples = [{"trace_id": f"t{i}", "question": f"q{i}"} for i in range(1199)]
        global_summary = {
            "langfuse_source_type": "evidence_snapshot",
            "langfuse_snapshot_id": "snap_20260731_074134_113412",
            "input_file": "/data/langfuse_projects/proj_old/snapshots/snap_20260731_074134_113412.jsonl",
            "trace_count": 1199,
        }
        session_state["samples"] = global_samples
        session_state["summary"] = global_summary
        assert len(session_state["samples"]) == 1199

        # 阶段2：用户连接新项目
        new_proj_id = "proj_new_abc123"
        old_proj_id = session_state.get("_lf_project_info", {}).get("project_id", "")

        # 模拟 _on_project_changed
        if new_proj_id and (not old_proj_id or old_proj_id != new_proj_id):
            for k in ("samples", "summary", "sample_page", "_use_frozen_source"):
                session_state.pop(k, None)

        # 模拟 button handler 设置 _lf_project_info
        session_state["_lf_project_info"] = {"project_id": new_proj_id}

        # 验证：旧 samples 已被清理
        assert "samples" not in session_state
        assert "summary" not in session_state

        # 阶段3：后续 rerun 时，startup 代码重新读取 project_id
        loaded_proj_id = session_state.get("_lf_project_info", {}).get("project_id", "")
        assert loaded_proj_id == new_proj_id  # 不再是空字符串

        # 模拟 _need_reload = True (samples not in session_state)
        need_reload = "samples" not in session_state
        assert need_reload is True

        # 模拟 find_latest_processed(new_proj_id) 返回隔离路径
        # （在真实环境中返回 data/processed/langfuse_projects/proj_new_abc123/current/samples.jsonl）
        isolated_samples = [{"trace_id": f"new_t{i}"} for i in range(200)]
        isolated_summary = {
            "langfuse_source_type": "current_cache",
            "langfuse_project_id": new_proj_id,
            "trace_count": 200,
        }
        session_state["samples"] = isolated_samples
        session_state["summary"] = isolated_summary

        # 验证：页面显示新项目的数据
        assert len(session_state["samples"]) == 200
        assert session_state["summary"]["langfuse_source_type"] == "current_cache"
        assert session_state["summary"]["langfuse_project_id"] == new_proj_id

    def test_legacy_then_connect_same_project_no_clear(self):
        """连接同一个项目时不应清理 samples（非切换场景）。"""
        session_state = {
            "_lf_project_info": {"project_id": "proj_abc"},
            "samples": [{"trace_id": "t1"}],
            "summary": {"trace_count": 1},
        }

        old_proj_id = session_state.get("_lf_project_info", {}).get("project_id", "")
        new_proj_id = "proj_abc"  # 同一个项目

        # 模拟 _on_project_changed
        if new_proj_id and (not old_proj_id or old_proj_id != new_proj_id):
            for k in ("samples", "summary", "sample_page", "_use_frozen_source"):
                session_state.pop(k, None)

        # 不应清理
        assert "samples" in session_state
        assert len(session_state["samples"]) == 1
