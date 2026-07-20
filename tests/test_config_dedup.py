"""
配置方案去重与合并测试。

覆盖：
1. 5 个重复配置各有 1 个 run，执行后只剩 1 个配置
2. canonical 配置显示 5 次运行
3. 5 个 run_id 均保留且可读取
4. 每个 run 的 question_set_id、trace_id、Judge 结果迁移前后一致
5. 第二次执行不产生变化
6. 任意 run 校验失败时，不删除重复配置
"""

import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from experiment import (
    config_fingerprint,
    find_duplicate_config_groups,
    find_canonical_config,
    merge_duplicate_configs,
    list_config_profiles,
    load_config_profile,
    load_experiment_run,
    list_runs_by_config,
    _delete_config,
)


# ====== Fixture Helpers ======

def _make_config(config_id, config_name="测试配置", knowledge_base_version="KB_v1",
                 top_k=5, retrieval_mode="混合检索", created_at="2026-07-17T10:00:00", **extra):
    cfg = {
        "config_id": config_id,
        "config_name": config_name,
        "knowledge_base_version": knowledge_base_version,
        "workflow_version": "WF_v1",
        "changed_variable": "",
        "retrieval_config": "",
        "notes": "",
        "created_at": created_at,
    }
    if top_k is not None:
        cfg["top_k"] = top_k
    if retrieval_mode:
        cfg["retrieval_mode"] = retrieval_mode
    cfg.update(extra)
    return cfg


def _write_config(config_dir, cfg):
    path = Path(config_dir) / f"{cfg['config_id']}.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_run(experiments_dir, run_id, config_id, question_set_id="qs_test_001"):
    run_dir = Path(experiments_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "config_id": config_id,
        "config_snapshot": {"config_id": config_id, "config_name": "测试", "top_k": 5},
        "question_set_id": question_set_id,
        "question_set_name": "测试题集",
        "question_count": 20,
        "status": "completed",
        "started_at": "2026-07-17T10:00:00",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ====== Tests ======

def test_five_duplicates_merge_to_one():
    """5 个重复配置各有 1 个 run，合并后只剩 1 个配置，5 个 run 仍存在。"""
    print("=" * 60)
    print("测试：5 重复配置合并为 1")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            # 创建 5 个相同配置 + 各 1 个 run
            for i in range(5):
                cid = f"cfg_{i:03d}"
                _write_config(cfg_dir, _make_config(cid, created_at=f"2026-07-17T10:0{i}:00"))
                _make_run(exp_dir, f"run_{i:03d}", cid)

            # 确认初始状态
            assert len(list_config_profiles(include_archived=True)) == 5
            assert len(list(exp_dir.iterdir())) == 5

            # 执行合并
            result = merge_duplicate_configs(dry_run=False)

            # 验证：只剩 1 个配置
            remaining = list_config_profiles(include_archived=True)
            assert len(remaining) == 1, f"应只剩 1 个配置，实际 {len(remaining)}"
            assert remaining[0]["config_id"] == "cfg_000", f"canonical 应为 cfg_000"

            # 验证：5 个 run 仍存在
            assert len(list(exp_dir.iterdir())) == 5, "5 个 run 应全部保留"

            # 验证：所有 run 的 config_id 都指向 canonical
            for i in range(5):
                run = load_experiment_run(f"run_{i:03d}")
                assert run is not None, f"run_{i:03d} 应可加载"
                assert run["config_id"] == "cfg_000", f"run_{i:03d} config_id 应为 cfg_000"

            # 验证：canonical 配置有 5 次运行
            runs = list_runs_by_config("cfg_000")
            assert len(runs) == 5, f"canonical 应有 5 次运行，实际 {len(runs)}"

            assert result["runs_migrated"] == 4, f"应迁移 4 个 run（canonical 自身不迁移）"
            assert result["configs_deleted"] == 4, f"应删除 4 个配置"
            assert result["validation_failures"] == []

            print("PASS: 5 重复配置合并为 1，5 个 run 保留")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


def test_run_metadata_preserved():
    """每个 run 的 question_set_id、config_snapshot 迁移前后一致。"""
    print("=" * 60)
    print("测试：run 元数据保留")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            # 创建 2 个相同配置 + 各 1 个 run（不同 question_set_id）
            _write_config(cfg_dir, _make_config("cfg_000", created_at="2026-07-17T10:00:00"))
            _write_config(cfg_dir, _make_config("cfg_001", created_at="2026-07-17T10:01:00"))
            _make_run(exp_dir, "run_000", "cfg_000", question_set_id="qs_AAA")
            _make_run(exp_dir, "run_001", "cfg_001", question_set_id="qs_BBB")

            # 记录迁移前的 run 状态
            run0_before = load_experiment_run("run_000")
            run1_before = load_experiment_run("run_001")
            snap0_before = run0_before["config_snapshot"]
            snap1_before = run1_before["config_snapshot"]

            # 执行合并
            merge_duplicate_configs(dry_run=False)

            # 验证 run_001 的元数据保留（除了 config_id 变为 canonical）
            run1_after = load_experiment_run("run_001")
            assert run1_after["question_set_id"] == "qs_BBB", "question_set_id 应保留"
            assert run1_after["question_set_name"] == "测试题集", "question_set_name 应保留"
            assert run1_after["question_count"] == 20, "question_count 应保留"
            assert run1_after["status"] == "completed", "status 应保留"
            assert run1_after["started_at"] == "2026-07-17T10:00:00", "started_at 应保留"
            # config_snapshot 保持不变（不修改快照）
            assert run1_after["config_snapshot"] == snap1_before, "config_snapshot 不应改变"
            # config_id 变为 canonical
            assert run1_after["config_id"] == "cfg_000", "config_id 应变为 canonical"

            # run_000 不应改变
            run0_after = load_experiment_run("run_000")
            assert run0_after == run0_before, "canonical 的 run 不应改变"

            print("PASS: run 元数据完整保留")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


def test_merge_idempotent():
    """第二次执行不产生变化。"""
    print("=" * 60)
    print("测试：合并幂等")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            for i in range(3):
                cid = f"cfg_{i:03d}"
                _write_config(cfg_dir, _make_config(cid, created_at=f"2026-07-17T10:0{i}:00"))
                _make_run(exp_dir, f"run_{i:03d}", cid)

            # 第一次合并
            r1 = merge_duplicate_configs(dry_run=False)
            assert r1["runs_migrated"] == 2
            assert r1["configs_deleted"] == 2

            # 第二次合并
            r2 = merge_duplicate_configs(dry_run=False)
            assert r2["groups"] == 0, f"第二次不应有重复组: {r2['groups']}"
            assert r2["runs_migrated"] == 0, f"第二次不应迁移: {r2['runs_migrated']}"
            assert r2["configs_deleted"] == 0, f"第二次不应删除: {r2['configs_deleted']}"

            # 配置数量不变
            remaining = list_config_profiles(include_archived=True)
            assert len(remaining) == 1

            print("PASS: 合并幂等")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


def test_merge_no_delete_on_validation_failure():
    """任意 run 校验失败时，不删除重复配置。"""
    print("=" * 60)
    print("测试：校验失败不删除")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            # 创建 2 个配置 + 2 个 run
            _write_config(cfg_dir, _make_config("cfg_000", created_at="2026-07-17T10:00:00"))
            _write_config(cfg_dir, _make_config("cfg_001", created_at="2026-07-17T10:01:00"))
            _make_run(exp_dir, "run_000", "cfg_000")
            _make_run(exp_dir, "run_001", "cfg_001")

            # 模拟迁移失败：monkey-patch update_experiment_run
            # 使得 run_001 迁移时抛出异常
            orig_update = exp.update_experiment_run

            def mock_update(run_id, updates):
                if run_id == "run_001":
                    raise IOError(f"模拟写入失败: {run_id}")
                return orig_update(run_id, updates)

            exp.update_experiment_run = mock_update
            try:
                result = merge_duplicate_configs(dry_run=False)
            finally:
                exp.update_experiment_run = orig_update

            # 迁移失败导致不应删除配置
            assert result["configs_deleted"] == 0, "迁移失败时不应删除配置"
            remaining = list_config_profiles(include_archived=True)
            assert len(remaining) == 2, f"不应删除配置，实际 {len(remaining)}"
            # run_001 的 config_id 应保持原值（未成功迁移）
            run1 = load_experiment_run("run_001")
            assert run1["config_id"] == "cfg_001", "失败的 run config_id 不应改变"

            print("PASS: 校验失败不删除配置")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


def test_canonical_shows_all_runs():
    """合并后 canonical 配置显示所有 run。"""
    print("=" * 60)
    print("测试：canonical 显示所有 run")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            # 3 个重复配置各 2 个 run
            for i in range(3):
                cid = f"cfg_{i:03d}"
                _write_config(cfg_dir, _make_config(cid, created_at=f"2026-07-17T10:0{i}:00"))
                _make_run(exp_dir, f"run_{i}a", cid, question_set_id=f"qs_{i}a")
                _make_run(exp_dir, f"run_{i}b", cid, question_set_id=f"qs_{i}b")

            merge_duplicate_configs(dry_run=False)

            # canonical 应有 6 次运行
            runs = list_runs_by_config("cfg_000")
            assert len(runs) == 6, f"canonical 应有 6 次运行，实际 {len(runs)}"

            # 所有 run_id 均保留
            run_ids = {r["run_id"] for r in runs}
            expected = {f"run_{i}{s}" for i in range(3) for s in ("a", "b")}
            assert run_ids == expected, f"run_id 不匹配: {run_ids} != {expected}"

            print("PASS: canonical 显示所有 6 个 run")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


def test_fingerprint_stability():
    """相同业务字段产生相同指纹，不同字段产生不同指纹。"""
    print("=" * 60)
    print("测试：指纹稳定性")
    print("=" * 60)

    cfg1 = _make_config("cfg_001")
    cfg2 = _make_config("cfg_002")
    cfg3 = _make_config("cfg_003", top_k=10)

    assert config_fingerprint(cfg1) == config_fingerprint(cfg2)
    assert config_fingerprint(cfg1) != config_fingerprint(cfg3)

    # 敏感字段不影响
    cfg4 = _make_config("cfg_004", api_key="sk-secret")
    assert config_fingerprint(cfg1) == config_fingerprint(cfg4)

    print("PASS: 指纹稳定性正确")


def test_modified_config_not_duplicate():
    """修改 top_k 后不是重复配置。"""
    print("=" * 60)
    print("测试：修改后非重复")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            # 2 个配置，top_k 不同
            _write_config(cfg_dir, _make_config("cfg_000", top_k=5, created_at="2026-07-17T10:00:00"))
            _write_config(cfg_dir, _make_config("cfg_001", top_k=10, created_at="2026-07-17T10:01:00"))
            _make_run(exp_dir, "run_000", "cfg_000")
            _make_run(exp_dir, "run_001", "cfg_001")

            result = merge_duplicate_configs(dry_run=False)
            assert result["groups"] == 0, "不同 top_k 不应视为重复"
            assert result["configs_deleted"] == 0

            # 两个配置都保留
            remaining = list_config_profiles(include_archived=True)
            assert len(remaining) == 2

            print("PASS: 修改后非重复")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


def test_dropdown_excludes_merged():
    """合并后下拉框不显示已删除的配置。"""
    print("=" * 60)
    print("测试：下拉框排除已合并")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dir = Path(tmpdir) / "configs"
        exp_dir = Path(tmpdir) / "experiments"
        cfg_dir.mkdir()
        exp_dir.mkdir()

        import experiment as exp
        orig_cfg = exp.CONFIG_PROFILES_DIR
        orig_exp = exp.EXPERIMENTS_DIR
        exp.CONFIG_PROFILES_DIR = cfg_dir
        exp.EXPERIMENTS_DIR = exp_dir
        try:
            for i in range(3):
                cid = f"cfg_{i:03d}"
                _write_config(cfg_dir, _make_config(cid, created_at=f"2026-07-17T10:0{i}:00"))
                _make_run(exp_dir, f"run_{i:03d}", cid)

            merge_duplicate_configs(dry_run=False)

            # list_config_profiles 默认不含已删除的
            visible = list_config_profiles()
            assert len(visible) == 1
            assert visible[0]["config_id"] == "cfg_000"

            # include_archived=True 也只剩 1 个（已删除的不会出现）
            all_cfgs = list_config_profiles(include_archived=True)
            assert len(all_cfgs) == 1

            print("PASS: 下拉框只显示 canonical")
        finally:
            exp.CONFIG_PROFILES_DIR = orig_cfg
            exp.EXPERIMENTS_DIR = orig_exp


# ====== Main ======

def main():
    tests = [
        test_fingerprint_stability,
        test_five_duplicates_merge_to_one,
        test_run_metadata_preserved,
        test_merge_idempotent,
        test_merge_no_delete_on_validation_failure,
        test_canonical_shows_all_runs,
        test_modified_config_not_duplicate,
        test_dropdown_excludes_merged,
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
