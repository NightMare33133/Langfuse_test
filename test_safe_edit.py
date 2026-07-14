"""
安全编辑测试。

测试内容：
1. 最少必填字段可创建配置和运行
2. 旧配置/旧 run 缺少新字段仍能加载和展示
3. 编辑配置方案后 config_id 不变
4. 编辑某个 run 的 snapshot 不影响其他 run 或配置方案
5. 尝试修改核心字段会被拒绝/忽略
6. 修改前快照/审计记录存在

不调用真实 API。
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from experiment import (
    create_config_profile, create_experiment_run,
    load_config_profile, load_experiment_run,
    update_config_profile_safe, update_run_snapshot,
    CONFIG_CORE_FIELDS, RUN_CORE_FIELDS, CONFIG_EDITABLE_FIELDS,
)


def test_minimal_required_fields():
    """最少必填字段可创建配置和运行。"""
    print("=" * 60)
    print("测试最少必填字段创建")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_dir = tmpdir / "config_profiles"
        exp_dir = tmpdir / "experiments"

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir), \
             patch("experiment.EXPERIMENTS_DIR", exp_dir):
            # 只填必填字段
            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                workflow_version="wf_v1",
            )

            assert config["config_id"], "应有 config_id"
            assert config["config_name"] == "测试配置"
            assert config["knowledge_base_version"] == "v1"
            assert config["workflow_version"] == "wf_v1"
            assert config["created_at"], "应有 created_at"
            # 可选字段不应自动填充
            assert "top_k" not in config, "top_k 不应自动填充"
            assert "embedding_model" not in config, "embedding_model 不应自动填充"
            print("[OK] 最少字段创建成功，可选字段未自动填充")

            # 创建运行
            run = create_experiment_run(config["config_id"], "测试题集", 5)
            manifest = load_experiment_run(run["run_id"])
            assert manifest["run_id"]
            assert manifest["config_id"] == config["config_id"]
            assert manifest["config_snapshot"]["config_name"] == "测试配置"
            print("[OK] 运行创建成功，快照包含配置信息")

    print()


def test_old_config_compatibility():
    """旧配置缺少新字段仍能加载和展示。"""
    print("=" * 60)
    print("测试旧配置兼容性")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_dir = tmpdir / "config_profiles"
        config_dir.mkdir()

        # 模拟旧配置（只有老字段）
        old_config = {
            "config_id": "cfg_old_test",
            "config_name": "旧配置",
            "knowledge_base_version": "old_kb",
            "workflow_version": "",
            "changed_variable": "",
            "retrieval_config": "",
            "notes": "",
            "created_at": "2026-07-08T00:00:00",
        }
        (config_dir / "cfg_old_test.json").write_text(
            json.dumps(old_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            loaded = load_config_profile("cfg_old_test")
            assert loaded is not None
            assert loaded["config_name"] == "旧配置"
            # 新字段不存在但不报错
            assert loaded.get("retrieval_mode") is None
            assert loaded.get("top_k") is None
            assert loaded.get("embedding_model") is None
            print("[OK] 旧配置可加载，缺失新字段不报错")

    print()


def test_edit_preserves_config_id():
    """编辑配置方案后 config_id 不变。"""
    print("=" * 60)
    print("测试编辑保留 config_id")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_dir = tmpdir / "config_profiles"

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            config = create_config_profile("测试", "v1", "wf1")
            original_id = config["config_id"]

            # 编辑
            updated = update_config_profile_safe(original_id, {
                "config_name": "修改后的名称",
                "top_k": 5,
                "retrieval_mode": "hybrid",
            }, edit_note="补充配置信息")

            assert updated["config_id"] == original_id, "config_id 不应改变"
            assert updated["config_name"] == "修改后的名称"
            assert updated["top_k"] == 5
            assert updated["retrieval_mode"] == "hybrid"
            assert updated["updated_at"], "应有 updated_at"
            assert updated["edit_note"] == "补充配置信息"
            print("[OK] config_id 不变，新字段正确保存")

    print()


def test_core_fields_rejected():
    """尝试修改核心字段会被拒绝/忽略。"""
    print("=" * 60)
    print("测试核心字段保护")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_dir = tmpdir / "config_profiles"

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            config = create_config_profile("测试", "v1", "wf1")
            original_id = config["config_id"]
            original_created = config["created_at"]

            # 尝试修改核心字段
            updated = update_config_profile_safe(original_id, {
                "config_id": "hacked_id",  # 应被忽略
                "created_at": "2099-01-01",  # 应被忽略
                "config_name": "合法修改",  # 应生效
            })

            assert updated["config_id"] == original_id, "config_id 不应被修改"
            assert updated["created_at"] == original_created, "created_at 不应被修改"
            assert updated["config_name"] == "合法修改", "可编辑字段应生效"
            print("[OK] 核心字段修改被拒绝，可编辑字段生效")

    print()


def test_run_snapshot_edit_isolation():
    """编辑某个 run 的 snapshot 不影响其他 run 或配置方案。"""
    print("=" * 60)
    print("测试运行快照编辑隔离")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_dir = tmpdir / "config_profiles"
        exp_dir = tmpdir / "experiments"

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir), \
             patch("experiment.EXPERIMENTS_DIR", exp_dir):
            config = create_config_profile("测试", "v1", "wf1")
            config_id = config["config_id"]

            # 创建两个运行
            run1 = create_experiment_run(config_id, "题集A", 5)
            run2 = create_experiment_run(config_id, "题集B", 10)

            run1_id = run1["run_id"]
            run2_id = run2["run_id"]

            # 编辑 run1 的 snapshot
            update_run_snapshot(run1_id, {
                "top_k": 10,
                "rerank_model": "bge-reranker",
                "config_id": "should_be_ignored",  # 核心字段应被忽略
            }, edit_note="补录 Rerank 配置")

            # 验证 run1 snapshot 已更新
            m1 = load_experiment_run(run1_id)
            snap1 = m1["config_snapshot"]
            assert snap1["top_k"] == 10
            assert snap1["rerank_model"] == "bge-reranker"
            assert snap1["config_id"] == config_id, "核心字段不应被修改"
            assert snap1["snapshot_updated_at"]
            assert snap1["snapshot_edit_note"] == "补录 Rerank 配置"

            # 验证 run2 snapshot 未受影响
            m2 = load_experiment_run(run2_id)
            snap2 = m2["config_snapshot"]
            assert snap2.get("top_k") is None, "run2 的 snapshot 不应被修改"
            assert snap2.get("rerank_model") is None

            # 验证配置方案未受影响
            cfg = load_config_profile(config_id)
            assert cfg.get("top_k") is None, "配置方案不应被 run 的 snapshot 修改影响"

            print("[OK] run1 snapshot 已更新，run2 和配置方案未受影响")

    print()


def test_snapshot_edit_history():
    """修改前快照/审计记录存在。"""
    print("=" * 60)
    print("测试修改前快照审计")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_dir = tmpdir / "config_profiles"
        exp_dir = tmpdir / "experiments"

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir), \
             patch("experiment.EXPERIMENTS_DIR", exp_dir):
            config = create_config_profile("测试", "v1", "wf1")
            run = create_experiment_run(config["config_id"], "题集", 5)
            run_id = run["run_id"]

            # 第一次修改
            update_run_snapshot(run_id, {"top_k": 5}, edit_note="第一次修改")

            # 第二次修改
            update_run_snapshot(run_id, {"top_k": 10, "rerank_model": "reranker"}, edit_note="第二次修改")

            manifest = load_experiment_run(run_id)
            history = manifest.get("snapshot_edit_history", [])
            assert len(history) == 2, f"应有 2 条历史记录，实际 {len(history)}"

            # 第一条记录应保存修改前的快照（没有 top_k）
            assert history[0]["edit_note"] == "第一次修改"
            assert "top_k" not in history[0]["before"], "第一次修改前的快照不应有 top_k"

            # 第二条记录应保存修改前的快照（top_k=5）
            assert history[1]["edit_note"] == "第二次修改"
            assert history[1]["before"]["top_k"] == 5, "第二次修改前的快照应有 top_k=5"

            # 当前值应为最新
            snap = manifest["config_snapshot"]
            assert snap["top_k"] == 10
            assert snap["rerank_model"] == "reranker"

            print("[OK] 修改前快照正确保存，审计记录完整")

    print()


def test_optional_fields_display():
    """缺失可选字段显示为'未记录'而不是报错。"""
    print("=" * 60)
    print("测试缺失字段显示")
    print("=" * 60)

    def _display_val(val):
        if val is None or str(val).strip() == "":
            return "未记录"
        return str(val)

    assert _display_val(None) == "未记录"
    assert _display_val("") == "未记录"
    assert _display_val("  ") == "未记录"
    assert _display_val("有值") == "有值"
    assert _display_val(0) == "0"
    assert _display_val(5) == "5"
    print("[OK] 缺失字段显示为'未记录'，有值字段正常显示")

    print()


def main():
    print("=" * 60)
    print("安全编辑测试")
    print("=" * 60)
    print()

    test_minimal_required_fields()
    test_old_config_compatibility()
    test_edit_preserves_config_id()
    test_core_fields_rejected()
    test_run_snapshot_edit_isolation()
    test_snapshot_edit_history()
    test_optional_fields_display()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
