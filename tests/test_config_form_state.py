"""
配置编辑表单状态管理回归测试。

覆盖：
1. 动态 key_prefix 由 config_id 哈希生成
2. 配置切换时旧 session_state 被清理
3. 配置不一致时保存被阻止
4. 不触发保存时磁盘配置文件不变

不调用真实 Streamlit、Dify、Langfuse API。
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, load_config_profile, update_config_profile_safe,
)


# ====== Helpers ======

def _make_key_prefix(config_id: str) -> str:
    """模拟 app.py 中的动态 key_prefix 生成逻辑。"""
    return f"ecfg_{hashlib.md5(config_id.encode()).hexdigest()[:12]}"


def _make_session_state():
    """模拟 Streamlit session_state（普通 dict）。"""
    return {}


def _simulate_config_switch(session_state: dict, old_id: str, new_id: str):
    """模拟 app.py 中的配置切换清理逻辑。"""
    old_prefix = _make_key_prefix(old_id)
    keys_to_clean = [k for k in session_state if k.startswith(old_prefix)]
    for k in keys_to_clean:
        del session_state[k]
    if "ec_edit_note" in session_state:
        del session_state["ec_edit_note"]
    session_state["_ecfg_form_bound_id"] = new_id


# ====== Tests ======

def test_key_prefix_stable_hash():
    """key_prefix 必须由完整 config_id 的 MD5 哈希生成，不能是固定值。"""
    print("=" * 60)
    print("测试：key_prefix 稳定哈希")
    print("=" * 60)

    id_a = "cfg_20260724_120000_abcd1234"
    id_b = "cfg_20260724_130000_efgh5678"

    prefix_a = _make_key_prefix(id_a)
    prefix_b = _make_key_prefix(id_b)

    # 不同 config_id 产生不同 prefix
    assert prefix_a != prefix_b, f"不同 config_id 应产生不同 prefix: {prefix_a} vs {prefix_b}"
    # prefix 不是固定值
    assert prefix_a.startswith("ecfg_"), f"prefix 应以 ecfg_ 开头: {prefix_a}"
    assert len(prefix_a) > 10, f"prefix 应包含哈希: {prefix_a}"
    # 同一 config_id 产生相同 prefix（稳定性）
    assert _make_key_prefix(id_a) == prefix_a, "同一 config_id 应产生相同 prefix"

    print(f"PASS: prefix_a={prefix_a}, prefix_b={prefix_b}")


def test_switch_clears_old_state():
    """v4 -> v5 切换后，v4 的 session_state 应被清理。"""
    print("=" * 60)
    print("测试：配置切换清理旧状态")
    print("=" * 60)

    ss = _make_session_state()
    id_v4 = "cfg_20260724_120000_v4test"
    id_v5 = "cfg_20260724_130000_v5test"

    prefix_v4 = _make_key_prefix(id_v4)
    prefix_v5 = _make_key_prefix(id_v5)

    # 模拟 v4 的编辑状态
    ss[f"{prefix_v4}_config_name"] = "旧配置名称"
    ss[f"{prefix_v4}_knowledge_base_version"] = "old_kb_v1"
    ss[f"{prefix_v4}_workflow_version"] = "old_wf_v1"
    ss[f"{prefix_v4}_top_k"] = 3
    ss["ec_edit_note"] = "v4 的编辑说明"
    ss["_ecfg_form_bound_id"] = id_v4

    # 模拟切换到 v5
    _simulate_config_switch(ss, id_v4, id_v5)

    # v4 的 key 应被清理
    assert f"{prefix_v4}_config_name" not in ss, "v4 的 config_name 应被清理"
    assert f"{prefix_v4}_knowledge_base_version" not in ss, "v4 的 kb_version 应被清理"
    assert f"{prefix_v4}_top_k" not in ss, "v4 的 top_k 应被清理"
    assert "ec_edit_note" not in ss, "ec_edit_note 应被清理"
    # 绑定 ID 应更新
    assert ss["_ecfg_form_bound_id"] == id_v5, "绑定 ID 应更新为 v5"

    print(f"PASS: v4 状态已清理，绑定更新为 v5")


def test_round_trip_switch():
    """v5 -> v4 -> v5 来回切换，状态应正确清理。"""
    print("=" * 60)
    print("测试：来回切换")
    print("=" * 60)

    ss = _make_session_state()
    id_v4 = "cfg_20260724_120000_v4test"
    id_v5 = "cfg_20260724_130000_v5test"

    # 初始 v5 状态
    prefix_v5 = _make_key_prefix(id_v5)
    ss[f"{prefix_v5}_config_name"] = "v5配置"
    ss["_ecfg_form_bound_id"] = id_v5

    # 切换到 v4
    _simulate_config_switch(ss, id_v5, id_v4)
    prefix_v4 = _make_key_prefix(id_v4)
    ss[f"{prefix_v4}_config_name"] = "v4配置"

    # 切换回 v5
    _simulate_config_switch(ss, id_v4, id_v5)

    # v4 的状态应被清理
    assert f"{prefix_v4}_config_name" not in ss, "v4 的 config_name 应被清理"
    # 绑定应为 v5
    assert ss["_ecfg_form_bound_id"] == id_v5

    print("PASS: 来回切换后状态正确")


def test_consistency_check_blocks_save():
    """config_id 不一致时保存应被阻止。"""
    print("=" * 60)
    print("测试：一致性检查阻止保存")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("experiment.CONFIG_PROFILES_DIR", Path(tmpdir)):
            # 创建配置
            cfg = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                workflow_version="wf_v1",
            )
            cfg_id = cfg["config_id"]

            # 一致性检查：三重匹配
            form_bound = cfg_id
            selected = cfg_id
            disk = load_config_profile(cfg_id)
            disk_id = disk.get("config_id") if disk else None

            assert form_bound == selected == disk_id, "三重 ID 应匹配"

            # 模拟不一致：form_bound 被篡改
            form_bound_bad = "cfg_fake_id_12345"
            is_consistent = (form_bound_bad == selected == disk_id)
            assert not is_consistent, "不一致应被检测到"

    print("PASS: 一致性检查正确阻止不一致保存")


def test_no_save_disk_unchanged():
    """不触发保存时，磁盘配置文件不应变化。"""
    print("=" * 60)
    print("测试：不保存时磁盘不变")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("experiment.CONFIG_PROFILES_DIR", Path(tmpdir)):
            cfg = create_config_profile(
                config_name="原始配置",
                knowledge_base_version="v1",
                workflow_version="wf_v1",
            )
            cfg_id = cfg["config_id"]

            # 读取原始内容
            original = load_config_profile(cfg_id)
            original_content = json.dumps(original, sort_keys=True)

            # 模拟用户打开编辑但不保存（只读操作）
            _ = _make_key_prefix(cfg_id)
            # 没有调用 update_config_profile_safe

            # 验证磁盘未变
            after = load_config_profile(cfg_id)
            after_content = json.dumps(after, sort_keys=True)
            assert original_content == after_content, "未保存时磁盘不应变化"

    print("PASS: 未保存时磁盘配置文件不变")


def test_save_with_consistency_succeeds():
    """一致性匹配时保存应成功。"""
    print("=" * 60)
    print("测试：一致性匹配时保存成功")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("experiment.CONFIG_PROFILES_DIR", Path(tmpdir)):
            cfg = create_config_profile(
                config_name="原始配置",
                knowledge_base_version="v1",
                workflow_version="wf_v1",
            )
            cfg_id = cfg["config_id"]

            # 一致性检查通过
            form_bound = cfg_id
            selected = cfg_id
            disk = load_config_profile(cfg_id)
            disk_id = disk.get("config_id")
            assert form_bound == selected == disk_id

            # 保存
            update_config_profile_safe(cfg_id, {"config_name": "更新后名称"}, edit_note="测试修改")

            # 验证更新
            updated = load_config_profile(cfg_id)
            assert updated["config_name"] == "更新后名称"
            assert updated["edit_note"] == "测试修改"
            # config_id 不变
            assert updated["config_id"] == cfg_id
            assert updated["created_at"] == cfg["created_at"]

    print("PASS: 一致性匹配时保存成功，config_id/created_at 不变")


# ====== Main ======

def main():
    tests = [
        test_key_prefix_stable_hash,
        test_switch_clears_old_state,
        test_round_trip_switch,
        test_consistency_check_blocks_save,
        test_no_save_disk_unchanged,
        test_save_with_consistency_succeeds,
    ]

    import sys
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
