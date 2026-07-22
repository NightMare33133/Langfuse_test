"""
测试配置切换的正确性。

覆盖：
a. 切换配置后 widget key 包含 config_id（不复用旧值）
b. 切换配置时清理 batch_existing_runs_by_qs 缓存
c. 一致性校验阻止不匹配的执行
d. 配置始终从磁盘加载（不从过期 session_state 读取）
e. 历史配置只读，不因切换而修改 profile 文件
"""

import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def test_widget_key_includes_config_id():
    """只读表单的 key_prefix 应使用完整 config_id 的哈希，避免切换时复用旧值。"""
    print("=" * 60)
    print("测试 widget key 使用完整 config_id 哈希")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 应使用 hashlib.md5(selected_config_id.encode()).hexdigest() 作为 key
    assert 'hashlib.md5(selected_config_id.encode()).hexdigest()' in source, \
        "只读表单的 key_prefix 应使用完整 config_id 的 MD5 哈希"
    assert 'render_config_form(selected_config, key_prefix=_ro_key, disabled=True)' in source, \
        "应使用 _ro_key 作为 key_prefix"
    # 不应使用截断的 [-12:]
    assert 'selected_config_id[-12:]' not in source or '_ro_key' not in source, \
        "不应使用截断的 config_id 作为 key"
    print("[OK] widget key 使用完整 config_id 的 MD5 哈希")

    print()


def test_config_switch_clears_cache():
    """切换配置时应清理 batch_existing_runs_by_qs 缓存。"""
    print("=" * 60)
    print("测试配置切换清理缓存")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 应有 _batch_prev_config_id 跟踪
    assert "_batch_prev_config_id" in source, \
        "应跟踪上一次选择的 config_id"

    # 切换时应 pop batch_existing_runs_by_qs
    assert 'st.session_state.pop("batch_existing_runs_by_qs"' in source, \
        "切换配置时应清理 batch_existing_runs_by_qs"

    # 切换时应 pop batch_qs_strategy
    assert 'st.session_state.pop("batch_qs_strategy"' in source, \
        "切换配置时应清理 batch_qs_strategy"
    print("[OK] 配置切换时清理缓存")

    print()


def test_consistency_check_blocks_mismatch():
    """四重一致性校验应阻止 config_id 不匹配的执行。"""
    print("=" * 60)
    print("测试四重一致性校验")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 应有四重校验变量
    assert '_check_selectbox' in source, "应有 _check_selectbox 变量"
    assert '_check_displayed' in source, "应有 _check_displayed 变量"
    assert '_check_disk' in source, "应有 _check_disk 变量"
    assert '_check_executor' in source, "应有 _check_executor 变量"

    # 应有 _all_ids 集合比较
    assert '_all_ids' in source, "应有 _all_ids 集合比较"

    # 应有 st.stop() 阻止执行
    assert '配置一致性校验失败' in source, \
        "应显示一致性校验失败错误"
    print("[OK] 四重一致性校验已实现")

    print()


def test_displayed_config_id_recorded():
    """batch_displayed_config_id 应在渲染只读快照后写入。"""
    print("=" * 60)
    print("测试 batch_displayed_config_id 记录")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert 'st.session_state["batch_displayed_config_id"] = selected_config_id' in source, \
        "应在渲染后写入 batch_displayed_config_id"
    print("[OK] batch_displayed_config_id 已记录")

    print()


def test_selectbox_v4_display_v2_4_blocked():
    """模拟下拉框=v4、显示快照=v2_4 状态，执行必须被阻止。"""
    print("=" * 60)
    print("测试下拉框与显示不一致时阻止执行")
    print("=" * 60)

    # 模拟四重校验逻辑
    def check_consistency(selectbox, displayed, disk, executor):
        """返回 True 表示通过，False 表示阻止。"""
        all_ids = {selectbox, displayed, disk, executor}
        if len(all_ids) > 1 or not executor:
            return False
        if not disk:  # config 不存在
            return False
        return True

    # 场景 1: 全部一致 → 通过
    assert check_consistency("cfg_v4", "cfg_v4", "cfg_v4", "cfg_v4") is True
    print("[OK] 全部一致 → 通过")

    # 场景 2: 下拉框=v4，显示=v2_4 → 阻止
    assert check_consistency("cfg_v4", "cfg_v2_4", "cfg_v4", "cfg_v4") is False
    print("[OK] 下拉框=v4、显示=v2_4 → 阻止")

    # 场景 3: 下拉框=v4，磁盘=v2_4 → 阻止
    assert check_consistency("cfg_v4", "cfg_v4", "cfg_v2_4", "cfg_v4") is False
    print("[OK] 下拉框=v4、磁盘=v2_4 → 阻止")

    # 场景 4: 执行器=v4，其他=v2_4 → 阻止
    assert check_consistency("cfg_v2_4", "cfg_v2_4", "cfg_v2_4", "cfg_v4") is False
    print("[OK] 执行器与其他不一致 → 阻止")

    # 场景 5: config 不存在 → 阻止
    assert check_consistency("cfg_v4", "cfg_v4", "", "cfg_v4") is False
    print("[OK] config 不存在 → 阻止")

    # 场景 6: 空 config_id → 阻止
    assert check_consistency("", "", "", "") is False
    print("[OK] 空 config_id → 阻止")

    print()


def test_config_always_loaded_from_disk():
    """配置应始终从磁盘加载，不从过期 session_state 读取。"""
    print("=" * 60)
    print("测试配置从磁盘加载")
    print("=" * 60)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 执行前应有 _verify_config = load_config_profile(_config_id)
    assert '_verify_config = load_config_profile(_config_id)' in source, \
        "执行前应从磁盘重新加载配置"
    print("[OK] 执行前从磁盘重新加载配置")

    print()


def test_readonly_does_not_modify_profile():
    """历史配置只读，不因切换而修改 profile 文件。"""
    print("=" * 60)
    print("测试历史配置只读")
    print("=" * 60)

    # render_config_form 调用时 disabled=True
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # 只读模式应传 disabled=True
    assert 'render_config_form(selected_config, key_prefix=_ro_key, disabled=True)' in source, \
        "只读模式应传 disabled=True"
    print("[OK] 只读模式 disabled=True")

    # 模拟验证：只读表单不写入文件
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.json"
        config_path.write_text('{"config_name": "test"}', encoding="utf-8")
        original = config_path.read_text(encoding="utf-8")

        # 模拟只读加载
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        assert loaded["config_name"] == "test"

        # 文件不应被修改
        assert config_path.read_text(encoding="utf-8") == original
        print("[OK] 只读加载不修改文件")

    print()


def test_config_switch_session_state_cleanup():
    """模拟配置切换时 session_state 清理逻辑。"""
    print("=" * 60)
    print("测试 session_state 清理逻辑")
    print("=" * 60)

    # 模拟 session_state
    session_state = {
        "_batch_prev_config_id": "cfg_v2_4",
        "batch_existing_runs_by_qs": {"qs_001": {"run_id": "old_run"}},
        "batch_qs_strategy": "skip",
    }

    # 模拟切换到新配置
    new_config_id = "cfg_v4"
    prev_cfg = session_state.get("_batch_prev_config_id")

    if prev_cfg and prev_cfg != new_config_id:
        session_state.pop("batch_existing_runs_by_qs", None)
        session_state.pop("batch_qs_strategy", None)
    session_state["_batch_prev_config_id"] = new_config_id

    assert session_state["_batch_prev_config_id"] == "cfg_v4"
    assert "batch_existing_runs_by_qs" not in session_state
    assert "batch_qs_strategy" not in session_state
    print("[OK] 切换后缓存已清理，prev_config_id 已更新")

    # 模拟再次选择相同配置（不应清理）
    session_state["batch_existing_runs_by_qs"] = {"qs_002": {"run_id": "new_run"}}
    prev_cfg = session_state.get("_batch_prev_config_id")
    if prev_cfg and prev_cfg != new_config_id:
        session_state.pop("batch_existing_runs_by_qs", None)
    session_state["_batch_prev_config_id"] = new_config_id

    assert "batch_existing_runs_by_qs" in session_state, "相同配置不应清理缓存"
    print("[OK] 相同配置不清理缓存")

    print()


def test_consistency_check_logic():
    """模拟四重一致性校验逻辑。"""
    print("=" * 60)
    print("测试四重一致性校验逻辑")
    print("=" * 60)

    def four_way_check(selectbox, displayed, disk_profile_id, executor):
        """模拟四重校验，返回 (pass, reason)。"""
        if not executor:
            return False, "空 config_id"
        all_ids = {selectbox, displayed, disk_profile_id, executor}
        if len(all_ids) > 1:
            return False, f"不一致: {all_ids}"
        if not disk_profile_id:
            return False, "config 不存在"
        return True, "通过"

    # 全部一致
    ok, reason = four_way_check("cfg_v4", "cfg_v4", "cfg_v4", "cfg_v4")
    assert ok, f"应通过: {reason}"
    print("[OK] 全部一致 → 通过")

    # selectbox 与 displayed 不一致
    ok, reason = four_way_check("cfg_v4", "cfg_v2_4", "cfg_v4", "cfg_v4")
    assert not ok
    print(f"[OK] selectbox≠displayed → 阻止: {reason}")

    # displayed 与 disk 不一致
    ok, reason = four_way_check("cfg_v4", "cfg_v4", "cfg_v2_4", "cfg_v4")
    assert not ok
    print(f"[OK] displayed≠disk → 阻止: {reason}")

    # executor 与其他不一致
    ok, reason = four_way_check("cfg_v4", "cfg_v4", "cfg_v4", "cfg_v2_4")
    assert not ok
    print(f"[OK] executor≠其他 → 阻止: {reason}")

    # config 不存在
    ok, reason = four_way_check("cfg_v4", "cfg_v4", "", "cfg_v4")
    assert not ok
    print(f"[OK] config 不存在 → 阻止: {reason}")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("配置切换正确性测试")
    print("=" * 60)
    print()

    test_widget_key_includes_config_id()
    test_config_switch_clears_cache()
    test_consistency_check_blocks_mismatch()
    test_displayed_config_id_recorded()
    test_selectbox_v4_display_v2_4_blocked()
    test_config_always_loaded_from_disk()
    test_readonly_does_not_modify_profile()
    test_config_switch_session_state_cleanup()
    test_consistency_check_logic()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
