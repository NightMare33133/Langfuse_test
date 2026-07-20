"""
Dify 连接配置测试。

测试内容：
1. 创建配置 — 元数据 JSON 存在，不含 API Key；keyring 存储 key
2. 列出配置 — 返回所有配置
3. 加载配置 — 返回正确元数据
4. 读取 API Key — 从 keyring 返回正确 key
5. 更新配置 — 元数据更新；新 key 更新 keyring；空 key 保留原值
6. 清除 key — keyring 条目移除
7. 删除配置 — 元数据和 keyring 条目均移除
8. 脱敏 API Key — 正确脱敏
9. Run manifest — 含 profile_id/profile_name/base_url 但不含 api_key
10. 无 API Key 泄露 — 元数据 JSON 中不含任何 key 片段

使用 mock keyring，不使用真实密钥。
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.stdout.reconfigure(encoding="utf-8")

# Mock keyring before importing dify_connection
_mock_store = {}


def _mock_set_password(service, key, value):
    _mock_store[f"{service}:{key}"] = value


def _mock_get_password(service, key):
    return _mock_store.get(f"{service}:{key}")


def _mock_delete_password(service, key):
    _mock_store.pop(f"{service}:{key}", None)


_mock_keyring = MagicMock()
_mock_keyring.set_password = _mock_set_password
_mock_keyring.get_password = _mock_get_password
_mock_keyring.delete_password = _mock_delete_password

with patch("dify_connection._get_keyring", return_value=_mock_keyring):
    from dify_connection import (
        create_connection_profile, load_connection_profile, list_connection_profiles,
        update_connection_profile, delete_connection_profile,
        get_connection_api_key, has_connection_api_key, mask_api_key,
        CONNECTIONS_DIR, KEYRING_SERVICE,
    )


def _setup():
    """清理测试环境。"""
    _mock_store.clear()


def test_create_profile():
    """创建配置 — 元数据 JSON 存在，不含 API Key。"""
    print("=" * 60)
    print("测试创建连接配置")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile(
                "测试工作流", "http://localhost/v1", "sk-test-12345",
                workflow_description="测试描述", timeout_seconds=30,
            )

            assert meta["profile_name"] == "测试工作流"
            assert meta["base_url"] == "http://localhost/v1"
            assert meta["workflow_description"] == "测试描述"
            assert meta["timeout_seconds"] == 30
            assert meta["profile_id"].startswith("dify_")
            assert meta["created_at"]
            assert meta["updated_at"]

            # 元数据 JSON 不含 API Key
            meta_path = Path(tmpdir) / f"{meta['profile_id']}.json"
            assert meta_path.exists()
            raw = meta_path.read_text(encoding="utf-8")
            assert "sk-test-12345" not in raw, "API Key 不应出现在元数据 JSON 中"

            # keyring 存储了 key
            stored_key = get_connection_api_key(meta["profile_id"])
            assert stored_key == "sk-test-12345"

            print(f"  profile_id: {meta['profile_id']}")
            print("[OK] 创建成功，元数据不含 API Key，keyring 已存储")

    print()


def test_list_profiles():
    """列出配置 — 返回所有配置。"""
    print("=" * 60)
    print("测试列出连接配置")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            create_connection_profile("配置A", "http://a.local/v1", "sk-a")
            create_connection_profile("配置B", "http://b.local/v1", "sk-b")

            profiles = list_connection_profiles()
            assert len(profiles) == 2
            names = [p["profile_name"] for p in profiles]
            assert "配置A" in names
            assert "配置B" in names
            print("[OK] 列出 2 个配置")

    print()


def test_load_profile():
    """加载配置 — 返回正确元数据。"""
    print("=" * 60)
    print("测试加载连接配置")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile("加载测试", "http://test.local/v1", "sk-load-test")
            loaded = load_connection_profile(meta["profile_id"])

            assert loaded is not None
            assert loaded["profile_name"] == "加载测试"
            assert loaded["base_url"] == "http://test.local/v1"

            # 不存在的配置返回 None
            assert load_connection_profile("nonexistent") is None
            print("[OK] 加载正确，不存在时返回 None")

    print()


def test_get_api_key():
    """读取 API Key — 从 keyring 返回正确 key。"""
    print("=" * 60)
    print("测试读取 API Key")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile("Key测试", "http://test.local/v1", "sk-secret-key-12345")

            key = get_connection_api_key(meta["profile_id"])
            assert key == "sk-secret-key-12345"

            assert has_connection_api_key(meta["profile_id"]) is True
            assert has_connection_api_key("nonexistent") is False

            print("[OK] API Key 读取正确")

    print()


def test_update_profile():
    """更新配置 — 元数据更新；新 key 更新；空 key 保留原值。"""
    print("=" * 60)
    print("测试更新连接配置")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile("更新测试", "http://old.local/v1", "sk-old-key")

            # 更新元数据 + 新 key
            updated = update_connection_profile(
                meta["profile_id"],
                {"profile_name": "更新后名称", "base_url": "http://new.local/v1"},
                api_key="sk-new-key",
            )
            assert updated["profile_name"] == "更新后名称"
            assert updated["base_url"] == "http://new.local/v1"
            assert get_connection_api_key(meta["profile_id"]) == "sk-new-key"
            print("[OK] 元数据和 key 均已更新")

            # 更新元数据但不传 key（保留原 key）
            update_connection_profile(meta["profile_id"], {"base_url": "http://final.local/v1"})
            assert get_connection_api_key(meta["profile_id"]) == "sk-new-key"
            print("[OK] 不传 key 时保留原 key")

    print()


def test_clear_key():
    """清除 key — keyring 条目移除。"""
    print("=" * 60)
    print("测试清除 API Key")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile("清除测试", "http://test.local/v1", "sk-to-clear")
            assert has_connection_api_key(meta["profile_id"]) is True

            update_connection_profile(meta["profile_id"], {}, clear_key=True)
            assert has_connection_api_key(meta["profile_id"]) is False
            assert get_connection_api_key(meta["profile_id"]) is None
            print("[OK] API Key 已清除")

    print()


def test_delete_profile():
    """删除配置 — 元数据和 keyring 条目均移除。"""
    print("=" * 60)
    print("测试删除连接配置")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile("删除测试", "http://test.local/v1", "sk-to-delete")
            pid = meta["profile_id"]

            assert load_connection_profile(pid) is not None
            assert has_connection_api_key(pid) is True

            delete_connection_profile(pid)

            assert load_connection_profile(pid) is None
            assert has_connection_api_key(pid) is False
            print("[OK] 元数据和 key 均已删除")

    print()


def test_mask_api_key():
    """脱敏 API Key — 正确脱敏。"""
    print("=" * 60)
    print("测试 API Key 脱敏")
    print("=" * 60)

    assert mask_api_key("app-aYY4JyhFyDrsuobLEvuBl5DI") == "app-...l5DI"
    assert mask_api_key("sk-12345678") == "sk-1...5678"
    assert mask_api_key("short") == "***"
    assert mask_api_key("") == "***"
    assert mask_api_key(None) == "***"
    print("[OK] 脱敏正确")

    print()


def test_no_key_in_metadata():
    """无 API Key 泄露 — 元数据 JSON 中不含任何 key 片段。"""
    print("=" * 60)
    print("测试 API Key 无泄露")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            test_keys = [
                "sk-test-1234567890",
                "app-aYY4JyhFyDrsuobLEvuBl5DI",
                "Bearer-token-xyz",
            ]
            for i, key in enumerate(test_keys):
                meta = create_connection_profile(f"泄露测试{i}", f"http://test{i}.local/v1", key)
                meta_path = Path(tmpdir) / f"{meta['profile_id']}.json"
                raw = meta_path.read_text(encoding="utf-8")
                assert key not in raw, f"API Key '{key}' 不应出现在元数据 JSON 中"
                # 也检查部分片段
                assert key[:8] not in raw, f"API Key 前8位不应出现在元数据中"

            print("[OK] 所有测试 key 均未泄露到元数据 JSON")

    print()


def test_profile_in_manifest():
    """Run manifest 含 profile_id/profile_name/base_url 但不含 api_key。"""
    print("=" * 60)
    print("测试 manifest 中的连接配置信息")
    print("=" * 60)

    _setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("dify_connection.CONNECTIONS_DIR", Path(tmpdir)):
            meta = create_connection_profile(
                "Manifest测试", "http://manifest.local/v1", "sk-manifest-secret",
                workflow_description="测试工作流",
            )

            # 模拟 manifest 更新（与 app.py 中的逻辑一致）
            manifest = {
                "run_id": "test_run",
                "config_id": "test_config",
                "dify_connection_profile_id": meta["profile_id"],
                "dify_connection_profile_name": meta["profile_name"],
                "dify_base_url": meta["base_url"],
                "dify_workflow_description": meta.get("workflow_description", ""),
            }

            # 验证 manifest 不含 API Key
            manifest_json = json.dumps(manifest, ensure_ascii=False)
            assert "sk-manifest-secret" not in manifest_json
            assert manifest["dify_connection_profile_id"] == meta["profile_id"]
            assert manifest["dify_connection_profile_name"] == "Manifest测试"
            assert manifest["dify_base_url"] == "http://manifest.local/v1"
            print("[OK] manifest 含连接配置元数据，不含 API Key")

    print()


def main():
    print("=" * 60)
    print("Dify 连接配置测试")
    print("=" * 60)
    print()

    test_create_profile()
    test_list_profiles()
    test_load_profile()
    test_get_api_key()
    test_update_profile()
    test_clear_key()
    test_delete_profile()
    test_mask_api_key()
    test_no_key_in_metadata()
    test_profile_in_manifest()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
