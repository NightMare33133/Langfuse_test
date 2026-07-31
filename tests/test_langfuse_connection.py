"""
Langfuse 连接配置管理测试。

覆盖：
a. CRUD 基本操作
b. Key 安全存储（keyring / .credentials fallback）
c. JSON 元数据不含 Key
d. 编辑留空 Key 时保持原值
e. 删除 profile 后凭据一并删除
f. Host 规范化
g. 名称唯一性
h. 测试连接脱敏
i. mask 脱敏函数
j. 空 profile 可新建
k. 旧 JSON 迁移
l. 凭据缺失状态
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Mock keyring
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

with patch("langfuse_connection._get_keyring", return_value=_mock_keyring):
    from langfuse_connection import (
        normalize_host, create_profile, update_profile, delete_profile,
        list_profiles, load_profile, check_connection,
        mask_public_key, mask_secret_key,
        get_profile_api_keys, has_profile_api_keys,
        _CONNECTIONS_DIR, _OLD_CONNECTIONS_FILE,
    )


@pytest.fixture(autouse=True)
def _use_tmp_dir(tmp_path):
    """每个测试使用独立临时目录。"""
    _mock_store.clear()
    with patch("langfuse_connection._CONNECTIONS_DIR", tmp_path / "langfuse_connections"), \
         patch("langfuse_connection._OLD_CONNECTIONS_FILE", tmp_path / "langfuse_connections.json"), \
         patch("langfuse_connection._CRED_FILE", tmp_path / "langfuse_connections" / ".credentials"):
        yield tmp_path


# ─── Host 规范化 ─────────────────────────────────────────────────────────────


class TestNormalizeHost:
    """Host 规范化测试。"""

    def test_valid_hosts(self):
        assert normalize_host("http://localhost:3000") == "http://localhost:3000"
        assert normalize_host("https://langfuse.example.com") == "https://langfuse.example.com"
        assert normalize_host("http://localhost:3000/") == "http://localhost:3000"
        assert normalize_host("  http://localhost:3000/  ") == "http://localhost:3000"

    def test_invalid_hosts(self):
        for bad in ["", "ftp://host", "localhost:3000", "ws://host"]:
            with pytest.raises(ValueError):
                normalize_host(bad)


# ─── CRUD ────────────────────────────────────────────────────────────────────


class TestCRUD:
    """CRUD 基本操作。"""

    def test_create_and_load(self):
        p1 = create_profile("测试A", "http://localhost:3000", "pk-lf-aaa", "sk-lf-bbb")
        assert p1["display_name"] == "测试A"
        assert p1["host"] == "http://localhost:3000"
        assert p1["profile_id"].startswith("lf_")
        # 元数据不含 Key
        assert "public_key" not in p1
        assert "secret_key" not in p1

        loaded = load_profile(p1["profile_id"])
        assert loaded is not None
        assert loaded["display_name"] == "测试A"
        assert "public_key" not in loaded
        assert "secret_key" not in loaded

    def test_list_profiles(self):
        create_profile("配置A", "http://localhost:3000", "pk-a", "sk-a")
        create_profile("配置B", "http://localhost:3001", "pk-b", "sk-b")
        profiles = list_profiles()
        assert len(profiles) == 2

    def test_update_profile(self):
        p = create_profile("原始名", "http://localhost:3000", "pk-orig", "sk-orig")
        pid = p["profile_id"]

        updated = update_profile(pid, "新名称", "http://localhost:4000", "pk-new", None)
        assert updated["display_name"] == "新名称"
        assert updated["host"] == "http://localhost:4000"

        # Key 更新
        pk, sk = get_profile_api_keys(pid)
        assert pk == "pk-new"
        assert sk == "sk-orig", "Secret Key 未传入时应保持原值"

    def test_delete_profile(self):
        p = create_profile("待删除", "http://localhost:3000", "pk-del", "sk-del")
        pid = p["profile_id"]
        assert delete_profile(pid) is True
        assert load_profile(pid) is None
        assert has_profile_api_keys(pid) is False
        assert delete_profile("nonexistent") is False

    def test_name_uniqueness(self):
        create_profile("唯一名称", "http://localhost:3000", "pk", "sk")
        with pytest.raises(ValueError, match="已存在"):
            create_profile("唯一名称", "http://localhost:3001", "pk2", "sk2")


# ─── Key 安全存储 ────────────────────────────────────────────────────────────


class TestKeyStorage:
    """Key 安全存储测试。"""

    def test_keys_not_in_metadata_json(self):
        """JSON 文件不含明文 Key。"""
        p = create_profile("安全测试", "http://localhost:3000", "pk-real", "sk-real")
        # 直接读取 profile 元数据文件
        loaded = load_profile(p["profile_id"])
        content = json.dumps(loaded, ensure_ascii=False)
        assert "pk-real" not in content
        assert "sk-real" not in content

    def test_keys_readable_via_api(self):
        """Key 可通过 get_profile_api_keys 读取。"""
        p = create_profile("Key测试", "http://localhost:3000", "pk-test", "sk-test")
        pk, sk = get_profile_api_keys(p["profile_id"])
        assert pk == "pk-test"
        assert sk == "sk-test"

    def test_has_profile_api_keys(self):
        """has_profile_api_keys 正确报告。"""
        p = create_profile("存在测试", "http://localhost:3000", "pk", "sk")
        assert has_profile_api_keys(p["profile_id"]) is True
        assert has_profile_api_keys("nonexistent") is False

    def test_edit_empty_key_keeps_original(self):
        """编辑留空 Key 保持原值。"""
        p = create_profile("留空测试", "http://localhost:3000", "pk-orig", "sk-orig")
        pid = p["profile_id"]

        # None 保持原值（不更新）
        update_profile(pid, "留空测试", "http://localhost:3000", None, None)
        pk, sk = get_profile_api_keys(pid)
        assert pk == "pk-orig"
        assert sk == "sk-orig"

        # 空字符串对 secret_key 也保持原值
        update_profile(pid, "留空测试", "http://localhost:3000", "pk-new", "")
        pk, sk = get_profile_api_keys(pid)
        assert pk == "pk-new"
        assert sk == "sk-orig"

    def test_delete_removes_credentials(self):
        """删除 profile 后凭据一并删除。"""
        p = create_profile("删除测试", "http://localhost:3000", "pk-del", "sk-del")
        pid = p["profile_id"]
        assert has_profile_api_keys(pid) is True

        delete_profile(pid)
        assert has_profile_api_keys(pid) is False


# ─── 脱敏函数 ────────────────────────────────────────────────────────────────


class TestMaskFunctions:
    """脱敏函数测试。"""

    def test_mask_public_key(self):
        assert mask_public_key("pk-lf-fe595c51-b982-40c6-9cbf-9dc0c52c6420") == "pk-lf-...6420"
        assert mask_public_key("short") == "***"
        assert mask_public_key("") == "***"

    def test_mask_secret_key(self):
        assert mask_secret_key("sk-lf-anything") == "已配置"
        assert mask_secret_key("") == ""


# ─── 测试连接 ────────────────────────────────────────────────────────────────


class TestCheckConnection:
    """测试连接脱敏。"""

    @patch("langfuse_connection._requests")
    def test_connection_failure_desensitized(self, mock_req):
        mock_req.get.side_effect = mock_req.RequestException("Connection refused")
        mock_req.RequestException = Exception
        ok, msg = check_connection("http://localhost:3000", "pk-lf-real", "sk-lf-real")
        assert ok is False
        assert "pk-lf-real" not in msg
        assert "sk-lf-real" not in msg

    @patch("langfuse_connection._requests")
    def test_auth_failure_desensitized(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_req.get.return_value = mock_resp
        mock_req.get.side_effect = None
        ok, msg = check_connection("http://localhost:3000", "pk-lf-real", "sk-lf-real")
        assert ok is False
        assert "pk-lf-real" not in msg

    @patch("langfuse_connection._requests")
    def test_success_desensitized(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"meta": {"totalItems": 42}}
        mock_req.get.return_value = mock_resp
        mock_req.get.side_effect = None
        ok, msg = check_connection("http://localhost:3000", "pk-lf-real", "sk-lf-real")
        assert ok is True
        assert "pk-lf-real" not in msg
        assert "42" in msg


# ─── 空 profile 新建 ────────────────────────────────────────────────────────


class TestEmptyProfiles:
    """空 profile 状态测试。"""

    def test_empty_list_returns_empty(self):
        assert list_profiles() == []

    def test_can_create_first_profile(self):
        """无 profile 时可新建。"""
        p = create_profile("首个配置", "http://localhost:3000", "pk-first", "sk-first")
        assert p["display_name"] == "首个配置"
        assert has_profile_api_keys(p["profile_id"]) is True

    def test_profile_survives_reread(self):
        """新建后重新读取仍存在（模拟重启）。"""
        p = create_profile("持久测试", "http://localhost:3000", "pk-persist", "sk-persist")
        pid = p["profile_id"]

        # 重新读取
        loaded = load_profile(pid)
        assert loaded is not None
        assert loaded["display_name"] == "持久测试"

        pk, sk = get_profile_api_keys(pid)
        assert pk == "pk-persist"
        assert sk == "sk-persist"


# ─── 旧 JSON 迁移 ───────────────────────────────────────────────────────────


class TestMigration:
    """旧 JSON 迁移测试。"""

    def test_migrate_old_json(self, tmp_path):
        """旧 langfuse_connections.json 迁移到新结构。"""
        old_file = tmp_path / "langfuse_connections.json"
        old_data = {
            "lf_old_001": {
                "profile_id": "lf_old_001",
                "display_name": "旧配置",
                "host": "http://localhost:3000",
                "public_key": "pk-old-key",
                "secret_key": "sk-old-key",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        }
        old_file.write_text(json.dumps(old_data), encoding="utf-8")

        with patch("langfuse_connection._OLD_CONNECTIONS_FILE", old_file), \
             patch("langfuse_connection._CONNECTIONS_DIR", tmp_path / "langfuse_connections"), \
             patch("langfuse_connection._CRED_FILE", tmp_path / "langfuse_connections" / ".credentials"):
            # 触发迁移
            profiles = list_profiles()

        # 验证迁移结果
        assert len(profiles) == 1
        assert profiles[0]["display_name"] == "旧配置"
        assert "public_key" not in profiles[0]
        assert "secret_key" not in profiles[0]

        # Key 已存入 keyring
        pk, sk = get_profile_api_keys("lf_old_001")
        assert pk == "pk-old-key"
        assert sk == "sk-old-key"

        # 旧文件已删除
        assert not old_file.exists()


# ─── Host 规范化集成 ─────────────────────────────────────────────────────────


class TestHostNormalization:
    """Host 创建/编辑时规范化。"""

    def test_normalized_on_create(self):
        p = create_profile("Host测试", "http://localhost:3000/", "pk", "sk")
        assert p["host"] == "http://localhost:3000"

    def test_normalized_on_update(self):
        p = create_profile("Host编辑", "http://localhost:3000", "pk", "sk")
        updated = update_profile(p["profile_id"], "Host编辑", "http://localhost:4000///", None, None)
        assert updated["host"] == "http://localhost:4000"


# ─── Key 为空拒绝 ────────────────────────────────────────────────────────────


class TestEmptyKeyRejected:
    """空 Key 拒绝测试。"""

    def test_empty_public_key_rejected(self):
        with pytest.raises(ValueError, match="Public Key"):
            create_profile("空PK", "http://localhost:3000", "", "sk")

    def test_empty_secret_key_rejected(self):
        with pytest.raises(ValueError, match="Secret Key"):
            create_profile("空SK", "http://localhost:3000", "pk", "")


# ─── Key 不泄露到错误消息 ────────────────────────────────────────────────────


class TestKeyNotInErrors:
    """Key 不泄露到错误消息。"""

    def test_name_duplicate_no_key_leak(self):
        real_pk = "pk-lf-fe595c51-b982-40c6-9cbf-9dc0c52c6420"
        real_sk = "sk-lf-a226c439-e9d1-4aad-8024-bbb56fdd42b9"
        create_profile("泄露测试", "http://localhost:3000", real_pk, real_sk)
        try:
            create_profile("泄露测试", "http://localhost:3000", real_pk, real_sk)
        except ValueError as e:
            assert real_pk not in str(e)
            assert real_sk not in str(e)

    def test_bad_host_no_key_leak(self):
        real_pk = "pk-lf-fe595c51-b982-40c6-9cbf-9dc0c52c6420"
        real_sk = "sk-lf-a226c439-e9d1-4aad-8024-bbb56fdd42b9"
        try:
            check_connection("ftp://bad", real_pk, real_sk)
        except ValueError as e:
            assert real_pk not in str(e)
            assert real_sk not in str(e)


# ─── gitignore 覆盖 ─────────────────────────────────────────────────────────


class TestGitignore:
    """gitignore 覆盖测试。"""

    def test_gitignore_covers_connections_dir(self):
        gitignore_path = ROOT / ".gitignore"
        assert gitignore_path.exists()
        content = gitignore_path.read_text(encoding="utf-8")
        assert "data/langfuse_connections/" in content
        assert "data/langfuse_connections/.credentials" in content
