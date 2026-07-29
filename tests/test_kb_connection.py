"""
知识库连接配置 CRUD 测试。

测试内容：
1. create_kb_profile — 创建配置，元数据不含 Key，Key 存入 keyring
2. list_kb_profiles — 列出所有配置
3. load_kb_profile — 加载单个配置
4. update_kb_profile — 更新元数据和 Key
5. delete_kb_profile — 删除配置和 Key
6. validate_dataset_key — 前缀校验
7. mask_api_key — 脱敏显示
8. 安全：元数据 JSON 不含明文 Key
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

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


with patch("dify_kb_connection._get_keyring", return_value=_mock_keyring):
    from dify_kb_connection import (
        create_kb_profile, list_kb_profiles, load_kb_profile,
        update_kb_profile, delete_kb_profile,
        get_kb_api_key, has_kb_api_key,
        validate_dataset_key, mask_api_key,
        CONNECTIONS_DIR,
    )


@pytest.fixture(autouse=True)
def _use_tmp_dir(tmp_path):
    """每个测试使用独立临时目录。"""
    _mock_store.clear()
    with patch("dify_kb_connection.CONNECTIONS_DIR", tmp_path / "kb_conn"):
        yield tmp_path


# ── validate_dataset_key ─────────────────────────────────────


class TestValidateDatasetKey:
    """测试 Key 前缀校验。"""

    def test_valid_dataset_key(self):
        """dataset- 前缀合法。"""
        ok, err = validate_dataset_key("dataset-abc123def456")
        assert ok is True
        assert err == ""

    def test_rejects_app_key(self):
        """app- 前缀被拒绝。"""
        ok, err = validate_dataset_key("app-abc123def456")
        assert ok is False
        assert "应用 API Key" in err
        assert "app-" in err

    def test_rejects_empty_key(self):
        """空 Key 被拒绝。"""
        ok, err = validate_dataset_key("")
        assert ok is False
        assert "缺少知识库 API Key" in err

    def test_rejects_dat_key(self):
        """dat- 前缀（非 dataset-）被拒绝。"""
        ok, err = validate_dataset_key("dat-abc123def456")
        assert ok is False
        assert "前缀不正确" in err
        assert "dataset-" in err

    def test_rejects_other_prefix(self):
        """其他前缀被拒绝。"""
        ok, err = validate_dataset_key("sk-abc123def456")
        assert ok is False
        assert "前缀不正确" in err


# ── mask_api_key ─────────────────────────────────────────────


class TestMaskApiKey:
    """测试 Key 脱敏显示。"""

    def test_mask_normal_key(self):
        """正常长度 Key 脱敏。"""
        masked = mask_api_key("dataset-abc123def456")
        assert masked.startswith("dataset-ab")
        assert masked.endswith("f456")
        assert "..." in masked

    def test_mask_short_key(self):
        """过短 Key 返回 ***。"""
        assert mask_api_key("short") == "***"
        assert mask_api_key("") == "***"
        assert mask_api_key(None) == "***"


# ── CRUD 操作 ────────────────────────────────────────────────


class TestCreateProfile:
    """测试创建配置。"""

    def test_create_basic(self):
        """创建基本配置。"""
        meta = create_kb_profile("测试配置", "http://localhost/v1", "dataset-abc123def456")
        assert meta["profile_name"] == "测试配置"
        assert meta["base_url"] == "http://localhost/v1"
        assert "profile_id" in meta
        assert meta["profile_id"].startswith("kb_")
        assert "created_at" in meta
        assert "updated_at" in meta

    def test_create_stores_masked_key(self):
        """元数据中只保存脱敏 Key。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        assert "key_masked" in meta
        assert "dataset-ab" in meta["key_masked"]
        # 明文 Key 不在元数据中
        assert "dataset-abc123def456" not in json.dumps(meta)

    def test_create_rejects_app_key(self):
        """app- Key 被拒绝。"""
        with pytest.raises(ValueError, match="应用 API Key"):
            create_kb_profile("测试", "http://localhost/v1", "app-abc123def456")

    def test_create_rejects_dat_key(self):
        """dat- Key 被拒绝。"""
        with pytest.raises(ValueError, match="前缀不正确"):
            create_kb_profile("测试", "http://localhost/v1", "dat-abc123def456")

    def test_create_saves_key_to_keyring(self):
        """Key 存入 keyring。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        saved = get_kb_api_key(meta["profile_id"])
        assert saved == "dataset-abc123def456"

    def test_create_no_key_in_metadata_file(self):
        """元数据文件不含明文 Key。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        # 使用 sys.modules 获取动态 patched 的 CONNECTIONS_DIR
        import sys
        kb_mod = sys.modules["dify_kb_connection"]
        meta_path = kb_mod.CONNECTIONS_DIR / f"{meta['profile_id']}.json"
        content = meta_path.read_text(encoding="utf-8")
        assert "dataset-abc123def456" not in content
        assert "key_masked" in content


class TestListProfiles:
    """测试列出配置。"""

    def test_empty_list(self):
        """无配置时返回空列表。"""
        assert list_kb_profiles() == []

    def test_list_returns_all(self):
        """返回所有配置。"""
        create_kb_profile("配置A", "http://a.local/v1", "dataset-aaa111aaa111")
        create_kb_profile("配置B", "http://b.local/v1", "dataset-bbb222bbb222")
        profiles = list_kb_profiles()
        assert len(profiles) == 2
        names = {p["profile_name"] for p in profiles}
        assert "配置A" in names
        assert "配置B" in names


class TestLoadProfile:
    """测试加载配置。"""

    def test_load_existing(self):
        """加载存在的配置。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        loaded = load_kb_profile(meta["profile_id"])
        assert loaded["profile_name"] == "测试"
        assert loaded["base_url"] == "http://localhost/v1"

    def test_load_nonexistent(self):
        """加载不存在的配置返回 None。"""
        assert load_kb_profile("nonexistent_id") is None


class TestUpdateProfile:
    """测试更新配置。"""

    def test_update_name_and_url(self):
        """更新名称和 URL。"""
        meta = create_kb_profile("旧名称", "http://old.local/v1", "dataset-abc123def456")
        updated = update_kb_profile(
            meta["profile_id"],
            {"profile_name": "新名称", "base_url": "http://new.local/v1"},
        )
        assert updated["profile_name"] == "新名称"
        assert updated["base_url"] == "http://new.local/v1"

    def test_update_key(self):
        """更新 Key。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-old111old111")
        update_kb_profile(
            meta["profile_id"],
            {},
            api_key="dataset-new222new222",
        )
        assert get_kb_api_key(meta["profile_id"]) == "dataset-new222new222"

    def test_update_rejects_app_key(self):
        """更新时 app- Key 被拒绝。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        with pytest.raises(ValueError, match="应用 API Key"):
            update_kb_profile(meta["profile_id"], {}, api_key="app-newkey")

    def test_clear_key(self):
        """清除 Key。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        update_kb_profile(meta["profile_id"], {}, clear_key=True)
        assert not has_kb_api_key(meta["profile_id"])

    def test_update_nonexistent_raises(self):
        """更新不存在的配置抛出 ValueError。"""
        with pytest.raises(ValueError, match="不存在"):
            update_kb_profile("nonexistent", {"profile_name": "x"})


class TestDeleteProfile:
    """测试删除配置。"""

    def test_delete_existing(self):
        """删除存在的配置。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        pid = meta["profile_id"]
        assert delete_kb_profile(pid) is True
        assert load_kb_profile(pid) is None
        assert not has_kb_api_key(pid)

    def test_delete_nonexistent(self):
        """删除不存在的配置返回 False。"""
        assert delete_kb_profile("nonexistent") is False


class TestKeySafety:
    """测试 Key 安全性。"""

    def test_metadata_json_no_plaintext_key(self):
        """元数据 JSON 不含明文 Key。"""
        secret = "dataset-SUPERSECRETKEY12345678"
        meta = create_kb_profile("安全测试", "http://localhost/v1", secret)
        import sys
        kb_mod = sys.modules["dify_kb_connection"]
        meta_path = kb_mod.CONNECTIONS_DIR / f"{meta['profile_id']}.json"
        content = meta_path.read_text(encoding="utf-8")
        assert secret not in content
        parsed = json.loads(content)
        assert secret not in json.dumps(parsed)
        assert "key_masked" in parsed

    def test_masked_key_in_metadata(self):
        """元数据包含脱敏 Key。"""
        meta = create_kb_profile("测试", "http://localhost/v1", "dataset-abc123def456")
        assert "key_masked" in meta
        assert meta["key_masked"].startswith("dataset-ab")
        assert "abc123def456" not in meta["key_masked"]
