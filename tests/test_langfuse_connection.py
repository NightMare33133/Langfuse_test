"""
Langfuse 连接配置管理测试。

覆盖：
a. CRUD 基本操作
b. Secret Key 不出现在 UI 状态、错误和导出
c. JSON 文件被 gitignore
d. 编辑留空 Secret Key 时保持原值
e. 删除 profile 后本地凭据一并删除
f. Host 规范化
g. 名称唯一性
h. 测试连接脱敏
i. mask 脱敏函数
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langfuse_connection import (
    normalize_host, create_profile, update_profile, delete_profile,
    list_profiles, load_profile, check_connection,
    mask_public_key, mask_secret_key,
    _CONNECTIONS_FILE, _DATA_DIR,
)


def _cleanup():
    if _CONNECTIONS_FILE.exists():
        _CONNECTIONS_FILE.unlink()


def _setup():
    _cleanup()


def _teardown():
    _cleanup()


# ─── Host 规范化 ─────────────────────────────────────────────────────────────

def test_normalize_host():
    print("=" * 60)
    print("测试：Host 规范化")
    print("=" * 60)

    assert normalize_host("http://localhost:3000") == "http://localhost:3000"
    assert normalize_host("https://langfuse.example.com") == "https://langfuse.example.com"
    assert normalize_host("http://localhost:3000/") == "http://localhost:3000"
    assert normalize_host("  http://localhost:3000/  ") == "http://localhost:3000"
    print("[OK] 正常 Host 规范化通过")

    for bad in ["", "ftp://host", "localhost:3000", "ws://host"]:
        try:
            normalize_host(bad)
            assert False, f"应抛出 ValueError: {bad}"
        except ValueError:
            pass
    print("[OK] 非法 Host 正确拒绝")
    print()


# ─── CRUD ────────────────────────────────────────────────────────────────────

def test_crud_basic():
    print("=" * 60)
    print("测试：CRUD 基本操作")
    print("=" * 60)
    _setup()

    p1 = create_profile("测试A", "http://localhost:3000", "pk-lf-aaa", "sk-lf-bbb")
    assert p1["display_name"] == "测试A"
    assert p1["host"] == "http://localhost:3000"
    assert p1["public_key"] == "pk-lf-aaa"
    assert p1["secret_key"] == "sk-lf-bbb"
    assert p1["profile_id"].startswith("lf_")
    print(f"[OK] 新建: {p1['profile_id']}")

    loaded = load_profile(p1["profile_id"])
    assert loaded["public_key"] == "pk-lf-aaa"
    assert loaded["secret_key"] == "sk-lf-bbb"
    print("[OK] 加载含凭据")

    p2 = create_profile("测试B", "http://localhost:3001", "pk-lf-ccc", "sk-lf-ddd")
    profiles = list_profiles()
    assert len(profiles) == 2
    print(f"[OK] 列表: {len(profiles)} 个")

    updated = update_profile(p1["profile_id"], "测试A改名", "http://localhost:3002", "pk-lf-new", None)
    assert updated["display_name"] == "测试A改名"
    assert updated["host"] == "http://localhost:3002"
    assert updated["public_key"] == "pk-lf-new"
    assert updated["secret_key"] == "sk-lf-bbb", "Secret Key 未传入时应保持原值"
    print("[OK] 编辑成功，SK 保持原值")

    assert delete_profile(p1["profile_id"]) is True
    assert load_profile(p1["profile_id"]) is None
    assert len(list_profiles()) == 1
    print("[OK] 删除成功")

    assert delete_profile("nonexistent") is False
    print("[OK] 删除不存在返回 False")

    _teardown()
    print()


# ─── 名称唯一性 ──────────────────────────────────────────────────────────────

def test_name_uniqueness():
    print("=" * 60)
    print("测试：名称唯一性")
    print("=" * 60)
    _setup()

    create_profile("唯一名称", "http://localhost:3000", "pk", "sk")
    try:
        create_profile("唯一名称", "http://localhost:3001", "pk2", "sk2")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "已存在" in str(e)
        print(f"[OK] 重复名称拒绝: {e}")

    _teardown()
    print()


# ─── 编辑留空 SK 保持原值 ────────────────────────────────────────────────────

def test_edit_empty_sk_keeps_original():
    print("=" * 60)
    print("测试：编辑留空 SK 保持原值")
    print("=" * 60)
    _setup()

    p = create_profile("SK测试", "http://localhost:3000", "pk-orig", "sk-orig")
    pid = p["profile_id"]

    # secret_key=None 表示不更新
    updated = update_profile(pid, "SK测试", "http://localhost:3000", "pk-new", None)
    assert updated["secret_key"] == "sk-orig", "None 应保持原值"
    print("[OK] None 保持原值")

    # secret_key="" 也保持原值
    updated2 = update_profile(pid, "SK测试", "http://localhost:3000", "pk-new", "")
    assert updated2["secret_key"] == "sk-orig", "空字符串应保持原值"
    print("[OK] 空字符串保持原值")

    # secret_key="sk-changed" 应更新
    updated3 = update_profile(pid, "SK测试", "http://localhost:3000", "pk-new", "sk-changed")
    assert updated3["secret_key"] == "sk-changed"
    print("[OK] 非空值更新成功")

    _teardown()
    print()


# ─── 删除后凭据一并删除 ─────────────────────────────────────────────────────

def test_delete_removes_credentials():
    print("=" * 60)
    print("测试：删除后凭据一并删除")
    print("=" * 60)
    _setup()

    p = create_profile("待删除", "http://localhost:3000", "pk-del", "sk-del")
    pid = p["profile_id"]

    # 确认 JSON 中有凭据
    with _CONNECTIONS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert pid in data
    assert data[pid]["secret_key"] == "sk-del"
    print("[OK] 删除前凭据存在")

    delete_profile(pid)

    with _CONNECTIONS_FILE.open("r", encoding="utf-8") as f:
        data2 = json.load(f)
    assert pid not in data2
    print("[OK] 删除后凭据已清除")

    _teardown()
    print()


# ─── 测试连接脱敏 ────────────────────────────────────────────────────────────

def test_connection_desensitized():
    print("=" * 60)
    print("测试：测试连接脱敏")
    print("=" * 60)

    with patch("langfuse_connection._requests") as mock_req:
        mock_req.get.side_effect = mock_req.RequestException("Connection refused")
        mock_req.RequestException = Exception
        ok, msg = check_connection("http://localhost:3000", "pk-lf-real", "sk-lf-real")
        assert ok is False
        assert "pk-lf-real" not in msg, "报错不应包含 Key"
        assert "sk-lf-real" not in msg, "报错不应包含 Key"
        print(f"[OK] 连接失败脱敏: {msg}")

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_req.get.return_value = mock_resp
        mock_req.get.side_effect = None
        ok, msg = check_connection("http://localhost:3000", "pk-lf-real", "sk-lf-real")
        assert ok is False
        assert "pk-lf-real" not in msg
        assert "sk-lf-real" not in msg
        print(f"[OK] 认证失败脱敏: {msg}")

        mock_resp.status_code = 200
        mock_resp.json.return_value = {"meta": {"totalItems": 42}}
        ok, msg = check_connection("http://localhost:3000", "pk-lf-real", "sk-lf-real")
        assert ok is True
        assert "pk-lf-real" not in msg
        assert "42" in msg
        print(f"[OK] 成功脱敏: {msg}")

    print()


# ─── mask 函数 ───────────────────────────────────────────────────────────────

def test_mask_functions():
    print("=" * 60)
    print("测试：mask 脱敏函数")
    print("=" * 60)

    assert mask_public_key("pk-lf-fe595c51-b982-40c6-9cbf-9dc0c52c6420") == "pk-lf-...6420"
    assert mask_public_key("short") == "***"
    assert mask_public_key("") == "***"
    print(f"[OK] mask_public_key: {mask_public_key('pk-lf-fe595c51-b982-40c6-9cbf-9dc0c52c6420')}")

    assert mask_secret_key("sk-lf-anything") == "已配置"
    assert mask_secret_key("") == ""
    print(f"[OK] mask_secret_key: '已配置'")

    print()


# ─── Key 不泄露到错误消息 ────────────────────────────────────────────────────

def test_key_not_in_errors():
    print("=" * 60)
    print("测试：Key 不泄露到错误消息")
    print("=" * 60)
    _setup()

    real_pk = "pk-lf-fe595c51-b982-40c6-9cbf-9dc0c52c6420"
    real_sk = "sk-lf-a226c439-e9d1-4aad-8024-bbb56fdd42b9"

    p = create_profile("泄露测试", "http://localhost:3000", real_pk, real_sk)

    # 检查所有可能的错误路径
    try:
        create_profile("泄露测试", "http://localhost:3000", real_pk, real_sk)
    except ValueError as e:
        assert real_pk not in str(e)
        assert real_sk not in str(e)
        print("[OK] 名称重复错误不含 Key")

    try:
        check_connection("ftp://bad", real_pk, real_sk)
    except ValueError as e:
        assert real_pk not in str(e)
        assert real_sk not in str(e)
        print("[OK] Host 非法错误不含 Key")

    _teardown()
    print()


# ─── Host 规范化集成 ─────────────────────────────────────────────────────────

def test_host_normalized_on_create():
    print("=" * 60)
    print("测试：Host 创建时规范化")
    print("=" * 60)
    _setup()

    p = create_profile("Host测试", "http://localhost:3000/", "pk", "sk")
    assert p["host"] == "http://localhost:3000"
    print(f"[OK] 创建时规范化: {p['host']}")

    updated = update_profile(p["profile_id"], "Host测试", "http://localhost:4000///", "pk", None)
    assert updated["host"] == "http://localhost:4000"
    print(f"[OK] 编辑时规范化: {updated['host']}")

    _teardown()
    print()


# ─── Key 为空拒绝 ────────────────────────────────────────────────────────────

def test_empty_key_rejected():
    print("=" * 60)
    print("测试：空 Key 拒绝")
    print("=" * 60)
    _setup()

    try:
        create_profile("空PK", "http://localhost:3000", "", "sk")
        assert False, "应拒绝空 PK"
    except ValueError as e:
        assert "Public Key" in str(e)
        print(f"[OK] 空 PK 拒绝: {e}")

    try:
        create_profile("空SK", "http://localhost:3000", "pk", "")
        assert False, "应拒绝空 SK"
    except ValueError as e:
        assert "Secret Key" in str(e)
        print(f"[OK] 空 SK 拒绝: {e}")

    _teardown()
    print()


# ─── JSON 文件 gitignore ─────────────────────────────────────────────────────

def test_gitignore_coverage():
    print("=" * 60)
    print("测试：JSON 文件 gitignore 覆盖")
    print("=" * 60)

    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore 应存在"
    content = gitignore_path.read_text(encoding="utf-8")
    assert "*.json" in content, ".gitignore 应包含 *.json 规则"
    print("[OK] .gitignore 包含 *.json 规则，覆盖 langfuse_connections.json")

    print()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Langfuse 连接配置管理测试")
    print("=" * 60)
    print()

    test_normalize_host()
    test_crud_basic()
    test_name_uniqueness()
    test_edit_empty_sk_keeps_original()
    test_delete_removes_credentials()
    test_connection_desensitized()
    test_mask_functions()
    test_key_not_in_errors()
    test_host_normalized_on_create()
    test_empty_key_rejected()
    test_gitignore_coverage()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
