"""
Langfuse 连接配置管理模块 — 命名连接配置 + 安全凭据存储。

对齐 dify_connection.py 的 CRUD 模式。
连接配置元数据存储在 data/langfuse_connections/<profile_id>.json（不含 Key）。
凭据通过 keyring 存储（macOS Keychain / Windows Credential Manager / Linux Secret Service）。

安全规则：
- Key 绝不写入元数据 JSON、manifest、JSONL、日志或报错信息
- 所有显示使用 mask_public_key() / mask_secret_key() 脱敏
- data/langfuse_connections/ 目录已加入 .gitignore
"""

import json
import re
from datetime import datetime
from pathlib import Path

import requests as _requests

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CONNECTIONS_DIR = _DATA_DIR / "langfuse_connections"
_OLD_CONNECTIONS_FILE = _DATA_DIR / "langfuse_connections.json"
_KEYRING_SERVICE = "langfuse-rag-eval.langfuse"


# ─── 脱敏工具 ────────────────────────────────────────────────────────────────


def mask_public_key(key: str) -> str:
    """Public Key 脱敏：显示前缀 + *** + 后4位。"""
    if not key or len(key) < 12:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def mask_secret_key(key: str) -> str:
    """Secret Key 固定返回 '已配置'，不显示任何字符。"""
    return "已配置" if key else ""


# ─── 内部工具 ────────────────────────────────────────────────────────────────


def _generate_id(name: str) -> str:
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    us = f"{now.microsecond:06d}"
    slug = re.sub(r"[^\w一-鿿]", "_", name.strip())[:20]
    return f"lf_{ts}_{us}_{slug}"


def _profile_path(profile_id: str) -> Path:
    return _CONNECTIONS_DIR / f"{profile_id}.json"


def _read_profile(profile_id: str) -> dict | None:
    """读取单个 profile 元数据。"""
    p = _profile_path(profile_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _write_profile(profile_id: str, metadata: dict):
    """写入单个 profile 元数据。"""
    _CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = _profile_path(profile_id)
    p.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_profile_file(profile_id: str) -> bool:
    """删除 profile 元数据文件。"""
    p = _profile_path(profile_id)
    if p.exists():
        p.unlink()
        return True
    return False


# ─── Keyring 操作 ────────────────────────────────────────────────────────────


def _get_keyring():
    """获取 keyring 模块，不可用时返回 None。"""
    try:
        import keyring
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _store_api_key(profile_id: str, key_type: str, api_key: str):
    """存储 API Key 到 keyring。key_type: 'public' 或 'secret'。"""
    if not api_key:
        return
    service = f"{_KEYRING_SERVICE}.{key_type}"
    kr = _get_keyring()
    if kr:
        try:
            kr.set_password(service, profile_id, api_key)
            return
        except Exception:
            pass
    _store_api_key_file(profile_id, key_type, api_key)


def _read_api_key(profile_id: str, key_type: str) -> str:
    """从 keyring 读取 API Key。"""
    service = f"{_KEYRING_SERVICE}.{key_type}"
    kr = _get_keyring()
    if kr:
        try:
            key = kr.get_password(service, profile_id)
            if key:
                return key
        except Exception:
            pass
    return _read_api_key_file(profile_id, key_type)


def _delete_api_key(profile_id: str, key_type: str):
    """从 keyring 删除 API Key。"""
    service = f"{_KEYRING_SERVICE}.{key_type}"
    kr = _get_keyring()
    if kr:
        try:
            kr.delete_password(service, profile_id)
        except Exception:
            pass
    _delete_api_key_file(profile_id, key_type)


def _delete_all_api_keys(profile_id: str):
    """删除 profile 的所有凭据。"""
    _delete_api_key(profile_id, "public")
    _delete_api_key(profile_id, "secret")


# ─── Fallback 文件存储（仅 keyring 不可用时） ───────────────────────────────

_CRED_FILE = _CONNECTIONS_DIR / ".credentials"


def _store_api_key_file(profile_id: str, key_type: str, api_key: str):
    """Fallback: 将 API Key 存储到文件。"""
    _CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    creds = _load_credentials_file()
    creds[f"{profile_id}:{key_type}"] = api_key
    _save_credentials_file(creds)


def _read_api_key_file(profile_id: str, key_type: str) -> str:
    """Fallback: 从文件读取 API Key。"""
    creds = _load_credentials_file()
    return creds.get(f"{profile_id}:{key_type}")


def _delete_api_key_file(profile_id: str, key_type: str):
    """Fallback: 从文件删除 API Key。"""
    creds = _load_credentials_file()
    key = f"{profile_id}:{key_type}"
    if key in creds:
        del creds[key]
        _save_credentials_file(creds)


def _load_credentials_file() -> dict:
    """加载凭据文件。"""
    if not _CRED_FILE.exists():
        return {}
    try:
        return json.loads(_CRED_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return {}


def _save_credentials_file(creds: dict):
    """保存凭据文件。"""
    _CRED_FILE.write_text(json.dumps(creds, ensure_ascii=False), encoding="utf-8")


# ─── Host 校验 ───────────────────────────────────────────────────────────────


def normalize_host(raw: str) -> str:
    """校验并规范化 Host。

    - 仅允许 http / https
    - 去除末尾 /
    """
    if not raw or not raw.strip():
        raise ValueError("Host 不能为空")
    host = raw.strip().rstrip("/")
    if not re.match(r"^https?://", host, re.IGNORECASE):
        raise ValueError("Host 必须以 http:// 或 https:// 开头")
    return host


# ─── 旧格式迁移 ─────────────────────────────────────────────────────────────


def _migrate_old_json():
    """将旧 data/langfuse_connections.json 迁移到新结构。"""
    if not _OLD_CONNECTIONS_FILE.exists():
        return
    try:
        old_data = json.loads(_OLD_CONNECTIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return

    _CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for pid, old_profile in old_data.items():
        if _profile_path(pid).exists():
            # 已存在新格式，跳过
            continue

        # 写入元数据（不含 Key）
        metadata = {
            "profile_id": pid,
            "display_name": old_profile.get("display_name", ""),
            "host": old_profile.get("host", ""),
            "created_at": old_profile.get("created_at", ""),
            "updated_at": old_profile.get("updated_at", ""),
        }
        _write_profile(pid, metadata)

        # 存储凭据到 keyring
        pk = old_profile.get("public_key", "")
        sk = old_profile.get("secret_key", "")
        if pk:
            _store_api_key(pid, "public", pk)
        if sk:
            _store_api_key(pid, "secret", sk)

    # 删除旧文件
    try:
        _OLD_CONNECTIONS_FILE.unlink()
    except OSError:
        pass


# ─── CRUD ────────────────────────────────────────────────────────────────────


def list_profiles() -> list:
    """返回全部 profile 列表（元数据，不含 Key），按创建时间倒序。"""
    _migrate_old_json()
    if not _CONNECTIONS_DIR.exists():
        return []

    profiles = []
    for p in _CONNECTIONS_DIR.glob("*.json"):
        try:
            profiles.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, IOError):
            continue
    return sorted(profiles, key=lambda x: x.get("created_at", ""), reverse=True)


def load_profile(profile_id: str) -> dict | None:
    """加载单个 profile 元数据（不含 Key），不存在返回 None。"""
    _migrate_old_json()
    return _read_profile(profile_id)


def get_profile_api_keys(profile_id: str) -> tuple[str, str]:
    """从安全存储读取 API Key。仅在内存中使用，不序列化。

    Returns:
        (public_key, secret_key)  不存在或缺失时返回空字符串。
    """
    return _read_api_key(profile_id, "public"), _read_api_key(profile_id, "secret")


def has_profile_api_keys(profile_id: str) -> bool:
    """检查是否已保存 API Key。"""
    pk = _read_api_key(profile_id, "public")
    sk = _read_api_key(profile_id, "secret")
    return bool(pk and sk)


def create_profile(display_name: str, host: str,
                   public_key: str, secret_key: str) -> dict:
    """新建配置。元数据存 JSON，Key 存入 keyring。

    Returns:
        新建的 profile 元数据 dict（不含 Key）。

    Raises:
        ValueError: 名称为空/重复、Host 非法、Key 为空。
    """
    if not display_name or not display_name.strip():
        raise ValueError("配置名称不能为空")
    display_name = display_name.strip()
    host = normalize_host(host)
    if not public_key or not public_key.strip():
        raise ValueError("Public Key 不能为空")
    if not secret_key or not secret_key.strip():
        raise ValueError("Secret Key 不能为空")

    # 检查名称唯一
    for p in list_profiles():
        if p.get("display_name") == display_name:
            raise ValueError(f"配置名称已存在: {display_name}")

    now = datetime.now().isoformat()
    pid = _generate_id(display_name)
    metadata = {
        "profile_id": pid,
        "display_name": display_name,
        "host": host,
        "created_at": now,
        "updated_at": now,
    }
    _write_profile(pid, metadata)
    _store_api_key(pid, "public", public_key.strip())
    _store_api_key(pid, "secret", secret_key.strip())

    return metadata


def update_profile(profile_id: str, display_name: str, host: str,
                   public_key: str = None, secret_key: str = None) -> dict:
    """编辑配置。

    Args:
        public_key: 新值（None=不更新）
        secret_key: 新值（None=不更新, ""=保持原值不修改）

    Raises:
        ValueError: 配置不存在、名称为空/重复、Host 非法。
    """
    metadata = _read_profile(profile_id)
    if metadata is None:
        raise ValueError(f"配置不存在: {profile_id}")

    if not display_name or not display_name.strip():
        raise ValueError("配置名称不能为空")
    display_name = display_name.strip()
    host = normalize_host(host)

    for p in list_profiles():
        pid = p.get("profile_id", "")
        if pid != profile_id and p.get("display_name") == display_name:
            raise ValueError(f"配置名称已存在: {display_name}")

    metadata["display_name"] = display_name
    metadata["host"] = host
    metadata["updated_at"] = datetime.now().isoformat()
    _write_profile(profile_id, metadata)

    # 更新 Key
    if public_key is not None:
        if not public_key.strip():
            raise ValueError("Public Key 不能为空")
        _store_api_key(profile_id, "public", public_key.strip())
    if secret_key is not None and secret_key.strip():
        _store_api_key(profile_id, "secret", secret_key.strip())

    return metadata


def delete_profile(profile_id: str) -> bool:
    """删除配置（含本地凭据）。"""
    _delete_all_api_keys(profile_id)
    return _delete_profile_file(profile_id)


# ─── 测试连接 ────────────────────────────────────────────────────────────────


def check_connection(host: str, public_key: str, secret_key: str) -> tuple:
    """测试 Langfuse 连接。

    Returns:
        (ok: bool, message: str)  message 已脱敏。

    Raises:
        ValueError: Host 非法或 Key 为空。
    """
    host = normalize_host(host)
    if not public_key or not public_key.strip():
        raise ValueError("Public Key 不能为空")
    if not secret_key or not secret_key.strip():
        raise ValueError("Secret Key 不能为空")

    url = f"{host}/api/public/traces"
    try:
        resp = _requests.get(url, auth=(public_key.strip(), secret_key.strip()),
                             params={"limit": 1}, timeout=10)
    except _requests.RequestException as e:
        return False, f"连接失败: {type(e).__name__}"

    if resp.status_code == 200:
        try:
            data = resp.json()
            total = data.get("meta", {}).get("totalItems", "?")
            return True, f"连接成功（共 {total} 条 trace）"
        except Exception:
            return True, "连接成功"
    elif resp.status_code in (401, 403):
        return False, "认证失败，请检查 Key"
    else:
        return False, f"HTTP {resp.status_code}"


def identify_project_info(host: str, public_key: str, secret_key: str) -> dict:
    """识别 Langfuse 项目信息。

    Returns:
        {"project_id", "project_name", "host", "key_masked", "total_traces"}

    Raises:
        RuntimeError: 连接失败
    """
    from langfuse_project import identify_project
    return identify_project(host, public_key, secret_key)
