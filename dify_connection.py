"""
Dify 连接配置管理模块 — 命名连接配置 + 安全 API Key 存储。

连接配置存储：
- 元数据 JSON：data/dify_connections/<profile_id>.json（不含 API Key）
- API Key：通过 keyring 存储（Windows Credential Manager / macOS Keychain / Linux Secret Service）

安全规则：
- API Key 绝不写入元数据 JSON、manifest、JSONL、日志或报错信息
- 所有显示使用 mask_api_key() 脱敏
"""

import json
import re
from datetime import datetime
from pathlib import Path

CONNECTIONS_DIR = Path(__file__).parent / "data" / "dify_connections"
KEYRING_SERVICE = "langfuse-rag-eval.dify"


def _get_keyring():
    """获取 keyring 模块，不可用时返回 None。"""
    try:
        import keyring
        # 测试 keyring 是否真正可用
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _generate_profile_id(name: str) -> str:
    """生成唯一 profile_id。"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"
    slug = re.sub(r'[^\w\u4e00-\u9fff]', '_', name.strip())
    slug = re.sub(r'_+', '_', slug).strip('_')[:20]
    return f"dify_{timestamp}_{slug or 'unnamed'}"


def mask_api_key(key: str) -> str:
    """将 API Key 脱敏为前缀 + *** + 后4位。"""
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def create_connection_profile(
    profile_name: str,
    base_url: str,
    api_key: str,
    workflow_description: str = "",
    timeout_seconds: int = 60,
    request_interval_seconds: float = 1.0,
) -> dict:
    """创建新的连接配置。

    元数据保存到 JSON 文件，API Key 保存到 keyring。

    Returns:
        dict: 配置元数据（不含 API Key）
    """
    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    profile_id = _generate_profile_id(profile_name)

    metadata = {
        "profile_id": profile_id,
        "profile_name": profile_name,
        "base_url": base_url,
        "workflow_description": workflow_description,
        "timeout_seconds": timeout_seconds,
        "request_interval_seconds": request_interval_seconds,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # 保存元数据
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存 API Key 到 keyring
    _store_api_key(profile_id, api_key)

    return metadata


def load_connection_profile(profile_id: str) -> dict:
    """加载连接配置元数据。"""
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def list_connection_profiles() -> list:
    """列出所有连接配置。"""
    if not CONNECTIONS_DIR.exists():
        return []

    profiles = []
    for meta_path in sorted(CONNECTIONS_DIR.glob("*.json"), reverse=True):
        try:
            profile = json.loads(meta_path.read_text(encoding="utf-8"))
            profiles.append(profile)
        except (json.JSONDecodeError, IOError):
            continue
    return profiles


def update_connection_profile(
    profile_id: str,
    updates: dict,
    api_key: str = None,
    clear_key: bool = False,
) -> dict:
    """更新连接配置。

    Args:
        profile_id: 配置 ID
        updates: 要更新的元数据字段（profile_id 不可修改）
        api_key: 新的 API Key（None=不更新, ""=清除）
        clear_key: 是否清除已保存的 API Key

    Returns:
        dict: 更新后的元数据
    """
    metadata = load_connection_profile(profile_id)
    if metadata is None:
        raise ValueError(f"连接配置不存在: {profile_id}")

    # 保护 profile_id
    updates.pop("profile_id", None)
    updates.pop("created_at", None)
    metadata.update(updates)
    metadata["updated_at"] = datetime.now().isoformat()

    # 保存元数据
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 更新 API Key
    if clear_key:
        _delete_api_key(profile_id)
    elif api_key is not None:
        _store_api_key(profile_id, api_key)

    return metadata


def delete_connection_profile(profile_id: str) -> bool:
    """删除连接配置（元数据 + API Key）。"""
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    deleted = False

    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    _delete_api_key(profile_id)
    return deleted


def get_connection_api_key(profile_id: str) -> str:
    """从安全存储读取 API Key。仅在内存中使用，不序列化。"""
    return _read_api_key(profile_id)


def has_connection_api_key(profile_id: str) -> bool:
    """检查是否已保存 API Key。"""
    return _read_api_key(profile_id) is not None


# ========== 内部 keyring 操作 ==========

def _store_api_key(profile_id: str, api_key: str):
    """存储 API Key 到 keyring。"""
    if not api_key:
        return
    kr = _get_keyring()
    if kr:
        try:
            kr.set_password(KEYRING_SERVICE, profile_id, api_key)
            return
        except Exception:
            pass
    # Fallback: 文件存储（仅在 keyring 不可用时）
    _store_api_key_file(profile_id, api_key)


def _read_api_key(profile_id: str) -> str:
    """从 keyring 读取 API Key。"""
    kr = _get_keyring()
    if kr:
        try:
            key = kr.get_password(KEYRING_SERVICE, profile_id)
            if key:
                return key
        except Exception:
            pass
    # Fallback: 文件存储
    return _read_api_key_file(profile_id)


def _delete_api_key(profile_id: str):
    """从 keyring 删除 API Key。"""
    kr = _get_keyring()
    if kr:
        try:
            kr.delete_password(KEYRING_SERVICE, profile_id)
        except Exception:
            pass
    # Fallback: 文件存储
    _delete_api_key_file(profile_id)


# ========== Fallback 文件存储（仅 keyring 不可用时） ==========

_CRED_FILE = CONNECTIONS_DIR / ".credentials"


def _store_api_key_file(profile_id: str, api_key: str):
    """Fallback: 将 API Key 存储到加密文件。"""
    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    creds = _load_credentials_file()
    creds[profile_id] = api_key
    _save_credentials_file(creds)


def _read_api_key_file(profile_id: str) -> str:
    """Fallback: 从文件读取 API Key。"""
    creds = _load_credentials_file()
    return creds.get(profile_id)


def _delete_api_key_file(profile_id: str):
    """Fallback: 从文件删除 API Key。"""
    creds = _load_credentials_file()
    if profile_id in creds:
        del creds[profile_id]
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
