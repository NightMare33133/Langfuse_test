"""
Dify 知识库 API 连接配置管理 — 独立于批量提问的 App API 配置。

连接配置存储：
- 元数据 JSON：data/dify_kb_connections/<profile_id>.json（不含 API Key）
- API Key：通过 keyring 存储（Windows Credential Manager / macOS Keychain / Linux Secret Service）

安全规则：
- API Key 绝不写入元数据 JSON、manifest、JSONL、日志或报错信息
- 所有显示使用 mask_api_key() 脱敏
- 仅接受 dataset- 开头的知识库专用 Key
"""

import json
import re
from datetime import datetime
from pathlib import Path

CONNECTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "dify_kb_connections"
KEYRING_SERVICE = "langfuse-rag-eval.dify-kb"

VALID_PREFIX = "dataset-"
INVALID_PREFIXES = {
    "app-": "这是应用 API Key（app-...），不能用于知识库探索。"
           "请使用 dataset- 开头的知识库专用 Key。",
}


def _get_keyring():
    """获取 keyring 模块，不可用时返回 None。"""
    try:
        import keyring
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _generate_profile_id(name: str) -> str:
    """生成唯一 profile_id。"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"
    slug = re.sub(r'[^\w一-鿿]', '_', name.strip())
    slug = re.sub(r'_+', '_', slug).strip('_')[:20]
    return f"kb_{timestamp}_{slug or 'unnamed'}"


def mask_api_key(key: str) -> str:
    """将 API Key 脱敏为前缀 + *** + 后4位。"""
    if not key or len(key) < 12:
        return "***"
    return f"{key[:10]}...{key[-4:]}"


def validate_dataset_key(api_key: str) -> tuple[bool, str]:
    """校验是否为合法的 dataset- 前缀 Key。

    Returns:
        (ok, error_message): ok=True 表示合法，error_message 为空
    """
    if not api_key:
        return False, "缺少知识库 API Key（DIFY_DATASET_API_KEY 未设置）"
    for prefix, hint in INVALID_PREFIXES.items():
        if api_key.startswith(prefix):
            return False, hint
    if not api_key.startswith(VALID_PREFIX):
        return False, (
            f"Key 前缀不正确: `{api_key[:10]}...`，"
            f"知识库 API 需要 {VALID_PREFIX} 开头的专用 Key。"
            f"请到 Dify 后台 → 知识库 → API 访问 获取。"
        )
    return True, ""


def create_kb_profile(
    profile_name: str,
    base_url: str,
    api_key: str,
) -> dict:
    """创建新的知识库连接配置。

    元数据保存到 JSON 文件，API Key 保存到 keyring。

    Args:
        profile_name: 配置名称
        base_url: Dify API Base URL
        api_key: Dataset API Key（必须 dataset- 开头）

    Returns:
        dict: 配置元数据（不含 API Key）

    Raises:
        ValueError: Key 前缀不合法
    """
    ok, err = validate_dataset_key(api_key)
    if not ok:
        raise ValueError(err)

    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    profile_id = _generate_profile_id(profile_name)

    metadata = {
        "profile_id": profile_id,
        "profile_name": profile_name,
        "base_url": base_url,
        "key_masked": mask_api_key(api_key),
        "key_source": "keyring",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # 保存元数据（不含 API Key）
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存 API Key 到 keyring
    _store_api_key(profile_id, api_key)

    return metadata


def create_kb_profile_from_env(
    profile_name: str,
    base_url: str,
    env_var_name: str = "DIFY_DATASET_API_KEY",
) -> dict:
    """从环境变量创建知识库连接配置（不保存 Key 明文）。

    元数据中记录 key_source="env:{env_var_name}"，
    运行时从 os.getenv 读取 Key。

    Args:
        profile_name: 配置名称
        base_url: Dify API Base URL
        env_var_name: 环境变量名

    Returns:
        dict: 配置元数据
    """
    import os
    api_key = os.getenv(env_var_name, "")
    if not api_key:
        raise ValueError(f"环境变量 {env_var_name} 未设置")

    ok, err = validate_dataset_key(api_key)
    if not ok:
        raise ValueError(f"环境变量 {env_var_name} 无效: {err}")

    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    profile_id = _generate_profile_id(profile_name)

    metadata = {
        "profile_id": profile_id,
        "profile_name": profile_name,
        "base_url": base_url,
        "key_masked": mask_api_key(api_key),
        "key_source": f"env:{env_var_name}",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # 仅保存元数据，不保存 Key 到 keyring（从环境变量读取）
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return metadata


def load_kb_profile(profile_id: str) -> dict:
    """加载知识库连接配置元数据。"""
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def list_kb_profiles() -> list:
    """列出所有知识库连接配置。"""
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


def update_kb_profile(
    profile_id: str,
    updates: dict,
    api_key: str = None,
    clear_key: bool = False,
) -> dict:
    """更新知识库连接配置。

    Args:
        profile_id: 配置 ID
        updates: 要更新的元数据字段（profile_id 不可修改）
        api_key: 新的 API Key（None=不更新, 必须 dataset- 开头）
        clear_key: 是否清除已保存的 API Key

    Returns:
        dict: 更新后的元数据

    Raises:
        ValueError: 配置不存在或 Key 不合法
    """
    metadata = load_kb_profile(profile_id)
    if metadata is None:
        raise ValueError(f"知识库连接配置不存在: {profile_id}")

    # 校验新 Key
    if api_key is not None and api_key:
        ok, err = validate_dataset_key(api_key)
        if not ok:
            raise ValueError(err)

    # 保护不可修改字段
    updates.pop("profile_id", None)
    updates.pop("created_at", None)
    metadata.update(updates)
    metadata["updated_at"] = datetime.now().isoformat()

    # 更新 Key
    if clear_key:
        _delete_api_key(profile_id)
        metadata["key_masked"] = ""
    elif api_key is not None and api_key:
        _store_api_key(profile_id, api_key)
        metadata["key_masked"] = mask_api_key(api_key)

    # 保存元数据
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return metadata


def delete_kb_profile(profile_id: str) -> bool:
    """删除知识库连接配置（元数据 + API Key）。"""
    meta_path = CONNECTIONS_DIR / f"{profile_id}.json"
    deleted = False

    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    _delete_api_key(profile_id)
    return deleted


def get_kb_api_key(profile_id: str) -> str:
    """从安全存储读取 API Key。仅在内存中使用，不序列化。

    支持两种来源：
    - keyring 存储的 Key（key_source="keyring" 或无 key_source）
    - 环境变量引用（key_source="env:VAR_NAME"）
    """
    import os
    metadata = load_kb_profile(profile_id)
    if metadata:
        key_source = metadata.get("key_source", "")
        if key_source.startswith("env:"):
            env_var = key_source[4:]
            return os.getenv(env_var, "")
    return _read_api_key(profile_id)


def has_kb_api_key(profile_id: str) -> bool:
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
    return _read_api_key_file(profile_id)


def _delete_api_key(profile_id: str):
    """从 keyring 删除 API Key。"""
    kr = _get_keyring()
    if kr:
        try:
            kr.delete_password(KEYRING_SERVICE, profile_id)
        except Exception:
            pass
    _delete_api_key_file(profile_id)


# ========== Fallback 文件存储 ==========

def _cred_file() -> Path:
    """返回凭据文件路径（动态计算，方便测试 mock）。"""
    return CONNECTIONS_DIR / ".credentials"


def _store_api_key_file(profile_id: str, api_key: str):
    """Fallback: 将 API Key 存储到文件。"""
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
    cf = _cred_file()
    if not cf.exists():
        return {}
    try:
        return json.loads(cf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return {}


def _save_credentials_file(creds: dict):
    """保存凭据文件。"""
    _cred_file().write_text(json.dumps(creds, ensure_ascii=False), encoding="utf-8")
