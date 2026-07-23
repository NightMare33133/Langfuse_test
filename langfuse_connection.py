"""
Langfuse 连接配置管理模块 — 命名连接配置 + 本地凭据存储。

对齐 dify_connection.py 的 CRUD 模式。
连接配置存储在 data/langfuse_connections.json，包含实际凭据。

安全规则：
- Key 绝不写入 run/config/batch/raw/processed/judged JSON、报告、CSV、HTML、日志或报错信息
- 所有显示使用 mask_public_key() / mask_secret_key() 脱敏
- data/langfuse_connections.json 已加入 .gitignore
"""

import json
import re
from datetime import datetime
from pathlib import Path

import requests as _requests

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CONNECTIONS_FILE = _DATA_DIR / "langfuse_connections.json"


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


def _read_all() -> dict:
    """读取全部配置，返回 {profile_id: profile_dict}。"""
    if not _CONNECTIONS_FILE.exists():
        return {}
    with _CONNECTIONS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_all(profiles: dict):
    """写入全部配置。"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _CONNECTIONS_FILE.open("w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def _generate_id(name: str) -> str:
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    us = f"{now.microsecond:06d}"
    slug = re.sub(r"[^\w\u4e00-\u9fff]", "_", name.strip())[:20]
    return f"lf_{ts}_{us}_{slug}"


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


# ─── CRUD ────────────────────────────────────────────────────────────────────


def list_profiles() -> list:
    """返回全部 profile 列表，按创建时间倒序。"""
    profiles = _read_all()
    return sorted(profiles.values(), key=lambda p: p.get("created_at", ""), reverse=True)


def load_profile(profile_id: str) -> dict | None:
    """加载单个 profile，不存在返回 None。"""
    return _read_all().get(profile_id)


def create_profile(display_name: str, host: str,
                   public_key: str, secret_key: str) -> dict:
    """新建配置。Key 存入本地 JSON。

    Returns:
        新建的 profile dict。

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

    profiles = _read_all()
    for p in profiles.values():
        if p.get("display_name") == display_name:
            raise ValueError(f"配置名称已存在: {display_name}")

    now = datetime.now().isoformat()
    pid = _generate_id(display_name)
    profile = {
        "profile_id": pid,
        "display_name": display_name,
        "host": host,
        "public_key": public_key.strip(),
        "secret_key": secret_key.strip(),
        "created_at": now,
        "updated_at": now,
    }
    profiles[pid] = profile
    _write_all(profiles)
    return profile


def update_profile(profile_id: str, display_name: str, host: str,
                   public_key: str = None, secret_key: str = None) -> dict:
    """编辑配置。

    Args:
        public_key: 新值（None=不更新）
        secret_key: 新值（None=不更新, ""=保持原值不修改）

    Raises:
        ValueError: 配置不存在、名称为空/重复、Host 非法。
    """
    profiles = _read_all()
    if profile_id not in profiles:
        raise ValueError(f"配置不存在: {profile_id}")

    if not display_name or not display_name.strip():
        raise ValueError("配置名称不能为空")
    display_name = display_name.strip()
    host = normalize_host(host)

    for pid, p in profiles.items():
        if pid != profile_id and p.get("display_name") == display_name:
            raise ValueError(f"配置名称已存在: {display_name}")

    p = profiles[profile_id]
    p["display_name"] = display_name
    p["host"] = host
    if public_key is not None:
        if not public_key.strip():
            raise ValueError("Public Key 不能为空")
        p["public_key"] = public_key.strip()
    if secret_key is not None and secret_key.strip():
        p["secret_key"] = secret_key.strip()
    p["updated_at"] = datetime.now().isoformat()
    _write_all(profiles)
    return p


def delete_profile(profile_id: str) -> bool:
    """删除配置（含本地凭据）。"""
    profiles = _read_all()
    if profile_id not in profiles:
        return False
    del profiles[profile_id]
    _write_all(profiles)
    return True


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
