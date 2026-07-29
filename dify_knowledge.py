"""
Dify 知识库探索模块 — 只读 API 封装。

功能：
- 列出知识库（datasets）
- 列出文档（documents）
- 列出分块（segments），支持 status 过滤和分页
- 内容规范化 SHA-256 哈希（用于重复分块检测）
- 生成 chunk catalog snapshot
- 导出 JSON / CSV

安全规则：
- 仅包含 GET 请求，禁止任何写入、删除、编辑、启用/禁用操作
- API Key 仅作为参数传入，不写入文件、日志或返回值
"""

import csv
import hashlib
import io
import json

import requests


# ── HTTP 基础 ─────────────────────────────────────────────────


def _get(api_key: str, base_url: str, path: str, params: dict = None,
         timeout: int = 30) -> dict:
    """发起 GET 请求并返回 JSON 响应。

    Args:
        api_key: Dify 知识库 API Key（dataset- 开头，仅在内存中使用）
        base_url: Dify 知识库 API Base URL
        path: API 路径（如 /datasets）
        params: 查询参数
        timeout: 超时秒数

    Returns:
        dict: 解析后的 JSON 响应

    Raises:
        RuntimeError: 请求失败或响应异常（含明确的错误分类）
    """
    url = base_url.rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"请求超时 ({timeout}s): {path}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"连接失败: {path} — 请检查 Base URL 是否正确且服务可达") from exc

    if resp.status_code != 200:
        # 针对 401 给出分类提示
        if resp.status_code == 401:
            if not api_key:
                raise RuntimeError("缺少知识库 API Key（DIFY_DATASET_API_KEY 未设置）")
            if api_key.startswith("app-"):
                raise RuntimeError(
                    f"认证失败 (401): 当前使用的是应用 Key（app-...），"
                    f"知识库 API 需要 dataset- 开头的知识库专用 Key。"
                    f"请到 Dify 后台 → 知识库 → API 访问 获取。"
                )
            raise RuntimeError(
                f"认证失败 (401): 知识库 API Key 无效或已过期。"
                f"请检查 DIFY_DATASET_API_KEY 是否正确。"
            )
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"JSON 解析失败: {resp.text[:200]}") from exc


def check_connection(api_key: str, base_url: str, timeout: int = 10) -> tuple[bool, str]:
    """测试知识库 API 连接。

    Returns:
        (ok, message): ok=True 表示成功，message 为描述信息
    """
    if not api_key:
        return False, "缺少知识库 API Key（DIFY_DATASET_API_KEY 未设置）"
    if not api_key.startswith("dataset-"):
        return False, (
            f"Key 类型错误: 当前是 `{api_key[:8]}...`（应用 Key），"
            f"知识库 API 需要 dataset- 开头的知识库专用 Key"
        )
    try:
        datasets = list_datasets(api_key, base_url, timeout=timeout)
        count = len(datasets)
        return True, f"成功连接，发现 {count} 个知识库"
    except RuntimeError as exc:
        return False, str(exc)


# ── 知识库 API ────────────────────────────────────────────────


def list_datasets(api_key: str, base_url: str, timeout: int = 30) -> list[dict]:
    """列出所有知识库（datasets）。

    Returns:
        list[dict]: 每个 dict 至少包含 id, name, document_count, word_count
    """
    data = _get(api_key, base_url, "/datasets", params={"page": 1, "limit": 100},
                timeout=timeout)
    return data.get("data", [])


def list_documents(api_key: str, base_url: str, dataset_id: str,
                   page: int = 1, limit: int = 20,
                   timeout: int = 30) -> dict:
    """列出指定知识库中的文档（带分页）。

    Args:
        dataset_id: 知识库 ID
        page: 页码（从 1 开始）
        limit: 每页数量（最大 100）

    Returns:
        dict: {"data": list[dict], "has_more": bool, "total": int}
        每个 dict 至少包含 id, name, word_count, status, created_at
    """
    limit = max(1, min(limit, 100))
    data = _get(api_key, base_url, f"/datasets/{dataset_id}/documents",
                params={"page": page, "limit": limit}, timeout=timeout)
    docs = data.get("data", [])
    has_more = data.get("has_more", False)
    total = data.get("total", 0)
    return {"data": docs, "has_more": has_more, "total": total}


def list_segments(api_key: str, base_url: str, dataset_id: str,
                  document_id: str, page: int = 1, limit: int = 20,
                  status_filter: str = "completed",
                  timeout: int = 30) -> dict:
    """列出指定文档的分块（segments），支持状态过滤和分页。

    Args:
        dataset_id: 知识库 ID
        document_id: 文档 ID
        page: 页码（从 1 开始）
        limit: 每页数量（最大 100）
        status_filter: 状态过滤（completed / indexing / error / 全部 = ""）
        timeout: 超时秒数

    Returns:
        dict: {"data": list[dict], "has_more": bool, "total": int}
    """
    limit = max(1, min(limit, 100))
    params = {"page": page, "limit": limit}
    if status_filter:
        params["status"] = status_filter

    data = _get(
        api_key, base_url,
        f"/datasets/{dataset_id}/documents/{document_id}/segments",
        params=params, timeout=timeout,
    )
    segments = data.get("data", [])
    has_more = data.get("has_more", False)
    total = data.get("total", 0)
    return {"data": segments, "has_more": has_more, "total": total}


# ── 内容哈希与重复检测 ────────────────────────────────────────


def compute_content_hash(content: str) -> str:
    """对内容做规范化 SHA-256 哈希。

    规范化：strip 首尾空白，统一换行符为 \\n。
    用于检测重复分块和后续评测回退匹配。
    """
    if not content:
        return ""
    normalized = content.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_chunk_catalog(segments: list[dict],
                        dataset_id: str = "",
                        document_id: str = "") -> list[dict]:
    """将原始 segments 转为标准化 chunk catalog。

    每条记录包含：
    segment_id, position, document_id, content, index_node_id,
    index_node_hash, tokens, word_count, enabled, status, content_hash

    Args:
        segments: Dify API 返回的 segment 列表
        dataset_id: 知识库 ID（可选，记入每条记录）
        document_id: 文档 ID（可选，记入每条记录）
    """
    catalog = []
    for seg in segments:
        content = seg.get("content", "") or ""
        # position: Dify 返回的 position 字段，或从 index_node_id 推断
        position = seg.get("position", None)
        if position is None:
            # 尝试从 word_count 等字段推断，否则留空
            position = seg.get("index", "")

        entry = {
            "segment_id": seg.get("id", ""),
            "position": position,
            "document_id": document_id or seg.get("document_id", ""),
            "dataset_id": dataset_id,
            "content": content,
            "index_node_id": seg.get("index_node_id", ""),
            "index_node_hash": seg.get("index_node_hash", ""),
            "tokens": seg.get("tokens", 0),
            "word_count": seg.get("word_count", 0),
            "enabled": seg.get("enabled", True),
            "status": seg.get("status", ""),
            "content_hash": compute_content_hash(content),
        }
        catalog.append(entry)
    return catalog


def detect_duplicates(catalog: list[dict]) -> dict[str, list[dict]]:
    """按 content_hash 检测重复分块。

    Returns:
        dict: {content_hash: [entry, ...]}，仅包含出现 >1 次的 hash。
    """
    hash_groups: dict[str, list[dict]] = {}
    for entry in catalog:
        h = entry.get("content_hash", "")
        if not h:
            continue
        hash_groups.setdefault(h, []).append(entry)
    return {h: entries for h, entries in hash_groups.items() if len(entries) > 1}


# ── 导出 ─────────────────────────────────────────────────────


# 导出的列顺序（与需求文档一致）
_EXPORT_COLUMNS = [
    "segment_id", "position", "document_id", "dataset_id",
    "content", "index_node_id", "index_node_hash",
    "tokens", "word_count", "enabled", "status", "content_hash",
]


def export_catalog_json(catalog: list[dict]) -> str:
    """导出 chunk catalog 为 JSON 字符串（缩进，ensure_ascii=False）。"""
    return json.dumps(catalog, ensure_ascii=False, indent=2)


def export_catalog_csv(catalog: list[dict]) -> bytes:
    """导出 chunk catalog 为 CSV（UTF-8 with BOM，Excel 友好）。"""
    output = io.StringIO()
    output.write("﻿")  # BOM
    writer = csv.DictWriter(output, fieldnames=_EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for entry in catalog:
        writer.writerow(entry)
    return output.getvalue().encode("utf-8-sig")
