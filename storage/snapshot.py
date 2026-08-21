"""一致性快照与版本回滚管理模块 (Consistency Snapshot & Rollback).

建立合同原件版本 (MinIO VersionId)、知识库索引 (Dify Document ID)、
以及结构化元数据 (Metadata) 的强一致性绑定与快照归档机制。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 快照持久化目录
SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"
SNAPSHOTS_FILE = SNAPSHOTS_DIR / "consistency_manifest.jsonl"


def _ensure_storage_dir():
    """确保快照存储目录存在。"""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_consistency_snapshot(
    file_name: str,
    content_hash: str,
    minio_version_id: str,
    dify_dataset_id: str,
    dify_document_id: str,
    metadata: dict[str, Any],
    contract_package: str = "baseline_2_4",
    minio_bucket: str = "contracts-vault",
    indexing_status: str = "completed",
    custom_note: str = "",
) -> dict[str, Any]:
    """创建并固化一份三位一体的一致性快照记录。

    Args:
        file_name: 合同文件名
        content_hash: 文件内容哈希 (SHA256/MD5)
        minio_version_id: MinIO 中的唯一版本号
        dify_dataset_id: Dify 知识库 ID
        dify_document_id: Dify 文档 ID
        metadata: 提取并绑定的元数据字典 (title, type, topics, summary 等)
        contract_package: 合同包分类
        minio_bucket: MinIO 桶名称
        indexing_status: 索引状态
        custom_note: 自定义版本备注

    Returns:
        dict: 生成的完整快照字典
    """
    _ensure_storage_dir()
    now_iso = datetime.now().astimezone().isoformat()
    ts_short = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"snap_{ts_short}_{content_hash[:8] if content_hash else 'auto'}"

    snapshot = {
        "snapshot_id": snapshot_id,
        "file_name": file_name,
        "content_hash": content_hash or "",
        "created_at": now_iso,
        "contract_package": contract_package or "baseline_2_4",
        "storage": {
            "bucket": minio_bucket,
            "object_name": file_name,
            "version_id": minio_version_id or "latest",
        },
        "knowledge_base": {
            "dataset_id": dify_dataset_id or "",
            "document_id": dify_document_id or "",
            "indexing_status": indexing_status or "completed",
        },
        "metadata": metadata or {},
        "custom_note": custom_note,
    }

    # 1. 追加写入全局 JSONL 台账
    with open(SNAPSHOTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    # 2. 同步写入 MinIO 伴生元数据文件 (.metadata.json)
    try:
        from storage.minio_vault import save_sidecar_metadata
        save_sidecar_metadata(file_name, snapshot, bucket_name=minio_bucket)
    except Exception:
        pass

    return snapshot


def list_consistency_snapshots(
    dataset_id: str = None,
    file_name: str = None,
) -> list[dict[str, Any]]:
    """查询一致性快照列表（最新快照排在前面）。"""
    _ensure_storage_dir()
    if not SNAPSHOTS_FILE.exists():
        return []

    snapshots = []
    with open(SNAPSHOTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if dataset_id and item.get("knowledge_base", {}).get("dataset_id") != dataset_id:
                    continue
                if file_name and item.get("file_name") != file_name:
                    continue
                snapshots.append(item)
            except Exception:
                continue

    snapshots.reverse()
    return snapshots


def get_snapshot_by_id(snapshot_id: str) -> dict[str, Any] | None:
    """根据 Snapshot ID 精准获取快照详情。"""
    snapshots = list_consistency_snapshots()
    for snap in snapshots:
        if snap.get("snapshot_id") == snapshot_id:
            return snap
    return None
