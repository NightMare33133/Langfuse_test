"""MinIO 对象存储工具模块。

提供合同原件的自动归档、多版本管理、安全预签名下载链接生成、
以及从 MinIO 资产库一键拉取文件回滚/重建索引的核心能力。
"""

from __future__ import annotations

import io
import json
import os
from datetime import timedelta
from typing import Any, BinaryIO

try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False
    Minio = None
    S3Error = Exception

# 默认配置（支持环境变量覆盖）
DEFAULT_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9005")
DEFAULT_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
DEFAULT_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
DEFAULT_MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes")
DEFAULT_CONTRACTS_BUCKET = os.getenv("MINIO_CONTRACTS_BUCKET", "contracts-vault")


def get_minio_client(
    endpoint: str = None,
    access_key: str = None,
    secret_key: str = None,
    secure: bool = None,
) -> Any:
    """获取 MinIO 客户端实例。"""
    if not HAS_MINIO:
        raise RuntimeError("未安装 minio 库，请运行: pip install minio")

    ep = endpoint or DEFAULT_MINIO_ENDPOINT
    ak = access_key or DEFAULT_MINIO_ACCESS_KEY
    sk = secret_key or DEFAULT_MINIO_SECRET_KEY
    sec = DEFAULT_MINIO_SECURE if secure is None else secure

    return Minio(
        ep,
        access_key=ak,
        secret_key=sk,
        secure=sec,
    )


def ensure_bucket(
    client: Any = None,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    enable_versioning: bool = True,
) -> bool:
    """确保目标 Bucket 存在并可选开启版本控制。"""
    cli = client or get_minio_client()
    try:
        if not cli.bucket_exists(bucket_name):
            cli.make_bucket(bucket_name)
        if enable_versioning:
            from minio.versioningconfig import VersioningConfig, ENABLED
            cli.set_bucket_versioning(bucket_name, VersioningConfig(ENABLED))
        return True
    except Exception:
        return False


def upload_file_to_vault(
    file_name: str,
    file_data: bytes | BinaryIO,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    metadata: dict[str, str] = None,
    client: Any = None,
) -> dict[str, Any]:
    """将合同原件上传并归档至 MinIO 资产库。

    Args:
        file_name: 文件名（作为 Object Key）
        file_data: 文件二进制 bytes 或文件流对象
        bucket_name: 桶名称（默认 contracts-vault）
        metadata: 自定义元数据字典（可选）
        client: MinIO 客户端实例（可选）

    Returns:
        dict: 包含 bucket, object_name, version_id, etag, size, presigned_url 的字典
    """
    cli = client or get_minio_client()
    ensure_bucket(cli, bucket_name, enable_versioning=True)

    if isinstance(file_data, bytes):
        length = len(file_data)
        stream = io.BytesIO(file_data)
    else:
        file_data.seek(0, os.SEEK_END)
        length = file_data.tell()
        file_data.seek(0)
        stream = file_data

    # 上传对象
    result = cli.put_object(
        bucket_name=bucket_name,
        object_name=file_name,
        data=stream,
        length=length,
        metadata=metadata or {},
    )

    version_id = getattr(result, "version_id", None) or ""
    etag = getattr(result, "etag", "") or ""

    # 生成 1 小时有效的预签名下载链接
    presigned_url = ""
    try:
        presigned_url = get_presigned_download_url(
            file_name,
            bucket_name=bucket_name,
            version_id=version_id if version_id else None,
            expires_hours=1,
            client=cli,
        )
    except Exception:
        pass

    return {
        "bucket": bucket_name,
        "object_name": file_name,
        "version_id": version_id,
        "etag": etag,
        "size": length,
        "presigned_url": presigned_url,
        "success": True,
    }


def get_presigned_download_url(
    object_name: str,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    version_id: str = None,
    expires_hours: int = 1,
    client: Any = None,
) -> str:
    """生成带有时效的安全预签名下载/预览链接。"""
    cli = client or get_minio_client()
    extra_query_params = {}
    if version_id:
        extra_query_params["versionId"] = version_id

    url = cli.presigned_get_object(
        bucket_name=bucket_name,
        object_name=object_name,
        expires=timedelta(hours=expires_hours),
        extra_query_params=extra_query_params,
    )
    return url


def list_vault_documents(
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    include_versions: bool = False,
    filter_sidecars: bool = True,
    client: Any = None,
) -> list[dict[str, Any]]:
    """列出 MinIO 资产库中的所有合同文档及元数据。"""
    cli = client or get_minio_client()
    if not cli.bucket_exists(bucket_name):
        return []

    objects = cli.list_objects(bucket_name, recursive=True, include_version=include_versions)
    results = []
    for obj in objects:
        if filter_sidecars and (
            obj.object_name.endswith(".metadata.json")
            or obj.object_name.endswith(".chunks.json")
            or obj.object_name.endswith(".cleaned.txt")
        ):
            continue
        url = ""
        try:
            url = get_presigned_download_url(
                obj.object_name,
                bucket_name=bucket_name,
                version_id=getattr(obj, "version_id", None),
                expires_hours=1,
                client=cli,
            )
        except Exception:
            pass

        # 转换为本地时区时间（中国标准时间 UTC+8）
        local_mtime = ""
        if obj.last_modified:
            try:
                local_mtime = obj.last_modified.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                local_mtime = obj.last_modified.isoformat()

        results.append({
            "bucket": bucket_name,
            "object_name": obj.object_name,
            "version_id": getattr(obj, "version_id", "") or "latest",
            "size": obj.size,
            "last_modified": local_mtime,
            "is_latest": getattr(obj, "is_latest", True),
            "presigned_url": url,
        })
    return results


def save_cleaned_text(
    file_name: str,
    cleaned_text: str,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    client: Any = None,
) -> dict[str, Any]:
    """在 MinIO 中保存数据清洗后但未分块的完整纯净文本（Cleaned Full Text）。

    用于后续文本 Diff 差异化对比与切片效果基准分析。
    """
    cli = client or get_minio_client()
    ensure_bucket(cli, bucket_name, enable_versioning=True)

    cleaned_obj_name = f"{file_name}.cleaned.txt"
    data_bytes = (cleaned_text or "").encode("utf-8")
    stream = io.BytesIO(data_bytes)

    res = cli.put_object(
        bucket_name=bucket_name,
        object_name=cleaned_obj_name,
        data=stream,
        length=len(data_bytes),
        content_type="text/plain; charset=utf-8",
    )

    return {
        "bucket": bucket_name,
        "object_name": cleaned_obj_name,
        "version_id": getattr(res, "version_id", "") or "",
        "length": len(data_bytes),
        "success": True,
    }


def get_cleaned_text(
    file_name: str,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    version_id: str = None,
    client: Any = None,
) -> str | None:
    """从 MinIO 读取清洗后的纯净文本（Cleaned Full Text）。"""
    cli = client or get_minio_client()
    cleaned_obj_name = f"{file_name}.cleaned.txt"
    try:
        data_bytes = get_vault_file_bytes(cleaned_obj_name, bucket_name=bucket_name, version_id=version_id, client=cli)
        return data_bytes.decode("utf-8")
    except Exception:
        return None


def save_sidecar_metadata(
    file_name: str,
    metadata: dict[str, Any],
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    client: Any = None,
) -> dict[str, Any]:
    """在 MinIO 中同步保存一份伴生元数据 JSON 文件 (Sidecar Metadata)。"""
    cli = client or get_minio_client()
    ensure_bucket(cli, bucket_name, enable_versioning=True)

    sidecar_name = f"{file_name}.metadata.json"
    data_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    stream = io.BytesIO(data_bytes)

    res = cli.put_object(
        bucket_name=bucket_name,
        object_name=sidecar_name,
        data=stream,
        length=len(data_bytes),
        content_type="application/json",
    )

    return {
        "bucket": bucket_name,
        "object_name": sidecar_name,
        "version_id": getattr(res, "version_id", "") or "",
        "success": True,
    }


def get_sidecar_metadata(
    file_name: str,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    version_id: str = None,
    client: Any = None,
) -> dict[str, Any] | None:
    """从 MinIO 获取某份合同的伴生元数据字典。"""
    cli = client or get_minio_client()
    sidecar_name = f"{file_name}.metadata.json"
    try:
        data_bytes = get_vault_file_bytes(sidecar_name, bucket_name=bucket_name, version_id=version_id, client=cli)
        return json.loads(data_bytes.decode("utf-8"))
    except Exception:
        return None


def get_vault_file_bytes(
    object_name: str,
    bucket_name: str = DEFAULT_CONTRACTS_BUCKET,
    version_id: str = None,
    client: Any = None,
) -> bytes:
    """从 MinIO 资产库下载指定版本文件的完整二进制 bytes。用于一键重新灌入知识库。"""
    cli = client or get_minio_client()
    response = cli.get_object(
        bucket_name,
        object_name,
        version_id=version_id if (version_id and version_id != "latest") else None,
    )
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
