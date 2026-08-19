"""Storage package for MinIO object storage and vault bridge."""

from .minio_vault import (
    DEFAULT_CONTRACTS_BUCKET,
    ensure_bucket,
    get_minio_client,
    get_presigned_download_url,
    get_vault_file_bytes,
    list_vault_documents,
    upload_file_to_vault,
)
from .vault_server import (
    app as vault_app,
    is_vault_server_running,
    start_vault_server_background,
)

__all__ = [
    "DEFAULT_CONTRACTS_BUCKET",
    "ensure_bucket",
    "get_minio_client",
    "get_presigned_download_url",
    "get_vault_file_bytes",
    "list_vault_documents",
    "upload_file_to_vault",
    "vault_app",
    "is_vault_server_running",
    "start_vault_server_background",
]
