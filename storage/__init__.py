"""Storage package for MinIO object storage, vault bridge, and consistency snapshots."""

from .minio_vault import (
    DEFAULT_CONTRACTS_BUCKET,
    ensure_bucket,
    get_minio_client,
    get_presigned_download_url,
    get_vault_file_bytes,
    list_vault_documents,
    upload_file_to_vault,
    save_sidecar_metadata,
    get_sidecar_metadata,
    save_cleaned_text,
    get_cleaned_text,
)
from .snapshot import (
    SNAPSHOTS_DIR,
    SNAPSHOTS_FILE,
    create_consistency_snapshot,
    get_snapshot_by_id,
    list_consistency_snapshots,
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
    "create_consistency_snapshot",
    "list_consistency_snapshots",
    "get_snapshot_by_id",
    "save_cleaned_text",
    "get_cleaned_text",
    "SNAPSHOTS_DIR",
    "SNAPSHOTS_FILE",
]
