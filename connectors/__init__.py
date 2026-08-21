"""Connectors package for external system integrations (Dify, Langfuse)."""

from .dify_connection import (
    create_connection_profile,
    delete_connection_profile,
    get_connection_api_key,
    list_connection_profiles,
    load_connection_profile,
    update_connection_profile,
)
from .dify_kb_connection import (
    create_kb_profile,
    delete_kb_profile,
    get_kb_api_key,
    list_kb_profiles,
    load_kb_profile,
    update_kb_profile,
)
from .dify_ingestion import (
    append_ingestion_record,
    build_ingestion_record,
    check_duplicate,
    compute_content_hash,
    ensure_required_metadata_fields,
    find_document_info_by_name,
    list_metadata_fields,
    load_ingestion_history,
    parse_auto_ingestion_outputs,
    run_auto_ingestion_workflow,
    upload_file,
)
from .dify_knowledge import (
    list_datasets,
    list_documents,
    list_segments,
    retrieve,
)
from .langfuse_connection import (
    create_profile as create_langfuse_profile,
    delete_profile as delete_langfuse_profile,
    get_profile_api_keys as get_langfuse_api_keys,
    list_profiles as list_langfuse_profiles,
    load_profile as load_langfuse_profile,
)
from .langfuse_project import (
    list_projects,
    load_project,
    register_project,
)
from .fetch_traces import (
    fetch_all,
    fetch_observations,
    fetch_traces,
)

__all__ = [
    "create_connection_profile",
    "load_connection_profile",
    "list_connection_profiles",
    "update_connection_profile",
    "delete_connection_profile",
    "get_connection_api_key",
    "create_kb_profile",
    "load_kb_profile",
    "list_kb_profiles",
    "update_kb_profile",
    "delete_kb_profile",
    "get_kb_api_key",
    "append_ingestion_record",
    "build_ingestion_record",
    "check_duplicate",
    "compute_content_hash",
    "ensure_required_metadata_fields",
    "find_document_info_by_name",
    "list_metadata_fields",
    "load_ingestion_history",
    "parse_auto_ingestion_outputs",
    "run_auto_ingestion_workflow",
    "upload_file",
    "list_datasets",
    "list_documents",
    "list_segments",
    "retrieve",
    "create_langfuse_profile",
    "delete_langfuse_profile",
    "get_langfuse_api_keys",
    "list_langfuse_profiles",
    "load_langfuse_profile",
    "list_projects",
    "load_project",
    "register_project",
    "fetch_all",
    "fetch_traces",
    "fetch_observations",
]
