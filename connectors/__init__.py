"""Connectors package for external system integrations (Dify, Langfuse)."""

from .dify_connection import (
    load_dify_connections,
    save_dify_connections,
    get_dify_connection,
    upsert_dify_connection,
    delete_dify_connection,
)
from .dify_kb_connection import (
    load_dify_kb_connections,
    save_dify_kb_connections,
    get_dify_kb_connection,
    upsert_dify_kb_connection,
    delete_dify_kb_connection,
)
from .dify_knowledge import (
    get_dataset_documents,
    get_document_segments,
    search_knowledge_segments,
)
from .dify_ingestion import (
    upload_file,
    run_auto_ingestion_workflow,
    load_ingestion_history,
    build_ingestion_record,
    append_ingestion_record,
    compute_content_hash,
    check_duplicate,
    list_metadata_fields,
    ensure_required_metadata_fields,
    parse_auto_ingestion_outputs,
    find_document_info_by_name,
)
from .langfuse_connection import (
    load_langfuse_connections,
    save_langfuse_connections,
    get_langfuse_connection,
    upsert_langfuse_connection,
    delete_langfuse_connection,
)
from .langfuse_project import (
    load_langfuse_projects,
    save_langfuse_projects,
    get_langfuse_project,
    upsert_langfuse_project,
    delete_langfuse_project,
)
from .fetch_traces import (
    fetch_all_traces,
    fetch_traces_page,
)
