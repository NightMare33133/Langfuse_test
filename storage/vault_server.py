"""MinIO Vault Bridge Server.

提供给 Dify Workflow HTTP 节点调用的轻量归档服务。
接收 Dify 传来的合同文件与分块切片，自动归档至 MinIO 并返回 version_id 与预签名安全下载链接。
"""

import json
import os
from typing import Any, Optional

import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .minio_vault import (
        DEFAULT_CONTRACTS_BUCKET,
        get_presigned_download_url,
        list_vault_documents,
        save_cleaned_text,
        save_sidecar_metadata,
        upload_file_to_vault,
    )
    from .snapshot import (
        create_consistency_snapshot,
        list_consistency_snapshots,
    )
except ImportError:
    from minio_vault import (
        DEFAULT_CONTRACTS_BUCKET,
        get_presigned_download_url,
        list_vault_documents,
        save_cleaned_text,
        save_sidecar_metadata,
        upload_file_to_vault,
    )
    from snapshot import (
        create_consistency_snapshot,
        list_consistency_snapshots,
    )

app = FastAPI(title="MinIO Contract Vault Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "minio-contract-vault"}


@app.post("/api/vault/upload")
async def archive_contract(
    file: UploadFile = File(...),
    bucket: str = Form(DEFAULT_CONTRACTS_BUCKET),
    contract_package: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
    document_id: Optional[str] = Form(None),
    metadata_json: Optional[str] = Form(None),
):
    """接收 Dify 上传的合同文件并归档至 MinIO 资产库。"""
    try:
        content = await file.read()
        metadata = {}
        if contract_package:
            metadata["contract_package"] = contract_package

        res = upload_file_to_vault(
            file_name=file.filename,
            file_data=content,
            bucket_name=bucket,
            metadata=metadata,
        )

        # 自动提取清洗后但未分块的纯净全文 (Cleaned Full Text) 并归档至 MinIO
        try:
            from generator.doc_parser import parse_document
            from storage.minio_vault import save_cleaned_text
            parsed = parse_document(file_bytes=content, file_name=file.filename)
            cleaned_text = parsed.get("text", "")
            if cleaned_text:
                save_cleaned_text(file_name=file.filename, cleaned_text=cleaned_text, bucket_name=bucket)
                res["cleaned_text_length"] = len(cleaned_text)
        except Exception as e:
            print(f"[VaultServer] Cleaned text extraction: {e}")

        # 解析传入的元数据
        meta_dict = {}
        if metadata_json:
            try:
                meta_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
            except Exception:
                pass

        # 记录一致性快照
        if dataset_id or document_id or meta_dict:
            create_consistency_snapshot(
                file_name=file.filename,
                content_hash=res.get("etag", ""),
                minio_version_id=res.get("version_id", ""),
                dify_dataset_id=dataset_id or "",
                dify_document_id=document_id or "",
                metadata=meta_dict,
                contract_package=contract_package or meta_dict.get("contract_package", "baseline_2_4"),
                minio_bucket=bucket,
            )

        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinIO 归档失败: {exc}")


@app.post("/api/vault/archive_chunks")
async def archive_chunks(payload: dict[str, Any] = Body(...)):
    """接收 Dify 知识库分块切片或自动拉取切片并归档至 MinIO 伴生文件。"""
    file_name = payload.get("file_name", "")
    dataset_id = payload.get("dataset_id", "")
    document_id = payload.get("document_id", "")
    bucket = payload.get("bucket", DEFAULT_CONTRACTS_BUCKET)
    raw_chunks = payload.get("chunks", [])

    if not file_name:
        raise HTTPException(status_code=400, detail="缺少 file_name 参数")

    try:
        # 如果未直接传入 chunks，尝试从 Dify 知识库 API 拉取
        if not raw_chunks and dataset_id and document_id:
            try:
                import os
                from connectors.dify_kb_connection import list_kb_profiles, load_kb_profile, get_kb_api_key
                profiles = list_kb_profiles()
                api_key = ""
                base_url = os.getenv("DIFY_DATASET_BASE_URL", "http://localhost/v1")
                if profiles:
                    p_id = profiles[0]["profile_id"]
                    prof = load_kb_profile(p_id)
                    api_key = get_kb_api_key(p_id)
                    base_url = prof.get("base_url", base_url)
                if not api_key:
                    api_key = os.getenv("DIFY_DATASET_API_KEY", "")

                from connectors.dify_knowledge import list_segments
                if api_key:
                    segments_res = list_segments(
                        api_key,
                        base_url,
                        dataset_id=dataset_id,
                        document_id=document_id,
                        limit=100,
                    )
                    raw_chunks = segments_res.get("data", [])
            except Exception as e:
                print(f"[VaultServer] Error fetching segments: {e}")

        chunks_data = {
            "file_name": file_name,
            "dataset_id": dataset_id,
            "document_id": document_id,
            "total_chunks": len(raw_chunks),
            "chunks": raw_chunks,
        }

        # 写入 MinIO 伴生切片文件: {file_name}.chunks.json
        import io
        from storage.minio_vault import get_minio_client
        cli = get_minio_client()
        chunks_bytes = json.dumps(chunks_data, ensure_ascii=False, indent=2).encode("utf-8")
        sidecar_name = f"{file_name}.chunks.json"
        
        cli.put_object(
            bucket_name=bucket,
            object_name=sidecar_name,
            data=io.BytesIO(chunks_bytes),
            length=len(chunks_bytes),
            content_type="application/json",
        )

        return {
            "status": "success",
            "file_name": file_name,
            "total_chunks": len(raw_chunks),
            "chunks_file": sidecar_name,
            "minio_bucket": bucket,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"切片归档失败: {exc}")


@app.post("/api/vault/archive_cleaned_text")
async def archive_cleaned_text(request: Request):
    """接收 Dify Pipeline 传来的清洗后纯净文本并归档至 MinIO (支持 JSON 与 Form-Data，兼容未转义换行符)。"""
    file_name = ""
    cleaned_text = ""
    bucket = DEFAULT_CONTRACTS_BUCKET

    content_type = request.headers.get("content-type", "")

    if "form" in content_type or "multipart" in content_type:
        try:
            form = await request.form()
            file_name = str(form.get("file_name") or "")
            cleaned_text = str(form.get("cleaned_text") or "")
            bucket = str(form.get("bucket") or DEFAULT_CONTRACTS_BUCKET)
        except Exception:
            pass
    else:
        # 优先直接解析 JSON；若因 Dify 原始文本插值导致控制字符错误，使用 strict=False 容错解析
        raw_bytes = await request.body()
        raw_str = raw_bytes.decode("utf-8", errors="ignore").strip()
        try:
            payload = json.loads(raw_str, strict=False)
            file_name = payload.get("file_name", "")
            cleaned_text = payload.get("cleaned_text", "")
            bucket = payload.get("bucket", DEFAULT_CONTRACTS_BUCKET)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"请求 Body 解析失败: {e}")

    if not file_name:
        raise HTTPException(status_code=400, detail="缺少 file_name 参数")

    try:
        res = save_cleaned_text(file_name=file_name, cleaned_text=cleaned_text, bucket_name=bucket)
        return {
            "status": "success",
            "file_name": file_name,
            "cleaned_file": f"{file_name}.cleaned.txt",
            "cleaned_text_length": len(cleaned_text or ""),
            "minio_bucket": bucket,
            "version_id": res.get("version_id", ""),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"清洗文本归档失败: {exc}")


@app.get("/api/vault/documents")
def list_documents(bucket: str = DEFAULT_CONTRACTS_BUCKET):
    """获取 MinIO 中的文档列表。"""
    try:
        return list_vault_documents(bucket_name=bucket)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


_server_thread = None


def is_vault_server_running(port: int = 8000) -> bool:
    """检查 Vault Server 是否在运行。"""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"User-Agent": "HealthChecker"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "ok"
    except Exception:
        pass
    return False


def start_vault_server_background(port: int = 8000) -> bool:
    """在后台线程中常驻启动 FastAPI 归档服务。"""
    global _server_thread
    if is_vault_server_running(port):
        return True

    import threading
    import time

    def run_server():
        cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        srv = uvicorn.Server(cfg)
        srv.run()

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    for _ in range(15):
        time.sleep(0.1)
        if is_vault_server_running(port):
            return True
    return False


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
