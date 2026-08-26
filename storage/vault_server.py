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
        get_cleaned_text,
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
        get_cleaned_text,
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

# API 安全密钥配置（默认密钥，可通过环境变量 VAULT_API_TOKEN 覆盖）
VAULT_API_TOKEN = os.getenv("VAULT_API_TOKEN", "contract-vault-secret-key-2026")


def verify_api_key(request: Request):
    """校验外部请求是否携带合法的 API Token，拒绝未授权公网访问。"""
    # 1. 检查 Authorization: Bearer <TOKEN>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token == VAULT_API_TOKEN:
            return True

    # 2. 检查 X-API-Key: <TOKEN>
    api_key_header = request.headers.get("X-API-Key", "").strip()
    if api_key_header == VAULT_API_TOKEN:
        return True

    # 3. 允许健康检查放行
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return True

    raise HTTPException(
        status_code=401,
        detail="未授权访问 (Unauthorized)：缺少或无效的 API Key / Bearer 签名凭据，拒绝访问敏感合同资产库。",
    )


app = FastAPI(title="MinIO Contract Vault Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 放行文档与健康检查
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    
    # 强制校验 API Key
    try:
        verify_api_key(request)
    except HTTPException as exc:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "detail": exc.detail})
    
    return await call_next(request)


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


@app.api_route("/api/vault/get_cleaned_text", methods=["GET", "POST"])
async def fetch_cleaned_text(
    request: Request,
    file_name: Optional[str] = None,
    bucket: str = DEFAULT_CONTRACTS_BUCKET,
    version_id: Optional[str] = None,
):
    """从 MinIO 调取指定合同清洗后的纯净基准文本 (Cleaned Full Text)。支持 GET/POST 与模糊文件名对齐。"""
    target_name = file_name or ""
    
    # 支持从 JSON / Form 中提取 file_name
    if not target_name and request.method == "POST":
        try:
            content_type = request.headers.get("content-type", "")
            if "form" in content_type:
                form = await request.form()
                target_name = str(form.get("file_name") or "")
                bucket = str(form.get("bucket") or bucket)
                version_id = str(form.get("version_id") or version_id) if form.get("version_id") else None
            else:
                body_bytes = await request.body()
                if body_bytes:
                    payload = json.loads(body_bytes.decode("utf-8", errors="ignore"), strict=False)
                    target_name = payload.get("file_name", "")
                    bucket = payload.get("bucket", bucket)
                    version_id = payload.get("version_id", version_id)
        except Exception:
            pass

    if not target_name:
        raise HTTPException(status_code=400, detail="缺少 file_name 参数，请指定要拉取的基线合同文件名")

    # 智能解析可能的全局前缀: "s3://bucket/file" 或 "bucket/file"
    clean_target = target_name.replace("s3://", "").replace("minio://", "").strip()
    if "/" in clean_target:
        parsed_bucket, parsed_file = clean_target.split("/", 1)
        if parsed_bucket and parsed_file:
            bucket = parsed_bucket
            target_name = parsed_file

    # 去除可能自带的 .cleaned.txt 后缀以保持一致性
    base_name = target_name.replace(".cleaned.txt", "")

    # 1. 尝试直接获取
    text = get_cleaned_text(file_name=base_name, bucket_name=bucket, version_id=version_id)

    # 2. 若未直接命中，根据知识库 document_titles 智能别名与语义关键词对齐 MinIO 实际文件名
    if text is None:
        try:
            alias_rules = [
                (["通用条款", "附件a", "appendix a", "gt", "采购条款", "it采购通用条款"], "Appendix A - General Terms for IT Purchasing 210909 bilingual.docx"),
                (["230404", "tech platform通用", "项目通用"], "Appendix A - General Terms for IT Purchasing 230404 bilingual.docx"),
                (["偏离表", "deviation", "a1", "appendix a1"], "Appendix A1. Deviation form to ABC Car Corporations General Terms for IT Purchasing_0714 Final Updates.docx"),
                (["安全要求", "安全最低要求", "appendix b", "security", "信息安全", "it安全"], "Appendix B. Minimum Information and IT Security Requirements v2 _ch.docx"),
                (["数据处理", "dpa", "appendix c", "数据修改", "委托处理"], "Appendix C. DPA 数据修改协议.docx"),
                (["中台", "master agreement", "框架主协议", "汽车合同中台", "框架协议", "it采购框架"], "Master Agreement of 汽车合同中台项目 for IT Purchasing.docx"),
                (["tech platform", "tech", "项目框架", "framework agreement"], "Framework Agreement of TECH Platfrom Project.docx"),
                (["准则", "行为准则", "code of conduct", "合作伙伴行为准则"], "Appendix F. Code of Conduct.docx"),
            ]
            
            target_lower = base_name.lower()
            matched_file = None
            for keywords, actual_doc in alias_rules:
                if any(kw in target_lower for kw in keywords):
                    matched_file = actual_doc
                    break

            if matched_file:
                text = get_cleaned_text(file_name=matched_file, bucket_name=bucket, version_id=version_id)
                if text:
                    base_name = matched_file

            # 3. 若别名未命中，遍历全部 MinIO 文档做字符重合度模糊打分
            if text is None:
                docs = list_vault_documents(bucket_name=bucket)
                best_doc = None
                best_score = 0
                for d in docs:
                    obj = d.get("object_name", "")
                    pure_doc = obj.replace(".cleaned.txt", "")
                    overlap = len(set(target_lower) & set(pure_doc.lower()))
                    if overlap > best_score:
                        best_score = overlap
                        best_doc = pure_doc
                if best_doc and best_score >= 2:
                    text = get_cleaned_text(file_name=best_doc, bucket_name=bucket, version_id=version_id)
                    if text:
                        base_name = best_doc
        except Exception:
            pass

    if text is None:
        raise HTTPException(
            status_code=404,
            detail=f"在 MinIO 存储桶 [{bucket}] 中未找到基线文本 [{target_name}.cleaned.txt]"
        )

    # 同步生成该基线原件（.docx）的 15 分钟临时预签名下载链接
    presigned_url = ""
    try:
        presigned_url = get_presigned_download_url(
            object_name=base_name,
            bucket_name=bucket,
            expires_hours=0.25, # 15分钟
        )
    except Exception:
        pass

    return {
        "status": "success",
        "file_name": base_name,
        "cleaned_file": f"{base_name}.cleaned.txt",
        "cleaned_text": text,
        "length": len(text),
        "minio_bucket": bucket,
        "presigned_url": presigned_url,
    }


@app.get("/api/vault/documents")
def list_documents(bucket: str = DEFAULT_CONTRACTS_BUCKET):
    """获取 MinIO 中的文档列表。"""
    try:
        return list_vault_documents(bucket_name=bucket)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.api_route("/api/vault/presigned_url", methods=["GET", "POST"])
async def get_file_presigned_url(
    request: Request,
    file_name: Optional[str] = None,
    bucket: str = DEFAULT_CONTRACTS_BUCKET,
    expires_minutes: int = 15,
):
    """生成带有时效性（默认15分钟、严格GET只读权限）的安全预签名临时下载链接。"""
    target_file = file_name or ""
    target_bucket = bucket
    exp_mins = expires_minutes

    if request.method == "POST":
        try:
            content_type = request.headers.get("content-type", "")
            if "json" in content_type:
                body = await request.json()
                target_file = body.get("file_name", target_file)
                target_bucket = body.get("bucket", target_bucket)
                exp_mins = int(body.get("expires_minutes", exp_mins))
            elif "form" in content_type:
                form = await request.form()
                target_file = str(form.get("file_name") or target_file)
                target_bucket = str(form.get("bucket") or target_bucket)
                if form.get("expires_minutes"):
                    exp_mins = int(form.get("expires_minutes"))
        except Exception:
            pass

    if not target_file:
        raise HTTPException(status_code=400, detail="缺少 file_name 参数")

    # 智能解析可能的全局前缀: "s3://bucket/file" 或 "bucket/file"
    clean_target = target_file.replace("s3://", "").replace("minio://", "").strip()
    if "/" in clean_target:
        parsed_b, parsed_k = clean_target.split("/", 1)
        if parsed_b and parsed_k:
            target_bucket = parsed_b
            target_file = parsed_k

    try:
        url = get_presigned_download_url(
            object_name=target_file,
            bucket_name=target_bucket,
            expires_hours=max(exp_mins / 60.0, 0.1),
        )
        return {
            "status": "success",
            "file_name": target_file,
            "bucket": target_bucket,
            "permission": "READ_ONLY",
            "expires_in_minutes": exp_mins,
            "presigned_url": url,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成预签名 URL 失败: {exc}")


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
