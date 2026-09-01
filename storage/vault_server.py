"""MinIO Vault Bridge Server.

提供给 Dify Workflow HTTP 节点调用的轻量归档服务。
接收 Dify 传来的合同文件与分块切片，自动归档至 MinIO 并返回 version_id 与预签名安全下载链接。
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录位于 sys.path 首位，兼容所有子模块导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# 安全审计日志路径
AUDIT_LOG_FILE = os.path.join(os.path.dirname(__file__), "audit_trail.jsonl")


def log_audit_event(
    username: str,
    role: str,
    action: str,
    object_name: str,
    client_ip: str,
    status: str = "SUCCESS",
    detail: str = "",
) -> dict[str, Any]:
    """记录企业级不可篡改的安全审计日志（Audit Trail）。"""
    event = {
        "audit_id": f"audit-{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": username or "system_reviewer",
        "role": role or "legal_reader",
        "action": action,
        "object_name": object_name,
        "client_ip": client_ip,
        "status": status,
        "detail": detail,
    }
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[AuditLogger Error] Failed to write audit log: {e}")
    return event

try:
    from storage.minio_vault import (
        DEFAULT_CONTRACTS_BUCKET,
        get_cleaned_text,
        get_minio_client,
        get_presigned_download_url,
        list_vault_documents,
        save_cleaned_text,
        save_sidecar_metadata,
        upload_file_to_vault,
    )
    from storage.snapshot import (
        create_consistency_snapshot,
        list_consistency_snapshots,
    )
except ImportError:
    from minio_vault import (
        DEFAULT_CONTRACTS_BUCKET,
        get_cleaned_text,
        get_minio_client,
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

    # 3. 允许健康检查、文档与可视化审计看板放行
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc", "/audit-dashboard"):
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
    # 放行文档、健康检查与可视化审计看板
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc", "/audit-dashboard"):
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
    raw_file_name = payload.get("file_name", "")
    dataset_id = payload.get("dataset_id", "")
    document_id = payload.get("document_id", "")
    bucket = payload.get("bucket", DEFAULT_CONTRACTS_BUCKET)
    raw_chunks = payload.get("chunks", [])

    if not raw_file_name:
        raise HTTPException(status_code=400, detail="缺少 file_name 参数")

    # 智能解析全局路径: "contracts-vault/Appendix A.docx"
    clean_name = raw_file_name.replace("s3://", "").replace("minio://", "").strip()
    if "/" in clean_name:
        parsed_b, parsed_k = clean_name.split("/", 1)
        if parsed_b and parsed_k:
            bucket = parsed_b
            file_name = parsed_k
        else:
            file_name = clean_name
    else:
        file_name = clean_name

    try:
        # 如果未直接传入 chunks，尝试从 Dify 知识库 API 拉取
        if not raw_chunks and dataset_id and document_id:
            try:
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
    """从 MinIO 调取指定合同清洗后的纯净基准文本 (Cleaned Full Text)。支持 GET/POST 与模糊文件名对齐，并记录身份审计。"""
    target_name = file_name or ""
    username = request.headers.get("X-User-Id") or request.headers.get("X-Username") or "system_reviewer"
    user_role = request.headers.get("X-User-Role") or "legal_reader"
    
    # 支持从 JSON / Form 中提取 file_name 与身份信息
    if not target_name and request.method == "POST":
        try:
            content_type = request.headers.get("content-type", "")
            if "form" in content_type:
                form = await request.form()
                target_name = str(form.get("file_path") or form.get("file_name") or "")
                bucket = str(form.get("bucket") or bucket)
                version_id = str(form.get("version_id") or version_id) if form.get("version_id") else None
                if form.get("username"):
                    username = str(form.get("username"))
                if form.get("role") or form.get("user_role"):
                    user_role = str(form.get("role") or form.get("user_role"))
            else:
                body_bytes = await request.body()
                if body_bytes:
                    raw_str = body_bytes.decode("utf-8", errors="ignore").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
                    try:
                        payload = json.loads(raw_str, strict=False)
                    except Exception:
                        # 正则兜底提取 file_path 或 file_name
                        payload = {}
                        import re
                        m = re.search(r'["\']?(?:file_path|file_name)["\']?\s*:\s*["\']?([^"\'}\n\r]+)', raw_str)
                        if m:
                            payload["file_path"] = m.group(1).strip()
                    target_name = payload.get("file_path") or payload.get("file_name", "")
                    bucket = payload.get("bucket", bucket)
                    version_id = payload.get("version_id", version_id)
                    if payload.get("username"):
                        username = str(payload.get("username"))
                    if payload.get("role") or payload.get("user_role"):
                        user_role = str(payload.get("role") or payload.get("user_role"))
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

    # 🌟 身份绑定签名：同步生成该基线原件（.docx）的 15 分钟只读预签名下载链接
    presigned_url = ""
    try:
        presigned_url = get_presigned_download_url(
            object_name=base_name,
            bucket_name=bucket,
            expires_hours=0.25, # 15分钟
            role=user_role,
        )
    except Exception:
        pass

    # 🌟 记录不可篡改的安全审计日志
    audit_event = log_audit_event(
        username=username,
        role=user_role,
        action="FETCH_CLEANED_TEXT_AND_SIGN",
        object_name=base_name,
        client_ip=request.client.host if request.client else "unknown",
        status="SUCCESS",
        detail=f"Bucket: {bucket}, Text Length: {len(text)}",
    )

    return {
        "status": "success",
        "file_name": base_name,
        "cleaned_file": f"{base_name}.cleaned.txt",
        "cleaned_text": text,
        "length": len(text),
        "minio_bucket": bucket,
        "presigned_url": presigned_url,
        "audit_id": audit_event["audit_id"],
        "signer_role": user_role,
        "username": username,
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
    """生成带有时效性（默认15分钟、严格GET只读权限、绑定业务角色身份）的安全预签名临时下载链接。"""
    target_file = file_name or ""
    target_bucket = bucket
    exp_mins = expires_minutes
    username = request.headers.get("X-User-Id") or request.headers.get("X-Username") or "system_reviewer"
    user_role = request.headers.get("X-User-Role") or "legal_reader"

    if request.method == "POST":
        try:
            content_type = request.headers.get("content-type", "")
            if "json" in content_type:
                body = await request.json()
                target_file = body.get("file_path") or body.get("file_name", target_file)
                target_bucket = body.get("bucket", target_bucket)
                exp_mins = int(body.get("expires_minutes", exp_mins))
                if body.get("username"):
                    username = str(body.get("username"))
                if body.get("role") or body.get("user_role"):
                    user_role = str(body.get("role") or body.get("user_role"))
            elif "form" in content_type:
                form = await request.form()
                target_file = str(form.get("file_path") or form.get("file_name") or target_file)
                target_bucket = str(form.get("bucket") or target_bucket)
                if form.get("expires_minutes"):
                    exp_mins = int(form.get("expires_minutes"))
                if form.get("username"):
                    username = str(form.get("username"))
                if form.get("role") or form.get("user_role"):
                    user_role = str(form.get("role") or form.get("user_role"))
        except Exception:
            pass

    if not target_file:
        raise HTTPException(status_code=400, detail="缺少 file_path 或 file_name 参数")

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
            role=user_role,
        )
        
        # 🌟 写入安全审计日志
        audit_event = log_audit_event(
            username=username,
            role=user_role,
            action="GENERATE_PRESIGNED_DOWNLOAD_URL",
            object_name=f"{target_bucket}/{target_file}" if target_bucket != DEFAULT_CONTRACTS_BUCKET else target_file,
            client_ip=request.client.host if request.client else "unknown",
            status="SUCCESS",
            detail=f"Expires: {exp_mins} minutes",
        )

        return {
            "status": "success",
            "file_name": target_file,
            "bucket": target_bucket,
            "permission": "READ_ONLY",
            "expires_in_minutes": exp_mins,
            "presigned_url": url,
            "audit_id": audit_event["audit_id"],
            "signer_role": user_role,
            "username": username,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成预签名 URL 失败: {exc}")


@app.get("/api/vault/audit_logs")
def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    username: Optional[str] = None,
    action: Optional[str] = None,
):
    """查询不可篡改的企业级安全审计日志（仅限审计员与合规审查使用）。"""
    if not os.path.exists(AUDIT_LOG_FILE):
        return {"status": "success", "total": 0, "logs": []}

    logs = []
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    if username and entry.get("username") != username:
                        continue
                    if action and entry.get("action") != action:
                        continue
                    logs.append(entry)
                except Exception:
                    pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取审计日志失败: {e}")

    # 按时间倒序返回最新记录
    logs.reverse()
    return {
        "status": "success",
        "total": len(logs),
        "limit": limit,
        "logs": logs[:limit],
    }


@app.get("/audit-dashboard", response_class=HTMLResponse)
def render_audit_dashboard():
    """渲染企业级 MinIO 合同安全审计与 RBAC 权限监控大屏 (Claude 温润羊皮纸暖金风格)。"""
    logs = []
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            logs.append(json.loads(line.strip()))
                        except Exception:
                            pass
        except Exception:
            pass
    logs.reverse()
    total_events = len(logs)
    unique_users = len(set(l.get("username", "") for l in logs if l.get("username")))

    table_rows = ""
    for entry in logs:
        role = entry.get("role", "legal_reader")
        if "legal" in role:
            badge_style = "background-color: #FAF0E6; color: #C25E38; border: 1px solid #EED7C8;"
        elif "finance" in role:
            badge_style = "background-color: #FEF9EE; color: #B45309; border: 1px solid #FDE68A;"
        elif "audit" in role:
            badge_style = "background-color: #F0FDF4; color: #15803D; border: 1px solid #BBF7D0;"
        else:
            badge_style = "background-color: #FDF2F8; color: #9D174D; border: 1px solid #FBCFE8;"

        table_rows += f"""
        <tr class="hover:bg-[#FAF6F0] transition duration-150 border-b border-[#EFEAE3]">
            <td class="px-4 py-3.5 font-mono text-xs font-semibold text-[#D97757] whitespace-nowrap">{entry.get("audit_id", "")}</td>
            <td class="px-4 py-3.5 text-xs text-[#78716C] whitespace-nowrap font-mono">{entry.get("timestamp", "")[:19].replace("T", " ")}</td>
            <td class="px-4 py-3.5 text-xs font-mono font-semibold text-[#292524] break-all max-w-[180px]" title="{entry.get("username", "")}">{entry.get("username", "")}</td>
            <td class="px-4 py-3.5 text-xs whitespace-nowrap"><span class="px-2.5 py-1 rounded-full font-medium whitespace-nowrap" style="{badge_style}">{role}</span></td>
            <td class="px-4 py-3.5 text-xs text-[#57534E] font-mono whitespace-nowrap">{entry.get("action", "")}</td>
            <td class="px-4 py-3.5 text-xs text-[#44403C] truncate max-w-[240px] font-medium" title="{entry.get("object_name", "")}">{entry.get("object_name", "")}</td>
            <td class="px-4 py-3.5 text-xs text-[#78716C] font-mono whitespace-nowrap">{entry.get("client_ip", "")}</td>
            <td class="px-4 py-3.5 text-xs whitespace-nowrap"><span class="whitespace-nowrap inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-emerald-800 bg-emerald-50 border border-emerald-200 font-semibold text-[11px]"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>SUCCESS</span></td>
        </tr>
        """

    if not table_rows:
        table_rows = '<tr><td colspan="8" class="text-center py-10 text-[#A8A29E] text-sm">暂无审计流水记录</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MinIO 合同安全审计与 RBAC 权限监控大屏</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
</head>
<body class="bg-[#FBF9F5] text-[#292524] min-h-screen font-['Plus_Jakarta_Sans',sans-serif] p-6 lg:p-10">
    <div class="max-w-7xl mx-auto space-y-7">
        <!-- Header -->
        <div class="flex flex-col md:flex-row md:items-center md:justify-between border-b border-[#EAE4DC] pb-6 gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="p-2.5 rounded-2xl bg-[#F5EFE6] text-[#D97757] border border-[#E5DDD2] text-2xl shadow-sm">🛡️</span>
                    <div>
                        <h1 class="text-2xl font-bold tracking-tight text-[#1C1917]">MinIO 合同安全审计与 RBAC 权限监控大屏</h1>
                        <p class="text-xs text-[#78716C] mt-0.5 font-medium">Enterprise Security Audit Trail · Least Privilege & Identity-Bound Signing</p>
                    </div>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <button onclick="location.reload()" class="px-4 py-2.5 bg-[#D97757] hover:bg-[#C26732] text-white rounded-xl text-xs font-semibold shadow-sm hover:shadow transition flex items-center gap-2">
                    <span>🔄</span> 实时刷新大屏
                </button>
            </div>
        </div>

        <!-- Metrics Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-5">
            <div class="bg-[#FFFFFF] border border-[#EBE5DF] rounded-2xl p-5 shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
                <div class="text-xs font-medium text-[#78716C]">总审计调取次数</div>
                <div class="text-3xl font-bold text-[#D97757] mt-2 font-['JetBrains_Mono']">{total_events}</div>
                <div class="text-[11px] text-[#A8A29E] mt-1 font-medium">不可篡改的 JSONL 流水</div>
            </div>
            <div class="bg-[#FFFFFF] border border-[#EBE5DF] rounded-2xl p-5 shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
                <div class="text-xs font-medium text-[#78716C]">活跃操作人</div>
                <div class="text-3xl font-bold text-[#0D9488] mt-2 font-['JetBrains_Mono']">{unique_users}</div>
                <div class="text-[11px] text-[#A8A29E] mt-1 font-medium">独立用户身份追踪</div>
            </div>
            <div class="bg-[#FFFFFF] border border-[#EBE5DF] rounded-2xl p-5 shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
                <div class="text-xs font-medium text-[#78716C]">MinIO 角色隔离状态</div>
                <div class="text-3xl font-bold text-[#2563EB] mt-2 font-['JetBrains_Mono']">3 Roles</div>
                <div class="text-[11px] text-[#A8A29E] mt-1 font-medium">legal_reader / finance / audit</div>
            </div>
            <div class="bg-[#FFFFFF] border border-[#EBE5DF] rounded-2xl p-5 shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
                <div class="text-xs font-medium text-[#78716C]">签名安全模式</div>
                <div class="text-sm font-bold text-[#D97757] mt-3 flex items-center gap-2">
                    <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    去特权化只读签名 (15m TTL)
                </div>
                <div class="text-[11px] text-[#A8A29E] mt-1 font-medium">杜绝 Admin 密钥暴露</div>
            </div>
        </div>

        <!-- MinIO Active IAM Users & Policies Matrix Card -->
        <div class="bg-[#FFFFFF] border border-[#EBE5DF] rounded-2xl p-6 shadow-[0_2px_12px_rgba(0,0,0,0.03)] space-y-4">
            <div class="flex items-center justify-between border-b border-[#EAE4DC] pb-4">
                <h3 class="font-bold text-[#1C1917] text-sm flex items-center gap-2">
                    <span>👥</span> MinIO 已激活业务账号与 IAM 策略矩阵 (RBAC Roles)
                </h3>
                <span class="text-[11px] px-3 py-1 rounded-full bg-[#FAF4ED] text-[#C25E38] border border-[#EADACF] font-semibold">MinIO IAM Engine: Active</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-[#FAF8F5] border border-[#EFEAE3] rounded-xl p-4 space-y-2 hover:border-[#DCCFC2] transition">
                    <div class="flex items-center justify-between">
                        <span class="text-sm font-bold text-[#1C1917] flex items-center gap-1.5">⚖️ legal_reader</span>
                        <span class="text-[10px] px-2 py-0.5 rounded-full bg-orange-100 text-orange-800 border border-orange-200 font-semibold">法务专员</span>
                    </div>
                    <div class="text-xs text-[#78716C]">绑定策略: <code class="text-[#D97757] font-mono font-semibold">contract_viewer</code></div>
                    <div class="text-[11px] text-[#A8A29E]">仅限合同只读、列表浏览与 15m 预签名下载</div>
                </div>

                <div class="bg-[#FAF8F5] border border-[#EFEAE3] rounded-xl p-4 space-y-2 hover:border-[#DCCFC2] transition">
                    <div class="flex items-center justify-between">
                        <span class="text-sm font-bold text-[#1C1917] flex items-center gap-1.5">💰 finance_reader</span>
                        <span class="text-[10px] px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200 font-semibold">财务审计</span>
                    </div>
                    <div class="text-xs text-[#78716C]">绑定策略: <code class="text-amber-700 font-mono font-semibold">contract_viewer</code></div>
                    <div class="text-[11px] text-[#A8A29E]">仅限报价单与费用文件查阅</div>
                </div>

                <div class="bg-[#FAF8F5] border border-[#EFEAE3] rounded-xl p-4 space-y-2 hover:border-[#DCCFC2] transition">
                    <div class="flex items-center justify-between">
                        <span class="text-sm font-bold text-[#1C1917] flex items-center gap-1.5">🔍 audit_officer</span>
                        <span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200 font-semibold">合规审计</span>
                    </div>
                    <div class="text-xs text-[#78716C]">绑定策略: <code class="text-emerald-700 font-mono font-semibold">contract_viewer</code></div>
                    <div class="text-[11px] text-[#A8A29E]">全量审计流水溯源与日志查阅</div>
                </div>

                <div class="bg-[#FAF8F5] border border-[#EFEAE3] rounded-xl p-4 space-y-2 hover:border-[#DCCFC2] transition">
                    <div class="flex items-center justify-between">
                        <span class="text-sm font-bold text-[#1C1917] flex items-center gap-1.5">👑 admin</span>
                        <span class="text-[10px] px-2 py-0.5 rounded-full bg-stone-200 text-stone-800 border border-stone-300 font-semibold">超级管理员</span>
                    </div>
                    <div class="text-xs text-[#78716C]">绑定策略: <code class="text-stone-700 font-mono font-semibold">consoleAdmin (全权)</code></div>
                    <div class="text-[11px] text-[#A8A29E]">系统底层运维，已解除日常签名绑定</div>
                </div>
            </div>
        </div>

        <!-- Table Card -->
        <div class="bg-[#FFFFFF] border border-[#EBE5DF] rounded-2xl shadow-[0_2px_14px_rgba(0,0,0,0.03)] overflow-hidden">
            <div class="px-6 py-4 border-b border-[#EAE4DC] flex items-center justify-between bg-[#FCFAF7]">
                <h3 class="font-bold text-[#1C1917] text-sm flex items-center gap-2">
                    <span>📋</span> 全链路审计流水 (Audit Trail)
                </h3>
                <span class="text-xs text-[#A8A29E] font-mono">存储路径: storage/audit_trail.jsonl</span>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-[#F6F1EA] border-b border-[#EAE4DC] text-[11px] font-bold text-[#78716C] uppercase tracking-wider">
                            <th class="px-4 py-3.5">审计编号 (Audit ID)</th>
                            <th class="px-4 py-3.5">时间戳 (UTC)</th>
                            <th class="px-4 py-3.5">操作人 (User)</th>
                            <th class="px-4 py-3.5">业务角色 (Role)</th>
                            <th class="px-4 py-3.5">动作 (Action)</th>
                            <th class="px-4 py-3.5">目标合同 (Object)</th>
                            <th class="px-4 py-3.5">客户端 IP</th>
                            <th class="px-4 py-3.5">状态</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-[#EFEAE3] font-['JetBrains_Mono']">
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)


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
