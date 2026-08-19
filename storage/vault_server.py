"""MinIO Vault Bridge Server.

提供给 Dify Workflow HTTP 节点调用的轻量归档服务。
接收 Dify 传来的合同文件，自动归档至 MinIO 并返回 version_id 与预签名安全下载链接。
"""

import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from minio_vault import (
    DEFAULT_CONTRACTS_BUCKET,
    get_presigned_download_url,
    list_vault_documents,
    upload_file_to_vault,
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
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinIO 归档失败: {exc}")


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
    import json
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
    port = int(os.getenv("VAULT_SERVER_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
