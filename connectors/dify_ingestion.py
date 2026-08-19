"""
Dify 知识库材料入库模块 — 写入型 Dataset API 封装。

功能：
- 上传文件到 Dify（POST /files/upload）
- 调用已发布 Workflow 提取结构化 metadata（POST /workflows/run）
- 校验 Workflow 返回 schema
- 创建文档（POST /datasets/{id}/document/create-by-text）
- 查询 metadata 字段列表（GET /datasets/{id}/metadata）
- 绑定文档 metadata（POST /datasets/{id}/documents/metadata）
- SHA-256 重复检测
- JSONL 入库历史记录（不含 API Key）

安全规则：
- API Key 仅作为参数传入，不写入文件、日志或返回值
- 所有 Key 只从环境变量读取
- 历史记录、报错信息中绝不包含 API Key
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import requests

# ── 常量 ─────────────────────────────────────────────────────

INGESTION_HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "ingestion_history"

VALID_WORKFLOW_PACKAGES = {"baseline_2_4", "tech_platform_2_5"}

WORKFLOW_RESULT_FIELDS = {
    "contract_package",
    "document_type",
    "document_title",
    "document_language",
    "document_summary",
    "topics",
}

WORKFLOW_USER = "ingestion-user"


# ── Key 校验 ─────────────────────────────────────────────────


def validate_workflow_key(api_key: str) -> tuple[bool, str]:
    """校验是否为合法的 app- 前缀 Workflow Key。

    Returns:
        (ok, error_message): ok=True 表示合法
    """
    if not api_key:
        return False, "缺少 Workflow API Key（DIFY_WORKFLOW_API_KEY 未设置）"
    if api_key.startswith("dataset-"):
        return False, (
            "这是知识库 API Key（dataset-...），不能用于 Workflow。"
            "请使用 app- 开头的应用 Key。"
        )
    if not api_key.startswith("app-"):
        return False, (
            f"Key 前缀不正确: `{api_key[:10]}...`，"
            f"Workflow API 需要 app- 开头的应用 Key。"
        )
    return True, ""


def validate_dataset_key(api_key: str) -> tuple[bool, str]:
    """校验是否为合法的 dataset- 前缀知识库 Key。

    Returns:
        (ok, error_message): ok=True 表示合法
    """
    if not api_key:
        return False, "缺少知识库 API Key（DIFY_DATASET_API_KEY 未设置）"
    if api_key.startswith("app-"):
        return False, (
            "这是应用 API Key（app-...），不能用于知识库写入。"
            "请使用 dataset- 开头的知识库专用 Key。"
        )
    if not api_key.startswith("dataset-"):
        return False, (
            f"Key 前缀不正确: `{api_key[:10]}...`，"
            f"知识库 API 需要 dataset- 开头的专用 Key。"
        )
    return True, ""


# ── HTTP 基础 ─────────────────────────────────────────────────


def _post_json(api_key: str, base_url: str, path: str, body: dict,
               timeout: int = 120) -> dict:
    """发起 JSON POST 请求并返回 JSON 响应。

    Args:
        api_key: API Key（仅在内存中使用）
        base_url: API Base URL
        path: API 路径
        body: 请求体（JSON 序列化）
        timeout: 超时秒数

    Returns:
        dict: 解析后的 JSON 响应

    Raises:
        RuntimeError: 请求失败或响应异常
    """
    url = base_url.rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"请求超时 ({timeout}s): {path}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"连接失败: {path} — 请检查 Base URL 是否正确且服务可达"
        ) from exc

    if resp.status_code not in (200, 201):
        _raise_http_error(resp, api_key, path)

    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"JSON 解析失败: {resp.text[:200]}") from exc


def _guess_mime_type(file_path: str) -> str:
    """根据文件扩展名返回 MIME Type。

    覆盖 Dify 常见文档格式；未知扩展名回退到
    application/octet-stream，由 requests 自动处理。
    """
    import mimetypes
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime

    ext = Path(file_path).suffix.lower()
    _EXTRA_MIME_MAP = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls":  "application/vnd.ms-excel",
        ".doc":  "application/msword",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".ppt":  "application/vnd.ms-powerpoint",
        ".csv":  "text/csv",
        ".md":   "text/markdown",
        ".jsonl": "application/jsonl",
        ".txt":  "text/plain",
        ".pdf":  "application/pdf",
    }
    return _EXTRA_MIME_MAP.get(ext, "application/octet-stream")


def _post_file(api_key: str, base_url: str, path: str, file_path: str,
               user: str = WORKFLOW_USER, timeout: int = 120,
               filename_override: str = None) -> dict:
    """发起 multipart 文件上传请求。

    根据文件扩展名显式设置 Content-Type，确保 Dify 能正确识别
    .docx / .xlsx / .pdf 等格式。

    Args:
        api_key: API Key
        base_url: API Base URL
        path: API 路径（如 /files/upload）
        file_path: 本地文件路径
        user: 终端用户标识
        timeout: 超时秒数
        filename_override: 可选的覆盖文件名（保留原始文件名，避免临时文件名如 tmpXXX.docx）

    Returns:
        dict: 解析后的 JSON 响应
    """
    url = base_url.rstrip("/") + path
    headers = {"Authorization": f"Bearer {api_key}"}
    file_name = filename_override or Path(file_path).name
    mime_type = _guess_mime_type(file_name)
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_name, f, mime_type)}
            data = {"user": user}
            resp = requests.post(
                url, headers=headers, files=files, data=data, timeout=timeout
            )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"文件上传超时 ({timeout}s): {file_name}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"文件上传连接失败 — 请检查 Base URL 是否正确且服务可达"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"文件不存在: {file_path}") from exc

    if resp.status_code not in (200, 201):
        _raise_http_error(resp, api_key, f"文件上传 {file_name}")

    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"文件上传响应解析失败: {resp.text[:200]}") from exc


def _get_json(api_key: str, base_url: str, path: str,
              params: dict = None, timeout: int = 30) -> dict:
    """发起 GET 请求并返回 JSON 响应。"""
    url = base_url.rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(
            url, headers=headers, params=params, timeout=timeout
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"请求超时 ({timeout}s): {path}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"连接失败: {path} — 请检查 Base URL 是否正确且服务可达"
        ) from exc

    if resp.status_code != 200:
        _raise_http_error(resp, api_key, path)

    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"JSON 解析失败: {resp.text[:200]}") from exc


def _raise_http_error(resp, api_key: str, context: str):
    """根据 HTTP 状态码抛出分类错误信息（不泄露 Key）。"""
    if resp.status_code == 401:
        if not api_key:
            raise RuntimeError(f"认证失败 ({context}): API Key 未设置")
        key_hint = api_key[:8] + "..." if len(api_key) >= 8 else "***"
        raise RuntimeError(
            f"认证失败 (401, {context}): Key `{key_hint}` 无效或已过期"
        )
    if resp.status_code == 403:
        raise RuntimeError(f"权限不足 (403, {context}): 请检查 API Key 权限")
    if resp.status_code == 404:
        raise RuntimeError(f"资源不存在 (404, {context}): 请检查 ID 或端点")
    if resp.status_code == 413:
        raise RuntimeError(f"文件过大 (413, {context})")
    raise RuntimeError(f"HTTP {resp.status_code} ({context}): {resp.text[:500]}")


# ── 文本提取 ─────────────────────────────────────────────────


def extract_text_from_file(file_path: str) -> str:
    """从文件中提取纯文本内容。

    支持格式：
    - .docx：使用 docx2txt 提取
    - .txt / .md / 其他纯文本：直接读取

    Args:
        file_path: 本地文件路径

    Returns:
        str: 提取的文本内容

    Raises:
        RuntimeError: 不支持的格式或提取失败
    """
    path = Path(file_path)
    if not path.exists():
        raise RuntimeError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()

    if ext == ".docx":
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            if not text or not text.strip():
                raise RuntimeError(f"docx 文件提取为空: {path.name}")
            return text
        except ImportError:
            raise RuntimeError(
                "缺少 docx2txt 库，无法处理 .docx 文件。"
                "请运行: pip install docx2txt"
            )
        except Exception as exc:
            raise RuntimeError(f"docx 提取失败: {exc}") from exc

    if ext in (".txt", ".md", ".csv", ".json", ".jsonl", ".xml", ".html"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                raise RuntimeError(f"文件内容为空: {path.name}")
            return text
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"文件读取失败: {exc}") from exc

    # 尝试作为纯文本读取
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            return text
    except Exception:
        pass

    raise RuntimeError(
        f"不支持的文件格式: {ext}（{path.name}）。"
        f"支持的格式: .docx, .txt, .md, .csv, .json, .xml, .html"
    )


def upload_text_as_file(api_key: str, base_url: str,
                        text: str, original_name: str,
                        user: str = WORKFLOW_USER) -> str:
    """将文本写入临时 .txt 文件并上传到 Dify。

    保留原文件名的 stem，后缀改为 .txt。

    Args:
        api_key: Workflow API Key
        base_url: Workflow API Base URL
        text: 文本内容
        original_name: 原始文件名（用于生成有意义的上传文件名）
        user: 终端用户标识

    Returns:
        str: 文件 ID
    """
    import tempfile

    # 保留原文件名但改为 .txt 后缀
    stem = Path(original_name).stem
    upload_name = f"{stem}.txt"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        result = _post_file(api_key, base_url, "/files/upload", tmp_path, user,
                            filename_override=upload_name)
        file_id = result.get("id")
        if not file_id:
            raise RuntimeError(
                f"文件上传成功但未返回 ID: "
                f"{json.dumps(result, ensure_ascii=False)[:200]}"
            )
        return file_id
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── 文件上传 ─────────────────────────────────────────────────


def upload_file(api_key: str, base_url: str, file_path: str,
                user: str = WORKFLOW_USER,
                filename_override: str = None) -> str:
    """上传文件到 Dify 并返回 file_id。

    先尝试以原始格式直接上传（显式设置 MIME Type 并保留原始文件名）。
    仅当 Dify 返回 HTTP 415（unsupported_file_type）时，
    才兜底提取文本并以 .txt 格式重新上传。

    Args:
        api_key: Workflow API Key（app- 开头）
        base_url: Workflow API Base URL
        file_path: 本地文件路径
        user: 终端用户标识
        filename_override: 可选的覆盖文件名（保留原始真实文件名）

    Returns:
        str: 文件 ID（用于 Workflow 文件引用）

    Raises:
        RuntimeError: 上传失败或文件处理失败
    """
    path = Path(file_path)
    real_name = filename_override or path.name

    # 1) 先尝试直接上传原始文件
    try:
        result = _post_file(api_key, base_url, "/files/upload", file_path, user,
                            filename_override=real_name)
        file_id = result.get("id")
        if not file_id:
            raise RuntimeError(
                f"文件上传成功但未返回 ID: "
                f"{json.dumps(result, ensure_ascii=False)[:200]}"
            )
        return file_id
    except RuntimeError as exc:
        # 仅对 415 unsupported_file_type 做兜底
        if "415" not in str(exc) and "unsupported_file_type" not in str(exc):
            raise

    # 2) 415 兜底：提取文本后以 .txt 上传
    text = extract_text_from_file(file_path)
    return upload_text_as_file(api_key, base_url, text, real_name, user)


# ── Workflow 调用 ────────────────────────────────────────────


def run_workflow(api_key: str, base_url: str,
                 file_ids: list[str], contract_package: str,
                 user: str = WORKFLOW_USER) -> dict:
    """调用已发布的 Dify Workflow 提取 metadata。

    Args:
        api_key: Workflow API Key（app- 开头）
        base_url: Workflow API Base URL
        file_ids: 已上传文件的 ID 列表
        contract_package: 合同包名称（baseline_2_4 或 tech_platform_2_5）
        user: 终端用户标识

    Returns:
        dict: Workflow 输出（data.outputs）

    Raises:
        ValueError: contract_package 不合法
        RuntimeError: Workflow 执行失败
    """
    if contract_package not in VALID_WORKFLOW_PACKAGES:
        raise ValueError(
            f"不支持的合同包: {contract_package}，"
            f"可选值: {', '.join(sorted(VALID_WORKFLOW_PACKAGES))}"
        )

    # 构造文件引用列表
    files = [
        {"type": "document", "transfer_method": "local_file", "upload_file_id": fid}
        for fid in file_ids
    ]

    body = {
        "inputs": {
            "contract_package": contract_package,
            "files": files,
        },
        "response_mode": "blocking",
        "user": user,
    }

    result = _post_json(api_key, base_url, "/workflows/run", body)

    # 检查 Workflow 执行状态
    data = result.get("data", {})
    status = data.get("status", "")
    if status != "succeeded":
        error_msg = data.get("error") or "未知错误"
        raise RuntimeError(
            f"Workflow 执行失败 (status={status}): {error_msg}"
        )

    outputs = data.get("outputs")
    if outputs is None:
        raise RuntimeError("Workflow 执行成功但未返回 outputs")

    return outputs


def run_auto_ingestion_workflow(
    api_key: str,
    base_url: str,
    file_ids: list[str],
    contract_package: str,
    dataset_id: str = None,
    user: str = WORKFLOW_USER,
    timeout: int = 300,
) -> dict:
    """调用已发布的 Dify 全流程入库 Workflow（自动提取 metadata + 知识库 Pipeline 入库 + 绑定 metadata）。

    Args:
        api_key: Workflow API Key（app- 开头）
        base_url: Workflow API Base URL
        file_ids: 已上传到 Dify (/files/upload) 的文件 ID 列表
        contract_package: 合同包名称（baseline_2_4 或 tech_platform_2_5）
        dataset_id: 目标知识库 ID（全流程 Workflow Start 节点必填变量）
        user: 终端用户标识
        timeout: 超时秒数（默认 300s，因为包含完整入库流水线）

    Returns:
        dict: Workflow 输出（data.outputs）

    Raises:
        ValueError: contract_package 不合法或缺少 dataset_id
        RuntimeError: Workflow 执行失败
    """
    if contract_package not in VALID_WORKFLOW_PACKAGES:
        raise ValueError(
            f"不支持的合同包: {contract_package}，"
            f"可选值: {', '.join(sorted(VALID_WORKFLOW_PACKAGES))}"
        )

    if not dataset_id or not str(dataset_id).strip():
        raise ValueError("全流程入库 Workflow 必须指定目标知识库 dataset_id（必填入参）")

    # 构造标准 Dify 文件引用列表
    files = [
        {"type": "document", "transfer_method": "local_file", "upload_file_id": fid}
        for fid in file_ids
    ]

    inputs = {
        "contract_package": contract_package,
        "files": files,
        "dataset_id": str(dataset_id).strip(),
    }

    body = {
        "inputs": inputs,
        "response_mode": "blocking",
        "user": user,
    }

    result = _post_json(api_key, base_url, "/workflows/run", body, timeout=timeout)

    data = result.get("data", {})
    status = data.get("status", "")
    if status != "succeeded":
        error_msg = data.get("error") or "未知错误"
        raise RuntimeError(
            f"全流程入库 Workflow 执行失败 (status={status}): {error_msg}"
        )

    outputs = data.get("outputs")
    if outputs is None:
        raise RuntimeError("Workflow 执行成功但未返回 outputs")

    return outputs


def _normalize_auto_item(item, expected_package: str, default_name: str = "") -> dict:
    """规范化单个全流程入库结果项。"""
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except Exception:
            return {
                "file_name": default_name,
                "document_id": "",
                "contract_package": expected_package,
                "document_type": "",
                "document_title": "",
                "document_language": "",
                "document_summary": "",
                "topics": [],
                "indexing_status": "error",
                "batch": "",
                "success": False,
                "error": f"条目解析失败（非合法 JSON）: {item[:100]}",
                "raw": item,
            }

    if not isinstance(item, dict):
        return {
            "file_name": default_name,
            "document_id": "",
            "contract_package": expected_package,
            "document_type": "",
            "document_title": "",
            "document_language": "",
            "document_summary": "",
            "topics": [],
            "indexing_status": "error",
            "batch": "",
            "success": False,
            "error": f"条目不是字典类型: {type(item).__name__}",
            "raw": item,
        }

    # 提取基本字段
    doc_id = str(item.get("document_id") or item.get("doc_id") or item.get("id") or "").strip()
    file_name = str(item.get("file_name") or item.get("name") or item.get("filename") or default_name).strip()
    pkg = str(item.get("contract_package") or expected_package).strip()
    doc_type = str(item.get("document_type") or "").strip()
    doc_title = str(item.get("document_title") or file_name).strip()
    doc_lang = str(item.get("document_language") or "").strip()
    doc_summary = str(item.get("document_summary") or "").strip()
    batch = str(item.get("batch") or "").strip()
    indexing_status = str(item.get("indexing_status") or ("completed" if not doc_id else "waiting")).strip()
    error = str(item.get("error") or item.get("error_message") or "").strip()

    # 处理 topics
    raw_topics = item.get("topics", [])
    if isinstance(raw_topics, str):
        try:
            raw_topics = json.loads(raw_topics)
        except Exception:
            raw_topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
    if not isinstance(raw_topics, list):
        raw_topics = [str(raw_topics)]
    topics = [str(t).strip() for t in raw_topics if str(t).strip()]

    # 判定 success：
    # 1. 有显式 error 判定为失败
    # 2. 有 doc_id 判定为成功
    # 3. 无 doc_id 时：若有完整的结构化元数据（doc_type/doc_summary/topics），判定为成功（由下游按文件名关联 document_id）
    # 4. 无 doc_id 且无完整元数据，判定为失败
    if error:
        success = False
    elif not doc_id:
        if doc_type and doc_summary:
            success = True
            indexing_status = "completed"
        else:
            success = False
            error = "Workflow 未返回 document_id，入库可能未完成"
    else:
        success = True

    return {
        "file_name": file_name,
        "document_id": doc_id,
        "contract_package": pkg,
        "document_type": doc_type,
        "document_title": doc_title,
        "document_language": doc_lang,
        "document_summary": doc_summary,
        "topics": topics,
        "indexing_status": indexing_status,
        "batch": batch,
        "success": success,
        "error": error,
        "raw": item,
    }



def parse_auto_ingestion_outputs(
    outputs,
    expected_package: str,
    file_names: list[str] = None,
) -> list[dict]:
    """解析并规范化全流程入库 Workflow 的 outputs。

    兼容多种返回结构：
    - {"results": [...]}
    - {"output": [...]}
    - {"documents": [...]}
    - [...] (直接列表)
    - dict (单文件返回)
    - json string 格式包装

    Args:
        outputs: Workflow outputs
        expected_package: 期望的 contract_package
        file_names: 可选的上传文件名列表（用于按顺序补充缺失的文件名）

    Returns:
        list[dict]: 规范化后的结果列表，每项包含 file_name, document_id, success, error 等
    """
    if isinstance(outputs, str):
        try:
            outputs = json.loads(outputs)
        except Exception as exc:
            return [{
                "file_name": file_names[0] if file_names else "",
                "document_id": "",
                "contract_package": expected_package,
                "document_type": "",
                "document_title": "",
                "document_language": "",
                "document_summary": "",
                "topics": [],
                "indexing_status": "error",
                "batch": "",
                "success": False,
                "error": f"outputs JSON 解析失败: {exc}",
                "raw": outputs,
            }]

    raw_items = []
    if isinstance(outputs, list):
        raw_items = outputs
    elif isinstance(outputs, dict):
        for key in ("results", "output", "documents", "data", "items"):
            if key in outputs and isinstance(outputs[key], list):
                raw_items = outputs[key]
                break
        if not raw_items:
            # 如果 dict 本身包含 document_id 或 contract_package，视为单个结果
            if "document_id" in outputs or "doc_id" in outputs or "contract_package" in outputs:
                raw_items = [outputs]
            else:
                # 扫描 values 寻找 list
                for v in outputs.values():
                    if isinstance(v, list):
                        raw_items = v
                        break
                if not raw_items:
                    if file_names and len(file_names) > 0:
                        raw_items = [{} for _ in file_names]
                    else:
                        raw_items = [outputs]

    if not raw_items:
        return [{
            "file_name": file_names[0] if file_names else "",
            "document_id": "",
            "contract_package": expected_package,
            "document_type": "",
            "document_title": "",
            "document_language": "",
            "document_summary": "",
            "topics": [],
            "indexing_status": "error",
            "batch": "",
            "success": False,
            "error": "未从 Workflow outputs 中解析出任何结果条目",
            "raw": outputs,
        }]

    parsed_results = []
    for idx, item in enumerate(raw_items):
        default_name = file_names[idx] if (file_names and idx < len(file_names)) else f"文件_{idx + 1}"
        parsed = _normalize_auto_item(item, expected_package, default_name=default_name)
        parsed_results.append(parsed)

    return parsed_results


# ── Workflow 结果校验 ────────────────────────────────────────


def validate_workflow_result(result: dict, expected_package: str) -> tuple[bool, str, dict]:
    """校验单个文件的 Workflow 返回结果。

    Args:
        result: Workflow 返回的单个文件结果
        expected_package: 期望的 contract_package 值

    Returns:
        (ok, error_message, cleaned): ok=True 时 cleaned 为清洗后的结果
    """
    if not isinstance(result, dict):
        return False, f"结果不是字典类型: {type(result).__name__}", {}

    # 检查必需字段
    missing = WORKFLOW_RESULT_FIELDS - set(result.keys())
    if missing:
        return False, f"缺少字段: {', '.join(sorted(missing))}", {}

    # 校验 contract_package
    pkg = result.get("contract_package", "")
    if pkg != expected_package:
        return False, (
            f"合同包不匹配: 期望 {expected_package}，"
            f"实际 {pkg}"
        ), {}

    # 校验 topics
    topics = result.get("topics")
    if not isinstance(topics, list):
        return False, f"topics 不是列表: {type(topics).__name__}", {}
    if len(topics) < 3 or len(topics) > 5:
        return False, (
            f"topics 数量不符: 期望 3-5 个，实际 {len(topics)} 个"
        ), {}
    for i, t in enumerate(topics):
        if not isinstance(t, str) or not t.strip():
            return False, f"topics[{i}] 不是非空字符串", {}

    # 清洗结果
    cleaned = {
        "contract_package": str(pkg).strip(),
        "document_type": str(result["document_type"]).strip(),
        "document_title": str(result["document_title"]).strip(),
        "document_language": str(result["document_language"]).strip(),
        "document_summary": str(result["document_summary"]).strip(),
        "topics": [str(t).strip() for t in topics],
    }

    # 基础非空校验
    for field in ("document_type", "document_title", "document_language"):
        if not cleaned[field]:
            return False, f"{field} 为空", {}

    return True, "", cleaned


def _extract_metadata_items(outputs) -> list:
    """从 Workflow outputs 中提取 metadata 条目列表。

    兼容三种 Dify 返回格式：
    1. {"output": [metadata, ...]}  — 常见，key 为 "output"
    2. [metadata, ...]              — 直接为列表
    3. metadata dict                — 单条 metadata（含 contract_package 等字段）

    Returns:
        list: metadata 条目列表
    """
    if isinstance(outputs, list):
        return outputs
    if isinstance(outputs, dict):
        # 优先检查 "output" 键（Dify Workflow 常见返回格式）
        if "output" in outputs and isinstance(outputs["output"], list):
            return outputs["output"]
        # 如果 dict 本身包含 metadata 特征字段，视为单条 metadata
        if "contract_package" in outputs or "document_type" in outputs:
            return [outputs]
        # 兜底：扫描 values 中的列表
        for v in outputs.values():
            if isinstance(v, list):
                return v
        return [outputs]
    return []


def validate_workflow_outputs(outputs, expected_package: str) -> list[dict]:
    """校验 Workflow 完整 outputs，返回每个文件的校验结果。

    兼容三种 Dify 返回格式：
    - {"output": [metadata, ...]}  — Dify Workflow 常见格式
    - [metadata, ...]              — 直接列表
    - metadata dict                — 单条 metadata

    Returns:
        list[dict]: 每个元素 {ok, error, cleaned, raw}
    """
    items = _extract_metadata_items(outputs)
    if not items:
        return [{"ok": False,
                 "error": f"outputs 中未找到 metadata 列表: {type(outputs).__name__}",
                 "cleaned": {}, "raw": outputs}]

    results = []
    for item in items:
        ok, err, cleaned = validate_workflow_result(item, expected_package)
        results.append({
            "ok": ok,
            "error": err,
            "cleaned": cleaned,
            "raw": item,
        })
    return results


# ── 文档创建 ─────────────────────────────────────────────────


def get_dataset_info(dataset_api_key: str, base_url: str,
                     dataset_id: str) -> dict:
    """获取单个知识库的详细信息。

    从 /datasets 列表中查找指定 ID 的知识库，返回其完整信息（含 doc_form）。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID

    Returns:
        dict: 知识库信息（含 id, name, doc_form 等）

    Raises:
        RuntimeError: 查询失败或未找到
    """
    data = _get_json(dataset_api_key, base_url, "/datasets",
                     params={"page": 1, "limit": 100})
    for ds in data.get("data", []):
        if ds.get("id") == dataset_id:
            return ds
    raise RuntimeError(f"未找到知识库: {dataset_id}")


def create_document(dataset_api_key: str, base_url: str,
                    dataset_id: str, name: str, text: str,
                    doc_form: str = "text_model") -> dict:
    """在 Dify 知识库中创建文档。

    Args:
        dataset_api_key: 知识库 API Key（dataset- 开头）
        base_url: 知识库 API Base URL
        dataset_id: 目标知识库 ID
        name: 文档名称
        text: 文档文本内容
        doc_form: 分段模式（text_model / hierarchical_model / qa_model），
                  必须与目标知识库的实际配置一致

    Returns:
        dict: 创建结果，包含 document.id 和 batch

    Raises:
        RuntimeError: 创建失败
    """
    path = f"/datasets/{dataset_id}/document/create-by-text"
    body = {
        "name": name,
        "text": text,
        "indexing_technique": "high_quality",
        "doc_form": doc_form,
        "process_rule": {"mode": "automatic"},
    }
    result = _post_json(dataset_api_key, base_url, path, body, timeout=120)

    doc = result.get("document")
    if not doc or not doc.get("id"):
        raise RuntimeError(
            f"文档创建成功但未返回 document: "
            f"{json.dumps(result, ensure_ascii=False)[:200]}"
        )
    return result


def create_document_by_file(dataset_api_key: str, base_url: str,
                            dataset_id: str, file_name: str,
                            file_bytes: bytes,
                            doc_form: str = "text_model") -> dict:
    """通过上传原始文件在 Dify 知识库中创建文档。

    使用 POST /datasets/{dataset_id}/document/create-by-file，
    让 Dify 自行解析 docx、xlsx、pdf 等格式，避免本地提取文本为空。

    Args:
        dataset_api_key: 知识库 API Key（dataset- 开头）
        base_url: 知识库 API Base URL
        dataset_id: 目标知识库 ID
        file_name: 原始文件名（含扩展名）
        file_bytes: 原始文件二进制内容
        doc_form: 分段模式（text_model / hierarchical_model / qa_model）

    Returns:
        dict: 创建结果，包含 document.id 和 batch

    Raises:
        RuntimeError: 创建失败或文件为空
    """
    import tempfile
    import io

    if not file_bytes:
        raise RuntimeError(f"文件内容为空: {file_name}")

    path = f"/datasets/{dataset_id}/document/create-by-file"
    url = base_url.rstrip("/") + path

    # data 字段：JSON 字符串
    # hierarchical_model 需要完整的父子分块规则，不能用 automatic
    if doc_form == "hierarchical_model":
        process_rule = {
            "mode": "hierarchical",
            "rules": {
                "pre_processing_rules": [
                    {"id": "remove_extra_spaces", "enabled": True},
                    {"id": "remove_urls_emails", "enabled": False},
                ],
                "parent_mode": "paragraph",
                "segmentation": {
                    "separator": "\n\n",
                    "max_tokens": 500,
                    "chunk_overlap": 50,
                },
                "subchunk_segmentation": {
                    "separator": "\n",
                    "max_tokens": 250,
                    "chunk_overlap": 25,
                },
            },
        }
    else:
        process_rule = {"mode": "automatic"}

    data_payload = json.dumps({
        "indexing_technique": "high_quality",
        "doc_form": doc_form,
        "process_rule": process_rule,
    }, ensure_ascii=False)

    headers = {"Authorization": f"Bearer {dataset_api_key}"}
    mime_type = _guess_mime_type(file_name)

    # 写入临时文件供 requests 读取
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file_name).suffix
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            files = {"file": (file_name, f, mime_type)}
            data = {"data": data_payload}
            try:
                resp = requests.post(
                    url, headers=headers, files=files, data=data,
                    timeout=180,
                )
            except requests.exceptions.Timeout:
                raise RuntimeError(f"文档创建超时: {file_name}")
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(
                    f"文档创建连接失败 — 请检查 Base URL 是否正确"
                ) from exc

        if resp.status_code not in (200, 201):
            _raise_http_error(resp, dataset_api_key, f"创建文档 {file_name}")

        try:
            result = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"文档创建响应解析失败: {resp.text[:200]}"
            ) from exc

        doc = result.get("document")
        if not doc or not doc.get("id"):
            raise RuntimeError(
                f"文档创建成功但未返回 document: "
                f"{resp.text[:200]}"
            )
        return result

    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Knowledge Pipeline API ───────────────────────────────────


def upload_pipeline_file(dataset_api_key: str, base_url: str,
                         file_name: str, file_bytes: bytes) -> dict:
    """上传文件到 Knowledge Pipeline。

    POST /datasets/pipeline/file-upload（multipart/form-data）。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        file_name: 原始文件名
        file_bytes: 原始文件二进制内容

    Returns:
        dict: 包含 id 字段的上传结果
    """
    import tempfile

    if not file_bytes:
        raise RuntimeError(f"文件内容为空: {file_name}")

    url = base_url.rstrip("/") + "/datasets/pipeline/file-upload"
    headers = {"Authorization": f"Bearer {dataset_api_key}"}
    mime_type = _guess_mime_type(file_name)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file_name).suffix
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            files = {"file": (file_name, f, mime_type)}
            try:
                resp = requests.post(
                    url, headers=headers, files=files, timeout=120,
                )
            except requests.exceptions.Timeout:
                raise RuntimeError(f"Pipeline 文件上传超时: {file_name}")
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError("Pipeline 文件上传连接失败") from exc

        if resp.status_code not in (200, 201):
            _raise_http_error(resp, dataset_api_key, f"Pipeline 上传 {file_name}")

        try:
            result = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Pipeline 上传响应解析失败: {resp.text[:200]}") from exc

        file_id = result.get("id")
        if not file_id:
            raise RuntimeError(
                f"Pipeline 上传成功但未返回 id: {resp.text[:200]}"
            )
        return result
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def list_pipeline_datasource_plugins(dataset_api_key: str, base_url: str,
                                     dataset_id: str) -> list[dict]:
    """查询已发布 Pipeline 的 datasource 节点。

    GET /datasets/{dataset_id}/pipeline/datasource-plugins?is_published=true

    Returns:
        list[dict]: datasource 节点列表
    """
    path = f"/datasets/{dataset_id}/pipeline/datasource-plugins"
    result = _get_json(dataset_api_key, base_url, path,
                       params={"is_published": "true"}, timeout=15)
    # Dify 可能直接返回列表，也可能包装在 data/plugins 键中
    if isinstance(result, list):
        return result
    return result.get("data", result.get("plugins", []))


def find_local_file_node_id(dataset_api_key: str, base_url: str,
                            dataset_id: str) -> str:
    """在已发布 Pipeline 中查找唯一的 local_file datasource 节点 ID。

    Returns:
        str: node_id

    Raises:
        RuntimeError: 未找到或找到多个
    """
    plugins = list_pipeline_datasource_plugins(
        dataset_api_key, base_url, dataset_id,
    )
    local_file_nodes = [
        p for p in plugins
        if p.get("datasource_type") == "local_file"
    ]
    if len(local_file_nodes) == 1:
        return local_file_nodes[0]["node_id"]
    if len(local_file_nodes) == 0:
        raise RuntimeError(
            "已发布知识库 Pipeline 中未找到本地文件数据源节点"
        )
    raise RuntimeError(
        f"已发布知识库 Pipeline 中找到 {len(local_file_nodes)} 个本地文件节点，"
        f"无法确定唯一节点"
    )


def run_knowledge_pipeline(dataset_api_key: str, base_url: str,
                           dataset_id: str, pipeline_file_id: str,
                           file_name: str, start_node_id: str) -> dict:
    """运行已发布的 Knowledge Pipeline。

    POST /datasets/{dataset_id}/pipeline/run

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID
        pipeline_file_id: upload_pipeline_file 返回的 id
        file_name: 原始文件名
        start_node_id: local_file datasource 节点 ID

    Returns:
        dict: Pipeline 运行结果
    """
    path = f"/datasets/{dataset_id}/pipeline/run"
    body = {
        "inputs": {},
        "datasource_type": "local_file",
        "datasource_info_list": [
            {"reference": pipeline_file_id, "name": file_name},
        ],
        "start_node_id": start_node_id,
        "is_published": True,
        "response_mode": "blocking",
    }
    return _post_json(dataset_api_key, base_url, path, body, timeout=300)


def try_pipeline_ingestion(dataset_api_key: str, base_url: str,
                           dataset_id: str, file_name: str,
                           file_bytes: bytes) -> tuple[bool, str, str, str]:
    """尝试通过 Knowledge Pipeline 入库。

    Returns:
        (success, document_id_or_error, batch, mode):
        - success=True: document_id 为文档 ID，mode="pipeline"
        - success=False: document_id_or_error 为错误信息，mode="fallback"
    """
    try:
        # 1. 查询 local_file 节点
        node_id = find_local_file_node_id(
            dataset_api_key, base_url, dataset_id,
        )

        # 2. 上传文件到 Pipeline
        upload_result = upload_pipeline_file(
            dataset_api_key, base_url, file_name, file_bytes,
        )
        pipeline_file_id = upload_result["id"]

        # 3. 运行 Pipeline
        run_result = run_knowledge_pipeline(
            dataset_api_key, base_url, dataset_id,
            pipeline_file_id, file_name, node_id,
        )

        # 4. 尝试从返回结果提取 document_id
        document_id = _extract_document_id_from_pipeline(run_result)

        if document_id:
            return True, document_id, "", "pipeline"

        # Pipeline 执行了但无法确定 document_id
        return False, "Pipeline 已执行，但无法唯一确认目标文档", "", "pipeline_no_doc_id"

    except RuntimeError as exc:
        return False, str(exc), "", "fallback"


def _extract_document_id_from_pipeline(run_result: dict) -> str:
    """从 Pipeline run 结果中提取 document_id。

    尝试多种返回格式：
    - result.outputs.document_id
    - result.outputs.document_ids[0]
    - result.data.outputs.document_id
    """
    # 直接在顶层查找
    for key in ("document_id", "doc_id"):
        val = run_result.get(key)
        if val:
            return str(val)

    # 在 outputs 中查找
    outputs = run_result.get("outputs") or run_result.get("data", {}).get("outputs") or {}
    for key in ("document_id", "doc_id"):
        val = outputs.get(key)
        if val:
            return str(val)

    # document_ids 列表
    doc_ids = outputs.get("document_ids") or run_result.get("document_ids")
    if isinstance(doc_ids, list) and len(doc_ids) == 1:
        return str(doc_ids[0])

    return ""


def find_document_by_name(dataset_api_key: str, base_url: str,
                          dataset_id: str, file_name: str) -> str:
    """通过文件名在知识库中查找文档 ID。

    优先返回最新的同名文档 ID。

    Returns:
        str: document_id 或空字符串
    """
    info = find_document_info_by_name(dataset_api_key, base_url, dataset_id, file_name)
    return str(info.get("id", "")) if info else ""


def find_document_info_by_name(dataset_api_key: str, base_url: str,
                               dataset_id: str, file_name: str) -> dict:
    """通过文件名在知识库中查找文档详情（包含 id, doc_metadata, indexing_status 等）。

    优先返回最新的同名文档记录。

    Returns:
        dict: 文档字典或空字典
    """
    path = f"/datasets/{dataset_id}/documents"
    try:
        result = _get_json(dataset_api_key, base_url, path,
                           params={"page": 1, "limit": 100}, timeout=15)
    except Exception:
        return {}

    docs = result.get("data", [])
    if not docs:
        return {}

    name_stem = Path(file_name).stem
    matching = [
        d for d in docs
        if (d.get("name") == file_name or d.get("name") == name_stem)
        and d.get("enabled") is not False
    ]
    if not matching:
        # 模糊包含匹配
        matching = [
            d for d in docs
            if (file_name in d.get("name", "") or name_stem in d.get("name", ""))
            and d.get("enabled") is not False
        ]

    if matching:
        # 按 created_at 降序排序，取最新入库的文档
        sorted_matching = sorted(
            matching,
            key=lambda x: x.get("created_at") or 0,
            reverse=True,
        )
        return sorted_matching[0]

    return {}


# ── 索引状态与分段查询 ──────────────────────────────────────


# 索引状态 → 中文文案映射（未知状态原样显示）
INDEXING_STATUS_LABELS = {
    "waiting": "已提交，等待处理",
    "parsing": "正在解析原始文件",
    "cleaning": "正在清洗文本",
    "splitting": "正在进行父子分块",
    "indexing": "正在建立检索索引",
    "completed": "索引完成",
    "error": "索引失败",
}


def get_document_indexing_status(dataset_api_key: str, base_url: str,
                                 dataset_id: str, batch: str) -> dict:
    """查询文档索引状态。

    使用 GET /datasets/{dataset_id}/documents/{batch}/indexing-status。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID
        batch: 文档创建时返回的 batch ID

    Returns:
        dict: 包含 id, indexing_status, error 等字段

    Raises:
        RuntimeError: 查询失败
    """
    path = f"/datasets/{dataset_id}/documents/{batch}/indexing-status"
    result = _get_json(dataset_api_key, base_url, path, timeout=15)

    # 规范化返回字段
    data = result.get("data", result)
    if isinstance(data, list) and data:
        data = data[0]
    return {
        "id": data.get("id", ""),
        "indexing_status": data.get("indexing_status", "unknown"),
        "processing_started_at": data.get("processing_started_at"),
        "parsing_completed_at": data.get("parsing_completed_at"),
        "cleaning_completed_at": data.get("cleaning_completed_at"),
        "splitting_completed_at": data.get("splitting_completed_at"),
        "completed_at": data.get("completed_at"),
        "error": data.get("error"),
    }


def get_document_segments(dataset_api_key: str, base_url: str,
                          dataset_id: str, document_id: str,
                          limit: int = 100) -> int:
    """获取文档的分段数量。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID
        document_id: 文档 ID
        limit: 每页数量

    Returns:
        int: 分段总数
    """
    path = f"/datasets/{dataset_id}/documents/{document_id}/segments"
    result = _get_json(dataset_api_key, base_url, path,
                       params={"limit": min(limit, 100)}, timeout=15)
    # Dify 返回格式: {"data": [...], "total": N, ...}
    if "total" in result:
        return result["total"]
    return len(result.get("data", []))


def wait_for_document_segments(
    dataset_api_key: str,
    base_url: str,
    dataset_id: str,
    document_id: str,
    timeout: int = 120,
    interval: float = 2.0,
    on_progress=None,
) -> dict:
    """轮询等待文档分段生成。

    Pipeline API 返回仅表示工作流执行结束，文档解析、分块和索引
    仍可能在后台异步进行。此函数轮询直到检测到分段或超时。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID
        document_id: 文档 ID
        timeout: 最长等待秒数
        interval: 轮询间隔秒数
        on_progress: 可选回调 fn(elapsed: float, segment_count: int)

    Returns:
        dict: {
            "status": "completed" | "processing" | "poll_error",
            "segment_count": int,
            "elapsed": float,
            "error": str,
        }
    """
    import time

    start = time.monotonic()
    last_error = ""
    consecutive_errors = 0

    while True:
        elapsed = time.monotonic() - start

        try:
            seg_count = get_document_segments(
                dataset_api_key, base_url, dataset_id, document_id,
            )
            consecutive_errors = 0

            if on_progress:
                on_progress(elapsed, seg_count)

            if seg_count > 0:
                return {
                    "status": "completed",
                    "segment_count": seg_count,
                    "elapsed": elapsed,
                    "error": "",
                }

            # segment_count == 0，继续等待
            if elapsed >= timeout:
                return {
                    "status": "processing",
                    "segment_count": 0,
                    "elapsed": elapsed,
                    "error": "",
                }

            time.sleep(interval)

        except RuntimeError as exc:
            last_error = str(exc)[:200]
            consecutive_errors += 1
            elapsed = time.monotonic() - start

            if on_progress:
                on_progress(elapsed, 0)

            # 持续失败且已超时
            if elapsed >= timeout:
                return {
                    "status": "poll_error",
                    "segment_count": 0,
                    "elapsed": elapsed,
                    "error": last_error,
                }

            # 偶发错误，继续重试
            time.sleep(interval)


# ── Metadata 操作 ────────────────────────────────────────────


def list_metadata_fields(dataset_api_key: str, base_url: str,
                         dataset_id: str) -> list[dict]:
    """列出知识库已定义的 metadata 字段。

    Returns:
        list[dict]: 每个元素包含 id, name, type, count
    """
    path = f"/datasets/{dataset_id}/metadata"
    result = _get_json(dataset_api_key, base_url, path)
    return result.get("doc_metadata", [])


def bind_document_metadata(dataset_api_key: str, base_url: str,
                           dataset_id: str, document_id: str,
                           metadata_items: list[dict]) -> dict:
    """为文档绑定 metadata。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID
        document_id: 文档 ID
        metadata_items: [{id, name, value}, ...]

    Returns:
        dict: API 响应
    """
    path = f"/datasets/{dataset_id}/documents/metadata"
    body = {
        "operation_data": [
            {
                "document_id": document_id,
                "metadata_list": metadata_items,
                "partial_update": True,
            }
        ]
    }
    return _post_json(dataset_api_key, base_url, path, body)


# ── 入库所需的 6 个 metadata 字段 ────────────────────────────

REQUIRED_METADATA_FIELDS = [
    {"name": "contract_package", "type": "string"},
    {"name": "document_type", "type": "string"},
    {"name": "document_title", "type": "string"},
    {"name": "document_language", "type": "string"},
    {"name": "document_summary", "type": "string"},
    {"name": "topics", "type": "string"},
]


def create_metadata_field(dataset_api_key: str, base_url: str,
                          dataset_id: str, name: str,
                          field_type: str = "string") -> dict:
    """在知识库中创建一个 metadata 字段。

    Args:
        dataset_api_key: 知识库 API Key（dataset- 开头）
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID
        name: 字段名称
        field_type: 字段类型（string / number / time）

    Returns:
        dict: 创建结果，包含 id, name, type

    Raises:
        RuntimeError: 创建失败
    """
    path = f"/datasets/{dataset_id}/metadata"
    body = {"name": name, "type": field_type}
    return _post_json(dataset_api_key, base_url, path, body)


def ensure_required_metadata_fields(dataset_api_key: str, base_url: str,
                                     dataset_id: str) -> tuple[list[dict], list[str]]:
    """确保知识库具有入库所需的全部 6 个 metadata 字段。

    只创建缺失的字段，绝不覆盖、删除或修改已存在字段。

    Args:
        dataset_api_key: 知识库 API Key
        base_url: 知识库 API Base URL
        dataset_id: 知识库 ID

    Returns:
        (created_fields, errors):
            created_fields: 本次新创建的字段列表 [{id, name, type}, ...]
            errors: 创建失败的错误信息列表

    Raises:
        RuntimeError: 读取已有字段失败
    """
    existing = list_metadata_fields(dataset_api_key, base_url, dataset_id)
    existing_names = {f["name"] for f in existing}

    created = []
    errors = []

    for field_def in REQUIRED_METADATA_FIELDS:
        if field_def["name"] in existing_names:
            continue
        try:
            result = create_metadata_field(
                dataset_api_key, base_url, dataset_id,
                field_def["name"], field_def["type"],
            )
            created.append(result)
        except RuntimeError as exc:
            errors.append(f"{field_def['name']}: {exc}")

    return created, errors


def compute_content_hash(content: str) -> str:
    """对内容做规范化 SHA-256 哈希（与 dify_knowledge 逻辑一致）。

    规范化：strip 首尾空白，统一换行符为 \\n。
    """
    if not content:
        return ""
    normalized = content.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_ingestion_history(dataset_id: str) -> list[dict]:
    """加载指定知识库的入库历史记录。

    Returns:
        list[dict]: 历史记录列表，按时间正序
    """
    history_path = INGESTION_HISTORY_DIR / f"{dataset_id}.jsonl"
    if not history_path.exists():
        return []
    records = []
    for line_num, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def check_duplicate(dataset_id: str, content_hash: str) -> dict | None:
    """检查指定知识库中是否已有相同内容的入库记录。

    Returns:
        dict: 匹配的历史记录，或 None
    """
    if not content_hash:
        return None
    for record in load_ingestion_history(dataset_id):
        if record.get("content_hash") == content_hash and record.get("ingestion_status") == "success":
            return record
    return None


def append_ingestion_record(record: dict) -> None:
    """追加一条入库记录到 JSONL 历史文件。

    安全规则：如果 record 中包含 API Key 字段，自动移除。
    """
    # 安全检查：移除可能的 Key 字段
    safe_record = {
        k: v for k, v in record.items()
        if k not in ("api_key", "dataset_api_key", "workflow_api_key", "key")
    }

    dataset_id = safe_record.get("dataset_id", "unknown")
    INGESTION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = INGESTION_HISTORY_DIR / f"{dataset_id}.jsonl"

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def build_ingestion_record(
    *,
    dataset_id: str,
    file_name: str,
    content_hash: str,
    document_id: str = "",
    metadata: dict = None,
    workflow_status: str = "pending",
    ingestion_status: str = "pending",
    error_message: str = "",
    batch: str = "",
    indexing_status: str = "",
    segment_count: int = -1,
    indexing_error: str = "",
) -> dict:
    """构造入库记录（不含 API Key）。

    新增可选字段（旧记录兼容）：
    - batch: Dify 文档创建批次 ID
    - indexing_status: 最终索引状态
    - segment_count: 分段数（-1 表示未查询）
    - indexing_error: 索引错误信息
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "file_name": file_name,
        "content_hash": content_hash,
        "document_id": document_id,
        "metadata": metadata or {},
        "workflow_status": workflow_status,
        "ingestion_status": ingestion_status,
        "error_message": error_message,
    }
    # 新增字段（可选，旧记录兼容）
    if batch:
        record["batch"] = batch
    if indexing_status:
        record["indexing_status"] = indexing_status
    if segment_count >= 0:
        record["segment_count"] = segment_count
    if indexing_error:
        record["indexing_error"] = indexing_error
    return record
