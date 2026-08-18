"""
材料入库模块测试。

测试内容：
1. validate_workflow_key — app- 前缀校验
2. validate_dataset_key — dataset- 前缀校验
3. validate_workflow_result — Workflow 返回 schema 校验
4. validate_workflow_outputs — outputs 格式适配
5. compute_content_hash — 内容规范化哈希
6. check_duplicate — 重复检测
7. 单文件失败不阻断批次
8. 创建文档不自动重试
9. 历史记录不含 API Key
10. 完整入库流程 mock
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dify_ingestion import (
    validate_workflow_key,
    validate_dataset_key,
    validate_workflow_result,
    validate_workflow_outputs,
    run_auto_ingestion_workflow,
    parse_auto_ingestion_outputs,
    compute_content_hash,
    check_duplicate,
    load_ingestion_history,
    append_ingestion_record,
    build_ingestion_record,
    upload_file,
    upload_text_as_file,
    extract_text_from_file,
    _post_file,
    _guess_mime_type,
    run_workflow,
    create_document,
    create_document_by_file,
    get_dataset_info,
    list_metadata_fields,
    bind_document_metadata,
    create_metadata_field,
    ensure_required_metadata_fields,
    REQUIRED_METADATA_FIELDS,
    get_document_indexing_status,
    get_document_segments,
    INDEXING_STATUS_LABELS,
    get_document_indexing_status,
    get_document_segments,
    INDEXING_STATUS_LABELS,
    upload_pipeline_file,
    list_pipeline_datasource_plugins,
    find_local_file_node_id,
    run_knowledge_pipeline,
    try_pipeline_ingestion,
    find_document_by_name,
    _extract_document_id_from_pipeline,
    wait_for_document_segments,
    INGESTION_HISTORY_DIR,
    VALID_WORKFLOW_PACKAGES,
    WORKFLOW_RESULT_FIELDS,
)


# ── Fixtures ──────────────────────────────────────────────────


def _mock_response(json_data, status_code=200):
    """构造 mock requests.Response。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data, ensure_ascii=False)
    return resp


def _sample_workflow_result():
    """返回一组合法的 Workflow 结果。"""
    return {
        "contract_package": "baseline_2_4",
        "document_type": "合同",
        "document_title": "测试合同标题",
        "document_language": "中文",
        "document_summary": "这是一份测试合同的摘要。",
        "topics": ["主题一", "主题二", "主题三"],
    }


@pytest.fixture(autouse=True)
def _use_tmp_history(tmp_path):
    """每个测试使用独立的临时历史目录。"""
    with patch("dify_ingestion.INGESTION_HISTORY_DIR", tmp_path / "history"):
        yield tmp_path


# ── TestValidateWorkflowKey ───────────────────────────────────


class TestValidateWorkflowKey:
    """Workflow Key 前缀校验。"""

    def test_valid_app_key(self):
        ok, err = validate_workflow_key("app-abcdef123456")
        assert ok is True
        assert err == ""

    def test_empty_key(self):
        ok, err = validate_workflow_key("")
        assert ok is False
        assert "未设置" in err

    def test_none_key(self):
        ok, err = validate_workflow_key(None)
        assert ok is False

    def test_dataset_key_rejected(self):
        ok, err = validate_workflow_key("dataset-abcdef123456")
        assert ok is False
        assert "dataset-" in err
        assert "app-" in err

    def test_random_prefix_rejected(self):
        ok, err = validate_workflow_key("sk-abcdef123456")
        assert ok is False
        assert "前缀不正确" in err


# ── TestValidateDatasetKey ────────────────────────────────────


class TestValidateDatasetKey:
    """Dataset Key 前缀校验。"""

    def test_valid_dataset_key(self):
        ok, err = validate_dataset_key("dataset-abcdef123456")
        assert ok is True
        assert err == ""

    def test_empty_key(self):
        ok, err = validate_dataset_key("")
        assert ok is False
        assert "未设置" in err

    def test_app_key_rejected(self):
        ok, err = validate_dataset_key("app-abcdef123456")
        assert ok is False
        assert "app-" in err

    def test_random_prefix_rejected(self):
        ok, err = validate_dataset_key("token-abcdef123456")
        assert ok is False
        assert "前缀不正确" in err


# ── TestValidateWorkflowResult ────────────────────────────────


class TestValidateWorkflowResult:
    """Workflow 返回结果 schema 校验。"""

    def test_valid_result(self):
        result = _sample_workflow_result()
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is True
        assert err == ""
        assert cleaned["contract_package"] == "baseline_2_4"
        assert cleaned["document_type"] == "合同"
        assert len(cleaned["topics"]) == 3

    def test_missing_field(self):
        result = _sample_workflow_result()
        del result["document_type"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is False
        assert "缺少字段" in err
        assert "document_type" in err

    def test_package_mismatch(self):
        result = _sample_workflow_result()
        ok, err, cleaned = validate_workflow_result(result, "tech_platform_2_5")
        assert ok is False
        assert "不匹配" in err

    def test_topics_not_list(self):
        result = _sample_workflow_result()
        result["topics"] = "单一主题"
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is False
        assert "不是列表" in err

    def test_topics_too_few(self):
        result = _sample_workflow_result()
        result["topics"] = ["主题一", "主题二"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is False
        assert "数量不符" in err

    def test_topics_too_many(self):
        result = _sample_workflow_result()
        result["topics"] = ["t1", "t2", "t3", "t4", "t5", "t6"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is False
        assert "数量不符" in err

    def test_topics_empty_string(self):
        result = _sample_workflow_result()
        result["topics"] = ["主题一", "", "主题三"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is False
        assert "非空字符串" in err

    def test_topics_exactly_three(self):
        result = _sample_workflow_result()
        result["topics"] = ["a", "b", "c"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is True

    def test_topics_exactly_five(self):
        result = _sample_workflow_result()
        result["topics"] = ["a", "b", "c", "d", "e"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is True

    def test_empty_title(self):
        result = _sample_workflow_result()
        result["document_title"] = "  "
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is False
        assert "为空" in err

    def test_not_dict(self):
        ok, err, cleaned = validate_workflow_result("not a dict", "baseline_2_4")
        assert ok is False
        assert "不是字典" in err

    def test_whitespace_trimmed(self):
        result = _sample_workflow_result()
        result["document_title"] = "  标题  "
        result["topics"] = ["  t1  ", " t2 ", "  t3"]
        ok, err, cleaned = validate_workflow_result(result, "baseline_2_4")
        assert ok is True
        assert cleaned["document_title"] == "标题"
        assert cleaned["topics"] == ["t1", "t2", "t3"]


# ── TestValidateWorkflowOutputs ───────────────────────────────


class TestValidateWorkflowOutputs:
    """Workflow outputs 格式适配。"""

    def test_list_outputs(self):
        outputs = [_sample_workflow_result()]
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is True

    def test_single_dict_output(self):
        outputs = _sample_workflow_result()
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is True

    def test_unsupported_type(self):
        # int 不是 dict/list，_extract_metadata_items 返回 []
        results = validate_workflow_outputs(12345, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is False
        assert "未找到" in results[0]["error"]

    def test_mixed_valid_invalid(self):
        outputs = [
            _sample_workflow_result(),
            {"contract_package": "baseline_2_4"},  # 缺字段
        ]
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 2
        assert results[0]["ok"] is True
        assert results[1]["ok"] is False

    def test_output_key_format(self):
        """Dify Workflow 真实返回：{"output": [metadata, ...]}。"""
        outputs = {
            "output": [
                _sample_workflow_result(),
            ]
        }
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is True
        assert results[0]["cleaned"]["document_title"] == "测试合同标题"

    def test_output_key_multiple_items(self):
        """{"output": [m1, m2]} 应拆分为两条校验结果。"""
        r1 = _sample_workflow_result()
        r2 = _sample_workflow_result()
        r2["document_title"] = "第二份合同"
        outputs = {"output": [r1, r2]}
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 2
        assert results[0]["ok"] is True
        assert results[1]["ok"] is True
        assert results[1]["cleaned"]["document_title"] == "第二份合同"

    def test_output_key_with_extra_keys(self):
        """{"output": [...], "extra": ...} 应忽略 extra，只取 output。"""
        outputs = {
            "output": [_sample_workflow_result()],
            "task_id": "abc-123",
            "elapsed_time": 1.5,
        }
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is True

    def test_output_key_empty_list(self):
        """{"output": []} 应返回错误。"""
        outputs = {"output": []}
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is False

    def test_output_key_invalid_items(self):
        """{"output": [invalid]} 应逐条校验。"""
        outputs = {"output": [{"contract_package": "baseline_2_4"}]}
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert len(results) == 1
        assert results[0]["ok"] is False
        assert "缺少字段" in results[0]["error"]


# ── TestComputeContentHash ────────────────────────────────────


class TestComputeContentHash:
    """内容规范化 SHA-256 哈希。"""

    def test_basic(self):
        h = compute_content_hash("hello world")
        assert len(h) == 64  # SHA-256 hex

    def test_empty(self):
        assert compute_content_hash("") == ""
        assert compute_content_hash(None) == ""

    def test_whitespace_stripped(self):
        h1 = compute_content_hash("  hello  ")
        h2 = compute_content_hash("hello")
        assert h1 == h2

    def test_crlf_normalized(self):
        h1 = compute_content_hash("line1\r\nline2")
        h2 = compute_content_hash("line1\nline2")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_content_hash("abc")
        h2 = compute_content_hash("def")
        assert h1 != h2


# ── TestDuplicateDetection ────────────────────────────────────


class TestDuplicateDetection:
    """入库重复检测。"""

    def test_no_history(self, _use_tmp_history):
        assert check_duplicate("ds_001", "abc123") is None

    def test_duplicate_found(self, _use_tmp_history):
        history_dir = _use_tmp_history / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        record = build_ingestion_record(
            dataset_id="ds_001",
            file_name="test.txt",
            content_hash="abc123",
            document_id="doc_001",
            ingestion_status="success",
        )
        (history_dir / "ds_001.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        found = check_duplicate("ds_001", "abc123")
        assert found is not None
        assert found["document_id"] == "doc_001"

    def test_different_hash_not_duplicate(self, _use_tmp_history):
        history_dir = _use_tmp_history / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        record = build_ingestion_record(
            dataset_id="ds_001",
            file_name="test.txt",
            content_hash="abc123",
            ingestion_status="success",
        )
        (history_dir / "ds_001.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        assert check_duplicate("ds_001", "xyz789") is None

    def test_failed_record_not_duplicate(self, _use_tmp_history):
        history_dir = _use_tmp_history / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        record = build_ingestion_record(
            dataset_id="ds_001",
            file_name="test.txt",
            content_hash="abc123",
            ingestion_status="failed",
        )
        (history_dir / "ds_001.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        assert check_duplicate("ds_001", "abc123") is None


# ── TestHistoryNoApiKey ───────────────────────────────────────


class TestHistoryNoApiKey:
    """历史记录不包含 API Key。"""

    def test_append_strips_api_keys(self, _use_tmp_history):
        record = {
            "dataset_id": "ds_001",
            "file_name": "test.txt",
            "content_hash": "abc",
            "api_key": "app-secret123",
            "dataset_api_key": "dataset-secret456",
            "workflow_api_key": "app-workflow789",
            "key": "some-key",
            "ingestion_status": "success",
        }
        append_ingestion_record(record)

        history_dir = _use_tmp_history / "history"
        lines = (history_dir / "ds_001.jsonl").read_text(encoding="utf-8").strip().split("\n")
        saved = json.loads(lines[0])

        assert "api_key" not in saved
        assert "dataset_api_key" not in saved
        assert "workflow_api_key" not in saved
        assert "key" not in saved
        assert saved["ingestion_status"] == "success"

    def test_build_record_no_key_fields(self):
        record = build_ingestion_record(
            dataset_id="ds_001",
            file_name="test.txt",
            content_hash="abc",
        )
        assert "api_key" not in record
        assert "dataset_api_key" not in record
        assert "timestamp" in record
        assert "content_hash" in record


# ── TestUploadFile ────────────────────────────────────────────


class TestUploadFile:
    """文件上传 — 始终先尝试原始格式，仅 415 兜底转文本。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({"id": "file-abc-123", "name": "test.pdf"})
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test content")
            f.flush()
            file_id = upload_file("app-key", "http://localhost/v1", f.name)
        assert file_id == "file-abc-123"

    @patch("dify_ingestion.requests.post")
    def test_no_id_in_response(self, mock_post):
        mock_post.return_value = _mock_response({"name": "test.pdf"})
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test content")
            f.flush()
            with pytest.raises(RuntimeError, match="未返回 ID"):
                upload_file("app-key", "http://localhost/v1", f.name)

    def test_file_not_found(self):
        with pytest.raises(RuntimeError, match="文件不存在"):
            upload_file("app-key", "http://localhost/v1", "/nonexistent/file.pdf")

    @patch("dify_ingestion.requests.post")
    def test_docx_direct_upload_success(self, mock_post, _use_tmp_history):
        """docx 默认直接上传原文件（不转 .txt）。"""
        mock_post.return_value = _mock_response({"id": "file-docx-001"})
        tmp_file = _use_tmp_history / "contract.docx"
        tmp_file.write_bytes(b"PK\x03\x04fake docx")
        file_id = upload_file("app-key", "http://localhost/v1", str(tmp_file))
        assert file_id == "file-docx-001"
        # 确认只调用了一次（直接上传），没有 fallback
        assert mock_post.call_count == 1

    @patch("dify_ingestion.requests.post")
    def test_415_fallback_to_text(self, mock_post, _use_tmp_history):
        """Dify 返回 415 时兜底提取文本以 .txt 上传。"""
        # 第一次调用返回 415，第二次成功
        mock_post.side_effect = [
            _mock_response({"code": "unsupported_file_type"}, status_code=415),
            _mock_response({"id": "file-txt-fallback"}),
        ]
        tmp_file = _use_tmp_history / "contract.docx"
        tmp_file.write_bytes(b"PK\x03\x04fake docx")
        with patch("dify_ingestion.extract_text_from_file", return_value="提取的文本"):
            file_id = upload_file("app-key", "http://localhost/v1", str(tmp_file))
        assert file_id == "file-txt-fallback"
        assert mock_post.call_count == 2

    @patch("dify_ingestion.requests.post")
    def test_non_415_error_no_fallback(self, mock_post, _use_tmp_history):
        """非 415 错误不触发兜底，直接抛出。"""
        mock_post.return_value = _mock_response(
            {"message": "server error"}, status_code=500
        )
        tmp_file = _use_tmp_history / "contract.docx"
        tmp_file.write_bytes(b"fake")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            upload_file("app-key", "http://localhost/v1", str(tmp_file))
        # 只调用了一次，没有 fallback
        assert mock_post.call_count == 1

    @patch("dify_ingestion.requests.post")
    def test_txt_direct_upload(self, mock_post, _use_tmp_history):
        """txt 直接上传。"""
        mock_post.return_value = _mock_response({"id": "file-txt-002"})
        tmp_file = _use_tmp_history / "contract.txt"
        tmp_file.write_text("合同内容", encoding="utf-8")
        file_id = upload_file("app-key", "http://localhost/v1", str(tmp_file))
        assert file_id == "file-txt-002"
        assert mock_post.call_count == 1


# ── TestExtractTextFromFile ────────────────────────────────────


class TestExtractTextFromFile:
    """文件文本提取。"""

    def test_txt_file(self, _use_tmp_history):
        tmp_file = _use_tmp_history / "test.txt"
        tmp_file.write_text("Hello World\n你好世界", encoding="utf-8")
        text = extract_text_from_file(str(tmp_file))
        assert text == "Hello World\n你好世界"

    def test_md_file(self, _use_tmp_history):
        tmp_file = _use_tmp_history / "readme.md"
        tmp_file.write_text("# 标题\n\n内容", encoding="utf-8")
        text = extract_text_from_file(str(tmp_file))
        assert "# 标题" in text

    def test_empty_txt_raises(self, _use_tmp_history):
        tmp_file = _use_tmp_history / "empty.txt"
        tmp_file.write_text("   ", encoding="utf-8")
        with pytest.raises(RuntimeError, match="内容为空"):
            extract_text_from_file(str(tmp_file))

    def test_nonexistent_raises(self):
        with pytest.raises(RuntimeError, match="文件不存在"):
            extract_text_from_file("/nonexistent/file.txt")

    def test_docx_with_docx2txt(self, _use_tmp_history):
        """docx 文件通过 docx2txt 提取。"""
        tmp_file = _use_tmp_history / "test.docx"
        tmp_file.write_bytes(b"fake docx")
        with patch("docx2txt.process", return_value="提取的合同内容"):
            text = extract_text_from_file(str(tmp_file))
        assert text == "提取的合同内容"

    def test_docx_empty_raises(self, _use_tmp_history):
        tmp_file = _use_tmp_history / "empty.docx"
        tmp_file.write_bytes(b"fake")
        with patch("docx2txt.process", return_value="   "):
            with pytest.raises(RuntimeError, match="提取为空"):
                extract_text_from_file(str(tmp_file))

    def test_unsupported_format_fallback(self, _use_tmp_history):
        """未知格式尝试作为纯文本读取。"""
        tmp_file = _use_tmp_history / "data.xyz"
        tmp_file.write_text("some content", encoding="utf-8")
        text = extract_text_from_file(str(tmp_file))
        assert text == "some content"

    def test_unsupported_binary_fallback(self, _use_tmp_history):
        """二进制未知格式会尝试作为纯文本读取（可能含替换字符）。"""
        tmp_file = _use_tmp_history / "data.bin"
        tmp_file.write_bytes(b"\x00\x00\x00\x00")
        # 由于 errors="replace"，null bytes 被替换为 U+FFFD，内容非空
        text = extract_text_from_file(str(tmp_file))
        assert len(text) > 0  # 替换字符不为空


class TestUploadTextAsFile:
    """文本作为 .txt 文件上传。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({"id": "file-txt-001"})
        file_id = upload_text_as_file(
            "app-key", "http://localhost/v1",
            "这是合同内容", "合同.docx"
        )
        assert file_id == "file-txt-001"

    @patch("dify_ingestion.requests.post")
    def test_no_id_raises(self, mock_post):
        mock_post.return_value = _mock_response({"name": "test.txt"})
        with pytest.raises(RuntimeError, match="未返回 ID"):
            upload_text_as_file(
                "app-key", "http://localhost/v1",
                "内容", "test.docx"
            )


class TestGuessMimeType:
    """_guess_mime_type 扩展名 → MIME 映射。"""

    @pytest.mark.parametrize("ext,expected", [
        (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (".xls",  "application/vnd.ms-excel"),
        (".pdf",  "application/pdf"),
        (".csv",  "text/csv"),
        (".txt",  "text/plain"),
    ])
    def test_known_extensions(self, ext, expected):
        result = _guess_mime_type(f"file{ext}")
        assert result == expected

    def test_md_extension(self):
        mime = _guess_mime_type("readme.md")
        assert mime == "text/markdown"

    def test_unknown_extension_fallback(self):
        mime = _guess_mime_type("data.xyz123")
        assert mime == "application/octet-stream"


class TestPostFileMimeType:
    """_post_file multipart 上传时显式设置 Content-Type。"""

    @pytest.mark.parametrize("ext,expected_mime", [
        (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (".xls",  "application/vnd.ms-excel"),
        (".pdf",  "application/pdf"),
        (".csv",  "text/csv"),
        (".txt",  "text/plain"),
    ])
    @patch("dify_ingestion.requests.post")
    def test_mime_type_in_multipart(self, mock_post, ext, expected_mime, _use_tmp_history):
        """上传时 files 元组的第三项应为正确的 MIME Type。"""
        mock_post.return_value = _mock_response({"id": "f1"})
        tmp_file = _use_tmp_history / f"test{ext}"
        tmp_file.write_bytes(b"content")

        _post_file("app-key", "http://localhost/v1", "/files/upload", str(tmp_file))

        # 从 mock 调用参数中提取 files 字典
        call_kwargs = mock_post.call_args[1]
        file_tuple = call_kwargs["files"]["file"]
        # file_tuple = (filename, fileobj, content_type)
        assert file_tuple[0] == f"test{ext}"
        assert file_tuple[2] == expected_mime

    @patch("dify_ingestion.requests.post")
    def test_docx_content_type_not_octet_stream(self, mock_post, _use_tmp_history):
        """docx 的 Content-Type 不应是 application/octet-stream。"""
        mock_post.return_value = _mock_response({"id": "f1"})
        tmp_file = _use_tmp_history / "contract.docx"
        tmp_file.write_bytes(b"PK\x03\x04")

        _post_file("app-key", "http://localhost/v1", "/files/upload", str(tmp_file))

        file_tuple = mock_post.call_args[1]["files"]["file"]
        assert file_tuple[2] != "application/octet-stream"
        assert "wordprocessingml" in file_tuple[2]

    @patch("dify_ingestion.requests.post")
    def test_xlsx_content_type_not_octet_stream(self, mock_post, _use_tmp_history):
        """xlsx 的 Content-Type 不应是 application/octet-stream。"""
        mock_post.return_value = _mock_response({"id": "f1"})
        tmp_file = _use_tmp_history / "data.xlsx"
        tmp_file.write_bytes(b"PK\x03\x04")

        _post_file("app-key", "http://localhost/v1", "/files/upload", str(tmp_file))

        file_tuple = mock_post.call_args[1]["files"]["file"]
        assert file_tuple[2] != "application/octet-stream"
        assert "spreadsheetml" in file_tuple[2]


# ── TestRunWorkflow ───────────────────────────────────────────


class TestRunWorkflow:
    """Workflow 调用。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        outputs = {"result": [_sample_workflow_result()]}
        mock_post.return_value = _mock_response({
            "data": {"status": "succeeded", "outputs": outputs}
        })
        result = run_workflow(
            "app-key", "http://localhost/v1",
            ["file-123"], "baseline_2_4"
        )
        assert result == outputs

    @patch("dify_ingestion.requests.post")
    def test_workflow_failed(self, mock_post):
        mock_post.return_value = _mock_response({
            "data": {"status": "failed", "error": "节点执行出错"}
        })
        with pytest.raises(RuntimeError, match="执行失败"):
            run_workflow(
                "app-key", "http://localhost/v1",
                ["file-123"], "baseline_2_4"
            )

    def test_invalid_package(self):
        with pytest.raises(ValueError, match="不支持的合同包"):
            run_workflow(
                "app-key", "http://localhost/v1",
                ["file-123"], "invalid_package"
            )

    @patch("dify_ingestion.requests.post")
    def test_no_outputs(self, mock_post):
        mock_post.return_value = _mock_response({
            "data": {"status": "succeeded", "outputs": None}
        })
        with pytest.raises(RuntimeError, match="未返回 outputs"):
            run_workflow(
                "app-key", "http://localhost/v1",
                ["file-123"], "baseline_2_4"
            )


# ── TestCreateDocument ────────────────────────────────────────


class TestCreateDocument:
    """文档创建。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({
            "document": {"id": "doc-abc", "name": "test.txt"},
            "batch": "20250101120000",
        })
        result = create_document(
            "dataset-key", "http://localhost/v1",
            "ds-001", "test.txt", "文档内容",
            doc_form="text_model",
        )
        assert result["document"]["id"] == "doc-abc"

    @patch("dify_ingestion.requests.post")
    def test_no_document_in_response(self, mock_post):
        mock_post.return_value = _mock_response({"batch": "123"})
        with pytest.raises(RuntimeError, match="未返回 document"):
            create_document(
                "dataset-key", "http://localhost/v1",
                "ds-001", "test.txt", "内容"
            )

    @patch("dify_ingestion.requests.post")
    def test_text_model_request_body(self, mock_post):
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.txt"}, "batch": "b1",
        })
        create_document(
            "key", "http://localhost/v1", "ds-001", "a.txt", "内容",
            doc_form="text_model",
        )
        body = mock_post.call_args[1]["json"]
        assert body["doc_form"] == "text_model"

    @patch("dify_ingestion.requests.post")
    def test_hierarchical_model_request_body(self, mock_post):
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.txt"}, "batch": "b1",
        })
        create_document(
            "key", "http://localhost/v1", "ds-001", "a.txt", "内容",
            doc_form="hierarchical_model",
        )
        body = mock_post.call_args[1]["json"]
        assert body["doc_form"] == "hierarchical_model"

    @patch("dify_ingestion.requests.post")
    def test_doc_form_not_hardcoded(self, mock_post):
        """doc_form 必须由调用方传入，不能硬编码。"""
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.txt"}, "batch": "b1",
        })
        create_document(
            "key", "http://localhost/v1", "ds-001", "a.txt", "内容",
            doc_form="qa_model",
        )
        body = mock_post.call_args[1]["json"]
        assert body["doc_form"] == "qa_model"


class TestCreateDocumentByFile:
    """通过上传原始文件创建文档。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({
            "document": {"id": "doc-file-001", "name": "contract.docx"},
            "batch": "b1",
        })
        result = create_document_by_file(
            "dataset-key", "http://localhost/v1", "ds-001",
            "contract.docx", b"PK\x03\x04fake docx content",
            doc_form="hierarchical_model",
        )
        assert result["document"]["id"] == "doc-file-001"

    @patch("dify_ingestion.requests.post")
    def test_empty_bytes_raises(self, mock_post):
        with pytest.raises(RuntimeError, match="内容为空"):
            create_document_by_file(
                "dataset-key", "http://localhost/v1", "ds-001",
                "empty.docx", b"",
            )

    @patch("dify_ingestion.requests.post")
    def test_docx_mime_type(self, mock_post, _use_tmp_history):
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.docx"}, "batch": "b1",
        })
        create_document_by_file(
            "key", "http://localhost/v1", "ds-001",
            "a.docx", b"PK\x03\x04",
        )
        call_kwargs = mock_post.call_args[1]
        file_tuple = call_kwargs["files"]["file"]
        assert file_tuple[2] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    @patch("dify_ingestion.requests.post")
    def test_xlsx_mime_type(self, mock_post, _use_tmp_history):
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "data.xlsx"}, "batch": "b1",
        })
        create_document_by_file(
            "key", "http://localhost/v1", "ds-001",
            "data.xlsx", b"PK\x03\x04",
        )
        file_tuple = mock_post.call_args[1]["files"]["file"]
        assert file_tuple[2] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    @patch("dify_ingestion.requests.post")
    def test_hierarchical_model_request_body(self, mock_post):
        """hierarchical_model 必须使用完整的父子分块规则，不能用 automatic。"""
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.docx"}, "batch": "b1",
        })
        create_document_by_file(
            "key", "http://localhost/v1", "ds-001",
            "a.docx", b"PK\x03\x04",
            doc_form="hierarchical_model",
        )
        call_kwargs = mock_post.call_args[1]
        data_payload = json.loads(call_kwargs["data"]["data"])
        assert data_payload["doc_form"] == "hierarchical_model"
        rule = data_payload["process_rule"]
        assert rule["mode"] == "hierarchical"
        assert rule["rules"]["parent_mode"] == "paragraph"
        assert "subchunk_segmentation" in rule["rules"]
        assert rule["rules"]["subchunk_segmentation"]["max_tokens"] == 250
        assert rule["rules"]["segmentation"]["max_tokens"] == 500

    @patch("dify_ingestion.requests.post")
    def test_text_model_request_body(self, mock_post):
        """text_model 使用 automatic 规则。"""
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.txt"}, "batch": "b1",
        })
        create_document_by_file(
            "key", "http://localhost/v1", "ds-001",
            "a.txt", b"content",
            doc_form="text_model",
        )
        data_payload = json.loads(mock_post.call_args[1]["data"]["data"])
        assert data_payload["process_rule"] == {"mode": "automatic"}

    @patch("dify_ingestion.requests.post")
    def test_hierarchical_not_automatic(self, mock_post):
        """hierarchical_model 绝不能传 automatic。"""
        mock_post.return_value = _mock_response({
            "document": {"id": "d1", "name": "a.docx"}, "batch": "b1",
        })
        create_document_by_file(
            "key", "http://localhost/v1", "ds-001",
            "a.docx", b"PK\x03\x04",
            doc_form="hierarchical_model",
        )
        data_payload = json.loads(mock_post.call_args[1]["data"]["data"])
        assert data_payload["process_rule"]["mode"] != "automatic"
        assert "parent_mode" in data_payload["process_rule"]["rules"]

    @patch("dify_ingestion.requests.post")
    def test_no_document_in_response(self, mock_post):
        mock_post.return_value = _mock_response({"batch": "b1"})
        with pytest.raises(RuntimeError, match="未返回 document"):
            create_document_by_file(
                "key", "http://localhost/v1", "ds-001",
                "a.txt", b"content",
            )

    @patch("dify_ingestion.requests.post")
    def test_415_error(self, mock_post):
        mock_post.return_value = _mock_response(
            {"code": "unsupported_file_type"}, status_code=415
        )
        with pytest.raises(RuntimeError, match="HTTP 415"):
            create_document_by_file(
                "key", "http://localhost/v1", "ds-001",
                "data.xyz", b"content",
            )


class TestGetDocumentIndexingStatus:
    """索引状态查询。"""

    @patch("dify_ingestion.requests.get")
    def test_completed(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": {
                "id": "doc-001",
                "indexing_status": "completed",
                "completed_at": 1700000000,
                "error": None,
            }
        })
        result = get_document_indexing_status(
            "key", "http://localhost/v1", "ds-001", "batch-001",
        )
        assert result["indexing_status"] == "completed"
        assert result["id"] == "doc-001"
        assert result["error"] is None

    @patch("dify_ingestion.requests.get")
    def test_error_status(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": {
                "id": "doc-001",
                "indexing_status": "error",
                "error": "解析失败: 文件格式不支持",
            }
        })
        result = get_document_indexing_status(
            "key", "http://localhost/v1", "ds-001", "batch-001",
        )
        assert result["indexing_status"] == "error"
        assert "解析失败" in result["error"]

    @patch("dify_ingestion.requests.get")
    def test_parsing_status(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": {"id": "doc-001", "indexing_status": "parsing"}
        })
        result = get_document_indexing_status(
            "key", "http://localhost/v1", "ds-001", "batch-001",
        )
        assert result["indexing_status"] == "parsing"

    @patch("dify_ingestion.requests.get")
    def test_normalized_fields(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": {"id": "d1", "indexing_status": "completed", "error": None}
        })
        result = get_document_indexing_status(
            "key", "http://localhost/v1", "ds-001", "b1",
        )
        for key in ["id", "indexing_status", "processing_started_at",
                     "parsing_completed_at", "cleaning_completed_at",
                     "splitting_completed_at", "completed_at", "error"]:
            assert key in result

    @patch("dify_ingestion.requests.get")
    def test_list_response_format(self, mock_get):
        """Dify 有时返回 data 为列表。"""
        mock_get.return_value = _mock_response({
            "data": [{"id": "d1", "indexing_status": "splitting"}]
        })
        result = get_document_indexing_status(
            "key", "http://localhost/v1", "ds-001", "b1",
        )
        assert result["indexing_status"] == "splitting"


class TestGetDocumentSegments:
    """分段数量查询。"""

    @patch("dify_ingestion.requests.get")
    def test_total_field(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}],
            "total": 3,
        })
        count = get_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
        )
        assert count == 3

    @patch("dify_ingestion.requests.get")
    def test_fallback_to_data_length(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [{"id": "s1"}, {"id": "s2"}],
        })
        count = get_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
        )
        assert count == 2

    @patch("dify_ingestion.requests.get")
    def test_empty_segments(self, mock_get):
        mock_get.return_value = _mock_response({"data": [], "total": 0})
        count = get_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
        )
        assert count == 0


class TestIndexingStatusLabels:
    """状态文案映射。"""

    def test_known_statuses(self):
        assert INDEXING_STATUS_LABELS["waiting"] == "已提交，等待处理"
        assert INDEXING_STATUS_LABELS["parsing"] == "正在解析原始文件"
        assert INDEXING_STATUS_LABELS["cleaning"] == "正在清洗文本"
        assert INDEXING_STATUS_LABELS["splitting"] == "正在进行父子分块"
        assert INDEXING_STATUS_LABELS["indexing"] == "正在建立检索索引"
        assert INDEXING_STATUS_LABELS["completed"] == "索引完成"
        assert INDEXING_STATUS_LABELS["error"] == "索引失败"

    def test_all_statuses_present(self):
        assert len(INDEXING_STATUS_LABELS) == 7


class TestIngestionRecordNewFields:
    """入库记录新增字段兼容性。"""

    def test_with_all_new_fields(self):
        record = build_ingestion_record(
            dataset_id="ds-001",
            file_name="test.docx",
            content_hash="abc",
            document_id="doc-001",
            batch="batch-001",
            indexing_status="completed",
            segment_count=5,
            indexing_error="",
        )
        assert record["batch"] == "batch-001"
        assert record["indexing_status"] == "completed"
        assert record["segment_count"] == 5
        assert "indexing_error" not in record  # 空值不写入

    def test_without_new_fields(self):
        record = build_ingestion_record(
            dataset_id="ds-001",
            file_name="test.txt",
            content_hash="abc",
        )
        assert "batch" not in record
        assert "indexing_status" not in record
        assert "segment_count" not in record

    def test_error_record(self):
        record = build_ingestion_record(
            dataset_id="ds-001",
            file_name="test.docx",
            content_hash="abc",
            document_id="doc-001",
            batch="batch-001",
            indexing_status="error",
            segment_count=-1,
            indexing_error="解析失败",
        )
        assert record["indexing_status"] == "error"
        assert record["indexing_error"] == "解析失败"
        assert "segment_count" not in record  # -1 不写入

    def test_empty_segment_record(self):
        record = build_ingestion_record(
            dataset_id="ds-001",
            file_name="test.docx",
            content_hash="abc",
            ingestion_status="completed_empty",
            batch="batch-001",
            indexing_status="completed",
            segment_count=0,
        )
        assert record["segment_count"] == 0
        assert record["ingestion_status"] == "completed_empty"


class TestUploadPipelineFile:
    """Pipeline 文件上传。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({"id": "pf-001", "name": "a.docx"})
        result = upload_pipeline_file("key", "http://localhost/v1", "a.docx", b"PK\x03\x04")
        assert result["id"] == "pf-001"

    @patch("dify_ingestion.requests.post")
    def test_docx_mime_type(self, mock_post):
        mock_post.return_value = _mock_response({"id": "pf-001"})
        upload_pipeline_file("key", "http://localhost/v1", "contract.docx", b"PK\x03\x04")
        file_tuple = mock_post.call_args[1]["files"]["file"]
        assert file_tuple[2] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    @patch("dify_ingestion.requests.post")
    def test_no_id_raises(self, mock_post):
        mock_post.return_value = _mock_response({"name": "a.txt"})
        with pytest.raises(RuntimeError, match="未返回 id"):
            upload_pipeline_file("key", "http://localhost/v1", "a.txt", b"content")

    def test_empty_bytes_raises(self):
        with pytest.raises(RuntimeError, match="内容为空"):
            upload_pipeline_file("key", "http://localhost/v1", "a.txt", b"")


class TestListPipelineDatasourcePlugins:
    """Pipeline datasource 节点查询。"""

    @patch("dify_ingestion.requests.get")
    def test_returns_plugins(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [
                {"node_id": "n1", "datasource_type": "local_file"},
                {"node_id": "n2", "datasource_type": "web_crawler"},
            ]
        })
        plugins = list_pipeline_datasource_plugins("key", "http://localhost/v1", "ds-001")
        assert len(plugins) == 2

    @patch("dify_ingestion.requests.get")
    def test_empty(self, mock_get):
        mock_get.return_value = _mock_response({"data": []})
        plugins = list_pipeline_datasource_plugins("key", "http://localhost/v1", "ds-001")
        assert plugins == []


class TestFindLocalFileNodeId:
    """local_file 节点定位。"""

    @patch("dify_ingestion.requests.get")
    def test_unique_node(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [
                {"node_id": "n1", "datasource_type": "local_file"},
                {"node_id": "n2", "datasource_type": "web_crawler"},
            ]
        })
        node_id = find_local_file_node_id("key", "http://localhost/v1", "ds-001")
        assert node_id == "n1"

    @patch("dify_ingestion.requests.get")
    def test_no_local_file_node(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [{"node_id": "n1", "datasource_type": "web_crawler"}]
        })
        with pytest.raises(RuntimeError, match="未找到"):
            find_local_file_node_id("key", "http://localhost/v1", "ds-001")

    @patch("dify_ingestion.requests.get")
    def test_multiple_local_file_nodes(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [
                {"node_id": "n1", "datasource_type": "local_file"},
                {"node_id": "n2", "datasource_type": "local_file"},
            ]
        })
        with pytest.raises(RuntimeError, match="无法确定唯一节点"):
            find_local_file_node_id("key", "http://localhost/v1", "ds-001")


class TestRunKnowledgePipeline:
    """Pipeline 运行。"""

    @patch("dify_ingestion.requests.post")
    def test_request_body(self, mock_post):
        mock_post.return_value = _mock_response({
            "data": {"status": "succeeded"}
        })
        run_knowledge_pipeline(
            "key", "http://localhost/v1", "ds-001",
            "pf-001", "contract.docx", "n1",
        )
        body = mock_post.call_args[1]["json"]
        assert body["is_published"] is True
        assert body["response_mode"] == "blocking"
        assert body["datasource_type"] == "local_file"
        assert body["start_node_id"] == "n1"
        assert body["datasource_info_list"][0]["reference"] == "pf-001"
        assert body["datasource_info_list"][0]["name"] == "contract.docx"


class TestExtractDocumentIdFromPipeline:
    """从 Pipeline 结果提取 document_id。"""

    def test_top_level_key(self):
        result = {"document_id": "doc-001"}
        assert _extract_document_id_from_pipeline(result) == "doc-001"

    def test_outputs_key(self):
        result = {"outputs": {"document_id": "doc-002"}}
        assert _extract_document_id_from_pipeline(result) == "doc-002"

    def test_document_ids_list(self):
        result = {"outputs": {"document_ids": ["doc-003"]}}
        assert _extract_document_id_from_pipeline(result) == "doc-003"

    def test_empty_result(self):
        assert _extract_document_id_from_pipeline({}) == ""

    def test_multiple_document_ids(self):
        """多个 document_id 无法确定，返回空。"""
        result = {"outputs": {"document_ids": ["doc-003", "doc-004"]}}
        assert _extract_document_id_from_pipeline(result) == ""


class TestFindDocumentByName:
    """通过文件名查找文档。"""

    @patch("dify_ingestion.requests.get")
    def test_unique_match(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [
                {"id": "d1", "name": "contract.docx", "enabled": True},
                {"id": "d2", "name": "other.docx", "enabled": True},
            ]
        })
        doc_id = find_document_by_name("key", "http://localhost/v1", "ds-001", "contract.docx")
        assert doc_id == "d1"

    @patch("dify_ingestion.requests.get")
    def test_no_match(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [{"id": "d1", "name": "other.docx", "enabled": True}]
        })
        assert find_document_by_name("key", "http://localhost/v1", "ds-001", "contract.docx") == ""

    @patch("dify_ingestion.requests.get")
    def test_multiple_matches(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [
                {"id": "d1", "name": "contract.docx", "enabled": True, "created_at": 100},
                {"id": "d2", "name": "contract.docx", "enabled": True, "created_at": 50},
            ]
        })
        assert find_document_by_name("key", "http://localhost/v1", "ds-001", "contract.docx") == "d1"

    @patch("dify_ingestion.requests.get")
    def test_disabled_doc_ignored(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [{"id": "d1", "name": "contract.docx", "enabled": False}]
        })
        assert find_document_by_name("key", "http://localhost/v1", "ds-001", "contract.docx") == ""


class TestTryPipelineIngestion:
    """Pipeline 入口整合。"""

    @patch("dify_ingestion.run_knowledge_pipeline")
    @patch("dify_ingestion.upload_pipeline_file")
    @patch("dify_ingestion.find_local_file_node_id", return_value="n1")
    def test_success_with_document_id(self, mock_find, mock_upload, mock_run):
        mock_upload.return_value = {"id": "pf-001"}
        mock_run.return_value = {"outputs": {"document_id": "doc-001"}}
        ok, doc_id, _, mode = try_pipeline_ingestion(
            "key", "http://localhost/v1", "ds-001", "a.docx", b"content",
        )
        assert ok is True
        assert doc_id == "doc-001"
        assert mode == "pipeline"

    @patch("dify_ingestion.run_knowledge_pipeline")
    @patch("dify_ingestion.upload_pipeline_file")
    @patch("dify_ingestion.find_local_file_node_id", return_value="n1")
    def test_no_document_id(self, mock_find, mock_upload, mock_run):
        mock_upload.return_value = {"id": "pf-001"}
        mock_run.return_value = {"outputs": {}}
        ok, msg, _, mode = try_pipeline_ingestion(
            "key", "http://localhost/v1", "ds-001", "a.docx", b"content",
        )
        assert ok is False
        assert "无法唯一确认" in msg
        assert mode == "pipeline_no_doc_id"

    @patch("dify_ingestion.find_local_file_node_id", side_effect=RuntimeError("未找到"))
    def test_fallback_when_no_pipeline(self, mock_find):
        ok, msg, _, mode = try_pipeline_ingestion(
            "key", "http://localhost/v1", "ds-001", "a.docx", b"content",
        )
        assert ok is False
        assert "未找到" in msg
        assert mode == "fallback"


class TestWaitForDocumentSegments:
    """分段轮询等待。"""

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("dify_ingestion.get_document_segments")
    def test_first_zero_then_success(self, mock_seg, mock_time, mock_sleep):
        """第一次返回 0，第二次返回 19，最终 completed。"""
        mock_time.side_effect = [0, 2, 4]
        mock_seg.side_effect = [0, 19]
        result = wait_for_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
            timeout=120, interval=2,
        )
        assert result["status"] == "completed"
        assert result["segment_count"] == 19

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("dify_ingestion.get_document_segments")
    def test_timeout_always_zero(self, mock_seg, mock_time, mock_sleep):
        """连续返回 0 直到超时，最终 processing。"""
        # 每次迭代: monotonic() → segments() → check → sleep
        # 超时那次: monotonic() → segments() → check(>=timeout) → return
        mock_time.side_effect = [0, 2, 4, 120]
        mock_seg.side_effect = [0, 0, 0]
        result = wait_for_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
            timeout=120, interval=2,
        )
        assert result["status"] == "processing"
        assert result["segment_count"] == 0

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("dify_ingestion.get_document_segments")
    def test_first_error_then_success(self, mock_seg, mock_time, mock_sleep):
        """第一次抛 RuntimeError，之后返回 19，最终成功。"""
        mock_time.side_effect = [0, 2, 4, 6]
        mock_seg.side_effect = [RuntimeError("连接超时"), 19]
        result = wait_for_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
            timeout=120, interval=2,
        )
        assert result["status"] == "completed"
        assert result["segment_count"] == 19

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("dify_ingestion.get_document_segments")
    def test_always_error(self, mock_seg, mock_time, mock_sleep):
        """查询始终抛 RuntimeError，最终 poll_error。"""
        mock_time.side_effect = [0, 2, 4, 120, 122]
        mock_seg.side_effect = RuntimeError("持续失败")
        result = wait_for_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
            timeout=120, interval=2,
        )
        assert result["status"] == "poll_error"
        assert "持续失败" in result["error"]

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("dify_ingestion.get_document_segments")
    def test_on_progress_called(self, mock_seg, mock_time, mock_sleep):
        """on_progress 能收到 elapsed 和当前 segment_count。"""
        mock_time.side_effect = [0, 2, 4]
        mock_seg.side_effect = [0, 5]
        progress_calls = []
        wait_for_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
            timeout=120, interval=2,
            on_progress=lambda e, s: progress_calls.append((e, s)),
        )
        assert len(progress_calls) == 2
        assert progress_calls[0] == (2, 0)
        assert progress_calls[1] == (4, 5)

    @patch("time.sleep")
    @patch("time.monotonic")
    @patch("dify_ingestion.get_document_segments")
    def test_immediate_success(self, mock_seg, mock_time, mock_sleep):
        """第一次就返回 > 0，立即完成。"""
        mock_time.side_effect = [0, 0]
        mock_seg.return_value = 19
        result = wait_for_document_segments(
            "key", "http://localhost/v1", "ds-001", "doc-001",
            timeout=120, interval=2,
        )
        assert result["status"] == "completed"
        assert result["segment_count"] == 19
        mock_sleep.assert_not_called()


class TestGetDatasetInfo:
    """获取单个知识库详情（含 doc_form）。"""

    @patch("dify_ingestion.requests.get")
    def test_found(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [
                {"id": "ds-001", "name": "知识库A", "doc_form": "hierarchical_model"},
                {"id": "ds-002", "name": "知识库B", "doc_form": "text_model"},
            ]
        })
        info = get_dataset_info("key", "http://localhost/v1", "ds-001")
        assert info["id"] == "ds-001"
        assert info["doc_form"] == "hierarchical_model"

    @patch("dify_ingestion.requests.get")
    def test_not_found(self, mock_get):
        mock_get.return_value = _mock_response({"data": [
            {"id": "ds-001", "name": "A"},
        ]})
        with pytest.raises(RuntimeError, match="未找到"):
            get_dataset_info("key", "http://localhost/v1", "ds-999")

    @patch("dify_ingestion.requests.get")
    def test_doc_form_preserved(self, mock_get):
        mock_get.return_value = _mock_response({
            "data": [{"id": "ds-001", "name": "A", "doc_form": "text_model"}]
        })
        info = get_dataset_info("key", "http://localhost/v1", "ds-001")
        assert "doc_form" in info


# ── TestNoRetry ───────────────────────────────────────────────


class TestNoRetry:
    """创建文档不自动重试（由调用方决定）。"""

    @patch("dify_ingestion.requests.post")
    def test_failure_raises_immediately(self, mock_post):
        mock_post.return_value = _mock_response(
            {"message": "server error"}, status_code=500
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            create_document(
                "dataset-key", "http://localhost/v1",
                "ds-001", "test.txt", "内容"
            )
        # 只调用了一次，没有重试
        assert mock_post.call_count == 1


# ── TestListMetadataFields ────────────────────────────────────


class TestListMetadataFields:
    """Metadata 字段查询。"""

    @patch("dify_ingestion.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response({
            "doc_metadata": [
                {"id": "m1", "name": "category", "type": "string", "count": 5},
                {"id": "m2", "name": "priority", "type": "number", "count": 3},
            ]
        })
        fields = list_metadata_fields("dataset-key", "http://localhost/v1", "ds-001")
        assert len(fields) == 2
        assert fields[0]["name"] == "category"

    @patch("dify_ingestion.requests.get")
    def test_empty(self, mock_get):
        mock_get.return_value = _mock_response({"doc_metadata": []})
        fields = list_metadata_fields("dataset-key", "http://localhost/v1", "ds-001")
        assert fields == []


# ── TestBindDocumentMetadata ──────────────────────────────────


class TestBindDocumentMetadata:
    """文档 metadata 绑定。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({"result": "success"})
        result = bind_document_metadata(
            "dataset-key", "http://localhost/v1",
            "ds-001", "doc-001",
            [{"id": "m1", "name": "category", "value": "合同"}]
        )
        assert result["result"] == "success"

        # 验证请求体
        body = mock_post.call_args[1]["json"]
        op = body["operation_data"][0]
        assert op["document_id"] == "doc-001"
        assert op["partial_update"] is True


# ── TestCreateMetadataField ──────────────────────────────────


class TestCreateMetadataField:
    """创建单个 metadata 字段。"""

    @patch("dify_ingestion.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({
            "id": "meta-001", "name": "contract_package", "type": "string"
        })
        result = create_metadata_field(
            "dataset-key", "http://localhost/v1", "ds-001",
            "contract_package", "string",
        )
        assert result["id"] == "meta-001"
        assert result["name"] == "contract_package"

    @patch("dify_ingestion.requests.post")
    def test_request_body(self, mock_post):
        mock_post.return_value = _mock_response({
            "id": "m1", "name": "topics", "type": "string"
        })
        create_metadata_field(
            "dataset-key", "http://localhost/v1", "ds-001", "topics"
        )
        body = mock_post.call_args[1]["json"]
        assert body == {"name": "topics", "type": "string"}

    @patch("dify_ingestion.requests.post")
    def test_failure(self, mock_post):
        mock_post.return_value = _mock_response(
            {"message": "duplicate"}, status_code=409
        )
        with pytest.raises(RuntimeError, match="HTTP 409"):
            create_metadata_field(
                "dataset-key", "http://localhost/v1", "ds-001", "category"
            )


# ── TestEnsureRequiredMetadataFields ─────────────────────────


class TestEnsureRequiredMetadataFields:
    """ensure_required_metadata_fields 批量初始化。"""

    def test_all_missing(self):
        """全部缺失 → 创建全部 6 个。"""
        created_fields = [
            {"id": f"m{i}", "name": f["name"], "type": f["type"]}
            for i, f in enumerate(REQUIRED_METADATA_FIELDS)
        ]
        with patch("dify_ingestion.list_metadata_fields", return_value=[]):
            with patch("dify_ingestion.create_metadata_field",
                       side_effect=created_fields) as mock_create:
                created, errors = ensure_required_metadata_fields(
                    "key", "http://localhost/v1", "ds-001"
                )
        assert len(created) == 6
        assert errors == []
        assert mock_create.call_count == 6

    def test_all_exist(self):
        """全部已存在 → 不创建任何字段。"""
        existing = [
            {"id": f"m{i}", "name": f["name"], "type": f["type"], "count": 0}
            for i, f in enumerate(REQUIRED_METADATA_FIELDS)
        ]
        with patch("dify_ingestion.list_metadata_fields", return_value=existing):
            with patch("dify_ingestion.create_metadata_field") as mock_create:
                created, errors = ensure_required_metadata_fields(
                    "key", "http://localhost/v1", "ds-001"
                )
        assert created == []
        assert errors == []
        mock_create.assert_not_called()

    def test_partial_missing(self):
        """部分缺失 → 只创建缺失的。"""
        existing = [
            {"id": "m0", "name": "contract_package", "type": "string", "count": 5},
            {"id": "m1", "name": "document_type", "type": "string", "count": 3},
        ]
        with patch("dify_ingestion.list_metadata_fields", return_value=existing):
            with patch("dify_ingestion.create_metadata_field",
                       return_value={"id": "m_new", "name": "x", "type": "string"}) as mock_create:
                created, errors = ensure_required_metadata_fields(
                    "key", "http://localhost/v1", "ds-001"
                )
        assert mock_create.call_count == 4  # 6 - 2 existing
        assert errors == []

    def test_creation_failure_no_overwrite(self):
        """某个字段创建失败 → 记录错误，不影响其他字段。"""
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("server error")
            return {"id": f"m{call_count[0]}", "name": "x", "type": "string"}

        with patch("dify_ingestion.list_metadata_fields", return_value=[]):
            with patch("dify_ingestion.create_metadata_field",
                       side_effect=side_effect):
                created, errors = ensure_required_metadata_fields(
                    "key", "http://localhost/v1", "ds-001"
                )
        assert len(created) == 5  # 6 - 1 failed
        assert len(errors) == 1
        assert "server error" in errors[0]

    def test_existing_fields_not_modified(self):
        """已存在字段绝不被修改或删除。"""
        existing = [
            {"id": "m_keep", "name": "contract_package", "type": "string", "count": 10},
        ]
        with patch("dify_ingestion.list_metadata_fields", return_value=existing):
            with patch("dify_ingestion.create_metadata_field",
                       return_value={"id": "m_new", "name": "x", "type": "string"}):
                created, errors = ensure_required_metadata_fields(
                    "key", "http://localhost/v1", "ds-001"
                )
        # contract_package 已存在，不会被调用创建
        assert len(created) == 5

    def test_required_fields_constant(self):
        """REQUIRED_METADATA_FIELDS 包含且仅包含 6 个字段。"""
        assert len(REQUIRED_METADATA_FIELDS) == 6
        names = {f["name"] for f in REQUIRED_METADATA_FIELDS}
        assert names == {
            "contract_package", "document_type", "document_title",
            "document_language", "document_summary", "topics",
        }
        # 全部为 string 类型
        for f in REQUIRED_METADATA_FIELDS:
            assert f["type"] == "string"


# ── TestFullIngestionFlow ─────────────────────────────────────


class TestFullIngestionFlow:
    """完整入库流程 mock 测试。"""

    @patch("dify_ingestion.requests.get")
    @patch("dify_ingestion.requests.post")
    def test_end_to_end(self, mock_post, mock_get, _use_tmp_history):
        """模拟：上传→运行workflow→校验→创建文档→绑定metadata→记录历史。"""
        # 1. 上传文件
        mock_post.return_value = _mock_response({"id": "file-001"})
        tmp_file = _use_tmp_history / "test.pdf"
        tmp_file.write_bytes(b"test pdf content")
        file_id = upload_file("app-key", "http://localhost/v1", str(tmp_file))
        assert file_id == "file-001"

        # 2. 运行 Workflow
        mock_post.return_value = _mock_response({
            "data": {
                "status": "succeeded",
                "outputs": [_sample_workflow_result()],
            }
        })
        outputs = run_workflow(
            "app-key", "http://localhost/v1",
            [file_id], "baseline_2_4"
        )

        # 3. 校验结果
        results = validate_workflow_outputs(outputs, "baseline_2_4")
        assert results[0]["ok"] is True
        cleaned = results[0]["cleaned"]

        # 4. 创建文档
        mock_post.return_value = _mock_response({
            "document": {"id": "doc-001", "name": "test.pdf"},
            "batch": "batch-001",
        })
        doc_result = create_document(
            "dataset-key", "http://localhost/v1",
            "ds-001", cleaned["document_title"], "文档内容"
        )
        document_id = doc_result["document"]["id"]

        # 5. 查询 metadata 字段
        mock_get.return_value = _mock_response({
            "doc_metadata": [
                {"id": "m1", "name": "contract_package", "type": "string"},
                {"id": "m2", "name": "document_type", "type": "string"},
                {"id": "m3", "name": "document_title", "type": "string"},
                {"id": "m4", "name": "document_language", "type": "string"},
                {"id": "m5", "name": "document_summary", "type": "string"},
                {"id": "m6", "name": "topics", "type": "string"},
            ]
        })
        fields = list_metadata_fields("dataset-key", "http://localhost/v1", "ds-001")
        field_map = {f["name"]: f["id"] for f in fields}

        # 6. 绑定 metadata
        metadata_items = [
            {"id": field_map["contract_package"], "name": "contract_package", "value": cleaned["contract_package"]},
            {"id": field_map["document_type"], "name": "document_type", "value": cleaned["document_type"]},
            {"id": field_map["document_title"], "name": "document_title", "value": cleaned["document_title"]},
            {"id": field_map["document_language"], "name": "document_language", "value": cleaned["document_language"]},
            {"id": field_map["document_summary"], "name": "document_summary", "value": cleaned["document_summary"]},
            {"id": field_map["topics"], "name": "topics", "value": json.dumps(cleaned["topics"], ensure_ascii=False)},
        ]
        mock_post.return_value = _mock_response({"result": "success"})
        bind_result = bind_document_metadata(
            "dataset-key", "http://localhost/v1",
            "ds-001", document_id, metadata_items
        )
        assert bind_result["result"] == "success"

        # 7. 记录历史
        content_hash = compute_content_hash("文档内容")
        record = build_ingestion_record(
            dataset_id="ds-001",
            file_name="test.pdf",
            content_hash=content_hash,
            document_id=document_id,
            metadata=cleaned,
            workflow_status="success",
            ingestion_status="success",
        )
        append_ingestion_record(record)

        # 验证历史记录
        history = load_ingestion_history("ds-001")
        assert len(history) == 1
        assert history[0]["document_id"] == "doc-001"
        assert "api_key" not in history[0]

    @patch("dify_ingestion.requests.post")
    def test_single_failure_does_not_block_batch(self, mock_post, _use_tmp_history):
        """单文件失败不阻断批次中其他文件。"""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一个文件：创建失败
                return _mock_response({"message": "server error"}, status_code=500)
            # 第二个文件：创建成功
            return _mock_response({
                "document": {"id": "doc-002", "name": "file2.txt"},
                "batch": "b2",
            })

        mock_post.side_effect = side_effect

        results = []
        for fname, content in [("file1.txt", "内容一"), ("file2.txt", "内容二")]:
            try:
                r = create_document(
                    "dataset-key", "http://localhost/v1",
                    "ds-001", fname, content
                )
                results.append({"file": fname, "ok": True, "doc_id": r["document"]["id"]})
            except RuntimeError as exc:
                results.append({"file": fname, "ok": False, "error": str(exc)})

        assert results[0]["ok"] is False
        assert results[1]["ok"] is True
        assert results[1]["doc_id"] == "doc-002"


# ── TestHttpErrorClassification ───────────────────────────────


class TestHttpErrorClassification:
    """HTTP 错误分类。"""

    @patch("dify_ingestion.requests.post")
    def test_401_error(self, mock_post):
        mock_post.return_value = _mock_response(
            {"message": "unauthorized"}, status_code=401
        )
        with pytest.raises(RuntimeError, match="认证失败"):
            _post_json_for_test("app-key123", "http://localhost/v1", "/test", {})

    @patch("dify_ingestion.requests.post")
    def test_404_error(self, mock_post):
        mock_post.return_value = _mock_response(
            {"message": "not found"}, status_code=404
        )
        with pytest.raises(RuntimeError, match="不存在"):
            _post_json_for_test("app-key", "http://localhost/v1", "/test", {})

    @patch("dify_ingestion.requests.post")
    def test_413_error(self, mock_post):
        mock_post.return_value = _mock_response(
            {"message": "too large"}, status_code=413
        )
        with pytest.raises(RuntimeError, match="文件过大"):
            _post_json_for_test("app-key", "http://localhost/v1", "/test", {})


def _post_json_for_test(api_key, base_url, path, body):
    """测试辅助：调用 _post_json。"""
    from dify_ingestion import _post_json
    return _post_json(api_key, base_url, path, body)


# ── TestRunAutoIngestionWorkflow ──────────────────────────────


class TestRunAutoIngestionWorkflow:
    """全流程入库 Workflow 调用测试。"""

    @patch("dify_ingestion.requests.post")
    def test_success_blocking_mode(self, mock_post):
        mock_post.return_value = _mock_response({
            "data": {
                "status": "succeeded",
                "outputs": {
                    "results": [
                        {
                            "document_id": "doc-auto-001",
                            "contract_package": "baseline_2_4",
                            "document_title": "测试主协议",
                            "indexing_status": "waiting",
                        }
                    ]
                },
            }
        })
        outputs = run_auto_ingestion_workflow(
            "app-key", "http://localhost/v1",
            ["f1", "f2"], "baseline_2_4", dataset_id="ds-123"
        )
        assert "results" in outputs
        assert outputs["results"][0]["document_id"] == "doc-auto-001"

        # 检查请求体
        body = mock_post.call_args[1]["json"]
        assert body["response_mode"] == "blocking"
        assert body["inputs"]["contract_package"] == "baseline_2_4"
        assert body["inputs"]["dataset_id"] == "ds-123"
        assert len(body["inputs"]["files"]) == 2
        assert body["inputs"]["files"][0]["upload_file_id"] == "f1"
        assert body["inputs"]["files"][0]["transfer_method"] == "local_file"

    def test_invalid_package_raises(self):
        with pytest.raises(ValueError, match="不支持的合同包"):
            run_auto_ingestion_workflow(
                "app-key", "http://localhost/v1", ["f1"], "invalid_pkg", dataset_id="ds-123"
            )

    def test_missing_dataset_id_raises(self):
        with pytest.raises(ValueError, match="必须指定目标知识库"):
            run_auto_ingestion_workflow(
                "app-key", "http://localhost/v1", ["f1"], "baseline_2_4", dataset_id=""
            )

    @patch("dify_ingestion.requests.post")
    def test_workflow_failed_raises(self, mock_post):
        mock_post.return_value = _mock_response({
            "data": {"status": "failed", "error": "Pipeline 节点执行超时"}
        })
        with pytest.raises(RuntimeError, match="执行失败"):
            run_auto_ingestion_workflow(
                "app-key", "http://localhost/v1", ["f1"], "baseline_2_4", dataset_id="ds-123"
            )

    @patch("dify_ingestion.requests.post")
    def test_no_outputs_raises(self, mock_post):
        mock_post.return_value = _mock_response({
            "data": {"status": "succeeded", "outputs": None}
        })
        with pytest.raises(RuntimeError, match="未返回 outputs"):
            run_auto_ingestion_workflow(
                "app-key", "http://localhost/v1", ["f1"], "baseline_2_4", dataset_id="ds-123"
            )


# ── TestParseAutoIngestionOutputs ─────────────────────────────


class TestParseAutoIngestionOutputs:
    """全流程入库输出解析器测试。"""

    def test_results_key_format(self):
        outputs = {
            "results": [
                {
                    "file_name": "contract.docx",
                    "document_id": "doc-001",
                    "contract_package": "baseline_2_4",
                    "document_type": "主协议",
                    "document_title": "IT采购框架协议",
                    "document_language": "中英双语",
                    "document_summary": "测试摘要",
                    "topics": ["IT采购", "保密", "安全"],
                    "indexing_status": "waiting",
                    "batch": "batch-001",
                }
            ]
        }
        parsed = parse_auto_ingestion_outputs(outputs, "baseline_2_4", ["contract.docx"])
        assert len(parsed) == 1
        res = parsed[0]
        assert res["success"] is True
        assert res["document_id"] == "doc-001"
        assert res["file_name"] == "contract.docx"
        assert res["document_title"] == "IT采购框架协议"
        assert res["contract_package"] == "baseline_2_4"
        assert res["indexing_status"] == "waiting"
        assert res["error"] == ""

    def test_output_key_format(self):
        outputs = {
            "output": [
                {
                    "document_id": "doc-002",
                    "document_title": "DPA协议",
                    "topics": '["数据处理", "安全事件"]',
                    "indexing_status": "completed",
                }
            ]
        }
        parsed = parse_auto_ingestion_outputs(outputs, "tech_platform_2_5", ["dpa.docx"])
        assert len(parsed) == 1
        res = parsed[0]
        assert res["success"] is True
        assert res["document_id"] == "doc-002"
        assert res["topics"] == ["数据处理", "安全事件"]
        assert res["file_name"] == "dpa.docx"

    def test_stringified_json_outputs(self):
        outputs_str = json.dumps({
            "results": [
                {
                    "document_id": "doc-003",
                    "document_title": "附录A",
                    "indexing_status": "waiting",
                }
            ]
        })
        parsed = parse_auto_ingestion_outputs(outputs_str, "baseline_2_4")
        assert len(parsed) == 1
        assert parsed[0]["success"] is True
        assert parsed[0]["document_id"] == "doc-003"

    def test_missing_document_id_marks_failure(self):
        outputs = {
            "results": [
                {
                    "file_name": "bad.docx",
                    "document_title": "失败文件",
                    "document_id": "",
                }
            ]
        }
        parsed = parse_auto_ingestion_outputs(outputs, "baseline_2_4")
        assert len(parsed) == 1
        assert parsed[0]["success"] is False
        assert "document_id" in parsed[0]["error"]

    def test_explicit_error_in_item(self):
        outputs = {
            "results": [
                {
                    "file_name": "err.docx",
                    "document_id": "doc-004",
                    "error": "Pipeline 执行崩溃",
                }
            ]
        }
        parsed = parse_auto_ingestion_outputs(outputs, "baseline_2_4")
        assert len(parsed) == 1
        assert parsed[0]["success"] is False
        assert "Pipeline 执行崩溃" in parsed[0]["error"]

    def test_direct_list_format(self):
        outputs = [
            {"document_id": "doc-1", "document_title": "文件1"},
            {"document_id": "doc-2", "document_title": "文件2"},
        ]
        parsed = parse_auto_ingestion_outputs(outputs, "baseline_2_4", ["f1.docx", "f2.docx"])
        assert len(parsed) == 2
        assert parsed[0]["document_id"] == "doc-1"
        assert parsed[0]["file_name"] == "f1.docx"
        assert parsed[1]["document_id"] == "doc-2"
        assert parsed[1]["file_name"] == "f2.docx"

