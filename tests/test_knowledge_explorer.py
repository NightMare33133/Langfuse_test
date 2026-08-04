"""
知识库探索模块测试。

测试内容：
1. list_datasets — 请求构造与响应解析
2. list_documents — 分页参数与响应结构
3. list_segments — status 过滤与分页
4. compute_content_hash — 规范化正确性
5. build_chunk_catalog — 字段完整性
6. detect_duplicates — 重复检测
7. export_catalog_json / export_catalog_csv — 格式正确性
8. API Key 不泄露到导出内容
"""

import json
import sys
import hashlib
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from dify_knowledge import (
    _get,
    list_datasets,
    list_documents,
    list_segments,
    list_all_documents,
    list_all_segments,
    retrieve,
    compute_content_hash,
    build_chunk_catalog,
    detect_duplicates,
    export_catalog_json,
    export_catalog_csv,
    export_full_kb_json,
    export_full_kb_csv,
    build_full_kb_catalog,
    check_connection,
    _EXPORT_COLUMNS,
)


# ── Fixtures ──────────────────────────────────────────────────


def _mock_response(json_data, status_code=200):
    """构造 mock requests.Response。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    return resp


def _sample_segments():
    """返回一组测试用 segment 数据。"""
    return [
        {
            "id": "seg_001",
            "position": 1,
            "document_id": "doc_abc",
            "content": "这是第一段内容。",
            "index_node_id": "node_001",
            "index_node_hash": "hash_001",
            "tokens": 50,
            "word_count": 7,
            "enabled": True,
            "status": "completed",
        },
        {
            "id": "seg_002",
            "position": 2,
            "document_id": "doc_abc",
            "content": "这是第二段内容。",
            "index_node_id": "node_002",
            "index_node_hash": "hash_002",
            "tokens": 60,
            "word_count": 7,
            "enabled": True,
            "status": "completed",
        },
        {
            "id": "seg_003",
            "position": 3,
            "document_id": "doc_abc",
            "content": "这是第一段内容。",  # 与 seg_001 重复
            "index_node_id": "node_003",
            "index_node_hash": "hash_003",
            "tokens": 50,
            "word_count": 7,
            "enabled": True,
            "status": "completed",
        },
    ]


# ── _get 基础请求 ────────────────────────────────────────────


class TestGetRequest:
    """测试 _get 函数的请求构造和错误处理。"""

    @patch("dify_knowledge.requests.get")
    def test_get_constructs_correct_url(self, mock_get):
        """URL 拼接正确（去除尾部斜杠）。"""
        mock_get.return_value = _mock_response({"ok": True})
        _get("test-key", "http://localhost/v1", "/datasets", params={"page": 1})
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost/v1/datasets"

    @patch("dify_knowledge.requests.get")
    def test_get_sends_auth_header(self, mock_get):
        """Authorization header 包含 Bearer token。"""
        mock_get.return_value = _mock_response({"ok": True})
        _get("my-secret-key", "http://localhost/v1", "/datasets")
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    @patch("dify_knowledge.requests.get")
    def test_get_returns_json(self, mock_get):
        """正常响应返回解析后的 JSON。"""
        data = {"data": [{"id": "1"}], "total": 1}
        mock_get.return_value = _mock_response(data)
        result = _get("key", "http://localhost/v1", "/test")
        assert result == data

    @patch("dify_knowledge.requests.get")
    def test_get_raises_on_non_200(self, mock_get):
        """非 200 状态码抛出 RuntimeError。"""
        mock_get.return_value = _mock_response({"error": "forbidden"}, status_code=403)
        with pytest.raises(RuntimeError, match="HTTP 403"):
            _get("key", "http://localhost/v1", "/test")

    @patch("dify_knowledge.requests.get")
    def test_get_raises_on_timeout(self, mock_get):
        """超时抛出 RuntimeError。"""
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()
        with pytest.raises(RuntimeError, match="请求超时"):
            _get("key", "http://localhost/v1", "/test")

    @patch("dify_knowledge.requests.get")
    def test_get_raises_on_connection_error(self, mock_get):
        """连接失败抛出 RuntimeError。"""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        with pytest.raises(RuntimeError, match="连接失败"):
            _get("key", "http://localhost/v1", "/test")


# ── list_datasets ─────────────────────────────────────────────


class TestListDatasets:
    """测试知识库列表接口。"""

    @patch("dify_knowledge.requests.get")
    def test_returns_data_list(self, mock_get):
        """返回 datasets 列表。"""
        datasets = [
            {"id": "ds1", "name": "知识库A", "document_count": 10},
            {"id": "ds2", "name": "知识库B", "document_count": 5},
        ]
        mock_get.return_value = _mock_response({"data": datasets})
        result = list_datasets("key", "http://localhost/v1")
        assert len(result) == 2
        assert result[0]["id"] == "ds1"

    @patch("dify_knowledge.requests.get")
    def test_empty_datasets(self, mock_get):
        """空知识库返回空列表。"""
        mock_get.return_value = _mock_response({"data": []})
        result = list_datasets("key", "http://localhost/v1")
        assert result == []

    @patch("dify_knowledge.requests.get")
    def test_passes_pagination_params(self, mock_get):
        """传递 page=1, limit=100 参数。"""
        mock_get.return_value = _mock_response({"data": []})
        list_datasets("key", "http://localhost/v1")
        call_args = mock_get.call_args
        assert call_args[1]["params"]["page"] == 1
        assert call_args[1]["params"]["limit"] == 100


# ── list_documents ────────────────────────────────────────────


class TestListDocuments:
    """测试文档列表接口。"""

    @patch("dify_knowledge.requests.get")
    def test_returns_paginated_result(self, mock_get):
        """返回包含 data, has_more, total 的分页结果。"""
        docs = [{"id": "d1", "name": "doc1", "status": "completed"}]
        mock_get.return_value = _mock_response({
            "data": docs, "has_more": False, "total": 1,
        })
        result = list_documents("key", "http://localhost/v1", "ds1")
        assert result["data"] == docs
        assert result["has_more"] is False
        assert result["total"] == 1

    @patch("dify_knowledge.requests.get")
    def test_passes_correct_path(self, mock_get):
        """请求路径包含 dataset_id。"""
        mock_get.return_value = _mock_response({"data": [], "has_more": False, "total": 0})
        list_documents("key", "http://localhost/v1", "ds_abc", page=2, limit=50)
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://localhost/v1/datasets/ds_abc/documents"
        assert call_args[1]["params"]["page"] == 2
        assert call_args[1]["params"]["limit"] == 50

    @patch("dify_knowledge.requests.get")
    def test_limit_clamped_to_100(self, mock_get):
        """limit 最大为 100。"""
        mock_get.return_value = _mock_response({"data": [], "has_more": False, "total": 0})
        list_documents("key", "http://localhost/v1", "ds1", limit=500)
        call_args = mock_get.call_args
        assert call_args[1]["params"]["limit"] == 100

    @patch("dify_knowledge.requests.get")
    def test_limit_minimum_1(self, mock_get):
        """limit 最小为 1。"""
        mock_get.return_value = _mock_response({"data": [], "has_more": False, "total": 0})
        list_documents("key", "http://localhost/v1", "ds1", limit=0)
        call_args = mock_get.call_args
        assert call_args[1]["params"]["limit"] == 1


# ── list_segments ─────────────────────────────────────────────


class TestListSegments:
    """测试分块列表接口。"""

    @patch("dify_knowledge.requests.get")
    def test_passes_status_filter(self, mock_get):
        """status_filter 参数正确传递。"""
        mock_get.return_value = _mock_response({"data": [], "has_more": False, "total": 0})
        list_segments("key", "http://localhost/v1", "ds1", "doc1",
                      status_filter="completed")
        call_args = mock_get.call_args
        assert call_args[1]["params"]["status"] == "completed"

    @patch("dify_knowledge.requests.get")
    def test_empty_status_filter_not_sent(self, mock_get):
        """status_filter 为空时不传递 status 参数。"""
        mock_get.return_value = _mock_response({"data": [], "has_more": False, "total": 0})
        list_segments("key", "http://localhost/v1", "ds1", "doc1",
                      status_filter="")
        call_args = mock_get.call_args
        assert "status" not in call_args[1]["params"]

    @patch("dify_knowledge.requests.get")
    def test_correct_path(self, mock_get):
        """请求路径包含 dataset_id 和 document_id。"""
        mock_get.return_value = _mock_response({"data": [], "has_more": False, "total": 0})
        list_segments("key", "http://localhost/v1", "ds1", "doc1")
        call_args = mock_get.call_args
        expected = "http://localhost/v1/datasets/ds1/documents/doc1/segments"
        assert call_args[0][0] == expected

    @patch("dify_knowledge.requests.get")
    def test_returns_segments(self, mock_get):
        """返回分块数据。"""
        segs = _sample_segments()
        mock_get.return_value = _mock_response({
            "data": segs, "has_more": True, "total": 100,
        })
        result = list_segments("key", "http://localhost/v1", "ds1", "doc1")
        assert len(result["data"]) == 3
        assert result["has_more"] is True
        assert result["total"] == 100


# ── compute_content_hash ─────────────────────────────────────


class TestContentHash:
    """测试内容哈希的规范化行为。"""

    def test_basic_hash(self):
        """基本哈希计算正确。"""
        content = "Hello World"
        expected = hashlib.sha256(b"Hello World").hexdigest()
        assert compute_content_hash(content) == expected

    def test_strips_whitespace(self):
        """首尾空白不影响哈希。"""
        h1 = compute_content_hash("  hello  ")
        h2 = compute_content_hash("hello")
        assert h1 == h2

    def test_normalizes_crlf(self):
        """\\r\\n 统一为 \\n 不影响哈希。"""
        h1 = compute_content_hash("line1\r\nline2")
        h2 = compute_content_hash("line1\nline2")
        assert h1 == h2

    def test_empty_content(self):
        """空内容返回空字符串。"""
        assert compute_content_hash("") == ""
        assert compute_content_hash(None) == ""

    def test_different_content_different_hash(self):
        """不同内容产生不同哈希。"""
        h1 = compute_content_hash("内容A")
        h2 = compute_content_hash("内容B")
        assert h1 != h2

    def test_hash_is_sha256_hex(self):
        """哈希值是 64 位十六进制字符串。"""
        h = compute_content_hash("test content")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── build_chunk_catalog ───────────────────────────────────────


class TestBuildCatalog:
    """测试 chunk catalog 构建。"""

    def test_catalog_fields_complete(self):
        """catalog 包含所有必需字段。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        assert len(catalog) == 3
        for entry in catalog:
            for col in _EXPORT_COLUMNS:
                assert col in entry, f"缺少字段: {col}"

    def test_catalog_preserves_segment_data(self):
        """catalog 保留原始 segment 数据。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        assert catalog[0]["segment_id"] == "seg_001"
        assert catalog[0]["content"] == "这是第一段内容。"
        assert catalog[0]["tokens"] == 50
        assert catalog[0]["word_count"] == 7
        assert catalog[0]["enabled"] is True
        assert catalog[0]["status"] == "completed"

    def test_catalog_adds_content_hash(self):
        """catalog 为每条记录添加 content_hash。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        for entry in catalog:
            assert entry["content_hash"] != ""
            assert len(entry["content_hash"]) == 64

    def test_catalog_includes_dataset_document_ids(self):
        """catalog 包含传入的 dataset_id 和 document_id。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds_xyz", "doc_abc")
        for entry in catalog:
            assert entry["dataset_id"] == "ds_xyz"
            assert entry["document_id"] == "doc_abc"

    def test_catalog_empty_segments(self):
        """空 segments 返回空 catalog。"""
        catalog = build_chunk_catalog([], "ds1", "doc1")
        assert catalog == []

    def test_catalog_handles_missing_fields(self):
        """segment 缺少可选字段时不崩溃。"""
        minimal_seg = [{"id": "s1", "content": "hello"}]
        catalog = build_chunk_catalog(minimal_seg, "ds1", "doc1")
        assert len(catalog) == 1
        assert catalog[0]["segment_id"] == "s1"
        assert catalog[0]["content_hash"] == compute_content_hash("hello")


# ── detect_duplicates ─────────────────────────────────────────


class TestDetectDuplicates:
    """测试重复分块检测。"""

    def test_detects_duplicates(self):
        """正确检测内容相同的分块。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        dupes = detect_duplicates(catalog)
        # seg_001 和 seg_003 内容相同
        assert len(dupes) == 1
        hash_key = list(dupes.keys())[0]
        assert len(dupes[hash_key]) == 2
        ids = {e["segment_id"] for e in dupes[hash_key]}
        assert ids == {"seg_001", "seg_003"}

    def test_no_duplicates(self):
        """无重复时返回空字典。"""
        segments = [
            {"id": "s1", "content": "内容A"},
            {"id": "s2", "content": "内容B"},
        ]
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        dupes = detect_duplicates(catalog)
        assert dupes == {}

    def test_empty_catalog(self):
        """空 catalog 返回空字典。"""
        assert detect_duplicates([]) == {}

    def test_multiple_duplicate_groups(self):
        """多组重复分别检测。"""
        segments = [
            {"id": "s1", "content": "A"},
            {"id": "s2", "content": "B"},
            {"id": "s3", "content": "A"},
            {"id": "s4", "content": "B"},
        ]
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        dupes = detect_duplicates(catalog)
        assert len(dupes) == 2


# ── 导出 ─────────────────────────────────────────────────────


class TestExport:
    """测试导出功能。"""

    def test_export_json_valid(self):
        """导出 JSON 是合法 JSON 且包含完整数据。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        json_str = export_catalog_json(catalog)
        parsed = json.loads(json_str)
        assert len(parsed) == 3
        assert parsed[0]["segment_id"] == "seg_001"

    def test_export_json_has_indent(self):
        """导出 JSON 有缩进（便于人工阅读）。"""
        catalog = build_chunk_catalog([{"id": "s1", "content": "x"}], "ds1", "doc1")
        json_str = export_catalog_json(catalog)
        assert "\n" in json_str  # 缩进会引入换行

    def test_export_csv_has_bom(self):
        """导出 CSV 以 UTF-8 BOM 开头（Excel 友好）。"""
        catalog = build_chunk_catalog([{"id": "s1", "content": "x"}], "ds1", "doc1")
        csv_bytes = export_catalog_csv(catalog)
        assert csv_bytes.startswith(b"\xef\xbb\xbf")

    def test_export_csv_has_header(self):
        """导出 CSV 包含表头行。"""
        catalog = build_chunk_catalog([{"id": "s1", "content": "x"}], "ds1", "doc1")
        csv_bytes = export_catalog_csv(catalog)
        first_line = csv_bytes.decode("utf-8-sig").split("\n")[0]
        for col in _EXPORT_COLUMNS:
            assert col in first_line

    def test_export_csv_row_count(self):
        """导出 CSV 行数 = 数据行数 + 表头行。"""
        segments = _sample_segments()
        catalog = build_chunk_catalog(segments, "ds1", "doc1")
        csv_bytes = export_catalog_csv(catalog)
        lines = csv_bytes.decode("utf-8-sig").strip().split("\n")
        assert len(lines) == 4  # 1 header + 3 data


# ── API Key 安全 ─────────────────────────────────────────────


class TestApiKeySafety:
    """测试 API Key 不泄露到导出内容。"""

    def test_api_key_not_in_json_export(self):
        """JSON 导出中不包含 API Key。"""
        secret_key = "app-SUPERSECRETKEY12345678"
        catalog = build_chunk_catalog(
            [{"id": "s1", "content": "test content"}], "ds1", "doc1"
        )
        json_str = export_catalog_json(catalog)
        assert secret_key not in json_str

    def test_api_key_not_in_csv_export(self):
        """CSV 导出中不包含 API Key。"""
        secret_key = "app-SUPERSECRETKEY12345678"
        catalog = build_chunk_catalog(
            [{"id": "s1", "content": "test content"}], "ds1", "doc1"
        )
        csv_bytes = export_catalog_csv(catalog).decode("utf-8-sig")
        assert secret_key not in csv_bytes

    def test_api_key_not_in_catalog_entries(self):
        """catalog 条目中不包含 API Key 字段。"""
        catalog = build_chunk_catalog(
            [{"id": "s1", "content": "test"}], "ds1", "doc1"
        )
        for entry in catalog:
            assert "api_key" not in entry
            assert "Authorization" not in entry


# ── 边界情况 ─────────────────────────────────────────────────


class TestEdgeCases:
    """边界情况测试。"""

    def test_segment_with_none_content(self):
        """content 为 None 时不崩溃。"""
        seg = [{"id": "s1", "content": None}]
        catalog = build_chunk_catalog(seg, "ds1", "doc1")
        assert catalog[0]["content"] == ""
        assert catalog[0]["content_hash"] == ""

    def test_segment_with_empty_content(self):
        """content 为空字符串时 hash 为空。"""
        seg = [{"id": "s1", "content": ""}]
        catalog = build_chunk_catalog(seg, "ds1", "doc1")
        assert catalog[0]["content_hash"] == ""

    def test_export_empty_catalog(self):
        """空 catalog 导出不崩溃。"""
        json_str = export_catalog_json([])
        assert json.loads(json_str) == []

        csv_bytes = export_catalog_csv([])
        lines = csv_bytes.decode("utf-8-sig").strip().split("\n")
        assert len(lines) == 1  # 仅表头

    @patch("dify_knowledge.requests.get")
    def test_list_datasets_error_propagates(self, mock_get):
        """API 错误正确传播。"""
        mock_get.return_value = _mock_response({"error": "unauthorized"}, status_code=403)
        with pytest.raises(RuntimeError, match="HTTP 403"):
            list_datasets("bad-key", "http://localhost/v1")


# ── 401 错误分类提示 ─────────────────────────────────────────


class TestAuthErrorMessages:
    """测试 401 错误的分类提示信息。"""

    @patch("dify_knowledge.requests.get")
    def test_401_with_app_key(self, mock_get):
        """使用 app- Key 访问 dataset API 时提示 Key 类型错误。"""
        mock_get.return_value = _mock_response(
            {"error": "unauthorized"}, status_code=401,
        )
        with pytest.raises(RuntimeError, match="应用 Key.*app-"):
            _get("app-abc123def456", "http://localhost/v1", "/datasets")

    @patch("dify_knowledge.requests.get")
    def test_401_with_dataset_key(self, mock_get):
        """使用 dataset- Key 但无效时提示 Key 无效。"""
        mock_get.return_value = _mock_response(
            {"error": "unauthorized"}, status_code=401,
        )
        with pytest.raises(RuntimeError, match="无效或已过期"):
            _get("dataset-abc123def456", "http://localhost/v1", "/datasets")

    @patch("dify_knowledge.requests.get")
    def test_401_with_empty_key(self, mock_get):
        """空 Key 时提示缺少 Key。"""
        mock_get.return_value = _mock_response(
            {"error": "unauthorized"}, status_code=401,
        )
        with pytest.raises(RuntimeError, match="缺少知识库 API Key"):
            _get("", "http://localhost/v1", "/datasets")

    @patch("dify_knowledge.requests.get")
    def test_401_with_unknown_prefix(self, mock_get):
        """其他前缀的 Key 提示无效。"""
        mock_get.return_value = _mock_response(
            {"error": "unauthorized"}, status_code=401,
        )
        with pytest.raises(RuntimeError, match="无效或已过期"):
            _get("sk-abc123def456", "http://localhost/v1", "/datasets")


# ── test_connection ───────────────────────────────────────────


class TestConnection:
    """测试连接测试函数。"""

    def test_missing_key(self):
        """空 Key 返回明确错误。"""
        ok, msg = check_connection("", "http://localhost/v1")
        assert ok is False
        assert "缺少知识库 API Key" in msg

    def test_app_key_rejected(self):
        """app- Key 被拒绝并提示类型错误。"""
        ok, msg = check_connection("app-abc123def456", "http://localhost/v1")
        assert ok is False
        assert "应用 Key" in msg
        assert "dataset-" in msg

    @patch("dify_knowledge.requests.get")
    def test_valid_dataset_key_success(self, mock_get):
        """有效的 dataset- Key 连接成功。"""
        mock_get.return_value = _mock_response({"data": [{"id": "ds1"}]})
        ok, msg = check_connection("dataset-abc123def456", "http://localhost/v1")
        assert ok is True
        assert "成功连接" in msg
        assert "1" in msg

    @patch("dify_knowledge.requests.get")
    def test_valid_dataset_key_auth_failure(self, mock_get):
        """dataset- Key 但 401 时返回认证失败消息。"""
        mock_get.return_value = _mock_response(
            {"error": "unauthorized"}, status_code=401,
        )
        ok, msg = check_connection("dataset-abc123def456", "http://localhost/v1")
        assert ok is False
        assert "无效或已过期" in msg

    @patch("dify_knowledge.requests.get")
    def test_connection_timeout(self, mock_get):
        """连接超时返回超时消息。"""
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()
        ok, msg = check_connection("dataset-abc123def456", "http://localhost/v1")
        assert ok is False
        assert "超时" in msg

    @patch("dify_knowledge.requests.get")
    def test_connection_refused(self, mock_get):
        """连接被拒绝返回连接失败消息。"""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")
        ok, msg = check_connection("dataset-abc123def456", "http://localhost/v1")
        assert ok is False


# ── retrieve 检索诊断 ────────────────────────────────────────


def _mock_post_response(json_data, status_code=200):
    """构造 mock requests.Response（POST）。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    return resp


class TestRetrieve:
    """测试检索诊断函数。"""

    @patch("dify_knowledge.requests.post")
    def test_retrieve_constructs_correct_request(self, mock_post):
        """POST 请求构造正确。"""
        mock_post.return_value = _mock_post_response({
            "records": [
                {"segment": {"id": "s1", "document_id": "d1", "content": "hello",
                             "enabled": True, "status": "completed"}, "score": 0.95},
            ]
        })
        retrieve("dataset-key", "http://localhost/v1", "ds1", "test query", top_k=3)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost/v1/datasets/ds1/retrieve"
        body = call_args[1]["json"]
        assert body["query"] == "test query"
        assert body["retrieval_model"]["top_k"] == 3

    @patch("dify_knowledge.requests.post")
    def test_retrieve_returns_parsed_records(self, mock_post):
        """返回解析后的记录列表。"""
        mock_post.return_value = _mock_post_response({
            "records": [
                {"segment": {"id": "s1", "document_id": "d1", "content": "chunk 1",
                             "enabled": True, "status": "completed"}, "score": 0.95},
                {"segment": {"id": "s2", "document_id": "d2", "content": "chunk 2",
                             "enabled": True, "status": "completed"}, "score": 0.80},
            ]
        })
        records = retrieve("key", "http://localhost/v1", "ds1", "query")
        assert len(records) == 2
        assert records[0]["position"] == 1
        assert records[0]["segment_id"] == "s1"
        assert records[0]["score"] == 0.95
        assert records[1]["position"] == 2
        assert records[1]["score"] == 0.80

    @patch("dify_knowledge.requests.post")
    def test_retrieve_topk_clamped(self, mock_post):
        """top_k 被钳制在 1-20 范围。"""
        mock_post.return_value = _mock_post_response({"records": []})
        retrieve("key", "http://localhost/v1", "ds1", "q", top_k=100)
        call_args = mock_post.call_args
        assert call_args[1]["json"]["retrieval_model"]["top_k"] == 20

        retrieve("key", "http://localhost/v1", "ds1", "q", top_k=0)
        call_args = mock_post.call_args
        assert call_args[1]["json"]["retrieval_model"]["top_k"] == 1

    @patch("dify_knowledge.requests.post")
    def test_retrieve_empty_results(self, mock_get):
        """空结果返回空列表。"""
        mock_get.return_value = _mock_post_response({"records": []})
        records = retrieve("key", "http://localhost/v1", "ds1", "no results")
        assert records == []

    @patch("dify_knowledge.requests.post")
    def test_retrieve_401_with_app_key(self, mock_post):
        """app- Key 时 401 提示 Key 类型错误。"""
        mock_post.return_value = _mock_post_response(
            {"error": "unauthorized"}, status_code=401,
        )
        with pytest.raises(RuntimeError, match="应用 Key.*app-"):
            retrieve("app-abc", "http://localhost/v1", "ds1", "q")

    @patch("dify_knowledge.requests.post")
    def test_retrieve_401_with_dataset_key(self, mock_post):
        """dataset- Key 无效时 401 提示无效。"""
        mock_post.return_value = _mock_post_response(
            {"error": "unauthorized"}, status_code=401,
        )
        with pytest.raises(RuntimeError, match="无效或已过期"):
            retrieve("dataset-abc123def456", "http://localhost/v1", "ds1", "q")


# ── build_full_kb_catalog ──────────────────────────────────────


def _sample_docs_page1():
    """返回第 1 页文档列表。"""
    return {
        "data": [
            {"id": "doc_a", "name": "文档A.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 1000},
            {"id": "doc_b", "name": "文档B.docx", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 2000},
        ],
        "has_more": False,
        "total": 2,
    }


def _sample_docs_with_skipped():
    """返回包含需跳过文档的列表。"""
    return {
        "data": [
            {"id": "doc_ok", "name": "正常文档.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 500},
            {"id": "doc_archived", "name": "已归档文档.pdf", "indexing_status": "completed", "enabled": True, "archived": True, "word_count": 300},
            {"id": "doc_error", "name": "失败文档.pdf", "indexing_status": "error", "enabled": True, "archived": False, "word_count": 100},
        ],
        "has_more": False,
        "total": 3,
    }


def _sample_segments_for_doc(doc_id, count=3):
    """为指定文档生成 segment 数据。"""
    return [
        {
            "id": f"seg_{doc_id}_{i:03d}",
            "position": i + 1,
            "document_id": doc_id,
            "content": f"文档 {doc_id} 的第 {i+1} 段内容。",
            "index_node_id": f"node_{doc_id}_{i}",
            "index_node_hash": f"hash_{doc_id}_{i}",
            "tokens": 50 + i,
            "word_count": 10 + i,
            "enabled": True,
            "status": "completed",
        }
        for i in range(count)
    ]


class TestBuildFullKbCatalog:
    """测试全知识库 catalog 构建。"""

    @patch("dify_knowledge.requests.get")
    def test_two_docs_multi_page_chunks(self, mock_get):
        """两个文档、每文档多页 chunks 时，导出总数正确。"""
        # 第 1 次调用：list_all_documents (page 1)
        # 第 2 次调用：list_all_segments doc_a page 1
        # 第 3 次调用：list_all_segments doc_a page 2
        # 第 4 次调用：list_all_segments doc_b page 1
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            params = kwargs.get("params", {})
            if "/documents" in url and "/segments" not in url:
                # list_documents
                return _mock_response({
                    "data": [
                        {"id": "doc_a", "name": "文档A.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 1000},
                        {"id": "doc_b", "name": "文档B.docx", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 2000},
                    ],
                    "has_more": False,
                    "total": 2,
                })
            elif "/segments" in url and "doc_a" in url:
                all_segs_a = _sample_segments_for_doc("doc_a", 3)
                if params.get("page") == 1:
                    return _mock_response({
                        "data": all_segs_a[:2],
                        "has_more": True,
                        "total": 3,
                    })
                else:
                    return _mock_response({
                        "data": all_segs_a[2:],
                        "has_more": False,
                        "total": 3,
                    })
            elif "/segments" in url and "doc_b" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc_b", 2),
                    "has_more": False,
                    "total": 2,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("dataset-key", "http://localhost/v1", "ds1", "测试知识库")

        assert result["metadata"]["total_documents"] == 2
        assert result["metadata"]["exported_documents"] == 2
        assert result["metadata"]["skipped_documents"] == 0
        assert result["metadata"]["failed_documents"] == 0
        assert result["metadata"]["total_chunks"] == 3 + 2  # doc_a: 3, doc_b: 2
        assert len(result["catalog"]) == 5
        assert result["metadata"]["dataset_name"] == "测试知识库"
        assert result["metadata"]["export_type"] == "full_knowledge_base"

    @patch("dify_knowledge.requests.get")
    def test_one_doc_failure_others_exported(self, mock_get):
        """一个文档 API 失败时，其他文档仍被导出且失败被记录。"""
        call_idx = [0]

        def side_effect(url, **kwargs):
            call_idx[0] += 1
            params = kwargs.get("params", {})
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc_ok", "name": "正常文档.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 500},
                        {"id": "doc_fail", "name": "失败文档.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 800},
                    ],
                    "has_more": False,
                    "total": 2,
                })
            elif "/segments" in url and "doc_ok" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc_ok", 2),
                    "has_more": False,
                    "total": 2,
                })
            elif "/segments" in url and "doc_fail" in url:
                return _mock_response({"error": "internal error"}, status_code=500)
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("dataset-key", "http://localhost/v1", "ds1")

        assert result["metadata"]["exported_documents"] == 1
        assert result["metadata"]["failed_documents"] == 1
        assert result["metadata"]["total_chunks"] == 2
        assert len(result["catalog"]) == 2

        failed_stats = [s for s in result["doc_stats"] if s["status"] == "error"]
        assert len(failed_stats) == 1
        assert failed_stats[0]["document_name"] == "失败文档.pdf"
        assert "HTTP 500" in failed_stats[0]["reason"]

    @patch("dify_knowledge.requests.get")
    def test_skipped_archived_and_error_docs(self, mock_get):
        """已归档和处理失败的文档被跳过，completed 文档正常导出。"""

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response(_sample_docs_with_skipped())
            elif "/segments" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc_ok", 1),
                    "has_more": False,
                    "total": 1,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("dataset-key", "http://localhost/v1", "ds1")

        assert result["metadata"]["total_documents"] == 3
        assert result["metadata"]["exported_documents"] == 1
        assert result["metadata"]["skipped_documents"] == 2
        assert result["metadata"]["total_chunks"] == 1

        skipped = [s for s in result["doc_stats"] if s["status"] == "skipped"]
        assert len(skipped) == 2
        skip_names = {s["document_name"] for s in skipped}
        assert "已归档文档.pdf" in skip_names
        assert "失败文档.pdf" in skip_names
        # 每个 skipped 文档有明确原因
        for s in skipped:
            assert s["reason"] != ""
            assert "api_fields" in s

    @patch("dify_knowledge.requests.get")
    def test_each_chunk_has_document_name(self, mock_get):
        """每个 chunk 都包含 document_name 和完整 content。"""
        mock_get.return_value = _mock_response({
            "data": [
                {"id": "doc_x", "name": "合同模板.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 500},
            ],
            "has_more": False,
            "total": 1,
        })

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc_x", "name": "合同模板.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 500},
                    ],
                    "has_more": False,
                    "total": 1,
                })
            elif "/segments" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc_x", 2),
                    "has_more": False,
                    "total": 2,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("dataset-key", "http://localhost/v1", "ds1", "测试库")

        for entry in result["catalog"]:
            assert entry["document_name"] == "合同模板.pdf"
            assert entry["dataset_id"] == "ds1"
            assert entry["content"] != ""
            assert len(entry["content"]) > 0

    @patch("dify_knowledge.requests.get")
    def test_no_api_key_in_catalog_or_metadata(self, mock_get):
        """JSON/CSV/日志不含 dataset API key 或 secret。"""
        secret_key = "dataset-SUPERSECRET1234567890"

        mock_get.return_value = _mock_response({
            "data": [
                {"id": "doc1", "name": "test.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 100},
            ],
            "has_more": False,
            "total": 1,
        })

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc1", "name": "test.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 100},
                    ],
                    "has_more": False,
                    "total": 1,
                })
            elif "/segments" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc1", 1),
                    "has_more": False,
                    "total": 1,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog(secret_key, "http://localhost/v1", "ds1")

        # 检查 catalog 不含 key
        json_str = export_full_kb_json(result["catalog"], result["metadata"])
        assert secret_key not in json_str

        csv_bytes = export_full_kb_csv(result["catalog"]).decode("utf-8-sig")
        assert secret_key not in csv_bytes

        # 检查每条记录不含 key
        for entry in result["catalog"]:
            assert "api_key" not in entry
            assert secret_key not in str(entry)

    @patch("dify_knowledge.requests.get")
    def test_progress_callback_called(self, mock_get):
        """progress_callback 被正确调用。"""
        mock_get.return_value = _mock_response({
            "data": [
                {"id": "doc1", "name": "doc1.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 100},
                {"id": "doc2", "name": "doc2.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 200},
            ],
            "has_more": False,
            "total": 2,
        })

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc1", "name": "doc1.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 100},
                        {"id": "doc2", "name": "doc2.pdf", "indexing_status": "completed", "enabled": True, "archived": False, "word_count": 200},
                    ],
                    "has_more": False,
                    "total": 2,
                })
            elif "/segments" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc1", 1),
                    "has_more": False,
                    "total": 1,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        calls = []

        def progress_cb(cur, total, doc_name, chunk_count):
            calls.append((cur, total, doc_name, chunk_count))

        build_full_kb_catalog("key", "http://localhost/v1", "ds1", progress_callback=progress_cb)

        assert len(calls) == 2
        assert calls[0][0] == 1  # current
        assert calls[0][1] == 2  # total
        assert calls[0][2] == "doc1.pdf"  # doc_name
        assert calls[0][3] == 1  # chunk_count


class TestIndexingStatusField:
    """测试 indexing_status 字段正确处理。"""

    @patch("dify_knowledge.requests.get")
    def test_six_completed_docs_all_exported(self, mock_get):
        """6 个 indexing_status=completed 的文档必须全部进入 chunks 拉取流程。"""
        docs = [
            {"id": f"doc_{i}", "name": f"文档{i}.pdf", "indexing_status": "completed",
             "enabled": True, "archived": False, "word_count": 100 * i}
            for i in range(1, 7)
        ]

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({"data": docs, "has_more": False, "total": 6})
            elif "/segments" in url:
                # 提取 document_id
                doc_id = url.split("/documents/")[1].split("/segments")[0]
                return _mock_response({
                    "data": _sample_segments_for_doc(doc_id, 1),
                    "has_more": False,
                    "total": 1,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("key", "http://localhost/v1", "ds1")

        assert result["metadata"]["total_documents"] == 6
        assert result["metadata"]["exported_documents"] == 6
        assert result["metadata"]["skipped_documents"] == 0
        assert result["metadata"]["failed_documents"] == 0
        assert result["metadata"]["total_chunks"] == 6

    @patch("dify_knowledge.requests.get")
    def test_missing_indexing_status_not_skip_all(self, mock_get):
        """indexing_status 字段缺失时不能全部跳过。"""

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                # 文档没有 indexing_status 字段
                return _mock_response({
                    "data": [
                        {"id": "doc1", "name": "无状态文档.pdf", "enabled": True, "word_count": 100},
                    ],
                    "has_more": False,
                    "total": 1,
                })
            elif "/segments" in url:
                return _mock_response({
                    "data": _sample_segments_for_doc("doc1", 2),
                    "has_more": False,
                    "total": 2,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("key", "http://localhost/v1", "ds1")

        assert result["metadata"]["exported_documents"] == 1
        assert result["metadata"]["skipped_documents"] == 0
        assert result["metadata"]["total_chunks"] == 2

    @patch("dify_knowledge.requests.get")
    def test_all_skipped_has_skipped_count(self, mock_get):
        """全部文档被跳过时，skipped_count 正确。"""

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc1", "name": "归档1.pdf", "indexing_status": "completed",
                         "enabled": True, "archived": True, "word_count": 100},
                        {"id": "doc2", "name": "归档2.pdf", "indexing_status": "completed",
                         "enabled": True, "archived": True, "word_count": 200},
                    ],
                    "has_more": False,
                    "total": 2,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("key", "http://localhost/v1", "ds1")

        assert result["metadata"]["exported_documents"] == 0
        assert result["metadata"]["skipped_documents"] == 2
        assert result["metadata"]["total_chunks"] == 0

        # 每个 skipped 文档有明确原因和 api_fields
        for stat in result["doc_stats"]:
            assert stat["status"] == "skipped"
            assert stat["reason"] != ""
            assert "api_fields" in stat
            assert stat["api_fields"]["archived"] is True

    @patch("dify_knowledge.requests.get")
    def test_indexing_status_indexing_skipped(self, mock_get):
        """indexing_status=indexing 的文档被跳过并显示原因。"""

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc1", "name": "处理中.pdf", "indexing_status": "indexing",
                         "enabled": True, "archived": False, "word_count": 100},
                    ],
                    "has_more": False,
                    "total": 1,
                })
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("key", "http://localhost/v1", "ds1")

        assert result["metadata"]["exported_documents"] == 0
        assert result["metadata"]["skipped_documents"] == 1
        stat = result["doc_stats"][0]
        assert stat["status"] == "skipped"
        assert "indexing" in stat["reason"]
        assert stat["api_fields"]["indexing_status"] == "indexing"

    @patch("dify_knowledge.requests.get")
    def test_empty_chunks_is_ok_not_skipped(self, mock_get):
        """chunks API 返回空列表时，文档算成功导出（0 chunks），不算 skipped。"""

        def side_effect(url, **kwargs):
            if "/documents" in url and "/segments" not in url:
                return _mock_response({
                    "data": [
                        {"id": "doc1", "name": "空文档.pdf", "indexing_status": "completed",
                         "enabled": True, "archived": False, "word_count": 0},
                    ],
                    "has_more": False,
                    "total": 1,
                })
            elif "/segments" in url:
                return _mock_response({"data": [], "has_more": False, "total": 0})
            return _mock_response({"data": []})

        mock_get.side_effect = side_effect

        result = build_full_kb_catalog("key", "http://localhost/v1", "ds1")

        assert result["metadata"]["exported_documents"] == 1
        assert result["metadata"]["skipped_documents"] == 0
        assert result["metadata"]["total_chunks"] == 0
        stat = result["doc_stats"][0]
        assert stat["status"] == "ok"
        assert stat["chunk_count"] == 0
        assert "空列表" in stat["reason"]


class TestExportFullKb:
    """测试全知识库导出格式。"""

    def test_export_full_kb_json_has_metadata(self):
        """JSON 导出包含 metadata 顶层包装。"""
        catalog = build_chunk_catalog(
            _sample_segments_for_doc("doc1", 2), "ds1", "doc1", "文档1.pdf"
        )
        metadata = {
            "export_type": "full_knowledge_base",
            "dataset_id": "ds1",
            "dataset_name": "测试库",
            "total_documents": 1,
            "exported_documents": 1,
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 2,
            "schema_version": "1.0",
        }
        json_str = export_full_kb_json(catalog, metadata)
        parsed = json.loads(json_str)

        assert "metadata" in parsed
        assert "catalog" in parsed
        assert parsed["metadata"]["export_type"] == "full_knowledge_base"
        assert len(parsed["catalog"]) == 2

    def test_export_full_kb_csv_has_document_name_column(self):
        """CSV 导出包含 document_name 列。"""
        catalog = build_chunk_catalog(
            _sample_segments_for_doc("doc1", 2), "ds1", "doc1", "文档1.pdf"
        )
        csv_bytes = export_full_kb_csv(catalog)
        first_line = csv_bytes.decode("utf-8-sig").split("\n")[0]
        assert "document_name" in first_line

    def test_export_full_kb_csv_row_count(self):
        """CSV 行数正确。"""
        catalog = build_chunk_catalog(
            _sample_segments_for_doc("doc1", 3), "ds1", "doc1", "文档1.pdf"
        )
        csv_bytes = export_full_kb_csv(catalog)
        lines = csv_bytes.decode("utf-8-sig").strip().split("\n")
        assert len(lines) == 4  # 1 header + 3 data


class TestExportColumnsIncludeDocumentName:
    """测试 _EXPORT_COLUMNS 包含 document_name。"""

    def test_export_columns_has_document_name(self):
        """导出列包含 document_name。"""
        assert "document_name" in _EXPORT_COLUMNS

    def test_catalog_entries_have_document_name(self):
        """build_chunk_catalog 生成的条目包含 document_name。"""
        catalog = build_chunk_catalog(
            [{"id": "s1", "content": "test"}], "ds1", "doc1", "测试文档.pdf"
        )
        assert catalog[0]["document_name"] == "测试文档.pdf"
