"""知识库导出 summary 字段测试。

覆盖：
- 有 summary 的分段会被保留
- 无 summary 的旧版响应仍兼容
- summary 不会拼接进 content
- 自定义 metadata 可用时被导出
- API Key、Cookie、Authorization 绝不能出现在 JSON/CSV 输出中
"""

import csv
import io
import json

import pytest

from dify_knowledge import (
    _EXPORT_COLUMNS,
    build_chunk_catalog,
    detect_duplicates,
    export_catalog_csv,
    export_catalog_json,
    export_full_kb_csv,
    export_full_kb_json,
)


# ── 测试数据 ──────────────────────────────────────────────────


def _make_segment(**overrides) -> dict:
    """构造一个最小合法 segment dict，可通过 overrides 定制字段。"""
    base = {
        "id": "seg-001",
        "position": 1,
        "document_id": "doc-abc",
        "content": "这是正文内容。",
        "index_node_id": "node-001",
        "index_node_hash": "hash-001",
        "tokens": 120,
        "word_count": 50,
        "enabled": True,
        "status": "completed",
    }
    base.update(overrides)
    return base


# ── 有 summary 的分段会被保留 ─────────────────────────────────


class TestSummaryPreserved:
    """当 API 响应包含 summary 时，catalog 中应原样保留。"""

    def test_summary_in_catalog_entry(self):
        seg = _make_segment(summary="这是一段摘要。")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        assert len(catalog) == 1
        assert catalog[0]["summary"] == "这是一段摘要。"

    def test_summary_in_json_export(self):
        seg = _make_segment(summary="JSON 摘要测试")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        exported = json.loads(export_catalog_json(catalog))
        assert exported[0]["summary"] == "JSON 摘要测试"

    def test_summary_in_csv_export(self):
        seg = _make_segment(summary="CSV 摘要测试")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        csv_bytes = export_catalog_csv(catalog)
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert rows[0]["summary"] == "CSV 摘要测试"

    def test_summary_in_full_kb_json_export(self):
        seg = _make_segment(summary="全库 JSON 摘要")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        meta = {"export_type": "full_knowledge_base", "dataset_id": "ds-1"}
        wrapper = json.loads(export_full_kb_json(catalog, meta))
        assert wrapper["catalog"][0]["summary"] == "全库 JSON 摘要"

    def test_summary_in_full_kb_csv_export(self):
        seg = _make_segment(summary="全库 CSV 摘要")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        csv_bytes = export_full_kb_csv(catalog)
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert rows[0]["summary"] == "全库 CSV 摘要"

    def test_multiple_segments_with_different_summaries(self):
        segs = [
            _make_segment(id="seg-1", summary="摘要一"),
            _make_segment(id="seg-2", summary="摘要二"),
            _make_segment(id="seg-3", summary="摘要三"),
        ]
        catalog = build_chunk_catalog(segs, "ds-1", "doc-abc", "文档A")
        assert [e["summary"] for e in catalog] == ["摘要一", "摘要二", "摘要三"]


# ── 无 summary 的旧版响应仍兼容 ──────────────────────────────


class TestBackwardCompatibility:
    """旧版 API 不返回 summary 字段时，catalog 中 summary 为空字符串。"""

    def test_missing_summary_defaults_to_empty(self):
        """segment dict 中无 summary 键时，catalog entry 的 summary 为空字符串。"""
        seg = _make_segment()  # 无 summary 字段
        assert "summary" not in seg
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        assert catalog[0]["summary"] == ""

    def test_none_summary_defaults_to_empty(self):
        """API 返回 summary=null 时，catalog entry 的 summary 为空字符串。"""
        seg = _make_segment(summary=None)
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        assert catalog[0]["summary"] == ""

    def test_empty_string_summary_preserved(self):
        """API 返回 summary="" 时，catalog entry 的 summary 为空字符串。"""
        seg = _make_segment(summary="")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        assert catalog[0]["summary"] == ""

    def test_whitespace_only_summary_stripped_to_empty(self):
        """summary 仅含空白字符时，strip 后为空字符串。"""
        seg = _make_segment(summary="   \n\t  ")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        assert catalog[0]["summary"] == ""

    def test_old_response_json_export_has_empty_summary(self):
        """旧版响应导出 JSON 时，summary 字段存在但为空。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        exported = json.loads(export_catalog_json(catalog))
        assert "summary" in exported[0]
        assert exported[0]["summary"] == ""

    def test_old_response_csv_export_has_summary_column(self):
        """旧版响应导出 CSV 时，summary 列存在但值为空。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        csv_bytes = export_catalog_csv(catalog)
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert "summary" in rows[0]
        assert rows[0]["summary"] == ""

    def test_mixed_segments_some_with_some_without_summary(self):
        """混合：部分有 summary，部分没有，全部正常导出。"""
        segs = [
            _make_segment(id="seg-1", summary="有摘要"),
            _make_segment(id="seg-2"),  # 无 summary
            _make_segment(id="seg-3", summary=None),  # null summary
            _make_segment(id="seg-4", summary="另一个摘要"),
        ]
        catalog = build_chunk_catalog(segs, "ds-1", "doc-abc", "文档A")
        summaries = [e["summary"] for e in catalog]
        assert summaries == ["有摘要", "", "", "另一个摘要"]


# ── summary 不会拼接进 content ──────────────────────────────


class TestSummaryNotInContent:
    """summary 字段绝不能被拼接到 content 中。"""

    def test_content_unchanged_when_summary_present(self):
        original_content = "原始正文内容，不应被修改。"
        seg = _make_segment(content=original_content, summary="这是摘要")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        assert catalog[0]["content"] == original_content
        assert "摘要" not in catalog[0]["content"]

    def test_content_hash_ignores_summary(self):
        """content_hash 只基于 content 计算，不受 summary 影响。"""
        seg_with = _make_segment(content="相同内容", summary="有摘要")
        seg_without = _make_segment(content="相同内容")
        cat_with = build_chunk_catalog([seg_with], "ds-1", "doc-abc", "文档A")
        cat_without = build_chunk_catalog([seg_without], "ds-1", "doc-abc", "文档A")
        assert cat_with[0]["content_hash"] == cat_without[0]["content_hash"]

    def test_content_and_summary_are_separate_fields(self):
        seg = _make_segment(content="正文", summary="摘要")
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        entry = catalog[0]
        assert entry["content"] == "正文"
        assert entry["summary"] == "摘要"
        assert entry["content"] != entry["summary"]


# ── 自定义 metadata 可用时被导出 ──────────────────────────────


class TestMetadataExported:
    """document-level metadata 应正确写入导出。"""

    def test_dataset_id_in_catalog(self):
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-123", "doc-abc", "文档A")
        assert catalog[0]["dataset_id"] == "ds-123"

    def test_document_name_in_catalog(self):
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "我的文档")
        assert catalog[0]["document_name"] == "我的文档"

    def test_full_kb_metadata_fields(self):
        """全知识库导出的 metadata 包含 dataset 信息。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        from datetime import datetime, timezone
        meta = {
            "export_type": "full_knowledge_base",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": "ds-1",
            "dataset_name": "测试知识库",
            "total_documents": 1,
            "exported_documents": 1,
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 1,
            "schema_version": "1.0",
        }
        wrapper = json.loads(export_full_kb_json(catalog, meta))
        assert wrapper["metadata"]["dataset_id"] == "ds-1"
        assert wrapper["metadata"]["dataset_name"] == "测试知识库"
        assert wrapper["metadata"]["total_chunks"] == 1

    def test_segment_level_api_fields_preserved(self):
        """segment 级别的 API 字段（tokens, word_count 等）正确保留。"""
        seg = _make_segment(
            tokens=256,
            word_count=100,
            enabled=False,
            status="disabled",
            index_node_id="custom-node-id",
            index_node_hash="custom-hash",
        )
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        entry = catalog[0]
        assert entry["tokens"] == 256
        assert entry["word_count"] == 100
        assert entry["enabled"] is False
        assert entry["status"] == "disabled"
        assert entry["index_node_id"] == "custom-node-id"
        assert entry["index_node_hash"] == "custom-hash"


# ── API Key、Cookie、Authorization 绝不能出现在输出中 ────────


class TestNoCredentialsInOutput:
    """导出文件中绝不能包含 API Key、Cookie、Authorization 等敏感信息。"""

    _SENSITIVE_PATTERNS = [
        "dataset-",
        "app-",
        "Bearer ",
        "Authorization",
        "Cookie",
        "api_key",
        "api-key",
        "apikey",
        "secret",
        "token",
        "password",
    ]

    def _assert_no_sensitive(self, text: str, label: str = ""):
        """断言文本中不包含敏感模式（允许出现在正常字段名中，但不允许出现在值中）。"""
        lower = text.lower()
        # 这些模式如果出现在值区域（即引号后的值）才算泄露
        # 简单检查：不包含典型的 API Key 格式
        for pat in ["dataset-", "app-", "Bearer sk-", "Cookie:", "Authorization:"]:
            assert pat not in text, (
                f"{label}: 导出内容中发现敏感信息 '{pat}'"
            )

    def test_json_export_no_credentials(self):
        """JSON 导出不包含凭据。"""
        seg = _make_segment(
            content="正常内容",
            summary="正常摘要",
        )
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        json_str = export_catalog_json(catalog)
        self._assert_no_sensitive(json_str, "JSON export")

    def test_csv_export_no_credentials(self):
        """CSV 导出不包含凭据。"""
        seg = _make_segment(
            content="正常内容",
            summary="正常摘要",
        )
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        csv_bytes = export_catalog_csv(catalog)
        text = csv_bytes.decode("utf-8-sig")
        self._assert_no_sensitive(text, "CSV export")

    def test_full_kb_json_no_credentials(self):
        """全知识库 JSON 导出不包含凭据。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        meta = {
            "export_type": "full_knowledge_base",
            "dataset_id": "ds-1",
            "dataset_name": "测试库",
        }
        json_str = export_full_kb_json(catalog, meta)
        self._assert_no_sensitive(json_str, "Full KB JSON export")

    def test_full_kb_csv_no_credentials(self):
        """全知识库 CSV 导出不包含凭据。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        csv_bytes = export_full_kb_csv(catalog)
        text = csv_bytes.decode("utf-8-sig")
        self._assert_no_sensitive(text, "Full KB CSV export")

    def test_content_with_sensitive_like_text_not_filtered(self):
        """如果 content 本身讨论 API Key（如文档内容），不应被过滤。
        但 build_chunk_catalog 不会注入凭据。"""
        seg = _make_segment(
            content="本文档说明如何使用 dataset-xxx 格式的 Key。",
            summary="关于 API Key 格式的说明",
        )
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        # content 原样保留（不做清洗），但 catalog 函数本身不注入凭据
        assert "dataset-xxx" in catalog[0]["content"]

    def test_catalog_entry_keys_no_credential_fields(self):
        """catalog entry 的字段名中不包含 credential 相关键。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        entry = catalog[0]
        for key in entry:
            lower_key = key.lower()
            for forbidden in ["api_key", "apikey", "api-key", "cookie",
                              "authorization", "bearer", "secret", "password"]:
                assert forbidden not in lower_key, (
                    f"catalog entry 包含敏感字段名: {key}"
                )

    def test_export_columns_no_credential_fields(self):
        """_EXPORT_COLUMNS 中不包含凭据相关列名。"""
        for col in _EXPORT_COLUMNS:
            lower_col = col.lower()
            for forbidden in ["api_key", "apikey", "api-key", "cookie",
                              "authorization", "bearer", "secret", "password"]:
                assert forbidden not in lower_col, (
                    f"_EXPORT_COLUMNS 包含敏感列名: {col}"
                )


# ── summary 在 _EXPORT_COLUMNS 中 ───────────────────────────


class TestExportColumnsSchema:
    """导出列定义应包含 summary。"""

    def test_summary_in_export_columns(self):
        assert "summary" in _EXPORT_COLUMNS

    def test_summary_after_content_in_columns(self):
        """summary 应在 content 之后。"""
        content_idx = _EXPORT_COLUMNS.index("content")
        summary_idx = _EXPORT_COLUMNS.index("summary")
        assert summary_idx == content_idx + 1

    def test_csv_header_contains_summary(self):
        """CSV 表头应包含 summary 列。"""
        seg = _make_segment()
        catalog = build_chunk_catalog([seg], "ds-1", "doc-abc", "文档A")
        csv_bytes = export_catalog_csv(catalog)
        text = csv_bytes.decode("utf-8-sig")
        first_line = text.split("\n")[0]
        assert "summary" in first_line


# ── 重复检测不受 summary 影响 ─────────────────────────────────


class TestDuplicateDetectionUnaffected:
    """重复检测基于 content_hash，summary 差异不影响结果。"""

    def test_same_content_different_summary_still_duplicate(self):
        segs = [
            _make_segment(id="seg-1", content="相同内容", summary="摘要A"),
            _make_segment(id="seg-2", content="相同内容", summary="摘要B"),
        ]
        catalog = build_chunk_catalog(segs, "ds-1", "doc-abc", "文档A")
        dupes = detect_duplicates(catalog)
        # 相同 content 应被检测为重复，即使 summary 不同
        assert len(dupes) == 1

    def test_different_content_same_summary_not_duplicate(self):
        segs = [
            _make_segment(id="seg-1", content="内容A", summary="相同摘要"),
            _make_segment(id="seg-2", content="内容B", summary="相同摘要"),
        ]
        catalog = build_chunk_catalog(segs, "ds-1", "doc-abc", "文档A")
        dupes = detect_duplicates(catalog)
        # 不同 content 不应被检测为重复
        assert len(dupes) == 0
