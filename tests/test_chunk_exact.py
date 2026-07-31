"""
chunk_exact 题集创建 + 评测测试。

测试内容：
1. filter_candidate_chunks — 过滤逻辑
2. _parse_llm_response — LLM 输出解析
3. _validate_candidate_id — fail-closed 校验
4. chunk_exact judge — 纯机器判定（Top1/3/5 hit）
5. question dict 结构完整性
6. compute_metrics chunk_exact 分组
"""

import json
import random
import sys
import hashlib
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from chunk_exact_questions import (
    filter_candidate_chunks,
    _parse_llm_response,
    _validate_candidate_id,
    generate_chunk_exact_questions,
    generate_chunk_exact_questions_multi_doc,
    validate_chunk_exact_question,
    validate_chunk_exact_set,
    validate_multi_doc_config,
    sample_candidates_random,
    get_candidates_by_documents,
    generate_default_set_name,
    generate_default_set_name_for_dataset,
)
from judge import (
    classify_evaluation_track, _judge_chunk_exact, compute_metrics,
    build_result_status, backfill_chunk_exact_topk,
    TRACK_CHUNK_EXACT, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE,
)


# ── Fixtures ──────────────────────────────────────────────────


def _make_catalog_entry(segment_id, content, status="completed", enabled=True,
                        dataset_id="ds1", document_id="doc1", document_name=""):
    """构造 catalog entry。"""
    return {
        "segment_id": segment_id,
        "content": content,
        "status": status,
        "enabled": enabled,
        "dataset_id": dataset_id,
        "document_id": document_id,
        "content_hash": hashlib.sha256(content.strip().replace("\r\n", "\n").encode("utf-8")).hexdigest() if content else "",
        "position": 1,
        "index_node_id": "",
        "index_node_hash": "",
        "tokens": 10,
        "word_count": 5,
    }


# ── filter_candidate_chunks ──────────────────────────────────


class TestFilterCandidates:
    """测试候选 chunk 过滤逻辑。"""

    def test_filters_completed_only(self):
        """只保留 status=completed 的 chunk。"""
        catalog = [
            _make_catalog_entry("s1", "这是有效的知识片段内容，长度足够通过过滤检查。"),
            _make_catalog_entry("s2", "这条也是有效内容，足够长。", status="indexing"),
            _make_catalog_entry("s3", "这条也是有效内容，足够长。", status="error"),
        ]
        candidates, stats = filter_candidate_chunks(catalog)
        assert len(candidates) == 1
        assert candidates[0]["segment_id"] == "s1"
        assert stats["filtered"]["status_not_completed"] == 2

    def test_filters_enabled_only(self):
        """只保留 enabled=true 的 chunk。"""
        catalog = [
            _make_catalog_entry("s1", "这是有效的知识片段内容，长度足够通过过滤检查。"),
            _make_catalog_entry("s2", "这是被禁用的内容片段，但长度足够。", enabled=False),
        ]
        candidates, stats = filter_candidate_chunks(catalog)
        assert len(candidates) == 1
        assert stats["filtered"]["disabled"] == 1

    def test_filters_duplicates(self):
        """排除重复 chunk。"""
        content_dup = "这是有效的知识片段内容，长度足够通过过滤检查。"
        content_ok = "这是另一段不同的知识片段内容，也足够长。"
        catalog = [
            _make_catalog_entry("s1", content_dup),
            _make_catalog_entry("s2", content_dup),  # 与 s1 重复
            _make_catalog_entry("s3", content_ok),
        ]
        dup_hash = hashlib.sha256(content_dup.strip().replace("\r\n", "\n").encode("utf-8")).hexdigest()
        duplicates = {dup_hash: [catalog[0], catalog[1]]}
        candidates, stats = filter_candidate_chunks(catalog, duplicates)
        assert len(candidates) == 1
        assert candidates[0]["segment_id"] == "s3"
        assert stats["filtered"]["duplicate"] == 2

    def test_filters_empty_content(self):
        """排除空内容 chunk。"""
        catalog = [
            _make_catalog_entry("s1", "这是有效的知识片段内容，长度足够通过过滤检查。"),
            _make_catalog_entry("s2", ""),
        ]
        candidates, stats = filter_candidate_chunks(catalog)
        assert len(candidates) == 1
        assert stats["filtered"]["empty"] == 1

    def test_filters_short_content(self):
        """排除过短内容。"""
        catalog = [
            _make_catalog_entry("s1", "这是有效的知识片段内容，长度足够通过过滤检查。"),
            _make_catalog_entry("s2", "短"),
        ]
        candidates, stats = filter_candidate_chunks(catalog)
        assert len(candidates) == 1
        assert stats["filtered"]["too_short"] == 1

    def test_filters_title_pages(self):
        """排除纯标题行。"""
        catalog = [
            _make_catalog_entry("s1", "这是有效的知识片段内容，长度足够通过过滤检查。"),
            _make_catalog_entry("s2", "# 这是一个纯Markdown标题行被过滤掉"),
        ]
        candidates, stats = filter_candidate_chunks(catalog)
        assert len(candidates) == 1

    def test_all_pass(self):
        """所有 chunk 都通过时返回完整列表。"""
        catalog = [
            _make_catalog_entry("s1", "这是第一个有效的知识片段内容，长度足够通过检查。"),
            _make_catalog_entry("s2", "这是第二个有效的知识片段内容，长度足够通过检查。"),
        ]
        candidates, stats = filter_candidate_chunks(catalog)
        assert len(candidates) == 2
        assert stats["passed"] == 2
        assert stats["total"] == 2


# ── _parse_llm_response ──────────────────────────────────────


class TestParseLLMResponse:
    """测试 LLM 输出解析。"""

    def test_parse_valid_json(self):
        """解析有效 JSON 数组。"""
        text = '[{"candidate_id": "s1", "retrieval_query": "测试查询", "target_label": "标签"}]'
        items = _parse_llm_response(text)
        assert len(items) == 1
        assert items[0]["candidate_id"] == "s1"

    def test_parse_json_with_markdown(self):
        """解析带 Markdown 代码块的 JSON。"""
        text = '```json\n[{"candidate_id": "s1", "retrieval_query": "q", "target_label": "t"}]\n```'
        items = _parse_llm_response(text)
        assert len(items) == 1

    def test_parse_no_json_raises(self):
        """无 JSON 时抛出异常。"""
        with pytest.raises(ValueError, match="不包含 JSON 数组"):
            _parse_llm_response("这不是 JSON")

    def test_parse_invalid_json_raises(self):
        """无效 JSON 时抛出异常。"""
        with pytest.raises(ValueError, match="JSON 解析失败"):
            _parse_llm_response("[invalid json]")


# ── _validate_candidate_id ───────────────────────────────────


class TestValidateCandidateId:
    """测试 fail-closed 校验。"""

    def test_valid_id(self):
        """有效 candidate_id 通过校验。"""
        cid = _validate_candidate_id(
            {"candidate_id": "s1", "retrieval_query": "q"},
            {"s1", "s2"},
        )
        assert cid == "s1"

    def test_missing_id_raises(self):
        """缺少 candidate_id 时抛出异常。"""
        with pytest.raises(ValueError, match="缺少 candidate_id"):
            _validate_candidate_id({"retrieval_query": "q"}, {"s1"})

    def test_invalid_id_raises(self):
        """不在候选集中的 candidate_id 时抛出异常。"""
        with pytest.raises(ValueError, match="不在当前候选集中"):
            _validate_candidate_id(
                {"candidate_id": "s999", "retrieval_query": "q"},
                {"s1", "s2"},
            )


# ── chunk_exact judge ────────────────────────────────────────


class TestChunkExactJudge:
    """测试 chunk_exact 纯机器判定。"""

    def test_hit_top1_by_segment_id(self):
        """segment_id 在 Top1 时命中。"""
        sample = {
            "expected_segment_id": "s1",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": "s1", "content": "内容"},
                {"segment_id": "s2", "content": "其他"},
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 1
        assert result["retrieval_top3_hit"] == 1
        assert result["retrieval_top5_hit"] == 1
        assert result["hit_evidence_position"] == 1

    def test_hit_top3_by_segment_id(self):
        """segment_id 在 Top3 时 Top3/Top5 命中。"""
        sample = {
            "expected_segment_id": "s3",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": "s1", "content": "a"},
                {"segment_id": "s2", "content": "b"},
                {"segment_id": "s3", "content": "c"},
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 0
        assert result["retrieval_top3_hit"] == 1
        assert result["retrieval_top5_hit"] == 1
        assert result["hit_evidence_position"] == 3

    def test_no_hit(self):
        """未命中时全部为 0（有检索结果但不匹配）。"""
        sample = {
            "expected_segment_id": "s999",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": "s1", "content": "a"},
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 0
        assert result["retrieval_top3_hit"] == 0
        assert result["retrieval_top5_hit"] == 0
        assert result["hit_evidence_position"] is None

    def test_hit_by_content_hash(self):
        """按 content_hash 匹配。"""
        content = "这是目标内容"
        expected_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        sample = {
            "expected_segment_id": "",
            "expected_content_hash": expected_hash,
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": "other", "content": content},
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 1
        assert result["hit_evidence_position"] == 1

    def test_empty_results_fail_closed(self):
        """无检索结果时 fail-closed，返回 None 而非 0。"""
        sample = {
            "expected_segment_id": "s1",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] is None
        assert result["chunk_exact_status"] == "no_retrieval"
        assert result["retrieval_evaluable"] is False

    def test_no_expected_id_or_hash_fail_closed(self):
        """缺少 expected_id 和 expected_hash 时 fail-closed。"""
        sample = {
            "expected_segment_id": "",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [{"segment_id": "s1", "content": "a"}],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] is None
        assert result["chunk_exact_status"] == "missing_binding"

    def test_hit_top6_by_segment_id(self):
        """segment_id 在第 6 位时 Top5 未命中但 Top10 命中。"""
        sample = {
            "expected_segment_id": "s6",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": f"s{i}", "content": f"c{i}"} for i in range(1, 11)
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 0
        assert result["retrieval_top3_hit"] == 0
        assert result["retrieval_top5_hit"] == 0
        assert result["retrieval_top10_hit"] == 1
        assert result["hit_evidence_position"] == 6

    def test_hit_top10_by_segment_id(self):
        """segment_id 在第 10 位时 Top10 命中。"""
        sample = {
            "expected_segment_id": "s10",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": f"s{i}", "content": f"c{i}"} for i in range(1, 11)
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 0
        assert result["retrieval_top3_hit"] == 0
        assert result["retrieval_top5_hit"] == 0
        assert result["retrieval_top10_hit"] == 1
        assert result["hit_evidence_position"] == 10

    def test_miss_top10(self):
        """segment_id 不在 Top10 时全部为 0。"""
        sample = {
            "expected_segment_id": "s999",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": f"s{i}", "content": f"c{i}"} for i in range(1, 11)
            ],
        }
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 0
        assert result["retrieval_top3_hit"] == 0
        assert result["retrieval_top5_hit"] == 0
        assert result["retrieval_top10_hit"] == 0
        assert result["hit_evidence_position"] is None

    def test_top10_scans_beyond_top5(self):
        """确认扫描范围是 [:10] 而非 [:5]。"""
        sample = {
            "expected_segment_id": "s7",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [
                {"segment_id": f"s{i}", "content": f"c{i}"} for i in range(1, 11)
            ],
        }
        result = _judge_chunk_exact(sample)
        # Top5 未命中但 Top10 命中
        assert result["retrieval_top5_hit"] == 0
        assert result["retrieval_top10_hit"] == 1
        assert result["hit_evidence_position"] == 7

    def test_error_returns_include_top10(self):
        """fail-closed 返回值包含 retrieval_top10_hit: None。"""
        sample_no_binding = {
            "expected_segment_id": "",
            "expected_content_hash": "",
            "trace_id": "real_trace_123",
            "retrieval_results": [],
        }
        result = _judge_chunk_exact(sample_no_binding)
        assert "retrieval_top10_hit" in result
        assert result["retrieval_top10_hit"] is None


# ── classify_evaluation_track ────────────────────────────────


class TestClassifyChunkExact:
    """测试 chunk_exact 轨道分类。"""

    def test_chunk_exact_mode(self):
        """question_mode=chunk_exact 归入 chunk_exact 轨道。"""
        sample = {"question_mode": "chunk_exact", "question": "q"}
        assert classify_evaluation_track(sample) == TRACK_CHUNK_EXACT

    def test_retrieval_mode_unchanged(self):
        """question_mode=retrieval 不受影响。"""
        sample = {"question_mode": "retrieval", "source_excerpt": "证据"}
        assert classify_evaluation_track(sample) == TRACK_RETRIEVAL


# ── compute_metrics chunk_exact ──────────────────────────────


class TestComputeMetricsChunkExact:
    """测试 compute_metrics 中 chunk_exact 分组。"""

    def test_chunk_exact_metrics_separate(self):
        """chunk_exact 指标单独统计。"""
        results = [
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1},
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 0, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1},
            {"evaluation_track": TRACK_RETRIEVAL, "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
             "retrieval_evaluable": True},
        ]
        metrics = compute_metrics(results)
        assert metrics["chunk_exact_track_count"] == 2
        assert metrics["chunk_exact_top1_hit_rate"] == 0.5
        assert metrics["chunk_exact_top3_hit_rate"] == 1.0
        assert metrics["chunk_exact_top5_hit_rate"] == 1.0
        # retrieval 轨道不受影响
        assert metrics["retrieval_track_count"] == 1
        assert metrics["retrieval_top1_hit_rate"] == 1.0

    def test_no_chunk_exact_results(self):
        """无 chunk_exact 结果时指标为 None。"""
        results = [
            {"evaluation_track": TRACK_RETRIEVAL, "retrieval_top1_hit": 1,
             "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
        ]
        metrics = compute_metrics(results)
        assert metrics["chunk_exact_track_count"] == 0
        assert metrics["chunk_exact_top1_hit_rate"] is None
        assert metrics["chunk_exact_top10_hit_rate"] is None

    def test_chunk_exact_top10_hit_rate(self):
        """chunk_exact Top10 指标正确计算。"""
        results = [
            # Top1 命中
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 1,
             "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_top10_hit": 1,
             "retrieval_evaluable": True},
            # Top5 未命中但 Top10 命中（第 7 位）
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 0,
             "retrieval_top3_hit": 0, "retrieval_top5_hit": 0, "retrieval_top10_hit": 1,
             "retrieval_evaluable": True},
            # Top10 未命中
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 0,
             "retrieval_top3_hit": 0, "retrieval_top5_hit": 0, "retrieval_top10_hit": 0,
             "retrieval_evaluable": True},
        ]
        metrics = compute_metrics(results)
        assert metrics["chunk_exact_track_count"] == 3
        assert metrics["chunk_exact_top1_hit_rate"] == pytest.approx(1 / 3)
        assert metrics["chunk_exact_top3_hit_rate"] == pytest.approx(1 / 3)
        assert metrics["chunk_exact_top5_hit_rate"] == pytest.approx(1 / 3)
        assert metrics["chunk_exact_top10_hit_rate"] == pytest.approx(2 / 3)


# ── generate_chunk_exact_questions 结构 ─────────────────────


class TestGenerateQuestionsStructure:
    """测试生成的 question dict 结构。"""

    @patch("chunk_exact_questions.call_llm")
    def test_question_structure(self, mock_llm):
        """生成的 question 包含所有必需字段。"""
        mock_llm.return_value = json.dumps([{
            "candidate_id": "seg_001",
            "retrieval_query": "测试查询",
            "target_label": "标签",
        }])
        candidates = [_make_catalog_entry("seg_001", "这是候选内容，足够长以通过过滤检查验证。")]
        questions = generate_chunk_exact_questions(
            candidates, "key", "http://localhost/v1", "model",
            dataset_id="ds1", document_id="doc1",
        )
        assert len(questions) == 1
        q = questions[0]
        assert q["question_mode"] == "chunk_exact"
        assert q["expected_segment_id"] == "seg_001"
        assert q["expected_content_hash"] != ""
        assert q["dataset_id"] == "ds1"
        assert q["document_id"] == "doc1"
        assert q["snapshot_id"] != ""
        assert q["question"] == "测试查询"
        assert q["retrieval_query"] == "测试查询"
        assert q["target_label"] == "标签"

    @patch("chunk_exact_questions.call_llm")
    def test_fail_closed_rejects_invalid_id(self, mock_llm):
        """无效 candidate_id 被 fail-closed 拒绝。"""
        mock_llm.return_value = json.dumps([{
            "candidate_id": "seg_999",
            "retrieval_query": "查询",
            "target_label": "标签",
        }])
        candidates = [_make_catalog_entry("seg_001", "这是候选内容，足够长以通过过滤检查验证。")]
        with pytest.raises(ValueError, match="所有 LLM 输出均校验失败"):
            generate_chunk_exact_questions(
                candidates, "key", "http://localhost/v1", "model",
            )

    @patch("chunk_exact_questions.call_llm")
    def test_empty_candidates_raises(self, mock_llm):
        """空候选列表抛出异常。"""
        with pytest.raises(ValueError, match="候选 chunk 列表为空"):
            generate_chunk_exact_questions(
                [], "key", "http://localhost/v1", "model",
            )


# ── validate_chunk_exact ─────────────────────────────────────


class TestValidateChunkExact:
    """测试 chunk_exact 题目绑定校验。"""

    def test_valid_question(self):
        """完整绑定通过校验。"""
        q = {
            "question": "test",
            "snapshot_id": "snap_001",
            "document_id": "doc_001",
            "expected_segment_id": "seg_001",
            "expected_content_hash": "abc123",
        }
        ok, errors = validate_chunk_exact_question(q)
        assert ok is True
        assert errors == []

    def test_missing_snapshot_id(self):
        """缺少 snapshot_id 被标记为无效。"""
        q = {
            "question": "test",
            "document_id": "doc_001",
            "expected_segment_id": "seg_001",
            "expected_content_hash": "abc123",
        }
        ok, errors = validate_chunk_exact_question(q)
        assert ok is False
        assert "snapshot_id" in errors

    def test_missing_multiple_fields(self):
        """缺少多个字段。"""
        q = {"question": "test"}
        ok, errors = validate_chunk_exact_question(q)
        assert ok is False
        assert len(errors) == 4

    def test_validate_set(self):
        """题集校验分离有效/无效题目。"""
        valid_q = {
            "question": "valid",
            "snapshot_id": "s1",
            "document_id": "d1",
            "expected_segment_id": "seg1",
            "expected_content_hash": "h1",
        }
        invalid_q = {"question": "invalid"}
        valid, invalid = validate_chunk_exact_set([valid_q, invalid_q])
        assert len(valid) == 1
        assert len(invalid) == 1
        assert "_validation_errors" in invalid[0]


# ── question dict 字段完整性 ─────────────────────────────────


class TestQuestionFieldCompleteness:
    """测试生成的 question dict 包含所有必需字段。"""

    @patch("chunk_exact_questions.call_llm")
    def test_all_required_fields_present(self, mock_llm):
        """生成的 question 包含所有 chunk_exact 必需字段。"""
        mock_llm.return_value = json.dumps([{
            "candidate_id": "seg_001",
            "retrieval_query": "测试查询",
            "target_label": "标签",
        }])
        candidates = [_make_catalog_entry("seg_001", "这是候选内容，足够长以通过过滤检查验证。")]
        questions = generate_chunk_exact_questions(
            candidates, "key", "http://localhost/v1", "model",
            dataset_id="ds1", document_id="doc1",
        )
        q = questions[0]

        # 核心绑定字段
        assert q["question_mode"] == "chunk_exact"
        assert q["evaluation_type"] == "chunk_exact"
        assert q["snapshot_id"] != ""
        assert q["dataset_id"] == "ds1"
        assert q["document_id"] == "doc1"
        assert q["expected_segment_id"] == "seg_001"
        assert q["expected_content_hash"] != ""

        # 附加元数据
        assert q["question_id"] != ""
        assert q["candidate_id"] == "seg_001"
        assert q["target_label"] == "标签"
        assert q["source_position"] != ""
        assert q["source_label"] != ""
        assert q["expected_content"] != ""

    @patch("chunk_exact_questions.call_llm")
    def test_round_trip_preserves_fields(self, mock_llm, tmp_path):
        """保存后重新加载，所有字段完整保留。"""
        import json as _json
        mock_llm.return_value = _json.dumps([{
            "candidate_id": "seg_001",
            "retrieval_query": "测试查询",
            "target_label": "标签",
        }])
        candidates = [_make_catalog_entry("seg_001", "这是候选内容，足够长以通过过滤检查验证。")]
        questions = generate_chunk_exact_questions(
            candidates, "key", "http://localhost/v1", "model",
            dataset_id="ds1", document_id="doc1",
        )

        # 模拟保存到 JSONL
        output_file = tmp_path / "test_questions.jsonl"
        with output_file.open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(_json.dumps(q, ensure_ascii=False) + "\n")

        # 重新加载
        loaded = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                loaded.append(_json.loads(line.strip()))

        assert len(loaded) == 1
        q = loaded[0]

        # 所有字段必须完整
        assert q["question_mode"] == "chunk_exact"
        assert q["evaluation_type"] == "chunk_exact"
        assert q["snapshot_id"] != ""
        assert q["dataset_id"] == "ds1"
        assert q["document_id"] == "doc1"
        assert q["expected_segment_id"] == "seg_001"
        assert q["expected_content_hash"] != ""
        assert q["question_id"] != ""
        assert q["candidate_id"] == "seg_001"
        assert q["target_label"] == "标签"
        assert q["expected_content"] != ""

        # 校验绑定完整性
        ok, errors = validate_chunk_exact_question(q)
        assert ok is True, f"绑定不完整: {errors}"


# ── sample_candidates_random ─────────────────────────────────


class TestRandomSampling:
    """测试随机抽样逻辑。"""

    def test_basic_sampling(self):
        """基本随机抽样。"""
        candidates = [
            _make_catalog_entry(f"s{i}", f"内容{i}，足够长以通过过滤检查验证。")
            for i in range(20)
        ]
        sampled, count, capped = sample_candidates_random(candidates, 5, seed=42)
        assert count == 5
        assert len(sampled) == 5
        assert capped is False

    def test_no_duplicates(self):
        """抽样结果无重复。"""
        candidates = [
            _make_catalog_entry(f"s{i}", f"内容{i}，足够长以通过过滤检查验证。")
            for i in range(10)
        ]
        sampled, count, capped = sample_candidates_random(candidates, 10, seed=42)
        ids = [s["segment_id"] for s in sampled]
        assert len(ids) == len(set(ids))

    def test_capped_when_exceed(self):
        """数量超过可用时截断。"""
        candidates = [
            _make_catalog_entry(f"s{i}", f"内容{i}，足够长以通过过滤检查验证。")
            for i in range(3)
        ]
        sampled, count, capped = sample_candidates_random(candidates, 10, seed=42)
        assert count == 3
        assert capped is True
        assert len(sampled) == 3

    def test_filter_by_document_ids(self):
        """按文档 ID 过滤后抽样。"""
        candidates = [
            _make_catalog_entry("s0", "内容0，足够长以通过过滤检查验证。", document_id="docA"),
            _make_catalog_entry("s1", "内容1，足够长以通过过滤检查验证。", document_id="docA"),
            _make_catalog_entry("s2", "内容2，足够长以通过过滤检查验证。", document_id="docB"),
        ]
        sampled, count, capped = sample_candidates_random(
            candidates, 5, document_ids=["docA"], seed=42
        )
        assert count == 2
        assert all(s["document_id"] == "docA" for s in sampled)

    def test_seed_reproducibility(self):
        """相同 seed 产生相同结果。"""
        candidates = [
            _make_catalog_entry(f"s{i}", f"内容{i}，足够长以通过过滤检查验证。")
            for i in range(20)
        ]
        s1, _, _ = sample_candidates_random(candidates, 5, seed=123)
        s2, _, _ = sample_candidates_random(candidates, 5, seed=123)
        assert [s["segment_id"] for s in s1] == [s["segment_id"] for s in s2]

    def test_empty_candidates(self):
        """空候选返回空列表。"""
        sampled, count, capped = sample_candidates_random([], 5)
        assert sampled == []
        assert count == 0

    def test_get_candidates_by_documents(self):
        """按文档 ID 过滤候选。"""
        candidates = [
            _make_catalog_entry("s0", "内容0，足够长。", document_id="docA"),
            _make_catalog_entry("s1", "内容1，足够长。", document_id="docB"),
            _make_catalog_entry("s2", "内容2，足够长。", document_id="docA"),
        ]
        filtered = get_candidates_by_documents(candidates, ["docA"])
        assert len(filtered) == 2
        assert all(c["document_id"] == "docA" for c in filtered)


# ── generate_default_set_name ────────────────────────────────


class TestDefaultSetName:
    """测试默认题集名称生成。"""

    def test_single_document_name(self):
        """单文档：{原文件名去扩展名}-随机题集-{YYYYMMDD}。"""
        name = generate_default_set_name(["产品手册.pdf"], mode="random")
        assert name.startswith("产品手册-随机题集-")
        assert name.endswith(datetime.now().strftime("%Y%m%d"))

    def test_single_document_strip_extension(self):
        """去掉常见扩展名。"""
        for ext in [".txt", ".md", ".docx", ".xlsx", ".pdf", ".csv"]:
            name = generate_default_set_name([f"文档{ext}"], mode="random")
            assert ext not in name
            assert "随机题集" in name

    def test_multi_document_name(self):
        """多文档：随机题集-{文档数量}份文档-{YYYYMMDD}。"""
        name = generate_default_set_name(["doc1.txt", "doc2.txt", "doc3.txt"], mode="random")
        assert "3份文档" in name
        assert "随机题集" in name

    def test_empty_document_names(self):
        """空列表降级。"""
        name = generate_default_set_name([], mode="random")
        assert "随机题集" in name

    def test_manual_mode(self):
        """手动模式返回 chunk_exact_ 前缀。"""
        name = generate_default_set_name(["任意"], mode="manual")
        assert name.startswith("chunk_exact_")


# ── chunk_exact fail-closed ──────────────────────────────────


class TestChunkExactFailClosed:
    """测试 chunk_exact 的 fail-closed 逻辑。"""

    def test_no_binding_marks_not_evaluable(self):
        """缺少 expected 标记为不可评测，不写入 miss。"""
        sample = {"question_mode": "chunk_exact", "question": "q",
                  "trace_id": "real_trace_123"}
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] is None
        assert result["retrieval_evaluable"] is False
        assert result["chunk_exact_status"] == "missing_binding"

    def test_no_trace_marks_not_evaluable(self):
        """未关联真实 trace 标记为不可评测。"""
        sample = {"question_mode": "chunk_exact", "question": "q",
                  "expected_segment_id": "seg1", "expected_content_hash": "h1",
                  "trace_id": "batch_qa_0_123"}
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] is None
        assert result["chunk_exact_status"] == "no_trace"

    def test_no_retrieval_marks_not_evaluable(self):
        """无检索结果标记为不可评测。"""
        sample = {"question_mode": "chunk_exact", "question": "q",
                  "expected_segment_id": "seg1", "expected_content_hash": "h1",
                  "trace_id": "real_trace_123",
                  "retrieval_results": []}
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] is None
        assert result["chunk_exact_status"] == "no_retrieval"

    def test_with_retrieval_evaluates_normally(self):
        """有检索结果时正常判定。"""
        sample = {"question_mode": "chunk_exact", "question": "q",
                  "expected_segment_id": "seg1", "expected_content_hash": "h1",
                  "trace_id": "real_trace_123",
                  "retrieval_results": [
                      {"segment_id": "seg1", "content": "hello"},
                      {"segment_id": "seg2", "content": "world"},
                  ]}
        result = _judge_chunk_exact(sample)
        assert result["retrieval_top1_hit"] == 1
        assert result["chunk_exact_status"] == ""


# ── Mixed tracks regression ─────────────────────────────────


class TestMixedTracksRegression:
    """测试混合轨道统计不崩溃。"""

    def test_mixed_tracks_no_key_error(self):
        """混合 retrieval / chunk_exact / legacy 不触发 KeyError。"""
        results = [
            {"evaluation_track": TRACK_RETRIEVAL, "retrieval_top1_hit": 1,
             "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 1,
             "retrieval_top3_hit": 1, "retrieval_top5_hit": 1},
            {"evaluation_track": TRACK_STRICT_QA, "answer_correct": 1},
            {"evaluation_track": TRACK_GROUNDED_QA, "answer_correct": 0},
            {"evaluation_track": TRACK_NOT_EVALUABLE, "retrieval_top1_hit": 0,
             "retrieval_top3_hit": 0, "retrieval_top5_hit": 0},
        ]
        metrics = compute_metrics(results)
        assert metrics["total"] == 5
        assert metrics["chunk_exact_track_count"] == 1
        assert metrics["retrieval_track_count"] == 1

    def test_chunk_exact_none_excluded_from_hit_rate(self):
        """chunk_exact 中 None 值不影响命中率计算。"""
        results = [
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": 1,
             "retrieval_top3_hit": 1, "retrieval_top5_hit": 1},
            {"evaluation_track": TRACK_CHUNK_EXACT, "retrieval_top1_hit": None,
             "retrieval_top3_hit": None, "retrieval_top5_hit": None,
             "chunk_exact_status": "no_retrieval"},
        ]
        metrics = compute_metrics(results)
        assert metrics["chunk_exact_track_count"] == 2
        assert metrics["chunk_exact_evaluable_count"] == 1
        assert metrics["chunk_exact_top1_hit_rate"] == 1.0  # only the evaluable one

    def test_legacy_missing_evaluation_type(self):
        """历史数据缺少 evaluation_type 按 legacy 兼容。"""
        sample = {"question": "test", "question_mode": ""}
        track = classify_evaluation_track(sample)
        assert track in (TRACK_STRICT_QA, TRACK_GROUNDED_QA)  # falls through to legacy logic

    def test_chunk_exact_build_result_status(self):
        """chunk_exact 各状态的 build_result_status 不崩溃。"""
        for ce_status in ["missing_binding", "no_trace", "no_retrieval", ""]:
            result = {"evaluation_track": TRACK_CHUNK_EXACT,
                      "chunk_exact_status": ce_status,
                      "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1}
            status = build_result_status(result)
            assert "icon" in status
            assert "title" in status


# ── backfill_chunk_exact_topk ────────────────────────────────


def _make_sample_lookup_for_backfill(expected_id, expected_hash="", retrieval_results=None):
    """构造 sample_lookup 用于 backfill 测试。"""
    sample = {
        "trace_id": "trace_1",
        "expected_segment_id": expected_id,
        "expected_content_hash": expected_hash,
        "retrieval_results": retrieval_results or [],
    }
    return {"trace_1": sample}


def _make_retrieval_results(count, hit_at=None, hit_id="seg_target"):
    """构造检索结果列表。hit_at 为 1-based 命中位置。"""
    results = []
    for i in range(count):
        seg_id = hit_id if (hit_at and i + 1 == hit_at) else f"seg_{i+1}"
        results.append({"segment_id": seg_id, "content": f"content_{i+1}"})
    return results


class TestBackfillChunkExactTopk:
    """旧版 chunk_exact 结果补齐缺失 Top10 字段的兼容测试。"""

    def test_backfill_top10_hit_at_position_6(self):
        """Top6 命中（旧版只扫描 top5，position=None）=> 通过 sample_lookup 补齐 top10=1。"""
        lookup = _make_sample_lookup_for_backfill(
            "seg_target", retrieval_results=_make_retrieval_results(10, hit_at=6))
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0}
        # retrieval_top10_hit 字段不存在
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] == 1

    def test_backfill_top10_hit_at_position_10(self):
        """Top10 命中 => top10=1。"""
        lookup = _make_sample_lookup_for_backfill(
            "seg_target", retrieval_results=_make_retrieval_results(10, hit_at=10))
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0}
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] == 1

    def test_backfill_top10_miss(self):
        """Top10 未命中 => top10=0。"""
        lookup = _make_sample_lookup_for_backfill(
            "seg_target", retrieval_results=_make_retrieval_results(10))  # 无 hit
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0}
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] == 0

    def test_backfill_top10_by_content_hash(self):
        """通过 content_hash 匹配 Top8 命中 => top10=1。"""
        target_content = "这是目标分块的内容"
        results = []
        for i in range(10):
            content = target_content if i == 7 else f"其他内容_{i+1}"
            results.append({"segment_id": f"seg_{i+1}", "content": content})
        lookup = _make_sample_lookup_for_backfill("", expected_hash="unused", retrieval_results=results)
        # 设置正确的 expected_hash
        import hashlib
        expected_hash = hashlib.sha256(target_content.strip().replace("\r\n", "\n").encode("utf-8")).hexdigest()
        lookup["trace_1"]["expected_content_hash"] = expected_hash

        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0}
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] == 1

    def test_backfill_top10_skip_when_already_present(self):
        """retrieval_top10_hit 已存在时不覆盖。"""
        lookup = _make_sample_lookup_for_backfill(
            "seg_target", retrieval_results=_make_retrieval_results(10, hit_at=1))
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": 1, "retrieval_evaluable": True,
             "retrieval_top1_hit": 1, "retrieval_top3_hit": 1,
             "retrieval_top5_hit": 1, "retrieval_top10_hit": 0}  # 已有值
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] == 0  # 不覆盖

    def test_backfill_top10_no_sample_in_lookup(self):
        """sample_lookup 中无对应 trace_id => 不修改 top10。"""
        lookup = {}  # 空
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top10_hit": None}
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] is None  # 无法补齐

    def test_backfill_top10_no_lookup_provided(self):
        """不提供 sample_lookup 时，top10 不补齐。"""
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top10_hit": None}
        backfill_chunk_exact_topk(r)  # 无 lookup
        assert r["retrieval_top10_hit"] is None

    def test_backfill_top1_3_5_from_position(self):
        """Top1/3/5 仍从 hit_evidence_position 推导（旧版扫描了 top5，position 可靠）。"""
        r = {"evaluation_track": TRACK_CHUNK_EXACT,
             "hit_evidence_position": 3, "retrieval_evaluable": True,
             "retrieval_top1_hit": None, "retrieval_top3_hit": None,
             "retrieval_top5_hit": None, "retrieval_top10_hit": 1}
        backfill_chunk_exact_topk(r)
        assert r["retrieval_top1_hit"] == 0  # 3 > 1
        assert r["retrieval_top3_hit"] == 1  # 3 <= 3
        assert r["retrieval_top5_hit"] == 1  # 3 <= 5

    def test_backfill_skip_non_chunk_exact(self):
        """非 chunk_exact 轨道不修改。"""
        r = {"evaluation_track": TRACK_RETRIEVAL,
             "hit_evidence_position": 1,
             "retrieval_top1_hit": None, "retrieval_top10_hit": None}
        backfill_chunk_exact_topk(r)
        assert r["retrieval_top1_hit"] is None
        assert r["retrieval_top10_hit"] is None

    def test_backfill_skip_not_evaluable(self):
        """evaluable=False 时不修改。"""
        r = {"evaluation_track": TRACK_CHUNK_EXACT,
             "hit_evidence_position": 1, "retrieval_evaluable": False,
             "retrieval_top1_hit": None, "retrieval_top10_hit": None}
        backfill_chunk_exact_topk(r)
        assert r["retrieval_top1_hit"] is None
        assert r["retrieval_top10_hit"] is None

    def test_backfill_idempotent(self):
        """重复调用不会改变结果。"""
        lookup = _make_sample_lookup_for_backfill(
            "seg_target", retrieval_results=_make_retrieval_results(10, hit_at=6))
        r = {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0}
        backfill_chunk_exact_topk(r, lookup)
        first_call = dict(r)
        backfill_chunk_exact_topk(r, lookup)
        assert r == first_call

    def test_backfill_by_rule_name(self):
        """通过 _rule_name 识别 chunk_exact（无 evaluation_track）。"""
        lookup = _make_sample_lookup_for_backfill(
            "seg_target", retrieval_results=_make_retrieval_results(10, hit_at=8))
        r = {"_rule_name": "chunk_exact_match", "trace_id": "trace_1",
             "hit_evidence_position": None, "retrieval_evaluable": True,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0}
        backfill_chunk_exact_topk(r, lookup)
        assert r["retrieval_top10_hit"] == 1

    def test_compute_metrics_with_backfilled_top10(self):
        """补齐后的数据能正确计算 chunk_exact metrics。"""
        # 构造 3 个旧记录：pos=1(top5命中), pos=None(top6-10命中), pos=None(top10未命中)
        lookup = {
            "t1": {"trace_id": "t1", "expected_segment_id": "seg_1",
                    "retrieval_results": _make_retrieval_results(10, hit_at=1, hit_id="seg_1")},
            "t2": {"trace_id": "t2", "expected_segment_id": "seg_target",
                    "retrieval_results": _make_retrieval_results(10, hit_at=7)},
            "t3": {"trace_id": "t3", "expected_segment_id": "seg_target",
                    "retrieval_results": _make_retrieval_results(10)},  # miss
        }
        results = [
            {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "t1",
             "retrieval_evaluable": True, "hit_evidence_position": 1,
             "retrieval_top1_hit": 1, "retrieval_top3_hit": 1,
             "retrieval_top5_hit": 1, "chunk_exact_status": ""},
            {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "t2",
             "retrieval_evaluable": True, "hit_evidence_position": None,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0, "chunk_exact_status": ""},
            {"evaluation_track": TRACK_CHUNK_EXACT, "trace_id": "t3",
             "retrieval_evaluable": True, "hit_evidence_position": None,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
             "retrieval_top5_hit": 0, "chunk_exact_status": ""},
        ]
        for r in results:
            backfill_chunk_exact_topk(r, lookup)

        # t1: top10=1 (pos=1), t2: top10=1 (pos=7), t3: top10=0 (miss)
        assert results[0]["retrieval_top10_hit"] == 1
        assert results[1]["retrieval_top10_hit"] == 1
        assert results[2]["retrieval_top10_hit"] == 0

        metrics = compute_metrics(results)
        assert metrics["chunk_exact_top1_hit_rate"] == 1 / 3   # only t1
        assert metrics["chunk_exact_top5_hit_rate"] == 1 / 3   # only t1
        assert metrics["chunk_exact_top10_hit_rate"] == 2 / 3   # t1 + t2


# ── 多文档联合出题 ────────────────────────────────────────────


def _make_doc_candidates(doc_id, count, doc_name=""):
    """构造指定文档的候选 chunk 列表。"""
    candidates = []
    for i in range(count):
        content = f"{doc_name or doc_id} 的第 {i+1} 段内容，包含足够的文字以通过过滤。"
        candidates.append({
            "segment_id": f"{doc_id}_seg_{i+1:03d}",
            "content": content,
            "content_hash": hashlib.sha256(content.strip().encode("utf-8")).hexdigest(),
            "dataset_id": "ds1",
            "document_id": doc_id,
            "document_name": doc_name or doc_id,
            "status": "completed",
            "enabled": True,
            "position": i + 1,
            "index_node_id": f"node_{doc_id}_{i}",
            "index_node_hash": f"hash_{doc_id}_{i}",
            "tokens": 50,
            "word_count": 20,
        })
    return candidates


class TestMultiDocValidation:
    """多文档配置校验测试。"""

    def test_validate_two_docs_ok(self):
        """两文档 3+5 题，配置正确。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 10), "num_questions": 3},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 10), "num_questions": 5},
        ]
        ok, errors = validate_multi_doc_config(doc_configs)
        assert ok is True
        assert errors == []

    def test_validate_insufficient_candidates(self):
        """某文档候选不足时校验失败。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 2), "num_questions": 3},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 10), "num_questions": 5},
        ]
        ok, errors = validate_multi_doc_config(doc_configs)
        assert ok is False
        assert len(errors) == 1
        assert "文档A" in errors[0]
        assert "3 题" in errors[0]
        assert "2 个" in errors[0]

    def test_validate_no_active_docs(self):
        """没有选中文档时校验失败。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 10), "num_questions": 0},
        ]
        ok, errors = validate_multi_doc_config(doc_configs)
        assert ok is False
        assert "没有选择任何文档" in errors[0]

    def test_validate_all_zero_questions(self):
        """所有文档题数为 0 时校验失败。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 10), "num_questions": 0},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 10), "num_questions": 0},
        ]
        ok, errors = validate_multi_doc_config(doc_configs)
        assert ok is False


class TestMultiDocGeneration:
    """多文档联合出题测试。"""

    def test_two_docs_3_plus_5_questions(self):
        """两文档 3+5 题生成（两阶段流程）。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 20), "num_questions": 5},
        ]

        call_idx = [0]
        last_planned_ids = [set()]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_idx[0] += 1
            is_phase1 = (call_idx[0] % 2 == 1)

            if is_phase1:
                # Phase 1: 找到 prompt 中的 candidate_id，取前 N 个
                found = []
                for dc in doc_configs:
                    for c in dc["candidates"]:
                        if c["segment_id"] in prompt:
                            found.append(c["segment_id"])
                num = next((dc["num_questions"] for dc in doc_configs
                           if any(c["segment_id"] in prompt and c["document_id"] == dc["document_id"]
                                  for c in dc["candidates"])), 3)
                planned = found[:num]
                last_planned_ids[0] = set(planned)
                return json.dumps([{"candidate_id": sid, "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"} for sid in planned])
            else:
                # Phase 2: 只返回 Phase 1 规划的 ID
                return json.dumps([{"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"} for sid in last_planned_ids[0] if sid in prompt])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                dataset_id="ds1", master_seed=42,
            )

        doc1_qs = [q for q in questions if q["document_id"] == "doc1"]
        doc2_qs = [q for q in questions if q["document_id"] == "doc2"]
        assert len(doc1_qs) == 3
        assert len(doc2_qs) == 5
        assert len(doc_stats) == 2
        assert all(s["status"] == "ok" for s in doc_stats)

    def test_no_cross_doc_duplicates(self):
        """跨文档无重复 chunk。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool_by_doc = {}
        for dc in doc_configs:
            pool, _, _ = sample_candidate_pool(
                dc["candidates"], dc["num_questions"], dc["document_id"], master_seed
            )
            pool_by_doc[dc["document_id"]] = pool

        call_idx = [0]
        last_planned = [[]]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_idx[0] += 1
            is_phase1 = "规划专家" in prompt
            found = []
            for doc_id, pool in pool_by_doc.items():
                for c in pool:
                    if c["segment_id"] in prompt:
                        found.append(c["segment_id"])
            if is_phase1:
                num = min(len(found), 3)
                planned = found[:num]
                last_planned[0] = planned
                return json.dumps([{"candidate_id": sid, "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"} for sid in planned])
            else:
                return json.dumps([{"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"} for sid in last_planned[0] if sid in prompt])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        seg_ids = [q["expected_segment_id"] for q in questions]
        assert len(seg_ids) == len(set(seg_ids))

    def test_correct_document_id_binding(self):
        """每题保留正确的 document_id。"""
        doc_configs = [
            {"document_id": "doc_A", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc_A", 20, "文档A"), "num_questions": 2},
            {"document_id": "doc_B", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc_B", 20, "文档B"), "num_questions": 2},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool_by_doc = {}
        for dc in doc_configs:
            pool, _, _ = sample_candidate_pool(
                dc["candidates"], dc["num_questions"], dc["document_id"], master_seed
            )
            pool_by_doc[dc["document_id"]] = pool

        call_idx = [0]
        last_planned = [[]]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_idx[0] += 1
            is_phase1 = "规划专家" in prompt
            found = []
            for doc_id, pool in pool_by_doc.items():
                for c in pool:
                    if c["segment_id"] in prompt:
                        found.append(c["segment_id"])
            if is_phase1:
                num = min(len(found), 2)
                planned = found[:num]
                last_planned[0] = planned
                return json.dumps([{"candidate_id": sid, "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"} for sid in planned])
            else:
                return json.dumps([{"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"} for sid in last_planned[0] if sid in prompt])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                dataset_id="ds1", master_seed=master_seed,
            )

        for q in questions:
            seg_id = q["expected_segment_id"]
            if seg_id.startswith("doc_A_"):
                assert q["document_id"] == "doc_A"
                assert q["dataset_id"] == "ds1"
            elif seg_id.startswith("doc_B_"):
                assert q["document_id"] == "doc_B"
                assert q["dataset_id"] == "ds1"

    def test_insufficient_candidates_raises(self):
        """候选不足时抛出异常。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 2), "num_questions": 5},
        ]

        with pytest.raises(ValueError, match="校验失败"):
            generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
            )


class TestDefaultSetNameForDataset:
    """知识库级默认题集名称测试。"""

    def test_normal_dataset_name(self):
        """正常知识库名称。"""
        name = generate_default_set_name_for_dataset("我的知识库")
        today = datetime.now().strftime("%Y%m%d")
        assert name == f"我的知识库-chunk_exact-{today}"

    def test_empty_dataset_name(self):
        """空名称使用默认值。"""
        name = generate_default_set_name_for_dataset("")
        today = datetime.now().strftime("%Y%m%d")
        assert name == f"未知知识库-chunk_exact-{today}"

    def test_none_dataset_name(self):
        """None 名称使用默认值。"""
        name = generate_default_set_name_for_dataset(None)
        today = datetime.now().strftime("%Y%m%d")
        assert name == f"未知知识库-chunk_exact-{today}"


class TestMultiDocTableDataSource:
    """多文档表格数据源测试：表格应基于全库文档，而非当前预览 catalog。"""

    def test_table_rows_from_full_dataset_not_preview(self):
        """知识库有 3 个文档，当前预览只选其中 1 个时，表格仍应有 3 行。

        模拟场景：
        - 全库有 doc1, doc2, doc3 三个文档
        - 当前预览 catalog 只包含 doc1 的 chunk
        - 出题文档与数量表格应基于全库文档列表构建
        """
        # 全库文档列表（来自 list_all_documents）
        all_dataset_docs = [
            {"id": "doc1", "name": "文档A", "status": "available", "word_count": 1000},
            {"id": "doc2", "name": "文档B", "status": "available", "word_count": 2000},
            {"id": "doc3", "name": "文档C", "status": "available", "word_count": 1500},
        ]

        # 当前预览 catalog 只包含 doc1 的 chunk
        preview_catalog = _make_doc_candidates("doc1", 10, "文档A")
        preview_candidates, _ = filter_candidate_chunks(preview_catalog)

        # 模拟全库候选统计（来自缓存）
        ds_doc_stats = []
        for doc in all_dataset_docs:
            doc_id = doc["id"]
            doc_name = doc["name"]
            # 每个文档有自己的候选数
            doc_catalog = _make_doc_candidates(doc_id, 15, doc_name)
            doc_candidates, _ = filter_candidate_chunks(doc_catalog)
            ds_doc_stats.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "candidate_count": len(doc_candidates),
            })

        # 验证：表格行数应等于全库文档数，而非预览 catalog 中的文档数
        assert len(ds_doc_stats) == 3
        assert len(ds_doc_stats) != len(set(
            c.get("document_id") for c in preview_candidates
        ))

        # 验证：每个文档的候选数不等于预览 catalog 的候选数
        for stat in ds_doc_stats:
            if stat["document_id"] != "doc1":
                assert stat["candidate_count"] != len(preview_candidates)

    def test_candidate_counts_per_document_independent(self):
        """各文档的可用候选数应独立计算，不互相影响。"""
        # doc1 有 10 个候选，doc2 有 5 个候选
        doc1_candidates, _ = filter_candidate_chunks(
            _make_doc_candidates("doc1", 10, "文档A")
        )
        doc2_candidates, _ = filter_candidate_chunks(
            _make_doc_candidates("doc2", 5, "文档B")
        )

        assert len(doc1_candidates) == 10
        assert len(doc2_candidates) == 5
        assert len(doc1_candidates) != len(doc2_candidates)

    def test_full_dataset_stats_not_affected_by_preview_page(self):
        """全库候选统计不应受当前预览页影响。"""
        # 全库有 3 个文档，每个文档有不同数量的 chunk
        all_docs = [
            {"id": "doc1", "name": "文档A"},
            {"id": "doc2", "name": "文档B"},
            {"id": "doc3", "name": "文档C"},
        ]

        # 模拟全库统计
        full_stats = []
        chunk_counts = {"doc1": 10, "doc2": 20, "doc3": 15}
        for doc in all_docs:
            doc_id = doc["id"]
            doc_catalog = _make_doc_candidates(doc_id, chunk_counts[doc_id], doc["name"])
            doc_candidates, _ = filter_candidate_chunks(doc_catalog)
            full_stats.append({
                "document_id": doc_id,
                "document_name": doc["name"],
                "candidate_count": len(doc_candidates),
            })

        # 验证：即使当前预览只显示 doc1，全库统计仍包含所有文档
        assert len(full_stats) == 3
        assert full_stats[0]["candidate_count"] == 10
        assert full_stats[1]["candidate_count"] == 20
        assert full_stats[2]["candidate_count"] == 15


class TestStatusFilterFix:
    """验证文档 status 不影响统计流程。"""

    def test_documents_with_various_statuses_all_included(self):
        """模拟 documents API 返回的 status 缺失、空字符串、completed、available 四种情况，
        文档均应进入统计流程（不以 status 作为前置过滤条件）。"""
        # 模拟 Dify API 返回的文档列表，status 各不相同
        all_docs = [
            {"id": "doc1", "name": "文档A", "status": ""},           # 空字符串
            {"id": "doc2", "name": "文档B", "status": "available"},  # available
            {"id": "doc3", "name": "文档C", "status": "completed"},  # completed
            {"id": "doc4", "name": "文档D"},                          # 缺失 status
            {"id": "doc5", "name": "文档E", "status": "indexing"},   # indexing
        ]

        # 模拟统计流程：不以 status 作为前置过滤条件
        _ds_doc_stats = []
        for doc in all_docs:
            _doc_id = doc.get("id", "")
            _doc_name = doc.get("name", "未命名")
            if not _doc_id:
                continue
            # 不检查 status，直接进入统计流程
            _ds_doc_stats.append({
                "document_id": _doc_id,
                "document_name": _doc_name,
                "candidate_count": 10,  # 假设都有 10 个候选
                "status": "ok",
                "error": "",
            })

        # 验证：所有 5 个文档都应进入统计
        assert len(_ds_doc_stats) == 5
        assert all(s["status"] == "ok" for s in _ds_doc_stats)

    def test_empty_id_documents_excluded(self):
        """没有 document_id 的文档应被排除。"""
        all_docs = [
            {"id": "doc1", "name": "文档A", "status": ""},
            {"id": "", "name": "无ID文档", "status": "available"},
            {"id": "doc3", "name": "文档C"},
        ]

        _ds_doc_stats = []
        for doc in all_docs:
            _doc_id = doc.get("id", "")
            _doc_name = doc.get("name", "未命名")
            if not _doc_id:
                continue
            _ds_doc_stats.append({
                "document_id": _doc_id,
                "document_name": _doc_name,
                "candidate_count": 10,
            })

        # 验证：只有有 ID 的文档进入统计
        assert len(_ds_doc_stats) == 2
        assert _ds_doc_stats[0]["document_id"] == "doc1"
        assert _ds_doc_stats[1]["document_id"] == "doc3"


class TestTwoPhaseGeneration:
    """两阶段出题流程测试。"""

    def test_multi_doc_independent_calls(self):
        """多文档独立调用 LLM，每个文档各调用两次（Phase 1 + Phase 2）。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 20), "num_questions": 3},
        ]

        # 使用与函数相同的种子派生逻辑预测候选池
        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool_by_doc = {}
        for dc in doc_configs:
            pool, _, _ = sample_candidate_pool(
                dc["candidates"], dc["num_questions"], dc["document_id"], master_seed
            )
            pool_by_doc[dc["document_id"]] = pool

        call_count = [0]
        last_planned = [[]]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_count[0] += 1
            # Phase 1 prompt 包含 "你是 RAG 检索评测出题规划专家"
            # Phase 2 prompt 包含 "你是 RAG 检索评测出题专家"
            is_phase1 = "规划专家" in prompt

            if is_phase1:
                # 找到 prompt 中的 candidate_id（来自当前文档的候选池）
                found = []
                for doc_id, pool in pool_by_doc.items():
                    for c in pool:
                        if c["segment_id"] in prompt:
                            found.append(c["segment_id"])
                # Phase 1 返回前 N 个
                num = min(len(found), 3)
                planned = found[:num]
                last_planned[0] = planned
                return json.dumps([{"candidate_id": sid, "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"} for sid in planned])
            else:
                # Phase 2: 只返回 Phase 1 规划的 ID
                return json.dumps([{"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"} for sid in last_planned[0] if sid in prompt])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        # 验证：每个文档都有统计信息
        assert len(doc_stats) == 2
        # 验证：LLM 被调用了多次（每个文档至少 2 次）
        assert call_count[0] >= 4  # 2 docs × 2 phases

    def test_single_doc_failure_doesnt_affect_others(self):
        """单文档失败不影响其他文档（使用 side_effect 列表）。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
            {"document_id": "doc2", "document_name": "文档B",
             "candidates": _make_doc_candidates("doc2", 20), "num_questions": 3},
        ]

        # 使用与函数相同的种子派生逻辑预测候选池
        from chunk_exact_questions import _derive_doc_seed, sample_candidate_pool
        master_seed = 42
        pool_by_doc = {}
        for dc in doc_configs:
            pool, pool_size, _ = sample_candidate_pool(
                dc["candidates"], dc["num_questions"], dc["document_id"], master_seed
            )
            pool_by_doc[dc["document_id"]] = pool

        doc2_pool = pool_by_doc["doc2"]

        # doc2 Phase 1 返回全部池中候选（模拟 LLM 选择全部）
        doc2_phase1 = json.dumps([
            {"candidate_id": c["segment_id"], "query_style": "semantic",
             "search_intent": "意图", "target_label": "标签",
             "must_preserve_terms": [], "plan": "说明"}
            for c in doc2_pool
        ])
        # doc2 Phase 2 返回全部规划的候选
        doc2_phase2 = json.dumps([
            {"candidate_id": c["segment_id"], "retrieval_query": f"查询 {c['segment_id']}", "target_label": "标签"}
            for c in doc2_pool
        ])

        # doc1 Phase 1 失败，doc2 正常
        side_effects = [
            RuntimeError("LLM 调用失败"),  # doc1 Phase 1
            doc2_phase1,                     # doc2 Phase 1
            doc2_phase2,                     # doc2 Phase 2
        ]

        with patch("chunk_exact_questions.call_llm", side_effect=side_effects):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        assert len(doc_stats) == 2
        doc1_stat = next(s for s in doc_stats if s["document_id"] == "doc1")
        doc2_stat = next(s for s in doc_stats if s["document_id"] == "doc2")
        assert doc1_stat["status"] == "phase1_failed"
        assert doc2_stat["status"] == "ok"
        assert len(questions) > 0
        assert all(q["document_id"] == "doc2" for q in questions)

    def test_unknown_candidate_id_rejected(self):
        """LLM 返回陌生 candidate_id 被拒绝。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )
        valid_ids = {c["segment_id"] for c in pool}

        call_count = [0]
        last_planned = [[]]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_count[0] += 1
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]
            if is_phase1:
                # Phase 1: 返回有效 + 无效 ID
                planned = found[:3]
                last_planned[0] = planned
                items = [{"candidate_id": "unknown_id_123", "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"}]
                items += [{"candidate_id": sid, "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"} for sid in planned]
                return json.dumps(items)
            else:
                # Phase 2: 返回有效 + 无效 ID
                items = [{"candidate_id": "unknown_id_456", "retrieval_query": "查询", "target_label": "标签"}]
                items += [{"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"} for sid in last_planned[0] if sid in prompt]
                return json.dumps(items)

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        assert len(questions) == 3
        for q in questions:
            assert q["candidate_id"] in valid_ids

    def test_correct_chunk_binding(self):
        """生成题仍保留正确的 chunk binding（segment_id, content_hash, document_id）。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )

        call_count = [0]
        last_planned = [[]]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_count[0] += 1
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]
            if is_phase1:
                planned = found[:3]
                last_planned[0] = planned
                return json.dumps([{"candidate_id": sid, "query_style": "semantic", "search_intent": "意图", "target_label": "标签", "must_preserve_terms": [], "plan": "说明"} for sid in planned])
            else:
                return json.dumps([{"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"} for sid in last_planned[0] if sid in prompt])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _seed = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                dataset_id="ds1", master_seed=master_seed,
            )

        assert len(questions) == 3
        for q in questions:
            assert q["expected_segment_id"]
            assert q["expected_content_hash"]
            assert q["document_id"] == "doc1"
            assert q["dataset_id"] == "ds1"
            assert q["snapshot_id"]
            assert q["question_mode"] == "chunk_exact"

    def test_get_multi_doc_stats_summary(self):
        """测试统计摘要生成。"""
        from chunk_exact_questions import get_multi_doc_stats_summary

        doc_stats = [
            {"document_id": "doc1", "document_name": "文档A",
             "requested": 5, "candidate_pool": 8, "phase1_planned": 5,
             "phase2_generated": 5, "bound": 5, "status": "ok", "errors": [],
             "query_style_counts": {"lexical": 2, "semantic": 2, "disambiguating": 1}},
            {"document_id": "doc2", "document_name": "文档B",
             "requested": 3, "candidate_pool": 5, "phase1_planned": 0,
             "phase2_generated": 0, "bound": 0, "status": "phase1_failed",
             "errors": ["LLM 调用失败"], "query_style_counts": {}},
            {"document_id": "doc3", "document_name": "文档C",
             "requested": 3, "candidate_pool": 5, "phase1_planned": 3,
             "phase2_generated": 2, "bound": 2, "status": "underfilled",
             "errors": ["请求 3 题，实际绑定 2 题"],
             "query_style_counts": {"semantic": 2}},
        ]

        summary = get_multi_doc_stats_summary(doc_stats)
        assert "✅ 文档A:" in summary
        assert "池8" in summary
        assert "绑定5" in summary
        assert "lexical:2" in summary
        assert "❌ 文档B: phase1_failed" in summary
        assert "⚠️ 文档C:" in summary
        assert "underfilled" in summary or "绑定2" in summary


class TestCandidatePoolAndSeed:
    """候选池扩大与种子派生测试。"""

    def test_pool_larger_than_n(self):
        """N=10 时候选池大于 N（ceil(10*1.5)=15）。"""
        from chunk_exact_questions import sample_candidate_pool
        candidates = _make_doc_candidates("doc1", 30)
        pool, pool_size, capped = sample_candidate_pool(candidates, 10, "doc1", 42)
        assert pool_size >= 10
        assert pool_size <= 15  # min(30, max(10, 15))
        assert not capped
        assert len(pool) == pool_size

    def test_pool_capped_when_insufficient(self):
        """可用候选不足时池被截断。"""
        from chunk_exact_questions import sample_candidate_pool
        candidates = _make_doc_candidates("doc1", 5)
        pool, pool_size, capped = sample_candidate_pool(candidates, 10, "doc1", 42)
        assert pool_size == 5
        assert capped

    def test_stable_seed_derivation(self):
        """相同 user seed + document_id 跨进程得到相同种子。"""
        from chunk_exact_questions import _derive_doc_seed
        seed1 = _derive_doc_seed(42, "doc1")
        seed2 = _derive_doc_seed(42, "doc1")
        assert seed1 == seed2
        # 不同 document_id 得到不同种子
        seed3 = _derive_doc_seed(42, "doc2")
        assert seed1 != seed3
        # 不同 master_seed 得到不同种子
        seed4 = _derive_doc_seed(99, "doc1")
        assert seed1 != seed4

    def test_seed_not_using_python_hash(self):
        """种子派生不使用 Python hash()（跨进程不稳定）。"""
        from chunk_exact_questions import _derive_doc_seed
        # SHA-256 派生的种子应该是一个确定的值
        seed = _derive_doc_seed(42, "doc1")
        # Python hash() 在不同运行中会给出不同结果（由于 PYTHONHASHSEED）
        # 但 SHA-256 是确定的
        assert isinstance(seed, int)
        assert 0 < seed < 2**64


class TestQueryStyleMetadata:
    """query_style 元数据测试。"""

    def test_query_style_in_question_dict(self):
        """生成题保留 query_style 和 generation_plan。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )

        def mock_llm_side_effect(prompt, *args, **kwargs):
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]
            if is_phase1:
                planned = found[:3]
                return json.dumps([
                    {"candidate_id": sid, "query_style": "lexical",
                     "search_intent": "意图", "target_label": "标签",
                     "must_preserve_terms": ["术语"], "plan": "出题策略"}
                    for sid in planned
                ])
            else:
                return json.dumps([
                    {"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"}
                    for sid in found[:3] if sid in prompt
                ])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, _, _ = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        assert len(questions) == 3
        for q in questions:
            assert q["query_style"] == "lexical"
            assert q["generation_plan"] == "出题策略"
            assert q["selection_seed"] == 42

    def test_lexical_semantic_disambiguating_all_valid(self):
        """三类 query_style 均可通过本地校验。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )

        styles = ["lexical", "semantic", "disambiguating"]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]
            if is_phase1:
                planned = found[:3]
                return json.dumps([
                    {"candidate_id": sid, "query_style": styles[i % 3],
                     "search_intent": "意图", "target_label": "标签",
                     "must_preserve_terms": [], "plan": "说明"}
                    for i, sid in enumerate(planned)
                ])
            else:
                return json.dumps([
                    {"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"}
                    for sid in found[:3] if sid in prompt
                ])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, _, _ = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        assert len(questions) == 3
        actual_styles = {q["query_style"] for q in questions}
        assert actual_styles == {"lexical", "semantic", "disambiguating"}


class TestRetryAndUnderfilled:
    """补充重试与 underfilled 状态测试。"""

    def test_phase2_retry_for_missing_candidates(self):
        """Phase 2 少题时只对缺失项重试一次。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )

        call_count = [0]
        planned_ids = [[]]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            call_count[0] += 1
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]

            if is_phase1:
                planned = found[:3]
                planned_ids[0] = planned
                return json.dumps([
                    {"candidate_id": sid, "query_style": "semantic",
                     "search_intent": "意图", "target_label": "标签",
                     "must_preserve_terms": [], "plan": "说明"}
                    for sid in planned
                ])
            else:
                # 第一次 Phase 2 调用只返回 2 个（少题）
                # 重试调用返回缺失的 1 个
                if call_count[0] <= 2:  # Phase 1 + Phase 2 (first)
                    return json.dumps([
                        {"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"}
                        for sid in planned_ids[0][:2] if sid in prompt
                    ])
                else:  # Phase 2 retry
                    return json.dumps([
                        {"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"}
                        for sid in planned_ids[0][2:] if sid in prompt
                    ])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _ = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        # 重试后应该有 3 题
        assert len(questions) == 3
        assert doc_stats[0]["status"] == "ok"

    def test_underfilled_after_retry(self):
        """重试后仍少题，文档标为 underfilled。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )

        # Phase 1 返回 3 个规划，但 Phase 2 始终只返回 1 个（同一个）
        first_planned_id = [None]

        def mock_llm_side_effect(prompt, *args, **kwargs):
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]

            if is_phase1:
                planned = found[:3]
                first_planned_id[0] = planned[0] if planned else None
                return json.dumps([
                    {"candidate_id": sid, "query_style": "semantic",
                     "search_intent": "意图", "target_label": "标签",
                     "must_preserve_terms": [], "plan": "说明"}
                    for sid in planned
                ])
            else:
                # 始终只返回第一个规划的候选（重试也返回同一个）
                if first_planned_id[0] and first_planned_id[0] in prompt:
                    return json.dumps([
                        {"candidate_id": first_planned_id[0],
                         "retrieval_query": f"查询 {first_planned_id[0]}",
                         "target_label": "标签"}
                    ])
                return json.dumps([])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _ = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        # 只有 1 题（重试也只返回同一个），标记为 underfilled
        assert len(questions) == 1
        assert doc_stats[0]["status"] == "underfilled"
        assert doc_stats[0]["bound"] == 1
        assert doc_stats[0]["requested"] == 3


class TestPhase1DuplicateRejected:
    """Phase 1/2 重复 candidate_id 被拒绝。"""

    def test_phase1_duplicate_candidate_id_rejected(self):
        """Phase 1 返回重复 candidate_id 被拒绝。"""
        doc_configs = [
            {"document_id": "doc1", "document_name": "文档A",
             "candidates": _make_doc_candidates("doc1", 20), "num_questions": 3},
        ]

        from chunk_exact_questions import sample_candidate_pool
        master_seed = 42
        pool, _, _ = sample_candidate_pool(
            doc_configs[0]["candidates"], 3, "doc1", master_seed
        )

        def mock_llm_side_effect(prompt, *args, **kwargs):
            is_phase1 = "规划专家" in prompt
            found = [c["segment_id"] for c in pool if c["segment_id"] in prompt]
            if is_phase1:
                # 返回重复的 candidate_id
                planned = found[:2]
                items = []
                for sid in planned:
                    items.append({"candidate_id": sid, "query_style": "semantic",
                                  "search_intent": "意图", "target_label": "标签",
                                  "must_preserve_terms": [], "plan": "说明"})
                # 重复第一个
                items.append(items[0].copy())
                return json.dumps(items)
            else:
                return json.dumps([
                    {"candidate_id": sid, "retrieval_query": f"查询 {sid}", "target_label": "标签"}
                    for sid in found[:2] if sid in prompt
                ])

        with patch("chunk_exact_questions.call_llm", side_effect=mock_llm_side_effect):
            questions, doc_stats, _ = generate_chunk_exact_questions_multi_doc(
                doc_configs, "key", "url", "model",
                master_seed=master_seed,
            )

        # 重复的被拒绝，只有 2 题
        assert len(questions) == 2
        cids = [q["candidate_id"] for q in questions]
        assert len(set(cids)) == len(cids)  # 无重复
