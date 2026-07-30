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
    validate_chunk_exact_question,
    validate_chunk_exact_set,
    sample_candidates_random,
    get_candidates_by_documents,
    generate_default_set_name,
)
from judge import (
    classify_evaluation_track, _judge_chunk_exact, compute_metrics,
    build_result_status,
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
