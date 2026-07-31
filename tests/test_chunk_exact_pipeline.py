"""chunk_exact 从样本准备到运行看板的回归测试。"""

import json
from pathlib import Path

from judge import (
    TRACK_CHUNK_EXACT,
    compute_chunk_exact_metrics,
    judge_sample,
    write_chunk_exact_retry_artifacts,
)
from parser import parse_langfuse_jsonl, backfill_reference_answers


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chunk_exact_question(question_id="q-chunk-1"):
    return {
        "question_id": question_id,
        "question": "目标分块的测试问题",
        "question_mode": "chunk_exact",
        "question_set_id": "qs_chunk_exact",
        "expected_segment_id": "segment-target",
        "expected_content_hash": "hash-target",
        "dataset_id": "dataset-1",
        "document_id": "document-1",
        "snapshot_id": "kb-snapshot-1",
        "target_label": "目标分块",
        "evaluation_type": "chunk_exact",
    }


def _trace_rows(question_id="q-chunk-1"):
    return [
        {
            "id": "trace-real-1",
            "traceId": "trace-real-1",
            "type": "TRACE",
            "name": "message",
            "input": {"sys.query": "目标分块的测试问题"},
            "output": {"answer": "ok"},
            "userId": f"rag_eval:run-chunk:{question_id}",
        },
        {
            "id": "obs-retrieval-1",
            "traceId": "trace-real-1",
            "type": "SPAN",
            "name": "知识检索",
            "input": {"query": "目标分块的测试问题"},
            "output": {
                "result": [
                    {
                        "title": "target",
                        "content": "target content",
                        "metadata": {
                            "position": 1,
                            "segment_id": "segment-target",
                        },
                    }
                ]
            },
            "metadata": {"node_type": "knowledge-retrieval"},
            "userId": f"rag_eval:run-chunk:{question_id}",
        },
    ]


def test_chunk_exact_question_id_backfill_preserves_bindings(tmp_path):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    _write_jsonl(questions_dir / "chunk.jsonl", [_chunk_exact_question()])
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, _trace_rows())

    samples, _ = parse_langfuse_jsonl(trace_path)
    # parse_langfuse_jsonl 使用默认题库；直接调用回填函数以注入隔离题库。
    from parser import backfill_reference_answers
    samples, _ = backfill_reference_answers(samples, questions_dir=questions_dir)

    sample = samples[0]
    for field in (
        "expected_segment_id", "expected_content_hash", "dataset_id",
        "document_id", "snapshot_id", "target_label", "evaluation_type",
    ):
        assert sample[field] == _chunk_exact_question()[field]


def test_chunk_exact_never_binds_by_question_text_without_question_id(tmp_path):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    _write_jsonl(questions_dir / "chunk.jsonl", [_chunk_exact_question()])
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, _trace_rows(question_id="unknown-question-id"))

    samples, _ = parse_langfuse_jsonl(trace_path)
    from parser import backfill_reference_answers
    samples, _ = backfill_reference_answers(samples, questions_dir=questions_dir)

    assert "expected_segment_id" not in samples[0]
    assert "expected_content_hash" not in samples[0]


def test_legacy_chunk_exact_recovers_only_verified_deterministic_question_id(tmp_path):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    legacy_question = _chunk_exact_question()
    legacy_question.pop("question_id")
    _write_jsonl(questions_dir / "legacy.jsonl", [legacy_question])

    import hashlib
    question_id = hashlib.md5(legacy_question["question"].encode("utf-8")).hexdigest()[:12]
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, _trace_rows(question_id=question_id))
    samples, _ = parse_langfuse_jsonl(trace_path)
    samples[0]["question_set_id"] = legacy_question["question_set_id"]

    samples, stats = backfill_reference_answers(samples, questions_dir=questions_dir)
    sample = samples[0]
    assert sample["expected_segment_id"] == "segment-target"
    assert sample["binding_source"] == "legacy_deterministic_question_id"
    assert sample["binding_recovery_verified"] is True
    assert stats["legacy_id_recovered"] == 1


def test_legacy_chunk_exact_rejects_deterministic_id_collision(tmp_path):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    first = _chunk_exact_question()
    first.pop("question_id")
    # Patch hashlib locally to model the only collision condition relevant to
    # this fail-closed index, without requiring an impractical real MD5 collision.
    second = dict(first, question="另一道题")
    _write_jsonl(questions_dir / "legacy.jsonl", [first, second])

    import parser
    real_md5 = parser.hashlib.md5
    class _FixedHash:
        def hexdigest(self):
            return "same-question-id" + "0" * 32
    parser.hashlib.md5 = lambda _: _FixedHash()
    try:
        trace_path = tmp_path / "trace.jsonl"
        _write_jsonl(trace_path, _trace_rows(question_id="same-questio"))
        samples, _ = parse_langfuse_jsonl(trace_path)
        samples[0]["question_set_id"] = first["question_set_id"]
        samples, stats = backfill_reference_answers(samples, questions_dir=questions_dir)
    finally:
        parser.hashlib.md5 = real_md5

    assert "expected_segment_id" not in samples[0]
    assert stats["legacy_id_recovery_rejected"] == 1


def test_chunk_exact_with_binding_and_retrieval_is_machine_evaluable():
    sample = {
        "trace_id": "real-langfuse-trace-id",
        "question": "目标分块的测试问题",
        "question_mode": "chunk_exact",
        "expected_segment_id": "segment-target",
        "expected_content_hash": "hash-target",
        "retrieval_results": [{"position": 1, "segment_id": "segment-target", "content": "x"}],
    }

    result = judge_sample(sample, "unused", "unused", "unused")
    assert result["evaluation_track"] == TRACK_CHUNK_EXACT
    assert result["retrieval_evaluable"] is True
    assert result["retrieval_top1_hit"] == 1
    assert result["retrieval_top3_hit"] == 1
    assert result["retrieval_top5_hit"] == 1


def test_chunk_exact_dashboard_metrics_keep_pending_out_of_topk():
    metrics = compute_chunk_exact_metrics([
        {
            "evaluation_track": TRACK_CHUNK_EXACT,
            "chunk_exact_status": "missing_binding",
            "retrieval_evaluable": False,
            "retrieval_top1_hit": None,
        },
        {
            "evaluation_track": TRACK_CHUNK_EXACT,
            "chunk_exact_status": "",
            "retrieval_evaluable": True,
            "retrieval_top1_hit": 1,
            "retrieval_top3_hit": 1,
            "retrieval_top5_hit": 1,
            "retrieval_top10_hit": 1,
        },
    ])

    assert metrics["total_count"] == 2
    assert metrics["evaluable_count"] == 1
    assert metrics["missing_binding_count"] == 1
    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["top10_hit_rate"] == 1.0

    app_source = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
    assert "compute_chunk_exact_metrics(chunk_exact_results)" in app_source
    assert "当前 chunk_exact Judge 结果不可正式使用" in app_source


def test_run_dashboard_prefers_new_chunk_exact_retry_over_old_pending(tmp_path, monkeypatch):
    import experiment
    experiments_dir = tmp_path / "experiments"
    run_id = "run-chunk"
    run_dir = experiments_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "question_count": 1,
        "question_set_id": "qs_chunk_exact", "question_set_name": "chunk",
    }), encoding="utf-8")
    monkeypatch.setattr(experiment, "EXPERIMENTS_DIR", experiments_dir)

    processed = tmp_path / "processed.jsonl"
    judged = tmp_path / "judged" / "eval_results.jsonl"
    judged.parent.mkdir()
    _write_jsonl(processed, [{"run_id": run_id, "trace_id": "real-trace"}])
    _write_jsonl(judged, [{
        "run_id": run_id, "trace_id": "real-trace", "evaluation_track": "chunk_exact",
        "chunk_exact_status": "missing_binding", "retrieval_evaluable": False,
    }])
    _write_jsonl(judged.parent / "chunk_exact_retry_20260730_000000_000000.jsonl", [{
        "run_id": run_id, "question_set_id": "qs_chunk_exact", "trace_id": "real-trace",
        "evaluation_track": "chunk_exact", "source_snapshot_id": "frozen-snap",
        "retrieval_evaluable": True, "retrieval_top1_hit": 1,
    }])

    status = experiment.get_run_status(
        run_id, processed_file=processed, judged_file=judged,
        include_judge_results=True,
    )
    assert status["judge_count"] == 1
    assert status["judge_results"][0]["retrieval_evaluable"] is True
    assert status["judge_results"][0].get("chunk_exact_status") != "missing_binding"


def test_chunk_exact_retry_artifacts_are_new_and_formally_evaluable(tmp_path):
    sample = {
        "trace_id": "real-langfuse-trace-id",
        "question_id": "q-retry",
        "question": "目标分块的测试问题",
        "question_mode": "chunk_exact",
        "question_set_id": "qs_chunk_exact",
        "expected_segment_id": "segment-target",
        "retrieval_results": [{"position": 1, "segment_id": "segment-target", "content": "x"}],
    }
    processed = tmp_path / "processed.jsonl"
    judged = tmp_path / "judged.jsonl"
    artifact = write_chunk_exact_retry_artifacts(
        [sample], processed, judged, source_snapshot_id="snap_frozen",
    )

    assert artifact["sample_count"] == 1
    assert artifact["metrics"]["evaluable_count"] == 1
    assert artifact["metrics"]["top1_hit_rate"] == 1.0
    assert json.loads(processed.read_text(encoding="utf-8"))["source_snapshot_id"] == "snap_frozen"
    assert json.loads(judged.read_text(encoding="utf-8"))["retry_of"] == "missing_binding"


def test_chunk_exact_retry_rejects_pseudo_trace_before_writing(tmp_path):
    sample = {
        "trace_id": "batch_qa_0_legacy",
        "question_mode": "chunk_exact",
        "expected_segment_id": "segment-target",
        "retrieval_results": [{"segment_id": "segment-target"}],
    }
    processed = tmp_path / "processed.jsonl"
    judged = tmp_path / "judged.jsonl"
    import pytest
    with pytest.raises(ValueError, match="real Langfuse trace"):
        write_chunk_exact_retry_artifacts([sample], processed, judged, "snap_frozen")
    assert not processed.exists()
    assert not judged.exists()
