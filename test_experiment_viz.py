"""
运行看板可视化增强测试。

测试内容：
1. 当前 run 过滤：Judge 结果只显示当前 run 的数据
2. 空轨道处理：无检索/QA 数据时不报错
3. 零样本处理：无除零错误
4. 跨 run 不混合：run A 的结果不出现在 run B 中
5. 历史 Judge 无 run_id fallback：通过 trace_id 正确关联
6. 多次运行比较：正确计算对比指标

不调用真实 Dify、Langfuse、LLM API。
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, create_experiment_run,
    load_experiment_run, update_experiment_run,
    get_run_status,
)
from judge import compute_metrics, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA


def _setup_test_env(tmpdir):
    """创建测试目录结构。"""
    dirs = {}
    for name in ["config_profiles", "experiments", "batch", "raw", "processed", "judged"]:
        d = tmpdir / name
        d.mkdir(parents=True)
        dirs[name] = d
    return dirs


def _create_run_with_data(config_id, config_dir, exp_dir, batch_dir, raw_dir, processed_dir, judged_dir,
                          question_count=10, run_label="测试题集"):
    """创建一个完整的运行（batch → raw → processed → judged）。"""
    with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
         patch('experiment.EXPERIMENTS_DIR', exp_dir):
        run_result = create_experiment_run(config_id, run_label, question_count)
        run_id = run_result["run_id"]
        update_experiment_run(run_id, {
            "question_set_id": f"qs_{run_label}",
            "question_set_name": run_label,
        })

    # Batch
    batch_file = f"batch_{run_id}.jsonl"
    batch_path = batch_dir / batch_file
    langfuse_ids = [str(uuid.uuid4()) for _ in range(question_count)]

    with batch_path.open("w", encoding="utf-8") as f:
        for i in range(question_count):
            f.write(json.dumps({
                "success": True,
                "sample": {
                    "trace_id": f"batch_qa_{i}_{run_id}",
                    "question": f"问题 {i+1}",
                    "run_id": run_id,
                    "user_id": f"rag_eval:{run_id}:q_{i:03d}",
                    "question_mode": "retrieval",
                    "reference_answer": f"答案 {i+1}",
                    "retrieval_results": [],
                },
            }, ensure_ascii=False) + "\n")

    with patch('experiment.EXPERIMENTS_DIR', exp_dir):
        update_experiment_run(run_id, {"batch_results_file": batch_file})

    # Raw
    raw_file = f"raw_{run_id}.jsonl"
    raw_path = raw_dir / raw_file
    with raw_path.open("w", encoding="utf-8") as f:
        for i in range(question_count):
            f.write(json.dumps({
                "trace_id": f"batch_qa_{i}_{run_id}",
                "question": f"问题 {i+1}",
                "run_id": run_id,
            }, ensure_ascii=False) + "\n")

    with patch('experiment.EXPERIMENTS_DIR', exp_dir):
        update_experiment_run(run_id, {"raw_results_file": raw_file, "status": "completed"})

    # Processed (real Langfuse trace_ids)
    processed_file = processed_dir / "langfuse_samples.jsonl"
    with processed_file.open("w", encoding="utf-8") as f:
        for i in range(question_count):
            f.write(json.dumps({
                "trace_id": langfuse_ids[i],
                "question": f"问题 {i+1}",
                "run_id": run_id,
                "user_id": f"rag_eval:{run_id}:q_{i:03d}",
                "question_mode": "retrieval",
                "reference_answer": f"答案 {i+1}",
                "retrieval_results": [],
            }, ensure_ascii=False) + "\n")

    # Judged
    judged_file = judged_dir / "eval_results.jsonl"
    return run_id, langfuse_ids, processed_file, judged_file


def test_run_filtering():
    """测试当前 run 过滤：Judge 结果只包含当前 run 的数据。"""
    print("=" * 60)
    print("测试当前 run 过滤")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_test_env(tmpdir)

        with patch('experiment.CONFIG_PROFILES_DIR', dirs["config_profiles"]), \
             patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            config = create_config_profile("测试配置", "kb_v1")
            config_id = config["config_id"]

        run_id, langfuse_ids, processed_file, judged_file = _create_run_with_data(
            config_id, dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], question_count=10,
        )

        # 写入 judged results
        with judged_file.open("w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": langfuse_ids[i],
                    "question": f"问题 {i+1}",
                    "run_id": run_id,
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 1 if i < 8 else 0,
                    "retrieval_top3_hit": 1 if i < 9 else 0,
                    "retrieval_top5_hit": 1,
                    "answer_correct": 1,
                }, ensure_ascii=False) + "\n")

        # 写入其他 run 的 judged results（不应被包含）
        with judged_file.open("a", encoding="utf-8") as f:
            for i in range(5):
                f.write(json.dumps({
                    "trace_id": f"other_trace_{i}",
                    "question": f"其他问题 {i+1}",
                    "run_id": "other_run_id",
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 0,
                    "retrieval_top3_hit": 0,
                    "retrieval_top5_hit": 0,
                    "answer_correct": 0,
                }, ensure_ascii=False) + "\n")

        with patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            status = get_run_status(
                run_id,
                batch_dir=str(dirs["batch"]),
                raw_dir=str(dirs["raw"]),
                processed_file=str(processed_file),
                judged_file=str(judged_file),
            )

        assert status["judge_count"] == 10, f"应有 10 条 Judge 结果，实际 {status['judge_count']}"
        judge_results = status.get("judge_results", [])
        for r in judge_results:
            assert r.get("run_id") == run_id or r.get("trace_id") in langfuse_ids, \
                f"结果不应包含其他 run 的数据: {r.get('trace_id')}"

        # 验证指标
        metrics = compute_metrics(judge_results)
        assert abs(metrics["top1_hit_rate"] - 0.8) < 0.01, f"Top1 应为 80%，实际 {metrics['top1_hit_rate']}"
        assert abs(metrics["top3_hit_rate"] - 0.9) < 0.01, f"Top3 应为 90%，实际 {metrics['top3_hit_rate']}"

        print("[OK] 当前 run 过滤正确，不包含其他 run 的数据")

    print()


def test_empty_tracks():
    """测试空轨道处理：无检索/QA 数据时不报错。"""
    print("=" * 60)
    print("测试空轨道处理")
    print("=" * 60)

    # 模拟只有严格问答，没有检索评测的情况
    strict_only_results = [
        {"trace_id": "t1", "run_id": "r1", "evaluation_track": TRACK_STRICT_QA,
         "answer_correct": 1, "question": "问题1"},
        {"trace_id": "t2", "run_id": "r1", "evaluation_track": TRACK_STRICT_QA,
         "answer_correct": 0, "question": "问题2"},
    ]

    metrics = compute_metrics(strict_only_results)
    assert metrics["retrieval_track_count"] == 0, "应无检索轨道"
    assert metrics["retrieval_top1_hit_rate"] is None, "无检索数据时 Top1 应为 None"
    assert metrics["strict_qa_track_count"] == 2, "应有 2 条严格问答"
    assert abs(metrics["strict_qa_answer_rate"] - 0.5) < 0.01, "QA 正确率应为 50%"
    print("[OK] 无检索轨道时指标为 None，不报错")

    # 模拟只有检索，没有 QA 的情况
    retrieval_only_results = [
        {"trace_id": "t1", "run_id": "r1", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
         "retrieval_evaluable": True, "question": "问题1"},
    ]

    metrics = compute_metrics(retrieval_only_results)
    assert metrics["strict_qa_track_count"] == 0, "应无严格问答轨道"
    assert metrics["strict_qa_answer_rate"] is None, "无 QA 数据时应为 None"
    assert metrics["grounded_qa_track_count"] == 0, "应无合理性问答轨道"
    print("[OK] 无 QA 轨道时指标为 None，不报错")

    # 模拟空结果列表
    empty_metrics = compute_metrics([])
    assert empty_metrics["total"] == 0
    assert empty_metrics["evaluated"] == 0
    assert empty_metrics["top1_hit_rate"] is None
    print("[OK] 空结果列表不报错")

    print()


def test_zero_samples():
    """测试零样本处理：无除零错误。"""
    print("=" * 60)
    print("测试零样本处理")
    print("=" * 60)

    # 计算指标时分母为 0 的情况
    metrics = compute_metrics([])
    assert metrics["total"] == 0
    assert metrics["errors"] == 0
    assert metrics["retrieval_top1_hit_rate"] is None
    assert metrics["strict_qa_answer_rate"] is None
    assert metrics["grounded_qa_answer_rate"] is None
    print("[OK] 零样本无除零错误")

    # 验证可视化代码中的除法安全
    # 模拟 completion rate 计算
    question_count = 0
    batch_total = 0
    batch_success = 0
    processed_count = 0
    judge_count = 0

    _denom = max(question_count, 1)
    _batch_rate = batch_success / max(batch_total, 1) if batch_total > 0 else 0
    _proc_rate = processed_count / _denom
    _judge_rate = judge_count / _denom

    assert _batch_rate == 0
    assert _proc_rate == 0
    assert _judge_rate == 0
    print("[OK] 完成率计算无除零错误")

    print()


def test_cross_run_isolation():
    """测试跨 run 不混合：run A 的结果不出现在 run B 中。"""
    print("=" * 60)
    print("测试跨 run 不混合")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_test_env(tmpdir)

        with patch('experiment.CONFIG_PROFILES_DIR', dirs["config_profiles"]), \
             patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            config = create_config_profile("测试配置", "kb_v1")
            config_id = config["config_id"]

        # 创建 run A
        run_a_id, ids_a, proc_file, judged_file = _create_run_with_data(
            config_id, dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], question_count=10, run_label="题集A",
        )

        # 创建 run B 的 processed 样本（追加到同一文件）
        ids_b = [str(uuid.uuid4()) for _ in range(10)]
        run_b_id = "run_B_test"
        with proc_file.open("a", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": ids_b[i],
                    "question": f"B问题 {i+1}",
                    "run_id": run_b_id,
                    "user_id": f"rag_eval:{run_b_id}:q_{i:03d}",
                    "question_mode": "retrieval",
                    "retrieval_results": [],
                }, ensure_ascii=False) + "\n")

        # 写入 judged results：run A 和 run B 混在一起
        with judged_file.open("w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": ids_a[i],
                    "question": f"A问题 {i+1}",
                    "run_id": run_a_id,
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 1,
                    "retrieval_top3_hit": 1,
                    "retrieval_top5_hit": 1,
                    "answer_correct": 1,
                }, ensure_ascii=False) + "\n")
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": ids_b[i],
                    "question": f"B问题 {i+1}",
                    "run_id": run_b_id,
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 0,
                    "retrieval_top3_hit": 0,
                    "retrieval_top5_hit": 0,
                    "answer_correct": 0,
                }, ensure_ascii=False) + "\n")

        # 查询 run A 的状态
        with patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            status_a = get_run_status(
                run_a_id,
                batch_dir=str(dirs["batch"]),
                raw_dir=str(dirs["raw"]),
                processed_file=str(proc_file),
                judged_file=str(judged_file),
            )

        # 查询 run B 的状态
        # 需要先为 run B 创建 manifest
        with patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            run_b_dir = dirs["experiments"] / run_b_id
            run_b_dir.mkdir(parents=True, exist_ok=True)
            manifest_b = {
                "run_id": run_b_id,
                "config_id": config_id,
                "question_count": 10,
                "question_set_name": "题集B",
                "started_at": "2026-07-14T00:00:00",
                "status": "completed",
            }
            (run_b_dir / "manifest.json").write_text(
                json.dumps(manifest_b, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            status_b = get_run_status(
                run_b_id,
                batch_dir=str(dirs["batch"]),
                raw_dir=str(dirs["raw"]),
                processed_file=str(proc_file),
                judged_file=str(judged_file),
            )

        # 验证隔离
        assert status_a["judge_count"] == 10, f"Run A 应有 10 条，实际 {status_a['judge_count']}"
        assert status_b["judge_count"] == 10, f"Run B 应有 10 条，实际 {status_b['judge_count']}"

        metrics_a = compute_metrics(status_a["judge_results"])
        metrics_b = compute_metrics(status_b["judge_results"])

        # Run A 应该是 100% Top1，Run B 应该是 0% Top1
        assert abs(metrics_a["top1_hit_rate"] - 1.0) < 0.01, \
            f"Run A Top1 应为 100%，实际 {metrics_a['top1_hit_rate']}"
        assert abs(metrics_b["top1_hit_rate"] - 0.0) < 0.01, \
            f"Run B Top1 应为 0%，实际 {metrics_b['top1_hit_rate']}"

        print("[OK] Run A 和 Run B 的指标完全隔离")

    print()


def test_history_judge_no_run_id_fallback():
    """测试历史 Judge 无 run_id 时通过 trace_id fallback 正确关联。"""
    print("=" * 60)
    print("测试历史 Judge 无 run_id fallback")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_test_env(tmpdir)

        with patch('experiment.CONFIG_PROFILES_DIR', dirs["config_profiles"]), \
             patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            config = create_config_profile("测试配置", "kb_v1")
            config_id = config["config_id"]

        run_id, langfuse_ids, processed_file, judged_file = _create_run_with_data(
            config_id, dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], question_count=10,
        )

        # 写入 judged results —— 没有 run_id 字段（模拟旧格式）
        with judged_file.open("w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": langfuse_ids[i],  # 与 processed 相同的真实 UUID
                    "question": f"问题 {i+1}",
                    # 没有 run_id！
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 1 if i < 8 else 0,
                    "retrieval_top3_hit": 1 if i < 9 else 0,
                    "retrieval_top5_hit": 1,
                    "answer_correct": 1,
                }, ensure_ascii=False) + "\n")

        with patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            status = get_run_status(
                run_id,
                batch_dir=str(dirs["batch"]),
                raw_dir=str(dirs["raw"]),
                processed_file=str(processed_file),
                judged_file=str(judged_file),
            )

        assert status["judge_count"] == 10, \
            f"无 run_id 时应通过 trace_id 关联到 10 条，实际 {status['judge_count']}"

        metrics = compute_metrics(status["judge_results"])
        assert abs(metrics["top1_hit_rate"] - 0.8) < 0.01, \
            f"Top1 应为 80%，实际 {metrics['top1_hit_rate']}"
        assert abs(metrics["top3_hit_rate"] - 0.9) < 0.01, \
            f"Top3 应为 90%，实际 {metrics['top3_hit_rate']}"
        assert abs(metrics["top5_hit_rate"] - 1.0) < 0.01, \
            f"Top5 应为 100%，实际 {metrics['top5_hit_rate']}"

        print(f"[OK] 历史 Judge 无 run_id 时通过 trace_id 关联正确")
        print(f"  Top1: {metrics['top1_hit_rate']:.0%}, Top3: {metrics['top3_hit_rate']:.0%}, Top5: {metrics['top5_hit_rate']:.0%}")

    print()


def test_multi_run_comparison():
    """测试多次运行比较：正确计算对比指标。"""
    print("=" * 60)
    print("测试多次运行比较")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_test_env(tmpdir)

        with patch('experiment.CONFIG_PROFILES_DIR', dirs["config_profiles"]), \
             patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            config = create_config_profile("测试配置", "kb_v1")
            config_id = config["config_id"]

        # 创建 run 1
        run1_id, ids1, proc_file, judged_file = _create_run_with_data(
            config_id, dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], question_count=10, run_label="题集1",
        )

        # 创建 run 2 的 processed 样本
        ids2 = [str(uuid.uuid4()) for _ in range(10)]
        run2_id = "run_comparison_2"
        with proc_file.open("a", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": ids2[i],
                    "question": f"问题2_{i+1}",
                    "run_id": run2_id,
                    "user_id": f"rag_eval:{run2_id}:q_{i:03d}",
                    "question_mode": "retrieval",
                    "retrieval_results": [],
                }, ensure_ascii=False) + "\n")

        # 创建 run 2 的 manifest
        run2_dir = dirs["experiments"] / run2_id
        run2_dir.mkdir(parents=True, exist_ok=True)
        manifest2 = {
            "run_id": run2_id,
            "config_id": config_id,
            "question_count": 10,
            "question_set_name": "题集2",
            "started_at": "2026-07-14T01:00:00",
            "status": "completed",
        }
        (run2_dir / "manifest.json").write_text(
            json.dumps(manifest2, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 写入两个 run 的 judged results
        with judged_file.open("w", encoding="utf-8") as f:
            # Run 1: Top1=80%, Top3=90%, Top5=100%
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": ids1[i],
                    "question": f"问题1_{i+1}",
                    "run_id": run1_id,
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 1 if i < 8 else 0,
                    "retrieval_top3_hit": 1 if i < 9 else 0,
                    "retrieval_top5_hit": 1,
                    "answer_correct": 1,
                }, ensure_ascii=False) + "\n")
            # Run 2: Top1=60%, Top3=70%, Top5=80%
            for i in range(10):
                f.write(json.dumps({
                    "trace_id": ids2[i],
                    "question": f"问题2_{i+1}",
                    "run_id": run2_id,
                    "evaluation_track": TRACK_RETRIEVAL,
                    "retrieval_top1_hit": 1 if i < 6 else 0,
                    "retrieval_top3_hit": 1 if i < 7 else 0,
                    "retrieval_top5_hit": 1 if i < 8 else 0,
                    "answer_correct": 1 if i < 5 else 0,
                }, ensure_ascii=False) + "\n")

        # 验证两个 run 的指标
        with patch('experiment.EXPERIMENTS_DIR', dirs["experiments"]):
            status1 = get_run_status(
                run1_id,
                batch_dir=str(dirs["batch"]),
                raw_dir=str(dirs["raw"]),
                processed_file=str(proc_file),
                judged_file=str(judged_file),
            )
            status2 = get_run_status(
                run2_id,
                batch_dir=str(dirs["batch"]),
                raw_dir=str(dirs["raw"]),
                processed_file=str(proc_file),
                judged_file=str(judged_file),
            )

        metrics1 = compute_metrics(status1["judge_results"])
        metrics2 = compute_metrics(status2["judge_results"])

        # 验证指标不同（不会混合）
        assert abs(metrics1["top1_hit_rate"] - 0.8) < 0.01, \
            f"Run1 Top1 应为 80%，实际 {metrics1['top1_hit_rate']}"
        assert abs(metrics2["top1_hit_rate"] - 0.6) < 0.01, \
            f"Run2 Top1 应为 60%，实际 {metrics2['top1_hit_rate']}"

        # 验证比较指标计算
        compare_data = []
        for rid, status in [(run1_id, status1), (run2_id, status2)]:
            j_results = status.get("judge_results", [])
            valid_j = [r for r in j_results if "error" not in r]
            retrieval_j = [r for r in valid_j if r.get("evaluation_track") == TRACK_RETRIEVAL]
            if retrieval_j:
                n = len(retrieval_j)
                t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_j) / n
                t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_j) / n
                t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_j) / n
            else:
                t1 = t3 = t5 = None
            compare_data.append({"rid": rid, "t1": t1, "t3": t3, "t5": t5})

        assert len(compare_data) == 2
        assert compare_data[0]["t1"] != compare_data[1]["t1"], "两次运行的 Top1 应不同"

        print("[OK] 多次运行比较指标正确且不混合")
        print(f"  Run1 Top1: {compare_data[0]['t1']:.0%}, Run2 Top1: {compare_data[1]['t1']:.0%}")

    print()


def test_mixed_tracks_in_run():
    """测试单个 run 内混合评测轨道的处理。"""
    print("=" * 60)
    print("测试混合评测轨道处理")
    print("=" * 60)

    # 模拟一个 run 同时有 retrieval 和 strict_qa 结果
    mixed_results = [
        {"trace_id": "t1", "run_id": "r1", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
         "retrieval_evaluable": True, "answer_correct": 1, "question": "检索题1"},
        {"trace_id": "t2", "run_id": "r1", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 0, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
         "retrieval_evaluable": True, "answer_correct": 0, "question": "检索题2"},
        {"trace_id": "t3", "run_id": "r1", "evaluation_track": TRACK_STRICT_QA,
         "answer_correct": 1, "question": "问答题1"},
        {"trace_id": "t4", "run_id": "r1", "evaluation_track": TRACK_GROUNDED_QA,
         "answer_correct": 0, "question": "问答题2"},
    ]

    metrics = compute_metrics(mixed_results)

    # 检索轨道：2 条
    assert metrics["retrieval_track_count"] == 2
    assert abs(metrics["retrieval_top1_hit_rate"] - 0.5) < 0.01

    # 严格问答：1 条
    assert metrics["strict_qa_track_count"] == 1
    assert abs(metrics["strict_qa_answer_rate"] - 1.0) < 0.01

    # 合理性问答：1 条
    assert metrics["grounded_qa_track_count"] == 1
    assert abs(metrics["grounded_qa_answer_rate"] - 0.0) < 0.01

    # 不混合
    assert metrics["retrieval_track_count"] != metrics["strict_qa_track_count"]
    print("[OK] 混合轨道指标按轨道分组，不混合")

    print()


def test_error_results_handling():
    """测试错误结果的处理。"""
    print("=" * 60)
    print("测试错误结果处理")
    print("=" * 60)

    results_with_errors = [
        {"trace_id": "t1", "run_id": "r1", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
         "retrieval_evaluable": True, "question": "正常题"},
        {"trace_id": "t2", "run_id": "r1", "error": "LLM 调用失败", "question": "错误题"},
        {"trace_id": "t3", "run_id": "r1", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 0, "retrieval_top3_hit": 0, "retrieval_top5_hit": 0,
         "retrieval_evaluable": True, "question": "未命中题"},
    ]

    metrics = compute_metrics(results_with_errors)

    assert metrics["total"] == 3
    assert metrics["evaluated"] == 2  # 只有 2 条有效
    assert metrics["errors"] == 1
    assert metrics["retrieval_track_count"] == 2
    print("[OK] 错误结果正确计数，不纳入指标计算")

    # 验证可视化代码中的筛选逻辑
    valid = [r for r in results_with_errors if "error" not in r]
    error = [r for r in results_with_errors if "error" in r]
    assert len(valid) == 2
    assert len(error) == 1
    print("[OK] 有效/错误结果筛选正确")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("运行看板可视化增强测试")
    print("=" * 60)
    print()

    test_run_filtering()
    test_empty_tracks()
    test_zero_samples()
    test_cross_run_isolation()
    test_history_judge_no_run_id_fallback()
    test_multi_run_comparison()
    test_mixed_tracks_in_run()
    test_error_results_handling()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
