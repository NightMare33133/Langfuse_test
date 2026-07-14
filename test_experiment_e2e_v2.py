"""
实验版本看板端到端测试 v2。

模拟真实情况：batch pseudo trace_id 与 Langfuse trace_id 不同。

测试内容：
1. batch 文件 trace_id：batch_qa_xxx
2. processed trace_id：真实 langfuse_uuid
3. judged trace_id：同一个真实 langfuse_uuid
4. 断言 get_run_status 返回正确数量
5. 断言指标来自正确的 judged results

不调用真实 Dify、Langfuse、LLM API。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, create_experiment_run,
    load_experiment_run, update_experiment_run,
    get_run_status, migrate_judged_results, migrate_processed_samples,
)
from judge import TRACK_RETRIEVAL, compute_metrics


def test_e2e_with_different_trace_ids():
    """测试 batch trace_id 与 Langfuse trace_id 不同的情况。"""
    print("=" * 60)
    print("测试 batch trace_id 与 Langfuse trace_id 不同的情况")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建目录
        config_dir = tmpdir / "config_profiles"
        exp_dir = tmpdir / "experiments"
        batch_dir = tmpdir / "batch"
        raw_dir = tmpdir / "raw"
        processed_dir = tmpdir / "processed"
        judged_dir = tmpdir / "judged"

        for d in [config_dir, exp_dir, batch_dir, raw_dir, processed_dir, judged_dir]:
            d.mkdir(parents=True)

        # ========== 1. 创建配置和运行 ==========
        print("\n[1] 创建配置和运行...")

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="kb_v1",
            )
            config_id = config["config_id"]

            run_result = create_experiment_run(
                config_id=config_id,
                question_set_source="测试题集",
                question_count=10,
            )
            run_id = run_result["run_id"]

            # 更新 manifest
            update_experiment_run(run_id, {
                "question_set_id": "qs_test_001",
                "question_set_name": "测试题集_检索评测",
            })

            manifest = load_experiment_run(run_id)
            assert manifest["question_set_id"] == "qs_test_001"
            print(f"  [OK] 配置 ID: {config_id}")
            print(f"  [OK] 运行 ID: {run_id}")

        # ========== 2. 创建 Batch 结果（pseudo trace_id） ==========
        print("\n[2] 创建 Batch 结果（pseudo trace_id）...")

        batch_file = "batch_results_20260713_120000.jsonl"
        batch_path = batch_dir / batch_file

        # 生成真实的 Langfuse UUID
        import uuid
        langfuse_trace_ids = [str(uuid.uuid4()) for _ in range(10)]

        batch_results = []
        for i in range(10):
            batch_results.append({
                "success": True,
                "question": f"测试问题 {i+1}",
                "sample": {
                    "trace_id": f"batch_qa_{i}_20260713_120000",  # pseudo trace_id
                    "question": f"测试问题 {i+1}",
                    "final_answer": f"测试回答 {i+1}",
                    "run_id": run_id,
                    "config_id": config_id,
                    "question_id": f"q_{i:03d}",
                    "question_set_id": "qs_test_001",
                    "question_set_name": "测试题集_检索评测",
                    "user_id": f"rag_eval:{run_id}:q_{i:03d}",
                    "retrieval_results": [],
                    "reference_answer": f"参考答案 {i+1}",
                    "question_mode": "retrieval",
                },
            })

        with batch_path.open("w", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 更新 manifest
        with patch('experiment.EXPERIMENTS_DIR', exp_dir):
            update_experiment_run(run_id, {
                "batch_results_file": batch_file,
            })

        print(f"  [OK] Batch 文件: {batch_file} ({len(batch_results)} 条)")

        # ========== 3. 创建 Raw 结果 ==========
        print("\n[3] 创建 Raw 结果...")

        raw_file = "batch_qa_20260713_120000.jsonl"
        raw_path = raw_dir / raw_file
        with raw_path.open("w", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r["sample"], ensure_ascii=False) + "\n")

        # 更新 manifest
        with patch('experiment.EXPERIMENTS_DIR', exp_dir):
            update_experiment_run(run_id, {
                "raw_results_file": raw_file,
                "status": "completed",
            })

        print(f"  [OK] Raw 文件: {raw_file} ({len(batch_results)} 条)")

        # ========== 4. 创建 Processed 样本（真实 Langfuse trace_id） ==========
        print("\n[4] 创建 Processed 样本（真实 Langfuse trace_id）...")

        processed_file = processed_dir / "langfuse_samples.jsonl"
        with processed_file.open("w", encoding="utf-8") as f:
            for i in range(10):
                sample = {
                    "trace_id": langfuse_trace_ids[i],  # 真实 Langfuse UUID
                    "question": f"测试问题 {i+1}",
                    "final_answer": f"测试回答 {i+1}",
                    "user_id": f"rag_eval:{run_id}:q_{i:03d}",
                    "run_id": run_id,
                    "config_id": config_id,
                    "question_id": f"q_{i:03d}",
                    "question_set_id": "qs_test_001",
                    "question_set_name": "测试题集_检索评测",
                    "question_mode": "retrieval",
                    "reference_answer": f"参考答案 {i+1}",
                    "retrieval_results": [],
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"  [OK] Processed 文件: {processed_file.name} ({10} 条)")
        print(f"  [OK] trace_id 与 batch 不同（真实 Langfuse UUID）")

        # ========== 5. 创建 Judge 结果（使用真实 Langfuse trace_id） ==========
        print("\n[5] 创建 Judge 结果（使用真实 Langfuse trace_id）...")

        judged_file = judged_dir / "eval_results.jsonl"
        with judged_file.open("w", encoding="utf-8") as f:
            for i in range(10):
                judge_result = {
                    "trace_id": langfuse_trace_ids[i],  # 与 processed 相同的真实 UUID
                    "question": f"测试问题 {i+1}",
                    # 注意：没有 run_id，模拟旧格式
                    "evaluation_track": TRACK_RETRIEVAL,
                    "has_reference": True,
                    "retrieval_evaluable": True,
                    "retrieval_top1_hit": 1 if i < 8 else 0,
                    "retrieval_top3_hit": 1 if i < 9 else 0,
                    "retrieval_top5_hit": 1,
                    "answer_correct": 1,
                    "reason": f"测试原因 {i+1}",
                }
                f.write(json.dumps(judge_result, ensure_ascii=False) + "\n")

        print(f"  [OK] Judge 文件: {judged_file.name} ({10} 条)")
        print(f"  [OK] trace_id 与 processed 相同（真实 Langfuse UUID）")

        # ========== 6. 验证运行状态（迁移前） ==========
        print("\n[6] 验证运行状态（迁移前）...")

        with patch('experiment.EXPERIMENTS_DIR', exp_dir):
            status = get_run_status(
                run_id,
                batch_dir=str(batch_dir),
                raw_dir=str(raw_dir),
                processed_file=str(processed_file),
                judged_file=str(judged_file),
            )

        print(f"  Batch: {status.get('batch_success')}/{status.get('batch_total')}")
        print(f"  Raw: {status.get('raw_count')}")
        print(f"  Processed: {status.get('processed_count')}")
        print(f"  Judge: {status.get('judge_count')}")

        # 验证
        assert status["batch_success"] == 10, f"Batch success 不正确: {status['batch_success']}"
        assert status["batch_total"] == 10, f"Batch total 不正确: {status['batch_total']}"
        assert status["raw_count"] == 10, f"Raw 不正确: {status['raw_count']}"
        assert status["processed_count"] == 10, f"Processed 不正确: {status['processed_count']}"
        assert status["judge_count"] == 10, f"Judge 不正确: {status['judge_count']}"
        print("  [OK] 迁移前状态正确")

        # ========== 7. 验证 Judge 指标 ==========
        print("\n[7] 验证 Judge 指标...")

        judge_results = status.get("judge_results", [])
        assert len(judge_results) == 10, f"Judge 结果数不正确: {len(judge_results)}"

        metrics = compute_metrics(judge_results)

        # Top1 Hit = 8/10 = 80%
        top1_expected = 8 / 10
        assert abs(metrics["top1_hit_rate"] - top1_expected) < 0.01, f"Top1 不正确: {metrics['top1_hit_rate']}"

        # Top3 Hit = 9/10 = 90%
        top3_expected = 9 / 10
        assert abs(metrics["top3_hit_rate"] - top3_expected) < 0.01, f"Top3 不正确: {metrics['top3_hit_rate']}"

        # Top5 Hit = 10/10 = 100%
        top5_expected = 1.0
        assert abs(metrics["top5_hit_rate"] - top5_expected) < 0.01, f"Top5 不正确: {metrics['top5_hit_rate']}"

        print(f"  Top1 Hit: {metrics['top1_hit_rate']:.0%}")
        print(f"  Top3 Hit: {metrics['top3_hit_rate']:.0%}")
        print(f"  Top5 Hit: {metrics['top5_hit_rate']:.0%}")
        print("  [OK] Judge 指标正确")

        # ========== 8. 测试迁移功能 ==========
        print("\n[8] 测试迁移功能...")

        # 迁移 judged 结果
        result = migrate_judged_results(
            processed_file=str(processed_file),
            judged_file=str(judged_file),
            backup=True,
        )
        print(f"  迁移 Judge 结果: {result['migrated']} 条")

        # 验证迁移后的 judged 结果有 run_id
        with judged_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                obj = json.loads(line)
                assert obj.get("run_id") == run_id, f"迁移后 run_id 不正确: {obj.get('run_id')}"
        print("  [OK] 迁移后 Judge 结果有正确的 run_id")

    print("\n" + "=" * 60)
    print("[OK] 端到端测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_e2e_with_different_trace_ids()
