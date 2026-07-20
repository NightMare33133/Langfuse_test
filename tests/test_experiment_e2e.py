"""
运行看板看板端到端测试。

模拟真实文件链路：
题集 JSONL → batch_results.jsonl → raw JSONL → processed samples JSONL → judged results JSONL

测试内容：
1. 题集信息正确保存到 manifest
2. Batch 10/10
3. Raw 10
4. Processed 10
5. Judge 10
6. 对应轨道指标

不调用真实 Dify、Langfuse、LLM API。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, create_experiment_run,
    load_experiment_run, update_experiment_run,
    get_run_status, backfill_manifest_from_batch,
)
from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA


def test_e2e_experiment_pipeline():
    """端到端测试运行看板看板。"""
    print("=" * 60)
    print("端到端测试运行看板看板")
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
                config_name="端到端测试配置",
                knowledge_base_version="kb_v1",
                workflow_version="wf_v1",
            )
            config_id = config["config_id"]
            print(f"  配置 ID: {config_id}")

            run_result = create_experiment_run(
                config_id=config_id,
                question_set_source="测试题集",
                question_count=10,
            )
            run_id = run_result["run_id"]
            print(f"  运行 ID: {run_id}")

            # 更新 manifest 添加题集信息
            update_experiment_run(run_id, {
                "question_set_id": "qs_test_001",
                "question_set_name": "测试题集_检索评测",
                "question_set_file": "questions_测试题集_20260713.jsonl",
            })

            manifest = load_experiment_run(run_id)
            assert manifest["question_set_id"] == "qs_test_001"
            assert manifest["question_set_name"] == "测试题集_检索评测"
            print("  [OK] 题集信息已保存到 manifest")

        # ========== 2. 创建 Batch 结果 ==========
        print("\n[2] 创建 Batch 结果...")

        batch_results = []
        for i in range(10):
            batch_results.append({
                "success": True,
                "question": f"测试问题 {i+1}",
                "sample": {
                    "trace_id": f"batch_qa_{i}_20260713_120000",
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

        batch_file = f"batch_results_20260713_120000.jsonl"
        batch_path = batch_dir / batch_file
        with batch_path.open("w", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  Batch 文件: {batch_file} ({len(batch_results)} 条)")

        # 更新 manifest
        with patch('experiment.EXPERIMENTS_DIR', exp_dir):
            update_experiment_run(run_id, {
                "batch_results_file": batch_file,
            })

        # ========== 3. 创建 Raw 结果 ==========
        print("\n[3] 创建 Raw 结果...")

        raw_file = "batch_qa_20260713_120000.jsonl"
        raw_path = raw_dir / raw_file
        with raw_path.open("w", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r["sample"], ensure_ascii=False) + "\n")

        print(f"  Raw 文件: {raw_file} ({len(batch_results)} 条)")

        # 更新 manifest
        with patch('experiment.EXPERIMENTS_DIR', exp_dir):
            update_experiment_run(run_id, {
                "raw_results_file": raw_file,
                "status": "completed",
            })

        # ========== 4. 创建 Processed 样本 ==========
        print("\n[4] 创建 Processed 样本...")

        processed_file = processed_dir / "langfuse_samples.jsonl"
        with processed_file.open("w", encoding="utf-8") as f:
            for r in batch_results:
                sample = r["sample"]
                # 模拟 parser 处理后的样本
                processed_sample = {
                    "trace_id": sample["trace_id"],
                    "question": sample["question"],
                    "final_answer": sample["final_answer"],
                    "user_id": sample["user_id"],
                    "run_id": run_id,
                    "config_id": config_id,
                    "question_id": sample["question_id"],
                    "question_set_id": sample["question_set_id"],
                    "question_set_name": sample["question_set_name"],
                    "question_mode": sample["question_mode"],
                    "reference_answer": sample["reference_answer"],
                    "retrieval_results": [],
                }
                f.write(json.dumps(processed_sample, ensure_ascii=False) + "\n")

        print(f"  Processed 文件: {processed_file.name} ({len(batch_results)} 条)")

        # ========== 5. 创建 Judge 结果 ==========
        print("\n[5] 创建 Judge 结果...")

        judged_file = judged_dir / "eval_results.jsonl"
        with judged_file.open("w", encoding="utf-8") as f:
            for i, r in enumerate(batch_results):
                sample = r["sample"]
                # 模拟 Judge 结果
                judge_result = {
                    "trace_id": sample["trace_id"],
                    "question": sample["question"],
                    "run_id": run_id,
                    "config_id": config_id,
                    "question_set_id": sample["question_set_id"],
                    "question_set_name": sample["question_set_name"],
                    "evaluation_track": TRACK_RETRIEVAL,
                    "has_reference": True,
                    "retrieval_evaluable": True,
                    "retrieval_top1_hit": 1 if i < 7 else 0,
                    "retrieval_top3_hit": 1 if i < 9 else 0,
                    "retrieval_top5_hit": 1,
                    "answer_correct": 1,
                    "reason": f"测试原因 {i+1}",
                }
                f.write(json.dumps(judge_result, ensure_ascii=False) + "\n")

        print(f"  Judge 文件: {judged_file.name} ({len(batch_results)} 条)")

        # ========== 6. 验证运行状态 ==========
        print("\n[6] 验证运行状态...")

        with patch('experiment.EXPERIMENTS_DIR', exp_dir):
            status = get_run_status(
                run_id,
                batch_dir=str(batch_dir),
                raw_dir=str(raw_dir),
                processed_file=str(processed_file),
                judged_file=str(judged_file),
            )

        print(f"  question_set_id: {status.get('question_set_id')}")
        print(f"  question_set_name: {status.get('question_set_name')}")
        print(f"  batch_success: {status.get('batch_success')}")
        print(f"  batch_total: {status.get('batch_total')}")
        print(f"  raw_count: {status.get('raw_count')}")
        print(f"  processed_count: {status.get('processed_count')}")
        print(f"  judge_count: {status.get('judge_count')}")

        # 断言
        assert status["question_set_id"] == "qs_test_001", f"question_set_id 不正确: {status['question_set_id']}"
        assert status["question_set_name"] == "测试题集_检索评测", f"question_set_name 不正确: {status['question_set_name']}"
        assert status["batch_success"] == 10, f"batch_success 不正确: {status['batch_success']}"
        assert status["batch_total"] == 10, f"batch_total 不正确: {status['batch_total']}"
        assert status["raw_count"] == 10, f"raw_count 不正确: {status['raw_count']}"
        assert status["processed_count"] == 10, f"processed_count 不正确: {status['processed_count']}"
        assert status["judge_count"] == 10, f"judge_count 不正确: {status['judge_count']}"

        print("  [OK] 运行状态正确")

        # ========== 7. 验证 Judge 指标 ==========
        print("\n[7] 验证 Judge 指标...")

        judge_results = status.get("judge_results", [])
        assert len(judge_results) == 10, f"Judge 结果数不正确: {len(judge_results)}"

        # 计算指标
        from judge import compute_metrics
        metrics = compute_metrics(judge_results)

        # Top1 Hit = 7/10 = 70%
        top1_expected = 7 / 10
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

        # ========== 8. 测试回填功能 ==========
        print("\n[8] 测试回填功能...")

        # 创建一个没有题集信息的运行
        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            run_result2 = create_experiment_run(
                config_id=config_id,
                question_set_source="测试题集2",
                question_count=5,
            )
            run_id2 = run_result2["run_id"]

            # 创建 batch 文件（带题集信息）
            batch_file2 = "batch_results_20260713_130000.jsonl"
            batch_path2 = batch_dir / batch_file2
            with batch_path2.open("w", encoding="utf-8") as f:
                for i in range(5):
                    r = {
                        "success": True,
                        "question": f"问题 {i+1}",
                        "sample": {
                            "question": f"问题 {i+1}",
                            "question_set_id": "qs_test_002",
                            "question_set_name": "测试题集2_全流程问答",
                            "run_id": run_id2,
                            "config_id": config_id,
                        },
                    }
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 更新 manifest
            update_experiment_run(run_id2, {
                "batch_results_file": batch_file2,
            })

            # 测试回填
            result = backfill_manifest_from_batch(run_id2, batch_dir=str(batch_dir))
            assert result == True, "回填应成功"

            manifest2 = load_experiment_run(run_id2)
            assert manifest2["question_set_id"] == "qs_test_002", f"回填后 question_set_id 不正确: {manifest2.get('question_set_id')}"
            assert manifest2["question_set_name"] == "测试题集2_全流程问答", f"回填后 question_set_name 不正确: {manifest2.get('question_set_name')}"
            print(f"  [OK] 回填成功: {manifest2['question_set_name']}")

    print("\n" + "=" * 60)
    print("[OK] 端到端测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_e2e_experiment_pipeline()
