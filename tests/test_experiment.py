"""
运行看板管理测试。

测试内容：
1. 同一 config_profile 创建两次运行，两个 run_id 必须不同
2. 两次运行可以使用不同题集，但关联同一 config_id
3. 基于历史配置另存为新方案，原配置内容不变
4. batch/raw 结果保留 run_id、config_id、question_id
5. 所有测试使用临时目录，不调用真实 API
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, load_config_profile, list_config_profiles,
    create_experiment_run, load_experiment_run, update_experiment_run,
    list_experiment_runs, list_runs_by_config,
    ensure_question_id, build_dify_user_field,
)
from batch_query import query_to_sample, save_batch_results, push_to_raw_dir


def test_config_profile_reuse():
    """测试同一配置方案可用于多次运行。"""
    print("=" * 60)
    print("测试同一配置方案可用于多次运行")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            # 创建配置方案
            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="kb_v1",
                workflow_version="wf_v1",
                changed_variable="chunk_size: 1000 -> 500",
                retrieval_config="hybrid / top_k=5",
            )
            config_id = config["config_id"]
            print(f"[OK] 创建配置方案: {config_id}")

            # 第一次运行
            run1 = create_experiment_run(
                config_id=config_id,
                question_set_source="题目集A",
                question_count=10,
            )
            run_id1 = run1["run_id"]
            print(f"[OK] 创建运行1: {run_id1}")

            # 第二次运行
            run2 = create_experiment_run(
                config_id=config_id,
                question_set_source="题目集B",
                question_count=20,
            )
            run_id2 = run2["run_id"]
            print(f"[OK] 创建运行2: {run_id2}")

            # 验证 run_id 不同
            assert run_id1 != run_id2, f"两次运行应有不同 run_id，但都是 {run_id1}"
            print("[OK] 两次运行有不同 run_id")

            # 验证两次运行关联同一 config_id
            manifest1 = load_experiment_run(run_id1)
            manifest2 = load_experiment_run(run_id2)
            assert manifest1["config_id"] == config_id
            assert manifest2["config_id"] == config_id
            print("[OK] 两次运行关联同一 config_id")

            # 验证配置快照一致
            assert manifest1["config_snapshot"]["config_name"] == "测试配置"
            assert manifest2["config_snapshot"]["config_name"] == "测试配置"
            print("[OK] 配置快照一致")

            # 验证题目来源不同
            assert manifest1["question_set_source"] == "题目集A"
            assert manifest2["question_set_source"] == "题目集B"
            print("[OK] 题目来源不同")

            # 验证 list_runs_by_config
            runs = list_runs_by_config(config_id)
            assert len(runs) == 2, f"应有 2 次运行，实际有 {len(runs)}"
            print("[OK] list_runs_by_config 正确")

    print()


def test_config_profile_save_as():
    """测试基于历史配置另存为新方案。"""
    print("=" * 60)
    print("测试基于历史配置另存为新方案")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir):
            # 创建原始配置
            original = create_config_profile(
                config_name="原始配置",
                knowledge_base_version="kb_v1",
                changed_variable="chunk_size: 1000",
            )
            original_id = original["config_id"]
            print(f"[OK] 创建原始配置: {original_id}")

            # 加载并另存为新方案
            original_config = load_config_profile(original_id)
            new_config = create_config_profile(
                config_name=f"{original_config['config_name']} (副本)",
                knowledge_base_version=original_config["knowledge_base_version"],
                workflow_version=original_config["workflow_version"],
                changed_variable=original_config["changed_variable"],
                retrieval_config=original_config["retrieval_config"],
                notes=original_config["notes"],
            )
            new_id = new_config["config_id"]
            print(f"[OK] 另存为新方案: {new_id}")

            # 验证原配置不变
            original_reloaded = load_config_profile(original_id)
            assert original_reloaded["config_name"] == "原始配置"
            assert original_reloaded["changed_variable"] == "chunk_size: 1000"
            print("[OK] 原配置内容不变")

            # 验证新配置正确
            new_reloaded = load_config_profile(new_id)
            assert new_reloaded["config_name"] == "原始配置 (副本)"
            assert new_reloaded["changed_variable"] == "chunk_size: 1000"
            print("[OK] 新配置内容正确")

            # 验证是两个不同的配置
            assert original_id != new_id
            configs = list_config_profiles()
            assert len(configs) == 2
            print("[OK] 两个配置都存在")

    print()


def test_batch_results_with_run_fields():
    """测试批量结果包含 run_id、config_id、question_id。"""
    print("=" * 60)
    print("测试批量结果包含 run_id、config_id、question_id")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir) / "batch"

        # 构造带运行字段的批量结果
        batch_results = [
            {
                "success": True,
                "question": "测试问题1",
                "sample": {
                    "trace_id": "batch_qa_0_20260713_120000",
                    "question": "测试问题1",
                    "final_answer": "测试回答1",
                    "run_id": "run_20260713_120000_123456_test",
                    "config_id": "cfg_20260713_120000_123456_test",
                    "question_id": "q_abc123",
                    "user_id": "rag_eval:run_20260713_120000_123456_test:q_abc123",
                    "retrieval_results": [],
                },
            },
        ]

        # 保存批量结果
        with patch('batch_query.BATCH_DIR', batch_dir):
            output_path, filename = save_batch_results(batch_results)
            print(f"[OK] 保存批量结果到: {output_path}")

            # 逐行读回并验证
            with output_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 1
            obj = json.loads(lines[0])
            sample = obj.get("sample", {})

            assert sample.get("run_id") == "run_20260713_120000_123456_test"
            assert sample.get("config_id") == "cfg_20260713_120000_123456_test"
            assert sample.get("question_id") == "q_abc123"
            assert sample.get("user_id", "").startswith("rag_eval:")
            print("[OK] 批量结果包含正确的 run_id、config_id、question_id")

    print()


def test_raw_results_with_run_fields():
    """测试 raw JSONL 包含 run_id、config_id、question_id。"""
    print("=" * 60)
    print("测试 raw JSONL 包含 run_id、config_id、question_id")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"

        batch_results = [
            {
                "success": True,
                "question": "测试问题1",
                "sample": {
                    "trace_id": "batch_qa_0_20260713_120000",
                    "question": "测试问题1",
                    "final_answer": "测试回答1",
                    "run_id": "run_20260713_120000_123456_test",
                    "config_id": "cfg_20260713_120000_123456_test",
                    "question_id": "q_abc123",
                    "user_id": "rag_eval:run_20260713_120000_123456_test:q_abc123",
                    "retrieval_results": [],
                },
            },
        ]

        with patch('batch_query.RAW_DIR', raw_dir):
            output_path, filename = push_to_raw_dir(batch_results)
            print(f"[OK] 推送 raw 结果到: {output_path}")

            with output_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 1
            obj = json.loads(lines[0])

            assert obj.get("run_id") == "run_20260713_120000_123456_test"
            assert obj.get("config_id") == "cfg_20260713_120000_123456_test"
            assert obj.get("question_id") == "q_abc123"
            assert obj.get("user_id", "").startswith("rag_eval:")
            print("[OK] raw JSONL 包含正确的 run_id、config_id、question_id")

    print()


def test_query_to_sample_with_run_fields():
    """测试 query_to_sample 保留 run_id、config_id。"""
    print("=" * 60)
    print("测试 query_to_sample 保留 run_id、config_id")
    print("=" * 60)

    dify_result = {
        "answer": "测试回答",
        "conversation_id": "conv_123",
        "message_id": "msg_456",
        "retriever_resources": [],
    }

    sample = query_to_sample(
        question="测试问题",
        dify_result=dify_result,
        index=0,
        timestamp="20260713_120000",
        question_id="q_abc123",
        run_id="run_20260713_120000_123456_test",
        config_id="cfg_20260713_120000_123456_test",
        dify_user="rag_eval:run_20260713_120000_123456_test:q_abc123",
    )

    assert sample["run_id"] == "run_20260713_120000_123456_test"
    assert sample["config_id"] == "cfg_20260713_120000_123456_test"
    assert sample["question_id"] == "q_abc123"
    assert sample["user_id"] == "rag_eval:run_20260713_120000_123456_test:q_abc123"
    print("[OK] query_to_sample 正确保留 run_id、config_id")

    # 测试旧格式兼容
    sample_old = query_to_sample(
        question="旧问题",
        dify_result=dify_result,
        index=0,
        timestamp="20260713_120000",
    )
    assert sample_old["user_id"] == "batch-query"
    assert not sample_old.get("run_id")
    assert not sample_old.get("config_id")
    print("[OK] 旧格式兼容性正确")

    print()


def test_dify_user_field():
    """测试 Dify user 字段格式。"""
    print("=" * 60)
    print("测试 Dify user 字段格式")
    print("=" * 60)

    user = build_dify_user_field("run_20260713_120000_123456_test", "q_abc123")
    assert user == "rag_eval:run_20260713_120000_123456_test:q_abc123"
    print(f"[OK] user 字段格式正确: {user}")

    print()


def test_per_run_judge_stats_not_overwrite():
    """连续运行两个不同 run 后，两个 run 的统计互不覆盖。"""
    print("=" * 60)
    print("测试：per-run Judge 统计互不覆盖")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            # 创建配置方案
            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 创建两个运行
            run1 = create_experiment_run(config_id, question_count=10)
            run2 = create_experiment_run(config_id, question_count=10)
            run1_id = run1["run_id"]
            run2_id = run2["run_id"]

            assert run1_id != run2_id, "两个 run_id 应不同"

            # 模拟第一次 Judge 完成（单 run 批次），写入 run1 的 manifest
            update_experiment_run(run1_id, {
                "judge_duration_seconds": 12.5,
                "judge_duration_scope": "run",
                "judge_llm_call_count": 8,
                "judge_prescreened_count": 2,
                "judge_content_cached_count": 0,
                "judge_concurrency": 3,
                "judge_completed_at": "2026-07-20T10:00:00",
                "judge_mode": "incremental",
                "judge_batch_id": "judge_20260720_100000",
            })

            # 模拟第二次 Judge 完成（单 run 批次），写入 run2 的 manifest
            update_experiment_run(run2_id, {
                "judge_duration_seconds": 25.3,
                "judge_duration_scope": "run",
                "judge_llm_call_count": 15,
                "judge_prescreened_count": 3,
                "judge_content_cached_count": 1,
                "judge_concurrency": 4,
                "judge_completed_at": "2026-07-20T11:00:00",
                "judge_mode": "retry_failed",
                "judge_batch_id": "judge_20260720_110000",
            })

            # 验证 run1 的统计未被覆盖
            r1 = load_experiment_run(run1_id)
            assert r1["judge_duration_seconds"] == 12.5, \
                f"run1 耗时应为 12.5，实际 {r1['judge_duration_seconds']}"
            assert r1["judge_llm_call_count"] == 8, \
                f"run1 LLM 调用应为 8，实际 {r1['judge_llm_call_count']}"
            assert r1["judge_concurrency"] == 3
            assert r1["judge_completed_at"] == "2026-07-20T10:00:00"
            assert r1["judge_mode"] == "incremental"
            assert r1["judge_duration_scope"] == "run"
            print(f"[OK] run1 统计正确: LLM={r1['judge_llm_call_count']}, "
                  f"耗时={r1['judge_duration_seconds']}s, scope=run")

            # 验证 run2 的统计正确
            r2 = load_experiment_run(run2_id)
            assert r2["judge_duration_seconds"] == 25.3
            assert r2["judge_llm_call_count"] == 15
            assert r2["judge_concurrency"] == 4
            assert r2["judge_completed_at"] == "2026-07-20T11:00:00"
            assert r2["judge_mode"] == "retry_failed"
            assert r2["judge_duration_scope"] == "run"
            print(f"[OK] run2 统计正确: LLM={r2['judge_llm_call_count']}, "
                  f"耗时={r2['judge_duration_seconds']}s, scope=run")

            # 验证看板读取：每个 run 读到自己的数据
            r1_again = load_experiment_run(run1_id)
            r2_again = load_experiment_run(run2_id)
            assert r1_again["judge_llm_call_count"] == 8
            assert r2_again["judge_llm_call_count"] == 15
            assert r1_again["judge_batch_id"] != r2_again["judge_batch_id"]
            print("[OK] 看板分别读到各自 run 的统计，互不干扰")

    print()


def test_cross_run_batch_stats():
    """跨 run 批次统计：每个 run 写入自己的 manifest，批次信息一致。"""
    print("=" * 60)
    print("测试：跨 run 批次统计")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="跨 run 测试",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            run1 = create_experiment_run(config_id, question_count=5)
            run2 = create_experiment_run(config_id, question_count=5)

            batch_id = "judge_20260720_120000"
            batch_elapsed = 30.0

            # 同一批次评测两个 run：跨 run 批次语义
            update_experiment_run(run1["run_id"], {
                "judge_batch_duration_seconds": batch_elapsed,
                "judge_duration_scope": "batch",
                "judge_llm_call_count": 4,
                "judge_prescreened_count": 1,
                "judge_content_cached_count": 0,
                "judge_concurrency": 3,
                "judge_completed_at": "2026-07-20T12:01:00",
                "judge_batch_id": batch_id,
                "judge_mode": "incremental",
            })
            update_experiment_run(run2["run_id"], {
                "judge_batch_duration_seconds": batch_elapsed,
                "judge_duration_scope": "batch",
                "judge_llm_call_count": 3,
                "judge_prescreened_count": 1,
                "judge_content_cached_count": 1,
                "judge_concurrency": 3,
                "judge_completed_at": "2026-07-20T12:01:00",
                "judge_batch_id": batch_id,
                "judge_mode": "incremental",
            })

            r1 = load_experiment_run(run1["run_id"])
            r2 = load_experiment_run(run2["run_id"])

            # 同一批次 ID
            assert r1["judge_batch_id"] == batch_id
            assert r2["judge_batch_id"] == batch_id

            # 各自的 LLM 调用数独立
            assert r1["judge_llm_call_count"] == 4
            assert r2["judge_llm_call_count"] == 3

            # 跨 run 批次：不应有 judge_duration_seconds（那是单 run 语义）
            assert "judge_duration_seconds" not in r1, \
                "跨 run 批次不应写入 judge_duration_seconds"
            assert "judge_duration_seconds" not in r2, \
                "跨 run 批次不应写入 judge_duration_seconds"

            # batch 总耗时通过 judge_batch_duration_seconds 保存
            assert r1["judge_batch_duration_seconds"] == batch_elapsed
            assert r2["judge_batch_duration_seconds"] == batch_elapsed

            # scope 标记为 batch
            assert r1["judge_duration_scope"] == "batch"
            assert r2["judge_duration_scope"] == "batch"

            print(f"[OK] run1: LLM={r1['judge_llm_call_count']}, "
                  f"run2: LLM={r2['judge_llm_call_count']}, "
                  f"batch耗时={batch_elapsed}s, scope=batch")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("运行看板管理测试")
    print("=" * 60)
    print()

    test_config_profile_reuse()
    test_config_profile_save_as()
    test_batch_results_with_run_fields()
    test_raw_results_with_run_fields()
    test_query_to_sample_with_run_fields()
    test_dify_user_field()
    test_per_run_judge_stats_not_overwrite()
    test_cross_run_batch_stats()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
