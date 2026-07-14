"""
运行看板看板测试。

测试内容：
1. Langfuse user_id 可正确解析 run_id/question_id
2. parser 回填 run/config/question_set 元数据
3. 一个配置下两个不同题集运行不会混合指标
4. 一个 run 的 Judge 结果能正确汇总到该运行
5. Legacy 数据仍可显示

不调用真实 Dify、Langfuse、LLM API。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, create_experiment_run,
    load_config_profile, load_experiment_run,
    list_config_profiles, list_runs_by_config,
    get_run_status, get_judge_metrics_by_run,
    parse_rag_eval_user_id,
)
from parser import (
    parse_rag_eval_user_id as parser_parse_rag_eval_user_id,
    backfill_reference_answers,
    extract_experiment_metadata_from_samples,
)


def test_parse_rag_eval_user_id():
    """测试解析 rag_eval:run_id:question_id 格式。"""
    print("=" * 60)
    print("测试解析 rag_eval user_id")
    print("=" * 60)

    # 正常格式
    result = parse_rag_eval_user_id("rag_eval:run_20260713_120000_测试:q_abc123")
    assert result["run_id"] == "run_20260713_120000_测试", f"run_id 不正确: {result.get('run_id')}"
    assert result["question_id"] == "q_abc123", f"question_id 不正确: {result.get('question_id')}"
    print("[OK] 正常格式解析正确")

    # 空值
    result = parse_rag_eval_user_id("")
    assert result == {}, f"空值应返回空 dict: {result}"
    print("[OK] 空值处理正确")

    # 非 rag_eval 格式
    result = parse_rag_eval_user_id("batch-query")
    assert result == {}, f"非 rag_eval 格式应返回空 dict: {result}"
    print("[OK] 非 rag_eval 格式处理正确")

    # None
    result = parse_rag_eval_user_id(None)
    assert result == {}, f"None 应返回空 dict: {result}"
    print("[OK] None 处理正确")

    # parser 模块中的函数一致
    result2 = parser_parse_rag_eval_user_id("rag_eval:run_123:q_456")
    assert result2["run_id"] == "run_123"
    assert result2["question_id"] == "q_456"
    print("[OK] parser 模块函数一致")

    print()


def test_extract_experiment_metadata():
    """测试从样本中提取实验元数据统计。"""
    print("=" * 60)
    print("测试提取实验元数据统计")
    print("=" * 60)

    samples = [
        {"user_id": "rag_eval:run_1:q_1"},
        {"user_id": "rag_eval:run_1:q_2"},
        {"user_id": "rag_eval:run_2:q_3"},
        {"user_id": "batch-query"},
        {"user_id": ""},
    ]

    stats = extract_experiment_metadata_from_samples(samples)
    assert stats["total_samples"] == 5
    assert stats["identified_count"] == 3
    assert stats["unidentified_count"] == 2
    assert stats["identified_runs"]["run_1"] == 2
    assert stats["identified_runs"]["run_2"] == 1
    print("[OK] 元数据统计正确")

    print()


def test_backfill_with_user_id():
    """测试从 user_id 解析并回填元数据。"""
    print("=" * 60)
    print("测试从 user_id 解析并回填元数据")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir) / "questions"

        # 创建题目文件
        questions_dir.mkdir(parents=True)
        questions = [
            {
                "question": "测试问题1",
                "reference_answer": "答案1",
                "question_id": "q_abc123",
                "question_set_id": "qs_001",
                "question_set_name": "测试题集",
            },
        ]
        q_path = questions_dir / "test.jsonl"
        with q_path.open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # 样本（从 Langfuse 解析，有 user_id）
        samples = [
            {
                "trace_id": "trace_1",
                "question": "测试问题1",
                "user_id": "rag_eval:run_123:q_abc123",
                "final_answer": "回答1",
            },
        ]

        # 直接传 questions_dir 参数
        samples, stats = backfill_reference_answers(samples, questions_dir=str(questions_dir))

        # 验证回填
        assert samples[0]["run_id"] == "run_123", f"run_id 不正确: {samples[0].get('run_id')}"
        assert samples[0]["question_id"] == "q_abc123", f"question_id 不正确: {samples[0].get('question_id')}"
        assert samples[0]["reference_answer"] == "答案1", f"reference_answer 不正确: {samples[0].get('reference_answer')}"
        assert samples[0]["question_set_id"] == "qs_001", f"question_set_id 不正确: {samples[0].get('question_set_id')}"
        assert samples[0]["question_set_name"] == "测试题集", f"question_set_name 不正确: {samples[0].get('question_set_name')}"
        assert stats["run_id_parsed"] == 1
        print("[OK] 从 user_id 解析并回填元数据正确")

    print()


def test_config_runs_not_mixed():
    """测试一个配置下两个不同题集运行不会混合指标。"""
    print("=" * 60)
    print("测试不同题集运行不会混合指标")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            # 创建配置
            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="kb_v1",
            )
            config_id = config["config_id"]

            # 创建两个运行
            run1 = create_experiment_run(config_id, "题集A", 10)
            run2 = create_experiment_run(config_id, "题集B", 20)

            # 验证两个运行关联同一配置
            runs = list_runs_by_config(config_id)
            assert len(runs) == 2, f"应有 2 次运行，实际有 {len(runs)}"

            # 验证运行有不同的 run_id
            run_ids = [r["run_id"] for r in runs]
            assert run_ids[0] != run_ids[1], "两次运行应有不同 run_id"

            # 验证题集信息正确
            for run in runs:
                if run["question_count"] == 10:
                    assert run.get("question_set_source") == "题集A"
                else:
                    assert run.get("question_set_source") == "题集B"

            print("[OK] 不同题集运行不会混合")

    print()


def test_judge_metrics_by_run():
    """测试按 run_id 过滤 Judge 结果。"""
    print("=" * 60)
    print("测试按 run_id 过滤 Judge 结果")
    print("=" * 60)

    # 模拟 Judge 结果
    judge_results = [
        {"trace_id": "batch_qa_0_20260713", "run_id": "run_1", "evaluation_track": "retrieval",
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "answer_correct": 1},
        {"trace_id": "batch_qa_1_20260713", "run_id": "run_1", "evaluation_track": "retrieval",
         "retrieval_top1_hit": 0, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "answer_correct": 0},
        {"trace_id": "batch_qa_2_20260713", "run_id": "run_2", "evaluation_track": "strict_qa",
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "answer_correct": 1},
    ]

    # 过滤 run_1 的结果
    metrics = get_judge_metrics_by_run(judge_results, "run_1")
    assert metrics is not None, "应返回指标"
    assert metrics["evaluated"] == 2, f"应有 2 条结果，实际有 {metrics['evaluated']}"
    assert metrics["retrieval_count"] == 2, f"应有 2 条检索结果，实际有 {metrics['retrieval_count']}"
    assert metrics["strict_qa_count"] == 0, f"应有 0 条严格问答，实际有 {metrics['strict_qa_count']}"
    print("[OK] run_1 指标正确")

    # 过滤 run_2 的结果
    metrics2 = get_judge_metrics_by_run(judge_results, "run_2")
    assert metrics2 is not None
    assert metrics2["evaluated"] == 1
    assert metrics2["strict_qa_count"] == 1
    print("[OK] run_2 指标正确")

    # 不同 run 的指标不会混合
    assert metrics["evaluated"] != metrics2["evaluated"], "不同 run 的指标不应混合"
    print("[OK] 不同 run 的指标不会混合")

    print()


def test_legacy_data_display():
    """测试 Legacy 数据仍可显示。"""
    print("=" * 60)
    print("测试 Legacy 数据仍可显示")
    print("=" * 60)

    # Legacy 样本（没有 run_id）
    samples = [
        {"user_id": "batch-query", "question": "旧问题1"},
        {"user_id": "", "question": "旧问题2"},
        {"user_id": None, "question": "旧问题3"},
    ]

    stats = extract_experiment_metadata_from_samples(samples)
    assert stats["total_samples"] == 3
    assert stats["identified_count"] == 0
    assert stats["unidentified_count"] == 3
    assert stats["identified_runs"] == {}
    print("[OK] Legacy 数据统计正确")

    # Legacy Judge 结果
    legacy_results = [
        {"trace_id": "old_trace_1", "evaluation_track": "strict_qa", "answer_correct": 1},
        {"trace_id": "old_trace_2", "evaluation_track": "grounded_qa", "answer_correct": 0},
    ]

    metrics = get_judge_metrics_by_run(legacy_results, "nonexistent_run")
    # Legacy 数据没有 run_id 匹配，应返回 None
    assert metrics is None, "没有匹配的 run 应返回 None"
    print("[OK] Legacy Judge 结果处理正确")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("运行看板看板测试")
    print("=" * 60)
    print()

    test_parse_rag_eval_user_id()
    test_extract_experiment_metadata()
    test_backfill_with_user_id()
    test_config_runs_not_mixed()
    test_judge_metrics_by_run()
    test_legacy_data_display()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
