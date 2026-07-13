"""
验证脚本：测试评测轨道（evaluation_track）的分类和处理逻辑。

测试内容：
a. retrieval + 有 source_excerpt：可评测检索题
b. strict_qa + 有 reference_answer：严格问答
c. grounded_qa + 无参考答案：合理性问答
d. retrieval + 缺少金标准：不可评测

不调用真实 LLM、Dify 或 Langfuse API。
"""

import json
from judge import (
    classify_evaluation_track, get_gold_evidence,
    TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE,
    pre_screen, judge_sample, build_judge_prompt,
    compute_metrics
)
from question_generator import MODE_RETRIEVAL, MODE_QA


def create_test_sample(mode, has_source_excerpt=False, has_reference_answer=False):
    """创建测试样本。"""
    sample = {
        "trace_id": f"test_{mode}_{has_source_excerpt}_{has_reference_answer}",
        "question": f"测试问题（mode={mode}）",
        "question_mode": mode,
        "retrieval_query": "测试查询",
        "retrieval_results": [
            {"position": 1, "score": 0.95, "content": "测试检索结果1", "document_name": "doc1"},
            {"position": 2, "score": 0.85, "content": "测试检索结果2", "document_name": "doc2"},
        ],
        "final_answer": "测试回答",
    }
    if has_source_excerpt:
        sample["source_excerpt"] = "测试来源摘录"
    if has_reference_answer:
        sample["reference_answer"] = "测试参考答案"
    return sample


def test_track_classification():
    """测试评测轨道分类。"""
    print("=" * 60)
    print("测试评测轨道分类")
    print("=" * 60)

    # a. retrieval + 有 source_excerpt：可评测检索题
    sample_a = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=True, has_reference_answer=False)
    track_a = classify_evaluation_track(sample_a)
    assert track_a == TRACK_RETRIEVAL, f"期望 {TRACK_RETRIEVAL}，实际 {track_a}"
    print(f"[OK] retrieval + source_excerpt -> {track_a}")

    # b. strict_qa + 有 reference_answer：严格问答
    sample_b = create_test_sample(MODE_QA, has_source_excerpt=False, has_reference_answer=True)
    track_b = classify_evaluation_track(sample_b)
    assert track_b == TRACK_STRICT_QA, f"期望 {TRACK_STRICT_QA}，实际 {track_b}"
    print(f"[OK] qa + reference_answer -> {track_b}")

    # c. grounded_qa + 无参考答案：合理性问答
    sample_c = create_test_sample(MODE_QA, has_source_excerpt=False, has_reference_answer=False)
    track_c = classify_evaluation_track(sample_c)
    assert track_c == TRACK_GROUNDED_QA, f"期望 {TRACK_GROUNDED_QA}，实际 {track_c}"
    print(f"[OK] qa + 无参考答案 -> {track_c}")

    # d. retrieval + 缺少金标准：不可评测
    sample_d = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=False, has_reference_answer=False)
    track_d = classify_evaluation_track(sample_d)
    assert track_d == TRACK_NOT_EVALUABLE, f"期望 {TRACK_NOT_EVALUABLE}，实际 {track_d}"
    print(f"[OK] retrieval + 缺少金标准 -> {track_d}")

    # e. 旧版/未知模式 + 有 reference_answer：严格问答
    sample_e = create_test_sample("", has_source_excerpt=False, has_reference_answer=True)
    track_e = classify_evaluation_track(sample_e)
    assert track_e == TRACK_STRICT_QA, f"期望 {TRACK_STRICT_QA}，实际 {track_e}"
    print(f"[OK] 旧版 + reference_answer -> {track_e}")

    # f. 旧版/未知模式 + 无参考答案：合理性问答
    sample_f = create_test_sample("", has_source_excerpt=False, has_reference_answer=False)
    track_f = classify_evaluation_track(sample_f)
    assert track_f == TRACK_GROUNDED_QA, f"期望 {TRACK_GROUNDED_QA}，实际 {track_f}"
    print(f"[OK] 旧版 + 无参考答案 -> {track_f}")

    # g. retrieval + 有 reference_answer（无 source_excerpt）：可评测检索题（用 reference_answer 作为次级金标准）
    sample_g = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=False, has_reference_answer=True)
    track_g = classify_evaluation_track(sample_g)
    assert track_g == TRACK_RETRIEVAL, f"期望 {TRACK_RETRIEVAL}，实际 {track_g}"
    print(f"[OK] retrieval + reference_answer -> {track_g}")

    print()


def test_gold_evidence():
    """测试金标准证据获取。"""
    print("=" * 60)
    print("测试金标准证据获取")
    print("=" * 60)

    # 优先 source_excerpt
    sample_a = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=True, has_reference_answer=True)
    evidence_a = get_gold_evidence(sample_a)
    assert evidence_a == "测试来源摘录", f"期望 '测试来源摘录'，实际 '{evidence_a}'"
    print(f"[OK] 优先 source_excerpt: '{evidence_a}'")

    # 次级 reference_answer
    sample_b = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=False, has_reference_answer=True)
    evidence_b = get_gold_evidence(sample_b)
    assert evidence_b == "测试参考答案", f"期望 '测试参考答案'，实际 '{evidence_b}'"
    print(f"[OK] 次级 reference_answer: '{evidence_b}'")

    # 都没有
    sample_c = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=False, has_reference_answer=False)
    evidence_c = get_gold_evidence(sample_c)
    assert evidence_c == "", f"期望空字符串，实际 '{evidence_c}'"
    print(f"[OK] 都没有: '{evidence_c}'")

    print()


def test_pre_screen():
    """测试规则预筛选。"""
    print("=" * 60)
    print("测试规则预筛选")
    print("=" * 60)

    # 不可评测的检索题
    sample_d = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=False, has_reference_answer=False)
    result_d = pre_screen(sample_d)
    assert result_d is not None, "不可评测样本应被规则预筛选"
    assert result_d.get("retrieval_evaluable") == False, "应标记为不可评测"
    print(f"[OK] 不可评测检索题被规则预筛选: {result_d.get('reason', '')}")

    # 可评测的检索题
    sample_a = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=True)
    result_a = pre_screen(sample_a)
    assert result_a is None, "可评测样本不应被规则预筛选"
    print(f"[OK] 可评测检索题不被规则预筛选")

    print()


def test_prompt_selection():
    """测试 prompt 选择。"""
    print("=" * 60)
    print("测试 prompt 选择")
    print("=" * 60)

    # 检索评测题
    sample_a = create_test_sample(MODE_RETRIEVAL, has_source_excerpt=True)
    prompt_a = build_judge_prompt(sample_a)
    assert "金标准证据" in prompt_a, "检索评测 prompt 应包含金标准证据"
    assert "retrieval_top1_hit" in prompt_a, "检索评测 prompt 应包含 retrieval_top1_hit"
    print(f"[OK] 检索评测题使用专用 prompt")

    # 严格问答题
    sample_b = create_test_sample(MODE_QA, has_reference_answer=True)
    prompt_b = build_judge_prompt(sample_b)
    assert "参考答案" in prompt_b, "严格问答 prompt 应包含参考答案"
    assert "answer_correct" in prompt_b, "严格问答 prompt 应包含 answer_correct"
    print(f"[OK] 严格问答题使用含参考答案 prompt")

    # 合理性问答题
    sample_c = create_test_sample(MODE_QA)
    prompt_c = build_judge_prompt(sample_c)
    assert "参考答案" not in prompt_c, "合理性问答 prompt 不应包含参考答案"
    assert "answer_correct" in prompt_c, "合理性问答 prompt 应包含 answer_correct"
    print(f"[OK] 合理性问答题使用无参考答案 prompt")

    print()


def test_metrics_computation():
    """测试指标计算。"""
    print("=" * 60)
    print("测试指标计算")
    print("=" * 60)

    # 模拟评测结果
    results = [
        {
            "trace_id": "test_1",
            "question": "检索题1",
            "evaluation_track": TRACK_RETRIEVAL,
            "retrieval_evaluable": True,
            "retrieval_top1_hit": 1,
            "retrieval_top3_hit": 1,
            "retrieval_top5_hit": 1,
            "answer_correct": 1,
        },
        {
            "trace_id": "test_2",
            "question": "检索题2",
            "evaluation_track": TRACK_RETRIEVAL,
            "retrieval_evaluable": True,
            "retrieval_top1_hit": 0,
            "retrieval_top3_hit": 1,
            "retrieval_top5_hit": 1,
            "answer_correct": 0,
        },
        {
            "trace_id": "test_3",
            "question": "严格问答题1",
            "evaluation_track": TRACK_STRICT_QA,
            "has_reference": True,
            "retrieval_top1_hit": 1,
            "retrieval_top3_hit": 1,
            "retrieval_top5_hit": 1,
            "answer_correct": 1,
        },
        {
            "trace_id": "test_4",
            "question": "合理性问答题1",
            "evaluation_track": TRACK_GROUNDED_QA,
            "has_reference": False,
            "retrieval_top1_hit": 0,
            "retrieval_top3_hit": 0,
            "retrieval_top5_hit": 1,
            "answer_correct": 1,
        },
        {
            "trace_id": "test_5",
            "question": "不可评测题1",
            "evaluation_track": TRACK_NOT_EVALUABLE,
            "retrieval_evaluable": False,
            "retrieval_top1_hit": 0,
            "retrieval_top3_hit": 0,
            "retrieval_top5_hit": 0,
            "answer_correct": 0,
            "not_evaluable_reason": "缺少金标准证据",
        },
    ]

    metrics = compute_metrics(results)

    # 验证分组指标
    assert metrics["retrieval_track_count"] == 2, f"期望 2 条可评测检索题，实际 {metrics['retrieval_track_count']}"
    assert metrics["strict_qa_track_count"] == 1, f"期望 1 条严格问答，实际 {metrics['strict_qa_track_count']}"
    assert metrics["grounded_qa_track_count"] == 1, f"期望 1 条合理性问答，实际 {metrics['grounded_qa_track_count']}"
    assert metrics["retrieval_not_evaluable_count"] == 1, f"期望 1 条不可评测，实际 {metrics['retrieval_not_evaluable_count']}"

    # 验证检索命中率
    assert metrics["retrieval_top1_hit_rate"] == 0.5, f"期望 Top1 Hit 50%，实际 {metrics['retrieval_top1_hit_rate']}"
    assert metrics["retrieval_top3_hit_rate"] == 1.0, f"期望 Top3 Hit 100%，实际 {metrics['retrieval_top3_hit_rate']}"
    assert metrics["retrieval_top5_hit_rate"] == 1.0, f"期望 Top5 Hit 100%，实际 {metrics['retrieval_top5_hit_rate']}"

    # 验证严格问答正确率
    assert metrics["strict_qa_answer_rate"] == 1.0, f"期望严格问答正确率 100%，实际 {metrics['strict_qa_answer_rate']}"

    # 验证合理性问答正确率
    assert metrics["grounded_qa_answer_rate"] == 1.0, f"期望合理性问答正确率 100%，实际 {metrics['grounded_qa_answer_rate']}"

    print(f"[OK] 检索评测指标: Top1={metrics['retrieval_top1_hit_rate']:.0%}, Top3={metrics['retrieval_top3_hit_rate']:.0%}, Top5={metrics['retrieval_top5_hit_rate']:.0%}")
    print(f"[OK] 严格问答指标: Answer={metrics['strict_qa_answer_rate']:.0%}")
    print(f"[OK] 合理性问答指标: Answer={metrics['grounded_qa_answer_rate']:.0%}")
    print(f"[OK] 不可评测样本数: {metrics['retrieval_not_evaluable_count']}")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("评测轨道（evaluation_track）验证测试")
    print("=" * 60)
    print()

    test_track_classification()
    test_gold_evidence()
    test_pre_screen()
    test_prompt_selection()
    test_metrics_computation()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
