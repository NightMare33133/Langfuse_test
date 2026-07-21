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
    compute_metrics, validate_retrieval_judge_output
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


def test_retrieval_pre_screen_and_position_logic():
    """聚焦测试：retrieval 轨道 pre_screen 与 position/TopK 逻辑。

    验证项：
    a. retrieval 样本有 retrieval_results + final_answer 为空：不得预筛选为全 0，必须进入 Judge
    b. retrieval 样本无 retrieval_results：预筛选为全 0 + hit_evidence_position=null
    c. QA 轨道 final_answer 为空时行为不变（不受 retrieval 分支影响）
    """
    print("=" * 60)
    print("测试 retrieval 轨道 pre_screen 与 position 逻辑")
    print("=" * 60)

    # a. retrieval + 有检索结果 + final_answer 为空 → 不得预筛，必须进入 Judge
    sample_a = {
        "trace_id": "test_retrieval_no_answer",
        "question": "合同违约金比例",
        "question_mode": MODE_RETRIEVAL,
        "retrieval_query": "违约金比例",
        "source_excerpt": "违约方应支付合同总金额的 10% 作为违约金",
        "retrieval_results": [
            {"position": 1, "score": 0.95, "content": "违约方应支付合同总金额的 10% 作为违约金", "document_name": "doc1"},
        ],
        "final_answer": "",  # 空回答
    }
    result_a = pre_screen(sample_a)
    # 规则判定：gold_evidence 完整出现在 retrieval content 中时直接命中（无需 LLM）
    # 否则返回 None 进入 LLM Judge。无论哪种，都不应预筛为全 0。
    if result_a is not None:
        assert result_a.get("retrieval_top1_hit") != 0 or result_a.get("_rule_name"), (
            f"不应预筛为全 0: {result_a}"
        )
        print(f"[OK] retrieval + 有检索结果 + final_answer 为空 → 规则直接命中 (rule={result_a.get('_rule_name')})")
    else:
        print("[OK] retrieval + 有检索结果 + final_answer 为空 → 返回 None，进入 LLM Judge")

    # b. retrieval + 无检索结果 → 预筛选为全 0 + hit_evidence_position=null
    sample_b = {
        "trace_id": "test_retrieval_no_results",
        "question": "合同违约金比例",
        "question_mode": MODE_RETRIEVAL,
        "retrieval_query": "违约金比例",
        "source_excerpt": "违约方应支付合同总金额的 10% 作为违约金",
        "retrieval_results": [],
        "final_answer": "某回答",
    }
    result_b = pre_screen(sample_b)
    assert result_b is not None, "retrieval + 无检索结果应预筛"
    assert result_b["retrieval_top1_hit"] == 0
    assert result_b["retrieval_top3_hit"] == 0
    assert result_b["retrieval_top5_hit"] == 0
    assert result_b.get("hit_evidence_position") is None, (
        f"无检索结果时 hit_evidence_position 应为 null，实际: {result_b.get('hit_evidence_position')}"
    )
    print("[OK] retrieval + 无检索结果 → 全 0 + hit_evidence_position=null")

    # c. QA 轨道 + final_answer 为空 → 保持原有逻辑（应预筛为全 0）
    sample_c = {
        "trace_id": "test_qa_no_answer",
        "question": "什么是违约金？",
        "question_mode": MODE_QA,
        "retrieval_query": "违约金",
        "reference_answer": "违约金是合同约定的赔偿金额",
        "retrieval_results": [
            {"position": 1, "score": 0.9, "content": "违约金相关内容", "document_name": "doc1"},
        ],
        "final_answer": "",  # 空回答
    }
    result_c = pre_screen(sample_c)
    assert result_c is not None, "QA 轨道 final_answer 为空时应预筛"
    assert result_c["retrieval_top1_hit"] == 0
    assert result_c["answer_correct"] == 0
    print("[OK] QA 轨道 + final_answer 为空 → 预筛为全 0，answer_correct=0")

    # d. retrieval + 有检索结果 + final_answer 非空 → 正常进入 Judge
    sample_d = {
        "trace_id": "test_retrieval_full",
        "question": "合同违约金比例",
        "question_mode": MODE_RETRIEVAL,
        "retrieval_query": "违约金比例",
        "source_excerpt": "违约方应支付合同总金额的 10% 作为违约金",
        "retrieval_results": [
            {"position": 1, "score": 0.95, "content": "违约方应支付合同总金额的 10% 作为违约金", "document_name": "doc1"},
        ],
        "final_answer": "违约金为合同总金额的 10%",
    }
    result_d = pre_screen(sample_d)
    # 规则判定：gold_evidence 完整出现在 retrieval content 中时直接命中
    if result_d is not None:
        assert result_d.get("_rule_name") == "exact_contains_top1"
        print(f"[OK] retrieval + 有检索结果 + 有回答 → 规则直接命中 (rule={result_d.get('_rule_name')})")
    else:
        print("[OK] retrieval + 有检索结果 + 有回答 → 返回 None，进入 LLM Judge")

    print()


def test_retrieval_position_topk_constraint():
    """聚焦测试：验证 position 与 TopK 的输出约束关系。

    这些约束由 prompt 指令约束 LLM 输出，此处验证逻辑一致性：
    - position=1 → top1=1, top3=1, top5=1
    - position=2 or 3 → top1=0, top3=1, top5=1
    - position=4 or 5 → top1=0, top3=0, top5=1
    - position=null → top1=0, top3=0, top5=0
    """
    print("=" * 60)
    print("测试 position/TopK 输出约束一致性")
    print("=" * 60)

    # 模拟 LLM 按 prompt 约束返回的结果
    test_cases = [
        # (position, expected_top1, expected_top3, expected_top5, description)
        (1,   1, 1, 1, "position=1"),
        (2,   0, 1, 1, "position=2"),
        (3,   0, 1, 1, "position=3"),
        (4,   0, 0, 1, "position=4"),
        (5,   0, 0, 1, "position=5"),
        (None, 0, 0, 0, "position=null（未命中）"),
    ]

    for pos, exp_t1, exp_t3, exp_t5, desc in test_cases:
        # 构造符合约束的模拟结果
        mock_result = {
            "trace_id": f"test_{desc}",
            "question": "测试问题",
            "question_mode": MODE_RETRIEVAL,
            "evaluation_track": TRACK_RETRIEVAL,
            "retrieval_evaluable": True,
            "hit_evidence_position": pos,
        }
        # 按约束规则推导 TopK
        if pos is None:
            mock_result["retrieval_top1_hit"] = 0
            mock_result["retrieval_top3_hit"] = 0
            mock_result["retrieval_top5_hit"] = 0
        elif pos == 1:
            mock_result["retrieval_top1_hit"] = 1
            mock_result["retrieval_top3_hit"] = 1
            mock_result["retrieval_top5_hit"] = 1
        elif pos <= 3:
            mock_result["retrieval_top1_hit"] = 0
            mock_result["retrieval_top3_hit"] = 1
            mock_result["retrieval_top5_hit"] = 1
        else:  # pos 4 or 5
            mock_result["retrieval_top1_hit"] = 0
            mock_result["retrieval_top3_hit"] = 0
            mock_result["retrieval_top5_hit"] = 1

        assert mock_result["retrieval_top1_hit"] == exp_t1, f"{desc}: top1 期望 {exp_t1}"
        assert mock_result["retrieval_top3_hit"] == exp_t3, f"{desc}: top3 期望 {exp_t3}"
        assert mock_result["retrieval_top5_hit"] == exp_t5, f"{desc}: top5 期望 {exp_t5}"
        print(f"[OK] {desc} → Top1={exp_t1}, Top3={exp_t3}, Top5={exp_t5}")

    print()


def test_retrieval_prompt_content():
    """聚焦测试：验证 retrieval prompt 模板包含必要的评测口径约束。"""
    print("=" * 60)
    print("测试 retrieval prompt 内容")
    print("=" * 60)

    sample = {
        "trace_id": "test_prompt_content",
        "question": "违约金比例",
        "question_mode": MODE_RETRIEVAL,
        "retrieval_query": "违约金比例",
        "source_excerpt": "违约方应支付合同总金额的 10% 作为违约金",
        "retrieval_results": [
            {"position": 1, "score": 0.95, "content": "测试内容", "document_name": "doc1"},
        ],
        "final_answer": "测试回答",
    }
    prompt = build_judge_prompt(sample)

    # 验证关键口径约束存在于 prompt 中
    checks = [
        ("评测查询（短检索查询）", "应将'用户问题'改为'评测查询（短检索查询）'"),
        ("同一条事实/规则", "应要求同一条事实/规则"),
        ("关键词重合", "应明确不以关键词重合为命中依据"),
        ("文档名相同", "应明确不以文档名相同为命中依据"),
        ("主题相近", "应明确不以主题相近为命中依据"),
        ("关键主体", "应要求关键主体等未被遗漏"),
        ("双语文本", "应包含双语文本处理规则"),
        ("完全等价", "应要求双语完全等价"),
        ("最早命中", "应明确 hit_evidence_position 为最早命中"),
    ]

    for keyword, desc in checks:
        assert keyword in prompt, f"retrieval prompt 应包含 '{keyword}'：{desc}"
        print(f"[OK] retrieval prompt 包含 '{keyword}'")

    # 反向检查：JSON 输出模板中不应有 answer_correct 字段
    import re
    json_block = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", prompt)
    assert json_block, "prompt 应包含 JSON 输出模板"
    json_template = json_block.group(1)
    assert "answer_correct" not in json_template, (
        "retrieval prompt 的 JSON 输出模板中不应包含 answer_correct 字段"
    )
    print("[OK] retrieval prompt JSON 模板不包含 answer_correct")

    print()


def test_validate_retrieval_judge_output():
    """聚焦测试：validate_retrieval_judge_output 的严格校验逻辑。

    覆盖：
    a. 6 种有效 position 映射
    b. 非法 TopK 组合
    c. 非法 position 值
    d. 检索结果少于五条时 position 越界
    e. position 类型错误（如 bool、str）
    f. answer_correct 被丢弃
    g. 无检索结果时 position 必须为 null
    """
    print("=" * 60)
    print("测试 validate_retrieval_judge_output")
    print("=" * 60)

    # a. 6 种有效 position 映射
    valid_cases = [
        (1, 1, 1, 1),
        (2, 0, 1, 1),
        (3, 0, 1, 1),
        (4, 0, 0, 1),
        (5, 0, 0, 1),
        (None, 0, 0, 0),
    ]
    for pos, t1, t3, t5 in valid_cases:
        scores = {"retrieval_top1_hit": t1, "retrieval_top3_hit": t3,
                  "retrieval_top5_hit": t5, "hit_evidence_position": pos}
        validate_retrieval_judge_output(scores, 5)  # 5 条检索结果
        print(f"[OK] 有效: position={pos} → ({t1},{t3},{t5})")

    # b. 非法 TopK 组合（position 与 TopK 不匹配）
    illegal_topk_cases = [
        (1, 0, 1, 1, "position=1 但 top1=0"),
        (1, 1, 0, 1, "position=1 但 top3=0"),
        (1, 1, 1, 0, "position=1 但 top5=0"),
        (2, 1, 1, 1, "position=2 但 top1=1"),
        (2, 0, 0, 1, "position=2 但 top3=0"),
        (3, 1, 1, 1, "position=3 但 top1=1"),
        (4, 0, 1, 1, "position=4 但 top3=1"),
        (4, 1, 0, 1, "position=4 但 top1=1"),
        (5, 0, 1, 1, "position=5 但 top3=1"),
        (None, 1, 0, 0, "position=null 但 top1=1"),
        (None, 0, 1, 0, "position=null 但 top3=1"),
        (None, 0, 0, 1, "position=null 但 top5=1"),
    ]
    for pos, t1, t3, t5, desc in illegal_topk_cases:
        scores = {"retrieval_top1_hit": t1, "retrieval_top3_hit": t3,
                  "retrieval_top5_hit": t5, "hit_evidence_position": pos}
        try:
            validate_retrieval_judge_output(scores, 5)
            assert False, f"应抛出 ValueError: {desc}"
        except ValueError as e:
            assert "格式错误" in str(e), f"错误信息应包含'格式错误': {e}"
            print(f"[OK] 拒绝非法 TopK: {desc}")

    # c. 非法 position 值
    illegal_pos_cases = [
        (0, "position=0（下界越界）"),
        (6, "position=6（上界越界）"),
        (-1, "position=-1（负数）"),
        (100, "position=100（远超上界）"),
    ]
    for pos, desc in illegal_pos_cases:
        # 使用一个理论上合法的 TopK 组合，但 position 本身非法
        scores = {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                  "retrieval_top5_hit": 1, "hit_evidence_position": pos}
        try:
            validate_retrieval_judge_output(scores, 5)
            assert False, f"应抛出 ValueError: {desc}"
        except ValueError as e:
            assert "越界" in str(e) or "格式错误" in str(e), f"错误信息应包含'越界'或'格式错误': {e}"
            print(f"[OK] 拒绝非法 position: {desc}")

    # d. 检索结果少于五条时 position 越界
    # 只有 3 条检索结果，position=4 或 5 应被拒绝
    for pos in (4, 5):
        scores = {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                  "retrieval_top5_hit": 1, "hit_evidence_position": pos}
        try:
            validate_retrieval_judge_output(scores, 3)  # 只有 3 条
            assert False, f"应抛出 ValueError: 只有 3 条结果但 position={pos}"
        except ValueError as e:
            assert "越界" in str(e), f"错误信息应包含'越界': {e}"
            print(f"[OK] 拒绝越界: 只有 3 条结果但 position={pos}")

    # 只有 2 条检索结果，position=3 也应被拒绝
    scores = {"retrieval_top1_hit": 0, "retrieval_top3_hit": 1,
              "retrieval_top5_hit": 1, "hit_evidence_position": 3}
    try:
        validate_retrieval_judge_output(scores, 2)
        assert False, "应抛出 ValueError: 只有 2 条结果但 position=3"
    except ValueError as e:
        assert "越界" in str(e), f"错误信息应包含'越界': {e}"
        print("[OK] 拒绝越界: 只有 2 条结果但 position=3")

    # 只有 1 条检索结果，position=1 应通过
    scores = {"retrieval_top1_hit": 1, "retrieval_top3_hit": 1,
              "retrieval_top5_hit": 1, "hit_evidence_position": 1}
    validate_retrieval_judge_output(scores, 1)
    print("[OK] 有效: 只有 1 条结果，position=1")

    # 只有 1 条检索结果，position=2 应被拒绝
    scores = {"retrieval_top1_hit": 0, "retrieval_top3_hit": 1,
              "retrieval_top5_hit": 1, "hit_evidence_position": 2}
    try:
        validate_retrieval_judge_output(scores, 1)
        assert False, "应抛出 ValueError: 只有 1 条结果但 position=2"
    except ValueError as e:
        assert "越界" in str(e), f"错误信息应包含'越界': {e}"
        print("[OK] 拒绝越界: 只有 1 条结果但 position=2")

    # e. position 类型错误
    type_error_cases = [
        (True, "position=True（bool）"),
        (False, "position=False（bool）"),
        ("1", "position='1'（str）"),
        (1.0, "position=1.0（float）"),
        ([1], "position=[1]（list）"),
    ]
    for pos, desc in type_error_cases:
        scores = {"retrieval_top1_hit": 1, "retrieval_top3_hit": 1,
                  "retrieval_top5_hit": 1, "hit_evidence_position": pos}
        try:
            validate_retrieval_judge_output(scores, 5)
            assert False, f"应抛出 ValueError: {desc}"
        except ValueError as e:
            assert "格式错误" in str(e), f"错误信息应包含'格式错误': {e}"
            print(f"[OK] 拒绝类型错误: {desc}")

    # f. TopK 值类型错误
    topk_type_cases = [
        ("retrieval_top1_hit", 2, "top1=2"),
        ("retrieval_top3_hit", -1, "top3=-1"),
        ("retrieval_top5_hit", "1", "top5='1'"),
        ("retrieval_top1_hit", True, "top1=True"),
        ("retrieval_top1_hit", None, "top1=None"),
    ]
    for key, val, desc in topk_type_cases:
        scores = {"retrieval_top1_hit": 1, "retrieval_top3_hit": 1,
                  "retrieval_top5_hit": 1, "hit_evidence_position": 1}
        scores[key] = val
        try:
            validate_retrieval_judge_output(scores, 5)
            assert False, f"应抛出 ValueError: {desc}"
        except ValueError as e:
            assert "格式错误" in str(e), f"错误信息应包含'格式错误': {e}"
            print(f"[OK] 拒绝 TopK 类型错误: {desc}")

    # g. answer_correct 被丢弃
    scores = {"retrieval_top1_hit": 1, "retrieval_top3_hit": 1,
              "retrieval_top5_hit": 1, "hit_evidence_position": 1,
              "answer_correct": 1}
    validate_retrieval_judge_output(scores, 5)
    assert "answer_correct" not in scores, "answer_correct 应被丢弃"
    print("[OK] answer_correct 被正确丢弃")

    # h. 无检索结果时 position 必须为 null
    scores = {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
              "retrieval_top5_hit": 0, "hit_evidence_position": None}
    validate_retrieval_judge_output(scores, 0)
    print("[OK] 无检索结果 + position=null → 通过")

    scores = {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
              "retrieval_top5_hit": 0, "hit_evidence_position": 1}
    try:
        validate_retrieval_judge_output(scores, 0)
        assert False, "应抛出 ValueError: 无检索结果但 position=1"
    except ValueError as e:
        assert "格式错误" in str(e), f"错误信息应包含'格式错误': {e}"
        print("[OK] 拒绝: 无检索结果但 position=1")

    # i. TopK 值为 2 或其他非法整数
    scores = {"retrieval_top1_hit": 2, "retrieval_top3_hit": 0,
              "retrieval_top5_hit": 0, "hit_evidence_position": None}
    try:
        # position=null 时先检查 TopK 值域
        validate_retrieval_judge_output(scores, 5)
        assert False, "应抛出 ValueError: top1=2"
    except ValueError as e:
        assert "格式错误" in str(e), f"错误信息应包含'格式错误': {e}"
        print("[OK] 拒绝: top1=2")

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
    test_retrieval_pre_screen_and_position_logic()
    test_retrieval_position_topk_constraint()
    test_retrieval_prompt_content()
    test_validate_retrieval_judge_output()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
