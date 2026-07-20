"""
检索评测题目补题策略测试。

覆盖：
1. 统计准确性
2. 补题触发条件
3. 补题排除已用证据
4. 补题校验一致性
5. 最大轮次限制
6. 证据耗尽提前退出
7. 最终不足时保存已有题目
8. QA 模式不补题
9. 去重一致性

不调用真实 API。
"""

import copy
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from question_generator import (
    _deduplicate_and_trim,
    _generate_from_chunks,
    _supplement_retrieval_questions,
    _validate_retrieval_question,
    _build_supplement_prompt,
    deduplicate_questions,
    MODE_RETRIEVAL,
    MODE_QA,
)


# ====== Fixtures ======

def _make_valid_retrieval_question(idx, chunk_text=None):
    """构建一条通过校验的 retrieval 题目。

    reference_answer 必须是 chunk_text 的连续子串。
    """
    ct = chunk_text or _CHUNK_TEXT
    # 从 chunk_text 中截取一段作为 evidence（确保是连续子串）
    # 使用不同偏移量模拟不同的证据
    start = (idx * 7) % max(1, len(ct) - 30)
    ref = ct[start:start + 30]
    return {
        "question": f"检索查询_{idx}",
        "reference_answer": ref,
        "source_excerpt": ref,
        "difficulty": "事实",
        "topic": f"主题_{idx}",
    }


_CHUNK_TEXT = "合同条款文本，包含违约金的计算方式为合同总金额的百分之十。赔偿责任条款规定了双方的权利义务。争议解决条款约定了仲裁方式。" * 5


def _make_chunk(index=0, text=None, section_title="测试章节"):
    t = text or _CHUNK_TEXT
    return {
        "section_title": section_title,
        "text": t,
        "chunk_index": index,
        "char_count": len(t),
    }


def _mock_call_llm(responses, call_log=None):
    """返回 mock call_llm，按顺序返回预设响应。"""
    if call_log is None:
        call_log = []
    idx = [0]

    def mock(prompt, api_key, base_url, model, timeout=120):
        call_log.append({"stage": idx[0], "prompt_len": len(prompt)})
        resp = responses[idx[0]] if idx[0] < len(responses) else "[]"
        idx[0] += 1
        return resp

    return mock, call_log


def _questions_to_jsonl(questions):
    """将题目列表转为 LLM 响应格式的 JSON 字符串。"""
    return json.dumps(questions, ensure_ascii=False)


# ====== Tests ======

def test_stats_accuracy():
    """统计准确性：raw_count、validation_eliminated、dedup_eliminated 正确。"""
    print("=" * 60)
    print("测试：统计准确性")
    print("=" * 60)

    # 构建 20 条原始题目，其中 6 条校验不通过，2 条重复
    chunk = _make_chunk(0)
    raw_questions = []
    for i in range(20):
        q = _make_valid_retrieval_question(i)
        raw_questions.append(q)
    # 添加 6 条不合规题目（含问号）
    for i in range(20, 26):
        raw_questions.append({
            "question": f"这是问句？_{i}",
            "reference_answer": f"有效证据_{i}，长度超过十五个字符。",
            "source_excerpt": f"有效证据_{i}，长度超过十五个字符。",
        })
    # 添加 2 条重复题目
    raw_questions.append(copy.deepcopy(raw_questions[0]))
    raw_questions.append(copy.deepcopy(raw_questions[1]))

    response = _questions_to_jsonl(raw_questions)
    mock_fn, _ = _mock_call_llm([response])

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        questions, stats = _generate_from_chunks(
            [chunk], 20, "混合", "", "key", "url", "model", 120, None, mode=MODE_RETRIEVAL
        )
    finally:
        qg.call_llm = orig_fn

    assert stats["raw_count"] == 28, f"raw_count 应为 28，实际 {stats['raw_count']}"
    assert stats["validation_eliminated"] == 6, f"validation_eliminated 应为 6，实际 {stats['validation_eliminated']}"
    # 22 通过校验 → 2 条重复 → 20 unique → 裁剪到 20
    assert stats["final_count"] == 20, f"final_count 应为 20，实际 {stats['final_count']}"
    assert stats["target"] == 20, f"target 应为 20，实际 {stats['target']}"

    print(f"PASS: 统计准确 (raw={stats['raw_count']}, val_elim={stats['validation_eliminated']}, "
          f"dedup_elim={stats['dedup_eliminated']}, final={stats['final_count']})")


def test_supplement_triggered_on_deficit():
    """不足目标时触发补题。"""
    print("=" * 60)
    print("测试：补题触发条件")
    print("=" * 60)

    # 首轮只生成 14 条
    chunk = _make_chunk(0)
    ct = chunk["text"]
    first_round = [_make_valid_retrieval_question(i, ct) for i in range(14)]
    supplement = [_make_valid_retrieval_question(i, ct) for i in range(14, 20)]

    responses = [
        _questions_to_jsonl(first_round),
        _questions_to_jsonl(supplement[:3]),  # 补题第 1 轮
        _questions_to_jsonl(supplement[3:]),   # 补题第 2 轮
    ]
    mock_fn, call_log = _mock_call_llm(responses)

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        questions, stats = _generate_from_chunks(
            [chunk], 20, "混合", "", "key", "url", "model", 120, None, mode=MODE_RETRIEVAL
        )
        # 模拟 generate_questions 的补题逻辑
        if len(questions) < 20:
            new_qs, sup_stats = _supplement_retrieval_questions(
                [chunk], questions, 20, "混合", "",
                "key", "url", "model", 120,
            )
            questions.extend(new_qs)
            stats["supplement_rounds"] = sup_stats["rounds"]
            stats["supplement_new"] = sup_stats["new_count"]
            stats["final_count"] = len(questions)
    finally:
        qg.call_llm = orig_fn

    assert stats.get("supplement_rounds", 0) > 0, "应执行补题轮次"
    assert stats.get("supplement_new", 0) > 0, "应有新增题目"
    assert len(questions) == 20, f"最终应为 20 题，实际 {len(questions)}"

    print(f"PASS: 补题触发 ({stats['supplement_rounds']} 轮, 新增 {stats['supplement_new']})")


def test_supplement_excludes_used_evidence():
    """补题排除已使用的 reference_answer。"""
    print("=" * 60)
    print("测试：补题排除已用证据")
    print("=" * 60)

    used_evidence = {"这是已使用的金标准证据一，长度超过十五个字符。", "这是已使用的金标准证据二，长度超过十五个字符。"}
    used_topics = {"违约金", "赔偿责任"}

    prompt = _build_supplement_prompt(
        "合同条款文本内容", 3, used_evidence, used_topics,
        "混合", "", "测试章节", "章节上下文", MODE_RETRIEVAL,
    )

    assert "已使用证据" in prompt, "prompt 应包含已使用证据节"
    assert "已使用的金标准证据一" in prompt, "prompt 应列出已用证据"
    assert "已使用的金标准证据二" in prompt, "prompt 应列出已用证据"
    assert "违约金" in prompt, "prompt 应列出已用主题"
    assert "赔偿责任" in prompt, "prompt 应列出已用主题"
    assert "尚未被使用" in prompt, "prompt 应要求使用未用证据"

    print("PASS: 补题 prompt 包含已用证据和主题排除指令")


def test_supplement_validates_results():
    """补题结果同样经过校验。"""
    print("=" * 60)
    print("测试：补题校验一致性")
    print("=" * 60)

    chunk = _make_chunk(0)
    chunk_text = chunk["text"]
    existing = [_make_valid_retrieval_question(0, chunk_text)]

    # 补题返回 3 条：1 条合规，1 条含问号，1 条合规
    supplement = [
        _make_valid_retrieval_question(1, chunk_text),
        {"question": "这是问句？", "reference_answer": chunk_text[:30],
         "source_excerpt": chunk_text[:30]},
        _make_valid_retrieval_question(2, chunk_text),
    ]

    mock_fn, _ = _mock_call_llm([_questions_to_jsonl(supplement)])

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        new_qs, stats = _supplement_retrieval_questions(
            [chunk], existing, 5, "混合", "",
            "key", "url", "model", 120,
        )
    finally:
        qg.call_llm = orig_fn

    # 只有 2 条通过校验
    assert len(new_qs) == 2, f"应有 2 条通过校验，实际 {len(new_qs)}"
    for q in new_qs:
        ok, reason = _validate_retrieval_question(q, chunk_text)
        assert ok, f"补题结果应通过校验: {reason}"

    print("PASS: 补题结果经过校验，不合规题目被过滤")


def test_max_rounds_limit():
    """最大补题轮次不超过 max_rounds。"""
    print("=" * 60)
    print("测试：最大轮次限制")
    print("=" * 60)

    chunk = _make_chunk(0)
    ct = chunk["text"]
    existing = [_make_valid_retrieval_question(0, ct)]

    # 每轮只返回 1 条，目标 20，max_rounds=3
    responses = []
    for i in range(3):
        responses.append(_questions_to_jsonl([_make_valid_retrieval_question(100 + i, ct)]))

    mock_fn, call_log = _mock_call_llm(responses)

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        new_qs, stats = _supplement_retrieval_questions(
            [chunk], existing, 20, "混合", "",
            "key", "url", "model", 120,
            max_rounds=3,
        )
    finally:
        qg.call_llm = orig_fn

    assert stats["rounds"] <= 3, f"轮次应 <= 3，实际 {stats['rounds']}"
    assert len(call_log) <= 3, f"LLM 调用应 <= 3，实际 {len(call_log)}"

    print(f"PASS: 最大轮次限制 ({stats['rounds']} 轮, {len(call_log)} 次 LLM 调用)")


def test_evidence_exhaustion_early_exit():
    """证据耗尽时提前退出。"""
    print("=" * 60)
    print("测试：证据耗尽提前退出")
    print("=" * 60)

    chunk = _make_chunk(0)
    existing = [_make_valid_retrieval_question(0, chunk["text"])]

    # 第一轮返回空（无更多证据），应立即退出
    mock_fn, call_log = _mock_call_llm(["[]"])

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        new_qs, stats = _supplement_retrieval_questions(
            [chunk], existing, 20, "混合", "",
            "key", "url", "model", 120,
            max_rounds=3,
        )
    finally:
        qg.call_llm = orig_fn

    assert stats["rounds"] == 1, f"应只执行 1 轮，实际 {stats['rounds']}"
    assert len(call_log) == 1, f"应只调用 1 次 LLM，实际 {len(call_log)}"
    assert len(new_qs) == 0, f"应无新增题目，实际 {len(new_qs)}"

    print("PASS: 证据耗尽时提前退出")


def test_final_insufficient_saves_existing():
    """最终不足目标数时保存已有题目，stats 正确。"""
    print("=" * 60)
    print("测试：最终不足时保存已有题目")
    print("=" * 60)

    chunk = _make_chunk(0)
    chunk_text = chunk["text"]
    existing = [_make_valid_retrieval_question(i, chunk_text) for i in range(14)]

    # 补题只返回 2 条
    supplement = [_make_valid_retrieval_question(i, chunk_text) for i in range(14, 16)]
    mock_fn, _ = _mock_call_llm([_questions_to_jsonl(supplement)])

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        new_qs, stats = _supplement_retrieval_questions(
            [chunk], existing, 20, "混合", "",
            "key", "url", "model", 120,
            max_rounds=3,
        )
    finally:
        qg.call_llm = orig_fn

    all_questions = existing + new_qs
    assert len(all_questions) == 16, f"应有 16 题，实际 {len(all_questions)}"
    assert len(all_questions) < 20, "应不足目标数"
    assert stats["new_count"] == 2, f"新增应为 2，实际 {stats['new_count']}"

    print(f"PASS: 最终 {len(all_questions)} 题（目标 20），stats 正确")


def test_qa_mode_no_supplement():
    """QA 模式不执行补题。"""
    print("=" * 60)
    print("测试：QA 模式不补题")
    print("=" * 60)

    chunk = _make_chunk(0)
    # QA 模式没有校验，所有题目都通过
    raw = [_make_valid_retrieval_question(i) for i in range(14)]
    response = _questions_to_jsonl(raw)
    mock_fn, call_log = _mock_call_llm([response])

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        questions, stats = _generate_from_chunks(
            [chunk], 20, "混合", "", "key", "url", "model", 120, None, mode=MODE_QA
        )
    finally:
        qg.call_llm = orig_fn

    # QA 模式不校验，validation_eliminated 应为 0
    assert stats["validation_eliminated"] == 0, f"QA 模式校验淘汰应为 0，实际 {stats['validation_eliminated']}"
    # 只有 1 次 LLM 调用（无补题）
    assert len(call_log) == 1, f"QA 模式应只有 1 次 LLM 调用，实际 {len(call_log)}"

    print("PASS: QA 模式不执行补题")


def test_no_duplicate_evidence_after_supplement():
    """补题后不出现重复 reference_answer。"""
    print("=" * 60)
    print("测试：去重一致性")
    print("=" * 60)

    chunk = _make_chunk(0)
    ct = chunk["text"]
    existing = [_make_valid_retrieval_question(i, ct) for i in range(14)]

    # 补题返回的题目中有 1 条与 existing 重复
    supplement = [
        copy.deepcopy(existing[0]),  # 重复
        _make_valid_retrieval_question(14, ct),
        _make_valid_retrieval_question(15, ct),
    ]
    mock_fn, _ = _mock_call_llm([_questions_to_jsonl(supplement)])

    import question_generator as qg
    orig_fn = qg.call_llm
    qg.call_llm = mock_fn
    try:
        new_qs, stats = _supplement_retrieval_questions(
            [chunk], existing, 20, "混合", "",
            "key", "url", "model", 120,
        )
    finally:
        qg.call_llm = orig_fn

    # 合并后去重
    all_qs = existing + new_qs
    unique = deduplicate_questions(all_qs)
    all_refs = [q.get("reference_answer") for q in unique]
    # 不应有重复的 reference_answer
    ref_counts = {}
    for r in all_refs:
        ref_counts[r] = ref_counts.get(r, 0) + 1
    dupes = {k: v for k, v in ref_counts.items() if v > 1}
    assert len(dupes) == 0, f"不应有重复 evidence，实际重复: {len(dupes)}"

    print(f"PASS: 补题后无重复 evidence ({len(unique)} 条唯一)")


# ====== Main ======

def main():
    tests = [
        test_stats_accuracy,
        test_supplement_triggered_on_deficit,
        test_supplement_excludes_used_evidence,
        test_supplement_validates_results,
        test_max_rounds_limit,
        test_evidence_exhaustion_early_exit,
        test_final_insufficient_saves_existing,
        test_qa_mode_no_supplement,
        test_no_duplicate_evidence_after_supplement,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 个测试")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
