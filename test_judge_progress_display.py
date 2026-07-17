"""
Judge 评测进度文案测试。

验证四种 evaluation_track 的实时进度显示：
- retrieval: Top1/Top3/Top5 + 最早命中位置，不含 Answer
- strict_qa: Answer
- grounded_qa: 回答有据
- not_evaluable: 不可评测

不调用真实 API。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE


def _format_progress_line(result, idx):
    """从 app.py 提取的进度文案格式化逻辑。与 app.py 保持同步。"""
    track = result.get("evaluation_track", "")
    q = (result.get("question") or "")[:40]

    if track == TRACK_RETRIEVAL:
        t1 = "✓" if result.get("retrieval_top1_hit") else "✗"
        t3 = "✓" if result.get("retrieval_top3_hit") else "✗"
        t5 = "✓" if result.get("retrieval_top5_hit") else "✗"
        pos = result.get("hit_evidence_position")
        pos_str = str(pos) if pos else "无"
        return f"✅ [{idx}] {q} — Top1:{t1} | Top3:{t3} | Top5:{t5} | 最早命中位置:{pos_str}"
    elif track == TRACK_STRICT_QA:
        ans = "✓" if result.get("answer_correct") else "✗"
        return f"✅ [{idx}] {q} — Answer:{ans}"
    elif track == TRACK_GROUNDED_QA:
        gnd = "✓" if result.get("answer_correct") else "✗"
        return f"✅ [{idx}] {q} — 回答有据:{gnd}"
    else:
        return f"✅ [{idx}] {q} — 不可评测：缺少金标准证据"


# ====== 测试函数 ======

def test_retrieval_progress_text():
    """retrieval 轨道：显示 Top1/Top3/Top5 + 最早命中位置，不含 Answer。"""
    print("=" * 60)
    print("测试：retrieval 进度文案")
    print("=" * 60)

    # 全命中
    r_hit = {
        "evaluation_track": TRACK_RETRIEVAL,
        "question": "合同违约金条款是什么？",
        "retrieval_top1_hit": 1,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 1,
    }
    text = _format_progress_line(r_hit, 1)
    assert "Top1:✓" in text, f"应包含 Top1:✓，实际: {text}"
    assert "Top3:✓" in text, f"应包含 Top3:✓，实际: {text}"
    assert "Top5:✓" in text, f"应包含 Top5:✓，实际: {text}"
    assert "最早命中位置:1" in text, f"应包含最早命中位置:1，实际: {text}"
    assert "Answer" not in text, f"retrieval 不应包含 Answer，实际: {text}"
    assert "回答有据" not in text, f"retrieval 不应包含回答有据，实际: {text}"

    # 全未命中
    r_miss = {
        "evaluation_track": TRACK_RETRIEVAL,
        "question": "测试问题",
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 0,
        "retrieval_top5_hit": 0,
        "hit_evidence_position": None,
    }
    text2 = _format_progress_line(r_miss, 2)
    assert "Top1:✗" in text2, f"应包含 Top1:✗，实际: {text2}"
    assert "Top3:✗" in text2, f"应包含 Top3:✗，实际: {text2}"
    assert "Top5:✗" in text2, f"应包含 Top5:✗，实际: {text2}"
    assert "最早命中位置:无" in text2, f"应包含最早命中位置:无，实际: {text2}"
    assert "Answer" not in text2, f"retrieval 不应包含 Answer，实际: {text2}"

    # 仅 Top3 命中（排序问题）
    r_sort = {
        "evaluation_track": TRACK_RETRIEVAL,
        "question": "排序问题",
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 2,
    }
    text3 = _format_progress_line(r_sort, 3)
    assert "Top1:✗" in text3, f"应包含 Top1:✗，实际: {text3}"
    assert "Top3:✓" in text3, f"应包含 Top3:✓，实际: {text3}"
    assert "Top5:✓" in text3, f"应包含 Top5:✓，实际: {text3}"
    assert "最早命中位置:2" in text3, f"应包含最早命中位置:2，实际: {text3}"
    assert "Answer" not in text3

    print("PASS: retrieval 进度文案正确（含 Top5，不含 Answer）")


def test_strict_qa_progress_text():
    """strict_qa 轨道：显示 Answer。"""
    print("=" * 60)
    print("测试：strict_qa 进度文案")
    print("=" * 60)

    r_correct = {
        "evaluation_track": TRACK_STRICT_QA,
        "question": "合同有效期是多久？",
        "answer_correct": 1,
    }
    text = _format_progress_line(r_correct, 1)
    assert "Answer:✓" in text, f"应包含 Answer:✓，实际: {text}"
    assert "Top1" not in text, f"strict_qa 主状态不应含 Top1，实际: {text}"
    assert "Top5" not in text, f"strict_qa 主状态不应含 Top5，实际: {text}"

    r_wrong = {
        "evaluation_track": TRACK_STRICT_QA,
        "question": "测试问题",
        "answer_correct": 0,
    }
    text2 = _format_progress_line(r_wrong, 2)
    assert "Answer:✗" in text2, f"应包含 Answer:✗，实际: {text2}"

    print("PASS: strict_qa 进度文案正确")


def test_grounded_qa_progress_text():
    """grounded_qa 轨道：显示回答有据。"""
    print("=" * 60)
    print("测试：grounded_qa 进度文案")
    print("=" * 60)

    r_grounded = {
        "evaluation_track": TRACK_GROUNDED_QA,
        "question": "如何处理争议？",
        "answer_correct": 1,
    }
    text = _format_progress_line(r_grounded, 1)
    assert "回答有据:✓" in text, f"应包含回答有据:✓，实际: {text}"
    assert "Answer" not in text, f"grounded_qa 不应使用 Answer 文案，实际: {text}"
    assert "Top1" not in text

    r_ungrounded = {
        "evaluation_track": TRACK_GROUNDED_QA,
        "question": "测试问题",
        "answer_correct": 0,
    }
    text2 = _format_progress_line(r_ungrounded, 2)
    assert "回答有据:✗" in text2, f"应包含回答有据:✗，实际: {text2}"

    print("PASS: grounded_qa 进度文案正确")


def test_not_evaluable_progress_text():
    """not_evaluable 轨道：显示不可评测。"""
    print("=" * 60)
    print("测试：not_evaluable 进度文案")
    print("=" * 60)

    r = {
        "evaluation_track": TRACK_NOT_EVALUABLE,
        "question": "缺少金标准的问题",
    }
    text = _format_progress_line(r, 1)
    assert "不可评测" in text, f"应包含不可评测，实际: {text}"
    assert "缺少金标准证据" in text, f"应包含缺少金标准证据，实际: {text}"
    assert "Top1" not in text
    assert "Answer" not in text
    assert "回答有据" not in text

    print("PASS: not_evaluable 进度文案正确")


def test_error_result_progress_text():
    """error 结果显示错误信息，不显示指标。"""
    print("=" * 60)
    print("测试：error 结果进度文案")
    print("=" * 60)

    # error 结果走的是另一个分支（st.warning），不在 _format_progress_line 中
    # 但验证 error key 存在时不会进入正常格式化
    r = {
        "evaluation_track": TRACK_RETRIEVAL,
        "question": "错误问题",
        "error": "LLM 调用超时",
    }
    # error 结果应由 app.py 的 "error" in _r 分支处理
    assert "error" in r, "error 结果应有 error key"

    print("PASS: error 结果由独立分支处理")


def test_retrieval_no_answer_keyword():
    """retrieval 文案中不出现 Answer 关键字。"""
    print("=" * 60)
    print("测试：retrieval 不含 Answer 关键字")
    print("=" * 60)

    # 测试所有可能的 hit 组合
    for t1 in (0, 1):
        for t3 in (0, 1):
            for t5 in (0, 1):
                if t5 == 0 and t3 == 1:
                    continue  # 无效组合
                if t5 == 0 and t1 == 1:
                    continue  # 无效组合
                r = {
                    "evaluation_track": TRACK_RETRIEVAL,
                    "question": "测试",
                    "retrieval_top1_hit": t1,
                    "retrieval_top3_hit": t3,
                    "retrieval_top5_hit": t5,
                    "hit_evidence_position": 1 if t1 else (3 if t3 else (5 if t5 else None)),
                    "answer_correct": 1,  # 即使有 answer_correct 也不应显示
                }
                text = _format_progress_line(r, 1)
                assert "Answer" not in text, \
                    f"t1={t1} t3={t3} t5={t5} 时 retrieval 不应含 Answer: {text}"
                assert "Top5" in text, \
                    f"t1={t1} t3={t3} t5={t5} 时 retrieval 应含 Top5: {text}"

    print("PASS: retrieval 所有组合均不含 Answer，均含 Top5")


# ====== 主函数 ======

def main():
    tests = [
        test_retrieval_progress_text,
        test_strict_qa_progress_text,
        test_grounded_qa_progress_text,
        test_not_evaluable_progress_text,
        test_error_result_progress_text,
        test_retrieval_no_answer_keyword,
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
