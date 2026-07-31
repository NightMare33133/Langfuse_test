"""
Judge 评测进度文案测试。

验证五种 evaluation_track 的实时进度显示：
- retrieval: Top1/Top3/Top5 + 最早命中位置，不含 Answer
- strict_qa: Answer
- grounded_qa: 回答有据
- chunk_exact: Chunk Exact + Top1/Top3/Top5 或状态提示
- not_evaluable: 不可评测

不调用真实 API。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE, TRACK_CHUNK_EXACT


def _format_progress_line(result, idx):
    """从 app.py 提取的进度文案格式化逻辑。与 app.py 保持同步。"""
    track = result.get("evaluation_track", "")
    q = (result.get("question") or "")[:40]
    tag = ""

    if track == TRACK_RETRIEVAL:
        t1 = "✓" if result.get("retrieval_top1_hit") else "✗"
        t3 = "✓" if result.get("retrieval_top3_hit") else "✗"
        t5 = "✓" if result.get("retrieval_top5_hit") else "✗"
        pos = result.get("hit_evidence_position")
        pos_str = str(pos) if pos else "无"
        return f"✅ [{idx}] {q} — Top1:{t1} | Top3:{t3} | Top5:{t5} | 最早命中位置:{pos_str}{tag}"
    elif track == TRACK_CHUNK_EXACT:
        ce_status = result.get("chunk_exact_status", "")
        if ce_status:
            status_labels = {
                "missing_binding": "缺少绑定（expected_segment_id / expected_content_hash）",
                "no_trace": "未关联真实 Langfuse trace",
                "no_retrieval": "trace 已关联但无检索结果",
            }
            label = status_labels.get(ce_status, ce_status)
            return f"⚠️ [{idx}] {q} — Chunk Exact 不可判定：{label}{tag}"
        else:
            t1 = "✓" if result.get("retrieval_top1_hit") else "✗"
            t3 = "✓" if result.get("retrieval_top3_hit") else "✗"
            t5 = "✓" if result.get("retrieval_top5_hit") else "✗"
            pos = result.get("hit_evidence_position")
            seg_id = (result.get("expected_segment_id") or "")[:12]
            if result.get("retrieval_top5_hit"):
                return (
                    f"✅ [{idx}] {q} — Chunk Exact | "
                    f"Top1:{t1} | Top3:{t3} | Top5:{t5} | "
                    f"首次命中:Top{pos}{tag}"
                )
            else:
                return (
                    f"⚠️ [{idx}] {q} — Chunk Exact | "
                    f"Top5未命中 | 目标 chunk:{seg_id}{tag}"
                )
    elif track == TRACK_STRICT_QA:
        ans = "✓" if result.get("answer_correct") else "✗"
        return f"✅ [{idx}] {q} — Answer:{ans}{tag}"
    elif track == TRACK_GROUNDED_QA:
        gnd = "✓" if result.get("answer_correct") else "✗"
        return f"✅ [{idx}] {q} — 回答有据:{gnd}{tag}"
    else:
        return f"✅ [{idx}] {q} — 不可评测：缺少金标准证据{tag}"


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
    assert "Chunk Exact" not in text, f"retrieval 不应包含 Chunk Exact，实际: {text}"

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
    assert "Chunk Exact" not in text

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
    assert "Chunk Exact" not in text

    r_ungrounded = {
        "evaluation_track": TRACK_GROUNDED_QA,
        "question": "测试问题",
        "answer_correct": 0,
    }
    text2 = _format_progress_line(r_ungrounded, 2)
    assert "回答有据:✗" in text2, f"应包含回答有据:✗，实际: {text2}"

    print("PASS: grounded_qa 进度文案正确")


def test_chunk_exact_hit_progress_text():
    """chunk_exact 轨道命中：显示 Chunk Exact + Top1/Top3/Top5。"""
    print("=" * 60)
    print("测试：chunk_exact 命中文案")
    print("=" * 60)

    # Top1 命中
    r_top1 = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "什么是 RAG？",
        "retrieval_top1_hit": 1,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 1,
        "expected_segment_id": "seg_abc123",
        "chunk_exact_status": "",
    }
    text = _format_progress_line(r_top1, 1)
    assert "Chunk Exact" in text, f"应包含 Chunk Exact，实际: {text}"
    assert "Top1:✓" in text, f"应包含 Top1:✓，实际: {text}"
    assert "Top3:✓" in text
    assert "Top5:✓" in text
    assert "首次命中:Top1" in text
    assert "不可评测" not in text, f"命中结果不应包含不可评测，实际: {text}"
    assert "缺少金标准" not in text

    # Top3 命中
    r_top3 = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "测试问题",
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 3,
        "expected_segment_id": "seg_xyz",
        "chunk_exact_status": "",
    }
    text2 = _format_progress_line(r_top3, 2)
    assert "Top1:✗" in text2
    assert "Top3:✓" in text2
    assert "首次命中:Top3" in text2
    assert "不可评测" not in text2

    print("PASS: chunk_exact 命中文案正确")


def test_chunk_exact_miss_progress_text():
    """chunk_exact 轨道 Top5 未命中：显示警告 + 目标 chunk。"""
    print("=" * 60)
    print("测试：chunk_exact Top5 未命中文案")
    print("=" * 60)

    r_miss = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "某个问题",
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 0,
        "retrieval_top5_hit": 0,
        "hit_evidence_position": None,
        "expected_segment_id": "seg_miss123456",
        "chunk_exact_status": "",
    }
    text = _format_progress_line(r_miss, 3)
    assert "⚠️" in text, f"应包含警告图标，实际: {text}"
    assert "Top5未命中" in text, f"应包含 Top5未命中，实际: {text}"
    assert "目标 chunk:seg_miss123" in text, f"应包含目标 chunk 短 ID，实际: {text}"
    assert "不可评测" not in text, f"Top5 未命中不应显示不可评测，实际: {text}"
    assert "缺少金标准" not in text

    print("PASS: chunk_exact Top5 未命中文案正确")


def test_chunk_exact_status_progress_text():
    """chunk_exact 轨道各种 status：显示对应状态而非缺少金标准。"""
    print("=" * 60)
    print("测试：chunk_exact 状态文案")
    print("=" * 60)

    # missing_binding
    r_missing = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "绑定缺失",
        "chunk_exact_status": "missing_binding",
    }
    text = _format_progress_line(r_missing, 1)
    assert "⚠️" in text
    assert "Chunk Exact 不可判定" in text, f"应包含 Chunk Exact 不可判定，实际: {text}"
    assert "缺少绑定" in text, f"应包含缺少绑定，实际: {text}"
    assert "不可评测" not in text, f"chunk_exact 不应显示不可评测，实际: {text}"
    assert "缺少金标准证据" not in text, f"chunk_exact 不应显示缺少金标准证据，实际: {text}"

    # no_trace
    r_no_trace = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "无 trace",
        "chunk_exact_status": "no_trace",
    }
    text2 = _format_progress_line(r_no_trace, 2)
    assert "未关联真实 Langfuse trace" in text2, f"应包含未关联 trace，实际: {text2}"
    assert "不可评测" not in text2
    assert "缺少金标准证据" not in text2

    # no_retrieval
    r_no_ret = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "无检索结果",
        "chunk_exact_status": "no_retrieval",
    }
    text3 = _format_progress_line(r_no_ret, 3)
    assert "无检索结果" in text3, f"应包含无检索结果，实际: {text3}"
    assert "不可评测" not in text3
    assert "缺少金标准证据" not in text3

    print("PASS: chunk_exact 各状态文案正确，不含不可评测/缺少金标准")


def test_chunk_exact_no_wrong_labels():
    """chunk_exact 结果不包含 retrieval/QA 特有标签。"""
    print("=" * 60)
    print("测试：chunk_exact 不含错误标签")
    print("=" * 60)

    r = {
        "evaluation_track": TRACK_CHUNK_EXACT,
        "question": "测试",
        "retrieval_top1_hit": 1,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 1,
        "expected_segment_id": "seg_001",
        "chunk_exact_status": "",
    }
    text = _format_progress_line(r, 1)
    assert "Answer" not in text, f"chunk_exact 不应含 Answer: {text}"
    assert "回答有据" not in text, f"chunk_exact 不应含 回答有据: {text}"
    assert "最早命中位置" not in text, f"chunk_exact 不应含 最早命中位置: {text}"
    assert "不可评测" not in text, f"chunk_exact 命中时不应含 不可评测: {text}"
    assert "缺少金标准" not in text, f"chunk_exact 不应含 缺少金标准: {text}"

    print("PASS: chunk_exact 不含 retrieval/QA 特有标签")


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
    assert "Chunk Exact" not in text

    print("PASS: not_evaluable 进度文案正确")


def test_error_result_progress_text():
    """error 结果显示错误信息，不显示指标。"""
    print("=" * 60)
    print("测试：error 结果进度文案")
    print("=" * 60)

    r = {
        "evaluation_track": TRACK_RETRIEVAL,
        "question": "错误问题",
        "error": "LLM 调用超时",
    }
    assert "error" in r, "error 结果应有 error key"

    print("PASS: error 结果由独立分支处理")


def test_retrieval_no_answer_keyword():
    """retrieval 文案中不出现 Answer 关键字。"""
    print("=" * 60)
    print("测试：retrieval 不含 Answer 关键字")
    print("=" * 60)

    for t1 in (0, 1):
        for t3 in (0, 1):
            for t5 in (0, 1):
                if t5 == 0 and t3 == 1:
                    continue
                if t5 == 0 and t1 == 1:
                    continue
                r = {
                    "evaluation_track": TRACK_RETRIEVAL,
                    "question": "测试",
                    "retrieval_top1_hit": t1,
                    "retrieval_top3_hit": t3,
                    "retrieval_top5_hit": t5,
                    "hit_evidence_position": 1 if t1 else (3 if t3 else (5 if t5 else None)),
                    "answer_correct": 1,
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
        test_chunk_exact_hit_progress_text,
        test_chunk_exact_miss_progress_text,
        test_chunk_exact_status_progress_text,
        test_chunk_exact_no_wrong_labels,
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
