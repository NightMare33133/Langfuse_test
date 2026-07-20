"""
回归测试：验证 build_result_status 函数的行为。

测试内容：
1. retrieval 结果只有 TopK 字段、没有 answer_correct 时，状态不得显示为错误
2. retrieval Top1=0 / Top3=1 时，标题必须体现"Top1 未命中，Top3 命中"
3. strict_qa 结果显示回答正确/错误
4. grounded_qa 结果显示回答有据/缺乏依据
5. not_evaluable 结果显示缺少金标准证据

不调用真实 LLM、Dify 或 Langfuse API。
"""

from judge import (
    build_result_status,
    TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE
)


def test_retrieval_status():
    """测试检索评测状态。"""
    print("=" * 60)
    print("测试检索评测状态")
    print("=" * 60)

    # 测试 1: retrieval 结果只有 TopK 字段、没有 answer_correct
    result_1 = {
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        # 注意：没有 answer_correct
    }
    status_1 = build_result_status(result_1)
    assert status_1["icon"] == "🔍", f"期望图标 search，实际 {status_1['icon']}"
    assert "❌" not in status_1["title"], f"标题不应包含 X，实际: {status_1['title']}"
    assert "✅" not in status_1["title"], f"标题不应包含 check，实际: {status_1['title']}"
    print(f"[OK] retrieval 无 answer_correct 时，图标为 search，无 X/check")
    print(f"     标题: {status_1['title']}")

    # 测试 2: retrieval Top1=0 / Top3=1 时，标题必须体现"Top1 未命中，Top3 命中"
    assert "Top1 未命中" in status_1["title"], f"标题应包含 'Top1 未命中'，实际: {status_1['title']}"
    assert "Top3 命中" in status_1["title"], f"标题应包含 'Top3 命中'，实际: {status_1['title']}"
    print(f"[OK] 标题正确体现 Top1 未命中、Top3 命中")

    # 测试 3: retrieval Top1=1 / Top3=1 / Top5=1
    result_2 = {
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_top1_hit": 1,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
    }
    status_2 = build_result_status(result_2)
    assert "Top1 命中" in status_2["title"], f"标题应包含 'Top1 命中'，实际: {status_2['title']}"
    print(f"[OK] 全命中时标题正确: {status_2['title']}")

    # 测试 4: retrieval Top1=0 / Top3=0 / Top5=0
    result_3 = {
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 0,
        "retrieval_top5_hit": 0,
    }
    status_3 = build_result_status(result_3)
    assert "Top1 未命中" in status_3["title"], f"标题应包含 'Top1 未命中'，实际: {status_3['title']}"
    assert "Top3 未命中" in status_3["title"], f"标题应包含 'Top3 未命中'，实际: {status_3['title']}"
    assert "Top5 未命中" in status_3["title"], f"标题应包含 'Top5 未命中'，实际: {status_3['title']}"
    print(f"[OK] 全未命中时标题正确: {status_3['title']}")

    print()


def test_strict_qa_status():
    """测试严格问答状态。"""
    print("=" * 60)
    print("测试严格问答状态")
    print("=" * 60)

    # 测试 1: 回答正确
    result_1 = {
        "evaluation_track": TRACK_STRICT_QA,
        "answer_correct": 1,
    }
    status_1 = build_result_status(result_1)
    assert status_1["icon"] == "✅", f"期望图标 check，实际 {status_1['icon']}"
    assert status_1["title"] == "回答正确", f"期望标题 '回答正确'，实际 {status_1['title']}"
    print(f"[OK] 回答正确时: {status_1['title']}")

    # 测试 2: 回答错误
    result_2 = {
        "evaluation_track": TRACK_STRICT_QA,
        "answer_correct": 0,
    }
    status_2 = build_result_status(result_2)
    assert status_2["icon"] == "❌", f"期望图标 X，实际 {status_2['icon']}"
    assert status_2["title"] == "回答错误", f"期望标题 '回答错误'，实际 {status_2['title']}"
    print(f"[OK] 回答错误时: {status_2['title']}")

    print()


def test_grounded_qa_status():
    """测试合理性问答状态。"""
    print("=" * 60)
    print("测试合理性问答状态")
    print("=" * 60)

    # 测试 1: 回答有据
    result_1 = {
        "evaluation_track": TRACK_GROUNDED_QA,
        "answer_correct": 1,
    }
    status_1 = build_result_status(result_1)
    assert status_1["icon"] == "✅", f"期望图标 check，实际 {status_1['icon']}"
    assert status_1["title"] == "回答有据", f"期望标题 '回答有据'，实际 {status_1['title']}"
    assert "正确" not in status_1["title"], f"不应出现 '正确'，实际 {status_1['title']}"
    print(f"[OK] 回答有据时: {status_1['title']}")

    # 测试 2: 回答缺乏依据
    result_2 = {
        "evaluation_track": TRACK_GROUNDED_QA,
        "answer_correct": 0,
    }
    status_2 = build_result_status(result_2)
    assert status_2["icon"] == "⚠️", f"期望图标 warning，实际 {status_2['icon']}"
    assert status_2["title"] == "回答缺乏依据", f"期望标题 '回答缺乏依据'，实际 {status_2['title']}"
    print(f"[OK] 回答缺乏依据时: {status_2['title']}")

    print()


def test_not_evaluable_status():
    """测试不可评测状态。"""
    print("=" * 60)
    print("测试不可评测状态")
    print("=" * 60)

    result = {
        "evaluation_track": TRACK_NOT_EVALUABLE,
    }
    status = build_result_status(result)
    assert status["icon"] == "⚠️", f"期望图标 warning，实际 {status['icon']}"
    assert "缺少金标准" in status["title"], f"标题应包含 '缺少金标准'，实际 {status['title']}"
    assert "❌" not in status["title"], f"标题不应包含 X，实际 {status['title']}"
    print(f"[OK] 不可评测时: {status['title']}")

    print()


def test_unknown_track():
    """测试未知轨道状态。"""
    print("=" * 60)
    print("测试未知轨道状态")
    print("=" * 60)

    result = {
        "evaluation_track": "unknown_track",
    }
    status = build_result_status(result)
    assert status["icon"] == "❓", f"期望图标 ?，实际 {status['icon']}"
    print(f"[OK] 未知轨道时: {status['title']}")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("build_result_status 回归测试")
    print("=" * 60)
    print()

    test_retrieval_status()
    test_strict_qa_status()
    test_grounded_qa_status()
    test_not_evaluable_status()
    test_unknown_track()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
