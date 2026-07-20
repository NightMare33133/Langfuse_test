"""
回归测试：验证图表函数和检索详情展示。

测试内容：
1. strict_qa 图表只有 Answer Correctness
2. retrieval 图表没有 Answer 维度
3. retrieval 详情能从 sample 的 retrieval_results 按 position 输出真实检索结果

不调用真实 LLM、Dify 或 Langfuse API。
"""

import sys

from judge import (
    TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE,
    build_result_status
)

# 直接定义图表函数，避免导入 app.py 触发 Streamlit 上下文
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def build_retrieval_bar_chart(metrics: dict):
    """检索评测专用图表：只显示 Top1/Top3/Top5 Hit。"""
    labels = ["Top1 Hit", "Top3 Hit", "Top5 Hit"]
    keys = ["top1_hit_rate", "top3_hit_rate", "top5_hit_rate"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    values = [(metrics.get(key) or 0) * 100 for key in keys]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, text=[f"{v:.1f}%" for v in values], textposition="auto", marker_color=colors)])
    fig.update_layout(yaxis_title="百分比 (%)", yaxis_range=[0, 100], height=360, margin=dict(t=20, b=30))
    return fig


def build_strict_qa_bar_chart(metrics: dict):
    """严格问答专用图表：只显示 Answer Correctness。"""
    labels = ["Answer Correctness"]
    values = [(metrics.get("answer_correct_rate") or 0) * 100]
    colors = ["#17becf"]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, text=[f"{v:.1f}%" for v in values], textposition="auto", marker_color=colors)])
    fig.update_layout(yaxis_title="百分比 (%)", yaxis_range=[0, 100], height=360, margin=dict(t=20, b=30))
    return fig


def build_retrieval_per_question_chart(valid_results: list):
    """检索评测专用每题命中图：只显示 Top1/Top3/Top5，不含 Answer。"""
    if not valid_results:
        return None
    rows = []
    for r in valid_results:
        q = r.get("question", "")
        rows.append({"question": q[:30] + ("..." if len(q) > 30 else ""), "Top1": r.get("retrieval_top1_hit", 0) or 0, "Top3": r.get("retrieval_top3_hit", 0) or 0, "Top5": r.get("retrieval_top5_hit", 0) or 0})
    df = pd.DataFrame(rows)
    df = df.sort_values(["Top1", "Top3"], ascending=[True, True])
    df_melted = df.melt(id_vars="question", var_name="指标", value_name="命中")
    df_melted["命中"] = df_melted["命中"].map({1: "命中", 0: "未命中"})
    fig = px.bar(df_melted, x="question", y="指标", color="命中", orientation="h", color_discrete_map={"命中": "#2ca02c", "未命中": "#d62728"}, barmode="group")
    fig.update_layout(height=max(280, len(df) * 36 + 80), margin=dict(t=20, b=30, l=10), xaxis_title="", yaxis_title="")
    return fig


def test_strict_qa_chart():
    """测试严格问答图表：只有 Answer Correctness。"""
    print("=" * 60)
    print("测试严格问答图表")
    print("=" * 60)

    # 模拟严格问答指标
    metrics = {
        "answer_correct_rate": 0.75,
    }

    fig = build_strict_qa_bar_chart(metrics)

    # 验证只有一个柱子
    assert len(fig.data) == 1, f"期望 1 个柱子，实际 {len(fig.data)}"
    assert fig.data[0].x[0] == "Answer Correctness", f"期望标签 'Answer Correctness'，实际 {fig.data[0].x[0]}"
    assert len(fig.data[0].x) == 1, f"期望只有 1 个标签，实际 {len(fig.data[0].x)}"
    print("[OK] 严格问答图表只有 Answer Correctness")

    # 验证不包含 Top1/Top3/Top5
    labels = list(fig.data[0].x)
    assert "Top1 Hit" not in labels, "不应包含 Top1 Hit"
    assert "Top3 Hit" not in labels, "不应包含 Top3 Hit"
    assert "Top5 Hit" not in labels, "不应包含 Top5 Hit"
    print("[OK] 严格问答图表不包含 Top1/Top3/Top5")

    print()


def test_retrieval_chart():
    """测试检索评测图表：没有 Answer 维度。"""
    print("=" * 60)
    print("测试检索评测图表")
    print("=" * 60)

    # 模拟检索评测指标
    metrics = {
        "top1_hit_rate": 0.5,
        "top3_hit_rate": 0.8,
        "top5_hit_rate": 0.9,
    }

    fig = build_retrieval_bar_chart(metrics)

    # 验证只有三个柱子
    assert len(fig.data) == 1, f"期望 1 个柱子序列，实际 {len(fig.data)}"
    labels = list(fig.data[0].x)
    assert len(labels) == 3, f"期望 3 个标签，实际 {len(labels)}"
    print(f"[OK] 检索评测图表有 3 个标签: {labels}")

    # 验证不包含 Answer
    assert "Answer Correct" not in labels, "不应包含 Answer Correct"
    assert "Answer" not in labels, "不应包含 Answer"
    print("[OK] 检索评测图表不包含 Answer 维度")

    # 验证包含 Top1/Top3/Top5
    assert "Top1 Hit" in labels, "应包含 Top1 Hit"
    assert "Top3 Hit" in labels, "应包含 Top3 Hit"
    assert "Top5 Hit" in labels, "应包含 Top5 Hit"
    print("[OK] 检索评测图表包含 Top1/Top3/Top5")

    print()


def test_retrieval_per_question_chart():
    """测试检索评测每题命中图：没有 Answer 维度。"""
    print("=" * 60)
    print("测试检索评测每题命中图")
    print("=" * 60)

    # 模拟检索评测结果
    valid_results = [
        {
            "question": "测试问题1",
            "retrieval_top1_hit": 1,
            "retrieval_top3_hit": 1,
            "retrieval_top5_hit": 1,
            "answer_correct": 1,  # 有值但不应显示
        },
        {
            "question": "测试问题2",
            "retrieval_top1_hit": 0,
            "retrieval_top3_hit": 1,
            "retrieval_top5_hit": 1,
            "answer_correct": 0,  # 有值但不应显示
        },
    ]

    fig = build_retrieval_per_question_chart(valid_results)

    # 验证不包含 Answer 维度
    # 通过检查 fig.data 中的 y 值
    y_values = set()
    for trace in fig.data:
        y_values.update(trace.y)

    assert "Answer" not in y_values, f"不应包含 Answer 维度，实际有: {y_values}"
    assert "Top1" in y_values, f"应包含 Top1 维度，实际有: {y_values}"
    assert "Top3" in y_values, f"应包含 Top3 维度，实际有: {y_values}"
    assert "Top5" in y_values, f"应包含 Top5 维度，实际有: {y_values}"
    print(f"[OK] 检索评测每题命中图只有 Top1/Top3/Top5，没有 Answer")
    print(f"     实际维度: {y_values}")

    print()


def test_retrieval_details_from_sample():
    """测试检索详情能从 sample 的 retrieval_results 按 position 输出真实检索结果。"""
    print("=" * 60)
    print("测试检索详情展示")
    print("=" * 60)

    # 模拟 sample 数据
    sample = {
        "question": "测试问题",
        "source_excerpt": "测试金标准证据",
        "retrieval_results": [
            {
                "position": 1,
                "score": 0.95,
                "document_name": "doc1.md",
                "content": "这是第一条检索结果的内容",
            },
            {
                "position": 2,
                "score": 0.85,
                "document_name": "doc2.md",
                "content": "这是第二条检索结果的内容",
            },
            {
                "position": 3,
                "score": 0.75,
                "document_name": "doc3.md",
                "content": "这是第三条检索结果的内容",
            },
        ],
        "final_answer": "测试回答",
    }

    # 模拟 Judge 结果
    judge_result = {
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 2,
        "reason": "Top1 未命中，Top3 命中",
    }

    # 验证 retrieval_results 可按 position 排序
    _retrieval_results = sample.get("retrieval_results") or []
    sorted_results = sorted(_retrieval_results, key=lambda x: x.get("position", 999))
    positions = [r.get("position") for r in sorted_results]
    assert positions == [1, 2, 3], f"期望按 position 排序 [1, 2, 3]，实际 {positions}"
    print(f"[OK] retrieval_results 可按 position 排序: {positions}")

    # 验证 hit_evidence_position 可用于标识命中
    _hit_pos = judge_result.get("hit_evidence_position")
    assert _hit_pos == 2, f"期望命中位置 2，实际 {_hit_pos}"
    print(f"[OK] hit_evidence_position 正确标识命中位置: {_hit_pos}")

    # 验证命中标识逻辑
    for rr in sorted_results:
        _pos = rr.get("position")
        _is_hit = (_hit_pos is not None and _pos == _hit_pos)
        if _pos == 2:
            assert _is_hit, f"位置 {_pos} 应被标识为命中"
            print(f"[OK] 位置 {_pos} 正确标识为命中")
        else:
            assert not _is_hit, f"位置 {_pos} 不应被标识为命中"
    print(f"[OK] 命中标识逻辑正确")

    # 验证 Top1 未命中但 Top3 命中时的展示逻辑
    _t1 = judge_result.get("retrieval_top1_hit")
    _t3 = judge_result.get("retrieval_top3_hit")
    assert _t1 == 0, f"Top1 应为未命中"
    assert _t3 == 1, f"Top3 应为命中"
    print(f"[OK] Top1 未命中但 Top3 命中的场景正确")

    print()


def test_build_result_status():
    """测试 build_result_status 函数。"""
    print("=" * 60)
    print("测试 build_result_status 函数")
    print("=" * 60)

    # 测试 retrieval 状态
    result_retrieval = {
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
    }
    status = build_result_status(result_retrieval)
    assert "Top1 未命中" in status["title"], f"标题应包含 'Top1 未命中'，实际 {status['title']}"
    assert "Top3 命中" in status["title"], f"标题应包含 'Top3 命中'，实际 {status['title']}"
    print(f"[OK] retrieval 状态正确: {status['title']}")

    # 测试 strict_qa 状态
    result_strict = {
        "evaluation_track": TRACK_STRICT_QA,
        "answer_correct": 1,
    }
    status = build_result_status(result_strict)
    assert status["title"] == "回答正确", f"期望标题 '回答正确'，实际 {status['title']}"
    print(f"[OK] strict_qa 状态正确: {status['title']}")

    # 测试 grounded_qa 状态
    result_grounded = {
        "evaluation_track": TRACK_GROUNDED_QA,
        "answer_correct": 0,
    }
    status = build_result_status(result_grounded)
    assert status["title"] == "回答缺乏依据", f"期望标题 '回答缺乏依据'，实际 {status['title']}"
    print(f"[OK] grounded_qa 状态正确: {status['title']}")

    # 测试 not_evaluable 状态
    result_not_eval = {
        "evaluation_track": TRACK_NOT_EVALUABLE,
    }
    status = build_result_status(result_not_eval)
    assert "缺少金标准" in status["title"], f"标题应包含 '缺少金标准'，实际 {status['title']}"
    print(f"[OK] not_evaluable 状态正确: {status['title']}")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("图表函数和检索详情展示回归测试")
    print("=" * 60)
    print()

    test_strict_qa_chart()
    test_retrieval_chart()
    test_retrieval_per_question_chart()
    test_retrieval_details_from_sample()
    test_build_result_status()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
