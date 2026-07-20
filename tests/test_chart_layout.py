
"""
图表 layout 单元测试。

断言 title/margin/height 配置满足不会裁切的最低条件：
1. 所有 bar/pie 图的 margin.t >= 40（为轴标题和标签留足空间）
2. 不在 Plotly figure 内设置 title（改用 st.markdown/st.caption）
3. height >= 280（保证最小可读高度）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from app import (
    build_retrieval_bar_chart,
    build_strict_qa_bar_chart,
    build_grounded_qa_bar_chart,
    build_answer_pye,
    build_retrieval_per_question_chart,
    build_per_question_chart,
)
from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA

MIN_TOP_MARGIN = 40
MIN_HEIGHT = 280


def _check_bar_chart(fig, name):
    layout = fig.layout
    margin_t = layout.margin.t if layout.margin and layout.margin.t else 0
    height = layout.height or 0
    has_title = bool(layout.title and layout.title.text)

    assert margin_t >= MIN_TOP_MARGIN, \
        f"{name}: margin.t={margin_t} < {MIN_TOP_MARGIN}"
    assert height >= MIN_HEIGHT, \
        f"{name}: height={height} < {MIN_HEIGHT}"
    assert not has_title, \
        f"{name}: title 应在图表外设置，不应在 Plotly figure 内设置 (got '{layout.title.text}')"
    print(f"  [OK] {name}: margin.t={margin_t}, height={height}, no_title={not has_title}")


def _check_pie_chart(fig, name):
    layout = fig.layout
    margin_t = layout.margin.t if layout.margin and layout.margin.t else 0
    height = layout.height or 0

    assert margin_t >= MIN_TOP_MARGIN, \
        f"{name}: margin.t={margin_t} < {MIN_TOP_MARGIN}"
    assert height >= MIN_HEIGHT, \
        f"{name}: height={height} < {MIN_HEIGHT}"
    print(f"  [OK] {name}: margin.t={margin_t}, height={height}")


def test_retrieval_bar_chart():
    """检索柱状图 margin 和 height 满足最低条件。"""
    metrics = {"top1_hit_rate": 0.8, "top3_hit_rate": 0.9, "top5_hit_rate": 1.0}
    fig = build_retrieval_bar_chart(metrics)
    _check_bar_chart(fig, "build_retrieval_bar_chart")


def test_strict_qa_bar_chart():
    """严格问答柱状图 margin 和 height 满足最低条件。"""
    metrics = {"answer_correct_rate": 0.75}
    fig = build_strict_qa_bar_chart(metrics)
    _check_bar_chart(fig, "build_strict_qa_bar_chart")


def test_grounded_qa_bar_chart():
    """合理性问答柱状图 margin 和 height 满足最低条件。"""
    metrics = {"answer_correct_rate": 0.6}
    fig = build_grounded_qa_bar_chart(metrics)
    _check_bar_chart(fig, "build_grounded_qa_bar_chart")


def test_answer_pie():
    """回答正确性饼图 margin 和 height 满足最低条件。"""
    results = [
        {"answer_correct": 1}, {"answer_correct": 0},
        {"answer_correct": 1}, {"answer_correct": 1},
    ]
    fig = build_answer_pye(results)
    _check_pie_chart(fig, "build_answer_pye")


def test_retrieval_per_question_chart():
    """检索每题命中图 margin 和 height 满足最低条件。"""
    results = [
        {"question": "测试问题1", "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1},
        {"question": "测试问题2", "retrieval_top1_hit": 0, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1},
    ]
    fig = build_retrieval_per_question_chart(results)
    assert fig is not None
    layout = fig.layout
    margin_t = layout.margin.t if layout.margin and layout.margin.t else 0
    assert margin_t >= MIN_TOP_MARGIN, \
        f"build_retrieval_per_question_chart: margin.t={margin_t} < {MIN_TOP_MARGIN}"
    print(f"  [OK] build_retrieval_per_question_chart: margin.t={margin_t}")


def test_per_question_chart():
    """通用每题命中图 margin 和 height 满足最低条件。"""
    results = [
        {"question": "测试问题1", "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "answer_correct": 1},
        {"question": "测试问题2", "retrieval_top1_hit": 0, "retrieval_top3_hit": 0, "answer_correct": 0},
    ]
    fig = build_per_question_chart(results)
    assert fig is not None
    layout = fig.layout
    margin_t = layout.margin.t if layout.margin and layout.margin.t else 0
    assert margin_t >= MIN_TOP_MARGIN, \
        f"build_per_question_chart: margin.t={margin_t} < {MIN_TOP_MARGIN}"
    print(f"  [OK] build_per_question_chart: margin.t={margin_t}")


def main():
    print("=" * 60)
    print("图表 layout 单元测试")
    print("=" * 60)
    print()

    test_retrieval_bar_chart()
    test_strict_qa_bar_chart()
    test_grounded_qa_bar_chart()
    test_answer_pie()
    test_retrieval_per_question_chart()
    test_per_question_chart()

    print()
    print("=" * 60)
    print("[OK] 所有图表 layout 测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
