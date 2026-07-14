import streamlit as st
from pathlib import Path
from datetime import datetime
import json
import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from parser import parse_langfuse_jsonl, save_results
from judge import judge_all, compute_metrics, call_llm, pre_screen, compute_content_hash, build_judge_prompt, load_prompt_template, load_prompt_template_with_ref, build_result_status
from question_generator import generate_questions, save_questions, export_csv_bytes, choose_strategy, STRATEGY_LABELS, MODE_RETRIEVAL, MODE_QA, MODE_LABELS, build_question_set_name
from batch_query import run_batch_query, save_batch_results, push_to_raw_dir, export_csv_bytes as batch_export_csv

load_dotenv(Path(__file__).parent / ".env")

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
JUDGED_DIR = Path(__file__).parent / "data" / "judged"
JUDGED_FILE = JUDGED_DIR / "eval_results.jsonl"
BATCH_DIR = Path(__file__).parent / "data" / "batch"
QUESTIONS_DIR = Path(__file__).parent / "data" / "questions"


# ---------- 配置表单统一 helper ----------

def render_config_form(config: dict, key_prefix: str, disabled: bool = False) -> dict:
    """根据 CONFIG_FIELD_SCHEMA 渲染配置表单，返回 {field_key: value} 字典。

    Args:
        config: 当前配置值（用于回填）
        key_prefix: Streamlit widget key 前缀（避免冲突）
        disabled: 是否只读模式

    Returns:
        dict: 各字段的当前值
    """
    from experiment import CONFIG_FIELD_SCHEMA

    values = {}
    required_fields = []
    optional_fields = []

    for key, label, required, widget, placeholder, help_text in CONFIG_FIELD_SCHEMA:
        if required:
            required_fields.append((key, label, required, widget, placeholder, help_text))
        else:
            optional_fields.append((key, label, required, widget, placeholder, help_text))

    # 必填字段
    st.markdown("**必填字段**")
    req_col1, req_col2 = st.columns(2)
    for i, (key, label, _, widget, placeholder, help_text) in enumerate(required_fields):
        with (req_col1 if i % 2 == 0 else req_col2):
            display_label = f"{label} *" if not disabled else label
            val = config.get(key, "")
            if widget == "textarea":
                values[key] = st.text_area(
                    display_label, value=str(val),
                    placeholder=placeholder, key=f"{key_prefix}_{key}",
                    height=68, disabled=disabled, help=help_text,
                )
            else:
                values[key] = st.text_input(
                    display_label, value=str(val),
                    placeholder=placeholder, key=f"{key_prefix}_{key}",
                    disabled=disabled, help=help_text,
                )

    # 可选字段（折叠区）
    with st.expander("补充实验参数（可选）", expanded=False):
        opt_col1, opt_col2 = st.columns(2)
        for i, (key, label, _, widget, placeholder, help_text) in enumerate(optional_fields):
            with (opt_col1 if i % 2 == 0 else opt_col2):
                val = config.get(key, "")
                if widget == "textarea":
                    values[key] = st.text_area(
                        label, value=str(val) if val is not None else "",
                        placeholder=placeholder, key=f"{key_prefix}_{key}",
                        height=68, disabled=disabled, help=help_text,
                    )
                elif widget == "number":
                    # number_input 需要 int 值
                    num_val = val if isinstance(val, (int, float)) else 0
                    values[key] = st.number_input(
                        label, value=int(num_val), min_value=0, step=1,
                        key=f"{key_prefix}_{key}", disabled=disabled, help=help_text,
                    )
                else:
                    values[key] = st.text_input(
                        label, value=str(val) if val is not None else "",
                        placeholder=placeholder, key=f"{key_prefix}_{key}",
                        disabled=disabled, help=help_text,
                    )

    return values


def collect_config_updates(form_values: dict) -> dict:
    """从表单值收集更新字典，过滤空值和零值。"""
    updates = {}
    for key, val in form_values.items():
        if isinstance(val, (int, float)):
            if val > 0:
                updates[key] = val
        elif isinstance(val, str) and val.strip():
            updates[key] = val.strip()
    return updates


# ---------- 评测结果可视化 / 导出辅助函数 ----------

def build_retrieval_bar_chart(metrics: dict):
    """检索评测专用图表：只显示 Top1/Top3/Top5 Hit。"""
    labels = ["Top1 Hit", "Top3 Hit", "Top5 Hit"]
    keys = ["top1_hit_rate", "top3_hit_rate", "top5_hit_rate"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    values = []
    for key in keys:
        val = metrics.get(key)
        values.append((val or 0) * 100)

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        marker_color=colors,
    )])
    fig.update_layout(
        yaxis_title="百分比 (%)", yaxis_range=[0, 100],
        height=360, margin=dict(t=40, b=30),
    )
    return fig


def build_strict_qa_bar_chart(metrics: dict):
    """严格问答专用图表：只显示 Answer Correctness。"""
    labels = ["Answer Correctness"]
    values = [(metrics.get("answer_correct_rate") or 0) * 100]
    colors = ["#17becf"]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        marker_color=colors,
    )])
    fig.update_layout(
        yaxis_title="百分比 (%)", yaxis_range=[0, 100],
        height=360, margin=dict(t=40, b=30),
    )
    return fig


def build_grounded_qa_bar_chart(metrics: dict):
    """合理性问答专用图表：只显示 Answer Grounded。"""
    labels = ["Answer Grounded"]
    values = [(metrics.get("answer_correct_rate") or 0) * 100]
    colors = ["#2ca02c"]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        marker_color=colors,
    )])
    fig.update_layout(
        yaxis_title="百分比 (%)", yaxis_range=[0, 100],
        height=360, margin=dict(t=40, b=30),
    )
    return fig


def build_answer_pye(valid_results: list, label_correct="正确", label_incorrect="错误"):
    """回答正确性饼图。"""
    correct = sum(1 for r in valid_results if r.get("answer_correct"))
    incorrect = len(valid_results) - correct
    fig = go.Figure(data=[go.Pie(
        labels=[label_correct, label_incorrect],
        values=[correct, incorrect],
        marker_colors=["#2ca02c", "#d62728"],
        hole=0.4,
        textinfo="label+value+percent",
    )])
    fig.update_layout(height=340, margin=dict(t=40, b=20))
    return fig


def build_retrieval_per_question_chart(valid_results: list):
    """检索评测专用每题命中图：只显示 Top1/Top3/Top5，不含 Answer。"""
    if not valid_results:
        return None
    rows = []
    for r in valid_results:
        q = r.get("question", "")
        rows.append({
            "question": q[:30] + ("..." if len(q) > 30 else ""),
            "Top1": r.get("retrieval_top1_hit", 0) or 0,
            "Top3": r.get("retrieval_top3_hit", 0) or 0,
            "Top5": r.get("retrieval_top5_hit", 0) or 0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["Top1", "Top3"], ascending=[True, True])
    df_melted = df.melt(id_vars="question", var_name="指标", value_name="命中")
    df_melted["命中"] = df_melted["命中"].map({1: "命中", 0: "未命中"})
    fig = px.bar(
        df_melted, x="question", y="指标", color="命中",
        orientation="h",
        color_discrete_map={"命中": "#2ca02c", "未命中": "#d62728"},
        barmode="group",
    )
    fig.update_layout(
        height=max(280, len(df) * 36 + 80),
        margin=dict(t=40, b=30, l=10),
        xaxis_title="", yaxis_title="",
    )
    return fig


def build_per_question_chart(valid_results: list):
    """通用每题命中图：显示 Top1/Top3/Answer（兼容旧版）。"""
    if not valid_results:
        return None
    rows = []
    for r in valid_results:
        q = r.get("question", "")
        rows.append({
            "question": q[:30] + ("..." if len(q) > 30 else ""),
            "Top1": r.get("retrieval_top1_hit", 0) or 0,
            "Top3": r.get("retrieval_top3_hit", 0) or 0,
            "Answer": r.get("answer_correct", 0) or 0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["Answer", "Top1"], ascending=[True, True])
    df_melted = df.melt(id_vars="question", var_name="指标", value_name="命中")
    df_melted["命中"] = df_melted["命中"].map({1: "命中", 0: "未命中"})
    fig = px.bar(
        df_melted, x="question", y="指标", color="命中",
        orientation="h",
        color_discrete_map={"命中": "#2ca02c", "未命中": "#d62728"},
        barmode="group",
    )
    fig.update_layout(
        height=max(280, len(df) * 36 + 80),
        margin=dict(t=40, b=30, l=10),
        xaxis_title="", yaxis_title="",
    )
    return fig


def build_csv_download(results: list) -> str:
    rows = []
    for r in results:
        rows.append({
            "trace_id": r.get("trace_id", ""),
            "question": r.get("question", ""),
            "retrieval_top1_hit": r.get("retrieval_top1_hit"),
            "retrieval_top3_hit": r.get("retrieval_top3_hit"),
            "retrieval_top5_hit": r.get("retrieval_top5_hit"),
            "answer_correct": r.get("answer_correct"),
            "reason": r.get("reason", ""),
            "error": r.get("error", ""),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8-sig")


def build_markdown_report(results: list) -> str:
    valid = [r for r in results if "error" not in r]
    m = compute_metrics(results)

    top1_miss = [r for r in valid if not r.get("retrieval_top1_hit")]

    def _rate(v):
        return f"{v:.0%}" if v is not None else "N/A"

    lines = [
        "# RAG 评测报告", "",
        "## 评测汇总",
        f"- 总样本数: {m['total']}",
        f"- 有效评测数: {m['evaluated']}",
        f"- 错误数: {m['errors']}", "",
        "### 命中率 / 正确率",
        f"| 指标 | 值 |",
        f"|------|------|",
        f"| Top1 Hit Rate | {_rate(m['top1_hit_rate'])} |",
        f"| Top3 Hit Rate | {_rate(m['top3_hit_rate'])} |",
        f"| Top5 Hit Rate | {_rate(m['top5_hit_rate'])} |",
        f"| Answer Correctness | {_rate(m['answer_correct_rate'])} |", "",
        "## Top1 未命中案例",
    ]

    if top1_miss:
        lines.append(f"共 {len(top1_miss)} 条：")
        lines.append("")
        lines.append("| # | 问题 | 原因 |")
        lines.append("|---|------|------|")
        for i, r in enumerate(top1_miss, 1):
            lines.append(f"| {i} | {r.get('question','')} | {r.get('reason','')} |")
    else:
        lines.append("无 Top1 未命中案例。")

    lines += ["", "## 每题详情", "", "| # | Question | Top1 | Top3 | Top5 | Answer | Reason |",
              "|---|----------|------|------|------|--------|--------|"]
    for i, r in enumerate(valid, 1):
        def _v(r, k):
            return "✓" if r.get(k) else "✗"
        lines.append(
            f"| {i} | {r.get('question','')[:40]} | {_v(r,'retrieval_top1_hit')} | "
            f"{_v(r,'retrieval_top3_hit')} | {_v(r,'retrieval_top5_hit')} | "
            f"{_v(r,'answer_correct')} | {r.get('reason','')} |"
        )
    return "\n".join(lines)


def _compute_subset_metrics(results, has_ref_filter):
    """计算指定子集的指标。has_ref_filter: True=仅有参考答案, False=仅无参考答案, None=全部。

    与 compute_metrics() 口径一致：has_reference 缺失时视为 False（无参考答案）。
    """
    if has_ref_filter is None:
        subset = [r for r in results if "error" not in r]
    else:
        subset = [r for r in results if "error" not in r and bool(r.get("has_reference")) == has_ref_filter]
    n = len(subset)
    if n == 0:
        return None
    return {
        "count": n,
        "top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in subset) / n,
        "top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in subset) / n,
        "top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in subset) / n,
        "answer_correct_rate": sum(r.get("answer_correct", 0) for r in subset) / n,
    }


# ---------- 评测详情渲染（Judge 页与运行看板共享） ----------

def render_retrieval_result_detail(result: dict, sample: dict, key_prefix: str = ""):
    """渲染单条检索评测详情。Judge 页和运行看板共用。"""
    _tid = result.get("trace_id", "")
    _q = result.get("question", "(无问题)")
    _t1 = result.get("retrieval_top1_hit")
    _t3 = result.get("retrieval_top3_hit")
    _t5 = result.get("retrieval_top5_hit")
    _hit_pos = result.get("hit_evidence_position")

    _result_status = build_result_status(result)
    _icon = _result_status["icon"]
    _title = _result_status["title"]

    _sample = sample or {}
    _has_sample = bool(_sample)

    with st.expander(f"{_icon} {_q[:45]}{'...' if len(_q) > 45 else ''} ｜{_title}"):
        # 1. 问题
        st.markdown(f"**问题**: {_q}")

        # 2. 金标准证据
        _gold = (_sample.get("source_excerpt") or "").strip()
        if not _gold:
            _gold = (_sample.get("reference_answer") or "").strip()
        if _gold:
            st.markdown("**金标准证据**")
            st.code(_gold[:1000], language=None)
        elif not _has_sample:
            st.caption("未找到关联样本，无法显示金标准证据")

        # 3. 实际检索结果（TopK）
        _retrieval_results = _sample.get("retrieval_results") or []
        if _retrieval_results:
            st.markdown("**实际检索结果**")
            for _rr in sorted(_retrieval_results, key=lambda x: x.get("position", 999)):
                _pos = _rr.get("position", "?")
                _score = _rr.get("score")
                _doc_name = _rr.get("document_name") or ""
                _content = (_rr.get("content") or "")[:300]
                _is_hit = (_hit_pos is not None and _pos == _hit_pos)

                _pos_label = f"Top{_pos}"
                _score_label = f"(score: {_score:.4f})" if _score is not None else ""
                _hit_label = " **命中金标准证据**" if _is_hit else ""

                with st.expander(f"{_pos_label} {_doc_name} {_score_label}{_hit_label}", expanded=_is_hit):
                    if _is_hit:
                        st.success("命中金标准证据")
                    st.caption(f"文档: {_doc_name}" if _doc_name else "")
                    st.code(_content, language=None)
                    if len(_rr.get("content") or "") > 300:
                        with st.expander("展开完整内容"):
                            st.text(_rr.get("content", ""))

        # 4. Top1/Top3/Top5 判定与 Judge 原因
        st.markdown("**检索命中判定**")
        st.markdown(f"Top1 {'✓ 命中' if _t1 else '✗ 未命中'} | Top3 {'✓ 命中' if _t3 else '✗ 未命中'} | Top5 {'✓ 命中' if _t5 else '✗ 未命中'}")
        st.markdown(f"**Judge 原因**: {result.get('reason', '(无)')}")

        # 5. 最终回答（辅助参考）
        if _has_sample:
            _final = _sample.get("final_answer") or "(无)"
            st.markdown("**最终回答（辅助参考）**")
            st.code(_final[:500], language=None)

        # 错误信息
        if result.get("error"):
            st.error(f"评测错误: {result['error']}")

        st.caption(f"trace_id: `{_tid}`")


def render_strict_qa_result_detail(result: dict, sample: dict, key_prefix: str = ""):
    """渲染单条严格问答详情。Judge 页和运行看板共用。"""
    _tid = result.get("trace_id", "")
    _q = result.get("question", "(无问题)")

    _result_status = build_result_status(result)
    _icon = _result_status["icon"]
    _title = _result_status["title"]

    _sample = sample or {}
    _has_sample = bool(_sample)

    with st.expander(f"{_icon} {_q[:50]}{'...' if len(_q) > 50 else ''} ｜{_title}"):
        st.markdown(f"**问题**: {_q}")

        # 最终回答
        _final = _sample.get("final_answer") or "(无)" if _has_sample else "(未找到关联样本)"
        st.markdown("**最终回答**")
        st.code(_final[:1500], language=None)

        # 参考答案
        _ref = (_sample.get("reference_answer") or "").strip() if _has_sample else ""
        if _ref:
            st.markdown("**参考答案**")
            st.code(_ref[:1500], language=None)

        # 检索诊断（辅助）
        _has_excerpt = bool((_sample.get("source_excerpt") or "").strip()) if _has_sample else False
        _has_topk = (result.get("retrieval_top1_hit") is not None
                     or result.get("retrieval_top3_hit") is not None
                     or result.get("retrieval_top5_hit") is not None)
        if _has_excerpt and _has_topk:
            with st.expander("检索诊断（辅助）", expanded=False):
                st.caption("辅助诊断，不计入严格回答正确率；用于定位回答错误是否由检索失败造成。")
                _t1 = result.get("retrieval_top1_hit")
                _t3 = result.get("retrieval_top3_hit")
                _t5 = result.get("retrieval_top5_hit")
                _hit_pos = result.get("hit_evidence_position")

                st.markdown("**TopK 命中状态**")
                st.markdown(f"Top1 {'✓ 命中' if _t1 else '✗ 未命中'} | Top3 {'✓ 命中' if _t3 else '✗ 未命中'} | Top5 {'✓ 命中' if _t5 else '✗ 未命中'}")

                _retrieval_results = _sample.get("retrieval_results") or []
                if _retrieval_results:
                    st.markdown("**实际检索结果**")
                    for _rr in sorted(_retrieval_results, key=lambda x: x.get("position", 999)):
                        _pos = _rr.get("position", "?")
                        _score = _rr.get("score")
                        _doc_name = _rr.get("document_name") or ""
                        _content = (_rr.get("content") or "")[:300]
                        _is_hit = (_hit_pos is not None and _pos == _hit_pos)

                        _pos_label = f"Top{_pos}"
                        _score_label = f"(score: {_score:.4f})" if _score is not None else ""
                        _hit_label = " **命中金标准证据**" if _is_hit else ""

                        with st.expander(f"{_pos_label} {_doc_name} {_score_label}{_hit_label}", expanded=_is_hit):
                            if _is_hit:
                                st.success("命中金标准证据")
                            st.caption(f"文档: {_doc_name}" if _doc_name else "")
                            st.code(_content, language=None)
                            if len(_rr.get("content") or "") > 300:
                                with st.expander("展开完整内容"):
                                    st.text(_rr.get("content", ""))

                _gold = (_sample.get("source_excerpt") or "").strip()
                if _gold:
                    st.markdown("**金标准证据**")
                    st.code(_gold[:500], language=None)

        st.markdown(f"**Judge 原因**: {result.get('reason', '(无)')}")

        if result.get("error"):
            st.error(f"评测错误: {result['error']}")

        st.caption(f"trace_id: `{_tid}`")


def render_grounded_qa_result_detail(result: dict, sample: dict, key_prefix: str = ""):
    """渲染单条合理性问答详情。Judge 页和运行看板共用。"""
    _tid = result.get("trace_id", "")
    _q = result.get("question", "(无问题)")

    _result_status = build_result_status(result)
    _icon = _result_status["icon"]
    _title = _result_status["title"]

    _sample = sample or {}
    _has_sample = bool(_sample)

    with st.expander(f"{_icon} {_q[:50]}{'...' if len(_q) > 50 else ''} ｜{_title}"):
        st.markdown(f"**问题**: {_q}")

        _final = _sample.get("final_answer") or "(无)" if _has_sample else "(未找到关联样本)"
        st.markdown("**最终回答**")
        st.code(_final[:1500], language=None)

        st.markdown(f"**Judge 原因**: {result.get('reason', '(无)')}")

        if result.get("error"):
            st.error(f"评测错误: {result['error']}")

        st.caption(f"trace_id: `{_tid}`")


def render_judge_result_detail(result: dict, sample: dict, key_prefix: str = ""):
    """根据 evaluation_track 分派到对应的详情渲染函数。"""
    track = result.get("evaluation_track", "")
    if track == TRACK_RETRIEVAL:
        render_retrieval_result_detail(result, sample, key_prefix)
    elif track == TRACK_STRICT_QA:
        render_strict_qa_result_detail(result, sample, key_prefix)
    elif track == TRACK_GROUNDED_QA:
        render_grounded_qa_result_detail(result, sample, key_prefix)
    else:
        # 通用回退：显示基本信息
        _q = result.get("question", "(无问题)")
        _tid = result.get("trace_id", "")
        with st.expander(f"❓ {_q[:50]} ｜{track or '未知'}"):
            st.markdown(f"**问题**: {_q}")
            if result.get("error"):
                st.error(f"评测错误: {result['error']}")
            st.markdown(f"**Judge 原因**: {result.get('reason', '(无)')}")
            st.caption(f"trace_id: `{_tid}` | evaluation_track: {track}")


def render_judge_results_list(results: list, sample_map: dict, key_prefix: str = "jr",
                               page_size: int = 20):
    """渲染 Judge 结果详情列表，带筛选和分页。Judge 页和运行看板共用。

    Args:
        results: 当前 run 的 Judge result 列表
        sample_map: {trace_id: sample_dict} 映射
        key_prefix: Streamlit widget key 前缀
        page_size: 每页渲染数量
    """
    if not results:
        st.info("暂无评测结果")
        return

    from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA

    # 筛选控件
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        track_options = ["全部"]
        tracks_present = set(r.get("evaluation_track", "") for r in results)
        if TRACK_RETRIEVAL in tracks_present:
            track_options.append("retrieval")
        if TRACK_STRICT_QA in tracks_present:
            track_options.append("strict_qa")
        if TRACK_GROUNDED_QA in tracks_present:
            track_options.append("grounded_qa")
        filter_track = st.selectbox("按评测轨道筛选", track_options, key=f"{key_prefix}_track")
    with filter_col2:
        filter_status = st.selectbox(
            "按结果状态筛选", ["全部", "命中/正确", "未命中/错误", "错误"],
            key=f"{key_prefix}_status",
        )
    with filter_col3:
        st.markdown("")
        st.markdown("")

    # 应用筛选
    filtered = list(results)
    if filter_track != "全部":
        filtered = [r for r in filtered if r.get("evaluation_track") == filter_track]
    if filter_status == "命中/正确":
        filtered = [r for r in filtered if "error" not in r and (
            r.get("retrieval_top1_hit") or r.get("answer_correct"))]
    elif filter_status == "未命中/错误":
        filtered = [r for r in filtered if "error" not in r and (
            not r.get("retrieval_top1_hit") and not r.get("answer_correct"))]
    elif filter_status == "错误":
        filtered = [r for r in filtered if "error" in r]

    st.caption(f"筛选后 {len(filtered)} 条结果（共 {len(results)} 条）")

    if not filtered:
        st.info("无匹配的评测结果")
        return

    # 分页
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if total_pages > 1:
        page_col1, page_col2, _ = st.columns([1, 1, 4])
        with page_col1:
            page_num = st.number_input(
                "页码", min_value=1, max_value=total_pages, value=1, key=f"{key_prefix}_page",
            )
        with page_col2:
            st.markdown("")
            st.caption(f"共 {total_pages} 页")
        start_idx = (page_num - 1) * page_size
        page_results = filtered[start_idx:start_idx + page_size]
    else:
        page_results = filtered

    # 渲染每条详情
    for r in page_results:
        tid = r.get("trace_id", "")
        sample = sample_map.get(tid, {})
        render_judge_result_detail(r, sample, key_prefix)


st.set_page_config(page_title="Langfuse RAG 评测工具", layout="wide")
st.title("Langfuse RAG 评测工具")

# --- Sidebar ---
st.sidebar.markdown(
    "RAG 检索 + 回答质量评测工具。"
    "从知识库生成题目，通过 Dify 批量提问，解析为结构化样本后用 LLM Judge 自动评分。"
    "运行看板按配置方案汇总累计指标、运行历史和单次运行详情。"
)
st.sidebar.divider()
st.sidebar.markdown("**四步工作流**")
st.sidebar.markdown(
    "1. **题目生成** — 上传知识库文件，自动按章节切分后调用 LLM 出题，"
    "生成带参考答案的评测题集\n"
    "2. **批量提问** — 选择题集和 RAG 配置方案，通过 Dify Workflow API 批量提问，"
    "收集回答与检索结果\n"
    "3. **样本准备** — 解析 Dify / Langfuse 记录为结构化样本，回填参考答案和运行元数据\n"
    "4. **Judge 评测** — 按评测轨道自动评分：检索评测关注 Top1/3/5 命中，"
    "问答评测关注回答正确性/合理性"
)
st.sidebar.divider()
st.sidebar.markdown("**运行看板** — 按配置方案查看累计结果、运行历史和单次运行详情")
st.sidebar.caption("切换上方 Tab 进入对应工作区，每个 Tab 内均有独立配置面板和详细说明。")

# Load existing samples if available
if "samples" not in st.session_state:
    samples_file = PROCESSED_DIR / "langfuse_samples.jsonl"
    summary_file = PROCESSED_DIR / "langfuse_summary.json"
    if samples_file.exists():
        with open(samples_file, "r", encoding="utf-8") as f:
            st.session_state["samples"] = [json.loads(line) for line in f if line.strip()]
    if summary_file.exists():
        st.session_state["summary"] = json.loads(summary_file.read_text(encoding="utf-8"))

samples = st.session_state.get("samples")
summary = st.session_state.get("summary") or {}

# --- Tabs ---
tab_qgen, tab_batch, tab_samples, tab_judge, tab_experiment = st.tabs(["题目生成", "批量提问", "样本准备", "Judge 评测", "运行看板"])

# ========== Tab: 题目生成 ==========
with tab_qgen:
    st.subheader("题目生成")
    st.caption("上传知识库文件，调用 LLM 自动生成测评题目")

    # ---------- 模块说明 ----------
    with st.expander("题目生成模块说明（点击展开）", expanded=False):
        st.markdown("""
**一句话总览：** 上传知识库文件，自动按章节切分后调用 LLM 生成带参考答案的测评题目，为后续严格评测提供标准答案。

---

**两种出题模式**

| 模式 | 适用场景 | 题目特点 | 评测目标 |
|------|---------|---------|---------|
| **检索评测（单跳）** | 测试 RAG 系统的检索能力 | 单知识点、单证据片段可回答；定义题、枚举题、事实题 | Top1/Top3/Top5 命中率 |
| **全流程问答评测** | 测试完整问答能力 | 可包含对比题、分析题、推理题 | 回答质量、严格评测 |

**如何选择？**
- 如果你主要想测试 RAG 系统能否检索到正确的内容 → 选「检索评测」
- 如果你想测试从检索到回答的全流程质量 → 选「全流程问答评测」

**检索评测模式明确禁止的题型：**
- ❌ 对比题 / 区别题（如"A 和 B 的区别"）
- ❌ 优缺点分析题
- ❌ 原因分析题
- ❌ 影响/意义题

---

**输入是什么？**

| 输入 | 说明 |
|------|------|
| 知识库文件 | .txt 或 .md 格式的知识库文档 |
| 出题模式 | 检索评测 / 全流程问答评测 |
| 生成数量 | 期望生成的题目数量 |
| 难度偏好 | 基础概念题 / 理解题 / 综合题（检索模式无综合题） / 混合 |
| 生成策略 | 自动 / 极速 / 标准 / 深度（区别在于文档切分粒度和 LLM 调用次数） |

---

**实际做什么？**

1. **文档切分** — 将知识库文件按章节/段落切分为多个 chunk
2. **逐 chunk 出题** — 对每个 chunk 调用 LLM 生成问题（根据模式使用不同 prompt）
3. **去重** — 自动去除相似度过高的重复题目
4. **保存** — 将题目列表保存为 JSONL 文件

---

**输出哪些字段？**

每道题目包含以下字段，这些字段会沿着整个评测链路传递：

| 字段 | 说明 | 后续用途 |
|------|------|---------|
| `question` | 问题文本 | 批量提问的输入，Judge 评测的问题 |
| `reference_answer` | 参考答案 | **严格评测的核心依据**，Judge 据此判断回答正确性 |
| `source_excerpt` | 来源摘录 | 参考答案对应的原文片段，辅助 Judge 理解上下文 |
| `difficulty` | 难度标签 | 分析不同难度题目的表现差异 |
| `topic` | 主题标签 | 分析不同主题的检索和回答质量 |

---

**输出到哪里？**

| 输出 | 路径 | 用途 |
|------|------|------|
| 题目文件 | `data/questions/questions_<时间戳>.jsonl` | 批量提问的输入，也是参考答案回填的来源 |

---

**为什么它对后续严格评测重要？**

- `reference_answer` 是严格评测的基准 — 没有它，Judge 只能做"合理性评测"（靠 LLM 自行判断对错）
- 题目库同时也是「样本准备」回填参考答案的来源 — 从 Langfuse 解析的样本如果没有 reference_answer，会尝试从题目库中匹配
- 如果跳过这一步直接用其他来源的问题，后续大概率只能走无参考答案评测
""")

    # --- Config section (collapsible) ---
    with st.expander("配置", expanded=True):
        qgen_uploaded = st.file_uploader("上传知识库文件", type=["txt", "md"], key="qgen_upload")

        # 出题模式选择（放在最显眼的位置）
        qgen_mode_selection = st.radio(
            "出题模式",
            ["检索评测", "全流程问答评测"],
            index=0,
            key="qgen_mode_selection",
            horizontal=True,
            help="检索评测：生成适合测试 RAG 检索命中率的题目；全流程问答评测：生成适合完整问答能力测试的题目"
        )
        mode_val = MODE_RETRIEVAL if qgen_mode_selection == "检索评测" else MODE_QA

        # 模式说明
        if mode_val == MODE_RETRIEVAL:
            st.info("""🔍 **检索评测模式（单跳检索）**

生成专注于测试 RAG 系统检索能力的题目（Top1/Top3/Top5 命中率）。

**题目特点：**
- ✅ 单知识点、单证据片段可回答
- ✅ 优先生成：定义题、枚举题、单点事实题、单概念解释题
- ❌ 明确禁止：对比题、区别题、优缺点分析、原因分析、影响题

**评测目标：** 验证检索是否命中正确的 chunk，而非测试问答质量""")
        else:
            st.info("💬 **全流程问答评测模式**：生成的题目将用于测试完整问答能力。题目特点：可包含综合分析题、对比题、推理题，适合后续 Judge 严格评测。")

        # 题集名称
        if qgen_uploaded:
            _default_set_name = build_question_set_name(qgen_uploaded.name, mode_val)
        else:
            _default_set_name = ""
        qgen_set_name = st.text_input(
            "题集名称",
            value=_default_set_name,
            placeholder="例如：IS5010期末复习_检索评测",
            key="qgen_set_name_input",
            help="用于标识这一套题，默认由文件名和出题模式生成"
        )

        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        with cfg_col1:
            qgen_num = st.select_slider("生成题目数量", options=[5, 10, 15, 20], value=10, key="qgen_num")
        with cfg_col2:
            # 检索模式下只提供"事实"和"基础"两个难度
            if mode_val == MODE_RETRIEVAL:
                difficulty_options = ["混合", "基础概念题"]
                difficulty_help = "检索测试模式仅支持「事实」和「基础」两个难度级别"
            else:
                difficulty_options = ["混合", "基础概念题", "理解题", "综合题"]
                difficulty_help = None
            qgen_difficulty = st.selectbox(
                "难度偏好", difficulty_options, index=0, key="qgen_diff",
                help=difficulty_help
            )
        with cfg_col3:
            qgen_topic_hint = st.text_input("主题提示（可选）", placeholder="如：金融科技基础概念", key="qgen_topic")
        with cfg_col4:
            qgen_strategy = st.selectbox(
                "生成策略",
                ["自动", "极速", "标准", "深度"],
                index=0,
                key="qgen_strategy",
            )
            st.caption("四种策略的区别在于文档切分粒度和 LLM 调用次数")

        with st.expander("策略说明", expanded=False):
            st.markdown("""
| 模式 | 切分方式 | LLM 调用次数 | 适合场景 |
|------|---------|-------------|---------|
| **极速** | 单 chunk：取前 3 个 markdown section，或截取前 6000 字 | 1 次 | 快速预览、短文档 |
| **标准** | chunk_document(max_chars=6000, max_chunks=5) | 3~5 次 | 日常使用，平衡速度与覆盖 |
| **深度** | chunk_document(max_chars=3000, max_chunks=20) | 最多 20 次 | 正式评测，覆盖完整 |
| **自动** | 根据文档字符数和 section 数自动选择上述三种 | 取决于文档 | 不确定时选这个 |
""")
            st.markdown("**自动模式的选择规则：**")
            st.code(
                "字符数 < 3,000 → 极速\n"
                "3,000 ≤ 字符数 < 15,000 且 section ≤ 3 → 极速\n"
                "3,000 ≤ 字符数 < 15,000 且 section > 3 → 标准\n"
                "15,000 ≤ 字符数 ≤ 50,000 → 标准\n"
                "字符数 > 50,000 → 深度"
            )

        with st.expander("API 配置", expanded=False):
            api_col1, api_col2, api_col3 = st.columns(3)
            with api_col1:
                qgen_api_key = st.text_input(
                    "API Key", type="password",
                    value=os.getenv("QGEN_API_KEY") or os.getenv("JUDGE_API_KEY", ""),
                    key="qgen_api_key",
                )
            with api_col2:
                qgen_base_url = st.text_input(
                    "Base URL",
                    value=os.getenv("QGEN_API_BASE") or os.getenv("JUDGE_API_BASE", "https://token-plan-cn.xiaomimimo.com/v1"),
                    key="qgen_base_url",
                )
            with api_col3:
                qgen_model = st.text_input(
                    "Model",
                    value=os.getenv("QGEN_MODEL") or os.getenv("JUDGE_MODEL", "mimo-v2.5-pro"),
                    key="qgen_model",
                )
            if st.button("测试连接", key="qgen_test_conn"):
                if not qgen_api_key:
                    st.error("请先输入 API Key")
                else:
                    with st.status("正在测试连接...", expanded=True) as status:
                        try:
                            resp = call_llm('请只输出 JSON：{"ok": true}', qgen_api_key, qgen_base_url, qgen_model, timeout=15)
                            status.update(label="连接成功", state="complete")
                            st.code(resp[:200])
                        except Exception as e:
                            status.update(label="连接失败", state="error")
                            st.error(str(e))

    # --- File preview ---
    if qgen_uploaded is not None:
        file_bytes = qgen_uploaded.getvalue()
        try:
            file_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            file_content = file_bytes.decode("gbk", errors="replace")

        file_size_kb = len(file_bytes) / 1024
        char_count = len(file_content)

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("文件名", qgen_uploaded.name)
        info_col2.metric("文件大小", f"{file_size_kb:.1f} KB")
        info_col3.metric("字符数", f"{char_count:,}")

        with st.expander("文件内容预览", expanded=False):
            preview_len = 500
            if char_count > preview_len:
                st.text(file_content[:preview_len] + "...")
                st.caption(f"（显示前 {preview_len} 字，共 {char_count:,} 字）")
            else:
                st.text(file_content)

        if char_count > 8000:
            st.info(f"文件较长（{char_count:,} 字），建议使用「标准」或「深度」策略以确保内容覆盖完整。")

        # --- Auto-mode analysis (show when strategy is auto) ---
        if qgen_strategy == "自动":
            from question_generator import _split_markdown_sections
            sections = _split_markdown_sections(file_content)
            is_plain = len(sections) == 1 and sections[0][0] == "(前言)"
            section_count = 0 if is_plain else len(sections)
            predicted = choose_strategy(file_content)

            # Determine reasoning
            if char_count < 3000:
                reason = f"字符数 {char_count:,} < 3,000，文档很短，1 次调用即可覆盖"
            elif char_count < 15000:
                if section_count <= 3:
                    reason = f"字符数 {char_count:,}，识别到 {section_count} 个 section（≤3），结构简单，选极速"
                else:
                    reason = f"字符数 {char_count:,}，识别到 {section_count} 个 section（>3），有结构，选标准以适度覆盖"
            elif char_count <= 50000:
                reason = f"字符数 {char_count:,}，中等长度文档，选标准平衡速度和覆盖"
            else:
                reason = f"字符数 {char_count:,} > 50,000，长文档需要完整覆盖，选深度"

            with st.container(border=True):
                st.markdown(f"**自动模式分析** → 将使用「{STRATEGY_LABELS[predicted]}」策略")
                acol1, acol2, acol3 = st.columns(3)
                acol1.metric("字符数", f"{char_count:,}")
                acol2.metric("Section 数", f"{section_count}" if not is_plain else "无（纯文本）")
                acol3.metric("判定结果", STRATEGY_LABELS[predicted])
                st.caption(f"判断依据：{reason}")

        # --- Prompt 示例 ---
        with st.expander("Prompt 示例（点击展开）", expanded=False):
            from question_generator import (
                load_qgen_prompt_template, chunk_document, allocate_questions,
                choose_strategy, _BALANCED_MAX_CHARS, _BALANCED_MAX_CHUNKS,
                MAX_CHUNK_CHARS, MAX_CHUNKS, _FAST_MAX_CHARS,
            )
            _qgen_template = load_qgen_prompt_template(mode_val)

            # 显示当前模式
            if mode_val == MODE_RETRIEVAL:
                st.markdown("**当前模式：检索评测** — Prompt 侧重生成具体、可定位、答案明确的检索测试题目")
            else:
                st.markdown("**当前模式：全流程问答评测** — Prompt 侧重生成适合完整问答能力测试的题目")

            # 构造示例参数
            _difficulty_map = {"混合": "混合", "基础概念题": "基础", "理解题": "理解", "综合题": "综合"}
            _qgen_diff_val = _difficulty_map.get(qgen_difficulty, "混合")

            _topic_hint_section = ""
            if qgen_topic_hint:
                _topic_hint_section = f"- 主题方向：{qgen_topic_hint}"

            # --- 运行机制说明 ---
            st.markdown("""
**题目生成的真实运行流程：**

1. **切分文档** — 将整篇知识库文件按章节/段落切分为多个 chunk
2. **分配题目数** — 将总题目数按 chunk 长度比例分配（每个 chunk 至少 1 题）
3. **逐 chunk 调用 LLM** — 每个 chunk 单独调用一次 LLM 生成其分配到的题目
4. **去重汇总** — 去除重复题目，按多样性裁剪到目标数量

因此，"题目数量 10"是整篇文档的总目标，不是单个 chunk 要出 10 道。
""")

            # --- chunk 分配预览 ---
            if qgen_uploaded is not None:
                strategy_map = {"自动": "auto", "极速": "fast", "标准": "balanced", "深度": "deep"}
                _strategy_val = strategy_map.get(qgen_strategy, "auto")

                # 根据策略选择切分参数
                if _strategy_val == "auto":
                    _predicted = choose_strategy(file_content)
                    _strategy_val = _predicted
                    _strategy_name = f"自动 → {STRATEGY_LABELS[_predicted]}"
                else:
                    _strategy_name = qgen_strategy

                if _strategy_val == "fast":
                    _chunks = []  # fast 模式不真正切分，只有 1 个 chunk
                    _chunk_count = 1
                    _alloc = [qgen_num]
                elif _strategy_val == "balanced":
                    _chunks = chunk_document(file_content, max_chars=_BALANCED_MAX_CHARS, max_chunks=_BALANCED_MAX_CHUNKS)
                    _chunk_count = len(_chunks)
                    _alloc = allocate_questions(_chunks, qgen_num)
                else:  # deep
                    _chunks = chunk_document(file_content, max_chars=MAX_CHUNK_CHARS, max_chunks=MAX_CHUNKS)
                    _chunk_count = len(_chunks)
                    _alloc = allocate_questions(_chunks, qgen_num)

                st.markdown(f"**当前文档切分预览**（策略：{_strategy_name}）：")
                _alloc_display = [f"chunk {i+1}: {n} 题" for i, n in enumerate(_alloc[:10])]
                if len(_alloc) > 10:
                    _alloc_display.append(f"...共 {_chunk_count} 个 chunk")
                st.caption(" → ".join(_alloc_display))

                # 展示前 3 个 chunk 的 prompt 示例
                if _chunks:
                    _preview_count = min(3, len(_chunks))
                    st.markdown(f"**Prompt 示例**（前 {_preview_count} 个 chunk，与真实执行完全一致）：")
                    for _pi in range(_preview_count):
                        _pc = _chunks[_pi]
                        _pa = _alloc[_pi]
                        _pc_len = _pc["char_count"]
                        _pc_context = f"\n当前章节：「{_pc['section_title']}」"
                        with st.expander(
                            f"chunk {_pi+1} — 「{_pc['section_title'][:30]}」"
                            f"（{_pc_len:,} 字 | 分配 {_pa} 题）",
                            expanded=(_pi == 0),
                        ):
                            _p = _qgen_template
                            _p = _p.replace("{content}", _pc["text"])
                            _p = _p.replace("{num_questions}", str(_pa))
                            _p = _p.replace("{difficulty}", _qgen_diff_val)
                            _p = _p.replace("{topic_hint_section}", _topic_hint_section)
                            _p = _p.replace("{section_context}", _pc_context)
                            if _pa <= 1:
                                _cov = "- 当前片段只需生成 1 道题，请聚焦于该片段中最核心、最有考查价值的知识点"
                            else:
                                _cov = f"- 当前片段需生成 {_pa} 道题，如果涉及多个知识点，尽量覆盖不同知识点出题"
                            _p = _p.replace("{coverage_instruction}", _cov)
                            st.code(_p, language=None)
                            st.caption(f"prompt 长度：{len(_p)} 字符（含 {_pc_len:,} 字 chunk 内容）")
                    if len(_chunks) > 3:
                        st.caption(f"...还有 {len(_chunks) - 3} 个 chunk，结构相同，每个 chunk 独立调用 LLM")
                else:
                    # fast 模式：单个 prompt
                    st.markdown("**Prompt 示例**（极速模式，整个文档前部）：")
                    _p = _qgen_template
                    _p = _p.replace("{content}", file_content[:800] + ("..." if len(file_content) > 800 else ""))
                    _p = _p.replace("{num_questions}", str(qgen_num))
                    _p = _p.replace("{difficulty}", _qgen_diff_val)
                    _p = _p.replace("{topic_hint_section}", _topic_hint_section)
                    _p = _p.replace("{section_context}", "\n当前章节：「文档前部」")
                    if qgen_num <= 1:
                        _p = _p.replace("{coverage_instruction}", "- 当前片段只需生成 1 道题，请聚焦于该片段中最核心、最有考查价值的知识点")
                    else:
                        _p = _p.replace("{coverage_instruction}", f"- 当前片段需生成 {qgen_num} 道题，如果涉及多个知识点，尽量覆盖不同知识点出题")
                    st.code(_p, language=None)
                    st.caption(f"prompt 长度：{len(_p)} 字符")
            else:
                st.markdown("**Prompt 模板结构**（上传文件后将展示真实 chunk 分配）：")
                _p = _qgen_template
                _p = _p.replace("{content}", "（上传知识库文件后，此处将展示实际文档内容片段）")
                _p = _p.replace("{num_questions}", str(qgen_num))
                _p = _p.replace("{difficulty}", _qgen_diff_val)
                _p = _p.replace("{topic_hint_section}", _topic_hint_section)
                _p = _p.replace("{section_context}", "")
                _p = _p.replace("{coverage_instruction}", f"- 当前片段需生成 {qgen_num} 道题，如果涉及多个知识点，尽量覆盖不同知识点出题")
                st.code(_p, language=None)
                st.caption(f"prompt 模板长度：{len(_p)} 字符")

        # --- Generate button ---
        if st.button("生成题目", type="primary", key="qgen_run", use_container_width=True):
            if not qgen_api_key:
                st.error("请在上方「API 配置」中输入 API Key")
            else:
                difficulty_map = {
                    "混合": "混合",
                    "基础概念题": "基础",
                    "理解题": "理解",
                    "综合题": "综合",
                }
                difficulty_val = difficulty_map.get(qgen_difficulty, "混合")
                strategy_map = {"自动": "auto", "极速": "fast", "标准": "balanced", "深度": "deep"}
                strategy_val = strategy_map.get(qgen_strategy, "auto")

                # 自动模式下先预测策略，显示给用户
                if strategy_val == "auto":
                    predicted = choose_strategy(file_content)
                    strategy_label = f"自动 → {STRATEGY_LABELS[predicted]}"
                else:
                    strategy_label = STRATEGY_LABELS[strategy_val]

                mode_label = MODE_LABELS[mode_val]
                with st.status(f"正在生成题目（{mode_label} | {strategy_label}模式）...", expanded=True) as gen_status:
                    status_text = st.empty()
                    status_text.write("正在切分文档...")

                    def _on_progress(chunk_idx, total_chunks, section_title):
                        status_text.write(
                            f"正在出题: 章节 {chunk_idx + 1}/{total_chunks} — {section_title[:40]}"
                        )

                    try:
                        questions = generate_questions(
                            file_content, qgen_api_key, qgen_base_url, qgen_model,
                            num_questions=qgen_num, difficulty=difficulty_val,
                            topic_hint=qgen_topic_hint,
                            progress_callback=_on_progress,
                            strategy=strategy_val,
                            mode=mode_val,
                        )
                        st.session_state["generated_questions"] = questions
                        st.session_state["qgen_last_generated_mode"] = mode_val  # 保存当前模式

                        # 获取题集名称（从 widget 读取）
                        _set_name = st.session_state.get("qgen_set_name_input", "") or \
                                    build_question_set_name(qgen_uploaded.name, mode_val)

                        # 保存到文件
                        output_path, fname, set_id = save_questions(
                            questions,
                            question_set_name=_set_name,
                            source_document_name=qgen_uploaded.name,
                            question_mode=mode_val,
                        )
                        st.session_state["qgen_saved_file"] = fname
                        st.session_state["qgen_saved_path"] = str(output_path)
                        st.session_state["qgen_set_id"] = set_id
                        st.session_state["qgen_generated_set_name"] = _set_name

                        # 验证文件是否保存成功
                        if output_path.exists():
                            file_size = output_path.stat().st_size
                            gen_status.update(
                                label=f"生成完成！共 {len(questions)} 道题目（{mode_label}）",
                                state="complete",
                                expanded=False,
                            )
                            st.success(f"✅ 题目已自动保存到：`data/questions/{fname}`（{file_size} 字节）")
                            st.caption(f"题集 ID: `{set_id}` | 题集名称: `{_set_name}`")
                        else:
                            gen_status.update(label="生成完成但保存失败", state="error")
                            st.error(f"题目生成成功，但文件保存失败：{output_path}")
                    except Exception as e:
                        gen_status.update(label="生成失败", state="error")
                        st.error(f"生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.info("请在上方「配置」区域上传知识库文件（.txt 或 .md）")

    # --- Results display ---
    questions = st.session_state.get("generated_questions")
    if questions:
        st.divider()
        st.subheader(f"生成结果（{len(questions)} 道题目）")

        # 显示保存状态
        saved_path = st.session_state.get("qgen_saved_path", "")
        if saved_path:
            st.success(f"✅ 题目已自动保存到：`{saved_path}`")

        diff_counts = {}
        for item in questions:
            d = item.get("difficulty", "未知")
            diff_counts[d] = diff_counts.get(d, 0) + 1
        mcols = st.columns(max(len(diff_counts), 1))
        for i, (d, c) in enumerate(diff_counts.items()):
            mcols[i].metric(d, c)

        df_q = pd.DataFrame(questions)
        df_q.index = range(1, len(df_q) + 1)
        df_q.index.name = "#"
        st.dataframe(
            df_q[["question", "difficulty", "topic"]],
            use_container_width=True,
            height=min(400, len(df_q) * 40 + 60),
        )

        for i, item in enumerate(questions, 1):
            with st.expander(f"#{i} {item.get('question', '')[:60]}"):
                st.markdown(f"**问题**: {item.get('question', '')}")
                st.markdown(f"**参考答案**: {item.get('reference_answer', '')}")
                st.markdown(f"**来源摘录**: {item.get('source_excerpt', '')}")
                st.markdown(f"**难度**: {item.get('difficulty', '')} | **主题**: {item.get('topic', '')}")

        st.divider()
        st.subheader("导出")
        dl_col1, dl_col2 = st.columns(2)

        saved_file = st.session_state.get("qgen_saved_file", "questions.jsonl")
        with dl_col1:
            jsonl_data = "\n".join(
                json.dumps(q, ensure_ascii=False) for q in questions
            ).encode("utf-8")
            st.download_button(
                label="下载 JSONL",
                data=jsonl_data,
                file_name=saved_file,
                mime="application/jsonl",
            )

        with dl_col2:
            csv_data = export_csv_bytes(questions)
            st.download_button(
                label="下载 CSV",
                data=csv_data,
                file_name=saved_file.replace(".jsonl", ".csv"),
                mime="text/csv",
            )

        with st.expander("输出说明", expanded=False):
            st.markdown("""
**自动保存位置**：`data/questions/questions_<时间戳>.jsonl`

每行一道题，JSONL 格式，字段如下：

| 字段 | 说明 |
|------|------|
| `question` | 题目文本 |
| `reference_answer` | 参考答案 |
| `source_excerpt` | 来源摘录（原文片段） |
| `difficulty` | 难度：基础 / 理解 / 综合 |
| `topic` | 题目主题 |

生成完成后也可点击上方按钮下载 JSONL 或 CSV 副本。
这些题目可直接用于「批量提问」tab → 选择「使用已生成的题目」。
""")


# ========== Tab: 批量提问 ==========
with tab_batch:
    st.subheader("批量提问")
    st.caption("将题目批量发送到 Dify Q&A 接口，自动收集回答和检索结果")

    # ---------- 模块说明 ----------
    with st.expander("批量提问模块说明（点击展开）", expanded=False):
        st.markdown("""
**一句话总览：** 选择题集和 RAG 配置方案，通过 Dify Workflow API 批量提问，收集回答与检索结果，生成可直接用于评测的结构化样本。

---

**输入是什么？**

| 来源 | 说明 |
|------|------|
| 已生成的题目 | 来自「题目生成」模块，自带 reference_answer、question_set_id 等元数据 |
| 手动输入问题 | 直接输入问题文本，无参考答案 |
| 从文件加载 | 上传 JSONL / CSV / TXT 文件，按格式解析问题 |
| 从历史记录加载 | 复用之前的题集记录，按 question_set_id / 文件名区分 |

如果输入来自「题目生成」，reference_answer 和题集信息会自动透传到输出样本中。

---

**RAG 配置方案**

批量提问需要关联一个配置方案，记录用户声明的 Dify 环境参数：
- **必填**：配置名称、知识库版本、工作流版本
- **可选**：分块策略、Embedding 模型、检索模式、Top K、Rerank 模型、备注等

配置方案仅记录参数声明，本工具不直接修改 Dify 知识库、Embedding、分块或工作流节点。
新建的配置可在「运行看板」中编辑，历史配置也可补充描述性字段。

---

**实际做什么？**

1. **标准化输入** — 将各种格式的问题统一为 list[dict]，保留 reference_answer 等元数据
2. **创建运行记录** — 为本次批量提问创建 run_id，关联配置方案快照
3. **逐条调用 Dify** — 对每个问题调用 Dify Workflow API（blocking 模式），user 字段格式为 `rag_eval:<run_id>:<question_id>`
4. **收集结果** — 从 Dify 响应中提取最终回答和检索结果，组装为结构化样本

---

**收集哪些结果？**

| 字段 | 来源 | 说明 |
|------|------|------|
| `final_answer` | Dify response.answer | LLM 最终回答 |
| `retrieval_results` | Dify response.metadata.retriever_resources | 检索结果列表（含 position、score、content 等） |
| `retrieval_query` | 原始问题 | Dify 不单独返回 retrieval_query，用原始问题代替 |
| `trace_id` | 自动生成 | `batch_qa_{序号}_{时间戳}`（注意：这不是 Langfuse trace_id） |
| `reference_answer` | 透传自输入 | 如果输入有参考答案，会保留到输出样本 |
| `run_id` / `config_id` | 自动关联 | 本次运行的 run_id 和配置方案 ID |
| `question_set_id` | 透传自题集 | 用于在运行看板中关联题集 |

---

**输出到哪里？**

| 操作 | 路径 | 用途 |
|------|------|------|
| 自动保存完整结果 | `data/batch/batch_results_<时间戳>.jsonl` | 包含每条问题的原始响应、成功/失败状态 |
| 推送到样本准备 | `data/raw/batch_qa_<时间戳>.jsonl` | 仅含成功结果，格式兼容后续解析和评测 |

---

**和「样本准备」的关系**

```
本模块产出 → 推送到 data/raw/ → 样本准备解析 → Judge 评测
```

- 推送后的文件在「样本准备」tab 中可见，选择并点击「解析」即可进入评测流程
- 解析时会从 `user_id` 字段回填 `run_id`、`question_id` 等元数据
- 样本准备产出的 processed samples 使用真实 Langfuse trace_id（来自 Dify 调用 Langfuse 记录的 UUID），**不是** `batch_qa_*` 伪 trace_id
""")

    # --- Question source ---
    with st.expander("问题来源", expanded=True):
        q_source = st.radio(
            "选择问题来源",
            ["使用已生成的题目", "手动输入问题", "从文件加载", "从历史记录加载"],
            horizontal=True,
            key="batch_q_source",
        )

        with st.expander("输入文件格式说明", expanded=False):
            st.markdown("**推荐格式：JSONL**（与题目生成结果直接兼容）")
            st.markdown("""
| 格式 | 解析规则 | 示例 |
|------|---------|------|
| **JSONL** | 逐行读取，每行一个 JSON 对象；优先取 `question`，其次取 `query` | `{"question": "什么是AISP?"}` |
| **TXT** | 每行一个问题，空行自动忽略 | `什么是AISP?` |
| **CSV** | 自动检测表头（识别 `question` / `query` / `问题` 列）；无表头则读第一列 | 见下方示例 |

**CSV 示例（有表头）**：
```
question,reference_answer
什么是AISP?,AISP是账户信息服务提供商
PISP和AISP的区别?,PISP发起支付，AISP仅查看
```

**CSV 示例（无表头，直接每行一个问题）**：
```
什么是AISP?
PISP和AISP的区别?
```

> 如果只是临时测试几个问题，TXT 最方便；如果需要批量管理题目和参考答案，建议用 JSONL。
""")

        questions_list = []

        if q_source == "使用已生成的题目":
            gen_qs = st.session_state.get("generated_questions")
            if gen_qs:
                # 传递完整 question 对象（含 reference_answer / source_excerpt）
                questions_list = [q for q in gen_qs if q.get("question")]
                st.success(f"已加载 {len(questions_list)} 道已生成的题目")
                has_ref = sum(1 for q in questions_list if q.get("reference_answer"))
                if has_ref:
                    st.caption(f"其中 {has_ref} 道带有参考答案，评测时将用于严格评判")

                # 显示题目模式信息
                q_mode = questions_list[0].get("question_mode") if questions_list else ""
                if q_mode == MODE_RETRIEVAL:
                    st.info("🔍 **检索评测题**：这些题目主要用于测试 RAG 检索命中率，Judge 评测时会重点关注 Top1/Top3/Top5 Hit")
                elif q_mode == MODE_QA:
                    st.info("💬 **全流程问答题**：这些题目用于测试完整问答能力，Judge 评测时会重点关注 Answer OK")

                with st.expander("预览题目", expanded=False):
                    for i, q in enumerate(questions_list, 1):
                        qtext = q.get("question", "")
                        ref = q.get("reference_answer", "")
                        if ref:
                            st.write(f"{i}. {qtext}")
                            st.caption(f"   参考答案: {ref[:80]}{'...' if len(ref) > 80 else ''}")
                        else:
                            st.write(f"{i}. {qtext}")
            else:
                st.warning("暂无已生成的题目，请先在「题目生成」tab 中生成题目，或选择其他来源")

        elif q_source == "手动输入问题":
            manual_input = st.text_area(
                "输入问题（每行一个）",
                height=200,
                placeholder="问题1\n问题2\n问题3",
                key="batch_manual_input",
            )
            if manual_input.strip():
                questions_list = [{"question": line.strip()} for line in manual_input.strip().split("\n") if line.strip()]
                st.caption(f"已输入 {len(questions_list)} 个问题")

        elif q_source == "从文件加载":
            q_file = st.file_uploader("上传问题文件", type=["jsonl", "txt", "csv"], key="batch_q_file")
            if q_file is not None:
                content = q_file.getvalue().decode("utf-8")
                if q_file.name.endswith(".jsonl"):
                    for line in content.strip().split("\n"):
                        try:
                            obj = json.loads(line)
                            q = obj.get("question") or obj.get("query") or ""
                            if q.strip():
                                # 保留 reference_answer / source_excerpt（如有）
                                item = {"question": q.strip()}
                                if obj.get("reference_answer"):
                                    item["reference_answer"] = obj["reference_answer"]
                                if obj.get("source_excerpt"):
                                    item["source_excerpt"] = obj["source_excerpt"]
                                questions_list.append(item)
                        except json.JSONDecodeError:
                            continue
                elif q_file.name.endswith(".csv"):
                    import csv as csv_mod
                    import io
                    reader = csv_mod.reader(io.StringIO(content))
                    header = None
                    for row in reader:
                        if not row:
                            continue
                        # 检测表头行：如果首行不含常见列名，当作数据行
                        if header is None and any(
                            h.lower() in ("question", "query", "问题", "questions")
                            for h in row
                        ):
                            header = [h.lower().strip() for h in row]
                            continue
                        # 优先从 question/query 列取值，否则取第一列
                        if header:
                            q = ""
                            ref = ""
                            for i, h in enumerate(header):
                                if h in ("question", "query", "问题") and i < len(row):
                                    q = row[i]
                                if h in ("reference_answer", "参考答案") and i < len(row):
                                    ref = row[i]
                        else:
                            q = row[0] if row else ""
                            ref = ""
                        if q.strip():
                            item = {"question": q.strip()}
                            if ref.strip():
                                item["reference_answer"] = ref.strip()
                            questions_list.append(item)
                else:
                    # TXT: 每行一个问题，统一为 dict 格式
                    questions_list = [{"question": line.strip()} for line in content.strip().split("\n") if line.strip()]
                st.success(f"从文件加载了 {len(questions_list)} 个问题")

        elif q_source == "从历史记录加载":
            # Scan data/questions/ and data/batch/ for JSONL files
            history_files = []
            for d in [QUESTIONS_DIR, BATCH_DIR]:
                if d.exists():
                    for f in sorted(d.glob("*.jsonl"), reverse=True):
                        history_files.append(f)

            if not history_files:
                st.warning("暂无历史记录，请先在「题目生成」或「批量提问」中生成/保存过结果")
            else:
                # 预读每个文件，检测 question_mode、question_set_name、题目数
                def _detect_file_info(filepath):
                    """读取文件前20行，检测模式、题集名称、题目数。"""
                    info = {
                        "modes": {MODE_RETRIEVAL: 0, MODE_QA: 0, "unknown": 0},
                        "set_name": "",
                        "set_id": "",
                        "question_count": 0,
                        "has_set_info": False,
                    }
                    try:
                        with filepath.open("r", encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                line = line.strip()
                                if not line:
                                    continue
                                info["question_count"] += 1
                                if i >= 20:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    mode = obj.get("question_mode", "")
                                    if mode == MODE_RETRIEVAL:
                                        info["modes"][MODE_RETRIEVAL] += 1
                                    elif mode == MODE_QA:
                                        info["modes"][MODE_QA] += 1
                                    else:
                                        info["modes"]["unknown"] += 1

                                    # 获取题集信息
                                    if obj.get("question_set_name") and not info["set_name"]:
                                        info["set_name"] = obj["question_set_name"]
                                        info["set_id"] = obj.get("question_set_id", "")
                                        info["has_set_info"] = True
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        pass
                    return info

                # 为每个文件生成带模式标签和题集名称的显示名
                file_info_cache = {}
                file_labels = []
                for f in history_files:
                    info = _detect_file_info(f)
                    file_info_cache[f] = info
                    modes = info["modes"]
                    total_sampled = sum(modes.values())
                    q_count = info["question_count"]

                    # 生成模式标签
                    if total_sampled == 0:
                        mode_tag = "[空文件]"
                    elif modes[MODE_RETRIEVAL] > 0 and modes[MODE_QA] > 0:
                        mode_tag = "[混合]"
                    elif modes[MODE_RETRIEVAL] > 0 and modes["unknown"] == 0:
                        mode_tag = "[检索评测]"
                    elif modes[MODE_QA] > 0 and modes["unknown"] == 0:
                        mode_tag = "[全流程问答]"
                    elif modes["unknown"] > 0 and modes[MODE_RETRIEVAL] == 0 and modes[MODE_QA] == 0:
                        mode_tag = "[旧版]"
                    elif modes[MODE_RETRIEVAL] > 0:
                        mode_tag = "[检索评测+旧版]"
                    elif modes[MODE_QA] > 0:
                        mode_tag = "[全流程问答+旧版]"
                    else:
                        mode_tag = "[旧版]"

                    # 生成显示标签：优先题集名称，附带时间戳和 set_id 后缀确保唯一
                    if info["has_set_info"] and info["set_name"]:
                        # 从 set_id 提取时间戳用于区分同名题集
                        # set_id 格式: qs_YYYYMMDD_HHMMSSffffff_slug
                        _sid = info.get("set_id", "")
                        _ts_display = ""
                        if _sid:
                            _parts = _sid.split("_", 3)
                            if len(_parts) >= 3:
                                _date_part = _parts[1]  # YYYYMMDD
                                _time_part = _parts[2]  # HHMMSSffffff
                                if len(_date_part) == 8 and len(_time_part) >= 6:
                                    _ts_display = f"{_date_part[:4]}-{_date_part[4:6]}-{_date_part[6:8]} {_time_part[:2]}:{_time_part[2:4]}"
                        # 用 set_id 时间戳微秒部分做短后缀，确保同名题集可区分
                        # 取 HHMMSSffffff 中的后 8 位（含微秒）
                        _sid_short = ""
                        if _sid and len(_sid) > 20:
                            _sid_short = f" · qs...{_sid[12:20]}"
                        _ts_part = f" · {_ts_display}" if _ts_display else ""
                        label = f"{mode_tag} {info['set_name']} · {q_count} 题{_ts_part}{_sid_short}"
                    else:
                        # 旧版文件，回退显示文件名
                        label = f"{mode_tag} {f.stem} · {q_count} 题 [旧版题集]"

                    file_labels.append(label)

                selected_idx = st.selectbox(
                    "选择历史题集",
                    range(len(file_labels)),
                    format_func=lambda i: file_labels[i],
                    key="batch_history_file",
                )
                selected_file = history_files[selected_idx]
                selected_info = file_info_cache[selected_file]

                # 显示次级信息
                st.caption(f"文件: `{selected_file.name}` | 生成时间: {datetime.fromtimestamp(selected_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")

                try:
                    raw_lines = selected_file.read_text(encoding="utf-8").strip().split("\n")
                    for line in raw_lines:
                        try:
                            obj = json.loads(line)
                            q = obj.get("question") or obj.get("query") or ""
                            if q.strip():
                                item = {"question": q.strip()}
                                if obj.get("reference_answer"):
                                    item["reference_answer"] = obj["reference_answer"]
                                if obj.get("source_excerpt"):
                                    item["source_excerpt"] = obj["source_excerpt"]
                                if obj.get("question_mode"):
                                    item["question_mode"] = obj["question_mode"]
                                if obj.get("question_set_id"):
                                    item["question_set_id"] = obj["question_set_id"]
                                if obj.get("question_set_name"):
                                    item["question_set_name"] = obj["question_set_name"]
                                questions_list.append(item)
                        except json.JSONDecodeError:
                            continue

                    st.success(f"从 `{selected_file.name}` 加载了 {len(questions_list)} 个问题")

                    # 统计并展示 question_mode 分布
                    mode_counts = {MODE_RETRIEVAL: 0, MODE_QA: 0, "unknown": 0}
                    for q in questions_list:
                        qm = q.get("question_mode", "")
                        if qm == MODE_RETRIEVAL:
                            mode_counts[MODE_RETRIEVAL] += 1
                        elif qm == MODE_QA:
                            mode_counts[MODE_QA] += 1
                        else:
                            mode_counts["unknown"] += 1

                    has_ref = sum(1 for q in questions_list if q.get("reference_answer"))
                    if has_ref:
                        st.caption(f"其中 {has_ref} 道带有参考答案，评测时将用于严格评判")

                    # 显示模式统计
                    mode_info_parts = []
                    if mode_counts[MODE_RETRIEVAL] > 0:
                        mode_info_parts.append(f"检索评测题: {mode_counts[MODE_RETRIEVAL]} 道")
                    if mode_counts[MODE_QA] > 0:
                        mode_info_parts.append(f"全流程问答题: {mode_counts[MODE_QA]} 道")
                    if mode_counts["unknown"] > 0:
                        mode_info_parts.append(f"旧版/未知模式: {mode_counts['unknown']} 道")

                    if mode_info_parts:
                        st.caption("题目模式分布：" + " | ".join(mode_info_parts))

                    # 如果主要是检索评测题，给出提示
                    if mode_counts[MODE_RETRIEVAL] > 0 and mode_counts[MODE_QA] == 0:
                        st.info("🔍 **检索评测题**：Judge 评测时会重点关注 Top1/Top3/Top5 Hit")
                    elif mode_counts[MODE_QA] > 0 and mode_counts[MODE_RETRIEVAL] == 0:
                        st.info("💬 **全流程问答题**：Judge 评测时会重点关注 Answer OK")

                    with st.expander("预览题目", expanded=False):
                        for i, q in enumerate(questions_list, 1):
                            qtext = q.get("question", "")
                            ref = q.get("reference_answer", "")
                            qm = q.get("question_mode", "")
                            mode_badge = ""
                            if qm == MODE_RETRIEVAL:
                                mode_badge = "🔍 "
                            elif qm == MODE_QA:
                                mode_badge = "💬 "

                            if ref:
                                st.write(f"{mode_badge}{i}. {qtext}")
                                st.caption(f"   参考答案: {ref[:80]}{'...' if len(ref) > 80 else ''}")
                            else:
                                st.write(f"{mode_badge}{i}. {qtext}")
                except Exception as e:
                    st.error(f"读取文件失败: {e}")

    # --- RAG 配置方案 ---
    with st.expander("RAG 配置方案", expanded=False):
        from experiment import (
            create_config_profile, load_config_profile, list_config_profiles,
            create_experiment_run, update_experiment_run, ensure_question_id,
            get_config_summary, get_config_display_value,
            CONFIG_FIELD_SCHEMA,
        )

        # 配置来源选择
        config_source = st.radio(
            "配置来源",
            ["新建配置方案", "使用历史配置"],
            horizontal=True,
            key="batch_config_source",
        )

        if config_source == "使用历史配置":
            historical_configs = list_config_profiles()
            if not historical_configs:
                st.warning("暂无历史配置，请选择「新建配置方案」")
                config_source = "新建配置方案"
            else:
                config_options = []
                for cfg in historical_configs:
                    config_options.append((cfg.get("config_id"), get_config_summary(cfg)))

                selected_config_id = st.selectbox(
                    "选择历史配置",
                    options=[c[0] for c in config_options],
                    format_func=lambda x: next((c[1] for c in config_options if c[0] == x), x),
                    key="batch_selected_config",
                )

                if selected_config_id:
                    selected_config = load_config_profile(selected_config_id)
                    if selected_config:
                        st.caption(f"当前使用历史配置: **{selected_config.get('config_name', '')}**")
                        # 只读摘要（与运行看板编辑一致）
                        with st.container(border=True):
                            st.markdown("**当前配置（只读）**")
                            render_config_form(selected_config, key_prefix="ro_batch", disabled=True)
                        # 另存为新方案
                        if st.button("基于此配置另存为新方案", key="batch_save_as_new"):
                            st.session_state["batch_config_source"] = "新建配置方案"
                            for key, _, _, _, _, _ in CONFIG_FIELD_SCHEMA:
                                val = selected_config.get(key, "")
                                st.session_state[f"batch_new_{key}"] = f"{val} (副本)" if key == "config_name" else val
                            st.rerun()

        if config_source == "新建配置方案":
            st.caption("创建新的 RAG 配置方案，可在此后的多次批量测试中复用")
            # 使用统一 schema 渲染表单
            _new_config_values = render_config_form({}, key_prefix="batch_new")
            # 必填字段检查提示
            if not _new_config_values.get("config_name", "").strip():
                st.warning("建议填写配置名称，否则将使用'未命名配置'")
            if not _new_config_values.get("knowledge_base_version", "").strip():
                st.warning("建议填写知识库版本")

    # --- Dify API config ---
    with st.expander("Dify API 配置", expanded=False):
        from dify_connection import (
            list_connection_profiles, load_connection_profile,
            create_connection_profile, update_connection_profile, delete_connection_profile,
            get_connection_api_key, has_connection_api_key, mask_api_key,
        )

        _env_api_key = os.getenv("DIFY_API_KEY", "")
        _env_base_url = os.getenv("DIFY_API_BASE", "http://localhost/v1")

        # 连接配置来源选择
        dify_conn_source = st.radio(
            "连接配置来源",
            ["使用已保存连接配置", "临时手动填写"],
            horizontal=True,
            key="dify_conn_source",
        )

        # 初始化变量
        dify_api_key = ""
        dify_base_url = _env_base_url
        dify_timeout = 60
        dify_delay = 1.0
        _selected_profile_id = ""
        _selected_profile_name = ""
        _selected_profile_desc = ""

        if dify_conn_source == "使用已保存连接配置":
            profiles = list_connection_profiles()
            if not profiles:
                st.info("暂无已保存的连接配置，请选择「临时手动填写」或创建新配置。")
                dify_conn_source = "临时手动填写"
            else:
                # 下拉选择
                profile_options = []
                for p in profiles:
                    pid = p.get("profile_id", "")
                    pname = p.get("profile_name", "未命名")
                    purl = pdesc = ""
                    if pid:
                        purl = p.get("base_url", "")
                        pdesc = p.get("workflow_description", "")
                    label = f"{pname} · {purl}"
                    if pdesc:
                        label += f" · {pdesc}"
                    profile_options.append((pid, label))

                _selected_profile_id = st.selectbox(
                    "选择连接配置",
                    options=[c[0] for c in profile_options],
                    format_func=lambda x: next((c[1] for c in profile_options if c[0] == x), x),
                    key="dify_selected_profile",
                )

                if _selected_profile_id:
                    _sel_meta = load_connection_profile(_selected_profile_id)
                    if _sel_meta:
                        _selected_profile_name = _sel_meta.get("profile_name", "")
                        _selected_profile_desc = _sel_meta.get("workflow_description", "")
                        dify_base_url = _sel_meta.get("base_url", _env_base_url)
                        dify_timeout = _sel_meta.get("timeout_seconds", 60)
                        dify_delay = _sel_meta.get("request_interval_seconds", 1.0)

                        # 显示掩码 API Key
                        _saved_key = get_connection_api_key(_selected_profile_id)
                        if _saved_key:
                            dify_api_key = _saved_key
                            st.caption(f"API Key: `{mask_api_key(_saved_key)}`（已从安全存储读取）")
                        else:
                            st.warning("该配置未保存 API Key，请在下方手动输入。")
                            _manual_key = st.text_input(
                                "临时 API Key", type="password", key="dify_temp_key_for_profile",
                                help="仅本次会话使用，不写入磁盘",
                            )
                            if _manual_key:
                                dify_api_key = _manual_key

                        st.caption(f"Base URL: `{dify_base_url}` | 超时: {dify_timeout}s | 间隔: {dify_delay}s")

                # 管理操作
                mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
                with mgmt_col1:
                    if st.button("新建连接配置", key="dify_new_profile"):
                        st.session_state["dify_show_new_profile_form"] = True
                with mgmt_col2:
                    if st.button("编辑连接配置", key="dify_edit_profile", disabled=not _selected_profile_id):
                        st.session_state["dify_show_edit_profile_form"] = True
                with mgmt_col3:
                    if st.button("删除连接配置", key="dify_delete_profile", disabled=not _selected_profile_id):
                        st.session_state["dify_show_delete_confirm"] = True

                # 新建配置表单
                if st.session_state.get("dify_show_new_profile_form"):
                    with st.form("new_dify_profile_form"):
                        st.markdown("**新建连接配置**")
                        np_name = st.text_input("配置名称 *", placeholder="例如：金融知识库工作流-v2", key="np_name")
                        np_url = st.text_input("Base URL *", value=_env_base_url, key="np_url")
                        np_key = st.text_input("API Key *", type="password", key="np_key")
                        np_desc = st.text_input("工作流说明（可选）", key="np_desc")
                        np_timeout = st.number_input("超时（秒）", value=60, min_value=10, max_value=300, key="np_timeout")
                        np_interval = st.number_input("请求间隔（秒）", value=1.0, min_value=0.0, max_value=10.0, step=0.5, key="np_interval")
                        np_submit = st.form_submit_button("保存")
                    if np_submit and np_name and np_url and np_key:
                        create_connection_profile(np_name, np_url, np_key, np_desc, np_timeout, np_interval)
                        st.success(f"连接配置「{np_name}」已保存（API Key 已安全存储）")
                        st.session_state["dify_show_new_profile_form"] = False
                        st.rerun()

                # 编辑配置表单
                if st.session_state.get("dify_show_edit_profile_form") and _selected_profile_id:
                    _edit_meta = load_connection_profile(_selected_profile_id)
                    if _edit_meta:
                        with st.form("edit_dify_profile_form"):
                            st.markdown(f"**编辑连接配置: {_edit_meta.get('profile_name', '')}**")
                            ep_name = st.text_input("配置名称", value=_edit_meta.get("profile_name", ""), key="ep_name")
                            ep_url = st.text_input("Base URL", value=_edit_meta.get("base_url", ""), key="ep_url")
                            ep_key = st.text_input("API Key（留空则保留现有）", type="password", key="ep_key")
                            ep_desc = st.text_input("工作流说明", value=_edit_meta.get("workflow_description", ""), key="ep_desc")
                            ep_timeout = st.number_input("超时（秒）", value=_edit_meta.get("timeout_seconds", 60), min_value=10, max_value=300, key="ep_timeout")
                            ep_interval = st.number_input("请求间隔（秒）", value=_edit_meta.get("request_interval_seconds", 1.0), min_value=0.0, max_value=10.0, step=0.5, key="ep_interval")
                            ep_clear = st.checkbox("清除已保存的 API Key", key="ep_clear_key")
                            ep_submit = st.form_submit_button("保存修改")
                        if ep_submit:
                            update_connection_profile(
                                _selected_profile_id,
                                {"profile_name": ep_name, "base_url": ep_url, "workflow_description": ep_desc,
                                 "timeout_seconds": ep_timeout, "request_interval_seconds": ep_interval},
                                api_key=ep_key if ep_key else None,
                                clear_key=ep_clear,
                            )
                            st.success("连接配置已更新")
                            st.session_state["dify_show_edit_profile_form"] = False
                            st.rerun()

                # 删除确认
                if st.session_state.get("dify_show_delete_confirm") and _selected_profile_id:
                    _del_meta = load_connection_profile(_selected_profile_id)
                    st.warning(f"确认删除连接配置「{_del_meta.get('profile_name', '') if _del_meta else ''}」？已保存的 API Key 将一并删除。历史运行记录不受影响。")
                    dc_col1, dc_col2 = st.columns(2)
                    with dc_col1:
                        if st.button("确认删除", key="dify_confirm_delete", type="primary"):
                            delete_connection_profile(_selected_profile_id)
                            st.success("已删除")
                            st.session_state["dify_show_delete_confirm"] = False
                            st.rerun()
                    with dc_col2:
                        if st.button("取消", key="dify_cancel_delete"):
                            st.session_state["dify_show_delete_confirm"] = False
                            st.rerun()

        if dify_conn_source == "临时手动填写":
            st.caption("本次填写的密钥仅用于当前会话，不会写入磁盘。如需保存，请勾选下方选项。")
            tm_col1, tm_col2 = st.columns(2)
            with tm_col1:
                dify_api_key = st.text_input(
                    "Dify API Key", type="password",
                    value=_env_api_key,
                    key="batch_dify_key",
                    help="来自 .env 的默认值" if _env_api_key else "",
                )
            with tm_col2:
                dify_base_url = st.text_input(
                    "Dify Base URL",
                    value=_env_base_url,
                    key="batch_dify_url",
                )
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                dify_timeout = st.number_input(
                    "请求超时（秒）", min_value=10, max_value=300, value=60, key="batch_timeout"
                )
            with opt_col2:
                dify_delay = st.number_input(
                    "请求间隔（秒）", min_value=0.0, max_value=10.0, value=1.0, step=0.5, key="batch_delay",
                    help="每次请求之间的等待时间，避免过快调用"
                )

            # 保存为命名配置
            _save_as_profile = st.checkbox("保存为命名连接配置", key="dify_save_as_profile")
            if _save_as_profile:
                sp_col1, sp_col2 = st.columns(2)
                with sp_col1:
                    _save_name = st.text_input("配置名称", placeholder="例如：金融知识库工作流-v2", key="dify_save_name")
                with sp_col2:
                    _save_desc = st.text_input("工作流说明（可选）", key="dify_save_desc")
                if st.button("保存连接配置", key="dify_save_profile_btn"):
                    if _save_name and dify_api_key and dify_base_url:
                        create_connection_profile(_save_name, dify_base_url, dify_api_key, _save_desc, dify_timeout, dify_delay)
                        st.success(f"连接配置「{_save_name}」已保存（API Key 已安全存储）")
                        st.rerun()
                    else:
                        st.warning("请填写配置名称、API Key 和 Base URL")

        # 环境变量提示
        if not dify_api_key and _env_api_key:
            st.caption("将使用 .env 中的 `DIFY_API_KEY` 作为默认密钥。")
            dify_api_key = _env_api_key

    # --- Run batch query ---
    st.divider()

    if st.button("开始提问", type="primary", disabled=len(questions_list) == 0, key="batch_run"):
        if not dify_api_key:
            st.error("请填写 Dify API Key（选择已保存连接配置或临时手动填写）")
        elif not questions_list:
            st.error("没有可提问的问题")
        else:
            # 获取配置来源
            _config_source = st.session_state.get("batch_config_source", "新建配置方案")

            # 获取或创建配置方案
            if _config_source == "使用历史配置":
                _config_id = st.session_state.get("batch_selected_config", "")
                if not _config_id:
                    st.error("请选择历史配置")
                    st.stop()
            else:
                # 创建新配置方案（从统一 schema 的 session_state 读取）
                _form_vals = {}
                for _key, _, _, _, _, _ in CONFIG_FIELD_SCHEMA:
                    _form_vals[_key] = st.session_state.get(f"batch_new_{_key}", "")
                # 必填字段兜底
                if not str(_form_vals.get("config_name", "")).strip():
                    _form_vals["config_name"] = "未命名配置"
                if not str(_form_vals.get("knowledge_base_version", "")).strip():
                    _form_vals["knowledge_base_version"] = "未指定"

                config_result = create_config_profile(**collect_config_updates(_form_vals))
                _config_id = config_result["config_id"]

            # 创建运行记录
            # 从 questions_list 中提取题集信息
            _q_set_id = ""
            _q_set_name = ""
            for q in questions_list:
                if q.get("question_set_id"):
                    _q_set_id = q["question_set_id"]
                    _q_set_name = q.get("question_set_name", "")
                    break

            run_result = create_experiment_run(
                config_id=_config_id,
                question_set_source=st.session_state.get("batch_q_source", ""),
                question_count=len(questions_list),
            )
            run_id = run_result["run_id"]
            run_dir = run_result["run_dir"]

            # 更新 manifest 添加题集信息和连接配置元数据（不含 API Key）
            _manifest_updates = {}
            if _q_set_id or _q_set_name:
                _manifest_updates["question_set_id"] = _q_set_id
                _manifest_updates["question_set_name"] = _q_set_name
            if _selected_profile_id:
                _manifest_updates["dify_connection_profile_id"] = _selected_profile_id
                _manifest_updates["dify_connection_profile_name"] = _selected_profile_name
                _manifest_updates["dify_base_url"] = dify_base_url
                _manifest_updates["dify_workflow_description"] = _selected_profile_desc
            elif dify_base_url:
                _manifest_updates["dify_base_url"] = dify_base_url
            if _manifest_updates:
                from experiment import update_experiment_run
                update_experiment_run(run_id, _manifest_updates)

            # 确保每个问题有 question_id
            question_ids = []
            for q in questions_list:
                q = ensure_question_id(q)
                question_ids.append(q.get("question_id", ""))

            st.info(f"运行已创建: `{run_id}` | 配置方案: `{_config_id}`")

            batch_results = []
            progress_bar = st.progress(0, text="准备开始...")
            status_container = st.container()

            for idx, total, result in run_batch_query(
                questions_list, dify_api_key, dify_base_url,
                timeout=dify_timeout, delay=dify_delay,
                run_id=run_id,
                config_id=_config_id,
                question_ids=question_ids,
            ):
                progress_bar.progress(
                    (idx + 1) / total,
                    text=f"正在提问第 {idx + 1} / {total} 条",
                )
                batch_results.append(result)

                with status_container:
                    if result["success"]:
                        answer_preview = (result["sample"].get("final_answer", "") or "")[:80]
                        st.success(f"✅ [{idx + 1}/{total}] {result['question'][:40]}... → {answer_preview}")
                    else:
                        st.error(f"❌ [{idx + 1}/{total}] {result['question'][:40]}... → {result['error'][:80]}")

            progress_bar.progress(1.0, text="提问完成！")
            st.session_state["batch_results"] = batch_results
            st.session_state["batch_run_id"] = run_id

            # 保存结果到运行目录
            run_batch_path = run_dir / "batch_results.jsonl"
            with run_batch_path.open("w", encoding="utf-8") as f:
                for r in batch_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 同时保存到 data/batch/
            batch_path, batch_filename = save_batch_results(batch_results)

            # 推送到 data/raw/
            raw_path, raw_filename = push_to_raw_dir(batch_results)

            # 更新运行 manifest
            update_experiment_run(run_id, {
                "batch_results_file": batch_filename,
                "raw_results_file": raw_filename,
                "status": "completed",
            })

            st.success(f"批量提问完成！成功 {sum(1 for r in batch_results if r['success'])} / {total} 条")
            st.caption(f"运行结果已保存到: `{run_dir}`")

    # --- Results display ---
    batch_results = st.session_state.get("batch_results")
    if batch_results:
        st.divider()
        st.subheader("提问结果")

        success_count = sum(1 for r in batch_results if r["success"])
        fail_count = len(batch_results) - success_count
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("总问题数", len(batch_results))
        res_col2.metric("成功", success_count)
        res_col3.metric("失败", fail_count)

        # Results table
        table_data = []
        for i, r in enumerate(batch_results):
            if r["success"]:
                sample = r.get("sample", {})
                table_data.append({
                    "序号": i + 1,
                    "问题": r["question"],
                    "回答": (sample.get("final_answer", "") or "")[:100],
                    "检索结果数": len(sample.get("retrieval_results", [])),
                    "状态": "✅ 成功",
                })
            else:
                table_data.append({
                    "序号": i + 1,
                    "问题": r["question"],
                    "回答": "",
                    "检索结果数": 0,
                    "状态": f"❌ {r.get('error', '未知错误')[:50]}",
                })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Expandable detail for each result
        for i, r in enumerate(batch_results):
            if r["success"]:
                sample = r.get("sample", {})
                with st.expander(f"✅ Q{i+1}: {r['question'][:60]}"):
                    st.markdown(f"**问题**: {r['question']}")
                    st.markdown(f"**回答**: {sample.get('final_answer', '')}")
                    retrieval_results = sample.get("retrieval_results", [])
                    if retrieval_results:
                        st.markdown(f"**检索结果** ({len(retrieval_results)} 条):")
                        for rr in retrieval_results:
                            st.write(f"  - [{rr.get('position')}] {rr.get('title', 'N/A')} (score: {rr.get('score', 'N/A')})")
                            if rr.get("content"):
                                st.caption(f"    {rr['content'][:200]}")
                    else:
                        st.caption("无检索结果")
                    with st.expander("原始响应"):
                        st.json(r.get("raw_response", {}))
            else:
                with st.expander(f"❌ Q{i+1}: {r['question'][:60]}"):
                    st.error(f"错误: {r.get('error', '未知错误')}")

        # --- Export & Push ---
        st.divider()
        st.subheader("导出与推送")

        with st.expander("输出文件说明", expanded=False):
            st.markdown("""
| 操作 | 保存位置 | 用途 |
|------|---------|------|
| 自动保存完整结果 | `data/batch/batch_results_<时间戳>.jsonl` | 包含每条问题的原始响应、成功/失败状态，用于排查 |
| 下载 JSONL / CSV | 本地下载 | 离线备份或分享 |
| 推送到样本准备 | `data/raw/batch_qa_<时间戳>.jsonl` | 仅含成功结果，格式兼容后续「样本准备」和「Judge 评测」 |

> 推送后请切换到「样本准备」tab，选择该文件并点击「解析」即可进入后续评测流程。
""")

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            # JSONL download
            jsonl_lines = []
            for r in batch_results:
                jsonl_lines.append(json.dumps(r, ensure_ascii=False))
            jsonl_data = "\n".join(jsonl_lines).encode("utf-8")
            st.download_button(
                label="📥 下载完整结果 (JSONL)",
                data=jsonl_data,
                file_name="batch_results.jsonl",
                mime="application/jsonl",
                use_container_width=True,
            )

        with exp_col2:
            # CSV download
            csv_data = batch_export_csv(batch_results)
            st.download_button(
                label="📥 下载结果 (CSV)",
                data=csv_data,
                file_name="batch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp_col3:
            # Push to raw dir for downstream consumption
            if st.button("📤 推送到样本准备", use_container_width=True,
                         help="将成功的结果保存到 data/raw/，可在「样本准备」tab 中解析"):
                successful = [r for r in batch_results if r["success"] and r.get("sample")]
                if successful:
                    push_path, push_name = push_to_raw_dir(batch_results)
                    st.success(f"已推送 {len(successful)} 条结果到 {push_name}")
                    st.caption("请切换到「样本准备」tab，选择该文件并点击「解析」")
                else:
                    st.warning("没有成功的结果可推送")

# ========== Tab: 样本准备 ==========
with tab_samples:
    st.subheader("样本准备")
    st.caption("导入 Langfuse 导出数据，解析并准备评测样本")

    # ---------- 模块说明 ----------
    with st.expander("样本准备模块说明（点击展开）", expanded=False):
        st.markdown(f"""
**一句话总览：** 将 Dify / Langfuse 的运行记录解析为结构化样本，回填参考答案和运行元数据，为 Judge 评测提供输入。

---

**这个模块做什么？**

Judge 不是直接读取原始 trace 文件，而是读取这里准备好的结构化样本。这个模块负责：

1. **导入原始记录** — 从 Langfuse 导出的 JSONL 文件、Langfuse API 或批量提问推送的 raw 文件获取数据
2. **解析为结构化样本** — 按 traceId 聚合 observations，提取关键字段：
   - 用户问题（question）
   - 检索查询（retrieval_query）
   - 检索结果列表（retrieval_results）
   - LLM 最终回答（final_answer）
   - trace_id、session_id 等标识信息
3. **回填参考答案和元数据** — 从题目库匹配 reference_answer、source_excerpt，从 `user_id` 回填 run_id、question_id、question_set_id 等运行元数据

---

**输入从哪来？**

| 来源 | 说明 |
|------|------|
| 上传文件 | 上传 Langfuse 导出的 .jsonl 文件 |
| API 拉取 | 直接从 Langfuse API 拉取 traces |
| 批量提问推送 | 在「批量提问」中成功的结果会推送到 `data/raw/`，然后在这里解析 |

---

**输出到哪去？**

| 输出 | 路径 | 用途 |
|------|------|------|
| 结构化样本 | `{PROCESSED_DIR.name}/langfuse_samples.jsonl` | Judge 评测的直接输入 |
| 解析摘要 | `{PROCESSED_DIR.name}/langfuse_summary.json` | 记录来源文件、样本数、回填统计等 |

---

**关联链说明**

```
run_id → processed sample → 真实 Langfuse trace_id → Judge result
```

- processed sample 的 `trace_id` 是真实的 Langfuse UUID（来自 Dify 调用 Langfuse 记录的 UUID）
- **不是** `batch_qa_*` 伪 trace_id（那是批量提问模块生成的文件标识）
- Judge 结果通过 processed sample 的 trace_id 关联，不通过 batch_qa_* 关联
- 运行看板通过 `run_id → processed trace_id → judged trace_id` 链路汇总指标

---

**参考答案回填规则**

解析时会自动从题目库（`data/questions/`）中匹配：

1. 如果样本本身已有 reference_answer → 跳过
2. 如果样本有 question_id → 按 ID 精确匹配
3. 否则 → 按 question 文本精确匹配
4. 匹配成功 → 回填 reference_answer + source_excerpt + difficulty + topic + question_mode + question_set_id
5. 匹配失败 → 保留为空，该样本在 Judge 中走无参考答案评测

解析完成后会显示回填统计，告诉你多少条成功回填、多少条没有匹配到。
""")

    # --- Data import section (collapsible) ---
    with st.expander("数据导入", expanded=not samples):
        # Step 1: Acquire data
        st.markdown("**第一步：获取 Langfuse 导出文件**")
        source_mode = st.radio(
            "获取方式",
            ["上传文件", "从 API 拉取"],
            horizontal=True,
            key="lf_source_mode",
            label_visibility="collapsed",
        )

        if source_mode == "上传文件":
            uploaded = st.file_uploader("上传 Langfuse 导出文件", type=["jsonl"], key="langfuse_upload")
            if uploaded is not None:
                save_path = RAW_DIR / uploaded.name
                save_path.write_bytes(uploaded.getvalue())
                st.success(f"已保存: {uploaded.name}")
                st.rerun()

        elif source_mode == "从 API 拉取":
            fetch_col1, fetch_col2 = st.columns(2)
            with fetch_col1:
                langfuse_host = st.text_input("Langfuse 地址", value=os.getenv("LANGFUSE_HOST", "http://localhost:3000"), key="lf_host")
                langfuse_pk = st.text_input("Public Key", value=os.getenv("LANGFUSE_PUBLIC_KEY", ""), key="lf_pk")
            with fetch_col2:
                langfuse_sk = st.text_input("Secret Key", value=os.getenv("LANGFUSE_SECRET_KEY", ""), type="password", key="lf_sk")
                fetch_limit = st.number_input("每页 trace 数", min_value=1, max_value=500, value=50, key="lf_limit")

            if st.button("拉取 Traces", key="fetch_traces"):
                if not langfuse_pk or not langfuse_sk:
                    st.error("请填写 Langfuse Public Key 和 Secret Key")
                else:
                    from fetch_traces import fetch_all
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"langfuse_api_export_{ts}.jsonl"
                    output_path = RAW_DIR / filename
                    with st.spinner(f"正在从 {langfuse_host} 拉取 Traces..."):
                        try:
                            count = 0
                            with output_path.open("w", encoding="utf-8") as f:
                                for row in fetch_all(langfuse_host, langfuse_pk, langfuse_sk, limit=fetch_limit):
                                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                    count += 1
                            st.success(f"拉取完成！共 {count} 行，已保存为 {filename}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"拉取失败: {e}")

        # Step 2: Select file & parse
        st.divider()
        st.markdown("**第二步：选择文件并解析**")

        raw_files = sorted(RAW_DIR.glob("*.jsonl"))
        if not raw_files:
            st.info("data/raw 目录下暂无 .jsonl 文件，请先通过上方方式获取数据")
            selected_name = None
            selected_path = None
        else:
            file_names = [f.name for f in raw_files]
            selected_name = st.selectbox("待解析文件", file_names, key="raw_select")
            selected_path = RAW_DIR / selected_name

            file_size_kb = selected_path.stat().st_size / 1024
            with open(selected_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            st.caption(f"文件大小: {file_size_kb:.1f} KB | 总行数: {line_count}")

            if st.button("开始解析", type="primary", key="parse_btn"):
                with st.spinner("正在解析..."):
                    samples, summary = parse_langfuse_jsonl(selected_path)
                    output_path = PROCESSED_DIR / "langfuse_samples.jsonl"
                    summary_path = PROCESSED_DIR / "langfuse_summary.json"
                    full_summary = save_results(samples, summary, output_path, summary_path)
                    st.session_state["samples"] = samples
                    st.session_state["summary"] = full_summary
                st.success(f"解析完成，共 {len(samples)} 条 trace")
                # 回填结果提示
                bs = summary.get("backfill_stats") or {}
                if bs:
                    bf = bs.get("backfilled", 0)
                    already = bs.get("already_has", 0)
                    no_ref = bs["total"] - bf - already
                    if bf > 0:
                        st.success(f"参考答案回填：**{bf}** 条样本匹配到题目库，已回填 reference_answer")
                    if already > 0:
                        st.info(f"**{already}** 条样本本身已带参考答案")
                    if no_ref > 0:
                        st.warning(f"**{no_ref}** 条样本未匹配到题目库，将走无参考答案评测")
                st.rerun()

    # --- Sample display section ---
    if not samples:
        st.info("请在上方「数据导入」区域上传或拉取 Langfuse 数据，然后点击「开始解析」")
    else:
        input_file = summary.get("input_file") or (selected_name if 'selected_name' in dir() and selected_name else "") or ""
        output_file = summary.get("output_file") or ""
        if input_file:
            st.caption(f"数据来源: `{Path(input_file).name}`" + (f" → 解析结果: `{Path(output_file).name}`" if output_file else ""))

        # Stats
        trace_count = summary.get("trace_count") or len(samples)
        bad_line_count = summary.get("bad_line_count") or 0
        retrieval_total = summary.get("total_retrieval_results")

        st.subheader("统计信息")
        col1, col_col2, col3 = st.columns(3)
        col1.metric("总 Trace 数", trace_count)
        col_col2.metric("成功解析", trace_count - bad_line_count)
        col3.metric("Retrieval 结果总数", retrieval_total if retrieval_total is not None else "N/A")

        if bad_line_count > 0:
            st.warning(f"有 {bad_line_count} 行解析失败")

        # Search filter
        search = st.text_input("搜索问题内容", "", key="sample_search")
        filtered = samples
        if search:
            filtered = [s for s in samples if search.lower() in (s.get("question") or "").lower()]

        for i, sample in enumerate(filtered):
            question = sample.get("question") or "(无问题)"
            retrieval_count = len(sample.get("retrieval_results", []))

            with st.expander(
                f"**Q:** {question[:60]}{'...' if len(question) > 60 else ''} | "
                f"检索: {retrieval_count} 条 | {sample.get('trace_id', '')[:8]}..."
            ):
                st.markdown("**问题**")
                st.code(sample.get("question") or "(无)", language=None)

                st.markdown("**检索查询 (retrieval_query)**")
                st.code(sample.get("retrieval_query") or "(无)", language=None)

                st.markdown(f"**检索结果 ({retrieval_count} 条)**")
                for r in sample.get("retrieval_results", []):
                    title = r.get("title") or "(无标题)"
                    score = r.get("score")
                    content = r.get("content") or ""
                    score_str = f" (score: {score})" if score is not None else ""
                    with st.expander(f"{title}{score_str}"):
                        st.text((content or "(无内容)")[:2000])

                st.markdown(f"**LLM 模型**: `{sample.get('llm_model') or 'N/A'}`")

                st.markdown("**LLM Input**")
                llm_input = sample.get("llm_input")
                if llm_input:
                    st.json(llm_input)
                else:
                    st.caption("(无)")

                st.markdown("**LLM Output**")
                llm_output = sample.get("llm_output")
                if llm_output:
                    st.json(llm_output)
                else:
                    st.caption("(无)")

                st.markdown("**最终回答 (final_answer)**")
                st.code(sample.get("final_answer") or "(无)", language=None)

                # --- 参考答案与评测模式 ---
                ref_answer = (sample.get("reference_answer") or "").strip()
                source_excerpt = (sample.get("source_excerpt") or "").strip()
                difficulty = sample.get("difficulty") or ""
                topic = sample.get("topic") or ""

                if ref_answer:
                    st.markdown("**参考答案 (reference_answer)**")
                    st.code(ref_answer, language=None)
                    if source_excerpt:
                        with st.expander("来源摘录 (source_excerpt)"):
                            st.text(source_excerpt[:2000])
                    # 题目元数据（如果有）
                    _meta_parts = []
                    if difficulty:
                        _meta_parts.append(f"难度: {difficulty}")
                    if topic:
                        _meta_parts.append(f"主题: {topic}")
                    if _meta_parts:
                        st.caption(" | ".join(_meta_parts))
                    st.success("评测模式：**严格评测**（有参考答案，将与参考答案对比评判）")
                else:
                    st.warning("评测模式：**无参考答案评测**（LLM 将基于问题和检索内容自行判断回答合理性）")

                st.markdown("**元数据**")
                st.json({
                    "trace_id": sample.get("trace_id"),
                    "trace_name": sample.get("trace_name"),
                    "session_id": sample.get("session_id"),
                    "user_id": sample.get("user_id"),
                    "workflow_run_id": sample.get("workflow_run_id"),
                    "observation_count": len(sample.get("observations", [])),
                })

# ========== Tab: Judge 评测 ==========
with tab_judge:
    st.subheader("Judge 评测")

    # ---------- 数据来源摘要 ----------
    if samples and summary:
        src_file = summary.get("input_file") or ""
        src_name = Path(src_file).name if src_file else "(未知来源)"
        trace_count = summary.get("trace_count") or len(samples)
        retrieval_total = summary.get("total_retrieval_results")

        # 统计评测轨道
        from judge import classify_evaluation_track, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE

        track_counts = {
            TRACK_RETRIEVAL: 0,
            TRACK_STRICT_QA: 0,
            TRACK_GROUNDED_QA: 0,
            TRACK_NOT_EVALUABLE: 0,
        }
        for s in samples:
            track = classify_evaluation_track(s)
            track_counts[track] += 1

        # 统计 question_mode（兼容旧版）
        retrieval_mode_count = sum(1 for s in samples if s.get("question_mode") == MODE_RETRIEVAL)
        qa_mode_count = sum(1 for s in samples if s.get("question_mode") == MODE_QA)
        unknown_mode_count = trace_count - retrieval_mode_count - qa_mode_count

        info_parts = [
            f"**来源文件**: `{src_name}`",
            f"**样本数**: {trace_count}",
            f"**检索结果总数**: {retrieval_total}" if retrieval_total else None,
        ]
        st.info(" | ".join(p for p in info_parts if p))

        # 题目目的构成
        st.markdown("##### 题目目的构成")
        mode_col1, mode_col2, mode_col3 = st.columns(3)
        with mode_col1:
            if retrieval_mode_count > 0:
                st.metric("检索评测题", retrieval_mode_count, help="question_mode=retrieval，主要用于测试 RAG 检索命中率")
            if qa_mode_count > 0:
                st.metric("全流程问答题", qa_mode_count, help="question_mode=qa，用于测试完整问答能力")
            if unknown_mode_count > 0:
                st.metric("旧版/未知模式", unknown_mode_count, help="缺少 question_mode 字段，按旧逻辑处理")

        # 评分依据构成
        st.markdown("##### 评分依据构成")
        track_col1, track_col2, track_col3, track_col4 = st.columns(4)
        with track_col1:
            if track_counts[TRACK_RETRIEVAL] > 0:
                st.metric("可评测检索题", track_counts[TRACK_RETRIEVAL],
                          help="有金标准证据（source_excerpt 或 reference_answer），可计算 TopK Hit")
        with track_col2:
            if track_counts[TRACK_STRICT_QA] > 0:
                st.metric("严格问答", track_counts[TRACK_STRICT_QA],
                          help="有 reference_answer，可评判回答正确性")
        with track_col3:
            if track_counts[TRACK_GROUNDED_QA] > 0:
                st.metric("合理性问答", track_counts[TRACK_GROUNDED_QA],
                          help="无参考答案，基于检索内容判断合理性")
        with track_col4:
            if track_counts[TRACK_NOT_EVALUABLE] > 0:
                st.metric("缺少金标准", track_counts[TRACK_NOT_EVALUABLE],
                          help="检索评测题但 source_excerpt 和 reference_answer 均为空，无法可靠计算 Hit")

        # 混合提示
        has_mixed_modes = (retrieval_mode_count > 0 and qa_mode_count > 0)
        has_mixed_tracks = (track_counts[TRACK_RETRIEVAL] > 0 and track_counts[TRACK_STRICT_QA] > 0) or \
                          (track_counts[TRACK_RETRIEVAL] > 0 and track_counts[TRACK_GROUNDED_QA] > 0)
        if has_mixed_modes or has_mixed_tracks:
            st.warning("**混合评测**：包含不同类型的题目和评分依据，指标将按评测轨道分组展示，避免混合口径。")
    else:
        st.caption("对解析后的样本进行自动评分")

    # ---------- 运行机制说明 ----------
    with st.expander("Judge 运行机制说明（点击展开）", expanded=False):
        st.markdown(f"""
**一句话总览：** Judge 从「样本准备」中取出候选样本，逐条调用 LLM 对检索质量和回答正确性进行评分，结果保存到评测结果文件。

---

**Judge 评什么？两层评测，不是一个总分**

Judge 不是只给一个"总分"，而是同时评两个独立维度：

| 评测维度 | 评什么 | 对应指标 | 含义 |
|---|---|---|---|
| RAG 检索层 | 检索结果是否召回了正确内容 | Top1 / Top3 / Top5 Hit | 检索链路质量 |
| LLM 回答层 | 最终回答是否正确完整 | Answer OK | 回答生成质量 |

这两层相互独立：
- 检索命中高，不代表回答一定对（LLM 可能理解错或生成错）
- 回答正确，也不代表检索一定好（LLM 可能靠自身知识推断）
- 两层都高，才说明 RAG 链路整体健康

---

**两种评测模式：有参考答案 vs 无参考答案**

Judge 支持两种评测模式，取决于样本是否带有 `reference_answer`（参考答案）：

| 模式 | 判断依据 | Answer Correct 含义 | 适用场景 |
|---|---|---|---|
| **严格评测**（有参考答案） | 将最终回答与参考答案对比 | 回答是否与参考答案一致、覆盖关键要点 | 题目生成链路产出的样本 |
| **合理性评测**（无参考答案） | LLM 基于问题和检索内容自行判断 | 回答是否看起来合理且完整 | 手动问题、Langfuse 导入等 |

- 参考答案来自题目生成模块（`reference_answer` 字段），随样本全链路传递
- 严格评测更可靠，因为有明确的正确答案作为基准
- 合理性评测更宽松，LLM 只能判断"看起来对不对"，不能保证与标准答案一致
- 页面指标区会显示当前是哪种模式（或混合模式）

---

**题目模式：检索评测 vs 全流程问答评测**

除了评测模式（有/无参考答案），样本还可能带有 `question_mode` 字段，标识这道题原本的出题目的：

| 题目模式 | 出题目的 | 重点关注指标 | 辅助指标 |
|---|---|---|---|
| **检索评测** (`retrieval`) | 测试 RAG 系统能否检索到正确内容 | Top1 / Top3 / Top5 Hit | Answer OK（仅作参考） |
| **全流程问答评测** (`qa`) | 测试从检索到回答的完整能力 | Answer OK | Top1 / Top3 / Top5 Hit |

- 如果题目来自「题目生成」模块的「检索评测模式」，`question_mode` 会自动标记为 `retrieval`
- 这个字段会随样本全链路透传：题目生成 → 批量提问 → 样本准备 → Judge
- 页面顶部会统计并显示当前样本的题目模式构成

---

**评测输入是什么？**

Judge 评的不是原始题目文件，而是经过「样本准备」解析后的结构化样本。

- 输入文件：`{PROCESSED_DIR.name}/langfuse_samples.jsonl`
- 每条样本包含：用户问题、检索查询、检索结果列表、最终回答、trace_id 等
- 如果样本带有 `reference_answer`，Judge 会用它进行严格评测
- 页面中的候选样本，就是从这份文件中加载的

---

**样本怎么选？**

| 配置项 | 效果 |
|---|---|
| 评测样本数 = N | 从样本准备中按顺序取前 N 条作为候选 |
| 只评前 1 条（快速测试） | 覆盖上述设置，仅取第 1 条候选样本 |
| 跳过已有成功结果 | 候选样本中已有成功评测记录的会被跳过 |
| 强制重新评测 | 不跳过任何候选样本，全部重新运行 |
| 只重试失败样本 | 切换评测对象：不走「前 N 条」逻辑，而是从已有结果中找出失败的样本重跑 |

---

**「缓存」是什么意思？**

Judge 有三层减少重复调用的机制：

1. **结果跳过**：读取已有评测结果文件（`{JUDGED_DIR.name}/{JUDGED_FILE.name}`），如果某条样本已有成功结果且选择「跳过已有成功结果」，则不会重复调用 LLM
2. **内容复用**：如果多条样本的问题、检索查询、回答内容完全相同，只需评测一次，其余复用结果
3. **规则预筛选**：对于明显无法评测的样本（如无问题、无回答、无检索结果），直接给出规则判定结果，不进入 LLM

这些在点击「预览优化策略」后可以看到具体节省了多少次 LLM 调用。

---

**结果保存到哪里？**

- 最新结果始终保存到：`{JUDGED_DIR.name}/{JUDGED_FILE.name}`
- 每次运行后还会生成带时间戳的历史快照（如 `eval_results_20250709_143000.jsonl`）
- 结果按 trace_id 合并更新：新评测结果会覆盖同一条样本的旧结果，未重跑的成功结果保留
- 这意味着结果文件会持续积累，不是每次运行都从零开始

---

**新样本怎么进入 Judge？**

新题目不会自动出现在 Judge 中，需要经过完整流程：

```
题目生成 → 批量提问(Dify) → 样本准备 → Judge 评测
```

1. **题目生成**：从知识库文件生成测试问题，产出题集（含 question_set_id）
2. **批量提问**：选择题集和配置方案，通过 Dify API 批量提问，产出 raw 文件（含 run_id）
3. **样本准备**：解析 raw 文件为 processed samples，使用真实 Langfuse trace_id，回填参考答案和元数据
4. **Judge 评测**：从 processed samples 中取出样本，按评测轨道调用 LLM 评分

**注意**：Judge 通过 processed sample 的 trace_id（真实 Langfuse UUID）关联结果，不是通过 `batch_qa_*` 伪 trace_id。
运行看板通过 `run_id → processed trace_id → judged trace_id` 链路汇总指标，兼容旧格式 Judge 结果（无 run_id 时通过 trace_id fallback 关联）。

只有完成前 3 步，新样本才会出现在 Judge 的候选列表中。
""")

    if not samples:
        st.info("请先切换到「样本准备」tab 导入并解析数据")
    else:
        # ---------- 已有结果加载 & 索引（放在 UI 前，供策略摘要使用） ----------
        existing_results_map = {}  # trace_id -> result dict
        if JUDGED_FILE.exists():
            with JUDGED_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                        tid = r.get("trace_id")
                        if tid:
                            existing_results_map[tid] = r
                    except json.JSONDecodeError:
                        pass

        # 补齐历史结果的 evaluation_track 等字段
        # 构建 sample 索引以便关联
        _sample_by_tid = {s.get("trace_id"): s for s in samples if s.get("trace_id")}
        _unmatched_results = []  # 无法关联当前 sample 的历史结果

        for tid, r in existing_results_map.items():
            # 如果已有 evaluation_track，跳过
            if r.get("evaluation_track"):
                continue

            # 尝试关联当前 sample
            sample = _sample_by_tid.get(tid)
            if sample:
                # 从 sample 补齐字段
                r["question_mode"] = (sample.get("question_mode") or "").strip()
                r["evaluation_track"] = classify_evaluation_track(sample)
                r["retrieval_evaluable"] = r["evaluation_track"] == TRACK_RETRIEVAL
                if r["evaluation_track"] == TRACK_NOT_EVALUABLE:
                    r["not_evaluable_reason"] = "检索评测题缺少金标准证据"
            else:
                # 无法关联 sample，尝试从结果本身推断
                has_ref = bool((r.get("reference_answer") or "").strip())
                if has_ref:
                    r["evaluation_track"] = TRACK_STRICT_QA
                else:
                    r["evaluation_track"] = TRACK_GROUNDED_QA
                r["retrieval_evaluable"] = False
                _unmatched_results.append(tid)

        # --- Judge config section (collapsible) ---
        with st.expander("评测配置", expanded=True):
            # API config
            with st.expander("API 配置", expanded=False):
                api_col1, api_col2, api_col3, api_col4 = st.columns(4)
                with api_col1:
                    judge_api_key = st.text_input("API Key", type="password", value=os.getenv("JUDGE_API_KEY", ""), key="judge_api_key")
                with api_col2:
                    judge_base_url = st.text_input("Base URL", value=os.getenv("JUDGE_API_BASE", "https://token-plan-cn.xiaomimimo.com/v1"), key="judge_base_url")
                with api_col3:
                    judge_model = st.text_input("Model", value=os.getenv("JUDGE_MODEL", "mimo-v2.5-pro"), key="judge_model")
                with api_col4:
                    judge_timeout = st.number_input(
                        "超时（秒）", min_value=10, max_value=180, value=60, step=10,
                        help="单次 LLM 请求的最大等待时间", key="judge_timeout",
                    )
                if st.button("测试 Judge 连接", key="judge_test_conn"):
                    if not judge_api_key:
                        st.error("请先输入 API Key")
                    else:
                        with st.status("正在测试连接...", expanded=True) as status:
                            try:
                                resp = call_llm('请只输出 JSON：{"ok": true}', judge_api_key, judge_base_url, judge_model, timeout=15)
                                status.update(label="连接成功", state="complete")
                                st.code(resp[:200])
                            except Exception as e:
                                status.update(label="连接失败", state="error")
                                st.error(str(e))

            # === 第一层：评测范围与模式 ===
            st.markdown("##### 评测范围")
            total_available = len(samples)

            # 评测轨道筛选
            track_filter_options = ["全部"]
            if track_counts[TRACK_RETRIEVAL] > 0:
                track_filter_options.append(f"仅检索评测题（{track_counts[TRACK_RETRIEVAL]} 条）")
            if track_counts[TRACK_STRICT_QA] > 0:
                track_filter_options.append(f"仅严格问答（{track_counts[TRACK_STRICT_QA]} 条）")
            if track_counts[TRACK_GROUNDED_QA] > 0:
                track_filter_options.append(f"仅合理性问答（{track_counts[TRACK_GROUNDED_QA]} 条）")
            if track_counts[TRACK_NOT_EVALUABLE] > 0:
                track_filter_options.append(f"仅缺少金标准（{track_counts[TRACK_NOT_EVALUABLE]} 条）")

            scope_col1, scope_col2, scope_col3 = st.columns(3)
            with scope_col1:
                track_filter = st.selectbox(
                    "评测轨道筛选",
                    options=track_filter_options,
                    index=0,
                    key="track_filter",
                    help="按评测轨道筛选样本，筛选后样本数、执行计划、实际结果必须一致"
                )

                # 根据筛选过滤样本
                if "检索评测题" in track_filter:
                    filtered_samples = [s for s in samples if classify_evaluation_track(s) == TRACK_RETRIEVAL]
                elif "严格问答" in track_filter:
                    filtered_samples = [s for s in samples if classify_evaluation_track(s) == TRACK_STRICT_QA]
                elif "合理性问答" in track_filter:
                    filtered_samples = [s for s in samples if classify_evaluation_track(s) == TRACK_GROUNDED_QA]
                elif "缺少金标准" in track_filter:
                    filtered_samples = [s for s in samples if classify_evaluation_track(s) == TRACK_NOT_EVALUABLE]
                else:
                    filtered_samples = samples

                filtered_count = len(filtered_samples)
                st.caption(f"筛选后样本数：**{filtered_count}** 条")

            with scope_col2:
                debug_limit = st.checkbox(
                    "只评前 1 条（快速测试）", value=False, key="debug_limit",
                    help="勾选后「评测样本数」不生效，仅评测第 1 条样本"
                )
                if debug_limit:
                    st.number_input(
                        "评测样本数", min_value=1, max_value=max(filtered_count, 1),
                        value=1, key="max_samples", disabled=True,
                    )
                else:
                    st.number_input(
                        "评测样本数", min_value=1, max_value=max(filtered_count, 1),
                        value=filtered_count, key="max_samples",
                    )
            with scope_col3:
                existing_success_count = sum(
                    1 for r in existing_results_map.values() if "error" not in r
                )

                eval_mode = st.radio(
                    "已有结果处理方式",
                    options=["skip", "rerun_all"],
                    format_func=lambda x: {
                        "skip": "跳过已有成功结果（推荐）",
                        "rerun_all": "强制重新评测全部样本",
                    }[x],
                    index=0,
                    key="eval_mode",
                    help="跳过模式：已有成功结果的样本不会重复消耗 token；强制模式：忽略所有缓存，全部重跑",
                )
                skip_existing = (eval_mode == "skip")
                force_rerun = (eval_mode == "rerun_all")

            # 统一从 session_state 读取，确保所有下游使用同一个值
            max_samples = st.session_state.get("max_samples", filtered_count)
            effective_count = 1 if debug_limit else max_samples

            # 候选样本数说明（放在控件区外，确保使用同一 effective_count）
            if debug_limit:
                st.caption("快速测试模式，仅评测第 1 条样本")
            else:
                st.caption(f"从 {filtered_count} 条筛选后样本中取前 **{effective_count}** 条作为候选")

            # === 第二层：高级选项 ===
            with st.expander("高级选项", expanded=False):
                show_debug = st.checkbox("显示 Judge Prompt 和原始响应", key="show_debug")

            # === 本次评测执行计划 ===
            st.markdown("---")
            st.markdown("##### 本次评测执行计划")

            # 模拟真实筛选逻辑
            _preview_candidates = filtered_samples[:effective_count]
            _preview_will_judge = []
            _preview_will_skip = []
            _preview_will_retry = []
            for _s in _preview_candidates:
                _tid = _s.get("trace_id")
                _existing = existing_results_map.get(_tid)
                if _existing and "error" not in _existing and skip_existing and not force_rerun:
                    _preview_will_skip.append(_s)
                elif _existing and "error" in _existing:
                    _preview_will_retry.append(_s)
                    _preview_will_judge.append(_s)
                else:
                    _preview_will_judge.append(_s)

            # --- 上半部分：候选样本来源 ---
            if debug_limit:
                st.markdown(f"**候选样本**：快速测试模式，仅取第 1 条")
            else:
                st.markdown(f"**候选样本**：从当前样本（共 {total_available} 条）中取前 **{effective_count}** 条")

            # 候选样本的评测模式构成
            _cand_with_ref = sum(1 for s in _preview_candidates if (s.get("reference_answer") or "").strip())
            _cand_no_ref = len(_preview_candidates) - _cand_with_ref
            if _cand_with_ref > 0 and _cand_no_ref > 0:
                st.caption(f"其中 {_cand_with_ref} 条走严格评测，{_cand_no_ref} 条走合理性评测")
            elif _cand_with_ref > 0:
                st.caption(f"全部 {_cand_with_ref} 条走严格评测（均有参考答案）")
            elif _cand_no_ref > 0:
                st.caption(f"全部 {_cand_no_ref} 条走合理性评测（均无参考答案）")

            # --- 下半部分：与历史结果的交叉分析 ---
            existing_success_count = sum(
                1 for r in existing_results_map.values() if "error" not in r
            )
            existing_error_count = sum(
                1 for r in existing_results_map.values() if "error" in r
            )
            total_historical = len(existing_results_map)

            if total_historical > 0:
                # 历史结果存在 — 展示交叉分析
                st.markdown(
                    f"**历史评测记录**：`{JUDGED_FILE.name}` 中已有 "
                    f"**{existing_success_count}** 条成功 + **{existing_error_count}** 条失败"
                )

                # 交叉匹配
                hit_count = len(_preview_will_skip) + len(_preview_will_retry)
                if hit_count > 0:
                    match_detail = []
                    if _preview_will_skip:
                        match_detail.append(f"{len(_preview_will_skip)} 条命中成功结果")
                    if _preview_will_retry:
                        match_detail.append(f"{len(_preview_will_retry)} 条命中失败结果")
                    st.markdown(
                        f"**交叉匹配**：{effective_count} 条候选中 "
                        f"**{hit_count}** 条在历史记录中找到（{'，'.join(match_detail)}），"
                        f"**{effective_count - hit_count}** 条为全新样本"
                    )
                else:
                    st.markdown(f"**交叉匹配**：{effective_count} 条候选均为全新样本，历史记录中无匹配")

                # 最终执行数
                if force_rerun:
                    st.markdown(f"**本次执行**：强制重评模式 → 全部 **{effective_count}** 条进入 Judge")
                elif skip_existing:
                    st.markdown(
                        f"**本次执行**：跳过模式 → 跳过 {len(_preview_will_skip)} 条已有成功结果"
                        + (f"，重试 {len(_preview_will_retry)} 条失败结果" if _preview_will_retry else "")
                        + f" → 实际调用 LLM **{len(_preview_will_judge)}** 条"
                    )
                else:
                    st.markdown(f"**本次执行**：全部 **{effective_count}** 条进入 Judge")
            else:
                # 无历史结果
                st.markdown(f"**历史评测记录**：暂无（`{JUDGED_FILE.name}` 不存在或为空）")
                st.markdown(f"**本次执行**：全部 **{effective_count}** 条将作为新样本进入 Judge")

            # --- 候选样本逐条预览 ---
            with st.expander("查看候选样本明细（点击展开）", expanded=False):
                if not _preview_candidates:
                    st.info("没有候选样本")
                else:
                    for _idx, _s in enumerate(_preview_candidates):
                        _q = (_s.get("question") or "(无问题)")[:60]
                        _has_ref = bool((_s.get("reference_answer") or "").strip())
                        _mode_tag = "严格" if _has_ref else "合理性"
                        _existing = existing_results_map.get(_s.get("trace_id"))
                        if _existing and "error" not in _existing and skip_existing and not force_rerun:
                            st.caption(f"  ⏭️ {_idx+1}. `{_q}` — 历史成功，将跳过 [{_mode_tag}]")
                        elif _existing and "error" in _existing:
                            st.caption(f"  🔄 {_idx+1}. `{_q}` — 历史失败，将重试 [{_mode_tag}]")
                        else:
                            st.caption(f"  ✅ {_idx+1}. `{_q}` — 新样本，将评测 [{_mode_tag}]")

            st.markdown("---")

            # === Prompt 示例（独立可查看） ===
            with st.expander("Prompt 示例（点击展开）", expanded=False):
                st.caption("系统会根据题目类型和金标准自动选择 Prompt，无需手动选择。")

                # 按 evaluation_track 分组筛选样本
                _sample_retrieval = next((s for s in _preview_candidates if classify_evaluation_track(s) == TRACK_RETRIEVAL), None)
                _sample_strict_qa = next((s for s in _preview_candidates if classify_evaluation_track(s) == TRACK_STRICT_QA), None)
                _sample_grounded_qa = next((s for s in _preview_candidates if classify_evaluation_track(s) == TRACK_GROUNDED_QA), None)

                # 统计各轨道数量
                _track_counts = {
                    TRACK_RETRIEVAL: sum(1 for s in _preview_candidates if classify_evaluation_track(s) == TRACK_RETRIEVAL),
                    TRACK_STRICT_QA: sum(1 for s in _preview_candidates if classify_evaluation_track(s) == TRACK_STRICT_QA),
                    TRACK_GROUNDED_QA: sum(1 for s in _preview_candidates if classify_evaluation_track(s) == TRACK_GROUNDED_QA),
                }

                def _show_prompt_for_track(sample, track_label, track_desc):
                    """展示单条样本的 prompt 示例。"""
                    if not sample:
                        st.info(f"当前样本中暂无{track_label}题目")
                        return
                    _q = (sample.get("question") or "(无问题)")[:60]

                    # 构建样本标题
                    if track_label == "检索命中":
                        _title_suffix = "检索命中评测（TopK）"
                    elif track_label == "回答正确性":
                        _title_suffix = "回答正确性评测"
                    else:
                        _title_suffix = "回答有据性评测"

                    st.markdown(f"**示例样本**：`{_q}` — {_title_suffix}")

                    # 显示金标准来源（仅检索评测）
                    if track_label == "检索命中":
                        _source_excerpt = (sample.get("source_excerpt") or "").strip()
                        _reference_answer = (sample.get("reference_answer") or "").strip()
                        if _source_excerpt:
                            st.caption(f"金标准来源：source_excerpt")
                        elif _reference_answer:
                            st.caption(f"金标准来源：reference_answer（次级）")

                    prompt = build_judge_prompt(sample)
                    st.code(prompt, language=None)
                    st.caption(f"prompt 长度：{len(prompt)} 字符")

                # 构建 tabs
                _tab_names = []
                if _track_counts[TRACK_RETRIEVAL] > 0:
                    _tab_names.append(f"检索命中 Prompt（{_track_counts[TRACK_RETRIEVAL]} 条）")
                if _track_counts[TRACK_STRICT_QA] > 0:
                    _tab_names.append(f"回答正确性 Prompt（{_track_counts[TRACK_STRICT_QA]} 条）")
                if _track_counts[TRACK_GROUNDED_QA] > 0:
                    _tab_names.append(f"回答有据性 Prompt（{_track_counts[TRACK_GROUNDED_QA]} 条）")

                if _tab_names:
                    tabs = st.tabs(_tab_names)
                    tab_idx = 0

                    # 检索命中 Prompt
                    if _track_counts[TRACK_RETRIEVAL] > 0:
                        with tabs[tab_idx]:
                            st.caption("仅判断正确证据是否进入 Top1 / Top3 / Top5，不评判最终回答质量")
                            _show_prompt_for_track(_sample_retrieval, "检索命中", "检索命中评测")
                        tab_idx += 1

                    # 回答正确性 Prompt
                    if _track_counts[TRACK_STRICT_QA] > 0:
                        with tabs[tab_idx]:
                            st.caption("有参考答案，判断最终回答是否正确、完整")
                            _show_prompt_for_track(_sample_strict_qa, "回答正确性", "回答正确性评测")
                        tab_idx += 1

                    # 回答有据性 Prompt
                    if _track_counts[TRACK_GROUNDED_QA] > 0:
                        with tabs[tab_idx]:
                            st.caption("无参考答案，判断最终回答是否被检索内容支持")
                            _show_prompt_for_track(_sample_grounded_qa, "回答有据性", "回答有据性评测")
                else:
                    st.info("当前无候选样本")

            # === 第三层：执行动作 ===
            # 计算重试按钮上下文
            failed_count = sum(1 for r in existing_results_map.values() if "error" in r)
            retry_label = "只重试失败样本" if failed_count == 0 else f"只重试失败样本（{failed_count} 条）"
            retry_disabled = (failed_count == 0)

            btn_preview, btn_run, btn_retry = st.columns([2, 3, 2])
            with btn_preview:
                preview_optimization = st.button(
                    "预览优化策略",
                    use_container_width=True,
                    help="查看实际需要调用 LLM 的次数，不消耗 token",
                )
            with btn_run:
                run_judge = st.button(
                    "运行 Judge 评测",
                    type="primary",
                    use_container_width=True,
                    help="按当前配置正式开始评测",
                )
            with btn_retry:
                if retry_disabled:
                    st.button(
                        retry_label,
                        use_container_width=True,
                        disabled=True,
                        help="暂无失败样本，无需重试",
                    )
                    retry_failed = False
                else:
                    retry_failed = st.button(
                        retry_label,
                        use_container_width=True,
                        help="仅重新评测之前失败的样本，不影响成功结果",
                    )

        def _load_existing_for_session():
            if "judge_results" not in st.session_state and existing_results_map:
                st.session_state["judge_results"] = list(existing_results_map.values())
                st.session_state["judge_results_source"] = "historical"

        def _compute_samples_to_judge(all_samples, limit, skip, force):
            """根据策略筛选需要评测的样本。返回 (samples_to_judge, skipped_count)"""
            candidates = all_samples[:limit]
            if force or not skip:
                return candidates, 0
            need_judge = []
            for s in candidates:
                tid = s.get("trace_id")
                existing = existing_results_map.get(tid)
                if existing and "error" not in existing:
                    continue  # 已有成功结果，跳过
                need_judge.append(s)
            return need_judge, len(candidates) - len(need_judge)

        def _merge_and_save(new_results):
            """按 trace_id 合并：新结果覆盖旧结果，未重跑的成功结果保留。"""
            from datetime import datetime
            merged = dict(existing_results_map)
            for r in new_results:
                tid = r.get("trace_id")
                if tid:
                    merged[tid] = {
                        k: v for k, v in r.items()
                        if k not in ("_prompt", "_raw_response", "_prescreened", "_content_cached")
                    }
            JUDGED_DIR.mkdir(parents=True, exist_ok=True)

            # 保存当前工作文件（页面读取用）
            with JUDGED_FILE.open("w", encoding="utf-8") as f:
                for r in merged.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 保存带时间戳的历史快照
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = JUDGED_DIR / f"eval_results_{ts}.jsonl"
            with history_file.open("w", encoding="utf-8") as f:
                for r in merged.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            st.session_state["judge_results"] = list(merged.values())
            st.session_state["judge_results_source"] = "fresh_run"
            st.session_state["judge_results_run_count"] = len(new_results)
            return merged, history_file.name

        def _run_judge_ui(samples_to_judge, label="Judge 评测"):
            """通用的评测执行 + 进度 UI。返回新结果列表。"""
            if not samples_to_judge:
                st.info("没有需要评测的样本（全部已有成功结果）")
                return []

            st.info(f"💡 本次共 **{len(samples_to_judge)}** 条样本，经规则预筛选和内容去重后，实际 LLM 请求数可能更少。")

            with st.status(f"正在运行 {label}...", expanded=True) as eval_status:
                progress_bar = st.progress(0, text="准备开始评测...")
                status_text = st.empty()
                question_text = st.empty()
                live_result_area = st.container()
                status_text.write("⏳ 状态：准备开始")

                def on_progress(done, total, result):
                    progress_bar.progress(
                        done / total,
                        text=f"评测进度: {done}/{total}",
                    )
                    question_text.caption(
                        f"当前问题: {(result.get('question') or '')[:80]}"
                    )
                    if "error" in result:
                        status_text.error(
                            f"⏳ 状态：正在评测第 {done}/{total} 条 — 出错: "
                            f"{result['error'][:120]}"
                        )
                    else:
                        status_text.success(
                            f"⏳ 状态：正在评测第 {done}/{total} 条 — 完成"
                        )

                new_results = []
                llm_call_count = 0
                for result in judge_all(
                    samples_to_judge, judge_api_key, judge_base_url,
                    judge_model, on_progress, timeout=judge_timeout,
                ):
                    new_results.append(result)
                    is_prescreened = result.get("_prescreened", False)
                    is_cached = result.get("_content_cached", False)
                    if not is_prescreened and not is_cached:
                        llm_call_count += 1
                    with live_result_area:
                        _r = result
                        tag = ""
                        if is_prescreened:
                            tag = " [规则判定]"
                        elif is_cached:
                            tag = " [内容复用]"
                        if "error" in _r:
                            st.warning(
                                f"❌ [{len(new_results)}] {(_r.get('question') or '')[:40]} — "
                                f"{_r['error'][:100]}{tag}"
                            )
                        else:
                            t1 = "✓" if _r.get("retrieval_top1_hit") else "✗"
                            t3 = "✓" if _r.get("retrieval_top3_hit") else "✗"
                            ans = "✓" if _r.get("answer_correct") else "✗"
                            st.write(
                                f"✅ [{len(new_results)}] {(_r.get('question') or '')[:40]} — "
                                f"Top1:{t1} Top3:{t3} Answer:{ans}{tag}"
                            )
                    if show_debug:
                        with st.expander(
                            f"调试 - 第 {len(new_results)} 条: "
                            f"{(result.get('question') or '')[:40]}"
                        ):
                            st.markdown("**Judge Prompt**")
                            st.code(result.get("_prompt", "(未记录)"), language=None)
                            st.markdown("**原始响应**")
                            st.code(
                                result.get("_raw_response", "(未记录)"), language=None
                            )
                            if "error" in result:
                                st.error(result["error"])

            prescreened_count = sum(1 for r in new_results if r.get("_prescreened"))
            cached_count = sum(1 for r in new_results if r.get("_content_cached"))
            if prescreened_count or cached_count:
                eval_status.update(label=f"{label}完成 — "
                    f"共 {len(new_results)} 条, "
                    f"LLM 调用 {llm_call_count} 次, "
                    f"规则判定 {prescreened_count} 条, "
                    f"内容复用 {cached_count} 条")

            return new_results

        # ---------- 预览优化策略 ----------
        if preview_optimization:
            candidates = filtered_samples[:max_samples]

            prescreen_results = []   # (sample, prescreen_result)
            need_llm = []
            content_seen = {}        # hash -> sample
            trace_skipped = 0
            content_dup_count = 0

            for s in candidates:
                tid = s.get("trace_id")
                existing = existing_results_map.get(tid)
                if existing and "error" not in existing and skip_existing and not force_rerun:
                    trace_skipped += 1
                    continue

                ps = pre_screen(s)
                if ps is not None:
                    prescreen_results.append((s, ps))
                    continue

                ch = compute_content_hash(s)
                if ch in content_seen:
                    content_dup_count += 1
                    continue
                content_seen[ch] = s
                need_llm.append(s)

            total = len(candidates)
            skipped_total = total - len(need_llm)

            # ===== 核心结论 =====
            st.subheader(f"按下「运行 Judge 评测」后，实际需要调用 LLM **{len(need_llm)}** 次")
            st.caption(f"（共 {total} 条候选样本，可节省 {skipped_total} 次请求）")

            # ===== 跳过明细 =====
            if skipped_total > 0:
                st.markdown("#### 以下样本会被跳过，不消耗 token：")
                if trace_skipped > 0:
                    st.markdown(f"**{trace_skipped} 条 — 已有成功评测结果**（trace_id 命中缓存）")
                    skipped_samples = [s for s in candidates
                                       if existing_results_map.get(s.get("trace_id"))
                                       and "error" not in existing_results_map.get(s.get("trace_id"), {})
                                       and skip_existing and not force_rerun]
                    for s in skipped_samples[:5]:
                        q = (s.get("question") or "(无问题)")[:60]
                        st.caption(f"  - `{q}`")
                    if trace_skipped > 5:
                        st.caption(f"  - ...还有 {trace_skipped - 5} 条")

                if prescreen_results:
                    st.markdown(f"**{len(prescreen_results)} 条 — 规则直接判定**（无检索结果/无回答，结果确定，不需要 LLM）")
                    for s, ps in prescreen_results[:5]:
                        q = (s.get("question") or "(无问题)")[:60]
                        st.caption(f"  - `{q}` → {ps.get('reason', '')}")
                    if len(prescreen_results) > 5:
                        st.caption(f"  - ...还有 {len(prescreen_results) - 5} 条")

                if content_dup_count > 0:
                    st.markdown(f"**{content_dup_count} 条 — 内容重复**（question + 回答 完全相同，复用首次评测结果）")

            # ===== 需要 LLM 的样本 =====
            if need_llm:
                _llm_with_ref = sum(1 for s in need_llm if (s.get("reference_answer") or "").strip())
                _llm_no_ref = len(need_llm) - _llm_with_ref
                _mode_desc = []
                if _llm_with_ref:
                    _mode_desc.append(f"{_llm_with_ref} 条严格评测")
                if _llm_no_ref:
                    _mode_desc.append(f"{_llm_no_ref} 条合理性评测")
                st.markdown(f"#### 以下 {len(need_llm)} 条样本需要调用 LLM（{'，'.join(_mode_desc)}）：")
                for s in need_llm[:5]:
                    q = (s.get("question") or "(无问题)")[:60]
                    retrieval_count = len(s.get("retrieval_results", []))
                    answer_preview = (s.get("final_answer") or "(无)")[:40]
                    st.caption(f"  - `{q}` | 检索 {retrieval_count} 条 | 回答: {answer_preview}")
                if len(need_llm) > 5:
                    st.caption(f"  - ...还有 {len(need_llm) - 5} 条")

                # prompt 长度预览
                st.markdown("#### Prompt 长度预览")
                sample_preview = need_llm[0]
                _has_ref = bool((sample_preview.get("reference_answer") or "").strip())

                # 标注当前示例代表哪种评测模式
                _q_preview = (sample_preview.get("question") or "(无问题)")[:50]
                if _has_ref:
                    st.caption(f"当前示例样本：`{_q_preview}` — **严格评测**（含参考答案，使用含参考答案模板）")
                else:
                    st.caption(f"当前示例样本：`{_q_preview}` — **合理性评测**（无参考答案，使用基础模板）")

                # 选择与样本匹配的模板（和 build_judge_prompt 内部逻辑一致）
                if _has_ref:
                    template = load_prompt_template_with_ref()
                else:
                    template = load_prompt_template()

                # 真正的原始版本：未清洗 metadata、未截断
                _raw_results = sample_preview.get("retrieval_results") or []
                if _raw_results:
                    _raw_lines = []
                    for _r in _raw_results:
                        _t = _r.get("title") or ""
                        _c = _r.get("content") or ""
                        _s = _r.get("score")
                        _p = _r.get("position")
                        _prefix = f"[{_p}]" if _p is not None else ""
                        _score = f" (score: {_s})" if _s is not None else ""
                        _raw_lines.append(f"{_prefix}{_t}{_score}: {_c}")
                    _raw_retrieval_text = "\n".join(_raw_lines)
                else:
                    _raw_retrieval_text = "(无检索结果)"
                raw_prompt = template
                raw_prompt = raw_prompt.replace("{question}", sample_preview.get("question") or "(无)")
                raw_prompt = raw_prompt.replace("{retrieval_query}", sample_preview.get("retrieval_query") or "(无)")
                raw_prompt = raw_prompt.replace("{retrieval_results}", _raw_retrieval_text)
                raw_prompt = raw_prompt.replace("{final_answer}", sample_preview.get("final_answer") or "(无)")
                if _has_ref:
                    raw_prompt = raw_prompt.replace("{reference_answer}", sample_preview.get("reference_answer") or "(无)")
                    raw_prompt = raw_prompt.replace("{source_excerpt}", sample_preview.get("source_excerpt") or "(无)")

                # 实际版本：清洗 metadata + 分层截断（build_judge_prompt 内部会自动选模板）
                actual_prompt = build_judge_prompt(sample_preview)

                save_chars = len(raw_prompt) - len(actual_prompt)
                pct = save_chars / len(raw_prompt) * 100 if len(raw_prompt) > 0 else 0
                st.caption(
                    f"原始 {len(raw_prompt)} 字符 → 清洗+截断后 {len(actual_prompt)} 字符"
                    f"（省 {pct:.0f}%）。"
                    f"策略：去除 metadata 块，分层保留正文 — "
                    f"Top-1: 2000字，Top-2/3: 1200字，Top-4/5: 1000字"
                )
                with st.expander("查看处理后的 prompt 示例"):
                    st.code(actual_prompt, language=None)
            else:
                st.success("所有样本都已被跳过或规则判定，不需要调用 LLM！")

            st.divider()

        # ---------- 按钮 1: 运行 Judge 评测 ----------
        if run_judge:
            if not judge_api_key:
                st.error("请在上方「API 配置」中输入 API Key")
            elif not judge_model:
                st.error("请在上方「API 配置」中输入 Model 名称")
            else:
                if force_rerun:
                    _load_existing_for_session()
                    samples_to_judge = filtered_samples[:max_samples]
                    skipped = 0
                else:
                    samples_to_judge, skipped = _compute_samples_to_judge(
                        filtered_samples, max_samples, skip_existing, force_rerun
                    )
                if skipped > 0:
                    st.info(f"⏭️ 跳过 {skipped} 条已有成功结果的样本（可取消「跳过已有成功结果」或勾选「强制重新评测」）")

                new_results = _run_judge_ui(samples_to_judge)
                _, history_name = _merge_and_save(new_results)
                st.success(f"评测完成！结果已保存到 {JUDGED_FILE.name}，历史快照: {history_name}")

        # ---------- 按钮 2: 只重试失败样本 ----------
        if retry_failed:
            if not judge_api_key:
                st.error("请在上方「API 配置」中输入 API Key")
            elif not judge_model:
                st.error("请在上方「API 配置」中输入 Model 名称")
            elif not existing_results_map:
                st.warning("没有找到已有评测结果，请先运行一次 Judge 评测")
            else:
                # 从已有结果中找 error 样本
                failed_trace_ids = {
                    tid for tid, r in existing_results_map.items() if "error" in r
                }
                if not failed_trace_ids:
                    st.success("没有失败的样本，所有评测均已成功！")
                else:
                    # 从原始 samples 中找回对应的 sample
                    failed_samples = [
                        s for s in samples if s.get("trace_id") in failed_trace_ids
                    ]
                    st.info(
                        f"🔄 找到 {len(failed_samples)} 条失败样本，"
                        f"预计消耗 **{len(failed_samples)}** 次 LLM 请求。"
                    )
                    new_results = _run_judge_ui(failed_samples, label="失败样本重试")
                    if new_results:
                        _, history_name = _merge_and_save(new_results)
                        st.success(f"重试完成！结果已合并保存到 {JUDGED_FILE.name}，历史快照: {history_name}")

        # Load existing judge results if not in session
        _load_existing_for_session()

        judge_results = st.session_state.get("judge_results") or []

        # 构建 trace_id -> sample 的查找表，用于结果详情展示原始数据
        _sample_map = {s.get("trace_id"): s for s in (samples or []) if s.get("trace_id")}

        if not judge_results:
            st.info("请在上方配置 API 后点击「运行 Judge 评测」")
        else:
            metrics = compute_metrics(judge_results)
            valid_results = [r for r in judge_results if "error" not in r]

            # ---------- Top5 提示 ----------
            st.caption(
                "💡 如果每题实际只召回 3 条检索结果，则 Top5 指标仅供参考；"
                "严格来说需要把 Dify 检索 topK 调到 5 后重新测试。"
            )

            # ---------- 指标数据来源标注 ----------
            _results_source = st.session_state.get("judge_results_source", "historical")
            _run_count = st.session_state.get("judge_results_run_count", 0)

            if _results_source == "fresh_run":
                st.success(
                    f"以下指标包含本次新评测的 **{_run_count}** 条结果"
                    f"（合并历史记录后共 {len(judge_results)} 条）"
                )
            else:
                _file_mtime = ""
                if JUDGED_FILE.exists():
                    from datetime import datetime
                    _ts = JUDGED_FILE.stat().st_mtime
                    _file_mtime = datetime.fromtimestamp(_ts).strftime("%Y-%m-%d %H:%M")
                st.warning(
                    f"以下指标来自历史记录 `{JUDGED_FILE.name}`"
                    + (f"（最后更新: {_file_mtime}）" if _file_mtime else "")
                    + f"，共 **{len(judge_results)}** 条结果"
                    + "。如需最新结果，请运行 Judge 评测。"
                )

            # ---------- 指标卡片 ----------
            st.subheader("评测指标")

            # 概览
            ov1, ov2, ov3 = st.columns(3)
            ov1.metric("总样本数", metrics["total"])
            ov2.metric("有效评测数", metrics["evaluated"])
            ov3.metric("错误数", metrics["errors"])

            # 无法归类的历史结果提示
            if _unmatched_results:
                st.warning(f"有 **{len(_unmatched_results)}** 条历史结果无法关联当前样本，已归入「历史/无法归类」视图")

            # 按评测轨道分组展示指标
            retrieval_count = metrics.get("retrieval_track_count", 0)
            strict_qa_count = metrics.get("strict_qa_track_count", 0)
            grounded_qa_count = metrics.get("grounded_qa_track_count", 0)
            not_evaluable_count = metrics.get("retrieval_not_evaluable_count", 0)

            has_retrieval = retrieval_count > 0
            has_strict_qa = strict_qa_count > 0
            has_grounded_qa = grounded_qa_count > 0
            has_not_evaluable = not_evaluable_count > 0

            # 检索评测区块：Top1/Top3/Top5 为正式核心指标
            if has_retrieval:
                st.markdown("##### 检索评测指标")
                ret_col1, ret_col2, ret_col3, ret_col4 = st.columns(4)
                ret_col1.metric("可评测样本数", retrieval_count)
                ret_col2.metric("Top1 Hit", f"{metrics['retrieval_top1_hit_rate']:.0%}" if metrics['retrieval_top1_hit_rate'] is not None else "N/A")
                ret_col3.metric("Top3 Hit", f"{metrics['retrieval_top3_hit_rate']:.0%}" if metrics['retrieval_top3_hit_rate'] is not None else "N/A")
                ret_col4.metric("Top5 Hit", f"{metrics['retrieval_top5_hit_rate']:.0%}" if metrics['retrieval_top5_hit_rate'] is not None else "N/A")
                st.caption("检索命中率为正式核心指标，用于评估 RAG 检索链路质量")
                if has_not_evaluable:
                    st.warning(f"有 **{not_evaluable_count}** 条检索评测题缺少金标准证据，不纳入检索命中率计算")

            # 严格问答区块：Answer Correctness 为正式核心指标
            if has_strict_qa:
                st.markdown("##### 严格问答指标")
                qa_col1, qa_col2 = st.columns(2)
                qa_col1.metric("样本数", strict_qa_count)
                qa_col2.metric("Answer Correctness", f"{metrics['strict_qa_answer_rate']:.0%}" if metrics['strict_qa_answer_rate'] is not None else "N/A")
                st.caption("有参考答案，评判回答是否与参考答案一致")

                # 检索诊断（辅助）：仅当有 source_excerpt 时显示
                _strict_with_excerpt = sum(1 for r in judge_results
                                          if r.get("evaluation_track") == TRACK_STRICT_QA
                                          and (r.get("source_excerpt") or "").strip())
                if _strict_with_excerpt > 0:
                    with st.expander("检索诊断（辅助）", expanded=False):
                        st.caption("以下检索指标用于定位回答错误是否由检索失败造成，不作为严格回答题的正式结论")
                        diag_col1, diag_col2, diag_col3 = st.columns(3)
                        # 计算有 source_excerpt 的严格问答样本的 TopK
                        _strict_with_excerpt_results = [r for r in valid_results
                                                        if r.get("evaluation_track") == TRACK_STRICT_QA
                                                        and (r.get("source_excerpt") or "").strip()]
                        if _strict_with_excerpt_results:
                            _n = len(_strict_with_excerpt_results)
                            _t1 = sum(r.get("retrieval_top1_hit", 0) for r in _strict_with_excerpt_results) / _n
                            _t3 = sum(r.get("retrieval_top3_hit", 0) for r in _strict_with_excerpt_results) / _n
                            _t5 = sum(r.get("retrieval_top5_hit", 0) for r in _strict_with_excerpt_results) / _n
                            diag_col1.metric("Top1 Hit", f"{_t1:.0%}")
                            diag_col2.metric("Top3 Hit", f"{_t3:.0%}")
                            diag_col3.metric("Top5 Hit", f"{_t5:.0%}")

            # 合理性问答区块：只显示 Answer Grounded，不展示 TopHit
            if has_grounded_qa:
                st.markdown("##### 合理性问答指标")
                gq_col1, gq_col2 = st.columns(2)
                gq_col1.metric("样本数", grounded_qa_count)
                gq_col2.metric("Answer Grounded", f"{metrics['grounded_qa_answer_rate']:.0%}" if metrics['grounded_qa_answer_rate'] is not None else "N/A")
                st.caption("无参考答案，基于检索内容判断回答合理性")

            # 各轨道正式指标总览（不混合口径）
            if has_retrieval or has_strict_qa or has_grounded_qa:
                st.markdown("---")
                st.markdown("##### 各轨道正式指标总览")
                overview_cols = st.columns(3)
                col_idx = 0

                if has_retrieval:
                    with overview_cols[col_idx]:
                        st.markdown("**检索评测**")
                        st.metric("Top1 Hit", f"{metrics['retrieval_top1_hit_rate']:.0%}" if metrics['retrieval_top1_hit_rate'] is not None else "N/A")
                        st.caption(f"样本数: {retrieval_count}")
                    col_idx += 1

                if has_strict_qa:
                    with overview_cols[col_idx]:
                        st.markdown("**严格问答**")
                        st.metric("Answer Correctness", f"{metrics['strict_qa_answer_rate']:.0%}" if metrics['strict_qa_answer_rate'] is not None else "N/A")
                        st.caption(f"样本数: {strict_qa_count}")
                    col_idx += 1

                if has_grounded_qa:
                    with overview_cols[col_idx]:
                        st.markdown("**合理性问答**")
                        st.metric("Answer Grounded", f"{metrics['grounded_qa_answer_rate']:.0%}" if metrics['grounded_qa_answer_rate'] is not None else "N/A")
                        st.caption(f"样本数: {grounded_qa_count}")

            # ---------- 视图切换 + 下游内容（用 tabs 实现） ----------
            def _render_judge_view(view_valid, view_all, metrics_subset, metrics_desc, view_label=""):
                """渲染一个视图下的全部内容：图表、诊断、详情。"""
                st.caption(metrics_desc)

                if metrics["errors"] > 0:
                    st.warning(f"有 {metrics['errors']} 条评测出错")

                # 可视化图表
                st.subheader("可视化")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("**RAG 检索命中率 & LLM 回答正确率**")
                    st.plotly_chart(build_eval_bar_chart(
                        _compute_subset_metrics(view_all, None) or metrics_subset
                    ), use_container_width=True)

                with chart_col2:
                    st.markdown("**LLM 回答：正确 vs 错误**")
                    if view_valid:
                        st.plotly_chart(build_answer_pye(view_valid), use_container_width=True)
                    else:
                        st.info("无有效评测数据")

                st.markdown("**每题检索命中情况**")
                pq_fig = build_per_question_chart(view_valid) if view_valid else None
                if pq_fig:
                    st.plotly_chart(pq_fig, use_container_width=True)
                else:
                    st.info("无有效评测数据")

                # Top1 未命中案例
                top1_miss = [r for r in view_valid if not r.get("retrieval_top1_hit")]
                if top1_miss:
                    st.subheader(f"RAG 检索：Top1 未命中案例 ({len(top1_miss)} 条)")
                    st.caption("以下问题 Top1 未命中 — 说明检索链路可能存在问题（如召回策略、向量相似度、关键词匹配等）")
                    for r in top1_miss:
                        _mode_tag = "参考答案" if r.get("has_reference") else "LLM判断"
                        _tid = r.get("trace_id", "")
                        _sample = _sample_map.get(_tid, {})
                        with st.expander(f"**{r.get('question', '(无问题)')[:60]}** [{_mode_tag}]"):
                            st.markdown(f"**问题**: {r.get('question', '')}")
                            t3 = "✓" if r.get("retrieval_top3_hit") else "✗"
                            t5 = "✓" if r.get("retrieval_top5_hit") else "✗"
                            ans = "✓" if r.get("answer_correct") else "✗"
                            st.markdown(f"**检索命中**: Top1 ✗ | Top3 {t3} | Top5 {t5}　　**回答正确**: {ans}")
                            _final = _sample.get("final_answer") or "(无)"
                            st.markdown("**最终回答**")
                            st.code(_final[:1000], language=None)
                            if r.get("has_reference"):
                                _ref = (_sample.get("reference_answer") or "").strip()
                                if _ref:
                                    st.markdown("**参考答案**")
                                    st.code(_ref[:1000], language=None)
                            st.markdown(f"**Judge 原因**: {r.get('reason', '(无)')}")
                            if r.get("retrieval_top3_hit") and not r.get("retrieval_top1_hit"):
                                st.caption("Top1 未命中但 Top3 命中 — 排序可能有问题，正确结果未排到第一位")
                            elif not r.get("retrieval_top5_hit"):
                                st.caption("Top5 也未命中 — 检索完全未召回正确内容，需检查检索策略")
                            if r.get("answer_correct"):
                                st.caption("虽然检索未命中 Top1，但 LLM 仍给出了正确回答 — 可能靠其他上下文推断")
                            st.caption(f"trace_id: `{_tid}`")

                # 回答错误但检索命中
                answer_wrong = [r for r in view_valid if r.get("retrieval_top1_hit") and not r.get("answer_correct")]
                if answer_wrong:
                    st.subheader(f"LLM 回答：检索命中但回答错误 ({len(answer_wrong)} 条)")
                    st.caption("以下问题检索已命中正确内容，但 LLM 未给出正确回答 — 说明回答生成环节可能存在问题")
                    for r in answer_wrong:
                        _mode_tag = "参考答案" if r.get("has_reference") else "LLM判断"
                        _tid = r.get("trace_id", "")
                        _sample = _sample_map.get(_tid, {})
                        with st.expander(f"**{r.get('question', '(无问题)')[:60]}** [{_mode_tag}]"):
                            st.markdown(f"**问题**: {r.get('question', '')}")
                            _final = _sample.get("final_answer") or "(无)"
                            st.markdown("**最终回答**")
                            st.code(_final[:1000], language=None)
                            if r.get("has_reference"):
                                _ref = (_sample.get("reference_answer") or "").strip()
                                if _ref:
                                    st.markdown("**参考答案**")
                                    st.code(_ref[:1000], language=None)
                                    st.caption("回答与参考答案不一致或遗漏关键点 — 需检查回答生成是否覆盖了参考答案的核心内容")
                            else:
                                st.caption("合理性评测：检索已命中，但 LLM 判断回答不合理 — 可能原因：回答生成模型能力不足、prompt 设计问题、或检索结果干扰")
                            st.markdown(f"**Judge 原因**: {r.get('reason', '(无)')}")
                            st.caption(f"trace_id: `{_tid}`")

                # 评测详情卡
                st.subheader("评测详情")
                if not view_all:
                    st.info("当前视图下无评测样本")
                for _idx, r in enumerate(view_all):
                    _tid = r.get("trace_id", "")
                    _sample = _sample_map.get(_tid, {})
                    _q = r.get("question", "(无问题)")
                    _has_ref = r.get("has_reference", False)
                    _mtag = "严格" if _has_ref else "合理性"
                    _ans_ok = r.get("answer_correct")
                    _t1 = r.get("retrieval_top1_hit")
                    _t3 = r.get("retrieval_top3_hit")
                    _t5 = r.get("retrieval_top5_hit")
                    _status = "✅" if _ans_ok else "❌"
                    _rs = f"T1:{'✓' if _t1 else '✗'} T3:{'✓' if _t3 else '✗'} T5:{'✓' if _t5 else '✗'}"

                    with st.expander(f"{_status} {_q[:55]}{'...' if len(_q) > 55 else ''}  [{_mtag}] {_rs}"):
                        st.markdown(f"**检索命中**: Top1 {'✓ 命中' if _t1 else '✗ 未命中'} | Top3 {'✓ 命中' if _t3 else '✗ 未命中'} | Top5 {'✓ 命中' if _t5 else '✗ 未命中'}")
                        if _t1:
                            st.caption("Top1 检索结果包含回答问题所需的关键信息")
                        elif _t3:
                            st.caption("Top1 未命中，但 Top3 内命中 — 正确结果排在第 2~3 位，排序可能有优化空间")
                        elif _t5:
                            st.caption("Top3 未命中，但 Top5 内命中 — 正确结果排在第 4~5 位，召回但排序较差")
                        else:
                            st.caption("Top5 全未命中 — 检索未召回正确内容，需检查检索策略")
                        st.markdown("---")
                        _final = _sample.get("final_answer") or "(无)"
                        st.markdown("**最终回答**")
                        st.code(_final[:1500], language=None)
                        if _has_ref:
                            _ref = (_sample.get("reference_answer") or "").strip()
                            if _ref:
                                st.markdown("**参考答案**")
                                st.code(_ref[:1500], language=None)
                            _excerpt = (_sample.get("source_excerpt") or "").strip()
                            if _excerpt:
                                with st.expander("来源摘录"):
                                    st.text(_excerpt[:1000])
                        st.markdown(f"**Judge 原因**: {r.get('reason', '(无)')}")
                        st.caption(f"trace_id: `{_tid}` | 评测模式: {_mtag}评测 | answer_correct: {_ans_ok}")

            # ---------- 根据视图过滤数据 ----------
            def _filter_by_view(results, mode):
                if mode == "mixed":
                    return results
                elif mode == "strict":
                    return [r for r in results if bool(r.get("has_reference"))]
                else:
                    return [r for r in results if not bool(r.get("has_reference"))]

            # ---------- 按评测轨道分组展示详情 ----------
            st.subheader("评测详情与可视化")

            # 按评测轨道分组
            def _filter_by_track(results, track):
                return [r for r in results if r.get("evaluation_track") == track]

            # 构建 tabs
            tab_names = []
            if has_retrieval:
                tab_names.append(f"检索评测（{retrieval_count} 条）")
            if has_strict_qa:
                tab_names.append(f"严格问答（{strict_qa_count} 条）")
            if has_grounded_qa:
                tab_names.append(f"合理性问答（{grounded_qa_count} 条）")
            if has_not_evaluable:
                tab_names.append(f"缺少金标准（{not_evaluable_count} 条）")
            if _unmatched_results:
                tab_names.append(f"历史/无法归类（{len(_unmatched_results)} 条）")

            if tab_names:
                tabs = st.tabs(tab_names)
                tab_idx = 0

                # 检索评测详情
                if has_retrieval:
                    with tabs[tab_idx]:
                        _sv = _filter_by_track(valid_results, TRACK_RETRIEVAL)
                        _sa = _filter_by_track(judge_results, TRACK_RETRIEVAL)

                        # 可视化图表：只显示 Top1/Top3/Top5，不显示 Answer Correct
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            st.markdown("**检索命中率（核心指标）**")
                            _ret_metrics = {
                                "top1_hit_rate": metrics.get("retrieval_top1_hit_rate", 0) or 0,
                                "top3_hit_rate": metrics.get("retrieval_top3_hit_rate", 0) or 0,
                                "top5_hit_rate": metrics.get("retrieval_top5_hit_rate", 0) or 0,
                            }
                            st.plotly_chart(build_retrieval_bar_chart(_ret_metrics), use_container_width=True)
                        with chart_col2:
                            st.markdown("**每题检索命中情况**")
                            pq_fig = build_retrieval_per_question_chart(_sv) if _sv else None
                            if pq_fig:
                                st.plotly_chart(pq_fig, use_container_width=True)
                            else:
                                st.info("无有效评测数据")

                        # 检索评测详情
                        st.markdown("##### 检索评测详情")
                        for r in _sa:
                            _tid = r.get("trace_id", "")
                            _sample = _sample_map.get(_tid, {})
                            render_retrieval_result_detail(r, _sample, f"judge_ret_{_tid[:8]}")
                    tab_idx += 1

                # 严格问答详情
                if has_strict_qa:
                    with tabs[tab_idx]:
                        _sv = _filter_by_track(valid_results, TRACK_STRICT_QA)
                        _sa = _filter_by_track(judge_results, TRACK_STRICT_QA)

                        # 可视化图表：只显示 Answer Correctness，不显示 Top1/Top3/Top5
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            st.markdown("**回答正确性（核心指标）**")
                            _qa_metrics = {
                                "answer_correct_rate": metrics.get("strict_qa_answer_rate", 0) or 0,
                            }
                            st.plotly_chart(build_strict_qa_bar_chart(_qa_metrics), use_container_width=True)
                        with chart_col2:
                            st.markdown("**Answer Correct vs Incorrect**")
                            if _sv:
                                st.plotly_chart(build_answer_pye(_sv), use_container_width=True)
                            else:
                                st.info("无有效评测数据")

                        # 检索诊断（辅助）：仅当有 source_excerpt 且有有效 TopK 判定时显示
                        _strict_with_excerpt_results = [r for r in _sv
                                                        if (r.get("source_excerpt") or "").strip()
                                                        and (r.get("retrieval_top1_hit") is not None
                                                             or r.get("retrieval_top3_hit") is not None
                                                             or r.get("retrieval_top5_hit") is not None)]
                        if _strict_with_excerpt_results:
                            with st.expander("检索诊断（辅助）", expanded=False):
                                st.caption("辅助诊断，不计入严格回答正确率；用于定位回答错误是否由检索失败造成。")
                                _n = len(_strict_with_excerpt_results)
                                _t1 = sum(r.get("retrieval_top1_hit", 0) for r in _strict_with_excerpt_results) / _n
                                _t3 = sum(r.get("retrieval_top3_hit", 0) for r in _strict_with_excerpt_results) / _n
                                _t5 = sum(r.get("retrieval_top5_hit", 0) for r in _strict_with_excerpt_results) / _n
                                diag_col1, diag_col2, diag_col3 = st.columns(3)
                                diag_col1.metric("Top1 Hit", f"{_t1:.0%}")
                                diag_col2.metric("Top3 Hit", f"{_t3:.0%}")
                                diag_col3.metric("Top5 Hit", f"{_t5:.0%}")
                                st.caption(f"基于 {_n} 条有 source_excerpt 且有有效 TopK 判定的样本")

                        # 严格问答详情
                        st.markdown("##### 严格问答详情")
                        for r in _sa:
                            _tid = r.get("trace_id", "")
                            _sample = _sample_map.get(_tid, {})
                            render_strict_qa_result_detail(r, _sample, f"judge_strict_{_tid[:8]}")
                    tab_idx += 1

                # 合理性问答详情
                if has_grounded_qa:
                    with tabs[tab_idx]:
                        _sv = _filter_by_track(valid_results, TRACK_GROUNDED_QA)
                        _sa = _filter_by_track(judge_results, TRACK_GROUNDED_QA)

                        # 可视化图表：只显示 Answer Grounded，不显示 TopHit
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            st.markdown("**回答有据性（核心指标）**")
                            _gq_metrics = {
                                "answer_correct_rate": metrics.get("grounded_qa_answer_rate", 0) or 0,
                            }
                            st.plotly_chart(build_grounded_qa_bar_chart(_gq_metrics), use_container_width=True)
                        with chart_col2:
                            st.markdown("**Answer Grounded vs Not Grounded**")
                            if _sv:
                                st.plotly_chart(build_answer_pye(_sv, "有据", "缺乏依据"), use_container_width=True)
                            else:
                                st.info("无有效评测数据")

                        # 合理性问答详情
                        st.markdown("##### 合理性问答详情")
                        for r in _sa:
                            _tid = r.get("trace_id", "")
                            _sample = _sample_map.get(_tid, {})
                            render_grounded_qa_result_detail(r, _sample, f"judge_grounded_{_tid[:8]}")
                    tab_idx += 1

                # 缺少金标准详情
                if has_not_evaluable:
                    with tabs[tab_idx]:
                        _sa = _filter_by_track(judge_results, TRACK_NOT_EVALUABLE)
                        st.warning(f"以下 **{not_evaluable_count}** 条检索评测题缺少金标准证据（source_excerpt 和 reference_answer 均为空），无法可靠计算检索命中率")
                        for r in _sa:
                            _tid = r.get("trace_id", "")
                            _q = r.get("question", "(无问题)")
                            st.caption(f"- `{_q[:60]}` — {r.get('not_evaluable_reason', '')}")
                    tab_idx += 1

                # 历史/无法归类详情
                if _unmatched_results:
                    with tabs[tab_idx]:
                        st.info(f"以下 **{len(_unmatched_results)}** 条历史结果无法关联当前样本，已按旧逻辑归类")
                        for tid in _unmatched_results:
                            r = existing_results_map.get(tid, {})
                            _q = r.get("question", "(无问题)")
                            _track = r.get("evaluation_track", "unknown")
                            _track_label = {
                                TRACK_STRICT_QA: "严格问答",
                                TRACK_GROUNDED_QA: "合理性问答",
                            }.get(_track, "未知")
                            st.caption(f"- `{_q[:60]}` — 归入: {_track_label} | trace_id: `{tid}`")

            # ---------- 导出按钮 ----------
            st.subheader("导出")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                csv_data = build_csv_download(judge_results)
                st.download_button(
                    label="下载 CSV",
                    data=csv_data,
                    file_name="eval_results.csv",
                    mime="text/csv",
                )

            with dl_col2:
                md_report = build_markdown_report(judge_results)
                st.download_button(
                    label="下载 Markdown 报告",
                    data=md_report.encode("utf-8"),
                    file_name="eval_report.md",
                    mime="text/markdown",
                )

# ========== Tab: 运行看板 ==========
with tab_experiment:
    st.subheader("配置与运行看板")
    st.caption("按评测配置查看累计结果、运行历史和单次运行详情。")

    # ---------- 模块说明 ----------
    with st.expander("运行看板说明（点击展开）", expanded=False):
        st.markdown("""
**一句话总览：** 按评测配置查看累计结果、运行历史和单次运行详情。

---

**页面结构**

| 区域 | 说明 |
|------|------|
| **配置方案卡片** | 显示配置名称、知识库版本、工作流版本、检索模式、Top K、Rerank 等摘要；可编辑描述性字段 |
| **配置方案总览** | 聚合当前配置下所有 run 的累计 Judge 指标，按评测轨道分组，按样本数加权汇总 |
| **运行记录** | 每次 run 的 Batch/Raw/Processed/Judge 状态、该 run 的图表和逐题明细 |
| **运行历史** | 所有 run 的时间趋势图和历史表格 |

---

**数据模型**

| 概念 | 说明 | 存储位置 |
|------|------|---------|
| **配置方案** | 可复用的 RAG 配置（知识库版本、检索配置等） | `data/config_profiles/<config_id>.json` |
| **运行记录** | 每次批量提问的运行记录，关联一个配置方案，包含配置快照 | `data/experiments/<run_id>/manifest.json` |

---

**累计指标聚合规则**

- 配置方案总览聚合当前 config 下所有 run 的 Judge 结果
- 指标按有效 Judge 样本数加权汇总（`命中总数 / 有效样本数`），**不是**各 run 百分比的简单平均
- 同一 trace_id 出现多次时，保留最新且无 error 的结果
- retrieval / strict_qa / grounded_qa 分轨道统计，不混合

---

**配置方案编辑**

- 配置方案的描述性字段（知识库版本、Top K、Rerank 模型等）可随时编辑
- 核心关联字段（config_id、created_at）不可编辑
- 每次 run 的 config_snapshot 也可以单独修正（不影响其他 run 或配置方案），修正历史保存在 `snapshot_edit_history` 中

---

**关联链路**

```
run_id → processed sample（真实 Langfuse trace_id）→ Judge result
```

- batch_qa_* 是批量提问生成的文件标识，不是 Langfuse trace_id
- Judge 结果通过 processed sample 的 trace_id 关联
- 历史 Judge 结果没有 run_id 时，通过 trace_id fallback 关联
""")

    # ---------- 导入 ----------
    from experiment import (
        list_config_profiles, list_experiment_runs, list_runs_by_config,
        load_config_profile, get_run_status, get_judge_metrics_by_run,
        backfill_manifest_from_batch, migrate_judged_results, migrate_processed_samples,
        get_config_display_value, get_config_summary,
        EXPERIMENTS_DIR,
    )
    from judge import compute_metrics, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA

    # ---------- 自动迁移：从 batch 文件回填 manifest ----------
    _all_runs = list_experiment_runs()
    _migrated_count = 0
    for _run in _all_runs:
        if not _run.get("question_set_id"):
            if backfill_manifest_from_batch(_run["run_id"], batch_dir=str(BATCH_DIR)):
                _migrated_count += 1
    if _migrated_count > 0:
        st.toast(f"已自动回填 {_migrated_count} 条运行记录的题集信息")

    # ---------- 数据迁移工具 ----------
    with st.expander("数据迁移工具", expanded=False):
        st.caption("为历史数据回填 run_id、config_id 等元数据，便于实验看板关联")
        mig_col1, mig_col2 = st.columns(2)
        with mig_col1:
            if st.button("迁移 Judge 结果（回填 run_id）", key="migrate_judged"):
                with st.spinner("正在迁移..."):
                    result = migrate_judged_results(
                        processed_file=str(PROCESSED_DIR / "langfuse_samples.jsonl"),
                        judged_file=str(JUDGED_FILE),
                        backup=True,
                    )
                    if result["migrated"] > 0:
                        st.success(f"已迁移 {result['migrated']} 条 Judge 结果，备份: {result['backup_path']}")
                    else:
                        st.info("无需迁移或迁移失败")
        with mig_col2:
            if st.button("迁移 Processed 样本（回填 config_id）", key="migrate_processed"):
                with st.spinner("正在迁移..."):
                    result = migrate_processed_samples(
                        processed_file=str(PROCESSED_DIR / "langfuse_samples.jsonl"),
                        experiments_dir=str(EXPERIMENTS_DIR),
                        backup=True,
                    )
                    if result["migrated"] > 0:
                        st.success(f"已迁移 {result['migrated']} 条样本，备份: {result['backup_path']}")
                    else:
                        st.info("无需迁移或迁移失败")

    # ---------- 选择配置方案 ----------
    st.markdown("---")
    st.markdown("##### 选择配置方案")

    configs = list_config_profiles()

    if not configs:
        st.info("暂无配置方案。在「批量提问」页面创建配置后，将自动记录在此。")
        st.stop()

    # 构建下拉选项
    config_options = []
    for cfg in configs:
        runs_count = len(list_runs_by_config(cfg.get("config_id", "")))
        label = f"{cfg.get('config_name', '未命名')} | {cfg.get('knowledge_base_version', '')} | {runs_count} 次运行"
        config_options.append((cfg.get("config_id"), label))

    selected_config_id = st.selectbox(
        "选择配置方案",
        options=[c[0] for c in config_options],
        format_func=lambda x: next((c[1] for c in config_options if c[0] == x), x),
        key="exp_selected_config",
    )

    if not selected_config_id:
        st.stop()

    # ---------- 配置方案卡片 ----------
    selected_config = load_config_profile(selected_config_id)
    if not selected_config:
        st.error(f"配置方案不存在: {selected_config_id}")
        st.stop()

    st.markdown("---")
    st.markdown(f"##### 配置方案: {selected_config.get('config_name', '未命名')}")

    # 摘要字段卡片（使用统一 schema）
    card_col1, card_col2 = st.columns(2)
    with card_col1:
        st.markdown(f"**配置名称**: {get_config_display_value(selected_config, 'config_name')}")
        st.markdown(f"**知识库版本**: {get_config_display_value(selected_config, 'knowledge_base_version')}")
        st.markdown(f"**工作流版本**: {get_config_display_value(selected_config, 'workflow_version')}")
    with card_col2:
        st.markdown(f"**检索模式**: {get_config_display_value(selected_config, 'retrieval_mode')}")
        _topk = get_config_display_value(selected_config, 'top_k')
        _rerank = get_config_display_value(selected_config, 'rerank_model')
        st.markdown(f"**Top K**: {_topk}　**Rerank**: {_rerank}")
        st.markdown(f"**备注**: {get_config_display_value(selected_config, 'notes')}")
    _updated = selected_config.get('updated_at', '')
    _created = selected_config.get('created_at', '')
    st.caption(f"创建时间: {_created[:19] if _created else '未知'}" +
               (f"　最后编辑: {_updated[:19]}" if _updated else ""))

    # 编辑配置说明
    with st.expander("编辑/查看配置详情", expanded=False):
        with st.form("edit_config_form"):
            st.markdown("**可编辑字段**（核心字段 config_id / created_at 不可修改）")
            ec_values = render_config_form(selected_config, key_prefix="ecfg")
            ec_note = st.text_input("修改说明（可选）", value="", key="ec_edit_note",
                                    help="简要说明本次修改原因，如：补录 Rerank 配置")
            ec_submit = st.form_submit_button("保存配置修改", type="primary")

        if ec_submit:
            from experiment import update_config_profile_safe
            updates = collect_config_updates(ec_values)
            update_config_profile_safe(selected_config_id, updates, edit_note=ec_note)
            st.success("配置已保存，config_id 未变。")
            st.rerun()

    # 技术详情（核心字段只读）
    with st.expander("技术详情（只读）", expanded=False):
        st.markdown(f"**config_id**: `{selected_config.get('config_id', '')}`")
        st.markdown(f"**created_at**: {selected_config.get('created_at', '')}")
        if selected_config.get('updated_at'):
            st.markdown(f"**updated_at**: {selected_config['updated_at']}")
        if selected_config.get('edit_note'):
            st.markdown(f"**edit_note**: {selected_config['edit_note']}")
        st.json(selected_config)

    # ---------- 配置方案总览 ----------
    config_runs = list_runs_by_config(selected_config_id)

    if config_runs:
        st.markdown("---")
        st.markdown(f"##### 配置方案总览（{selected_config.get('config_name', '')}）")

        # 收集所有 run 的状态和 Judge 结果
        _all_run_statuses = []
        _all_judge_results_raw = []
        _total_questions = 0
        _total_batch_success = 0
        _total_batch_total = 0
        _total_raw = 0
        _total_processed = 0
        _total_judge = 0
        _status_counts = {}
        _latest_run = None
        _latest_time = ""

        for run in config_runs:
            rid = run.get("run_id", "")
            rs = get_run_status(
                rid,
                batch_dir=str(BATCH_DIR),
                raw_dir=str(RAW_DIR),
                processed_file=str(PROCESSED_DIR / "langfuse_samples.jsonl"),
                judged_file=str(JUDGED_FILE),
            )
            _all_run_statuses.append(rs)
            _total_questions += run.get("question_count", 0)
            _total_batch_success += rs.get("batch_success", 0)
            _total_batch_total += rs.get("batch_total", 0)
            _total_raw += rs.get("raw_count", 0)
            _total_processed += rs.get("processed_count", 0)
            _total_judge += rs.get("judge_count", 0)

            run_status = run.get("status", "unknown")
            _status_counts[run_status] = _status_counts.get(run_status, 0) + 1

            started = run.get("started_at", "")
            if started > _latest_time:
                _latest_time = started
                _latest_run = run

            # 收集 Judge 结果（带 run_id 标记）
            for r in rs.get("judge_results", []):
                r_copy = dict(r)
                r_copy["_source_run_id"] = rid
                _all_judge_results_raw.append(r_copy)

        # 去重：同一 trace_id 保留最新且无 error 的结果
        # 优先级：无 error > 有 error；同优先级时后出现的覆盖先出现的（后出现 = 更新的 run）
        _seen_trace = {}
        for r in _all_judge_results_raw:
            tid = r.get("trace_id", "")
            if not tid:
                continue
            existing = _seen_trace.get(tid)
            if existing is None:
                _seen_trace[tid] = r
            elif "error" in existing and "error" not in r:
                # 新结果无 error，覆盖旧的有 error 结果
                _seen_trace[tid] = r
            else:
                # 后出现的 run 更新，覆盖（即使都有 error 或都无 error）
                _seen_trace[tid] = r
        all_judge_results = list(_seen_trace.values())

        # 概览指标
        ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
        with ov_col1:
            st.metric("总运行次数", len(config_runs))
            _status_parts = [f"{k}: {v}" for k, v in _status_counts.items()]
            if _status_parts:
                st.caption("状态: " + " / ".join(_status_parts))
        with ov_col2:
            st.metric("题目总数", _total_questions)
            st.metric("Batch 成功", f"{_total_batch_success}/{_total_batch_total}")
        with ov_col3:
            st.metric("Raw 总数", _total_raw)
            st.metric("Processed 总数", _total_processed)
        with ov_col4:
            st.metric("Judge 已评测", _total_judge)
            if _latest_run:
                _latest_qs = _latest_run.get("question_set_name", "") or "—"
                st.caption(f"最近运行: {_latest_time[:19]}")
                st.caption(f"题集: {_latest_qs}")

        # 累计 Judge 指标（按 track 加权汇总，去重后）
        valid_all = [r for r in all_judge_results if "error" not in r]
        error_all = [r for r in all_judge_results if "error" in r]

        retrieval_all = [r for r in valid_all if r.get("evaluation_track") == TRACK_RETRIEVAL]
        strict_qa_all = [r for r in valid_all if r.get("evaluation_track") == TRACK_STRICT_QA]
        grounded_qa_all = [r for r in valid_all if r.get("evaluation_track") == TRACK_GROUNDED_QA]

        st.markdown("---")
        st.markdown("**累计 Judge 指标**")
        st.caption("按样本加权汇总（命中总数 / 有效样本数），去重后统计。不同评测轨道不混合。")

        has_any_track = retrieval_all or strict_qa_all or grounded_qa_all
        if not has_any_track:
            st.info("暂无数据")
        else:
            track_col1, track_col2, track_col3 = st.columns(3)

            if retrieval_all:
                with track_col1:
                    st.markdown("**检索评测**")
                    n = len(retrieval_all)
                    t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_all) / n
                    t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_all) / n
                    t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_all) / n
                    st.metric("Top1 Hit", f"{t1:.0%}")
                    st.metric("Top3 Hit", f"{t3:.0%}")
                    st.metric("Top5 Hit", f"{t5:.0%}")
                    st.caption(f"有效样本数 n={n}")
            else:
                with track_col1:
                    st.markdown("**检索评测**")
                    st.info("暂无数据")

            if strict_qa_all:
                with track_col2:
                    st.markdown("**严格问答**")
                    n = len(strict_qa_all)
                    acc = sum(r.get("answer_correct", 0) for r in strict_qa_all) / n
                    st.metric("Answer Correctness", f"{acc:.0%}")
                    st.caption(f"有效样本数 n={n}")
            else:
                with track_col2:
                    st.markdown("**严格问答**")
                    st.info("暂无数据")

            if grounded_qa_all:
                with track_col3:
                    st.markdown("**合理性问答**")
                    n = len(grounded_qa_all)
                    acc = sum(r.get("answer_correct", 0) for r in grounded_qa_all) / n
                    st.metric("Answer Groundedness", f"{acc:.0%}")
                    st.caption(f"有效样本数 n={n}")
            else:
                with track_col3:
                    st.markdown("**合理性问答**")
                    st.info("暂无数据")

        # 累计可视化
        st.markdown("**配置方案累计结果**")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if retrieval_all:
                n = len(retrieval_all)
                _cum_ret_m = {
                    "top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in retrieval_all) / n,
                    "top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in retrieval_all) / n,
                    "top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in retrieval_all) / n,
                }
                st.caption(f"检索命中率 (n={n})")
                fig_cum_ret = build_retrieval_bar_chart(_cum_ret_m)
                st.plotly_chart(fig_cum_ret, use_container_width=True, key="cum_ret_bar")
            else:
                st.info("暂无检索评测数据")

        with chart_col2:
            if strict_qa_all or grounded_qa_all:
                # QA 累计指标图
                qa_labels = []
                qa_values = []
                if strict_qa_all:
                    n = len(strict_qa_all)
                    qa_labels.append(f"严格问答 (n={n})")
                    qa_values.append(sum(r.get("answer_correct", 0) for r in strict_qa_all) / n * 100)
                if grounded_qa_all:
                    n = len(grounded_qa_all)
                    qa_labels.append(f"合理性问答 (n={n})")
                    qa_values.append(sum(r.get("answer_correct", 0) for r in grounded_qa_all) / n * 100)
                fig_qa_cum = go.Figure(data=[go.Bar(
                    x=qa_labels, y=qa_values,
                    marker_color="#17becf",
                    text=[f"{v:.1f}%" for v in qa_values], textposition="auto",
                )])
                fig_qa_cum.update_layout(
                    yaxis_title="百分比 (%)", yaxis_range=[0, 100],
                    height=360, margin=dict(t=40, b=30),
                )
                st.plotly_chart(fig_qa_cum, use_container_width=True, key="cum_qa_bar")
            else:
                st.info("暂无问答评测数据")

        # 轨道分布和结果状态分布
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Judge 轨道分布**")
            dist_labels = []
            dist_values = []
            if retrieval_all:
                dist_labels.append("检索评测")
                dist_values.append(len(retrieval_all))
            if strict_qa_all:
                dist_labels.append("严格问答")
                dist_values.append(len(strict_qa_all))
            if grounded_qa_all:
                dist_labels.append("合理性问答")
                dist_values.append(len(grounded_qa_all))
            if error_all:
                dist_labels.append("错误")
                dist_values.append(len(error_all))
            if dist_labels:
                fig_track_dist = go.Figure(data=[go.Pie(
                    labels=dist_labels, values=dist_values,
                    hole=0.4, textinfo="label+value+percent",
                )])
                fig_track_dist.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_track_dist, use_container_width=True, key="cum_track_dist")

        with dist_col2:
            st.markdown("**结果状态分布**")
            hit_count = sum(1 for r in retrieval_all if r.get("retrieval_top1_hit"))
            miss_count = len(retrieval_all) - hit_count
            qa_correct = sum(1 for r in strict_qa_all + grounded_qa_all if r.get("answer_correct"))
            qa_wrong = len(strict_qa_all + grounded_qa_all) - qa_correct
            status_labels = []
            status_values = []
            if hit_count:
                status_labels.append("检索 Top1 命中")
                status_values.append(hit_count)
            if miss_count:
                status_labels.append("检索 Top1 未命中")
                status_values.append(miss_count)
            if qa_correct:
                status_labels.append("QA 回答正确")
                status_values.append(qa_correct)
            if qa_wrong:
                status_labels.append("QA 回答错误")
                status_values.append(qa_wrong)
            if error_all:
                status_labels.append("评测错误")
                status_values.append(len(error_all))
            if status_labels:
                fig_status_dist = go.Figure(data=[go.Pie(
                    labels=status_labels, values=status_values,
                    hole=0.4, textinfo="label+value+percent",
                )])
                fig_status_dist.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_status_dist, use_container_width=True, key="cum_status_dist")

    # ---------- 运行记录 ----------
    st.markdown("---")
    st.markdown(f"##### 运行记录（配置: {selected_config.get('config_name', '')}）")

    if not config_runs:
        st.info("该配置方案暂无运行记录。在「批量提问」页面使用此配置开始提问后，运行记录将自动记录在此。")
    else:
        st.markdown(f"**共 {len(config_runs)} 次运行**")

        # 运行记录表格
        run_table = []
        for run in config_runs:
            # 获取真实状态
            run_status = get_run_status(
                run["run_id"],
                batch_dir=str(BATCH_DIR),
                raw_dir=str(RAW_DIR),
                processed_file=str(PROCESSED_DIR / "langfuse_samples.jsonl"),
                judged_file=str(JUDGED_FILE),
            )
            run_table.append({
                "运行 ID": run.get("run_id", ""),
                "题集名称": run_status.get("question_set_name") or run.get("question_set_name", "") or "旧版题集",
                "题集 ID": run_status.get("question_set_id") or run.get("question_set_id", "") or "—",
                "题目数": run.get("question_count", 0),
                "Batch": f"{run_status.get('batch_success', 0)}/{run_status.get('batch_total', 0)}",
                "Processed": run_status.get("processed_count", 0),
                "Judge": run_status.get("judge_count", 0),
                "创建时间": run.get("started_at", "")[:19],
            })

        st.dataframe(run_table, use_container_width=True)

        # 运行详情和状态看板
        for run in config_runs:
            run_id = run.get("run_id", "")

            # 获取真实状态
            run_status = get_run_status(
                run_id,
                batch_dir=str(BATCH_DIR),
                raw_dir=str(RAW_DIR),
                processed_file=str(PROCESSED_DIR / "langfuse_samples.jsonl"),
                judged_file=str(JUDGED_FILE),
            )

            q_set_name = run_status.get("question_set_name") or run.get("question_set_name", "") or "旧版题集"
            q_set_id = run_status.get("question_set_id") or run.get("question_set_id", "")
            batch_success = run_status.get("batch_success", 0)
            batch_total = run_status.get("batch_total", 0)
            processed_count = run_status.get("processed_count", 0)
            judge_count = run_status.get("judge_count", 0)
            question_count = run.get("question_count", 0)

            # 状态图标
            if judge_count > 0:
                status_icon = "✅"
            elif batch_success > 0:
                status_icon = "⏳"
            else:
                status_icon = "❌"

            with st.expander(f"{status_icon} {run_id} | 题集: {q_set_name}", expanded=False):
                # 基本信息
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**运行 ID**: `{run_id}`")
                    st.markdown(f"**题集名称**: {q_set_name}")
                    st.markdown(f"**题集 ID**: `{q_set_id or '未指定'}`")
                    st.markdown(f"**题目来源**: {run.get('question_set_source', '') or '未指定'}")
                with info_col2:
                    st.markdown(f"**题目数量**: {question_count}")
                    st.markdown(f"**创建时间**: {run.get('started_at', '')}")
                    st.markdown(f"**状态**: {run.get('status', '')}")
                    st.markdown(f"**配置 ID**: `{run.get('config_id', '')}`")

                # 运行状态看板
                st.markdown("---")
                st.markdown("**运行状态看板**")

                status_col1, status_col2, status_col3, status_col4 = st.columns(4)
                with status_col1:
                    st.metric("Batch", f"{batch_success}/{batch_total}")
                with status_col2:
                    st.metric("Raw", run_status.get("raw_count", 0))
                with status_col3:
                    st.metric("样本准备", processed_count)
                with status_col4:
                    st.metric("Judge", judge_count)

                # 流程完成率进度条
                _denom = max(question_count, 1)
                _batch_rate = batch_success / max(batch_total, 1) if batch_total > 0 else 0
                _proc_rate = processed_count / _denom
                _judge_rate = judge_count / _denom

                prog_col1, prog_col2, prog_col3 = st.columns(3)
                with prog_col1:
                    st.caption(f"Batch 成功率: {_batch_rate:.0%}")
                    st.progress(min(_batch_rate, 1.0))
                with prog_col2:
                    st.caption(f"样本准备率: {_proc_rate:.0%} ({processed_count}/{_denom})")
                    st.progress(min(_proc_rate, 1.0))
                with prog_col3:
                    st.caption(f"Judge 覆盖率: {_judge_rate:.0%} ({judge_count}/{_denom})")
                    st.progress(min(_judge_rate, 1.0))

                # 关联文件
                st.markdown("**关联文件**")
                batch_file = run.get("batch_results_file")
                raw_file = run.get("raw_results_file")
                file_col1, file_col2 = st.columns(2)
                with file_col1:
                    if batch_file:
                        st.markdown(f"Batch 结果: `{batch_file}`")
                    else:
                        st.caption("Batch 结果: 无")
                with file_col2:
                    if raw_file:
                        st.markdown(f"Raw 结果: `{raw_file}`")
                    else:
                        st.caption("Raw 结果: 无")

                # Judge 指标
                if judge_count > 0:
                    st.markdown("---")
                    st.markdown("**Judge 评测指标**")

                    judge_results = run_status.get("judge_results", [])
                    if judge_results:
                        # 计算指标
                        valid_results = [r for r in judge_results if "error" not in r]
                        if valid_results:
                            metrics = compute_metrics(judge_results)

                            # 按轨道分组
                            retrieval_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_RETRIEVAL]
                            strict_qa_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_STRICT_QA]
                            grounded_qa_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_GROUNDED_QA]

                            # 显示指标
                            metric_cols = st.columns(3)

                            if retrieval_results:
                                with metric_cols[0]:
                                    st.markdown("**检索评测**")
                                    n = len(retrieval_results)
                                    t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_results) / n
                                    t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_results) / n
                                    t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_results) / n
                                    st.metric("Top1 Hit", f"{t1:.0%}")
                                    st.metric("Top3 Hit", f"{t3:.0%}")
                                    st.metric("Top5 Hit", f"{t5:.0%}")
                                    st.caption(f"样本数: {n}")

                            if strict_qa_results:
                                with metric_cols[1]:
                                    st.markdown("**严格问答**")
                                    n = len(strict_qa_results)
                                    acc = sum(r.get("answer_correct", 0) for r in strict_qa_results) / n
                                    st.metric("Answer Correctness", f"{acc:.0%}")
                                    st.caption(f"样本数: {n}")

                            if grounded_qa_results:
                                with metric_cols[2]:
                                    st.markdown("**合理性问答**")
                                    n = len(grounded_qa_results)
                                    acc = sum(r.get("answer_correct", 0) for r in grounded_qa_results) / n
                                    st.metric("Answer Grounded", f"{acc:.0%}")
                                    st.caption(f"样本数: {n}")

                # ========== 评测结果可视化 ==========
                if judge_count > 0:
                    judge_results_viz = run_status.get("judge_results", [])
                    if judge_results_viz:
                        valid_viz = [r for r in judge_results_viz if "error" not in r]
                        error_viz = [r for r in judge_results_viz if "error" in r]

                        # 加载当前 run 的 processed samples 构建 sample_map
                        _run_sample_map = {}
                        _proc_path = PROCESSED_DIR / "langfuse_samples.jsonl"
                        if _proc_path.exists():
                            try:
                                with _proc_path.open("r", encoding="utf-8") as _pf:
                                    for _pline in _pf:
                                        if not _pline.strip():
                                            continue
                                        _pobj = json.loads(_pline)
                                        _p_run_id = _pobj.get("run_id", "")
                                        if not _p_run_id:
                                            _p_uid = _pobj.get("user_id", "")
                                            if _p_uid.startswith("rag_eval:"):
                                                _p_parts = _p_uid.split(":", 2)
                                                if len(_p_parts) == 3:
                                                    _p_run_id = _p_parts[1]
                                        if _p_run_id == run_id:
                                            _ptid = _pobj.get("trace_id", "")
                                            if _ptid:
                                                _run_sample_map[_ptid] = _pobj
                            except (json.JSONDecodeError, IOError):
                                pass

                        st.markdown("---")
                        st.markdown("##### 评测结果可视化")

                        # 检索评测轨道
                        retrieval_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_RETRIEVAL]
                        strict_qa_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_STRICT_QA]
                        grounded_qa_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_GROUNDED_QA]

                        # -- 检索评测图表 --
                        if retrieval_viz:
                            st.markdown("**检索评测**")
                            ret_chart_col1, ret_chart_col2 = st.columns(2)
                            with ret_chart_col1:
                                n = len(retrieval_viz)
                                _ret_m = {
                                    "top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in retrieval_viz) / n,
                                    "top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in retrieval_viz) / n,
                                    "top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in retrieval_viz) / n,
                                }
                                st.plotly_chart(build_retrieval_bar_chart(_ret_m), use_container_width=True, key=f"exp_ret_bar_{run_id}")
                            with ret_chart_col2:
                                pq_fig = build_retrieval_per_question_chart(retrieval_viz)
                                if pq_fig:
                                    st.plotly_chart(pq_fig, use_container_width=True, key=f"exp_ret_pq_{run_id}")
                                else:
                                    st.info("无有效评测数据")
                        else:
                            st.info("当前运行无检索评测轨道数据")

                        # -- QA 指标卡片 --
                        if strict_qa_viz or grounded_qa_viz:
                            qa_chart_col1, qa_chart_col2 = st.columns(2)
                            if strict_qa_viz:
                                with qa_chart_col1:
                                    n = len(strict_qa_viz)
                                    acc = sum(r.get("answer_correct", 0) for r in strict_qa_viz) / n
                                    st.plotly_chart(build_strict_qa_bar_chart({"answer_correct_rate": acc}), use_container_width=True, key=f"exp_strict_qa_{run_id}")
                                    st.caption(f"严格问答样本数: {n}")
                            if grounded_qa_viz:
                                with qa_chart_col2:
                                    n = len(grounded_qa_viz)
                                    acc = sum(r.get("answer_correct", 0) for r in grounded_qa_viz) / n
                                    st.plotly_chart(build_grounded_qa_bar_chart({"answer_correct_rate": acc}), use_container_width=True, key=f"exp_grounded_qa_{run_id}")
                                    st.caption(f"合理性问答样本数: {n}")

                        # -- 结果分布 --
                        st.markdown("**结果分布**")
                        dist_col1, dist_col2 = st.columns(2)
                        with dist_col1:
                            # 按评测轨道分布
                            track_labels = []
                            track_values = []
                            if retrieval_viz:
                                track_labels.append("检索评测")
                                track_values.append(len(retrieval_viz))
                            if strict_qa_viz:
                                track_labels.append("严格问答")
                                track_values.append(len(strict_qa_viz))
                            if grounded_qa_viz:
                                track_labels.append("合理性问答")
                                track_values.append(len(grounded_qa_viz))
                            if error_viz:
                                track_labels.append("错误")
                                track_values.append(len(error_viz))
                            if track_labels:
                                fig_dist = go.Figure(data=[go.Pie(
                                    labels=track_labels, values=track_values,
                                    hole=0.4, textinfo="label+value+percent",
                                )])
                                fig_dist.update_layout(height=300, margin=dict(t=40, b=20))
                                st.plotly_chart(fig_dist, use_container_width=True, key=f"exp_dist_{run_id}")
                        with dist_col2:
                            # 检索命中分布（仅检索轨道）
                            if retrieval_viz:
                                hit_count = sum(1 for r in retrieval_viz if r.get("retrieval_top1_hit"))
                                miss_count = len(retrieval_viz) - hit_count
                                fig_hit = go.Figure(data=[go.Pie(
                                    labels=["Top1 命中", "Top1 未命中"],
                                    values=[hit_count, miss_count],
                                    marker_colors=["#2ca02c", "#d62728"],
                                    hole=0.4, textinfo="label+value+percent",
                                )])
                                fig_hit.update_layout(height=300, margin=dict(t=40, b=20))
                                st.plotly_chart(fig_hit, use_container_width=True, key=f"exp_hit_{run_id}")
                            elif strict_qa_viz or grounded_qa_viz:
                                all_qa = strict_qa_viz + grounded_qa_viz
                                correct_count = sum(1 for r in all_qa if r.get("answer_correct"))
                                wrong_count = len(all_qa) - correct_count
                                fig_ans = go.Figure(data=[go.Pie(
                                    labels=["回答正确", "回答错误"],
                                    values=[correct_count, wrong_count],
                                    marker_colors=["#2ca02c", "#d62728"],
                                    hole=0.4, textinfo="label+value+percent",
                                )])
                                fig_ans.update_layout(height=300, margin=dict(t=40, b=20))
                                st.plotly_chart(fig_ans, use_container_width=True, key=f"exp_ans_{run_id}")

                        # -- 评测详情（本次运行） --
                        st.markdown("---")
                        st.markdown("##### 评测详情（本次运行）")
                        render_judge_results_list(
                            judge_results_viz, _run_sample_map,
                            key_prefix=f"exp_detail_{run_id}", page_size=20,
                        )

                # 配置快照 + 修正
                snapshot = run.get("config_snapshot", {})
                with st.expander("配置快照详情（可修正）", expanded=False):
                    st.json(snapshot)
                    st.markdown("---")
                    st.markdown("**修正本次运行的配置记录**")
                    st.caption("仅修正描述性字段，不影响其他运行或配置方案。用于补录旧 run 的实际参数。")
                    with st.form(f"edit_snapshot_{run_id}"):
                        ss_col1, ss_col2 = st.columns(2)
                        with ss_col1:
                            ss_kb = st.text_input("知识库版本", value=snapshot.get("knowledge_base_version", ""), key=f"ss_kb_{run_id}")
                            ss_wf = st.text_input("工作流版本", value=snapshot.get("workflow_version", ""), key=f"ss_wf_{run_id}")
                            ss_topk = st.text_input("Top K", value=str(snapshot.get("top_k", "")), key=f"ss_topk_{run_id}")
                            ss_rerank = st.text_input("Rerank 模型", value=snapshot.get("rerank_model", ""), key=f"ss_rerank_{run_id}")
                        with ss_col2:
                            ss_embed = st.text_input("Embedding 模型", value=snapshot.get("embedding_model", ""), key=f"ss_embed_{run_id}")
                            ss_mode = st.text_input("检索模式", value=snapshot.get("retrieval_mode", ""), key=f"ss_mode_{run_id}")
                            ss_chunk = st.text_input("分块策略", value=snapshot.get("chunk_strategy", ""), key=f"ss_chunk_{run_id}")
                            ss_notes = st.text_area("备注", value=snapshot.get("notes", ""), key=f"ss_notes_{run_id}", height=68)
                        ss_note = st.text_input("修正说明", value="", key=f"ss_note_{run_id}",
                                                help="如：补录实际使用的 Rerank 配置")
                        ss_submit = st.form_submit_button("保存修正", type="primary")

                    if ss_submit:
                        from experiment import update_run_snapshot
                        ss_updates = {
                            "knowledge_base_version": ss_kb,
                            "workflow_version": ss_wf,
                            "embedding_model": ss_embed,
                            "retrieval_mode": ss_mode,
                            "chunk_strategy": ss_chunk,
                            "notes": ss_notes,
                        }
                        if ss_topk.strip():
                            try:
                                ss_updates["top_k"] = int(ss_topk)
                            except ValueError:
                                ss_updates["top_k"] = ss_topk
                        if ss_rerank.strip():
                            ss_updates["rerank_model"] = ss_rerank
                        update_run_snapshot(run_id, ss_updates, edit_note=ss_note)
                        st.success(f"本次运行的配置记录已修正，不影响其他运行。")
                        st.rerun()

        # ========== 运行历史 ==========
        if len(config_runs) >= 1:
            st.markdown("---")
            with st.expander("运行历史（点击展开）", expanded=False):
                st.markdown(f"**配置 {selected_config.get('config_name', '')} 下共 {len(config_runs)} 次运行**")

                # 收集每次运行的指标（按最新运行时间倒序）
                history_rows = []
                history_metrics = []  # (run_time, t1, t3, t5, qa_acc)
                for run in config_runs:
                    rid = run.get("run_id", "")
                    rs = get_run_status(
                        rid,
                        batch_dir=str(BATCH_DIR),
                        raw_dir=str(RAW_DIR),
                        processed_file=str(PROCESSED_DIR / "langfuse_samples.jsonl"),
                        judged_file=str(JUDGED_FILE),
                    )
                    j_results = rs.get("judge_results", [])
                    valid_j = [r for r in j_results if "error" not in r]
                    retrieval_j = [r for r in valid_j if r.get("evaluation_track") == TRACK_RETRIEVAL]
                    strict_qa_j = [r for r in valid_j if r.get("evaluation_track") == TRACK_STRICT_QA]

                    t1 = t3 = t5 = qa_acc = None
                    if retrieval_j:
                        n = len(retrieval_j)
                        t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_j) / n
                        t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_j) / n
                        t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_j) / n
                    if strict_qa_j:
                        n = len(strict_qa_j)
                        qa_acc = sum(r.get("answer_correct", 0) for r in strict_qa_j) / n

                    run_time = run.get("started_at", "")[:19]
                    history_rows.append({
                        "运行 ID": rid,
                        "运行时间": run_time,
                        "题集": rs.get("question_set_name") or run.get("question_set_name", "") or "旧版",
                        "题数": run.get("question_count", 0),
                        "Judge 数": rs.get("judge_count", 0),
                        "Top1": f"{t1:.0%}" if t1 is not None else "N/A",
                        "Top3": f"{t3:.0%}" if t3 is not None else "N/A",
                        "Top5": f"{t5:.0%}" if t5 is not None else "N/A",
                        "QA 正确率": f"{qa_acc:.0%}" if qa_acc is not None else "N/A",
                    })
                    if t1 is not None:
                        history_metrics.append((run_time, t1, t3, t5))

                # 按运行时间倒序
                history_rows.sort(key=lambda x: x["运行时间"], reverse=True)
                st.dataframe(history_rows, use_container_width=True)

                # 轻量时间趋势图：横轴运行时间，纵轴 Top1/Top3/Top5
                if len(history_metrics) >= 2:
                    history_metrics.sort(key=lambda x: x[0])  # 按时间正序
                    trend_times = [m[0] for m in history_metrics]
                    trend_t1 = [m[1] * 100 for m in history_metrics]
                    trend_t3 = [m[2] * 100 for m in history_metrics]
                    trend_t5 = [m[3] * 100 for m in history_metrics]

                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=trend_times, y=trend_t1, mode="lines+markers",
                        name="Top1 Hit", line=dict(color="#1f77b4"),
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=trend_times, y=trend_t3, mode="lines+markers",
                        name="Top3 Hit", line=dict(color="#2ca02c"),
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=trend_times, y=trend_t5, mode="lines+markers",
                        name="Top5 Hit", line=dict(color="#9467bd"),
                    ))
                    fig_trend.update_layout(
                        yaxis_title="百分比 (%)", yaxis_range=[0, 100],
                        height=350, margin=dict(t=40, b=30),
                    )
                    st.caption("检索指标变化趋势")
                    st.plotly_chart(fig_trend, use_container_width=True, key="history_trend")
