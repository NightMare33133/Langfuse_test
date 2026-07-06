import streamlit as st
from pathlib import Path
import json
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from parser import parse_langfuse_jsonl, save_results
from judge import judge_all, compute_metrics, call_llm

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
JUDGED_DIR = Path(__file__).parent / "data" / "judged"
JUDGED_FILE = JUDGED_DIR / "eval_results.jsonl"


# ---------- 评测结果可视化 / 导出辅助函数 ----------

def build_eval_bar_chart(metrics: dict):
    labels = ["Top1 Hit", "Top3 Hit", "Top5 Hit", "Answer Correct"]
    values = [
        (metrics.get("top1_hit_rate") or 0) * 100,
        (metrics.get("top3_hit_rate") or 0) * 100,
        (metrics.get("top5_hit_rate") or 0) * 100,
        (metrics.get("answer_correct_rate") or 0) * 100,
    ]
    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        marker_color=["#1f77b4", "#2ca02c", "#9467bd", "#17becf"],
    )])
    fig.update_layout(
        yaxis_title="百分比 (%)", yaxis_range=[0, 100],
        height=360, margin=dict(t=20, b=30),
    )
    return fig


def build_answer_pye(valid_results: list):
    correct = sum(1 for r in valid_results if r.get("answer_correct"))
    incorrect = len(valid_results) - correct
    fig = go.Figure(data=[go.Pie(
        labels=["正确", "错误"],
        values=[correct, incorrect],
        marker_colors=["#2ca02c", "#d62728"],
        hole=0.4,
        textinfo="label+value+percent",
    )])
    fig.update_layout(height=340, margin=dict(t=20, b=20))
    return fig


def build_per_question_chart(valid_results: list):
    if not valid_results:
        return None
    rows = []
    for r in valid_results:
        q = r.get("question", "")
        rows.append({
            "question": q[:30] + ("…" if len(q) > 30 else ""),
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
        margin=dict(t=20, b=30, l=10),
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


st.set_page_config(page_title="Langfuse RAG 评测工具", layout="wide")
st.title("Langfuse RAG 评测工具")

# --- Sidebar: file selection and parsing ---
st.sidebar.header("数据导入")

# Upload section
uploaded = st.sidebar.file_uploader("上传 Langfuse 导出文件", type=["jsonl"])
if uploaded is not None:
    save_path = RAW_DIR / uploaded.name
    save_path.write_bytes(uploaded.getvalue())
    st.sidebar.success(f"已保存: {uploaded.name}")
    st.rerun()

raw_files = sorted(RAW_DIR.glob("*.jsonl"))
if not raw_files:
    st.sidebar.warning("data/raw 目录下没有找到 .jsonl 文件，请上传")
    st.stop()

file_names = [f.name for f in raw_files]
selected_name = st.sidebar.selectbox("选择 Langfuse 导出文件", file_names)
selected_path = RAW_DIR / selected_name

# Show file info
file_size_kb = selected_path.stat().st_size / 1024
with open(selected_path, "r", encoding="utf-8") as f:
    line_count = sum(1 for _ in f)
st.sidebar.caption(f"文件大小: {file_size_kb:.1f} KB | 总行数: {line_count}")

# Parse button
if st.sidebar.button("开始解析", type="primary"):
    with st.spinner("正在解析..."):
        samples, summary = parse_langfuse_jsonl(selected_path)
        output_path = PROCESSED_DIR / "langfuse_samples.jsonl"
        summary_path = PROCESSED_DIR / "langfuse_summary.json"
        full_summary = save_results(samples, summary, output_path, summary_path)
        st.session_state["samples"] = samples
        st.session_state["summary"] = full_summary
    st.sidebar.success(f"解析完成，共 {len(samples)} 条 trace")

# Load existing results if available
if "samples" not in st.session_state:
    samples_file = PROCESSED_DIR / "langfuse_samples.jsonl"
    summary_file = PROCESSED_DIR / "langfuse_summary.json"
    if samples_file.exists():
        with open(samples_file, "r", encoding="utf-8") as f:
            st.session_state["samples"] = [json.loads(line) for line in f if line.strip()]
    if summary_file.exists():
        st.session_state["summary"] = json.loads(summary_file.read_text(encoding="utf-8"))

# --- Sidebar: Judge config ---
st.sidebar.divider()
st.sidebar.header("Judge 评测")

judge_api_key = st.sidebar.text_input("API Key", type="password")
judge_base_url = st.sidebar.text_input("Base URL", value="https://token-plan-cn.xiaomimimo.com/v1")
judge_model = st.sidebar.text_input("Model", value="mimo-v2.5-pro")

# Test connection
if st.sidebar.button("测试 Judge 连接"):
    if not judge_api_key:
        st.sidebar.error("请先输入 API Key")
    else:
        with st.sidebar.status("正在测试连接...", expanded=True) as status:
            try:
                resp = call_llm('请只输出 JSON：{"ok": true}', judge_api_key, judge_base_url, judge_model, timeout=15)
                status.update(label="连接成功 ✓", state="complete")
                st.sidebar.code(resp[:200])
            except Exception as e:
                status.update(label="连接失败", state="error")
                st.sidebar.error(str(e))

run_judge = st.sidebar.button("运行 Judge 评测")

# Debug options
st.sidebar.markdown("**调试选项**")
debug_limit = st.sidebar.checkbox("只评测前 1 条样本", value=True)
if debug_limit:
    max_samples = 1
else:
    total_available = len(st.session_state.get("samples") or [])
    max_samples = st.sidebar.number_input("最多评测样本数", min_value=1, max_value=max(total_available, 1), value=min(10, total_available))
show_debug = st.sidebar.checkbox("显示 Judge Prompt 和原始响应")

# --- Main content ---
samples = st.session_state.get("samples")
summary = st.session_state.get("summary") or {}

if not samples:
    st.info("请在左侧选择文件并点击「开始解析」")
    st.stop()

# --- Tabs ---
tab_samples, tab_judge = st.tabs(["样本列表", "评测结果"])

# ========== Tab: 样本列表 ==========
with tab_samples:
    # Stats
    trace_count = summary.get("trace_count") or len(samples)
    bad_line_count = summary.get("bad_line_count") or 0
    retrieval_total = summary.get("total_retrieval_results")

    st.subheader("统计信息")
    col1, col2, col3 = st.columns(3)
    col1.metric("总 Trace 数", trace_count)
    col2.metric("成功解析", trace_count - bad_line_count)
    col3.metric("Retrieval 结果总数", retrieval_total if retrieval_total is not None else "N/A")

    if bad_line_count > 0:
        st.warning(f"有 {bad_line_count} 行解析失败")

    # Search filter
    search = st.text_input("搜索问题内容", "")
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

            st.markdown("**元数据**")
            st.json({
                "trace_id": sample.get("trace_id"),
                "trace_name": sample.get("trace_name"),
                "session_id": sample.get("session_id"),
                "user_id": sample.get("user_id"),
                "workflow_run_id": sample.get("workflow_run_id"),
                "observation_count": len(sample.get("observations", [])),
            })

# ========== Tab: 评测结果 ==========
with tab_judge:
    # Run judge if button pressed
    if run_judge:
        if not judge_api_key:
            st.error("请在左侧输入 API Key")
        elif not judge_model:
            st.error("请在左侧输入 Model 名称")
        else:
            judge_samples = samples[:max_samples]
            st.session_state["judge_results"] = []

            with st.status("正在运行 Judge 评测...", expanded=True) as eval_status:
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

                results = []
                for result in judge_all(
                    judge_samples, judge_api_key, judge_base_url,
                    judge_model, on_progress,
                ):
                    results.append(result)
                    # 实时刷新当前结果预览
                    with live_result_area:
                        _r = result
                        if "error" in _r:
                            st.warning(
                                f"❌ [{len(results)}] {(_r.get('question') or '')[:40]} — "
                                f"{_r['error'][:100]}"
                            )
                        else:
                            t1 = "✓" if _r.get("retrieval_top1_hit") else "✗"
                            t3 = "✓" if _r.get("retrieval_top3_hit") else "✗"
                            ans = "✓" if _r.get("answer_correct") else "✗"
                            st.write(
                                f"✅ [{len(results)}] {(_r.get('question') or '')[:40]} — "
                                f"Top1:{t1} Top3:{t3} Answer:{ans}"
                            )
                    if show_debug:
                        with st.expander(
                            f"调试 - 第 {len(results)} 条: "
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

            st.session_state["judge_results"] = results

            # Save to disk (strip debug fields before saving)
            JUDGED_DIR.mkdir(parents=True, exist_ok=True)
            with JUDGED_FILE.open("w", encoding="utf-8") as f:
                for r in results:
                    save_r = {
                        k: v for k, v in r.items()
                        if k not in ("_prompt", "_raw_response")
                    }
                    f.write(json.dumps(save_r, ensure_ascii=False) + "\n")

            st.success(f"评测完成，结果已保存到 {JUDGED_FILE.name}")

    # Load existing judge results if not in session
    if "judge_results" not in st.session_state and JUDGED_FILE.exists():
        with JUDGED_FILE.open("r", encoding="utf-8") as f:
            st.session_state["judge_results"] = [
                json.loads(line) for line in f if line.strip()
            ]

    judge_results = st.session_state.get("judge_results") or []

    if not judge_results:
        st.info("请在左侧配置 API 后点击「运行 Judge 评测」")
    else:
        metrics = compute_metrics(judge_results)
        valid_results = [r for r in judge_results if "error" not in r]

        # ---------- Top5 提示 ----------
        st.caption(
            "💡 如果每题实际只召回 3 条检索结果，则 Top5 指标仅供参考；"
            "严格来说需要把 Dify 检索 topK 调到 5 后重新测试。"
        )

        # ---------- 指标卡片 ----------
        st.subheader("评测指标")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("总样本数", metrics["total"])
        c2.metric("有效评测数", metrics["evaluated"])
        c3.metric("错误数", metrics["errors"])
        c4.metric(
            "Top1 Hit",
            f"{metrics['top1_hit_rate']:.0%}" if metrics["top1_hit_rate"] is not None else "N/A",
        )
        c5.metric(
            "Top3 Hit",
            f"{metrics['top3_hit_rate']:.0%}" if metrics["top3_hit_rate"] is not None else "N/A",
        )
        c6.metric(
            "Top5 Hit",
            f"{metrics['top5_hit_rate']:.0%}" if metrics["top5_hit_rate"] is not None else "N/A",
        )
        c7.metric(
            "Answer OK",
            f"{metrics['answer_correct_rate']:.0%}" if metrics["answer_correct_rate"] is not None else "N/A",
        )

        if metrics["errors"] > 0:
            st.warning(f"有 {metrics['errors']} 条评测出错")

        # ---------- 可视化图表 ----------
        st.subheader("可视化")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**命中率 / 正确率概览**")
            st.plotly_chart(build_eval_bar_chart(metrics), use_container_width=True)

        with chart_col2:
            st.markdown("**Answer 正确 vs 错误**")
            if valid_results:
                st.plotly_chart(build_answer_pye(valid_results), use_container_width=True)
            else:
                st.info("无有效评测数据")

        st.markdown("**每题命中情况**")
        pq_fig = build_per_question_chart(valid_results) if valid_results else None
        if pq_fig:
            st.plotly_chart(pq_fig, use_container_width=True)
        else:
            st.info("无有效评测数据")

        # ---------- Top1 未命中案例 ----------
        top1_miss = [r for r in valid_results if not r.get("retrieval_top1_hit")]
        if top1_miss:
            st.subheader(f"Top1 未命中案例 ({len(top1_miss)} 条)")
            st.caption("以下问题 Top1 未命中但可能 Top3 命中，可用于分析检索质量问题")
            for r in top1_miss:
                with st.expander(f"**{r.get('question', '(无问题)')[:60]}**"):
                    st.markdown(f"**问题**: {r.get('question', '')}")
                    st.markdown(f"**原因**: {r.get('reason', '(无)')}")
                    t3 = "✓" if r.get("retrieval_top3_hit") else "✗"
                    t5 = "✓" if r.get("retrieval_top5_hit") else "✗"
                    st.markdown(f"**Top3**: {t3} | **Top5**: {t5}")
                    st.markdown(f"**trace_id**: `{r.get('trace_id', '')}`")

        # ---------- 评测详情表格 ----------
        st.subheader("评测详情表格")
        table_rows = []
        for r in judge_results:
            table_rows.append({
                "question": r.get("question", ""),
                "retrieval_top1_hit": r.get("retrieval_top1_hit"),
                "retrieval_top3_hit": r.get("retrieval_top3_hit"),
                "retrieval_top5_hit": r.get("retrieval_top5_hit"),
                "answer_correct": r.get("answer_correct"),
                "reason": r.get("reason", ""),
                "trace_id": r.get("trace_id", ""),
                "error": r.get("error", ""),
            })
        df_results = pd.DataFrame(table_rows)
        st.dataframe(
            df_results,
            use_container_width=True,
            height=min(400, len(df_results) * 40 + 60),
        )

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
