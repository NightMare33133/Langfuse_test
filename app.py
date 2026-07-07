import streamlit as st
from pathlib import Path
import json
import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from parser import parse_langfuse_jsonl, save_results
from judge import judge_all, compute_metrics, call_llm, pre_screen, compute_content_hash, build_judge_prompt, load_prompt_template

load_dotenv(Path(__file__).parent / ".env")

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

# API fetch section
st.sidebar.divider()
st.sidebar.subheader("从 Langfuse API 拉取")
langfuse_host = st.sidebar.text_input("Langfuse 地址", value=os.getenv("LANGFUSE_HOST", "http://localhost:3000"))
langfuse_pk = st.sidebar.text_input("Public Key", value=os.getenv("LANGFUSE_PUBLIC_KEY", ""))
langfuse_sk = st.sidebar.text_input("Secret Key", value=os.getenv("LANGFUSE_SECRET_KEY", ""), type="password")
fetch_limit = st.sidebar.number_input("每页 trace 数", min_value=1, max_value=500, value=50)

if st.sidebar.button("拉取 Traces"):
    if not langfuse_pk or not langfuse_sk:
        st.sidebar.error("请填写 Langfuse Public Key 和 Secret Key")
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
                st.sidebar.success(f"拉取完成！共 {count} 行，已保存为 {filename}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"拉取失败: {e}")

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

# 1. API 配置
judge_api_key = st.sidebar.text_input("API Key", type="password", value=os.getenv("JUDGE_API_KEY", ""))
judge_base_url = st.sidebar.text_input("Base URL", value=os.getenv("JUDGE_API_BASE", "https://token-plan-cn.xiaomimimo.com/v1"))
judge_model = st.sidebar.text_input("Model", value=os.getenv("JUDGE_MODEL", "mimo-v2.5-pro"))
judge_timeout = st.sidebar.number_input(
    "请求超时时间（秒）", min_value=10, max_value=180, value=60, step=10,
    help="单次 LLM 请求的最大等待时间，建议 60 秒以上"
)

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

# 2. 选择评测范围
st.sidebar.divider()
st.sidebar.subheader("评测范围")

total_available = len(st.session_state.get("samples") or [])
debug_limit = st.sidebar.checkbox("只评前 1 条（快速测试）", value=True)
if debug_limit:
    max_samples = 1
else:
    max_samples = st.sidebar.number_input(
        "评测样本数", min_value=1, max_value=max(total_available, 1),
        value=min(10, total_available),
        help="从样本列表中取前 N 条进行评测"
    )

skip_existing = st.sidebar.checkbox(
    "跳过已有成功结果",
    value=True,
    help="已有评测结果的样本不会重复调用 LLM"
)

# 3. 预览 & 运行
st.sidebar.divider()
st.sidebar.subheader("执行评测")

preview_optimization = st.sidebar.button("预览优化策略", use_container_width=True, help="先看看有多少样本需要调 LLM，不会消耗 token")
run_judge = st.sidebar.button("运行 Judge 评测", type="primary", use_container_width=True)

# 4. 高级选项（折叠）
with st.sidebar.expander("高级选项"):
    show_debug = st.sidebar.checkbox("显示 Judge Prompt 和原始响应")
    force_rerun = st.sidebar.checkbox("强制重新评测（忽略所有缓存）", value=False)
    retry_failed = st.sidebar.button("只重试失败样本")

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
    input_file = summary.get("input_file") or selected_name
    output_file = summary.get("output_file") or ""
    st.caption(f"数据来源: `{Path(input_file).name}`" + (f" → 解析结果: `{Path(output_file).name}`" if output_file else ""))

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
    st.caption(f"评测结果文件: `{JUDGED_FILE.name}`")

    # ---------- 已有结果加载 & 索引 ----------
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

    def _load_existing_for_session():
        if "judge_results" not in st.session_state and existing_results_map:
            st.session_state["judge_results"] = list(existing_results_map.values())

    def _compute_samples_to_judge(all_samples, limit, skip, force):
        """根据策略筛选需要评测的样本列表。返回 (samples_to_judge, skipped_count)"""
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
        candidates = samples[:max_samples]

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
            st.markdown(f"#### 以下 {len(need_llm)} 条样本需要调用 LLM：")
            for s in need_llm[:5]:
                q = (s.get("question") or "(无问题)")[:60]
                retrieval_count = len(s.get("retrieval_results", []))
                answer_preview = (s.get("final_answer") or "(无)")[:40]
                st.caption(f"  - `{q}` | 检索 {retrieval_count} 条 | 回答: {answer_preview}")
            if len(need_llm) > 5:
                st.caption(f"  - ...还有 {len(need_llm) - 5} 条")

            # prompt 裁剪预览
            st.markdown("#### Prompt 裁剪效果")
            sample_preview = need_llm[0]
            template = load_prompt_template()
            full_prompt = build_judge_prompt(sample_preview, template, max_content_chars=999999)
            trimmed_prompt = build_judge_prompt(sample_preview, template, max_content_chars=300)
            save_chars = len(full_prompt) - len(trimmed_prompt)
            pct = save_chars / len(full_prompt) * 100 if len(full_prompt) > 0 else 0
            st.caption(
                f"每条 prompt 从 {len(full_prompt)} 字符裁剪到 {len(trimmed_prompt)} 字符"
                f"（省 {pct:.0f}%），主要裁剪检索结果正文"
            )
            with st.expander("查看裁剪后的 prompt 示例"):
                st.code(trimmed_prompt, language=None)
        else:
            st.success("所有样本都已被跳过或规则判定，不需要调用 LLM！")

        st.divider()

    # ---------- 按钮 1: 运行 Judge 评测 ----------
    if run_judge:
        if not judge_api_key:
            st.error("请在左侧输入 API Key")
        elif not judge_model:
            st.error("请在左侧输入 Model 名称")
        else:
            if force_rerun:
                _load_existing_for_session()
                samples_to_judge = samples[:max_samples]
                skipped = 0
            else:
                samples_to_judge, skipped = _compute_samples_to_judge(
                    samples, max_samples, skip_existing, force_rerun
                )
            if skipped > 0:
                st.info(f"⏭️ 跳过 {skipped} 条已有成功结果的样本（可在侧边栏取消「跳过已有成功结果」或勾选「强制重新评测」）")

            new_results = _run_judge_ui(samples_to_judge)
            _, history_name = _merge_and_save(new_results)
            st.success(f"评测完成！结果已保存到 {JUDGED_FILE.name}，历史快照: {history_name}")

    # ---------- 按钮 2: 只重试失败样本 ----------
    if retry_failed:
        if not judge_api_key:
            st.error("请在左侧输入 API Key")
        elif not judge_model:
            st.error("请在左侧输入 Model 名称")
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
