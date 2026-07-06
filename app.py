import streamlit as st
from pathlib import Path
import json

from parser import parse_langfuse_jsonl, save_results
from judge import judge_all, compute_metrics

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
JUDGED_DIR = Path(__file__).parent / "data" / "judged"
JUDGED_FILE = JUDGED_DIR / "eval_results.jsonl"


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
judge_base_url = st.sidebar.text_input("Base URL", value="https://api.openai.com/v1")
judge_model = st.sidebar.text_input("Model", value="gpt-4o-mini")

run_judge = st.sidebar.button("运行 Judge 评测")

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
            st.session_state["judge_results"] = []
            progress_bar = st.progress(0, text="正在评测...")
            status_area = st.empty()

            def on_progress(done, total, result):
                progress_bar.progress(done / total, text=f"评测进度: {done}/{total}")
                if "error" in result:
                    status_area.warning(f"第 {done} 条出错: {result['error'][:80]}")
                else:
                    status_area.caption(f"第 {done} 条完成 ✓")

            results = list(judge_all(samples, judge_api_key, judge_base_url, judge_model, on_progress))
            st.session_state["judge_results"] = results

            # Save to disk
            JUDGED_DIR.mkdir(parents=True, exist_ok=True)
            with JUDGED_FILE.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            progress_bar.progress(1.0, text="评测完成")
            st.success(f"评测完成，结果已保存到 {JUDGED_FILE.name}")

    # Load existing judge results if not in session
    if "judge_results" not in st.session_state and JUDGED_FILE.exists():
        with JUDGED_FILE.open("r", encoding="utf-8") as f:
            st.session_state["judge_results"] = [json.loads(line) for line in f if line.strip()]

    judge_results = st.session_state.get("judge_results") or []

    if not judge_results:
        st.info("请在左侧配置 API 后点击「运行 Judge 评测」")
    else:
        # Metrics
        metrics = compute_metrics(judge_results)
        st.subheader("评测指标")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("总样本", metrics["total"])
        m2.metric("已评测", metrics["evaluated"])
        m3.metric("Top1 Hit", f"{metrics['top1_hit_rate']:.0%}" if metrics["top1_hit_rate"] is not None else "N/A")
        m4.metric("Top3 Hit", f"{metrics['top3_hit_rate']:.0%}" if metrics["top3_hit_rate"] is not None else "N/A")
        m5.metric("Answer OK", f"{metrics['answer_correct_rate']:.0%}" if metrics["answer_correct_rate"] is not None else "N/A")

        if metrics["errors"] > 0:
            st.warning(f"有 {metrics['errors']} 条评测出错")

        # Results list
        st.subheader("评测详情")
        for r in judge_results:
            has_error = "error" in r
            label = f"**{r.get('question', '(无问题)')[:50]}**"
            if has_error:
                label += f" ⚠️ {r['error'][:40]}..."

            with st.expander(label):
                if has_error:
                    st.error(f"错误: {r['error']}")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Top1", "✓" if r.get("retrieval_top1_hit") else "✗")
                    c2.metric("Top3", "✓" if r.get("retrieval_top3_hit") else "✗")
                    c3.metric("Top5", "✓" if r.get("retrieval_top5_hit") else "✗")
                    c4.metric("Answer", "✓" if r.get("answer_correct") else "✗")
                    st.markdown(f"**理由**: {r.get('reason', '(无)')}")

                st.markdown(f"**trace_id**: `{r.get('trace_id', '')}`")
