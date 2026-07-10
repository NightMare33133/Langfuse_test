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
from judge import judge_all, compute_metrics, call_llm, pre_screen, compute_content_hash, build_judge_prompt, load_prompt_template, load_prompt_template_with_ref
from question_generator import generate_questions, save_questions, export_csv_bytes, choose_strategy, STRATEGY_LABELS
from batch_query import run_batch_query, save_batch_results, push_to_raw_dir, export_csv_bytes as batch_export_csv

load_dotenv(Path(__file__).parent / ".env")

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
JUDGED_DIR = Path(__file__).parent / "data" / "judged"
JUDGED_FILE = JUDGED_DIR / "eval_results.jsonl"
BATCH_DIR = Path(__file__).parent / "data" / "batch"
QUESTIONS_DIR = Path(__file__).parent / "data" / "questions"


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


st.set_page_config(page_title="Langfuse RAG 评测工具", layout="wide")
st.title("Langfuse RAG 评测工具")

# --- Sidebar ---
st.sidebar.markdown(
    "基于 Langfuse 数据的 RAG 检索 + 回答质量自动评测工具。"
    "支持从知识库文件自动生成测评题目、批量提问收集检索与回答、"
    "解析结构化样本并回填参考答案、通过 LLM Judge 进行严格评测或合理性评测。"
)
st.sidebar.divider()
st.sidebar.markdown("**四步工作流**")
st.sidebar.markdown(
    "1. **题目生成** — 上传知识库文件（.txt / .md），"
    "自动按章节切分后调用 LLM 出题，生成带参考答案的测评题目\n"
    "2. **批量提问** — 将题目批量发送到 Dify Q&A 接口，"
    "自动收集回答和检索结果，参考答案随题目透传\n"
    "3. **样本准备** — 导入 Langfuse / Dify 记录，解析为结构化样本，"
    "并从题目库回填 reference_answer，为评测做准备\n"
    "4. **Judge 评测** — 配置评测参数，对样本进行自动评分。"
    "有参考答案走严格评测，无参考答案走合理性评测。"
    "结果可视化并支持 CSV / Markdown 报告导出"
)
st.sidebar.divider()
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
tab_qgen, tab_batch, tab_samples, tab_judge = st.tabs(["题目生成", "批量提问", "样本准备", "Judge 评测"])

# ========== Tab: 题目生成 ==========
with tab_qgen:
    st.subheader("题目生成")
    st.caption("上传知识库文件，调用 LLM 自动生成测评题目")

    # ---------- 模块说明 ----------
    with st.expander("题目生成模块说明（点击展开）", expanded=False):
        st.markdown("""
**一句话总览：** 上传知识库文件，自动按章节切分后调用 LLM 生成带参考答案的测评题目，为后续严格评测提供标准答案。

---

**输入是什么？**

| 输入 | 说明 |
|------|------|
| 知识库文件 | .txt 或 .md 格式的知识库文档 |
| 生成数量 | 期望生成的题目数量 |
| 难度偏好 | 基础概念题 / 理解题 / 综合题 / 混合 |
| 生成策略 | 自动 / 极速 / 标准 / 深度（区别在于文档切分粒度和 LLM 调用次数） |

---

**实际做什么？**

1. **文档切分** — 将知识库文件按章节/段落切分为多个 chunk
2. **逐 chunk 出题** — 对每个 chunk 调用 LLM 生成问题
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

        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        with cfg_col1:
            qgen_num = st.select_slider("生成题目数量", options=[5, 10, 15, 20], value=10, key="qgen_num")
        with cfg_col2:
            qgen_difficulty = st.selectbox(
                "难度偏好", ["混合", "基础概念题", "理解题", "综合题"], index=0, key="qgen_diff"
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

                with st.status(f"正在生成题目（{strategy_label}模式）...", expanded=True) as gen_status:
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
                        )
                        st.session_state["generated_questions"] = questions
                        output_path, fname = save_questions(questions)
                        st.session_state["qgen_saved_file"] = fname
                        gen_status.update(
                            label=f"生成完成！共 {len(questions)} 道题目",
                            state="complete",
                            expanded=False,
                        )
                    except Exception as e:
                        gen_status.update(label="生成失败", state="error")
                        st.error(f"生成失败: {e}")
    else:
        st.info("请在上方「配置」区域上传知识库文件（.txt 或 .md）")

    # --- Results display ---
    questions = st.session_state.get("generated_questions")
    if questions:
        st.divider()
        st.subheader(f"生成结果（{len(questions)} 道题目）")

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
**一句话总览：** 将题目逐条发送到 Dify Q&A 接口，收集每次问答的最终回答和检索结果，生成可直接用于评测的结构化样本。

---

**输入是什么？**

| 来源 | 说明 |
|------|------|
| 已生成的题目 | 来自「题目生成」模块，自带 reference_answer 等元数据 |
| 手动输入问题 | 直接输入问题文本，无参考答案 |
| 从文件加载 | 上传 JSONL / CSV / TXT 文件，按格式解析问题 |
| 从历史记录加载 | 复用之前的批量提问记录 |

如果输入来自「题目生成」，reference_answer 会自动透传到输出样本中。

---

**实际做什么？**

1. **标准化输入** — 将各种格式的问题统一为 list[dict]，保留 reference_answer 等元数据
2. **逐条调用 Dify** — 对每个问题调用 Dify chat-messages API（blocking 模式）
3. **收集结果** — 从 Dify 响应中提取最终回答和检索结果
4. **组装样本** — 将提问结果转换为与 Langfuse 格式兼容的结构化样本

---

**收集哪些结果？**

| 字段 | 来源 | 说明 |
|------|------|------|
| `final_answer` | Dify response.answer | LLM 最终回答 |
| `retrieval_results` | Dify response.metadata.retriever_resources | 检索结果列表（含 position、score、content 等） |
| `retrieval_query` | 原始问题 | Dify 不单独返回 retrieval_query，用原始问题代替 |
| `trace_id` | 自动生成 | `batch_qa_{序号}_{时间戳}`，用于后续去重和结果关联 |
| `reference_answer` | 透传自输入 | 如果输入有参考答案，会保留到输出样本 |

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
- 如果输入题目有 reference_answer，推送的样本会保留它，解析后可直接进入严格评测
- 如果输入是手动输入或无参考答案的文件，解析后走无参考答案评测
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
                file_labels = [f"{f.parent.name}/{f.name}" for f in history_files]
                selected_idx = st.selectbox(
                    "选择历史文件",
                    range(len(file_labels)),
                    format_func=lambda i: file_labels[i],
                    key="batch_history_file",
                )
                selected_file = history_files[selected_idx]
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
                                questions_list.append(item)
                        except json.JSONDecodeError:
                            continue
                    st.success(f"从 {selected_file.name} 加载了 {len(questions_list)} 个问题")
                    has_ref = sum(1 for q in questions_list if q.get("reference_answer"))
                    if has_ref:
                        st.caption(f"其中 {has_ref} 道带有参考答案，评测时将用于严格评判")
                    with st.expander("预览题目", expanded=False):
                        for i, q in enumerate(questions_list, 1):
                            qtext = q.get("question", "")
                            ref = q.get("reference_answer", "")
                            if ref:
                                st.write(f"{i}. {qtext}")
                                st.caption(f"   参考答案: {ref[:80]}{'...' if len(ref) > 80 else ''}")
                            else:
                                st.write(f"{i}. {qtext}")
                except Exception as e:
                    st.error(f"读取文件失败: {e}")

    # --- Dify API config ---
    with st.expander("Dify API 配置", expanded=False):
        dify_col1, dify_col2 = st.columns(2)
        with dify_col1:
            dify_api_key = st.text_input(
                "Dify API Key", type="password",
                value=os.getenv("DIFY_API_KEY", ""),
                key="batch_dify_key",
            )
        with dify_col2:
            dify_base_url = st.text_input(
                "Dify Base URL",
                value=os.getenv("DIFY_API_BASE", "http://localhost/v1"),
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

    # --- Run batch query ---
    st.divider()

    if st.button("开始提问", type="primary", disabled=len(questions_list) == 0, key="batch_run"):
        if not dify_api_key:
            st.error("请填写 Dify API Key")
        elif not questions_list:
            st.error("没有可提问的问题")
        else:
            batch_results = []
            progress_bar = st.progress(0, text="准备开始...")
            status_container = st.container()

            for idx, total, result in run_batch_query(
                questions_list, dify_api_key, dify_base_url,
                timeout=dify_timeout, delay=dify_delay,
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

            # 自动保存结果
            save_batch_results(batch_results)
            st.success(f"批量提问完成！成功 {sum(1 for r in batch_results if r['success'])} / {total} 条")

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
**一句话总览：** 将 Langfuse / Dify 的运行记录解析为结构化样本，并尝试回填参考答案，为 Judge 评测提供输入。

---

**这个模块做什么？**

Judge 不是直接读取原始 trace 文件，而是读取这里准备好的结构化样本。这个模块负责：

1. **导入原始记录** — 从 Langfuse 导出的 JSONL 文件或 Langfuse API 获取 trace 数据
2. **解析为结构化样本** — 按 traceId 聚合 observations，提取关键字段：
   - 用户问题（question）
   - 检索查询（retrieval_query）
   - 检索结果列表（retrieval_results）
   - LLM 最终回答（final_answer）
   - trace_id、session_id 等标识信息
3. **参考答案回填** — 如果题目是从「题目生成」模块产出的，会自动从题目库中匹配并回填：
   - reference_answer（参考答案）
   - source_excerpt（来源摘录）
   - difficulty（难度）、topic（主题）

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

**和 Judge 的关系**

```
原始 trace 文件 → 【本模块】→ 结构化样本 → 【Judge 评测】→ 评分结果
```

- Judge 只消费本模块产出的结构化样本，不会回溯到原始 trace
- 如果本模块没有解析过数据，Judge 会提示「请先切换到样本准备 tab」
- reference_answer 回填也在这里完成，Judge 只看样本是否已带参考答案

---

**参考答案回填规则**

解析时会自动从题目库（`data/questions/`）中匹配：

1. 如果样本本身已有 reference_answer → 跳过
2. 如果样本有 question_id → 按 ID 精确匹配
3. 否则 → 按 question 文本精确匹配
4. 匹配成功 → 回填 reference_answer + source_excerpt + difficulty + topic
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
        has_ref_count = sum(1 for s in samples if (s.get("reference_answer") or "").strip())
        no_ref_count = trace_count - has_ref_count

        info_parts = [
            f"**来源文件**: `{src_name}`",
            f"**样本数**: {trace_count}",
            f"**检索结果总数**: {retrieval_total}" if retrieval_total else None,
        ]
        st.info(" | ".join(p for p in info_parts if p))

        # 评测模式构成
        if has_ref_count > 0 and no_ref_count > 0:
            st.warning(
                f"**混合评测模式**：{has_ref_count} 条严格评测（有参考答案）"
                f" + {no_ref_count} 条合理性评测（无参考答案）。"
                f"两者的 Answer OK 口径不同，指标需谨慎解读。"
            )
        elif has_ref_count > 0:
            st.success(f"**纯严格评测**：全部 {has_ref_count} 条样本均带参考答案，Answer OK = 与参考答案对比的正确性")
        elif no_ref_count > 0:
            st.info(f"**纯合理性评测**：全部 {no_ref_count} 条样本无参考答案，Answer OK = 基于检索内容判断的合理性")
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

1. **题目生成**：从知识库文件生成测试问题
2. **批量提问**：将问题发送给 Dify，获取检索结果和回答
3. **样本准备**：将 Dify 返回结果解析为结构化样本，为评测做准备
4. **Judge 评测**：从样本准备中取出样本，配置参数并调用 LLM 评分

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

            scope_col1, scope_col2 = st.columns(2)
            with scope_col1:
                debug_limit = st.checkbox(
                    "只评前 1 条（快速测试）", value=False, key="debug_limit",
                    help="勾选后「评测样本数」不生效，仅评测第 1 条样本"
                )
                if debug_limit:
                    max_samples = 1
                else:
                    max_samples = st.number_input(
                        "评测样本数", min_value=1, max_value=max(total_available, 1),
                        value=total_available, key="max_samples",
                    )
                    st.caption(f"从样本准备前 {total_available} 条中按顺序取前 {max_samples} 条作为候选")
            with scope_col2:
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

            # === 第二层：高级选项 ===
            with st.expander("高级选项", expanded=False):
                show_debug = st.checkbox("显示 Judge Prompt 和原始响应", key="show_debug")

            # === 本次评测执行计划 ===
            effective_count = max_samples if not debug_limit else 1
            st.markdown("---")
            st.markdown("##### 本次评测执行计划")

            # 模拟真实筛选逻辑
            _preview_candidates = samples[:effective_count]
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
                # 判断当前候选样本是否混合模式
                _cand_has_ref = sum(1 for s in _preview_candidates if (s.get("reference_answer") or "").strip())
                _prompt_is_mixed = _cand_has_ref > 0 and _cand_has_ref < len(_preview_candidates)

                # 从候选样本中挑出严格/合理性各一条
                _sample_strict = next((s for s in _preview_candidates if (s.get("reference_answer") or "").strip()), None)
                _sample_reasonable = next((s for s in _preview_candidates if not (s.get("reference_answer") or "").strip()), None)

                def _show_prompt_for_sample(sample, mode_label):
                    """展示单条样本的 prompt 示例。"""
                    if not sample:
                        st.info(f"当前候选样本中无{mode_label}样本")
                        return
                    _q = (sample.get("question") or "(无问题)")[:60]
                    st.markdown(f"**示例样本**：`{_q}` — {mode_label}")
                    prompt = build_judge_prompt(sample)
                    st.code(prompt, language=None)
                    st.caption(f"prompt 长度：{len(prompt)} 字符")

                if _prompt_is_mixed:
                    ptab_strict, ptab_reasonable = st.tabs([
                        f"严格评测 Prompt（含参考答案）",
                        f"合理性评测 Prompt（无参考答案）",
                    ])
                    with ptab_strict:
                        _show_prompt_for_sample(_sample_strict, "严格评测")
                    with ptab_reasonable:
                        _show_prompt_for_sample(_sample_reasonable, "合理性评测")
                else:
                    # 纯模式：直接展示
                    _any_sample = _preview_candidates[0] if _preview_candidates else None
                    if _cand_has_ref > 0:
                        _show_prompt_for_sample(_any_sample, "严格评测")
                    else:
                        _show_prompt_for_sample(_any_sample, "合理性评测")

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
                    samples_to_judge = samples[:max_samples]
                    skipped = 0
                else:
                    samples_to_judge, skipped = _compute_samples_to_judge(
                        samples, max_samples, skip_existing, force_rerun
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

            with_ref = metrics.get("with_ref_count", 0)
            without_ref = metrics.get("without_ref_count", 0)
            is_mixed = with_ref > 0 and without_ref > 0

            # 计算分组指标
            _strict_metrics = _compute_subset_metrics(judge_results, True)
            _reasonable_metrics = _compute_subset_metrics(judge_results, False)

            # 概览
            ov1, ov2, ov3 = st.columns(3)
            ov1.metric("总样本数", metrics["total"])
            ov2.metric("有效评测数", metrics["evaluated"])
            ov3.metric("错误数", metrics["errors"])

            # 评测模式构成
            if is_mixed:
                st.warning(
                    f"**混合模式**：严格评测 **{with_ref}** 条 + 合理性评测 **{without_ref}** 条。"
                    f"严格评测的 Answer OK = 与参考答案对比；合理性评测的 Answer OK = 基于检索内容判断。"
                    f"两者口径不同，下方可分别查看。"
                )
            elif with_ref > 0:
                st.success(f"**纯严格评测**：全部 **{with_ref}** 条均有参考答案")
            else:
                st.info(f"**纯合理性评测**：全部 **{without_ref}** 条均无参考答案")

            # ---------- 指标视图：tabs 或单视图 ----------
            def _render_metric_view(m, label, description):
                """渲染一组指标卡片。"""
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Top1 Hit", f"{m['top1_hit_rate']:.0%}")
                mc2.metric("Top3 Hit", f"{m['top3_hit_rate']:.0%}")
                mc3.metric("Top5 Hit", f"{m['top5_hit_rate']:.0%}")
                mc4.metric("Answer OK", f"{m['answer_correct_rate']:.0%}")
                st.caption(description)

            # ---------- 视图切换 + 下游内容（用 tabs 实现） ----------
            def _render_judge_view(view_valid, view_all, metrics_subset, metrics_desc, view_label=""):
                """渲染一个视图下的全部内容：指标、图表、诊断、详情。"""
                # 指标卡片
                _render_metric_view(metrics_subset, view_label, metrics_desc)

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

            # ---------- 混合模式用 tabs，纯模式直接渲染 ----------
            _mixed_metrics = {
                "top1_hit_rate": metrics["top1_hit_rate"],
                "top3_hit_rate": metrics["top3_hit_rate"],
                "top5_hit_rate": metrics["top5_hit_rate"],
                "answer_correct_rate": metrics["answer_correct_rate"],
            }

            if is_mixed:
                tab_all, tab_strict, tab_reasonable = st.tabs([
                    f"混合总览（{metrics['evaluated']} 条）",
                    f"严格评测（{with_ref} 条）",
                    f"合理性评测（{without_ref} 条）",
                ])

                with tab_all:
                    _render_judge_view(
                        valid_results, judge_results,
                        _mixed_metrics,
                        "混合口径：Top1~Top5 为统一标准，Answer OK 包含两种评测模式的结果，需谨慎解读",
                        "混合总览",
                    )

                with tab_strict:
                    _sv = _filter_by_view(valid_results, "strict")
                    _sa = _filter_by_view(judge_results, "strict")
                    if _strict_metrics:
                        _render_judge_view(
                            _sv, _sa,
                            _strict_metrics,
                            "严格口径：Answer OK = 回答是否与参考答案一致、覆盖关键要点",
                            "严格评测",
                        )
                    else:
                        st.info("无严格评测样本")

                with tab_reasonable:
                    _rv = _filter_by_view(valid_results, "reasonable")
                    _ra = _filter_by_view(judge_results, "reasonable")
                    if _reasonable_metrics:
                        _render_judge_view(
                            _rv, _ra,
                            _reasonable_metrics,
                            "合理性口径：Answer OK = 回答是否基于检索内容看起来合理、完整",
                            "合理性评测",
                        )
                    else:
                        st.info("无合理性评测样本")
            else:
                # 纯模式：不需要 tab 切换
                if with_ref > 0:
                    _render_judge_view(
                        valid_results, judge_results,
                        _mixed_metrics,
                        "严格口径：Answer OK = 回答是否与参考答案一致、覆盖关键要点",
                        "纯严格评测",
                    )
                else:
                    _render_judge_view(
                        valid_results, judge_results,
                        _mixed_metrics,
                        "合理性口径：Answer OK = 回答是否基于检索内容看起来合理、完整",
                        "纯合理性评测",
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
