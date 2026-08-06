import streamlit as st
from pathlib import Path
from datetime import datetime
import json
import io
import os
import re
import time

import psutil

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from parser import parse_langfuse_jsonl, save_results
from judge import judge_all, compute_metrics, call_llm, pre_screen, compute_content_hash, build_judge_prompt, load_prompt_template, load_prompt_template_with_ref, build_result_status, backfill_chunk_exact_topk
from question_generator import generate_questions, save_questions, export_csv_bytes, choose_strategy, STRATEGY_LABELS, MODE_RETRIEVAL, MODE_QA, MODE_LABELS, build_question_set_name
from batch_query import run_batch_query, push_to_raw_dir, export_csv_bytes as batch_export_csv

load_dotenv(Path(__file__).parent / ".env")

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
JUDGED_DIR = Path(__file__).parent / "data" / "judged"
JUDGED_FILE = JUDGED_DIR / "eval_results.jsonl"
BATCH_DIR = Path(__file__).parent / "data" / "batch"
QUESTIONS_DIR = Path(__file__).parent / "data" / "questions"
REPORTS_DIR = Path(__file__).parent / "data" / "reports"


def list_langfuse_export_files(raw_dir):
    """列出 raw_dir 下合法的 Langfuse 导出文件，按修改时间倒序。

    只保留：
    - langfuse_api_export*.jsonl（API 拉取）
    - Langfuse UI 导出文件（文件名含 lf-events-export 或首行含 traceId）
    - 首行含 traceId 字段的合法 JSONL

    排除：
    - batch_qa_*.jsonl（批量执行结果）
    - batch_results_*.jsonl
    - questions_*.jsonl（题集）
    - eval_results_*.jsonl（评测结果）
    - langfuse_samples.jsonl（解析产物）

    Returns:
        list[dict]: [{"path": Path, "name": str, "label": str, "mtime": float, "size_kb": float}, ...]
    """
    exclude_prefixes = ("batch_qa_", "batch_results_", "questions_", "eval_results_")
    exclude_names = {"langfuse_samples.jsonl"}
    result = []

    if not raw_dir.exists():
        return result

    for f in raw_dir.glob("*.jsonl"):
        name = f.name
        # 排除已知非导出文件
        if name in exclude_names:
            continue
        if any(name.startswith(p) for p in exclude_prefixes):
            continue

        # 判断是否为 Langfuse 导出文件
        is_export = False
        # 名称匹配：API 拉取或 UI 导出
        if name.startswith("langfuse_api_export") or "lf-events-export" in name:
            is_export = True
        else:
            # 内容匹配：首行含 traceId
            try:
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if "traceId" in obj:
                                is_export = True
                        except json.JSONDecodeError:
                            pass
                        break  # 只检查首行
            except Exception:
                pass

        if not is_export:
            continue

        stat = f.stat()
        mtime = stat.st_mtime
        size_kb = stat.st_size / 1024
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        label = f"{name}  |  {mtime_str}  |  {size_kb:.1f} KB"
        result.append({
            "path": f,
            "name": name,
            "label": label,
            "mtime": mtime,
            "size_kb": size_kb,
        })

    # 按修改时间倒序
    result.sort(key=lambda x: x["mtime"], reverse=True)
    return result


# ─── 缓存函数（避免每次 rerun 重新扫描磁盘） ────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _build_question_set_index(cache_key=""):
    """扫描 data/questions/ 目录，返回题集元数据列表（轻量级，不含题目内容）。

    cache_key: 传入目录 mtime 或手动递增的值，用于控制缓存失效。
    返回 list of dict，每个 dict 包含 Path-serializable 的元数据。
    """
    if not QUESTIONS_DIR.exists():
        return []

    results = []
    for f in QUESTIONS_DIR.glob("*.jsonl"):
        # 条件 A：有 manifest
        manifest_path = f.parent / f"{f.stem}_manifest.json"
        is_qs = manifest_path.exists()
        # 条件 B：前3行含 question_set_id
        if not is_qs:
            try:
                with f.open("r", encoding="utf-8") as fh:
                    checked = 0
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        checked += 1
                        if checked > 3:
                            break
                        try:
                            obj = json.loads(line)
                            if obj.get("question_set_id"):
                                is_qs = True
                                break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        if not is_qs:
            continue

        # 检测文件信息（前20行）
        info = {
            "modes": {"retrieval": 0, "qa": 0, "chunk_exact": 0, "unknown": 0},
            "set_name": "", "set_id": "", "question_count": 0,
            "has_set_info": False, "source_format": "", "source_file_name": "",
            "evaluation_type": "",
        }
        try:
            with f.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    info["question_count"] += 1
                    if i >= 20:
                        continue
                    try:
                        obj = json.loads(line)
                        mode = obj.get("question_mode", "")
                        if mode == "retrieval":
                            info["modes"]["retrieval"] += 1
                        elif mode == "qa":
                            info["modes"]["qa"] += 1
                        elif mode == "chunk_exact":
                            info["modes"]["chunk_exact"] += 1
                        else:
                            info["modes"]["unknown"] += 1
                        if obj.get("question_set_name") and not info["set_name"]:
                            info["set_name"] = obj["question_set_name"]
                            info["set_id"] = obj.get("question_set_id", "")
                            info["has_set_info"] = True
                        if obj.get("source_format") and not info["source_format"]:
                            info["source_format"] = obj["source_format"]
                            info["source_file_name"] = obj.get("source_file_name", "")
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        # created_at + evaluation_type from manifest
        created_at = None
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                ca = m.get("created_at")
                if ca:
                    created_at = ca  # ISO string
                et = m.get("evaluation_type", "")
                if et and not info["evaluation_type"]:
                    info["evaluation_type"] = et
            except Exception:
                pass
        if not created_at:
            sid = info.get("set_id", "")
            parts = sid.split("_", 3)
            if len(parts) >= 3 and len(parts[1]) == 8:
                created_at = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"

        results.append({
            "path": str(f),
            "name": f.name,
            "modes": info["modes"],
            "set_name": info["set_name"],
            "set_id": info["set_id"],
            "question_count": info["question_count"],
            "has_set_info": info["has_set_info"],
            "source_format": info["source_format"],
            "source_file_name": info["source_file_name"],
            "created_at": created_at,
        })

    results.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return results


@st.cache_data(ttl=60, show_spinner=False)
def _build_run_summary_index(cache_key=""):
    """扫描 experiments 目录，返回所有 run 的轻量摘要（不含 config_snapshot）。

    返回 list of dict: run_id, config_id, question_set_id, question_set_name,
    status, question_count, started_at。
    """
    from experiment import EXPERIMENTS_DIR
    if not EXPERIMENTS_DIR.exists():
        return []

    summaries = []
    for run_dir in EXPERIMENTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            summaries.append({
                "run_id": m.get("run_id", ""),
                "config_id": m.get("config_id", ""),
                "question_set_id": m.get("question_set_id", ""),
                "question_set_name": m.get("question_set_name", ""),
                "status": m.get("status", ""),
                "question_count": m.get("question_count", 0),
                "started_at": m.get("started_at", ""),
            })
        except (json.JSONDecodeError, IOError):
            continue
    return summaries


def _get_questions_dir_mtime():
    """获取 questions 目录的 mtime，用作缓存 key。"""
    if not QUESTIONS_DIR.exists():
        return ""
    try:
        return str(QUESTIONS_DIR.stat().st_mtime)
    except Exception:
        return ""


def _get_experiments_dir_mtime():
    """获取 experiments 目录的 mtime，用作缓存 key。"""
    from experiment import EXPERIMENTS_DIR
    if not EXPERIMENTS_DIR.exists():
        return ""
    try:
        return str(EXPERIMENTS_DIR.stat().st_mtime)
    except Exception:
        return ""


# ─── RSS 内存采样 ────────────────────────────────────────────────────────────

def _get_rss_mb():
    """返回当前进程 RSS（MB）。"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _render_ce_topk(evaluable_count: int, top1: int, top3: int,
                    top5: int, top10: int, total: int = 0):
    """渲染 Chunk Exact TopK 指标的紧凑两行布局。

    第一行：Top1 / Top3 / Top5 / Top10 百分比（st.metric）。
    第二行：X / Y 命中（st.caption）。
    可评测数量单独显示在最左侧。
    """
    n = evaluable_count
    if n <= 0:
        return

    metrics = [
        ("Top1", top1),
        ("Top3", top3),
        ("Top5", top5),
        ("Top10", top10),
    ]
    labels = ["可评测"] + [m[0] for m in metrics]
    cols = st.columns(len(labels))

    with cols[0]:
        _eval_label = f"{n}/{total}" if total else str(n)
        st.metric("可评测", _eval_label)

    for i, (label, hit) in enumerate(metrics):
        with cols[i + 1]:
            pct = hit / n
            st.metric(label, f"{pct:.1%}")

    # 第二行：分子分母
    hit_parts = [f"{label}: **{hit}** / {n}" for label, hit in metrics]
    st.caption("命中 — " + " | ".join(hit_parts))


def _record_rss(stage):
    """记录当前 RSS 到 session_state，用于内存用量分析。"""
    if "_rss_log" not in st.session_state:
        st.session_state["_rss_log"] = []
    st.session_state["_rss_log"].append({
        "stage": stage,
        "rss_mb": round(_get_rss_mb(), 1),
        "ts": datetime.now().strftime("%H:%M:%S"),
    })


# ─── 缓存加载 ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, max_entries=4, show_spinner=False)
def _load_sample_lookup(cache_key="", proc_path_str=""):
    """加载 processed samples 为 {trace_id: sample} 查找表（不含 observations）。

    缓存 120 秒、最多 4 个条目，避免实验看板切换 run 时累积占用。
    cache_key 应包含 proc_mtime 以在文件更新后自动失效。
    proc_path_str: 显式指定的 processed 文件路径（隔离路径优先）。
    """
    lookup = {}
    if proc_path_str:
        proc_path = Path(proc_path_str)
    else:
        proc_path = PROCESSED_DIR / "langfuse_samples.jsonl"
    if not proc_path.exists():
        return lookup
    try:
        with proc_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    obj.pop("observations", None)
                    tid = obj.get("trace_id")
                    if tid:
                        lookup[tid] = obj
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass
    return lookup


def _clear_session_samples():
    """清理 session 中的解析结果（不删除磁盘文件）。"""
    for _k in ("samples", "summary", "sample_page", "_use_frozen_source"):
        st.session_state.pop(_k, None)


def _on_project_changed(old_proj_id: str, new_proj_id: str):
    """项目切换时清理 session 中的旧解析结果。

    当 old_proj_id 为空（启动时无项目 / legacy fallback）且 new_proj_id 非空，
    或两者不同且均非空时，清理 session 中的 samples/summary。
    """
    if new_proj_id and (not old_proj_id or old_proj_id != new_proj_id):
        _clear_session_samples()


def _load_samples_from_path(proc_path_str):
    """从指定路径加载 processed samples（不含 observations），返回 {trace_id: sample}。"""
    lookup = {}
    if not proc_path_str:
        return lookup
    proc_path = Path(proc_path_str)
    if not proc_path.exists():
        return lookup
    try:
        with proc_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    obj.pop("observations", None)
                    tid = obj.get("trace_id")
                    if tid:
                        lookup[tid] = obj
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass
    return lookup


def _build_merged_sample_lookup(config_runs):
    """为报告导出构建跨 run 的 merged sample lookup。

    按每个 run 的 provenance 定位 processed file，合并所有 samples。
    返回 (merged_lookup, provenance_info)：
      - merged_lookup: {trace_id: sample}
      - provenance_info: dict 含 source_paths、total_loaded、fallback_count
    """
    from langfuse_project import find_processed_for_run

    merged = {}
    source_paths = {}  # path -> sample_count
    fallback_count = 0

    for run in config_runs:
        run_id = run.get("run_id", "")
        if not run_id:
            continue
        proc_path = find_processed_for_run(run_id)
        if not proc_path or not Path(proc_path).exists():
            fallback_count += 1
            continue
        samples = _load_samples_from_path(proc_path)
        source_paths[proc_path] = len(samples)
        merged.update(samples)

    provenance_info = {
        "source_paths": source_paths,
        "total_loaded": len(merged),
        "run_count": len(config_runs),
        "fallback_count": fallback_count,
    }
    return merged, provenance_info


def _resolve_processed_path():
    """解析当前最合适的 processed 文件路径（隔离路径优先，回退全局）。

    返回 (path_str, mtime_key) 供 _load_sample_lookup 使用。
    """
    _proj_id = st.session_state.get("_lf_project_info", {}).get("project_id", "")
    if _proj_id:
        try:
            from langfuse_project import find_latest_processed as _flp
        except ImportError:
            _flp = None
        if _flp:
            try:
                _s, _ = _flp(_proj_id)
                if _s and _s.exists():
                    return str(_s), str(_s.stat().st_mtime)
            except Exception:
                pass
    _gp = PROCESSED_DIR / "langfuse_samples.jsonl"
    return str(_gp), str(_gp.stat().st_mtime) if _gp.exists() else ""


def _get_created_at(filepath, info):
    """获取题集的创建时间，优先级：manifest created_at > set_id 时间戳 > 文件名时间戳。

    返回 datetime 对象；若均无法解析则返回 None（排序时排在最后）。
    """
    # 1. 检查 manifest 文件
    manifest_path = filepath.parent / f"{filepath.stem}_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            created_at = manifest.get("created_at")
            if created_at:
                return datetime.fromisoformat(created_at)
        except Exception:
            pass

    # 2. 从 set_id 解析时间戳（格式: qs_YYYYMMDD_HHMMSSffffff_slug）
    set_id = info.get("set_id", "")
    if set_id:
        parts = set_id.split("_", 3)
        if len(parts) >= 3:
            date_part = parts[1]  # YYYYMMDD
            time_part = parts[2]  # HHMMSSffffff
            try:
                if len(date_part) == 8 and len(time_part) >= 6:
                    ts_str = date_part + time_part[:6]
                    return datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            except (ValueError, IndexError):
                pass

    # 3. 从文件名解析时间戳
    match = re.search(r'(\d{8}_\d{6})', filepath.stem)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    return None


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

        # 4. TopK 判定与 Judge 原因
        st.markdown("**检索命中判定**")
        _track = result.get("evaluation_track", "")
        _topk_line = f"Top1 {'✓ 命中' if _t1 else '✗ 未命中'} | Top3 {'✓ 命中' if _t3 else '✗ 未命中'} | Top5 {'✓ 命中' if _t5 else '✗ 未命中'}"
        if _track == "chunk_exact":
            _t10 = result.get("retrieval_top10_hit")
            if _t10 is not None:
                _topk_line += f" | Top10 {'✓ 命中' if _t10 else '✗ 未命中'}"
        st.markdown(_topk_line)
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
    elif track == "chunk_exact":
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

    from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_CHUNK_EXACT

    # ── 筛选控件 ──
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        track_options = ["全部"]
        tracks_present = set(r.get("evaluation_track", "") for r in results)
        if TRACK_RETRIEVAL in tracks_present:
            track_options.append("retrieval")
        if TRACK_CHUNK_EXACT in tracks_present:
            track_options.append("chunk_exact")
        if TRACK_STRICT_QA in tracks_present:
            track_options.append("strict_qa")
        if TRACK_GROUNDED_QA in tracks_present:
            track_options.append("grounded_qa")
        filter_track = st.selectbox("按评测轨道筛选", track_options, key=f"{key_prefix}_track")
    with filter_col2:
        filter_status = st.selectbox(
            "按结果状态筛选",
            ["全部", "命中/正确", "未命中/错误", "Top1 未命中", "Top3 未命中", "Top5 未命中", "Top10 未命中", "错误"],
            key=f"{key_prefix}_status",
        )
    with filter_col3:
        filter_keyword = st.text_input("搜索题目关键字", "", key=f"{key_prefix}_kw")

    # ── 应用筛选 ──
    filtered = list(results)
    if filter_track != "全部":
        filtered = [r for r in filtered if r.get("evaluation_track") == filter_track]
    if filter_status == "命中/正确":
        filtered = [r for r in filtered if "error" not in r and (
            r.get("retrieval_top1_hit") or r.get("answer_correct"))]
    elif filter_status == "未命中/错误":
        filtered = [r for r in filtered if "error" not in r and (
            not r.get("retrieval_top1_hit") and not r.get("answer_correct"))]
    elif filter_status == "Top1 未命中":
        filtered = [r for r in filtered if "error" not in r
                    and r.get("evaluation_track") == TRACK_RETRIEVAL
                    and not r.get("retrieval_top1_hit")]
    elif filter_status == "Top3 未命中":
        filtered = [r for r in filtered if "error" not in r
                    and r.get("evaluation_track") == TRACK_RETRIEVAL
                    and not r.get("retrieval_top3_hit")]
    elif filter_status == "Top5 未命中":
        filtered = [r for r in filtered if "error" not in r
                    and r.get("evaluation_track") == TRACK_RETRIEVAL
                    and not r.get("retrieval_top5_hit")]
    elif filter_status == "Top10 未命中":
        filtered = [r for r in filtered if "error" not in r
                    and r.get("evaluation_track") == TRACK_CHUNK_EXACT
                    and not r.get("retrieval_top10_hit")]
    elif filter_status == "错误":
        filtered = [r for r in filtered if "error" in r]
    if filter_keyword:
        _kw = filter_keyword.lower()
        filtered = [r for r in filtered if _kw in (r.get("question") or "").lower()]

    st.caption(f"筛选后 {len(filtered)} 条结果（共 {len(results)} 条）")

    if not filtered:
        st.info("无匹配的评测结果")
        return

    # ── 分页 ──
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

    # ── 渲染当前页 ──
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
    "1. **题目生成** — 上传知识库文件（.txt/.md/.docx/.xlsx/.xls/.csv），自动按章节切分后调用 LLM 出题，"
    "生成带参考答案的评测题集\n"
    "2. **批量提问** — 选择题集和 RAG 配置方案，通过 Dify Workflow API 批量提问，"
    "收集回答与检索结果\n"
    "3. **样本准备** — 解析 Dify / Langfuse 记录为结构化样本，回填参考答案和运行元数据\n"
    "4. **Judge 评测** — 按评测轨道自动评分：检索评测关注 Top1/3/5 命中，"
    "问答评测关注回答正确性/合理性"
)
st.sidebar.divider()
st.sidebar.markdown("**运行看板** — 按配置方案查看累计结果、运行历史和单次运行详情")
st.sidebar.markdown("**知识库探索** — 浏览 Dify 知识库、文档和分块内容，检测重复分块")
st.sidebar.caption("切换上方 Tab 进入对应工作区，每个 Tab 内均有独立配置面板和详细说明。")

# --- 内存用量显示 ---
_rss_log = st.session_state.get("_rss_log", [])
if _rss_log:
    with st.sidebar.expander("内存用量 (RSS)", expanded=False):
        for entry in _rss_log:
            st.caption(f"{entry['ts']} | {entry['stage']} | {entry['rss_mb']:.0f} MB")
        _delta = _rss_log[-1]["rss_mb"] - _rss_log[0]["rss_mb"]
        st.caption(f"累计变化: {_delta:+.0f} MB")

# ── 自动恢复上次连接的项目 ──
# session_state 在新浏览器 tab 中为空，但项目注册表在磁盘上持久化。
# 启动时自动加载最近同步的项目，使 find_latest_processed 能找到隔离路径。
if "_lf_project_info" not in st.session_state:
    try:
        from langfuse_project import list_projects as _lp_list
        _registered = _lp_list()
        if _registered:
            # 取最近同步的项目
            _registered.sort(key=lambda p: p.get("last_sync_at", "") or "", reverse=True)
            _auto_proj = _registered[0]
            st.session_state["_lf_project_info"] = {
                "project_id": _auto_proj["project_id"],
                "project_name": _auto_proj.get("project_name", ""),
                "host": _auto_proj.get("host", ""),
                "key_masked": _auto_proj.get("key_masked", ""),
            }
    except Exception:
        pass

# ── 清理跨项目残留的 _use_frozen_source ──
_loaded_proj_id = st.session_state.get("_lf_project_info", {}).get("project_id", "")
_frozen_src = st.session_state.get("_use_frozen_source")
if _frozen_src and _loaded_proj_id:
    _frozen_pid = _frozen_src.get("project_id", "")
    if _frozen_pid and _frozen_pid != _loaded_proj_id:
        st.session_state.pop("_use_frozen_source", None)
        _frozen_src = None

# ── 强制刷新：当前缓存有效时，从磁盘重新加载最新解析结果 ──
# 核心问题：Streamlit session_state 跨浏览器 tab 持久化。
# 上次会话解析的 frozen snapshot 样本会残留，即使当前缓存已有更新数据。
# 解决：比较当前缓存 trace_count 与 session 中样本数，不一致则强制从磁盘重新加载。
# 保护：_frozen_source_just_set 标记防止用户刚选择的冻结源被误清。
# 重新读取 project_id（button handler 可能已更新 session_state）
_loaded_proj_id = st.session_state.get("_lf_project_info", {}).get("project_id", "")
_need_reload = "samples" not in st.session_state
_just_set = st.session_state.pop("_frozen_source_just_set", False)
if not _need_reload and _loaded_proj_id and not _just_set:
    try:
        from langfuse_project import get_current_cache_stats as _gccs_check
        _ccs_check = _gccs_check(_loaded_proj_id)
        _cc_trace = _ccs_check.get("trace_count", 0)
        _session_count = len(st.session_state.get("samples") or [])
        # 当前缓存有数据 且 数量不同 → 强制刷新（旧 session 残留数据过期）
        if _cc_trace > 0 and _cc_trace != _session_count:
            _need_reload = True
            # 清除过期的 _use_frozen_source（来自上一次 session）
            st.session_state.pop("_use_frozen_source", None)
            _frozen_src = None
    except Exception:
        pass

if _need_reload:
    try:
        from langfuse_project import find_latest_processed as _flp_startup
        _samples_file, _summary_file = _flp_startup(_loaded_proj_id)
    except ImportError:
        _samples_file = PROCESSED_DIR / "langfuse_samples.jsonl"
        _summary_file = PROCESSED_DIR / "langfuse_summary.json"
    if _samples_file and _samples_file.exists():
        with open(_samples_file, "r", encoding="utf-8") as f:
            _loaded = [json.loads(line) for line in f if line.strip()]
        for _s in _loaded:
            _s.pop("observations", None)
        st.session_state["samples"] = _loaded
    if _summary_file and _summary_file.exists():
        st.session_state["summary"] = json.loads(_summary_file.read_text(encoding="utf-8"))

samples = st.session_state.get("samples")
summary = st.session_state.get("summary") or {}

# 记录初始 RSS
if "_rss_init" not in st.session_state:
    _record_rss("初始加载")
    st.session_state["_rss_init"] = True

# --- Tabs ---
tab_kb, tab_qgen, tab_batch, tab_samples, tab_judge, tab_experiment = st.tabs(["知识库探索", "题目生成", "批量提问", "样本准备", "Judge 评测", "运行看板"])

# ========== Tab: 知识库探索 ==========
with tab_kb:
    st.subheader("知识库探索")
    st.caption("浏览 Dify 知识库、文档和分块内容，检测重复分块，导出 chunk catalog snapshot")

    with st.expander("知识库探索模块说明（点击展开）", expanded=False):
        st.markdown("""
**一句话总览：** 连接 Dify 知识库 API，浏览数据集 → 文档 → 分块的层级结构，对分块内容计算 SHA-256 哈希以检测重复，并导出可复现的 chunk catalog snapshot。

---

**功能说明**

| 功能 | 说明 |
|------|------|
| 列出知识库 | 调用 `GET /datasets` 获取所有数据集 |
| 列出文档 | 调用 `GET /datasets/{id}/documents` 分页展示文档列表 |
| 分块浏览 | 调用 `GET /datasets/{id}/documents/{id}/segments`，支持 status 过滤和分页 |
| 重复检测 | 对分块内容做规范化 SHA-256 哈希，标记内容完全相同的分块 |
| 导出 | JSON 和 CSV 格式导出 chunk catalog（含 content_hash） |

**安全说明**
- 本模块仅使用 GET 请求，**不会**上传、删除、编辑、启用或禁用任何文档或分块
- API Key 仅从本地环境变量或连接配置的安全存储读取，不写入任何导出文件、日志或 Git

**分页参数**
- `page`：页码（从 1 开始）
- `limit`：每页数量，最大 100
""")

    # ── 连接配置（仅知识库 API Key） ──────────────────────────
    from dify_kb_connection import (
        list_kb_profiles, load_kb_profile, create_kb_profile,
        create_kb_profile_from_env, update_kb_profile, delete_kb_profile,
        get_kb_api_key, has_kb_api_key,
        mask_api_key as kb_mask_api_key,
        validate_dataset_key,
    )
    from dify_knowledge import (
        list_datasets, list_documents, list_segments, list_all_documents, retrieve,
        build_chunk_catalog, detect_duplicates,
        export_catalog_json, export_catalog_csv,
        check_connection,
    )

    _kb_env_api_key = os.getenv("DIFY_DATASET_API_KEY", "")
    _kb_env_base_url = os.getenv("DIFY_DATASET_API_BASE", "") or "http://localhost/v1"

    # ── 用户偏好（非敏感） ──
    _prefs_path = Path("data/user_preferences.json")

    def _load_prefs():
        if _prefs_path.exists():
            try:
                return json.loads(_prefs_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_prefs(prefs):
        _prefs_path.parent.mkdir(parents=True, exist_ok=True)
        _prefs_path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_last_kb_profile_id():
        return _load_prefs().get("last_selected_kb_profile_id", "")

    def _set_last_kb_profile_id(pid):
        prefs = _load_prefs()
        if pid:
            prefs["last_selected_kb_profile_id"] = pid
        else:
            prefs.pop("last_selected_kb_profile_id", None)
        _save_prefs(prefs)

    with st.expander("知识库 API 连接配置", expanded=True):
        st.caption("本页仅使用 **知识库 API Key**（`dataset-` 开头），与批量提问的 App Key（`app-` 开头）完全独立。")

        # 列出已保存的配置
        kb_profiles = list_kb_profiles()

        # ── 自动选择逻辑 ──
        _auto_selected_pid = ""
        _auto_reason = ""
        _last_pid = _get_last_kb_profile_id()

        if kb_profiles:
            _valid_pids = {p.get("profile_id", "") for p in kb_profiles}

            if _last_pid and _last_pid in _valid_pids:
                # 优先级 1：上次选择的 profile 仍然存在
                _auto_selected_pid = _last_pid
                _auto_reason = "已恢复上次使用的连接"
            elif len(kb_profiles) == 1:
                # 优先级 2：只有一个 profile，自动选择
                _auto_selected_pid = kb_profiles[0].get("profile_id", "")
                _auto_reason = "已自动选择唯一可用连接"
            # 优先级 4：多个 profile 无历史选择 → 保持 "(请选择)"
        elif _kb_env_api_key:
            # 优先级 3：无 profile 但有环境变量
            _auto_selected_pid = "__env__"
            _auto_reason = "已自动选择环境变量默认连接"

        # 如果有历史选择但 profile 已被删除，清除失效偏好
        if _last_pid and _last_pid not in {p.get("profile_id", "") for p in kb_profiles} and _last_pid != "__env__":
            _set_last_kb_profile_id("")
            if _auto_reason == "":
                st.caption("⚠️ 上次使用的连接已删除，已清除历史偏好")
        kb_api_key = ""
        kb_base_url = _kb_env_base_url

        if kb_profiles:
            kb_profile_options = []
            for p in kb_profiles:
                pid = p.get("profile_id", "")
                pname = p.get("profile_name", "未命名")
                purl = p.get("base_url", "")
                pmasked = p.get("key_masked", "")
                label = f"{pname} · {purl}"
                if pmasked:
                    label += f" · Key: {pmasked}"
                kb_profile_options.append((pid, label))

            # 计算默认选中索引
            _options_list = [""] + [c[0] for c in kb_profile_options]
            _default_idx = 0
            if _auto_selected_pid and _auto_selected_pid in _options_list:
                _default_idx = _options_list.index(_auto_selected_pid)

            kb_selected_pid = st.selectbox(
                "选择知识库连接配置",
                options=_options_list,
                index=_default_idx,
                format_func=lambda x: (
                    "（请选择）" if not x
                    else next((c[1] for c in kb_profile_options if c[0] == x), x)
                ),
                key="kb_selected_profile",
            )

            # 用户手动切换后更新偏好
            if kb_selected_pid and kb_selected_pid != _last_pid:
                _set_last_kb_profile_id(kb_selected_pid)

            # 显示自动选择提示
            if _auto_reason and kb_selected_pid == _auto_selected_pid:
                _auto_profile_name = next(
                    (c[1].split(" · ")[0] for c in kb_profile_options if c[0] == _auto_selected_pid),
                    _auto_selected_pid
                )
                st.caption(f"✅ {_auto_reason}：{_auto_profile_name}")

            if kb_selected_pid:
                kb_sel_meta = load_kb_profile(kb_selected_pid)
                if kb_sel_meta:
                    kb_base_url = kb_sel_meta.get("base_url", _kb_env_base_url)
                    _sel_key_source = kb_sel_meta.get("key_source", "keyring")
                    kb_saved_key = get_kb_api_key(kb_selected_pid)
                    if kb_saved_key:
                        kb_api_key = kb_saved_key
                        _source_label = "环境变量引用" if _sel_key_source.startswith("env:") else "安全存储"
                        st.markdown(f"**当前使用：{kb_sel_meta.get('profile_name', '未命名')}**")
                        st.caption(f"Key 来源：{_source_label}（`{kb_mask_api_key(kb_saved_key)}`）")
                        st.caption(f"Base URL：`{kb_base_url}`")
                    else:
                        st.warning("该配置未保存 Key，请重新创建或手动输入。")

            # 管理操作
            mgmt_c1, mgmt_c2, mgmt_c3 = st.columns(3)
            with mgmt_c1:
                if st.button("➕ 新建配置", key="kb_new_profile_btn"):
                    st.session_state["kb_show_new_form"] = True
            with mgmt_c2:
                if st.button("✏️ 编辑配置", key="kb_edit_profile_btn",
                             disabled=not kb_selected_pid):
                    st.session_state["kb_show_edit_form"] = True
            with mgmt_c3:
                if st.button("🗑️ 删除配置", key="kb_delete_profile_btn",
                             disabled=not kb_selected_pid):
                    st.session_state["kb_show_delete_confirm"] = True

            # ── 新建配置表单 ──
            if st.session_state.get("kb_show_new_form"):
                with st.form("kb_new_profile_form"):
                    st.markdown("**新建知识库连接配置**")
                    np_name = st.text_input("配置名称 *", placeholder="例如：产品知识库", key="kb_np_name")
                    np_url = st.text_input("Base URL *", value=_kb_env_base_url, key="kb_np_url")
                    np_key = st.text_input(
                        "Dataset API Key *",
                        type="password",
                        key="kb_np_key",
                        placeholder="dataset-...",
                    )
                    np_submit = st.form_submit_button("保存")
                if np_submit and np_name and np_url and np_key:
                    ok, err = validate_dataset_key(np_key)
                    if not ok:
                        st.error(f"Key 校验失败: {err}")
                    else:
                        try:
                            create_kb_profile(np_name, np_url, np_key)
                            st.success(f"知识库连接配置「{np_name}」已保存")
                            st.session_state["kb_show_new_form"] = False
                            st.rerun()
                        except ValueError as exc:
                            st.error(f"保存失败: {exc}")

            # ── 编辑配置表单 ──
            if st.session_state.get("kb_show_edit_form") and kb_selected_pid:
                _edit_meta = load_kb_profile(kb_selected_pid)
                if _edit_meta:
                    with st.form("kb_edit_profile_form"):
                        st.markdown(f"**编辑: {_edit_meta.get('profile_name', '')}**")
                        ep_name = st.text_input("配置名称", value=_edit_meta.get("profile_name", ""), key="kb_ep_name")
                        ep_url = st.text_input("Base URL", value=_edit_meta.get("base_url", ""), key="kb_ep_url")
                        ep_key = st.text_input(
                            "新 Dataset API Key（留空则保留现有）",
                            type="password",
                            key="kb_ep_key",
                            placeholder="dataset-...",
                        )
                        ep_submit = st.form_submit_button("保存")
                    if ep_submit:
                        try:
                            update_kb_profile(
                                kb_selected_pid,
                                {"profile_name": ep_name, "base_url": ep_url},
                                api_key=ep_key if ep_key else None,
                            )
                            st.success("配置已更新")
                            st.session_state["kb_show_edit_form"] = False
                            st.rerun()
                        except ValueError as exc:
                            st.error(f"更新失败: {exc}")

            # ── 删除确认 ──
            if st.session_state.get("kb_show_delete_confirm") and kb_selected_pid:
                _del_meta = load_kb_profile(kb_selected_pid)
                _del_name = _del_meta.get("profile_name", "") if _del_meta else ""
                st.warning(f"确认删除配置「{_del_name}」？此操作不可撤销。")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("确认删除", key="kb_confirm_delete"):
                        delete_kb_profile(kb_selected_pid)
                        st.success(f"已删除「{_del_name}」")
                        st.session_state["kb_show_delete_confirm"] = False
                        st.rerun()
                with dc2:
                    if st.button("取消", key="kb_cancel_delete"):
                        st.session_state["kb_show_delete_confirm"] = False
                        st.rerun()

        else:
            # 无已保存配置：检查环境变量
            if _kb_env_api_key:
                _env_ok, _env_err = validate_dataset_key(_kb_env_api_key)
                if _env_ok:
                    kb_api_key = _kb_env_api_key
                    # 保存环境变量选择偏好
                    if _auto_selected_pid == "__env__" and _last_pid != "__env__":
                        _set_last_kb_profile_id("__env__")
                    st.markdown("**当前使用：环境变量默认连接（未保存为命名配置）**")
                    if _auto_reason:
                        st.caption(f"✅ {_auto_reason}")
                    st.caption(f"Key 来源：`DIFY_DATASET_API_KEY`（`{kb_mask_api_key(_kb_env_api_key)}`）")
                    st.caption(f"Base URL：`{kb_base_url}`")
                    # 保存为命名配置
                    if st.button("💾 保存为连接配置", key="kb_save_env_as_profile"):
                        try:
                            create_kb_profile_from_env(
                                "环境变量默认连接",
                                kb_base_url,
                            )
                            st.success("已保存为命名配置「环境变量默认连接」")
                            st.rerun()
                        except ValueError as exc:
                            st.error(f"保存失败: {exc}")
                else:
                    st.warning(f"环境变量 `DIFY_DATASET_API_KEY` 无效: {_env_err}")
            else:
                # 真正的空状态
                st.info("暂无可用的知识库连接配置。请新增连接配置，或在 .env 中设置 `DIFY_DATASET_API_KEY`。")

        # 手动输入（无配置且无环境变量时的入口）
        if not kb_api_key:
            st.markdown("**手动输入知识库 API 连接信息**")
            _kb_manual_key = st.text_input(
                "Dataset API Key（dataset-...）",
                type="password",
                key="kb_manual_key",
                value="",
                help="必须是 dataset- 开头的知识库专用 API Key，不可使用 app- 开头的应用 Key",
            )
            if _kb_manual_key:
                _manual_ok, _manual_err = validate_dataset_key(_kb_manual_key)
                if _manual_ok:
                    kb_api_key = _kb_manual_key
                else:
                    st.error(f"Key 校验失败: {_manual_err}")

            kb_base_url = st.text_input(
                "知识库 API Base URL",
                value=kb_base_url,
                key="kb_base_url_input",
                help="默认从 DIFY_DATASET_API_BASE 读取",
            )

        if not kb_api_key:
            st.warning(
                "请配置知识库 API Key（`dataset-` 开头）以使用知识库探索功能。\n\n"
                "**获取方式：** Dify 后台 → 知识库 → 选择知识库 → API 访问 → 复制 Dataset API Key\n\n"
                "**配置方式：** 在 .env 中设置 `DIFY_DATASET_API_KEY=dataset-xxx`，"
                "或点击「➕ 新建配置」保存，或在下方手动输入。"
            )

    # ── 以下为需要 Dataset API Key 的功能 ──
    # 使用 guard 条件渲染，不调用 st.stop()，避免阻塞其他 tab
    if not kb_api_key:
        # 无 key 时：知识库探索页显示提示，chunk_exact 入口禁用
        st.info("配置 Dataset API Key 后，即可使用知识库浏览、分块导出和 chunk_exact 题集创建功能。")
    else:
        # 连接测试
        if st.button("🔗 测试知识库连接", key="kb_test_conn"):
            ok, msg = check_connection(kb_api_key, kb_base_url)
            if ok:
                st.success(f"连接成功: {msg}")
            else:
                st.error(f"连接失败: {msg}")

        st.caption(f"Base URL: `{kb_base_url}`")

    # ── 知识库选择 ────────────────────────────────────────────
    if kb_api_key:
        st.markdown("### 知识库列表")

        try:
            with st.spinner("正在加载知识库列表..."):
                datasets = list_datasets(kb_api_key, kb_base_url)
        except RuntimeError as exc:
            st.error(f"获取知识库列表失败: {exc}")
            datasets = []

        if not datasets:
            st.info("未找到任何知识库。请检查连接配置和 API Key 权限。")
        else:
            # 构建选择列表
            ds_options = []
            ds_name_map = {}  # ds_id -> ds_name
            for ds in datasets:
                ds_id = ds.get("id", "")
                ds_name = ds.get("name", "未命名")
                ds_doc_count = ds.get("document_count", 0)
                ds_word_count = ds.get("word_count", 0)
                label = f"{ds_name}（{ds_doc_count} 篇文档，{ds_word_count:,} 词）"
                ds_options.append((ds_id, label))
                ds_name_map[ds_id] = ds_name

            selected_ds_id = st.selectbox(
                "选择知识库",
                options=[c[0] for c in ds_options],
                format_func=lambda x: next(
                    (c[1] for c in ds_options if c[0] == x), x
                ),
                key="kb_selected_dataset",
            )

            # 切换知识库时清除其他 dataset 的全库候选统计缓存
            if selected_ds_id:
                _current_stats_key = f"_ce_ds_stats_{selected_ds_id}"
                _stale_keys = [
                    k for k in st.session_state
                    if k.startswith("_ce_ds_stats_") and k != _current_stats_key
                ]
                for k in _stale_keys:
                    del st.session_state[k]

                # 缓存全库文档列表（供全知识库导出和出题使用）
                _kb_docs_cache_key = f"_kb_all_docs_{selected_ds_id}"
                if _kb_docs_cache_key not in st.session_state:
                    try:
                        _cached_all_docs = list_all_documents(
                            kb_api_key, kb_base_url, selected_ds_id,
                        )
                        st.session_state[_kb_docs_cache_key] = _cached_all_docs
                    except Exception:
                        st.session_state[_kb_docs_cache_key] = []
                st.session_state["_kb_all_docs"] = st.session_state[_kb_docs_cache_key]

            if selected_ds_id:
                # ── 文档列表 ──────────────────────────────────────
                st.markdown("### 文档列表")

                doc_page = st.session_state.get("kb_doc_page", 1)
                doc_limit = 20

                try:
                    with st.spinner("正在加载文档列表..."):
                        doc_result = list_documents(
                            kb_api_key, kb_base_url, selected_ds_id,
                            page=doc_page, limit=doc_limit,
                        )
                except RuntimeError as exc:
                    st.error(f"获取文档列表失败: {exc}")
                    doc_result = {"data": [], "has_more": False, "total": 0}

                documents = doc_result.get("data", [])
                doc_total = doc_result.get("total", 0)
                doc_has_more = doc_result.get("has_more", False)

                if not documents:
                    st.info("该知识库中没有文档。")
                else:
                    # 文档分页控件
                    if doc_total > doc_limit or doc_has_more:
                        doc_page_col1, doc_page_col2, doc_page_col3 = st.columns([1, 2, 3])
                        with doc_page_col1:
                            if st.button("⬅ 上一页", key="kb_doc_prev", disabled=(doc_page <= 1)):
                                st.session_state["kb_doc_page"] = max(1, doc_page - 1)
                                st.rerun()
                        with doc_page_col2:
                            st.caption(f"第 {doc_page} 页（共 {doc_total} 篇文档）")
                        with doc_page_col3:
                            if st.button("下一页 ➡", key="kb_doc_next",
                                         disabled=(not doc_has_more)):
                                st.session_state["kb_doc_page"] = doc_page + 1
                                st.rerun()

                    # 文档表格
                    doc_rows = []
                    for doc in documents:
                        doc_rows.append({
                            "文档ID": doc.get("id", ""),
                            "文档名称": doc.get("name", ""),
                            "词数": doc.get("word_count", 0),
                            "状态": doc.get("status", ""),
                            "创建时间": doc.get("created_at", ""),
                        })
                    st.dataframe(doc_rows, use_container_width=True, key="kb_doc_table")

                    # ── 按文档随机出题（知识库级，不依赖预览文档） ──
                    st.divider()
                    st.markdown("### 按文档随机出题（chunk_exact）")
                    st.caption(
                        "从当前知识库的全部文档中按文档独立采样，生成 chunk_exact 题集。"
                        "不依赖当前预览文档，适用于多文档联合出题场景。"
                    )

                    # 导入所需函数
                    from chunk_exact_questions import (
                        filter_candidate_chunks, generate_chunk_exact_questions_multi_doc,
                        save_chunk_exact_questions, validate_chunk_exact_set,
                        validate_multi_doc_config, sample_candidates_random,
                        generate_default_set_name_for_dataset,
                        get_multi_doc_stats_summary,
                        CHUNK_EXACT_PHASE1_PROMPT, CHUNK_EXACT_PHASE2_PROMPT,
                    )
                    from dify_knowledge import list_all_segments, list_all_documents, build_chunk_catalog as _build_catalog

                    # 缓存 key：按 dataset_id 缓存全库文档+候选统计
                    _ds_stats_key = f"_ce_ds_stats_{selected_ds_id}"

                    # 加载/刷新全库候选统计按钮
                    _stats_loaded = _ds_stats_key in st.session_state
                    _stats_label = "🔄 刷新全库候选统计" if _stats_loaded else "📥 加载全库候选统计"
                    if st.button(_stats_label, key="ce_load_ds_stats"):
                        try:
                            with st.spinner("正在拉取知识库全部文档和候选 chunk 统计..."):
                                # 拉取全部文档（自动分页）
                                all_docs = list_all_documents(
                                    kb_api_key, kb_base_url, selected_ds_id,
                                )
                                # 为每个文档拉取 segments 并统计可用候选数
                                _ds_doc_stats = []
                                _total_candidates = 0
                                for doc in all_docs:
                                    _doc_id = doc.get("id", "")
                                    _doc_name = doc.get("name", "未命名")
                                    if not _doc_id:
                                        continue
                                    try:
                                        all_segs = list_all_segments(
                                            kb_api_key, kb_base_url,
                                            selected_ds_id, _doc_id,
                                        )
                                        doc_catalog = _build_catalog(
                                            all_segs, selected_ds_id,
                                            _doc_id, _doc_name,
                                        )
                                        doc_candidates, _ = filter_candidate_chunks(
                                            doc_catalog
                                        )
                                        _cnt = len(doc_candidates)
                                        _total_candidates += _cnt
                                        _ds_doc_stats.append({
                                            "document_id": _doc_id,
                                            "document_name": _doc_name,
                                            "candidate_count": _cnt,
                                            "status": "ok",
                                            "error": "",
                                        })
                                    except Exception as doc_exc:
                                        _ds_doc_stats.append({
                                            "document_id": _doc_id,
                                            "document_name": _doc_name,
                                            "candidate_count": 0,
                                            "status": "error",
                                            "error": str(doc_exc)[:80],
                                        })
                                st.session_state[_ds_stats_key] = _ds_doc_stats
                                st.success(f"已加载 {len(_ds_doc_stats)} 个文档，{_total_candidates} 个可用候选 chunk")
                                st.rerun()
                        except Exception as exc:
                            st.error(f"加载全库统计失败: {exc}")

                    if _ds_stats_key not in st.session_state:
                        st.info("请点击上方按钮加载知识库全部文档和候选 chunk 统计，以便配置多文档出题。")
                    else:
                        _ds_doc_stats = st.session_state[_ds_stats_key]

                        if not _ds_doc_stats:
                            st.warning("该知识库没有找到任何文档。请检查知识库是否有文档。")
                        else:
                            # 构建出题文档与数量表格
                            st.markdown("##### 出题文档与数量")
                            st.caption("勾选要纳入的文档，设置每文档生成题数。总题数为各文档题数之和。")

                            _doc_table_rows = []
                            for _ds in _ds_doc_stats:
                                _did = _ds["document_id"]
                                _dname = _ds["document_name"]
                                _cnt = _ds["candidate_count"]
                                _status = _ds.get("status", "ok")
                                _error = _ds.get("error", "")
                                _avail_label = str(_cnt)
                                if _status == "error":
                                    _avail_label = f"加载失败: {_error}"
                                _doc_table_rows.append({
                                    "纳入": _status == "ok" and _cnt > 0,
                                    "文档名": _dname,
                                    "文档ID": _did,
                                    "可用chunk数": _cnt,
                                    "状态": _avail_label,
                                    "生成题数": min(_cnt, 5) if _cnt > 0 else 0,
                                })

                            _edited_rows = st.data_editor(
                                _doc_table_rows,
                                column_config={
                                    "纳入": st.column_config.CheckboxColumn(
                                        "纳入", default=True, width="small",
                                    ),
                                    "文档名": st.column_config.TextColumn(
                                        "文档名", disabled=True, width="medium",
                                    ),
                                    "文档ID": st.column_config.TextColumn(
                                        "文档ID", disabled=True, width="small",
                                    ),
                                    "可用chunk数": st.column_config.NumberColumn(
                                        "可用chunk数", disabled=True, width="small",
                                    ),
                                    "状态": st.column_config.TextColumn(
                                        "状态", disabled=True, width="medium",
                                    ),
                                    "生成题数": st.column_config.NumberColumn(
                                        "生成题数", min_value=0, max_value=1000, step=1, width="small",
                                    ),
                                },
                                disabled=["文档名", "文档ID", "可用chunk数", "状态"],
                                hide_index=True,
                                key="ce_doc_table",
                            )

                            # 解析表格结果
                            _active_docs = [
                                r for r in _edited_rows if r["纳入"] and r["生成题数"] > 0
                            ]
                            _total_gen = sum(r["生成题数"] for r in _active_docs)

                            if _active_docs:
                                _summary_parts = [
                                    f"{r['文档名'][:12]}…({r['生成题数']}题)"
                                    for r in _active_docs
                                ]
                                st.caption(f"已选 {len(_active_docs)} 个文档，共 {_total_gen} 题：" + "、".join(_summary_parts))
                            else:
                                st.info("请至少勾选一个文档并设置生成题数 > 0。")

                            # ── 自动出题如何工作 ──
                            with st.expander("📖 自动出题如何工作", expanded=False):
                                st.markdown("""
**两阶段自动流程（无需手动确认）：**

1. **本地抽样** — 按每文档配额从完整候选 chunk 中独立随机抽取，不跨文档抢占，不重复
2. **Phase 1 · 出题规划** — 逐文档调用 LLM，筛选有独立检索价值的 chunk，排除纯标题、无价值、重复片段
3. **Phase 2 · 生成检索查询** — 逐文档调用 LLM，为 Phase 1 保留的 chunk 生成短检索查询与 target_label
4. **本地绑定校验** — fail-closed 校验 candidate_id，从候选 chunk 绑定 segment_id / content_hash / document_id

**安全规则：**
- LLM 不产生 reference_answer，不能修改 gold chunk
- LLM 不输出 segment_id 或 content_hash（由系统自动绑定）
- 单个文档失败只跳过该文档，其余继续
""")
                                _n_docs = len(_active_docs)
                                _n_total_q = _total_gen
                                _n_llm_calls = _n_docs * 2  # 每文档 Phase 1 + Phase 2
                                st.caption(f"本次：{_n_docs} 个文档 · 请求 {_n_total_q} 题 · 预计 {_n_llm_calls} 次 LLM 调用")

                            # ── 查看提示词与本次输入预览 ──
                            with st.expander("🔍 查看提示词与本次输入预览", expanded=False):
                                st.markdown("**Phase 1 Prompt 模板（出题规划）**")
                                st.code(CHUNK_EXACT_PHASE1_PROMPT[:1500] + "\n...", language="text")
                                st.markdown("**Phase 2 Prompt 模板（生成检索查询）**")
                                st.code(CHUNK_EXACT_PHASE2_PROMPT[:1500] + "\n...", language="text")

                                st.markdown("---")
                                st.markdown("**本次输入预览**（只读，不调用 LLM，不写题集）")
                                if st.button("📋 生成只读预览", key="ce_preview_sampling"):
                                    _preview_seed = st.session_state.get("ce_random_seed", 0)
                                    _preview_seed = _preview_seed if _preview_seed > 0 else None
                                    for _prev_r in _active_docs:
                                        _prev_did = _prev_r["文档ID"]
                                        _prev_dname = _prev_r["文档名"]
                                        _prev_num = _prev_r["生成题数"]
                                        _prev_cnt = _prev_r["可用chunk数"]

                                        # 从缓存中获取该文档的候选（用于预览抽样）
                                        _prev_candidates = [
                                            {"segment_id": f"preview_{i}", "content": f"预览内容 {i}"}
                                            for i in range(_prev_cnt)
                                        ]
                                        # 尝试从 session 缓存获取真实候选数
                                        _prev_sampled, _prev_actual, _prev_capped = sample_candidates_random(
                                            _prev_candidates, _prev_num,
                                            seed=hash(_prev_did) % (2**31) if not _preview_seed else _preview_seed,
                                        )

                                        _cap_note = "（候选不足，已截取全部）" if _prev_capped else ""
                                        st.markdown(
                                            f"**{_prev_dname}** — 抽样 {_prev_actual}/{_prev_num} 题{_cap_note}，"
                                            f"预计 2 次 LLM 调用"
                                        )

                                    _prev_total_llm = len(_active_docs) * 2
                                    st.caption(f"预览完成：{len(_active_docs)} 个文档，共 {_total_gen} 题，预计 {_prev_total_llm} 次 LLM 调用")

                            # 出题模型凭据（复用 Judge 配置）
                            ce_api_base = os.getenv("JUDGE_API_BASE", "")
                            ce_model = os.getenv("JUDGE_MODEL", "")
                            _ce_judge_key = os.getenv("JUDGE_API_KEY", "")

                            with st.expander("出题模型凭据（默认复用 Judge 模型）", expanded=False):
                                st.caption("Base URL 和模型与 Judge 保持一致；如需临时使用另一把同服务商 Key，可仅替换 API Key。")
                                st.markdown(f"**Base URL：** `{ce_api_base or '（未配置）'}`")
                                st.markdown(f"**模型：** `{ce_model or '（未配置）'}`")
                                ce_api_key = st.text_input(
                                    "出题 API Key（可选覆盖）",
                                    value="",
                                    type="password",
                                    key="ce_api_key_random",
                                    placeholder="留空则使用 JUDGE_API_KEY",
                                    help="填写时仅覆盖本次出题调用的 API Key，不保存到任何配置文件。",
                                )
                                if not ce_api_key:
                                    ce_api_key = _ce_judge_key

                            # Judge 配置完整性检查
                            _ce_config_ok = bool(ce_api_base and ce_model and ce_api_key)
                            if not ce_api_base or not ce_model:
                                st.error("Judge Base URL 或 Model 未配置。请先在「Judge 评测」tab 配置 Judge API。")

                            ce_col1, ce_col2, ce_col3 = st.columns(3)
                            with ce_col1:
                                # 随机种子
                                ce_random_seed = st.number_input(
                                    "随机种子（可选）",
                                    min_value=0,
                                    value=0,
                                    step=1,
                                    key="ce_random_seed",
                                    help="设为 0 表示随机；正整数可复现",
                                )
                            with ce_col2:
                                # 生成默认名称（使用知识库名称）
                                _ds_name = ds_name_map.get(selected_ds_id, "")
                                _default_name = generate_default_set_name_for_dataset(_ds_name)
                                ce_random_name = st.text_input(
                                    "题集名称",
                                    value=_default_name,
                                    key="ce_random_name",
                                )
                            with ce_col3:
                                ce_max_workers = st.selectbox(
                                    "文档并发数",
                                    options=[1, 2, 3],
                                    index=1,  # 默认 2
                                    key="ce_max_workers",
                                    help="并发越高越快，但越可能遇到模型限流。建议 2。",
                                )

                            # 校验：每个选中文档的题数不超过可用 chunk 数
                            _validation_errors = []
                            for r in _active_docs:
                                if r["生成题数"] > r["可用chunk数"]:
                                    _validation_errors.append(
                                        f"文档「{r['文档名']}」需要 {r['生成题数']} 题，"
                                        f"但仅有 {r['可用chunk数']} 个可用候选 chunk"
                                    )

                            if _validation_errors:
                                for _err in _validation_errors:
                                    st.error(_err)

                            if st.button("🎯 随机生成 chunk_exact 题集", key="ce_random_generate",
                                         disabled=not (_ce_config_ok and _active_docs and not _validation_errors)):
                                _n_docs = len(_active_docs)
                                _total_steps = _n_docs * 2 + 2  # 候选加载 + Phase1 + Phase2 + 校验保存
                                _gen_progress = st.progress(0, text="准备开始...")
                                _gen_status = st.empty()
                                _failed_docs = []

                                try:
                                    # ── 阶段 1/4：候选加载（0-20%）──
                                    doc_configs = []
                                    for _gi, r in enumerate(_active_docs):
                                        _doc_id = r["文档ID"]
                                        _doc_name = r["文档名"]
                                        _num_q = r["生成题数"]
                                        _pct = int((_gi / max(_n_docs, 1)) * 20)
                                        _gen_progress.progress(_pct, text=f"候选加载 {_gi+1}/{_n_docs} — {_doc_name[:12]}…")
                                        _gen_status.caption(f"📥 正在拉取「{_doc_name[:15]}」完整 chunk catalog...")

                                        try:
                                            all_segs = list_all_segments(
                                                kb_api_key, kb_base_url,
                                                selected_ds_id, _doc_id,
                                            )
                                            doc_catalog = _build_catalog(
                                                all_segs, selected_ds_id,
                                                _doc_id, _doc_name,
                                            )
                                            doc_candidates, _ = filter_candidate_chunks(doc_catalog)
                                            doc_configs.append({
                                                "document_id": _doc_id,
                                                "document_name": _doc_name,
                                                "candidates": doc_candidates,
                                                "num_questions": _num_q,
                                            })
                                        except Exception as _load_exc:
                                            _failed_docs.append((_doc_name, "候选加载", str(_load_exc)[:60]))

                                    _gen_progress.progress(20, text=f"候选加载完成 — {len(doc_configs)}/{_n_docs} 文档就绪")

                                    # 校验
                                    ok, errors = validate_multi_doc_config(doc_configs)
                                    if not ok:
                                        for _err in errors:
                                            st.error(_err)
                                    else:
                                        # ── 阶段 2-3/4：Phase 1 + Phase 2（20-90%）──
                                        def _update_progress(done, total, message):
                                            # done/total 是文档级步数（每文档 2 步）
                                            _pct = 20 + int((done / max(total, 1)) * 70)
                                            _gen_progress.progress(min(_pct, 90), text=message)
                                            _gen_status.caption(message)

                                        ce_questions, ce_doc_stats, _actual_seed = generate_chunk_exact_questions_multi_doc(
                                            doc_configs,
                                            ce_api_key, ce_api_base, ce_model,
                                            dataset_id=selected_ds_id,
                                            master_seed=ce_random_seed if ce_random_seed > 0 else 0,
                                            timeout=60,
                                            progress_callback=_update_progress,
                                            max_workers=ce_max_workers,
                                        )

                                        # ── 阶段 4/4：校验保存（90-100%）──
                                        _gen_progress.progress(90, text="本地绑定校验...")
                                        _gen_status.caption("🔍 校验 chunk binding 完整性...")

                                        ce_valid, ce_invalid = validate_chunk_exact_set(ce_questions)
                                        _bind_ok = len(ce_valid)
                                        _bind_fail = len(ce_invalid)

                                        if not ce_valid:
                                            _gen_progress.progress(100, text="生成完成 — 无有效题目")
                                            _gen_status.empty()
                                            st.error("所有题目均绑定不完整，无法保存。")
                                        else:
                                            # 保存
                                            _doc_qc = {r["文档ID"]: r["生成题数"] for r in _active_docs}
                                            _dnm = {ds["document_id"]: ds["document_name"] for ds in _ds_doc_stats}

                                            ce_output_path, ce_filename, ce_set_id = save_chunk_exact_questions(
                                                ce_valid,
                                                question_set_name=ce_random_name or None,
                                                dataset_id=selected_ds_id,
                                                document_id=_active_docs[0]["文档ID"] if len(_active_docs) == 1 else "",
                                                selection_mode="random",
                                                selected_document_ids=[r["文档ID"] for r in _active_docs],
                                                doc_question_counts=_doc_qc,
                                                random_seed=_actual_seed,
                                            )

                                            _gen_progress.progress(100, text="生成完成")
                                            _gen_status.empty()

                                            # ── 成功摘要 ──
                                            _total_requested = sum(s.get("requested", 0) for s in ce_doc_stats)
                                            _total_final_bound = sum(s.get("final_bound", s.get("bound", 0)) for s in ce_doc_stats)
                                            _total_first_rejected = sum(s.get("first_rejected", 0) for s in ce_doc_stats)
                                            _total_binding_failed = sum(s.get("binding_failed", 0) for s in ce_doc_stats)
                                            st.success(
                                                f"chunk_exact 题集已生成！\n\n"
                                                f"- **题集 ID:** `{ce_set_id}`\n"
                                                f"- **请求题数:** {_total_requested} → **最终绑定:** {_total_final_bound} → **绑定通过:** {_bind_ok}\n"
                                                f"- **质量校验拒绝:** {_total_first_rejected}（不等于绑定失败）\n"
                                                f"- **绑定失败:** {_total_binding_failed}\n"
                                                f"- **文件:** `{ce_filename}`\n"
                                                f"- **文档:** {_n_docs} 份"
                                            )

                                            # ── 按文档详细统计 ──
                                            with st.expander("📊 每文档生成详情", expanded=True):
                                                for _s in ce_doc_stats:
                                                    _s_name = _s.get("document_name", "")[:15]
                                                    _s_status = _s.get("status", "")
                                                    _s_req = _s.get("requested", 0)
                                                    _s_pool = _s.get("candidate_pool", 0)
                                                    _s_p1 = _s.get("phase1_planned", 0)
                                                    _s_p2_first = _s.get("phase2_first_returned", _s.get("phase2_generated", 0))
                                                    _s_first_rej = _s.get("first_rejected", 0)
                                                    _s_retry_att = _s.get("retry_attempted", 0)
                                                    _s_retry_rec = _s.get("retry_recovered", 0)
                                                    _s_final = _s.get("final_bound", _s.get("bound", 0))
                                                    _s_bind_fail = _s.get("binding_failed", 0)
                                                    _s_styles = _s.get("query_style_counts", {})
                                                    _style_str = " | ".join(f"{k}:{v}" for k, v in sorted(_s_styles.items())) if _s_styles else ""

                                                    if _s_status == "ok":
                                                        _icon = "✅"
                                                        _detail = (
                                                            f"请求{_s_req}→池{_s_pool}→计划{_s_p1}"
                                                            f"→首轮{_s_p2_first}→校验拒绝{_s_first_rej}"
                                                            f"→重试{_s_retry_att}/恢复{_s_retry_rec}"
                                                            f"→最终绑定{_s_final}"
                                                        )
                                                        if _style_str:
                                                            _detail += f" [{_style_str}]"
                                                    elif _s_status == "underfilled":
                                                        _icon = "⚠️"
                                                        _detail = (
                                                            f"请求{_s_req}→池{_s_pool}→计划{_s_p1}"
                                                            f"→首轮{_s_p2_first}→校验拒绝{_s_first_rej}"
                                                            f"→重试{_s_retry_att}/恢复{_s_retry_rec}"
                                                            f"→最终绑定{_s_final}"
                                                        )
                                                        if _style_str:
                                                            _detail += f" [{_style_str}]"
                                                    elif _s_status == "insufficient_candidates":
                                                        _icon = "❌"
                                                        _err = (_s.get("errors") or ["未知"])[0][:50]
                                                        _detail = f"候选不足 — {_err}"
                                                    elif _s_status == "phase1_failed":
                                                        _icon = "❌"
                                                        _err = (_s.get("errors") or ["未知"])[0][:50]
                                                        _detail = f"Phase1 失败 — {_err}"
                                                    elif _s_status == "phase1_empty":
                                                        _icon = "⚠️"
                                                        _detail = "Phase1 未返回有效规划"
                                                    elif _s_status == "phase2_failed":
                                                        _icon = "❌"
                                                        _err = (_s.get("errors") or ["未知"])[0][:50]
                                                        _detail = f"Phase2 失败 — {_err}"
                                                    else:
                                                        _icon = "⚠️"
                                                        _detail = _s_status

                                                    st.markdown(f"{_icon} **{_s_name}** — {_detail}")

                                                # 拒绝诊断
                                                _all_diag = []
                                                for _s in ce_doc_stats:
                                                    for _d in _s.get("rejection_diagnostics", []):
                                                        _d["doc"] = _s.get("document_name", "")[:10]
                                                        _all_diag.append(_d)
                                                if _all_diag:
                                                    with st.expander(f"🔍 校验拒绝详情（{len(_all_diag)} 条）", expanded=False):
                                                        for _d in _all_diag[:20]:
                                                            _retry_tag = " [重试]" if _d.get("retry") else ""
                                                            st.markdown(
                                                                f"- **{_d.get('doc', '')}** / `{_d.get('candidate_id', '')[:12]}`{_retry_tag}: "
                                                                f"查询=`{_d.get('query', '')[:30]}` — {'; '.join(_d.get('errors', [])[:2])}"
                                                            )

                                            # 显示实际种子
                                            st.caption(f"🎲 实际随机种子：{_actual_seed}（可用于复现本次出题）")

                                            # 题目预览
                                            with st.expander("题目预览", expanded=False):
                                                for i, q in enumerate(ce_valid):
                                                    seg_id = q.get("expected_segment_id", "")
                                                    seg_short = seg_id[:12] + "..." if len(str(seg_id)) > 12 else seg_id
                                                    pos = q.get("source_position", "")
                                                    doc_id = q.get("document_id", "")
                                                    doc_label = _dnm.get(doc_id, doc_id[:8] + "...")
                                                    _qstyle = q.get("query_style", "")
                                                    _style_badge = f" `{_qstyle}`" if _qstyle else ""
                                                    _intent = q.get("retrieval_intent", "")
                                                    _intent_line = f"\n  ↳ 检索意图: {_intent}" if _intent else ""
                                                    _fact = q.get("target_fact", "")
                                                    _fact_line = f"\n  ↳ 证据锚点: {_fact}" if _fact else ""
                                                    st.markdown(
                                                        f"**{i+1}.** {q.get('retrieval_query', '')} "
                                                        f"(`{q.get('target_label', '')}`{_style_badge}) "
                                                        f"→ [{doc_label}] segment: `{seg_short}` pos:{pos}"
                                                        f"{_intent_line}{_fact_line}"
                                                    )

                                except Exception as exc:
                                    _gen_progress.progress(100, text="生成失败")
                                    _gen_status.empty()
                                    st.error(f"生成失败: {exc}")

                                finally:
                                    # 清理 progress 占位符（保留错误/成功消息）
                                    pass

                    # ── 文档选择与分块浏览 ────────────────────────────
                    st.markdown("### 分块浏览")

                    doc_id_options = []
                    doc_name_map = {}  # doc_id -> doc_name
                    for doc in documents:
                        doc_id = doc.get("id", "")
                        doc_name = doc.get("name", "未命名")
                        doc_name_map[doc_id] = doc_name
                        doc_id_options.append((doc_id, f"{doc_name}（{doc_id[:8]}...）"))

                    selected_doc_id = st.selectbox(
                        "选择文档",
                        options=[c[0] for c in doc_id_options],
                        format_func=lambda x: next(
                            (c[1] for c in doc_id_options if c[0] == x), x
                        ),
                        key="kb_selected_document",
                    )

                    if selected_doc_id:
                        # 状态过滤与分页设置
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            seg_status = st.radio(
                                "状态过滤",
                                ["completed", "indexing", "error", "全部"],
                                horizontal=True,
                                key="kb_seg_status",
                                help="默认仅显示已完成的分块",
                            )
                            status_filter = "" if seg_status == "全部" else seg_status

                        with filter_col2:
                            seg_limit = st.number_input(
                                "每页数量",
                                min_value=1,
                                max_value=100,
                                value=20,
                                step=5,
                                key="kb_seg_limit",
                                help="最大 100",
                            )

                        seg_page = st.session_state.get("kb_seg_page", 1)

                        try:
                            with st.spinner("正在加载分块列表..."):
                                seg_result = list_segments(
                                    kb_api_key, kb_base_url, selected_ds_id, selected_doc_id,
                                    page=seg_page, limit=seg_limit,
                                    status_filter=status_filter,
                                )
                        except RuntimeError as exc:
                            st.error(f"获取分块列表失败: {exc}")
                            seg_result = {"data": [], "has_more": False, "total": 0}

                        segments = seg_result.get("data", [])
                        seg_total = seg_result.get("total", 0)
                        seg_has_more = seg_result.get("has_more", False)

                        if not segments:
                            st.info("未找到符合条件的分块。")
                        else:
                            # 分块分页控件
                            if seg_total > seg_limit or seg_has_more:
                                seg_page_col1, seg_page_col2, seg_page_col3 = st.columns([1, 2, 3])
                                with seg_page_col1:
                                    if st.button("⬅ 上一页", key="kb_seg_prev", disabled=(seg_page <= 1)):
                                        st.session_state["kb_seg_page"] = max(1, seg_page - 1)
                                        st.rerun()
                                with seg_page_col2:
                                    st.caption(f"第 {seg_page} 页（共 {seg_total} 个分块）")
                                with seg_page_col3:
                                    if st.button("下一页 ➡", key="kb_seg_next",
                                                 disabled=(not seg_has_more)):
                                        st.session_state["kb_seg_page"] = seg_page + 1
                                        st.rerun()

                            # 构建 catalog
                            selected_doc_name = doc_name_map.get(selected_doc_id, "")
                            catalog = build_chunk_catalog(segments, selected_ds_id, selected_doc_id, selected_doc_name)

                            # 重复检测
                            duplicates = detect_duplicates(catalog)
                            if duplicates:
                                dup_count = sum(len(v) for v in duplicates.values())
                                dup_hash_count = len(duplicates)
                                st.warning(
                                    f"检测到 {dup_hash_count} 组重复内容（共 {dup_count} 个分块）"
                                )
                                dup_hashes = set(duplicates.keys())
                            else:
                                dup_hashes = set()
                                st.success("当前页未检测到重复分块。")

                            # 分块表格
                            seg_rows = []
                            for entry in catalog:
                                content_preview = entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry["content"]
                                is_dup = entry["content_hash"] in dup_hashes
                                seg_rows.append({
                                    "重复": "⚠️ 是" if is_dup else "",
                                    "segment_id": entry["segment_id"],
                                    "position": entry["position"],
                                    "document_id": entry["document_id"][:12] + "..." if len(str(entry["document_id"])) > 12 else entry["document_id"],
                                    "content": content_preview,
                                    "index_node_id": entry["index_node_id"][:12] + "..." if len(str(entry["index_node_id"])) > 12 else entry["index_node_id"],
                                    "index_node_hash": entry["index_node_hash"][:12] + "..." if len(str(entry["index_node_hash"])) > 12 else entry["index_node_hash"],
                                    "tokens": entry["tokens"],
                                    "word_count": entry["word_count"],
                                    "enabled": entry["enabled"],
                                    "status": entry["status"],
                                    "content_hash": entry["content_hash"][:16] + "..." if entry["content_hash"] else "",
                                })

                            st.dataframe(seg_rows, use_container_width=True, key="kb_seg_table")

                            # 展开查看完整内容
                            if catalog:
                                with st.expander("查看分块完整内容", expanded=False):
                                    for entry in catalog:
                                        is_dup = entry["content_hash"] in dup_hashes
                                        title = f"**{entry['segment_id'][:12]}...**"
                                        if is_dup:
                                            title += " ⚠️ 重复"
                                        with st.expander(title, expanded=False):
                                            st.text_area(
                                                "内容",
                                                value=entry["content"],
                                                height=150,
                                                disabled=True,
                                                key=f"kb_seg_content_{entry['segment_id']}",
                                            )
                                            st.caption(
                                                f"tokens: {entry['tokens']} | "
                                                f"word_count: {entry['word_count']} | "
                                                f"status: {entry['status']} | "
                                                f"enabled: {entry['enabled']} | "
                                                f"content_hash: `{entry['content_hash'][:16]}...`"
                                            )

                            # ── 导出当前文档 ──────────────────────────────────
                            st.markdown("### 导出 Chunk Catalog")

                            export_col1, export_col2 = st.columns(2)

                            with export_col1:
                                json_bytes = export_catalog_json(catalog).encode("utf-8")
                                st.download_button(
                                    label="📥 导出当前文档 JSON",
                                    data=json_bytes,
                                    file_name=f"chunk_catalog_{selected_doc_id[:8]}.json",
                                    mime="application/json",
                                    key="kb_export_json",
                                )

                            with export_col2:
                                csv_bytes = export_catalog_csv(catalog)
                                st.download_button(
                                    label="📥 导出当前文档 CSV",
                                    data=csv_bytes,
                                    file_name=f"chunk_catalog_{selected_doc_id[:8]}.csv",
                                    mime="text/csv",
                                    key="kb_export_csv",
                                )

                            st.caption(
                                f"当前文档导出包含 {len(catalog)} 条分块记录，"
                                f"字段：segment_id, position, document_id, document_name, content, summary, "
                                f"index_node_id, index_node_hash, tokens, word_count, "
                                f"enabled, status, content_hash"
                            )

                            # ── 导出全知识库 ────────────────────────────────
                            st.divider()
                            st.markdown("### 导出全知识库 Chunk Catalog")
                            st.caption(
                                "导出当前知识库全部文档的 chunk catalog，不依赖上方「分块预览」选中的文档。"
                            )

                            # 展示知识库信息
                            _ds_name_for_export = ds_name_map.get(selected_ds_id, "未知")
                            _all_docs_for_export = st.session_state.get("_kb_all_docs", [])
                            _total_docs_for_export = len(_all_docs_for_export)

                            st.markdown(
                                f"**知识库：** {_ds_name_for_export} | "
                                f"**dataset_id：** `{selected_ds_id[:16]}...` | "
                                f"**文档总数：** {_total_docs_for_export}"
                            )

                            _fk_export_col1, _fk_export_col2 = st.columns(2)

                            with _fk_export_col1:
                                _fk_json_clicked = st.button(
                                    "📥 导出全知识库 JSON",
                                    key="kb_export_full_json_btn",
                                    disabled=(not _total_docs_for_export),
                                )

                            with _fk_export_col2:
                                _fk_csv_clicked = st.button(
                                    "📥 导出全知识库 CSV",
                                    key="kb_export_full_csv_btn",
                                    disabled=(not _total_docs_for_export),
                                )

                            # 处理全知识库导出
                            if _fk_json_clicked or _fk_csv_clicked:
                                from dify_knowledge import (
                                    build_full_kb_catalog,
                                    export_full_kb_json,
                                    export_full_kb_csv,
                                )

                                _fk_progress_bar = st.progress(0, text="准备导出...")
                                _fk_status_text = st.empty()
                                _fk_total_chunks = [0]

                                def _fk_progress_cb(cur, total, doc_name, chunk_count):
                                    _fk_total_chunks[0] += chunk_count
                                    pct = int(cur / max(total, 1) * 100)
                                    _fk_progress_bar.progress(
                                        pct,
                                        text=f"正在读取第 {cur}/{total} 个文档",
                                    )
                                    _fk_status_text.caption(
                                        f"📄 {doc_name[:30]}… | 已累计 {_fk_total_chunks[0]} 个 chunk"
                                    )

                                try:
                                    _fk_result = build_full_kb_catalog(
                                        kb_api_key, kb_base_url,
                                        selected_ds_id, _ds_name_for_export,
                                        progress_callback=_fk_progress_cb,
                                    )
                                    _fk_progress_bar.progress(100, text="导出完成")
                                    _fk_status_text.empty()

                                    _fk_meta = _fk_result["metadata"]
                                    _fk_stats = _fk_result["doc_stats"]
                                    _fk_catalog = _fk_result["catalog"]

                                    # 显示导出摘要
                                    _fk_ok = sum(1 for s in _fk_stats if s["status"] == "ok")
                                    _fk_skip = sum(1 for s in _fk_stats if s["status"] == "skipped")
                                    _fk_err = sum(1 for s in _fk_stats if s["status"] == "error")
                                    _fk_total_chunks = _fk_meta["total_chunks"]

                                    if _fk_ok == 0:
                                        st.warning(
                                            f"⚠️ 无文档成功导出：**{_fk_skip}** 个跳过 / "
                                            f"**{_fk_err}** 个失败 / 共 **{_fk_total_chunks}** 个 chunk"
                                        )
                                    else:
                                        st.success(
                                            f"导出完成：**{_fk_ok}** 个文档成功 / "
                                            f"**{_fk_skip}** 个跳过 / "
                                            f"**{_fk_err}** 个失败 / "
                                            f"共 **{_fk_total_chunks}** 个 chunk"
                                        )

                                    # 逐份显示跳过文档详情（含 API 字段摘要）
                                    _fk_skipped = [s for s in _fk_stats if s["status"] == "skipped"]
                                    if _fk_skipped:
                                        with st.expander(f"⏭️ 跳过文档详情（{len(_fk_skipped)} 个）", expanded=(_fk_ok == 0)):
                                            for _fs in _fk_skipped:
                                                _api = _fs.get("api_fields", {})
                                                _api_summary = ", ".join(
                                                    f"{k}={v}" for k, v in _api.items()
                                                    if v not in (None, "", True, False)
                                                ) or "（无额外字段）"
                                                st.markdown(
                                                    f"- **{_fs['document_name']}** "
                                                    f"(`{_fs['document_id'][:12]}...`)\n"
                                                    f"  跳过原因: {_fs['reason']}\n"
                                                    f"  API 字段: {_api_summary}"
                                                )

                                    # 显示失败文档详情
                                    _fk_failed = [s for s in _fk_stats if s["status"] == "error"]
                                    if _fk_failed:
                                        with st.expander(f"❌ 失败文档详情（{len(_fk_failed)} 个）", expanded=False):
                                            for _fs in _fk_failed:
                                                st.markdown(
                                                    f"- **{_fs['document_name']}** (`{_fs['document_id'][:12]}...`): "
                                                    f"{_fs['reason']}"
                                                )

                                    # 0 chunk 的成功文档提示
                                    _fk_zero_chunk_ok = [
                                        s for s in _fk_stats
                                        if s["status"] == "ok" and s["chunk_count"] == 0
                                    ]
                                    if _fk_zero_chunk_ok:
                                        with st.expander(f"ℹ️ 成功但 0 chunks 的文档（{len(_fk_zero_chunk_ok)} 个）", expanded=False):
                                            for _fs in _fk_zero_chunk_ok:
                                                st.markdown(
                                                    f"- **{_fs['document_name']}** "
                                                    f"(`{_fs['document_id'][:12]}...`): "
                                                    f"{_fs['reason'] or 'chunks API 返回空列表'}"
                                                )

                                    # 生成下载（total_chunks == 0 时禁用）
                                    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    _ds_short = selected_ds_id[:8]
                                    _fk_dl_disabled = (_fk_total_chunks == 0)

                                    if _fk_json_clicked:
                                        _fk_json_str = export_full_kb_json(_fk_catalog, _fk_meta)
                                        _fk_json_bytes = _fk_json_str.encode("utf-8")
                                        st.download_button(
                                            label=f"⬇️ 下载全知识库 JSON（{_fk_total_chunks} chunks）",
                                            data=_fk_json_bytes,
                                            file_name=f"chunk_catalog_{_ds_short}_all_{_ts}.json",
                                            mime="application/json",
                                            key="kb_export_full_json_dl",
                                            disabled=_fk_dl_disabled,
                                        )

                                    if _fk_csv_clicked:
                                        _fk_csv_bytes = export_full_kb_csv(_fk_catalog)
                                        st.download_button(
                                            label=f"⬇️ 下载全知识库 CSV（{_fk_total_chunks} chunks）",
                                            data=_fk_csv_bytes,
                                            file_name=f"chunk_catalog_{_ds_short}_all_{_ts}.csv",
                                            mime="text/csv",
                                            key="kb_export_full_csv_dl",
                                            disabled=_fk_dl_disabled,
                                        )

                                    # 缓存结果供出题模块使用
                                    st.session_state["_fk_last_catalog"] = _fk_catalog
                                    st.session_state["_fk_last_metadata"] = _fk_meta
                                    st.session_state["_fk_last_stats"] = _fk_stats

                                except Exception as exc:
                                    _fk_progress_bar.progress(100, text="导出失败")
                                    _fk_status_text.empty()
                                    st.error(f"全知识库导出失败: {exc}")

                            # ── 基于当前预览文档手动创建题集 ──
                            st.divider()
                            st.markdown("### 基于当前预览文档手动创建题集")
                            st.caption(
                                "仅使用「分块浏览」当前选中文档、本页已加载的候选 chunk。"
                                "适合针对关键条款、异常切分或特定内容做专项检索诊断；"
                                "如需跨多个文档自动出题，请使用上方「按文档随机」模式。"
                            )

                            # 显示当前文档上下文
                            _cur_doc_name = doc_name_map.get(selected_doc_id, selected_doc_id[:12])
                            _cur_seg_total = seg_total if 'seg_total' in dir() else 0
                            _cur_seg_page = seg_page if 'seg_page' in dir() else 1
                            st.markdown(
                                f"**当前文档：** `{_cur_doc_name}` | "
                                f"**页码：** 第 {_cur_seg_page} 页 | "
                                f"**本页分块数：** {len(segments)} | "
                                f"**文档总分块数：** {_cur_seg_total}"
                            )

                            # 过滤候选 chunk
                            from chunk_exact_questions import (
                                filter_candidate_chunks, generate_chunk_exact_questions,
                                save_chunk_exact_questions, validate_chunk_exact_set,
                            )

                            ce_candidates, ce_filter_stats = filter_candidate_chunks(catalog, duplicates)

                            st.markdown(
                                f"**候选 chunk：** {ce_filter_stats['passed']} / {ce_filter_stats['total']} 通过过滤"
                            )
                            if ce_filter_stats["filtered"]:
                                filter_desc = "、".join(f"{reason}: {count}" for reason, count in ce_filter_stats["filtered"].items())
                                st.caption(f"过滤原因: {filter_desc}")

                            if not ce_candidates:
                                st.warning("当前预览文档没有符合条件的候选 chunk。请切换到有更多 completed+enabled 分块的文档。")
                            else:
                                # 出题模型凭据（复用 Judge 配置）
                                ce_api_base = os.getenv("JUDGE_API_BASE", "")
                                ce_model = os.getenv("JUDGE_MODEL", "")
                                _ce_judge_key = os.getenv("JUDGE_API_KEY", "")

                                with st.expander("出题模型凭据（默认复用 Judge 模型）", expanded=False):
                                    st.caption("Base URL 和模型与 Judge 保持一致；如需临时使用另一把同服务商 Key，可仅替换 API Key。")
                                    st.markdown(f"**Base URL：** `{ce_api_base or '（未配置）'}`")
                                    st.markdown(f"**模型：** `{ce_model or '（未配置）'}`")
                                    ce_api_key = st.text_input(
                                        "出题 API Key（可选覆盖）",
                                        value="",
                                        type="password",
                                        key="ce_api_key",
                                        placeholder="留空则使用 JUDGE_API_KEY",
                                        help="填写时仅覆盖本次出题调用的 API Key，不保存到任何配置文件。",
                                    )
                                    if not ce_api_key:
                                        ce_api_key = _ce_judge_key

                                # Judge 配置完整性检查
                                _ce_config_ok = bool(ce_api_base and ce_model and ce_api_key)
                                if not ce_api_base or not ce_model:
                                    st.error("Judge Base URL 或 Model 未配置。请先在「Judge 评测」tab 配置 Judge API。")

                                # ── 手动模式：从当前预览文档选择 chunk ──
                                    with st.expander("查看和调整候选 chunk", expanded=False):
                                        ce_exclude_ids = set()
                                        for c in ce_candidates:
                                            label = f"{c['segment_id'][:16]}... | {c['content'][:60]}..."
                                            if st.checkbox(label, value=True, key=f"ce_cand_{c['segment_id']}"):
                                                pass
                                            else:
                                                ce_exclude_ids.add(c["segment_id"])

                                        ce_final_candidates = [c for c in ce_candidates if c["segment_id"] not in ce_exclude_ids]
                                        st.caption(f"已选择 {len(ce_final_candidates)} 个候选 chunk")

                                    if not ce_final_candidates:
                                        st.info("请至少选择一个候选 chunk。")
                                    else:
                                        ce_col1, ce_col2 = st.columns(2)
                                        with ce_col1:
                                            ce_num_questions = st.number_input(
                                                "生成数量",
                                                min_value=1,
                                                max_value=len(ce_final_candidates),
                                                value=min(len(ce_final_candidates), 10),
                                                step=1,
                                                key="ce_num_questions",
                                            )
                                        with ce_col2:
                                            ce_set_name = st.text_input(
                                                "题集名称",
                                                value="",
                                                key="ce_set_name",
                                                placeholder="留空自动生成",
                                            )

                                        if st.button("🎯 生成 chunk_exact 题集", key="ce_generate",
                                                     disabled=not _ce_config_ok):
                                            try:
                                                with st.spinner("正在生成 chunk_exact 题集..."):
                                                    ce_questions = generate_chunk_exact_questions(
                                                        ce_final_candidates,
                                                        ce_api_key, ce_api_base, ce_model,
                                                        num_questions=ce_num_questions,
                                                        dataset_id=selected_ds_id,
                                                        document_id=selected_doc_id,
                                                        timeout=60,
                                                    )

                                                ce_valid, ce_invalid = validate_chunk_exact_set(ce_questions)
                                                if ce_invalid:
                                                    st.warning(f"{len(ce_invalid)} 道题绑定不完整，已剔除。剩余 {len(ce_valid)} 道。")
                                                    ce_questions = ce_valid

                                                if not ce_questions:
                                                    st.error("所有题目均绑定不完整，无法保存。")
                                                else:
                                                    ce_output_path, ce_filename, ce_set_id = save_chunk_exact_questions(
                                                        ce_questions,
                                                        question_set_name=ce_set_name or None,
                                                        dataset_id=selected_ds_id,
                                                        document_id=selected_doc_id,
                                                        selection_mode="manual",
                                                    )

                                                    st.success(
                                                        f"chunk_exact 题集已生成！\n\n"
                                                        f"- **题集 ID:** `{ce_set_id}`\n"
                                                        f"- **题目数量:** {len(ce_questions)}\n"
                                                        f"- **文件:** `{ce_filename}`\n"
                                                        f"- **知识库:** `{selected_ds_id[:12]}...`\n"
                                                        f"- **文档:** `{selected_doc_id[:12]}...`"
                                                    )

                                                    with st.expander("题目预览", expanded=False):
                                                        for i, q in enumerate(ce_questions):
                                                            seg_id = q.get("expected_segment_id", "")
                                                            seg_short = seg_id[:12] + "..." if len(str(seg_id)) > 12 else seg_id
                                                            pos = q.get("source_position", "")
                                                            _intent = q.get("retrieval_intent", "")
                                                            _intent_line = f"\n  ↳ 检索意图: {_intent}" if _intent else ""
                                                            _fact = q.get("target_fact", "")
                                                            _fact_line = f"\n  ↳ 证据锚点: {_fact}" if _fact else ""
                                                            st.markdown(
                                                                f"**{i+1}.** {q.get('retrieval_query', '')} "
                                                                f"(`{q.get('target_label', '')}`) "
                                                                f"→ segment: `{seg_short}` pos:{pos}"
                                                                f"{_intent_line}{_fact_line}"
                                                            )

                                            except Exception as exc:
                                                st.error(f"生成失败: {exc}")

                                    # ── 检索诊断 ──────────────────────────────────────────────
                                    st.divider()
                                    st.markdown("### 检索诊断")
                                    st.caption(
                                        "对指定知识库执行语义检索，查看返回 chunk 排名和命中情况。"
                                        "所有结果标为「诊断结果」，不写入正式 run/judge 指标。"
                                    )

                                    with st.expander("检索诊断说明（点击展开）", expanded=False):
                                        st.markdown("""
                        **用途：** 快速验证知识库检索质量，确认关键 chunk 能否被正确召回。

                        **操作流程：**
                        1. 输入一条短查询（模拟用户提问）
                        2. 设置 TopK（返回前 K 条结果）
                        3. 点击「运行诊断」查看排名和分数
                        4. 可选：输入一个预期 segment_id，直观查看 Top1/Top3/Top5 是否命中

                        **重要提示：**
                        - 诊断结果**仅供参考**，不代表正式评测结果
                        - 正式评测必须使用 Dify 工作流真实调用和真实 Langfuse trace
                        - 本功能仅调用 Dataset API 的 retrieve 能力，不会修改知识库内容
                        """)

                                    diag_col1, diag_col2 = st.columns([3, 1])
                                    with diag_col1:
                                        diag_query = st.text_input(
                                            "检索查询",
                                            placeholder="例如：如何申请退款？",
                                            key="kb_diag_query",
                                            help="输入一条短查询，模拟用户提问",
                                        )
                                    with diag_col2:
                                        diag_topk = st.number_input(
                                            "TopK",
                                            min_value=1,
                                            max_value=20,
                                            value=5,
                                            step=1,
                                            key="kb_diag_topk",
                                        )

                                    # 预期 segment_id（可选）
                                    diag_expected = st.text_input(
                                        "预期 segment_id（可选）",
                                        placeholder="输入期望命中的 segment_id，用于检查 Top1/Top3/Top5 命中",
                                        key="kb_diag_expected",
                                        help="留空则不检查命中。可从上方分块浏览中复制 segment_id。",
                                    )

                                    if st.button("🔍 运行诊断", key="kb_diag_run", disabled=not diag_query.strip()):
                                        try:
                                            with st.spinner("正在检索..."):
                                                diag_records = retrieve(
                                                    kb_api_key, kb_base_url, selected_ds_id,
                                                    diag_query.strip(), top_k=diag_topk,
                                                )
                                        except RuntimeError as exc:
                                            st.error(f"检索失败: {exc}")
                                            diag_records = []

                                        if diag_records:
                                            # 构建结果表格
                                            diag_rows = []
                                            hit_top1 = hit_top3 = hit_top5 = False
                                            expected_id = diag_expected.strip()

                                            for rec in diag_records:
                                                rank = rec["position"]
                                                seg_id = rec["segment_id"]
                                                is_expected = (expected_id and seg_id == expected_id)

                                                if is_expected:
                                                    if rank <= 1:
                                                        hit_top1 = True
                                                    if rank <= 3:
                                                        hit_top3 = True
                                                    if rank <= 5:
                                                        hit_top5 = True

                                                content_preview = rec["content"][:80] + "..." if len(rec["content"]) > 80 else rec["content"]
                                                diag_rows.append({
                                                    "排名": rank,
                                                    "命中": "✅ 预期" if is_expected else "",
                                                    "segment_id": seg_id[:16] + "..." if len(str(seg_id)) > 16 else seg_id,
                                                    "document_id": rec["document_id"][:12] + "..." if len(str(rec["document_id"])) > 12 else rec["document_id"],
                                                    "score": f"{rec['score']:.4f}" if isinstance(rec["score"], (int, float)) else str(rec["score"]),
                                                    "content": content_preview,
                                                    "enabled": rec["enabled"],
                                                    "status": rec["status"],
                                                })

                                            st.dataframe(diag_rows, use_container_width=True, key="kb_diag_table")

                                            # 命中指示
                                            if expected_id:
                                                st.markdown("**命中结果：**")
                                                hit_c1, hit_c2, hit_c3 = st.columns(3)
                                                with hit_c1:
                                                    if hit_top1:
                                                        st.success("✅ Top1 命中")
                                                    else:
                                                        st.error("❌ Top1 未命中")
                                                with hit_c2:
                                                    if hit_top3:
                                                        st.success("✅ Top3 命中")
                                                    else:
                                                        st.error("❌ Top3 未命中")
                                                with hit_c3:
                                                    if hit_top5:
                                                        st.success("✅ Top5 命中")
                                                    else:
                                                        st.error("❌ Top5 未命中")

                                                found_rank = None
                                                for rec in diag_records:
                                                    if rec["segment_id"] == expected_id:
                                                        found_rank = rec["position"]
                                                        break
                                                if found_rank:
                                                    st.info(f"预期分块在第 **{found_rank}** 位被召回")
                                                else:
                                                    st.warning(f"预期分块 `{expected_id[:16]}...` 未在 Top{diag_topk} 中出现")

                                            st.caption("⚠️ 以上为诊断结果，仅供参考。正式评测请使用 Dify 工作流真实调用。")
                                        else:
                                            st.info("未返回任何检索结果。请检查查询内容和知识库是否有数据。")

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
| 知识库文件 | .txt / .md / .docx / .xlsx 格式的知识库文档 |
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
        qgen_uploaded = st.file_uploader("上传知识库文件", type=["txt", "md", "docx", "xlsx", "xls", "csv"], key="qgen_upload")

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

生成短检索查询，测试 RAG 系统能否从知识库召回包含正确原文证据的 chunk（Top1/Top3/Top5 命中率）。

**查询特点：**
- ✅ 短检索查询：词、词组、短语或单一检索意图（非问句）
- ✅ 金标准证据：从当前 chunk 逐字复制的原文片段
- ✅ 优先生成：定义类、枚举类、单点事实类查询
- ❌ 明确禁止：问句、对比类、分析类、多子问题查询

**评测目标：** 验证检索是否命中正确的 chunk，而非测试 LLM 问答质量""")
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
        from doc_parser import parse_document, format_parse_summary, is_supported_file

        file_bytes = qgen_uploaded.getvalue()
        file_name = qgen_uploaded.name

        # 检查是否为 Word 临时文件
        if file_name.startswith("~$"):
            st.warning(f"已跳过 Word 临时文件: {file_name}")
            st.stop()

        # 统一解析
        try:
            parse_result = parse_document(file_bytes=file_bytes, file_name=file_name)
            file_content = parse_result["text"]
        except ValueError as e:
            st.error(f"文件解析失败: {e}")
            st.stop()

        file_size_kb = len(file_bytes) / 1024
        char_count = len(file_content)

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("文件名", file_name)
        info_col2.metric("文件大小", f"{file_size_kb:.1f} KB")
        info_col3.metric("字符数", f"{char_count:,}")

        # 电子表格格式额外指标
        if parse_result.get("source_type") in ("xlsx", "xls", "csv"):
            _summary = parse_result.get("summary", {})
            _ex1, _ex2, _ex3 = st.columns(3)
            _ex1.metric("工作表数", _summary.get("sheet_count", "?"))
            _ex2.metric("数据行数", _summary.get("row_count", "?"))
            _ex3.metric("表格数", _summary.get("table_count", "?"))

        # 解析摘要
        _parse_summary = format_parse_summary(parse_result)
        st.caption(_parse_summary)

        # 解析警告
        _warnings = parse_result.get("warnings", [])
        if _warnings:
            with st.expander(f"解析警告（{len(_warnings)} 条）", expanded=False):
                for w in _warnings:
                    st.warning(w)

        # ── 电子表格检索模式：本地解析 + 预览（不调用 LLM）──
        _is_spreadsheet_preview = (
            parse_result.get("source_type") in ("xlsx", "xls", "csv")
            and mode_val == MODE_RETRIEVAL
        )
        if _is_spreadsheet_preview:
            try:
                from spreadsheet_question_generator import (
                    parse_xlsx_to_sheet_contexts,
                    parse_csv_to_sheet_contexts,
                    parse_xls_to_sheet_contexts,
                    analyze_table_schema,
                    generate_questions_from_schema,
                    _col_letter,
                )

                _ext = Path(file_name).suffix.lower()
                if _ext == ".xlsx":
                    _schema_sheets = parse_xlsx_to_sheet_contexts(file_bytes)
                elif _ext == ".xls":
                    _schema_sheets = parse_xls_to_sheet_contexts(file_bytes)
                else:
                    _schema_sheets = parse_csv_to_sheet_contexts(file_bytes, file_name)

                # 检测文件是否更换（清除旧 schema）
                _prev_file = st.session_state.get("_schema_file_name")
                if _prev_file and _prev_file != file_name:
                    st.session_state.pop("_schema_analysis", None)
                    st.session_state.pop("_schema_analysis_done", None)
                    st.session_state.pop("_schema_llm_calls", None)

                st.session_state["_schema_sheets"] = _schema_sheets
                st.session_state["_schema_file_name"] = file_name
                st.session_state["_schema_file_bytes"] = file_bytes

                # 本地预览摘要
                _total_rows = sum(s.max_row for s in _schema_sheets)
                _total_cols = sum(s.max_col for s in _schema_sheets)
                _merged = sum(len(s.merged_cells) for s in _schema_sheets)
                _formula_warn = sum(len(s.formula_cells_without_cache) for s in _schema_sheets)

                _col1, _col2, _col3, _col4 = st.columns(4)
                _col1.metric("工作表", len(_schema_sheets))
                _col2.metric("总行数", _total_rows)
                _col3.metric("合并单元格", _merged)
                _col4.metric("公式警告", _formula_warn)
                st.caption("本地解析完成，未调用 LLM")

                # 汇总行/风险提示
                from spreadsheet_question_generator import _is_summary_row
                _summary_rows = []
                for s in _schema_sheets:
                    for r in range(len(s.rows)):
                        if _is_summary_row(s.rows[r]):
                            _summary_rows.append(f"{s.sheet_name} 第 {r + 1} 行")
                if _summary_rows:
                    st.warning(f"检测到汇总/总计行（将自动排除）: {', '.join(_summary_rows[:5])}" +
                               (f" 等 {len(_summary_rows)} 行" if len(_summary_rows) > 5 else ""))

                if _formula_warn:
                    st.warning(f"有 {_formula_warn} 个公式单元格缺少缓存值，出题时将自动规避")

                # ── Phase 1 按钮 ──
                _schema_done = st.session_state.get("_schema_analysis_done", False)

                if not _schema_done:
                    if st.button("分析表格结构（调用 LLM）", type="primary", key="btn_analyze_schema", use_container_width=True):
                        if not qgen_api_key:
                            st.error("请在上方「API 配置」中输入 API Key")
                        else:
                            with st.status("Phase 1: 正在理解表格结构...", expanded=True):
                                try:
                                    _analysis = analyze_table_schema(
                                        _schema_sheets,
                                        qgen_api_key, qgen_base_url, qgen_model,
                                        file_bytes=file_bytes,
                                    )
                                    _attempts = _analysis.get("llm_call_attempts", 1)
                                    _durations = _analysis.get("llm_attempt_durations", [])
                                    _total_dur = _analysis.get("llm_call_duration_sec", 0)
                                    st.session_state["_schema_analysis"] = _analysis
                                    st.session_state["_schema_analysis_done"] = True
                                    st.session_state["_schema_llm_calls"] = _attempts
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Phase 1 分析失败（已重试 {_attempts if '_attempts' in dir() else 1} 次）: {e}")
                else:
                    # Phase 1 已完成：展示摘要
                    _analysis = st.session_state.get("_schema_analysis", {})
                    _sf = _analysis.get("safe_question_fields", [])
                    _excl = _analysis.get("excluded_rows", [])
                    _record_id_cnt = len(_analysis.get("record_identifier_fields", []))
                    _context_cnt = len(_analysis.get("context_fields", []))
                    _metric_cnt = len(_analysis.get("metric_fields", []))
                    _cost_cnt = len(_analysis.get("cost_fields", []))
                    _categorical_cnt = len(_analysis.get("categorical_fields", []))

                    _src = _analysis.get("schema_source", "llm")
                    _dur = _analysis.get("llm_call_duration_sec", 0)
                    _attempts = _analysis.get("llm_call_attempts", 1)
                    _attempt_str = f"{_attempts} 次" if _attempts > 1 else "1 次"
                    st.success(
                        f"实际 LLM 分析完成 | 已调用 LLM {_attempt_str}（总耗时 {_dur:.1f}s）| "
                        f"来源: {_src} | 模型: {_analysis.get('analysis_model', 'N/A')} | "
                        f"表用途: {_analysis.get('table_purpose', '未知')}"
                    )

                    _scol1, _scol2, _scol3, _scol4, _scol5 = st.columns(5)
                    _scol1.metric("记录标识", _record_id_cnt)
                    _scol2.metric("上下文", _context_cnt)
                    _scol3.metric("数值/费用", _metric_cnt + _cost_cnt)
                    _scol4.metric("分类", _categorical_cnt)
                    _scol5.metric("排除行", len(_excl))

                    if _sf:
                        st.caption(f"可出题目标字段: {', '.join(_sf)}")

                    # 展示 question_plan 摘要
                    _qp = _analysis.get("question_plan", {})
                    if _qp:
                        _patterns = _qp.get("recommended_question_patterns", [])
                        if _patterns:
                            st.info(f"**推荐题型**: {', '.join(_patterns)}")

                        _priority = _qp.get("target_field_priority", [])
                        if _priority:
                            _priority_text = " | ".join(
                                f"{tfp['field']}（{tfp.get('role', '')}，优先级 {tfp.get('priority', '')}）"
                                for tfp in _priority
                            )
                            st.caption(f"**目标字段优先级**: {_priority_text}")

                        _forbidden = _qp.get("forbidden_patterns", [])
                        if _forbidden:
                            with st.expander("禁止题型", expanded=False):
                                for fb in _forbidden:
                                    st.caption(f"- {fb}")

                        _rationale = _qp.get("rationale", "")
                        if _rationale:
                            st.caption(f"**分析理由**: {_rationale}")

                    # 折叠区：分析依据
                    with st.expander("查看分析依据（可选）", expanded=False):
                        st.markdown(f"**表格用途**: {_analysis.get('table_purpose', '未知')}")
                        st.markdown(f"**分析模型**: {_analysis.get('analysis_model', 'N/A')} | **分析时间**: {_analysis.get('analysis_timestamp', 'N/A')}")

                        _role_labels = {
                            "record_identifier_fields": "记录标识字段",
                            "context_fields": "上下文/分组字段",
                            "metric_fields": "数值度量字段",
                            "cost_fields": "费用字段",
                            "categorical_fields": "分类字段",
                        }
                        for role_key, role_label in _role_labels.items():
                            fields = _analysis.get(role_key, [])
                            if not fields:
                                continue
                            labels = [f"{f['source_label']}（列 {_col_letter(f['col_index'])}，置信度 {f['confidence']:.0%}）" for f in fields]
                            st.markdown(f"**{role_label}**: {', '.join(labels)}")

                        if _excl:
                            st.markdown(f"**排除行**: {', '.join(str(r) for r in _excl)}")
                        _reasoning = _analysis.get("reasoning", "")
                        if _reasoning:
                            st.markdown(f"**分析理由**: {_reasoning}")

                        with st.expander("Phase 1 原始 JSON", expanded=False):
                            st.code(json.dumps(_analysis, ensure_ascii=False, indent=2), language="json")

                    # 重新分析按钮
                    if st.button("重新分析结构", key="btn_reanalyze_schema"):
                        st.session_state.pop("_schema_analysis", None)
                        st.session_state.pop("_schema_analysis_done", None)
                        st.session_state.pop("_schema_llm_calls", None)
                        st.rerun()

            except Exception as e:
                st.error(f"表格解析失败: {e}")
        else:
            with st.expander("文件内容预览", expanded=False):
                preview_len = 500
                if char_count > preview_len:
                    st.text(file_content[:preview_len] + "...")
                    st.caption(f"（显示前 {preview_len} 字，共 {char_count:,} 字）")
                else:
                    st.text(file_content)

        if char_count > 8000 and not _is_spreadsheet_preview:
            st.info(f"文件较长（{char_count:,} 字），建议使用「标准」或「深度」策略以确保内容覆盖完整。")

        # --- Auto-mode analysis (show when strategy is auto) ---
        # 电子表格检索模式不使用 chunking 策略，跳过此分析
        if qgen_strategy == "自动" and not _is_spreadsheet_preview:
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
            if _is_spreadsheet_preview:
                st.markdown("**当前模式：电子表格两阶段检索评测** — Phase 1 分析结构，Phase 2 生成题目")
                st.caption("Phase 1 和 Phase 2 的实际 Prompt 将在生成完成后显示在审计区。")
                st.markdown("""
**Phase 2 Prompt 结构**（脱敏示例）：

Phase 2 的 LLM 收到以下内容：
1. **Schema 与出题计划** — 记录定位字段、上下文字段、目标字段优先级、推荐题型、禁止题型
2. **表格内容** — 仅包含涉及字段列的 Markdown 表格
3. **候选目录（candidate catalog）** — 按角色分组的候选列表，每项含 `candidate_id`、`record_locator`、`target_field`

LLM 输出格式：
```json
[{"candidate_id": "...", "question": "短检索查询", "target_field_label": "...", "difficulty": "事实", "topic": "..."}]
```

LLM 只能从候选目录中选择 `candidate_id`，不得自造 anchor_range、字段或数值。
`reference_answer` 由本地程序从真实单元格渲染，LLM 不输出。
""")
            else:
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
        # 电子表格模式：Phase 1 未完成时禁用
        _is_spreadsheet_direct = (
            parse_result.get("source_type") in ("xlsx", "xls", "csv")
            and mode_val == MODE_RETRIEVAL
        )
        _schema_ready = st.session_state.get("_schema_analysis_done", False)
        _gen_disabled = _is_spreadsheet_direct and not _schema_ready

        if _gen_disabled:
            st.button("生成题目（请先点击上方「分析表格结构」）", type="primary",
                      key="qgen_run_disabled", use_container_width=True, disabled=True)
        elif st.button("生成题目", type="primary", key="qgen_run", use_container_width=True):
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
                        if _is_spreadsheet_direct:
                            from spreadsheet_question_generator import (
                                generate_questions_from_schema,
                            )

                            _schema_sheets = st.session_state.get("_schema_sheets")
                            _cached_schema = st.session_state.get("_schema_analysis")

                            if not _cached_schema or not _schema_sheets:
                                st.error("Phase 1 未完成，请先点击「分析表格结构」")
                                st.stop()

                            # ── Phase 2: 基于已缓存 schema 出题（不重复调用 Phase 1）──
                            status_text.write("Phase 2: 正在生成检索题...")

                            def _on_progress_spreadsheet(step, total, desc):
                                status_text.write(f"Phase 2: {desc} ({step}/{total})")

                            _phase1_calls = st.session_state.get("_schema_llm_calls", 0)

                            questions, gen_stats = generate_questions_from_schema(
                                _schema_sheets,
                                _cached_schema,
                                qgen_api_key, qgen_base_url, qgen_model,
                                num_questions=qgen_num, difficulty=difficulty_val,
                                topic_hint=qgen_topic_hint,
                                timeout=120,
                                progress_callback=_on_progress_spreadsheet,
                                file_name=file_name,
                            )

                            # 统计 LLM 调用次数
                            _phase2_calls = gen_stats.get("first_raw_count", 0) + gen_stats.get("supplement_count", 0)
                            _total_llm_calls = _phase1_calls + (1 if gen_stats.get("first_raw_count", 0) > 0 else 0) + (1 if gen_stats.get("supplement_count", 0) > 0 else 0)
                            st.session_state["_schema_llm_calls"] = _total_llm_calls

                            # 存储 schema 供审计展示
                            st.session_state["_last_schema_analysis"] = _cached_schema

                            status_text.write(
                                f"Phase 2 出题完成，已调用 LLM {_total_llm_calls} 次"
                            )
                        else:
                            questions, gen_stats = generate_questions(
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

                        # 构建统计摘要
                        _stats_parts = [f"目标 {gen_stats.get('target', qgen_num)}"]
                        _stats_parts.append(f"LLM 原始生成 {gen_stats.get('raw_count', '?')}")
                        _stats_parts.append(f"校验淘汰 {gen_stats.get('validation_eliminated', 0)}")
                        _stats_parts.append(f"去重淘汰 {gen_stats.get('dedup_eliminated', 0)}")
                        if gen_stats.get("supplement_rounds"):
                            _stats_parts.append(
                                f"补题 {gen_stats['supplement_rounds']} 轮，新增 {gen_stats['supplement_new']}"
                            )
                        _stats_parts.append(f"最终 {gen_stats.get('final_count', len(questions))}")
                        # 电子表格特有统计
                        if gen_stats.get("sheet_count"):
                            _stats_parts.append(f"工作表 {gen_stats['sheet_count']}")
                        if gen_stats.get("block_count"):
                            _stats_parts.append(f"表格块 {gen_stats['block_count']}")
                        if gen_stats.get("formula_warnings"):
                            _stats_parts.append(f"公式警告 {gen_stats['formula_warnings']}")
                        _stats_summary = " | ".join(_stats_parts)

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
                            st.caption(f"📊 {_stats_summary}")

                            # 若不足目标数，显示说明
                            if gen_stats.get("final_count", len(questions)) < qgen_num:
                                st.warning(
                                    f"目标 {qgen_num}，最终 {gen_stats.get('final_count', len(questions))}；"
                                    f"已完成 {gen_stats.get('supplement_rounds', 0)} 轮补题，"
                                    f"源文本中仅发现 {gen_stats.get('final_count', len(questions))} "
                                    f"条唯一且合格的可检索证据。"
                                )
                        else:
                            gen_status.update(label="生成完成但保存失败", state="error")
                            st.error(f"题目生成成功，但文件保存失败：{output_path}")
                    except Exception as e:
                        gen_status.update(label="生成失败", state="error")
                        st.error(f"生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.info("请在上方「配置」区域上传知识库文件（.txt / .md / .docx / .xlsx）")

    # --- Results display ---
    questions = st.session_state.get("generated_questions")
    if questions:
        # 检测当前题集模式，决定字段展示标签
        _qgen_mode = st.session_state.get("qgen_last_generated_mode", MODE_QA)
        _is_retrieval = (_qgen_mode == MODE_RETRIEVAL)
        _label_question = "检索查询" if _is_retrieval else "问题"
        _label_ref_answer = "金标准原文证据" if _is_retrieval else "参考答案"

        st.divider()
        st.subheader(f"生成结果（{len(questions)} 道{'查询' if _is_retrieval else '题目'}）")

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
        df_display = df_q[["question", "difficulty", "topic"]].rename(columns={"question": _label_question})
        st.dataframe(
            df_display,
            use_container_width=True,
            height=min(400, len(df_q) * 40 + 60),
        )

        for i, item in enumerate(questions, 1):
            with st.expander(f"#{i} {item.get('question', '')[:60]}"):
                st.markdown(f"**{_label_question}**: {item.get('question', '')}")
                st.markdown(f"**{_label_ref_answer}**: {item.get('reference_answer', '')}")
                if item.get("evidence_schema_display"):
                    st.markdown(f"**字段摘要**: {item['evidence_schema_display']}")
                st.markdown(f"**难度**: {item.get('difficulty', '')} | **主题**: {item.get('topic', '')}")

        # 电子表格两阶段分析依据（可选审计）
        _last_schema = st.session_state.get("_last_schema_analysis")
        if _last_schema:
            with st.expander("查看分析依据（可选）", expanded=False):
                _total_calls = st.session_state.get("_schema_llm_calls", 0)
                st.markdown(f"**LLM 调用次数**: {_total_calls} 次（Phase 1: 1 次 + Phase 2: {_total_calls - 1} 次）")
                st.markdown(f"**表格用途**: {_last_schema.get('table_purpose', '未知')}")
                st.markdown(f"**分析模型**: {_last_schema.get('analysis_model', 'N/A')} | **分析时间**: {_last_schema.get('analysis_timestamp', 'N/A')}")

                _role_labels = {
                    "group_fields": "上下文/分组字段",
                    "record_fields": "记录标识字段",
                    "metric_fields": "数值度量字段",
                    "cost_fields": "费用字段",
                    "ambiguous_fields": "待确认字段",
                }
                for role_key, role_label in _role_labels.items():
                    fields = _last_schema.get(role_key, [])
                    if not fields:
                        continue
                    labels = [f"{f['source_label']}（列 {_col_letter(f['col_index'])}，置信度 {f['confidence']:.0%}）" for f in fields]
                    st.markdown(f"**{role_label}**: {', '.join(labels)}")

                _excluded = _last_schema.get("excluded_rows", [])
                if _excluded:
                    st.markdown(f"**排除行**: {', '.join(str(r) for r in _excluded)}")

                _rl = _last_schema.get("record_locator_fields", [])
                _qt = _last_schema.get("question_target_fields", [])
                if _rl:
                    st.markdown(f"**记录定位字段**: {', '.join(_rl)}")
                if _qt:
                    st.markdown(f"**目标字段（可出题）**: {', '.join(_qt)}")

                # question_plan 展示
                _qp = _last_schema.get("question_plan", {})
                if _qp.get("target_field_plans"):
                    st.markdown("**题型计划**:")
                    for _plan in _qp["target_field_plans"]:
                        st.markdown(
                            f"- {_plan.get('question_type', '')}: "
                            f"目标={_plan.get('target_field', '')} | "
                            f"{_plan.get('description', '')}"
                        )
                    _cr = _qp.get("coverage_rule", "")
                    if _cr:
                        st.caption(f"覆盖规则: {_cr}")

                _reasoning = _last_schema.get("reasoning", "")
                if _reasoning:
                    st.markdown(f"**分析理由**: {_reasoning}")

                with st.expander("Phase 1 原始 JSON", expanded=False):
                    st.code(json.dumps(_last_schema, ensure_ascii=False, indent=2), language="json")

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
            # 使用缓存的题集索引（避免每次 rerun 扫描磁盘）
            _qs_index = _build_question_set_index(_get_questions_dir_mtime())

            if not _qs_index:
                st.warning("暂无历史记录，请先在「题目生成」或「批量提问」中生成/保存过结果")
            else:
                # 构建显示标签
                history_files = [Path(item["path"]) for item in _qs_index]
                file_info_cache = {}
                file_labels = []
                for item in _qs_index:
                    fp = Path(item["path"])
                    modes = item["modes"]
                    total_sampled = sum(modes.values())
                    q_count = item["question_count"]

                    # 模式标签
                    eval_type = item.get("evaluation_type", "")
                    if total_sampled == 0:
                        mode_tag = "[空文件]"
                    elif eval_type == "chunk_exact" or (modes["chunk_exact"] > 0 and modes["unknown"] == 0):
                        mode_tag = "[Chunk 精确匹配]"
                    elif modes["retrieval"] > 0 and modes["qa"] > 0:
                        mode_tag = "[混合]"
                    elif modes["retrieval"] > 0 and modes["unknown"] == 0:
                        mode_tag = "[检索评测]"
                    elif modes["qa"] > 0 and modes["unknown"] == 0:
                        mode_tag = "[全流程问答]"
                    elif modes["unknown"] > 0 and modes["retrieval"] == 0 and modes["qa"] == 0 and modes["chunk_exact"] == 0:
                        mode_tag = "[旧版]"
                    elif modes["retrieval"] > 0:
                        mode_tag = "[检索评测+旧版]"
                    elif modes["qa"] > 0:
                        mode_tag = "[全流程问答+旧版]"
                    elif modes["chunk_exact"] > 0:
                        mode_tag = "[Chunk 精确匹配+旧版]"
                    else:
                        mode_tag = "[旧版]"

                    if item["has_set_info"] and item["set_name"]:
                        _sid = item.get("set_id", "")
                        _ts_display = ""
                        if _sid:
                            _parts = _sid.split("_", 3)
                            if len(_parts) >= 3:
                                _date_part = _parts[1]
                                _time_part = _parts[2]
                                if len(_date_part) == 8 and len(_time_part) >= 6:
                                    _ts_display = f"{_date_part[:4]}-{_date_part[4:6]}-{_date_part[6:8]} {_time_part[:2]}:{_time_part[2:4]}"
                        _sid_short = f" · qs...{_sid[12:20]}" if _sid and len(_sid) > 20 else ""
                        _ts_part = f" · {_ts_display}" if _ts_display else ""
                        _src_fmt = ""
                        if item.get("source_format") == "xlsx":
                            _src_fmt = " · 来源: Excel"
                            if item.get("source_file_name"):
                                _src_fmt += f"（{item['source_file_name']}）"
                        label = f"{mode_tag} {item['set_name']} · {q_count} 题{_ts_part}{_sid_short}{_src_fmt}"
                    else:
                        label = f"{mode_tag} {fp.stem} · {q_count} 题 [旧版题集]"

                    file_labels.append(label)
                    # 构建兼容旧代码的 file_info_cache（Path key）
                    file_info_cache[fp] = {
                        "modes": modes,
                        "set_name": item["set_name"],
                        "set_id": item["set_id"],
                        "question_count": q_count,
                        "has_set_info": item["has_set_info"],
                        "source_format": item.get("source_format", ""),
                        "source_file_name": item.get("source_file_name", ""),
                        "created_at": item.get("created_at"),
                    }

                # 清理旧版单选 session_state
                st.session_state.pop("batch_history_file", None)

                selected_indices = st.multiselect(
                    "选择历史题集（可多选）",
                    range(len(file_labels)),
                    format_func=lambda i: file_labels[i],
                    default=[],
                    key="batch_history_files",
                )

                def _load_questions_from_file(filepath):
                    """从题集文件加载问题列表（仅在选择后调用）。

                    保留所有字段（chunk_exact 元数据、question_id 等），不做过滤。
                    """
                    qs = []
                    raw_lines = filepath.read_text(encoding="utf-8").strip().split("\n")
                    for line in raw_lines:
                        try:
                            obj = json.loads(line)
                            q = obj.get("question") or obj.get("query") or ""
                            if q.strip():
                                # 保留所有字段，确保 chunk_exact 元数据不丢失
                                item = dict(obj)
                                item["question"] = q.strip()
                                qs.append(item)
                        except json.JSONDecodeError:
                            continue
                    return qs

                if selected_indices:
                    # 构建多题集数据
                    selected_sets = []
                    for idx in selected_indices:
                        fp = history_files[idx]
                        info = file_info_cache[fp]
                        qs = _load_questions_from_file(fp)
                        selected_sets.append({"file": fp, "info": info, "questions": qs})

                    total_questions = sum(len(s["questions"]) for s in selected_sets)

                    # 汇总显示
                    st.success(f"已选 **{len(selected_sets)}** 个题集，共 **{total_questions}** 题")
                    for i, ss in enumerate(selected_sets, 1):
                        info = ss["info"]
                        modes = info["modes"]
                        eval_type = info.get("evaluation_type", "")
                        if eval_type == "chunk_exact" or modes["chunk_exact"] > 0:
                            mode_tag = "Chunk 精确匹配"
                        elif modes["retrieval"] > 0 and modes["qa"] > 0:
                            mode_tag = "混合"
                        elif modes["retrieval"] > 0:
                            mode_tag = "检索评测"
                        elif modes["qa"] > 0:
                            mode_tag = "全流程问答"
                        else:
                            mode_tag = "旧版"
                        _sid = info.get("set_id", "")
                        _sid_short = f"...{_sid[-8:]}" if len(_sid) > 8 else _sid
                        st.caption(
                            f"  {i}. {info['set_name']} · {len(ss['questions'])} 题"
                            f" · {mode_tag} · {_sid_short}"
                        )

                    # 检查跨题集 question_id 重复
                    all_qids = []
                    for ss in selected_sets:
                        for q in ss["questions"]:
                            qid = q.get("question_id") or q.get("question", "")[:20]
                            all_qids.append(qid)
                    if len(all_qids) != len(set(all_qids)):
                        st.info(
                            "⚠️ 不同题集含相同 question_id，"
                            "运行关联以 question_set_id + run_id 为准。"
                        )

                    # 合并 questions_list（用于后续执行，保留 question_set_id 来源）
                    questions_list = []
                    for ss in selected_sets:
                        questions_list.extend(ss["questions"])

                    # 分组预览题目
                    with st.expander("预览题目（按题集分组）", expanded=False):
                        for ss in selected_sets:
                            info = ss["info"]
                            qs = ss["questions"]
                            _set_has_invalid_chunk_exact = False
                            with st.expander(f"{info['set_name']} · {len(qs)} 题", expanded=False):
                                for i, q in enumerate(qs, 1):
                                    qtext = q.get("question", "")
                                    ref = q.get("reference_answer", "")
                                    qm = q.get("question_mode", "")
                                    if qm == "chunk_exact":
                                        mode_badge = "🎯 "
                                        doc_id = q.get("document_id", "")
                                        seg_id = q.get("expected_segment_id", "")
                                        content_hash = q.get("expected_content_hash", "")
                                        target = q.get("target_label", "")
                                        snap_id = q.get("snapshot_id", "")
                                        position = q.get("source_position", "")
                                        candidate_id = q.get("candidate_id", "")
                                        expected_content = q.get("expected_content", "")

                                        # 校验绑定完整性
                                        missing = []
                                        if not snap_id: missing.append("snapshot_id")
                                        if not doc_id: missing.append("document_id")
                                        if not seg_id: missing.append("expected_segment_id")
                                        if not content_hash: missing.append("expected_content_hash")

                                        if missing:
                                            _set_has_invalid_chunk_exact = True
                                            st.error(f"❌ {i}. {qtext} — 无效绑定，缺少: {', '.join(missing)}")
                                            st.caption("   需重新从 snapshot 生成此题集")
                                        else:
                                            doc_short = doc_id[:12] + "..." if len(str(doc_id)) > 12 else doc_id
                                            seg_short = seg_id[:12] + "..." if len(str(seg_id)) > 12 else seg_id
                                            pos_str = f"pos:{position}" if position != "" else ""
                                            st.write(f"{mode_badge}{i}. {qtext}")
                                            st.caption(
                                                f"   文档: {doc_short} | segment: {seg_short}"
                                                f" | {pos_str} | 标签: {target}"
                                            )
                                            if expected_content:
                                                _ec_len = len(expected_content)
                                                _ec_label = f"预期 Chunk 证据（{_ec_len:,} 字符）"
                                                # 检测历史 500 字符截断
                                                _is_legacy_truncated = (_ec_len == 500)
                                                with st.expander(_ec_label, expanded=False):
                                                    if _is_legacy_truncated:
                                                        st.warning(
                                                            "⚠️ 历史题集仅保存了前 500 字符的预览；"
                                                            "机器判定仍基于 segment_id/content_hash。"
                                                        )
                                                    st.text(expected_content)
                                    elif qm == MODE_RETRIEVAL:
                                        mode_badge = "🔍 "
                                        if ref:
                                            st.write(f"{mode_badge}{i}. {qtext}")
                                            st.caption(f"   参考答案: {ref[:80]}{'...' if len(ref) > 80 else ''}")
                                        else:
                                            st.write(f"{mode_badge}{i}. {qtext}")
                                    elif qm == MODE_QA:
                                        mode_badge = "💬 "
                                        if ref:
                                            st.write(f"{mode_badge}{i}. {qtext}")
                                            st.caption(f"   参考答案: {ref[:80]}{'...' if len(ref) > 80 else ''}")
                                        else:
                                            st.write(f"{mode_badge}{i}. {qtext}")
                                    else:
                                        if ref:
                                            st.write(f"{i}. {qtext}")
                                            st.caption(f"   参考答案: {ref[:80]}{'...' if len(ref) > 80 else ''}")
                                        else:
                                            st.write(f"{i}. {qtext}")

                            if _set_has_invalid_chunk_exact:
                                st.warning(
                                    f"题集「{info['set_name']}」含无效绑定的 chunk_exact 题目，"
                                    f"需重新从 snapshot 生成。禁止加入批量提问。"
                                )

    # --- RAG 配置方案 ---
    with st.expander("RAG 配置方案", expanded=False):
        from experiment import (
            create_config_profile, load_config_profile, list_config_profiles,
            create_experiment_run, update_experiment_run, ensure_question_id,
            get_config_summary, get_config_display_value,
            CONFIG_FIELD_SCHEMA,
            config_fingerprint, find_canonical_config,
            merge_duplicate_configs,
        )

        # 配置来源选择（"另存为新方案"按钮通过 trigger flag 在 widget 渲染前切换）
        if st.session_state.pop("_batch_switch_to_new", False):
            st.session_state["batch_config_source"] = "新建配置方案"
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
                    _cid = cfg.get("config_id", "")
                    _cid_sfx = _cid[-8:] if len(_cid) > 8 else _cid
                    _created = cfg.get("created_at", "")
                    _ts = _created[:16].replace("T", " ") if _created else ""
                    _summary = get_config_summary(cfg)
                    config_options.append((_cid, f"{_summary} | {_cid_sfx} | {_ts}"))

                selected_config_id = st.selectbox(
                    "选择历史配置",
                    options=[c[0] for c in config_options],
                    format_func=lambda x: next((c[1] for c in config_options if c[0] == x), x),
                    key="batch_selected_config",
                )

                if selected_config_id:
                    # 配置切换时清理依赖旧配置的缓存
                    _prev_cfg = st.session_state.get("_batch_prev_config_id")
                    if _prev_cfg and _prev_cfg != selected_config_id:
                        st.session_state.pop("batch_existing_runs_by_qs", None)
                        st.session_state.pop("batch_qs_strategy", None)
                    st.session_state["_batch_prev_config_id"] = selected_config_id

                    selected_config = load_config_profile(selected_config_id)
                    if selected_config:
                        st.caption(f"当前使用历史配置: **{selected_config.get('config_name', '')}**")
                        # 只读摘要 — key_prefix 使用完整 config_id 的哈希，确保切换时 widget 状态完全隔离
                        import hashlib as _hashlib
                        _ro_key = f"ro_{_hashlib.md5(selected_config_id.encode()).hexdigest()}"
                        with st.container(border=True):
                            st.markdown("**当前配置（只读）**")
                            render_config_form(selected_config, key_prefix=_ro_key, disabled=True)
                        # 记录实际渲染的 config_id，用于执行前一致性校验
                        st.session_state["batch_displayed_config_id"] = selected_config_id
                                        # 另存为新方案（通过 trigger flag 在 widget 渲染前切换 radio）
                        if st.button("基于此配置另存为新方案", key="batch_save_as_new"):
                            st.session_state["_batch_switch_to_new"] = True
                            for _k, _, _, _, _, _ in CONFIG_FIELD_SCHEMA:
                                _val = selected_config.get(_k, "")
                                st.session_state[f"batch_new_{_k}"] = f"{_val} (副本)" if _k == "config_name" else _val
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

        _env_api_key = os.getenv("DIFY_APP_API_KEY", "") or os.getenv("DIFY_API_KEY", "")
        _env_base_url = os.getenv("DIFY_APP_API_BASE", "") or os.getenv("DIFY_API_BASE", "http://localhost/v1")

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
                    help="来自 .env 的默认值（DIFY_APP_API_KEY）" if _env_api_key else "",
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
                    help="串行模式下每次请求之间的等待时间。并发模式下此设置不生效。"
                )

            dify_concurrency = st.number_input(
                "并发数", min_value=1, max_value=8, value=3, step=1, key="batch_concurrency",
                help='同时发起的 Dify 请求数。设为 1 即传统串行模式（受「请求间隔」控制）；'
                     '大于 1 时忽略请求间隔，多题同时提问以缩短总耗时。'
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
            st.caption("将使用 .env 中的 `DIFY_APP_API_KEY` 作为默认密钥。")
            dify_api_key = _env_api_key

    # --- Run batch query ---
    st.divider()

    # 执行前预检（使用缓存的 run 摘要，通过显式按钮触发）
    # 注意：config_id 解析延迟到按钮点击时，避免 multiselect/radio rerun 时
    # 调用 list_config_profiles() 扫描磁盘
    _existing_runs_by_qs = {}  # question_set_id -> run summary
    _qs_rerun_strategy = "skip"

    # 仅在历史题集模式且有选中题集时显示预检按钮
    _show_precheck = (
        q_source == "从历史记录加载"
        and 'selected_sets' in dir()
        and len(selected_sets) > 0
    )

    if _show_precheck:
        # 显式检查按钮（点击时才解析 config_id + 扫描 run 摘要）
        if st.button("检查已完成运行", key="btn_check_existing_runs"):
            # 解析 config_id（仅在按钮点击时执行）
            _pre_config_source = st.session_state.get("batch_config_source", "新建配置方案")
            _pre_config_id = ""
            if _pre_config_source == "使用历史配置":
                _pre_config_id = st.session_state.get("batch_selected_config", "")
            else:
                _pre_form_vals = {}
                for _key, _, _, _, _, _ in CONFIG_FIELD_SCHEMA:
                    _pre_form_vals[_key] = st.session_state.get(f"batch_new_{_key}", "")
                _pre_name = str(_pre_form_vals.get("config_name", "")).strip()
                if _pre_name:
                    _pre_clean = collect_config_updates(_pre_form_vals)
                    _pre_existing = list_config_profiles(include_archived=False)
                    _pre_fp = config_fingerprint(_pre_clean)
                    _pre_canonical = find_canonical_config(_pre_fp, _pre_existing)
                    if _pre_canonical:
                        _pre_config_id = _pre_canonical["config_id"]

            if _pre_config_id:
                # 强制刷新缓存后读取 run 摘要
                _build_run_summary_index.clear()
                _run_summaries = _build_run_summary_index(_get_experiments_dir_mtime())
                _selected_qs_ids = {ss["info"].get("set_id", "") for ss in selected_sets}
                for _rs in _run_summaries:
                    if (_rs.get("config_id") == _pre_config_id
                        and _rs.get("status") == "completed"
                        and _rs.get("question_set_id")
                        and _rs["question_set_id"] in _selected_qs_ids):
                        _existing_runs_by_qs[_rs["question_set_id"]] = _rs

                # 缓存到 session_state，避免下次 rerun 丢失
                st.session_state["batch_existing_runs_by_qs"] = _existing_runs_by_qs
            else:
                st.session_state.pop("batch_existing_runs_by_qs", None)

        # 从 session_state 恢复已检查的结果（不触发磁盘扫描）
        _existing_runs_by_qs = st.session_state.get("batch_existing_runs_by_qs", {})

    # 显示已有 run 信息和策略选择
    if _existing_runs_by_qs:
        st.markdown("#### 已有完成记录")
        st.caption("以下题集在当前配置下已有 completed run：")
        for _qs_id, _run in _existing_runs_by_qs.items():
            _qs_name = _run.get("question_set_name", _qs_id)
            _run_id = _run.get("run_id", "")
            _completed_at = _run.get("started_at", "")
            _q_count = _run.get("question_count", "?")
            _completed_str = _completed_at[:16].replace("T", " ") if _completed_at else "未知时间"
            st.caption(
                f"  · {_qs_name} · run: `{_run_id}`"
                f" · 完成于 {_completed_str} · {_q_count} 题"
            )

        _qs_rerun_strategy = st.radio(
            "执行策略",
            ["skip", "rerun_all"],
            format_func=lambda x: {
                "skip": "跳过已完成题集（推荐）",
                "rerun_all": "为所有已选题集重新执行",
            }[x],
            index=0,
            key="batch_qs_strategy",
            help="跳过：有 completed run 的题集不创建新 run；重新执行：每个题集创建全新 run，旧 run 完整保留",
        )

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

                # ── 四重一致性校验（fail-closed） ──
                # (1) selectbox 选中的 config_id
                _check_selectbox = _config_id
                # (2) 上次渲染只读快照时记录的 config_id
                _check_displayed = st.session_state.get("batch_displayed_config_id", "")
                # (3) 从磁盘重新加载的 profile.config_id
                _verify_config = load_config_profile(_config_id)
                _check_disk = _verify_config.get("config_id", "") if _verify_config else ""
                # (4) 即将传入执行器的 config_id（即 _config_id 本身）
                _check_executor = _config_id

                _all_ids = {_check_selectbox, _check_displayed, _check_disk, _check_executor}
                if len(_all_ids) > 1 or not _config_id:
                    st.error(
                        f"配置一致性校验失败，禁止执行。"
                        f"下拉框: `{_check_selectbox[:20]}`，"
                        f"显示快照: `{_check_displayed[:20]}`，"
                        f"磁盘加载: `{_check_disk[:20]}`，"
                        f"执行器: `{_check_executor[:20]}`。"
                        f"请重新选择配置后重试。"
                    )
                    st.stop()

                if not _verify_config:
                    st.error(f"配置 {_config_id} 不存在或已被删除")
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

                _clean_vals = collect_config_updates(_form_vals)

                # 去重检查：查找 fingerprint 相同的既有配置
                _existing_all = list_config_profiles(include_archived=False)
                _fp = config_fingerprint(_clean_vals)
                _canonical = find_canonical_config(_fp, _existing_all)

                if _canonical:
                    _cid_suffix = _canonical["config_id"][-8:]
                    _cname = _canonical.get("config_name", "")
                    st.info(
                        f"检测到内容相同的已有配置，已复用：{_cname}（...{_cid_suffix}）"
                    )
                    _config_id = _canonical["config_id"]
                else:
                    config_result = create_config_profile(**_clean_vals)
                    _config_id = config_result["config_id"]

            # 判断是否为多题集模式
            _is_multi_qs = (
                q_source == "从历史记录加载"
                and 'selected_sets' in dir()
                and len(selected_sets) > 1
            )

            # 构建连接配置 manifest 更新字段（不含 API Key）
            def _build_manifest_updates(q_set_id="", q_set_name=""):
                updates = {}
                if q_set_id or q_set_name:
                    updates["question_set_id"] = q_set_id
                    updates["question_set_name"] = q_set_name
                if _selected_profile_id:
                    updates["dify_connection_profile_id"] = _selected_profile_id
                    updates["dify_connection_profile_name"] = _selected_profile_name
                    updates["dify_base_url"] = dify_base_url
                    updates["dify_workflow_description"] = _selected_profile_desc
                elif dify_base_url:
                    updates["dify_base_url"] = dify_base_url
                return updates

            # 执行单个题集的批量提问，返回 (run_id, run_dir, batch_results)
            def _execute_single_qs(questions, q_set_id, q_set_name, label=""):
                run_result = create_experiment_run(
                    config_id=_config_id,
                    question_set_source=st.session_state.get("batch_q_source", ""),
                    question_count=len(questions),
                )
                _run_id = run_result["run_id"]
                _run_dir = run_result["run_dir"]

                manifest_up = _build_manifest_updates(q_set_id, q_set_name)
                if manifest_up:
                    update_experiment_run(_run_id, manifest_up)

                question_ids = []
                for q in questions:
                    q = ensure_question_id(q)
                    question_ids.append(q.get("question_id", ""))

                st.info(f"运行已创建: `{_run_id}` | 题集: {q_set_name or q_set_id or '未知'}")

                _batch_results = []
                _completed = [0]  # 用列表包装以在闭包中修改
                _progress = st.progress(0, text=f"{label}准备开始...")
                _status = st.container()
                _concurrency = st.session_state.get("batch_concurrency", 3)

                for idx, total, result in run_batch_query(
                    questions, dify_api_key, dify_base_url,
                    timeout=dify_timeout, delay=dify_delay,
                    run_id=_run_id,
                    config_id=_config_id,
                    question_ids=question_ids,
                    max_workers=_concurrency,
                ):
                    _completed[0] += 1
                    _progress.progress(
                        _completed[0] / total,
                        text=f"{label}已完成 {_completed[0]} / {total}（第 {idx + 1} 题）",
                    )
                    _batch_results.append(result)

                    with _status:
                        if result["success"]:
                            answer_preview = (result["sample"].get("final_answer", "") or "")[:80]
                            st.success(f"✅ [{idx + 1}/{total}] {result['question'][:40]}... → {answer_preview}")
                        else:
                            st.error(f"❌ [{idx + 1}/{total}] {result['question'][:40]}... → {result['error'][:80]}")

                # 并发模式下结果按完成顺序收集，保存前按原始索引稳定排序
                if _concurrency > 1:
                    _batch_results.sort(key=lambda r: r.get("_original_index", 0))

                _progress.progress(1.0, text=f"{label}提问完成！")

                # 保存结果到运行目录
                run_batch_path = _run_dir / "batch_results.jsonl"
                with run_batch_path.open("w", encoding="utf-8") as f:
                    for r in _batch_results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                raw_path, raw_filename = push_to_raw_dir(_batch_results)

                update_experiment_run(_run_id, {
                    "batch_results_file": "batch_results.jsonl",
                    "raw_results_file": raw_filename,
                    "status": "completed",
                })

                return _run_id, _run_dir, _batch_results

            if _is_multi_qs:
                # 多题集模式：逐个执行
                completed_runs = []
                failed_runs = []
                skipped_runs = []
                global_done = 0
                global_total = sum(len(s["questions"]) for s in selected_sets)
                global_progress = st.progress(0, text="多题集执行开始...")

                for set_idx, qs_info in enumerate(selected_sets):
                    qs = qs_info["questions"]
                    info = qs_info["info"]
                    q_set_id = info.get("set_id", "")
                    q_set_name = info.get("set_name", "")
                    set_label = f"题集 {set_idx + 1}/{len(selected_sets)}: {q_set_name} — "

                    # 跳过策略检查
                    if _qs_rerun_strategy == "skip" and q_set_id in _existing_runs_by_qs:
                        _existing_run = _existing_runs_by_qs[q_set_id]
                        skipped_runs.append({
                            "run_id": _existing_run.get("run_id", ""),
                            "q_set_name": q_set_name,
                            "q_set_id": q_set_id,
                            "count": len(qs),
                        })
                        global_done += len(qs)
                        global_progress.progress(
                            global_done / global_total,
                            text=f"全局进度: {global_done}/{global_total} 题（跳过已完成）",
                        )
                        st.info(
                            f"⏭️ {set_label}{len(qs)} 题 — 已跳过"
                            f"（已有 run: `{_existing_run.get('run_id', '')}`）"
                        )
                        continue

                    st.markdown(f"### {set_label}{len(qs)} 题")

                    if _qs_rerun_strategy == "rerun_all" and q_set_id in _existing_runs_by_qs:
                        st.warning(
                            f"将为 {q_set_name} 创建新的独立 run，"
                            f"旧 run `{_existing_runs_by_qs[q_set_id].get('run_id', '')}` 完整保留。"
                        )

                    try:
                        _run_id, _run_dir, _batch_results = _execute_single_qs(
                            qs, q_set_id, q_set_name, label=set_label,
                        )
                        _success = sum(1 for r in _batch_results if r["success"])
                        completed_runs.append({
                            "run_id": _run_id,
                            "q_set_name": q_set_name,
                            "q_set_id": q_set_id,
                            "count": len(qs),
                            "success": _success,
                        })
                        global_done += len(qs)
                        global_progress.progress(
                            global_done / global_total,
                            text=f"全局进度: {global_done}/{global_total} 题已完成",
                        )
                    except Exception as e:
                        failed_runs.append({
                            "q_set_name": q_set_name,
                            "q_set_id": q_set_id,
                            "count": len(qs),
                            "error": str(e),
                        })
                        global_done += len(qs)
                        global_progress.progress(
                            global_done / global_total,
                            text=f"全局进度: {global_done}/{global_total} 题（含失败）",
                        )
                        st.error(f"题集 {q_set_name} 执行失败: {e}")

                # 最终汇总
                global_progress.progress(1.0, text="全部执行完成")
                _build_run_summary_index.clear()  # 执行完成后刷新缓存
                st.session_state.pop("batch_existing_runs_by_qs", None)  # 清除旧预检结果
                st.markdown("### 执行汇总")

                _actual_executed = sum(r["count"] for r in completed_runs) + sum(r["count"] for r in failed_runs)

                if completed_runs:
                    _total_success = sum(r["success"] for r in completed_runs)
                    _total_count = sum(r["count"] for r in completed_runs)
                    st.success(
                        f"成功: {len(completed_runs)} 个题集 ({_total_count} 题, "
                        f"{_total_success} 条成功回答)"
                    )
                    for r in completed_runs:
                        st.caption(
                            f"  ✓ {r['q_set_name']} · {r['count']} 题"
                            f" · run_id: `{r['run_id']}`"
                        )

                if failed_runs:
                    _total_failed_count = sum(r["count"] for r in failed_runs)
                    st.error(f"失败: {len(failed_runs)} 个题集 ({_total_failed_count} 题)")
                    for r in failed_runs:
                        st.caption(
                            f"  ✗ {r['q_set_name']} · {r['count']} 题"
                            f" · 错误: {r['error'][:80]}"
                        )

                if skipped_runs:
                    _total_skipped_count = sum(r["count"] for r in skipped_runs)
                    st.info(f"跳过: {len(skipped_runs)} 个题集 ({_total_skipped_count} 题)")
                    for r in skipped_runs:
                        st.caption(
                            f"  ⏭️ {r['q_set_name']} · {r['count']} 题"
                            f" · 既有 run: `{r['run_id']}`"
                        )

                st.caption(
                    f"实际执行: {_actual_executed}/{global_total} 题"
                    + (f"（跳过 {sum(r['count'] for r in skipped_runs)} 题）" if skipped_runs else "")
                )

                # 存储最后一个成功 run 的结果到 session（兼容后续结果展示）
                if completed_runs:
                    last_run = completed_runs[-1]
                    st.session_state["batch_run_id"] = last_run["run_id"]

            else:
                # 单题集模式（兼容现有行为，含跳过逻辑）
                _q_set_id = ""
                _q_set_name = ""
                for q in questions_list:
                    if q.get("question_set_id"):
                        _q_set_id = q["question_set_id"]
                        _q_set_name = q.get("question_set_name", "")
                        break

                if _qs_rerun_strategy == "skip" and _q_set_id and _q_set_id in _existing_runs_by_qs:
                    _existing_run = _existing_runs_by_qs[_q_set_id]
                    st.info(
                        f"⏭️ 题集 {_q_set_name} 已跳过"
                        f"（已有 run: `{_existing_run.get('run_id', '')}`）"
                    )
                    st.session_state["batch_run_id"] = _existing_run.get("run_id", "")
                else:
                    if _qs_rerun_strategy == "rerun_all" and _q_set_id and _q_set_id in _existing_runs_by_qs:
                        st.warning(
                            f"将为 {_q_set_name} 创建新的独立 run，"
                            f"旧 run `{_existing_runs_by_qs[_q_set_id].get('run_id', '')}` 完整保留。"
                        )

                    run_id, run_dir, batch_results = _execute_single_qs(
                        questions_list, _q_set_id, _q_set_name,
                    )

                    st.session_state["batch_results"] = batch_results
                    st.session_state["batch_run_id"] = run_id
                    _build_run_summary_index.clear()  # 执行完成后刷新缓存
                    st.session_state.pop("batch_existing_runs_by_qs", None)  # 清除旧预检结果

                    _success = sum(1 for r in batch_results if r["success"])
                    st.success(f"批量提问完成！成功 {_success} / {len(batch_results)} 条")
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
    _record_rss("样本准备页")
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
            ["从 API 拉取", "上传文件"],
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
                # 上传后自动选中新文件
                st.session_state["raw_select"] = uploaded.name
                st.rerun()

        elif source_mode == "从 API 拉取":
            from langfuse_connection import (
                list_profiles, load_profile, create_profile, update_profile,
                delete_profile, check_connection, mask_public_key,
                identify_project_info, get_profile_api_keys, has_profile_api_keys,
            )
            from langfuse_project import (
                list_projects, load_project, register_project, get_project_stats,
                incremental_sync, load_project_traces, generate_project_id,
                backfill_observations, get_observation_coverage,
                list_cleanup_candidates, cleanup_files,
                list_parseable_sources, list_snapshots, get_current_snapshot_id,
                export_snapshot_as_jsonl, export_current_cache_as_jsonl,
                get_current_cache_stats, compute_file_fingerprint,
                mark_snapshot_parsed,
                can_cleanup_snapshot, cleanup_old_snapshots,
                create_frozen_snapshot,
                get_current_eval_cache, update_eval_cache,
                _find_snapshot_references,
                get_processed_paths, find_latest_processed,
                find_processed_for_run, update_run_index, backfill_run_index_all,
                PROJECTS_DIR, PROCESSED_DIR as _LP_PROCESSED_DIR,
            )

            # 连接模式选择
            _conn_mode = st.radio(
                "连接方式",
                ["使用已保存连接（推荐）", "临时手动填写"],
                horizontal=True,
                key="lf_conn_mode",
            )

            langfuse_host = ""
            langfuse_pk = ""
            langfuse_sk = ""

            if _conn_mode == "使用已保存连接（推荐）":
                profiles = list_profiles()
                _has_profiles = bool(profiles)
                if not _has_profiles:
                    st.info("暂无已保存的 Langfuse 连接配置。请在下方新建，或切换为「临时手动填写」。")

                # 选择器（有 profile 时显示）
                selected_pid = ""
                if _has_profiles:
                    _pid_options = {p["profile_id"]: f"{p['display_name']} | {p['host']}" for p in profiles}
                    _pid_list = list(_pid_options.keys())
                    selected_pid = st.selectbox(
                        "选择连接配置",
                        options=_pid_list,
                        format_func=lambda x: _pid_options.get(x, x),
                        key="lf_profile_select",
                    )
                    _sel_profile = load_profile(selected_pid)
                    if _sel_profile:
                        _has_keys = has_profile_api_keys(selected_pid)
                        if _has_keys:
                            _pk, _sk = get_profile_api_keys(selected_pid)
                            st.caption(
                                f"Host: {_sel_profile['host']}　|　"
                                f"Public Key: {mask_public_key(_pk)}　|　"
                                f"Secret Key: 已配置"
                            )
                        else:
                            st.warning("已保存配置但凭据缺失，请编辑并填入 Key。")

                # 管理连接配置（新建始终可用）
                with st.expander("管理连接配置", expanded=not _has_profiles):
                    _mgmt_options = ["新建配置"]
                    if _has_profiles:
                        _mgmt_options.extend(["编辑当前", "删除当前"])
                    _mgmt_action = st.radio(
                        "操作",
                        _mgmt_options,
                        horizontal=True,
                        key="lf_mgmt_action",
                    )

                    if _mgmt_action == "新建配置":
                        with st.form("lf_new_profile_form", clear_on_submit=True):
                            _new_name = st.text_input("配置名称", key="lf_new_name", placeholder="例如：本地 Langfuse 测试环境")
                            _new_host = st.text_input("Host", value="http://localhost:3000", key="lf_new_host")
                            _new_pk = st.text_input("Public Key", key="lf_new_pk")
                            _new_sk = st.text_input("Secret Key", key="lf_new_sk", type="password")
                            if st.form_submit_button("保存"):
                                try:
                                    create_profile(_new_name, _new_host, _new_pk, _new_sk)
                                    st.success(f"已保存: {_new_name}")
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))

                    elif _mgmt_action == "编辑当前" and selected_pid:
                        _ep = load_profile(selected_pid)
                        if _ep:
                            with st.form("lf_edit_profile_form"):
                                _ed_name = st.text_input("配置名称", value=_ep["display_name"], key="lf_ed_name")
                                _ed_host = st.text_input("Host", value=_ep["host"], key="lf_ed_host")
                                _ed_pk = st.text_input("Public Key（留空保持原值）", key="lf_ed_pk")
                                _ed_sk = st.text_input("Secret Key（留空保持原值）", key="lf_ed_sk", type="password")
                                if st.form_submit_button("保存修改"):
                                    try:
                                        pk_val = _ed_pk if _ed_pk else None
                                        sk_val = _ed_sk if _ed_sk else None
                                        update_profile(_ep["profile_id"], _ed_name, _ed_host, pk_val, sk_val)
                                        st.success("已更新")
                                        st.rerun()
                                    except ValueError as e:
                                        st.error(str(e))

                    elif _mgmt_action == "删除当前" and selected_pid:
                        _dp = load_profile(selected_pid)
                        if _dp:
                            st.warning(f"确认删除配置「{_dp['display_name']}」？此操作不可撤销，本地凭据将一并删除。")
                            if st.button("确认删除", key="lf_confirm_delete"):
                                delete_profile(_dp["profile_id"])
                                for k in ("lf_profile_select",):
                                    if k in st.session_state:
                                        del st.session_state[k]
                                st.success(f"已删除: {_dp['display_name']}")
                                st.rerun()

                # 测试连接 + 项目识别
                if selected_pid:
                    _tp = load_profile(selected_pid)
                    _tp_has_keys = has_profile_api_keys(selected_pid)
                    if _tp and _tp_has_keys and st.button("🔗 测试连接并识别项目", key="lf_test_conn"):
                        try:
                            _tp_pk, _tp_sk = get_profile_api_keys(selected_pid)
                            ok, msg = check_connection(_tp["host"], _tp_pk, _tp_sk)
                            if ok:
                                st.success(msg)
                                try:
                                    _proj_info = identify_project_info(_tp["host"], _tp_pk, _tp_sk)
                                    _old_pid = st.session_state.get("_lf_project_info", {}).get("project_id", "")
                                    _on_project_changed(_old_pid, _proj_info.get("project_id", ""))
                                    st.session_state["_lf_project_info"] = _proj_info
                                except Exception as _pe:
                                    st.warning(f"项目识别失败: {_pe}")
                            else:
                                st.error(msg)
                        except ValueError as e:
                            st.error(str(e))
                    elif _tp and not _tp_has_keys:
                        st.warning("凭据缺失，请编辑配置并填入 Key 后再测试连接。")

                # 从 profile 读取凭据
                if selected_pid and has_profile_api_keys(selected_pid):
                    _cp = load_profile(selected_pid)
                    if _cp:
                        langfuse_host = _cp.get("host", "")
                        langfuse_pk, langfuse_sk = get_profile_api_keys(selected_pid)

            else:
                # 临时手动填写模式
                fetch_col1, fetch_col2 = st.columns(2)
                with fetch_col1:
                    langfuse_host = st.text_input("Langfuse 地址", value="http://localhost:3000", key="lf_host")
                    langfuse_pk = st.text_input("Public Key", key="lf_pk")
                with fetch_col2:
                    langfuse_sk = st.text_input("Secret Key", key="lf_sk", type="password")

                # 临时模式也支持测试连接
                if langfuse_host and langfuse_pk and langfuse_sk:
                    if st.button("🔗 测试连接并识别项目", key="lf_test_conn_temp"):
                        try:
                            ok, msg = check_connection(langfuse_host, langfuse_pk, langfuse_sk)
                            if ok:
                                st.success(msg)
                                try:
                                    _proj_info = identify_project_info(langfuse_host, langfuse_pk, langfuse_sk)
                                    _old_pid = st.session_state.get("_lf_project_info", {}).get("project_id", "")
                                    _on_project_changed(_old_pid, _proj_info.get("project_id", ""))
                                    st.session_state["_lf_project_info"] = _proj_info
                                except Exception as _pe:
                                    st.warning(f"项目识别失败: {_pe}")
                            else:
                                st.error(msg)
                        except ValueError as e:
                            st.error(str(e))

            # ── 项目信息 + 同步选项 ──
            if langfuse_host and langfuse_pk and langfuse_sk:
                _proj_info = st.session_state.get("_lf_project_info")

                if _proj_info:
                    _proj_id = _proj_info["project_id"]
                    _proj_stats = get_project_stats(_proj_id)

                    # 项目信息卡片
                    st.markdown("#### 📊 项目信息")
                    pi_col1, pi_col2, pi_col3, pi_col4, pi_col5 = st.columns(5)
                    with pi_col1:
                        st.metric("项目", _proj_info.get("project_name", ""))
                    with pi_col2:
                        st.metric("远端 Trace", _proj_info.get("total_traces", "?"))
                    with pi_col3:
                        _local_count = _proj_stats.get("total_traces_synced", 0)
                        st.metric("本地 Trace", _local_count)
                    with pi_col4:
                        _obs_count = _proj_stats.get("total_observations_synced", 0)
                        st.metric("Observation", _obs_count)
                    with pi_col5:
                        _cache_size = _proj_stats.get("file_size_mb", 0) + _proj_stats.get("obs_file_size_mb", 0)
                        st.metric("本地缓存", f"{_cache_size:.2f} MB")

                    _last_sync = _proj_stats.get("last_sync_at")
                    if _last_sync:
                        st.caption(f"上次同步: {_last_sync}　|　游标: {_proj_stats.get('last_trace_timestamp', '无')}")

                    # Observation 覆盖率
                    _obs_cov = get_observation_coverage(_proj_id)
                    _cov_total = _obs_cov["total_traces"]
                    _cov_has = _obs_cov["traces_with_obs"]
                    _cov_pct = _obs_cov["coverage_pct"]
                    if _cov_total > 0:
                        if _cov_pct < 100:
                            st.warning(
                                f"⚠️ Observation 覆盖率: **{_cov_has} / {_cov_total}**"
                                f"（{_cov_pct}%）。"
                                f"未覆盖的 trace 无法解析 retrieval 结果。"
                                f"请使用「回填历史 Observation」补全。"
                            )
                        else:
                            st.success(f"✅ Observation 覆盖率: {_cov_has} / {_cov_total}（{_cov_pct}%）")

                        # 回填按钮
                        if _cov_pct < 100:
                            if st.button("🔄 回填历史 Observation", key="lf_backfill_obs"):
                                st.session_state["_backfilling"] = True
                                bf_status = st.status("正在回填历史 Observation...", expanded=True)
                                bf_progress = st.progress(0, text="开始回填...")
                                bf_detail = st.empty()

                                def _on_backfill(phase, done, total, new_obs, errors):
                                    if phase == "starting":
                                        bf_progress.progress(0, text="正在扫描...")
                                    elif phase == "backfilling" and total > 0:
                                        pct = min(done / total, 1.0)
                                        bf_progress.progress(pct,
                                            text=f"回填进度: {done}/{total} | 新增 obs: {new_obs}")
                                        bf_detail.caption(f"错误: {errors}" if errors else "")
                                    elif phase == "done":
                                        bf_progress.progress(1.0, text="回填完成")

                                try:
                                    bf_result = backfill_observations(
                                        _proj_id, langfuse_host, langfuse_pk, langfuse_sk,
                                        progress_callback=_on_backfill,
                                    )
                                    bf_status.update(label="回填完成", state="complete", expanded=False)
                                    st.success(
                                        f"回填完成：扫描 {bf_result['traces_backfilled']} 条 trace，"
                                        f"新增 {bf_result['new_observations']} 条 observation"
                                        f"（{bf_result['errors']} 个错误）"
                                        f" | 耗时 {bf_result['elapsed']:.1f}s"
                                    )
                                    # 更新逻辑快照
                                    try:
                                        from langfuse_project import _update_current_snapshot
                                        _update_current_snapshot(_proj_id)
                                    except Exception:
                                        pass
                                    st.rerun()
                                except Exception as e:
                                    bf_status.update(label="回填失败", state="error")
                                    st.error(f"回填失败: {e}")
                                finally:
                                    st.session_state["_backfilling"] = False

                    # 检查同项目多 Key
                    _all_projects = list_projects()
                    _same_host_projects = [p for p in _all_projects
                                          if p.get("host", "").rstrip("/").lower() == langfuse_host.strip().rstrip("/").lower()
                                          and p.get("project_id") != _proj_id]
                    if _same_host_projects:
                        st.info(f"同一 Langfuse 服务器上还有 {len(_same_host_projects)} 个其他已注册项目")

                    # 同步选项
                    st.markdown("#### 🔄 同步选项")
                    sync_mode = st.radio(
                        "同步方式",
                        ["仅同步新增（推荐）", "按时间范围导入", "首次全量导入"],
                        key="lf_sync_mode",
                    )

                    from_ts = None
                    max_pages = 50

                    if sync_mode == "按时间范围导入":
                        ts_col1, ts_col2 = st.columns(2)
                        with ts_col1:
                            _from_date = st.date_input("起始日期", key="lf_from_date")
                        with ts_col2:
                            _from_time = st.time_input("起始时间", value=None, key="lf_from_time")
                        if _from_date:
                            _dt = datetime.combine(_from_date, _from_time or datetime.min.time())
                            from_ts = _dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                    elif sync_mode == "首次全量导入":
                        max_pages = st.number_input(
                            "最大页数", min_value=1, max_value=500, value=50,
                            key="lf_max_pages",
                            help="每页 50 条 trace，50 页 = 2500 条",
                        )

                    fetch_limit = st.number_input("每页 trace 数", min_value=1, max_value=500, value=50, key="lf_limit")
                    _force_full = (sync_mode == "首次全量导入")

                    # 同步按钮
                    if st.button("🚀 开始同步", key="fetch_traces",
                                 disabled=st.session_state.get("_fetching", False)):
                        st.session_state["_fetching"] = True
                        fetch_status = st.status(f"正在同步 {langfuse_host} ...", expanded=True)
                        progress_bar = st.progress(0, text="连接中...")
                        detail_text = st.empty()

                        try:
                            def _on_sync_progress(phase, new, skipped, pages, total):
                                if phase == "connecting":
                                    progress_bar.progress(0, text="连接中...")
                                elif phase == "syncing":
                                    if total and total > 0:
                                        pct = min(pages * fetch_limit / total, 1.0)
                                    else:
                                        pct = 0
                                    progress_bar.progress(pct, text=f"已同步 {new} 条新 trace（跳过 {skipped} 条重复）")
                                    detail_text.caption(f"已完成 {pages} 页")
                                elif phase == "done":
                                    progress_bar.progress(1.0, text="同步完成")

                            result = incremental_sync(
                                _proj_id, langfuse_host, langfuse_pk, langfuse_sk,
                                limit=fetch_limit, max_pages=max_pages,
                                from_timestamp=from_ts,
                                progress_callback=_on_sync_progress,
                                force_full=_force_full,
                            )

                            fetch_status.update(label="同步完成", state="complete", expanded=False)
                            _obs_new = result.get("new_observations", 0)
                            _snap_created = result.get("snapshot_created", False)
                            st.success(
                                f"同步完成：**{result['new_traces']}** 条新 trace"
                                f" + **{_obs_new}** 条 observation"
                                f"（跳过 {result['skipped']} 条重复）"
                                f" | {result['pages']} 页 | 耗时 {result['elapsed']:.1f}s"
                            )
                            if _snap_created:
                                st.info("✅ 同步缓存已更新")
                            elif result["new_traces"] == 0 and _obs_new == 0:
                                st.info("无新增数据，复用当前缓存快照")
                            if _obs_new == 0 and result["new_traces"] > 0:
                                st.warning("未获取到 observation 数据。同步快照将为索引快照，不可解析为评测样本。请检查 API 权限。")

                            # 注册项目
                            register_project(
                                _proj_id, _proj_info.get("project_name", ""),
                                langfuse_host, _proj_info.get("key_masked", ""),
                            )

                            # 更新 stats（同步不切换项目，保留 _use_frozen_source）
                            st.session_state["_lf_project_info"] = _proj_info
                            st.rerun()

                        except Exception as e:
                            fetch_status.update(label="同步失败", state="error")
                            st.error(f"同步失败: {e}")
                        finally:
                            st.session_state["_fetching"] = False

                    # ── 当前动态缓存 ──
                    if _proj_id:
                        _cache_stats = get_current_cache_stats(_proj_id)
                        _cs_trace = _cache_stats.get("trace_count", 0)
                        _cs_obs = _cache_stats.get("observation_count", 0)
                        _cs_has_obs = _cache_stats.get("has_observations", False)
                        _cs_synced = _cache_stats.get("last_sync_at", "")
                        _cs_size_mb = round(_cache_stats.get("file_size_bytes", 0) / (1024 * 1024), 2)

                        st.markdown("**📋 当前动态缓存**")
                        _cache_info = f"{_cs_trace} traces / {_cs_obs} obs | {_cs_size_mb} MB"
                        if _cs_synced:
                            _cache_info += f" | 上次同步: {_cs_synced}"
                        if _cs_has_obs:
                            _cache_info += " | ✅ 含检索证据"
                        else:
                            _cache_info += " | ⚠️ 不含检索证据"
                        st.caption(_cache_info)

                        # 冻结按钮（可选留档）
                        _can_freeze = _cs_trace > 0 and _cs_has_obs
                        if st.button(
                            "📦 冻结当前缓存（可选留档）",
                            key="lf_freeze_cache",
                            disabled=not _can_freeze,
                            help="将当前动态缓存复制为不可变历史版本，供未来手动回退。正常解析不需要此步骤。"
                        ):
                            try:
                                with st.spinner("正在冻结当前缓存..."):
                                    _frozen = create_frozen_snapshot(_proj_id)
                                st.success(f"✅ 已冻结: `{_frozen['snapshot_id']}`（仅供历史复查）")
                                st.rerun()
                            except Exception as e:
                                st.error(f"冻结失败: {e}")

                        if not _can_freeze and _cs_trace > 0:
                            st.caption("💡 冻结需要 observation 数据。请检查 API 权限。")

                        # 历史冻结缓存（默认折叠）
                        _snap_list = list_snapshots(_proj_id)
                        _frozen_all = [
                            s for s in _snap_list
                            if s.get("snapshot_type") == "frozen"
                        ]
                        _frozen_evidence = [
                            s for s in _frozen_all
                            if s.get("has_observations", False)
                        ]
                        if _frozen_evidence:
                            with st.expander(f"📖 历史冻结缓存（{len(_frozen_evidence)} 个，仅供手动回退）", expanded=False):
                                for _of in sorted(_frozen_evidence, key=lambda s: s.get("created_at", ""), reverse=True):
                                    _of_id = _of.get("snapshot_id", "")
                                    _of_trace = _of.get("trace_count", 0)
                                    _of_obs = _of.get("observation_count", 0)
                                    _of_created = _of.get("created_at", "")[:16].replace("T", " ")
                                    _refs = _find_snapshot_references(_proj_id, _of_id)
                                    _ref_note = f" | 被 {len(_refs)} 个产物引用" if _refs else ""
                                    st.caption(
                                        f"`{_of_id}` | {_of_trace}t/{_of_obs}obs"
                                        f" | {_of_created}{_ref_note}"
                                    )

                    # ── 清理旧导出文件 ──
                    _cleanup_candidates = list_cleanup_candidates()
                    if _cleanup_candidates:
                        st.markdown("#### 🧹 清理旧导出文件")
                        _total_old_size = sum(c["size_mb"] for c in _cleanup_candidates)
                        st.caption(
                            f"发现 {len(_cleanup_candidates)} 个旧版全量导出文件"
                            f"（共 {_total_old_size:.1f} MB），"
                            f"已迁移至项目增量存储后可清理。"
                        )
                        with st.expander("查看清理候选", expanded=False):
                            for c in _cleanup_candidates:
                                st.caption(f"`{c['name']}` — {c['size_mb']} MB — {c['mtime']}")

                            _to_clean = st.multiselect(
                                "选择要删除的文件",
                                options=[c["path"] for c in _cleanup_candidates],
                                format_func=lambda x: Path(x).name,
                                key="lf_cleanup_select",
                            )
                            if _to_clean:
                                st.warning(f"将删除 {len(_to_clean)} 个文件，此操作不可撤销！")
                                if st.button("🗑️ 确认删除选中文件", key="lf_cleanup_confirm"):
                                    deleted, failed = cleanup_files(_to_clean)
                                    st.success(f"已删除 {deleted} 个文件" + (f"，{failed} 个失败" if failed else ""))
                                    st.rerun()

                else:
                    st.info("请点击「测试连接并识别项目」以开始。")

        # Step 2: 解析当前缓存
        st.divider()
        st.markdown("**第二步：解析当前缓存**")

        # ── 权威来源：当前动态缓存 ──
        _proj_id_for_sources = st.session_state.get("_lf_project_info", {}).get("project_id")
        _current_cache_stats = get_current_cache_stats(_proj_id_for_sources) if _proj_id_for_sources else {}
        _cc_has_obs = _current_cache_stats.get("has_observations", False)
        _cc_trace = _current_cache_stats.get("trace_count", 0)

        # ── 统一初始化解析变量 ──
        selected_source = None
        selected_name = None
        selected_path = None
        source_type = None
        is_gzip = False
        has_obs = False
        _can_parse = False

        # 已解析结果来源检测
        _parsed_samples = st.session_state.get("samples")
        _parsed_summary = st.session_state.get("summary", {})
        _parsed_source_type = _parsed_summary.get("langfuse_source_type", "")

        # ── 默认路径：当前动态缓存 ──
        if _proj_id_for_sources and _cc_trace > 0 and _cc_has_obs:
            source_type = "current_cache"
            _can_parse = True
            has_obs = True

            st.markdown(
                f"**当前数据源：** 当前动态缓存"
                f" — `{_proj_id_for_sources[:20]}...`"
            )
            _info_parts = [
                f"{_cc_trace} traces / {_current_cache_stats.get('observation_count', 0)} obs",
                "✅ 含检索证据",
            ]
            st.caption(" | ".join(_info_parts))

            # 已解析结果来源检测
            if _parsed_samples and _parsed_source_type:
                if _parsed_source_type != "current_cache":
                    st.warning(
                        f"⚠️ 当前展示的是历史解析结果（来源类型：`{_parsed_source_type}`）。"
                        f"点击下方按钮解析当前缓存。"
                    )
                elif _parsed_source_type == "current_cache":
                    # 检查缓存是否在上次解析后被更新（fingerprint 变化）
                    _parsed_fp = _parsed_summary.get("source_file_fingerprint", "")
                    if _parsed_fp:
                        try:
                            from langfuse_project import _traces_path as __tp, _obs_path as __op
                            _cur_trace_fp = compute_file_fingerprint(__tp(_proj_id_for_sources))
                            _cur_obs_fp = compute_file_fingerprint(__op(_proj_id_for_sources))
                            _cur_fp = f"{_cur_trace_fp}|{_cur_obs_fp}" if _cur_trace_fp else ""
                            if _cur_fp and _parsed_fp != _cur_fp:
                                st.warning(
                                    "⚠️ 当前解析结果不是最新动态缓存（缓存已更新）。"
                                    "请点击「解析当前缓存」刷新。"
                                )
                        except Exception:
                            pass

        elif _proj_id_for_sources and _cc_trace > 0 and not _cc_has_obs:
            st.warning("当前缓存不含 observation 数据。请检查 API 权限后重新同步。")
        elif _proj_id_for_sources:
            st.info("当前缓存无数据。请先点击「同步新增」获取 trace 数据。")
        else:
            st.info("请先在上方选择 Langfuse 连接并同步数据。")

        # ── 历史冻结缓存选择（可选，折叠） ──
        parseable_sources = list_parseable_sources(_proj_id_for_sources)
        _frozen_evidence = [
            s for s in parseable_sources
            if s["source_type"] == "evidence_snapshot"
        ]
        _legacy_sources = [
            s for s in parseable_sources
            if s["source_type"] == "legacy_raw"
        ]

        if _frozen_evidence or _legacy_sources:
            _frozen_labels = []
            _frozen_sources = []
            for _fs in _frozen_evidence:
                _fid = _fs.get("snapshot_id", "")
                _ft = _fs.get("trace_count", 0)
                _fo = _fs.get("observation_count", 0)
                _frozen_labels.append(f"📦 冻结: {_fid} ({_ft}t/{_fo}obs)")
                _frozen_sources.append(_fs)
            for _ls in _legacy_sources:
                _frozen_labels.append(f"📁 {_ls['label']} ({_ls['size_mb']} MB)")
                _frozen_sources.append(_ls)

            with st.expander(f"🔄 切换到历史缓存（{len(_frozen_labels)} 个可选）", expanded=False):
                if _frozen_labels:
                    _sel_frozen_label = st.selectbox(
                        "选择历史数据源",
                        _frozen_labels,
                        key="frozen_source_select",
                    )
                    _sel_frozen_idx = _frozen_labels.index(_sel_frozen_label)
                    _sel_frozen = _frozen_sources[_sel_frozen_idx]

                    if st.button("使用此历史数据源解析", key="use_frozen_btn"):
                        # 切换到历史数据源
                        selected_source = _sel_frozen
                        selected_name = _sel_frozen.get("source_id", "")
                        selected_path = _sel_frozen.get("path", "")
                        source_type = _sel_frozen.get("source_type", "")
                        is_gzip = selected_path.endswith(".gz")
                        has_obs = _sel_frozen.get("has_observations", False)
                        _can_parse = True
                        st.session_state["_use_frozen_source"] = _sel_frozen
                        # 标记为当前 session 的主动选择，防止 startup 清理误删
                        st.session_state["_frozen_source_just_set"] = True
                        st.rerun()

        # 检查是否用户选择了历史数据源
        _use_frozen = st.session_state.get("_use_frozen_source")
        if _use_frozen and not _can_parse:
            selected_source = _use_frozen
            selected_name = _use_frozen.get("source_id", "")
            selected_path = _use_frozen.get("path", "")
            source_type = _use_frozen.get("source_type", "")
            is_gzip = selected_path.endswith(".gz")
            has_obs = _use_frozen.get("has_observations", False)
            _can_parse = True
            st.info(f"📌 使用历史数据源: `{_use_frozen.get('snapshot_id', '') or _use_frozen.get('source_id', '')}`")
            if st.button("↩️ 切回当前缓存", key="back_to_current"):
                st.session_state.pop("_use_frozen_source", None)
                st.rerun()

        # ── 解析按钮与执行逻辑 ──
        if _can_parse:
            _parse_btn_label = "解析当前缓存" if source_type == "current_cache" else "开始解析"
            if st.button(
                _parse_btn_label, type="primary", key="parse_btn",
                disabled=st.session_state.get("_parsing", False),
            ):
                st.session_state["_parsing"] = True
                parse_status = st.status("正在解析...", expanded=True)
                progress_bar = st.progress(0, text="准备中...")
                detail_text = st.empty()
                t0 = time.time()

                # 准备解析文件
                actual_parse_path = None
                temp_jsonl = None
                try:
                    if source_type == "current_cache":
                        # 当前动态缓存：直接合并 traces + observations
                        progress_bar.progress(0, text="正在合并当前缓存...")
                        actual_parse_path = export_current_cache_as_jsonl(_proj_id_for_sources)
                    elif source_type == "evidence_snapshot" and selected_source.get("snapshot_id"):
                        # 冻结快照：合并 traces + observations
                        progress_bar.progress(0, text="正在解冻快照...")
                        actual_parse_path = export_snapshot_as_jsonl(
                            selected_source["project_id"],
                            selected_source["snapshot_id"],
                        )
                    elif is_gzip:
                        # 通用 gzip 解压
                        progress_bar.progress(0, text="正在解压 gzip 文件...")
                        import gzip as _gzip
                        temp_jsonl = Path(selected_path).with_suffix("")
                        with _gzip.open(selected_path, "rt", encoding="utf-8") as fin, \
                             temp_jsonl.open("w", encoding="utf-8") as fout:
                            for line in fin:
                                fout.write(line)
                        actual_parse_path = temp_jsonl
                    else:
                        actual_parse_path = Path(selected_path)

                    def _on_progress(phase, current, total, traces, retrieval):
                        if phase == "counting":
                            progress_bar.progress(0, text="正在统计行数...")
                            detail_text.caption("正在预扫描文件...")
                        elif phase == "reading" and total > 0:
                            pct = min(current / total, 1.0)
                            progress_bar.progress(pct, text=f"正在读取 JSONL: {current}/{total}")
                            detail_text.caption(f"已识别 trace: {traces} | retrieval: {retrieval}")
                        elif phase == "building":
                            if total > 0:
                                pct = min(current / total, 1.0)
                                progress_bar.progress(pct, text=f"正在构建样本: {current}/{total}")
                            detail_text.caption(f"已识别 trace: {traces}")
                        elif phase == "backfilling":
                            progress_bar.progress(0.95, text="正在回填参考答案...")
                        elif phase == "saving":
                            progress_bar.progress(0.99, text="正在写入文件...")

                    samples, summary = parse_langfuse_jsonl(
                        actual_parse_path, progress_callback=_on_progress,
                    )

                    _on_progress("saving", 0, 0, 0, 0)

                    # ── 隔离路径：按 project_id 写入 ──
                    _src_pid = _proj_id_for_sources or ""
                    _src_snap = selected_source.get("snapshot_id", "") if selected_source else ""
                    _src_sid = selected_source.get("source_id", "") if selected_source else ""
                    output_path, summary_path = get_processed_paths(
                        source_type or "current_cache", project_id=_src_pid,
                        snapshot_id=_src_snap, source_id=_src_sid,
                    )
                    if output_path.exists():
                        st.info(f"将覆盖已有解析结果：`{output_path}`")

                    # ── 计算文件指纹 ──
                    _trace_fp = ""
                    _obs_fp = ""
                    if source_type == "current_cache":
                        from langfuse_project import _traces_path, _obs_path
                        _trace_fp = compute_file_fingerprint(_traces_path(_src_pid))
                        _obs_fp = compute_file_fingerprint(_obs_path(_src_pid))

                    # ── 构建 provenance（落盘前固化到 summary 和每个 sample） ──
                    _provenance = {
                        "langfuse_project_id": _src_pid,
                        "langfuse_snapshot_id": _src_snap,
                        "langfuse_source_type": source_type or "",
                        "source_file": str(actual_parse_path) if actual_parse_path else "",
                        "cache_last_sync_at": _current_cache_stats.get("last_sync_at", ""),
                        "cache_trace_count": _current_cache_stats.get("trace_count", 0),
                        "cache_observation_count": _current_cache_stats.get("observation_count", 0),
                        "source_file_fingerprint": f"{_trace_fp}|{_obs_fp}" if _trace_fp else "",
                    }
                    summary.update(_provenance)
                    for _s in samples:
                        _s.update(_provenance)
                        # 确保每条 sample 有真实 trace_id（非 batch_qa_*）
                        # trace_id 由 parser 从原始数据提取，此处不覆盖
                    full_summary = save_results(samples, summary, output_path, summary_path)

                    # ── 更新 processed run index ──
                    _run_ids_seen = set()
                    for _s in samples:
                        _rid = _s.get("run_id", "")
                        if not _rid:
                            _uid = _s.get("user_id", "")
                            if _uid.startswith("rag_eval:"):
                                _parts = _uid.split(":", 2)
                                if len(_parts) == 3:
                                    _rid = _parts[1]
                        if _rid and _rid not in _run_ids_seen:
                            _run_ids_seen.add(_rid)
                            update_run_index(
                                _rid,
                                str(output_path),
                                str(summary_path),
                                project_id=_src_pid,
                                source_type=source_type or "",
                                fingerprint=full_summary.get("source_file_fingerprint", ""),
                            )

                    for _s in samples:
                        _s.pop("observations", None)
                    st.session_state["samples"] = samples
                    st.session_state["summary"] = full_summary
                    # 解析完成后重置页码，确保显示最新数据
                    st.session_state["sample_page"] = 1

                    # 标记快照已解析（仅冻结快照）
                    if source_type == "evidence_snapshot" and selected_source.get("snapshot_id"):
                        mark_snapshot_parsed(
                            selected_source["project_id"],
                            selected_source["snapshot_id"],
                        )

                    elapsed = time.time() - t0
                    progress_bar.progress(1.0, text="解析完成")
                    parse_status.update(label="解析完成", state="complete", expanded=False)
                    _record_rss("JSONL 解析完成")

                    bs = summary.get("backfill_stats") or {}
                    bf = bs.get("backfilled", 0)
                    already = bs.get("already_has", 0)
                    bad = summary.get("bad_line_count", 0)
                    st.success(
                        f"解析完成：**{len(samples)}** 条 Trace | "
                        f"**{summary.get('total_retrieval_results', 0)}** 条 retrieval 结果 | "
                        f"耗时 {elapsed:.1f}s"
                    )
                    if bf > 0:
                        st.success(f"参考答案回填：**{bf}** 条匹配到题目库")
                    if already > 0:
                        st.info(f"**{already}** 条样本本身已带参考答案")
                    no_ref = bs.get("total", 0) - bf - already
                    if no_ref > 0:
                        st.warning(f"**{no_ref}** 条样本未匹配到题目库")
                    if bad > 0:
                        bad_reasons = summary.get("bad_lines", [])
                        reason_summary = "; ".join(
                            f"行 {b['line']}: {b['error'][:40]}" for b in bad_reasons[:5]
                        )
                        suffix = "..." if len(bad_reasons) > 5 else ""
                        st.warning(f"跳过 {bad} 行（原因：{reason_summary}{suffix}）")

                    st.rerun()

                except Exception as e:
                    parse_status.update(label="解析失败", state="error")
                    st.error(f"解析失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    st.session_state["_parsing"] = False
                    # 清理临时解压文件
                    if temp_jsonl and temp_jsonl.exists():
                        try:
                            temp_jsonl.unlink()
                        except Exception:
                            pass
                    # 清理 current_cache 临时文件
                    if source_type == "current_cache" and actual_parse_path and actual_parse_path.exists():
                        try:
                            actual_parse_path.unlink()
                        except Exception:
                            pass

    # --- Sample display section ---
    if not samples:
        st.info("请在上方「数据导入」区域上传或拉取 Langfuse 数据，然后点击「开始解析」")
    else:
        _src_type = summary.get("langfuse_source_type", "")
        _src_snap = summary.get("langfuse_snapshot_id", "")
        _src_pid = summary.get("langfuse_project_id", "")
        input_file = summary.get("input_file") or ""
        output_file = summary.get("output_file") or ""

        # 显示实际数据来源（区分动态缓存 / 冻结快照 / 旧版文件）
        if _src_type == "current_cache":
            _src_label = f"当前动态缓存（项目 `{_src_pid[:20]}...`）"
        elif _src_type == "evidence_snapshot" and _src_snap:
            _src_label = f"冻结快照 `{_src_snap}`"
        elif input_file:
            _src_label = f"`{Path(input_file).name}`"
        else:
            _src_label = "未知来源"

        _output_label = f" → 解析结果: `{output_file}`" if output_file else ""
        st.caption(f"数据来源: {_src_label}{_output_label}")

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

        # ── 排序控制 ──
        _sort_col1, _sort_col2 = st.columns([1, 2])
        with _sort_col1:
            _sort_mode = st.radio(
                "排序方式",
                ["最新优先", "最早优先"],
                key="sample_sort_mode",
                horizontal=True,
            )

        # ── 搜索筛选 ──
        search = st.text_input("搜索问题内容", "", key="sample_search")

        # ── 页码重置：排序或搜索变化时自动回到第 1 页 ──
        _prev_sort = st.session_state.get("_prev_sample_sort")
        _prev_search = st.session_state.get("_prev_sample_search", "")
        if _sort_mode != _prev_sort or search != _prev_search:
            st.session_state["sample_page"] = 1
            st.session_state["_prev_sample_sort"] = _sort_mode
            st.session_state["_prev_sample_search"] = search

        # 排序 → 筛选 → 分页（顺序不可颠倒）
        _sort_newest_first = (_sort_mode == "最新优先")

        from parser import _parse_ts_for_sort
        _with_ts = []
        _no_ts = []
        for s in samples:
            ts = s.get("trace_timestamp") or s.get("earliest_obs_time")
            dt = _parse_ts_for_sort(ts)
            if dt is None:
                _no_ts.append(s)
            else:
                _with_ts.append((dt, s.get("trace_id", ""), s))

        if _sort_newest_first:
            _with_ts.sort(key=lambda x: (-x[0].timestamp(), x[1]))
        else:
            _with_ts.sort(key=lambda x: (x[0].timestamp(), x[1]))
        _no_ts.sort(key=lambda x: x.get("trace_id", ""))

        sorted_samples = [item[2] for item in _with_ts] + _no_ts

        if search:
            filtered = [s for s in sorted_samples if search.lower() in (s.get("question") or "").lower()]
        else:
            filtered = sorted_samples

        # ── 分页 ──
        _SAMPLE_PAGE_SIZE = 20
        _total_pages = max(1, (len(filtered) + _SAMPLE_PAGE_SIZE - 1) // _SAMPLE_PAGE_SIZE)
        if _total_pages > 1:
            _pg_col1, _pg_col2, _ = st.columns([1, 1, 4])
            with _pg_col1:
                _page_num = st.number_input(
                    "页码", min_value=1, max_value=_total_pages, value=1, key="sample_page",
                )
            with _pg_col2:
                st.caption(f"共 {_total_pages} 页（{len(filtered)} 条样本）")
            _start = (_page_num - 1) * _SAMPLE_PAGE_SIZE
            page_items = filtered[_start:_start + _SAMPLE_PAGE_SIZE]
        else:
            page_items = filtered

        for i, sample in enumerate(page_items):
            question = sample.get("question") or "(无问题)"
            retrieval_calls = sample.get("retrieval_calls") or []
            retrieval_results = sample.get("retrieval_results") or []
            retrieval_call_count = sample.get("retrieval_call_count") or len(retrieval_calls)
            trace_id = sample.get("trace_id", "")
            eval_track = sample.get("evaluation_track", "")

            # ── 时间显示 ──
            from parser import _parse_ts_for_sort
            _ts_raw = sample.get("trace_timestamp") or sample.get("earliest_obs_time")
            _ts_dt = _parse_ts_for_sort(_ts_raw)
            if _ts_dt is not None:
                _ts_local = _ts_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                _ts_local = "时间未知"

            # ── 检索状态分类显示（优先使用 retrieval_calls） ──
            if not trace_id or trace_id.startswith("batch_qa_"):
                _retrieval_badge = "⚠️ 未关联 trace"
            elif retrieval_calls:
                _total_results = sum(len(c.get("results") or []) for c in retrieval_calls)
                _retrieval_badge = f"🔍 {retrieval_call_count} 次检索, {_total_results} 条结果"
            elif retrieval_results:
                _retrieval_badge = f"检索 {len(retrieval_results)} 条"
            elif sample.get("retrieval_query"):
                _retrieval_badge = "检索 0 条（trace 已关联，无命中）"
            else:
                _retrieval_badge = "无检索数据"

            _track_badge = ""
            if eval_track == "chunk_exact":
                _track_badge = " 🎯"
            elif eval_track == "retrieval":
                _track_badge = " 🔍"

            _q_short = question[:50] + "..." if len(question) > 50 else question
            with st.expander(
                f"{_ts_local} | {_q_short} | {_retrieval_badge}{_track_badge} | {trace_id[:12]}..."
            ):
                # 时间和 trace_id 信息栏
                st.caption(f"🕐 {_ts_local}　|　trace_id: `{trace_id}`")

                st.markdown("**问题**")
                st.code(sample.get("question") or "(无)", language=None)

                # chunk_exact 题显示绑定信息
                if eval_track == "chunk_exact":
                    _exp_seg = sample.get("expected_segment_id", "")
                    _exp_hash = sample.get("expected_content_hash", "")
                    if _exp_seg or _exp_hash:
                        st.markdown("**预期绑定**")
                        if _exp_seg:
                            st.caption(f"expected_segment_id: `{_exp_seg}`")
                        if _exp_hash:
                            st.caption(f"expected_content_hash: `{_exp_hash[:16]}...`")

                # ── 检索结果展示（优先 retrieval_calls，回退 retrieval_results） ──
                if not trace_id or trace_id.startswith("batch_qa_"):
                    st.info("此样本未关联真实 Langfuse trace，无法获取检索结果。请先在「批量提问」中执行并同步 trace。")
                elif retrieval_calls:
                    # 多检索模式：按 call 展示
                    st.markdown(f"**检索调用 ({retrieval_call_count} 次)**")
                    for call in retrieval_calls:
                        _call_order = call.get("order", "?")
                        _call_query = call.get("query") or "(无 query)"
                        _call_latency = call.get("latency_ms")
                        _call_results = call.get("results") or []
                        _latency_str = f"{_call_latency}ms" if _call_latency is not None else "N/A"
                        with st.expander(
                            f"检索 #{_call_order}: {_call_query[:60]} "
                            f"({len(_call_results)} 条, {_latency_str})"
                        ):
                            st.caption(
                                f"observation: `{call.get('observation_id', '')[:16]}` | "
                                f"延迟: {_latency_str} | "
                                f"时间: {call.get('start_time', '')[:19]}"
                            )
                            for r in _call_results:
                                _title = r.get("title") or "(无标题)"
                                _score = r.get("score")
                                _content = r.get("content") or ""
                                _score_str = f" (score: {_score})" if _score is not None else ""
                                with st.expander(f"{_title}{_score_str}"):
                                    st.text((_content or "(无内容)")[:2000])
                elif retrieval_results:
                    # 单检索模式（向后兼容）
                    st.markdown(f"**检索查询 (retrieval_query)**")
                    st.code(sample.get("retrieval_query") or "(无)", language=None)
                    st.markdown(f"**检索结果 ({len(retrieval_results)} 条)**")
                    for r in retrieval_results:
                        title = r.get("title") or "(无标题)"
                        score = r.get("score")
                        content = r.get("content") or ""
                        score_str = f" (score: {score})" if score is not None else ""
                        with st.expander(f"{title}{score_str}"):
                            st.text((content or "(无内容)")[:2000])
                elif sample.get("retrieval_query"):
                    st.info("trace 已关联但 Dify 未返回检索结果（embedding 未命中或检索配置为空）。")
                else:
                    st.info("此样本缺少 retrieval_query，可能是旧数据或非检索类题目。")

                st.markdown(f"**LLM 模型**: `{sample.get('llm_model') or 'N/A'}`")

                st.markdown("**LLM Input**")
                llm_input = sample.get("llm_input")
                if llm_input:
                    _input_str = json.dumps(llm_input, ensure_ascii=False)
                    if len(_input_str) > 2000:
                        st.code(_input_str[:2000] + "\n... (已截断，共 " + str(len(_input_str)) + " 字符)", language="json")
                    else:
                        st.json(llm_input)
                else:
                    st.caption("(无)")

                st.markdown("**LLM Output**")
                llm_output = sample.get("llm_output")
                if llm_output:
                    _output_str = json.dumps(llm_output, ensure_ascii=False)
                    if len(_output_str) > 2000:
                        st.code(_output_str[:2000] + "\n... (已截断，共 " + str(len(_output_str)) + " 字符)", language="json")
                    else:
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
                })


def build_judge_plan(filtered_samples, existing_results_map, mode):
    """根据模式和历史结果，计算评测执行计划。

    Args:
        filtered_samples: 经轨道筛选后的样本列表
        existing_results_map: dict[trace_id -> result_dict]，已有评测结果
        mode: "quick_test" | "incremental" | "retry_failed" | "force_all"

    Returns:
        dict: samples, new_count, retry_count, prescreen_count,
              llm_count, success_count, total_filtered,
              selected_sample_preview (quick_test only)
    """
    from judge import classify_evaluation_track, pre_screen, compute_content_hash

    total_filtered = len(filtered_samples)

    # 先统计所有样本的分类（无论模式）
    selected = []
    new_count = 0
    retry_count = 0
    success_count = 0
    prescreen_count = 0
    selected_sample_preview = None

    # 统计已成功数（所有模式都需要）
    for s in filtered_samples:
        tid = s.get("trace_id")
        existing = existing_results_map.get(tid)
        if existing and "error" not in existing:
            success_count += 1

    if mode == "quick_test":
        # 找第一条：待评 且 能实际进入 LLM Judge（pre_screen 返回 None）
        for s in filtered_samples:
            tid = s.get("trace_id")
            existing = existing_results_map.get(tid)
            if existing and "error" not in existing:
                continue  # 已成功，跳过
            ps = pre_screen(s)
            if ps is not None:
                continue  # 规则预筛，不消耗 LLM，跳过
            # 找到符合条件的样本
            selected = [s]
            track = classify_evaluation_track(s)
            selected_sample_preview = {
                "question": (s.get("question") or "(无问题)")[:60],
                "trace_id_suffix": (tid or "")[-8:],
                "evaluation_track": track,
            }
            if existing and "error" in existing:
                retry_count = 1
            else:
                new_count = 1
            break

    elif mode == "incremental":
        for s in filtered_samples:
            tid = s.get("trace_id")
            existing = existing_results_map.get(tid)
            if existing and "error" not in existing:
                continue  # 已成功，跳过
            selected.append(s)
            if existing and "error" in existing:
                retry_count += 1
            else:
                new_count += 1

    elif mode == "retry_failed":
        for s in filtered_samples:
            tid = s.get("trace_id")
            existing = existing_results_map.get(tid)
            if existing and "error" in existing:
                selected.append(s)
                retry_count += 1

    elif mode == "force_all":
        selected = list(filtered_samples)
        for s in filtered_samples:
            tid = s.get("trace_id")
            existing = existing_results_map.get(tid)
            if existing and "error" not in existing:
                pass  # success_count already counted above
            elif existing and "error" in existing:
                retry_count += 1
            else:
                new_count += 1

    # 计算 prescreen 数和 LLM 调用数（含内容去重）
    content_seen = {}
    llm_count = 0
    for s in selected:
        ps = pre_screen(s)
        if ps is not None:
            prescreen_count += 1
            continue
        ch = compute_content_hash(s)
        if ch not in content_seen:
            content_seen[ch] = True
            llm_count += 1

    return {
        "samples": selected,
        "new_count": new_count,
        "retry_count": retry_count,
        "prescreen_count": prescreen_count,
        "llm_count": llm_count,
        "success_count": success_count,
        "total_filtered": total_filtered,
        "selected_sample_preview": selected_sample_preview,
    }


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
        from judge import classify_evaluation_track, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE, TRACK_CHUNK_EXACT

        from collections import Counter
        track_counts = Counter(classify_evaluation_track(s) for s in samples)

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
        _track_metric_items = [
            (TRACK_RETRIEVAL, "可评测检索题", "有金标准证据，可计算 TopK Hit"),
            (TRACK_STRICT_QA, "严格问答", "有 reference_answer，可评判回答正确性"),
            (TRACK_GROUNDED_QA, "合理性问答", "无参考答案，基于检索内容判断合理性"),
            (TRACK_NOT_EVALUABLE, "缺少金标准", "检索评测题但缺少金标准证据"),
            (TRACK_CHUNK_EXACT, "Chunk 精确匹配", "按 segment_id / content_hash 纯机器判定"),
        ]
        _active_tracks = [(t, label, desc) for t, label, desc in _track_metric_items if track_counts[t] > 0]
        if _active_tracks:
            _cols = st.columns(len(_active_tracks))
            for col, (track, label, desc) in zip(_cols, _active_tracks):
                with col:
                    st.metric(label, track_counts[track], help=desc)

        # 未知轨道告警
        _known_tracks = {t for t, _, _ in _track_metric_items}
        _unknown_tracks = {t for t in track_counts if t not in _known_tracks and track_counts[t] > 0}
        if _unknown_tracks:
            st.warning(f"发现未知评测轨道: {', '.join(_unknown_tracks)}，这些样本不会参与正式指标统计。")

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

        # 补齐旧版 chunk_exact 结果缺失的 TopK 字段
        for r in existing_results_map.values():
            backfill_chunk_exact_topk(r, _sample_by_tid)

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
            if track_counts[TRACK_CHUNK_EXACT] > 0:
                track_filter_options.append(f"仅 Chunk 精确匹配（{track_counts[TRACK_CHUNK_EXACT]} 条）")

            # 清理废弃的 session_state 键
            for stale_key in ("debug_limit", "max_samples", "eval_mode"):
                st.session_state.pop(stale_key, None)

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
            elif "Chunk 精确匹配" in track_filter:
                filtered_samples = [s for s in samples if classify_evaluation_track(s) == TRACK_CHUNK_EXACT]
            else:
                filtered_samples = samples

            filtered_count = len(filtered_samples)
            st.caption(f"筛选后样本数：**{filtered_count}** 条")

            # === 第二层：高级选项 ===
            with st.expander("高级选项", expanded=False):
                show_debug = st.checkbox("显示 Judge Prompt 和原始响应", key="show_debug")
                judge_concurrency = st.slider(
                    "并发数",
                    min_value=1, max_value=8, value=3, step=1,
                    key="judge_concurrency",
                    help="同时发起的 LLM 请求数。并发数过高可能触发 API 限流（429），建议从 3 开始逐步调整。",
                )
                st.markdown("---")
                st.warning("强制重新评测会忽略所有已有结果，重复消耗 token。")
                btn_force = st.button(
                    "强制重新评测全部",
                    use_container_width=True,
                    help="忽略所有缓存，对筛选范围内的全部样本重新评测",
                )
                if btn_force:
                    st.session_state["judge_mode"] = "force_all"

            # === 本次评测执行计划（基于 build_judge_plan，默认预览增量模式） ===
            st.markdown("---")
            st.markdown("##### 本次评测执行计划")

            # 使用统一的计划函数，默认预览增量模式
            _preview_mode = st.session_state.get("judge_mode", "incremental")
            _plan = build_judge_plan(filtered_samples, existing_results_map, _preview_mode)
            _preview_candidates = _plan["samples"]

            # 候选样本来源说明
            _mode_labels = {
                "quick_test": "快速测试 1 条",
                "incremental": "增量评测全部（仅新样本/未成功样本）",
                "retry_failed": "仅重试失败样本",
                "force_all": "强制重新评测全部样本",
            }
            st.markdown(f"**当前模式**：{_mode_labels.get(_preview_mode, _preview_mode)}")
            st.markdown(
                f"**计划概览**：筛选后 {_plan['total_filtered']} 条"
                f" | 已成功 {_plan['success_count']} 条"
                f" | 待评新样本 {_plan['new_count']} 条"
                f" | 重试失败 {_plan['retry_count']} 条"
                f" | 规则预筛 {_plan['prescreen_count']} 条"
                f" | 预计 LLM 调用 **{_plan['llm_count']}** 次"
            )

            # 候选样本的评测模式构成
            _cand_with_ref = sum(1 for s in _preview_candidates if (s.get("reference_answer") or "").strip())
            _cand_no_ref = len(_preview_candidates) - _cand_with_ref
            if _cand_with_ref > 0 and _cand_no_ref > 0:
                st.caption(f"其中 {_cand_with_ref} 条走严格评测，{_cand_no_ref} 条走合理性评测")
            elif _cand_with_ref > 0:
                st.caption(f"全部 {_cand_with_ref} 条走严格评测（均有参考答案）")
            elif _cand_no_ref > 0:
                st.caption(f"全部 {_cand_no_ref} 条走合理性评测（均无参考答案）")

            # --- 与历史结果的交叉分析 ---
            existing_success_count = sum(
                1 for r in existing_results_map.values() if "error" not in r
            )
            existing_error_count = sum(
                1 for r in existing_results_map.values() if "error" in r
            )
            total_historical = len(existing_results_map)

            if total_historical > 0:
                st.markdown(
                    f"**历史评测记录**：`{JUDGED_FILE.name}` 中已有 "
                    f"**{existing_success_count}** 条成功 + **{existing_error_count}** 条失败"
                )
            else:
                st.markdown(f"**历史评测记录**：暂无（`{JUDGED_FILE.name}` 不存在或为空）")

            # --- 候选样本逐条预览 ---
            with st.expander("查看候选样本明细（点击展开）", expanded=False):
                if not _preview_candidates:
                    st.info("当前模式下没有候选样本")
                else:
                    for _idx, _s in enumerate(_preview_candidates):
                        _q = (_s.get("question") or "(无问题)")[:60]
                        _has_ref = bool((_s.get("reference_answer") or "").strip())
                        _mode_tag = "严格" if _has_ref else "合理性"
                        _tid = _s.get("trace_id")
                        _existing = existing_results_map.get(_tid)
                        if _existing and "error" in _existing:
                            st.caption(f"  🔄 {_idx+1}. `{_q}` — 历史失败，将重试 [{_mode_tag}]")
                        elif _existing and "error" not in _existing:
                            # force_all 模式下已成功样本也会出现
                            st.caption(f"  ⏭️ {_idx+1}. `{_q}` — 已成功（强制重评） [{_mode_tag}]")
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
                _track_counts = Counter(classify_evaluation_track(s) for s in _preview_candidates)

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
            # 预计算各模式的计划，用于按钮状态和预览
            _plan_quick = build_judge_plan(filtered_samples, existing_results_map, "quick_test")
            _plan_retry = build_judge_plan(filtered_samples, existing_results_map, "retry_failed")
            _plan_incremental = build_judge_plan(filtered_samples, existing_results_map, "incremental")

            # 快速测试按钮说明
            _quick_disabled = len(_plan_quick["samples"]) == 0
            if _quick_disabled:
                if _plan_quick["success_count"] == _plan_quick["total_filtered"]:
                    _quick_help = "所有样本已有成功结果，无需测试"
                else:
                    _quick_help = "当前范围内没有待评且可调用 LLM 的样本"
            else:
                _preview = _plan_quick["selected_sample_preview"]
                _quick_help = f"将评测：{_preview['question']}（{_preview['trace_id_suffix']}）"

            # 重试按钮说明
            _retry_disabled = len(_plan_retry["samples"]) == 0

            # 增量模式摘要
            _inc = _plan_incremental
            _inc_summary = (
                f"总计 {_inc['total_filtered']} 条 | "
                f"已成功 {_inc['success_count']} 条 | "
                f"新样本 {_inc['new_count']} 条 | "
                f"重试失败 {_inc['retry_count']} 条 | "
                f"预计 LLM 调用 {_inc['llm_count']} 次"
            )

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                btn_quick = st.button(
                    "快速测试 1 条",
                    use_container_width=True,
                    disabled=_quick_disabled,
                    help=_quick_help,
                )
                if btn_quick:
                    st.session_state["judge_mode"] = "quick_test"
            with btn_col2:
                btn_incremental = st.button(
                    "增量评测全部",
                    type="primary",
                    use_container_width=True,
                    help=_inc_summary,
                )
                if btn_incremental:
                    st.session_state["judge_mode"] = "incremental"
            with btn_col3:
                if _retry_disabled:
                    st.button(
                        "仅重试失败样本",
                        use_container_width=True,
                        disabled=True,
                        help="暂无失败样本，无需重试",
                    )
                    btn_retry = False
                else:
                    btn_retry = st.button(
                        f"仅重试失败样本（{_plan_retry['retry_count']} 条）",
                        use_container_width=True,
                        help="仅重新评测之前失败的样本，不影响成功结果",
                    )
                    if btn_retry:
                        st.session_state["judge_mode"] = "retry_failed"

            # 预览优化策略按钮
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                preview_optimization = st.button(
                    "预览优化策略",
                    use_container_width=True,
                    help="查看实际需要调用 LLM 的次数，不消耗 token",
                )
            with preview_col2:
                preview_rules = st.button(
                    "快速规则预览（零 LLM）",
                    use_container_width=True,
                    help="仅用确定性规则判定所有候选样本，不调用 LLM，不写入正式结果",
                )

        def _load_existing_for_session():
            if "judge_results" not in st.session_state and existing_results_map:
                st.session_state["judge_results"] = list(existing_results_map.values())
                st.session_state["judge_results_source"] = "historical"

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
            """通用的评测执行 + 进度 UI。返回 (new_results, stats)。"""
            if not samples_to_judge:
                st.info("没有需要评测的样本（全部已有成功结果）")
                return [], {}

            st.info(f"💡 本次共 **{len(samples_to_judge)}** 条样本，经规则预筛选和内容去重后，实际 LLM 请求数可能更少。")

            # 跟踪最新 info 用于完成摘要
            _latest_info = {}

            with st.status(f"正在运行 {label}...", expanded=True) as eval_status:
                progress_bar = st.progress(0, text="准备开始评测...")
                status_text = st.empty()
                stats_text = st.empty()
                question_text = st.empty()
                live_result_area = st.container()
                status_text.write("⏳ 状态：准备开始")

                def _fmt_elapsed(secs):
                    """格式化秒数为 MM:SS。"""
                    m, s = divmod(int(secs), 60)
                    return f"{m:02d}分{s:02d}秒"

                def on_progress(done, total, result, info):
                    _latest_info.update(info)
                    llm_done = info.get("llm_done", 0)
                    llm_total = info.get("llm_total", 0)
                    elapsed = info.get("elapsed", 0)
                    eta_text = info.get("eta_text", "计算中")
                    throughput = info.get("throughput", 0.0)
                    prescreened = info.get("prescreened_count", 0)
                    cached = info.get("cached_count", 0)
                    concurrency = info.get("concurrency", 1)

                    progress_bar.progress(
                        done / total,
                        text=f"评测进度: {done}/{total}",
                    )

                    # 两阶段状态显示
                    if llm_total > 0 and llm_done < llm_total:
                        # LLM 阶段进行中
                        status_text.info(
                            f"**规则阶段**: 已完成 {prescreened} 条直接判定"
                            f" | **LLM 阶段**: {llm_done}/{llm_total}，并发 {concurrency}"
                        )
                    elif llm_total > 0:
                        # LLM 阶段完成
                        status_text.success(
                            f"**规则阶段**: {prescreened} 条直接判定"
                            f" | **LLM 阶段**: {llm_done}/{llm_total} 完成"
                        )
                    else:
                        # 仍在规则阶段
                        status_text.info(
                            f"**规则阶段**: 已处理 {done}/{total}（规则判定 {prescreened} 条，去重 {cached} 条）"
                        )

                    # 最后完成的题目
                    _q = (result.get("question") or "")[:60]
                    if "error" in result:
                        question_text.error(f"最后完成: {_q} — 出错: {result['error'][:80]}")
                    else:
                        question_text.caption(f"最后完成: {_q}")

                    # 统计栏：耗时 + 吞吐 + ETA
                    _elapsed_str = _fmt_elapsed(elapsed)
                    if llm_done >= 2 and throughput > 0:
                        _tp_str = f"{throughput:.2f} 条/秒"
                        stats_text.caption(
                            f"⏱️ {_elapsed_str} | LLM {llm_done}/{llm_total} | "
                            f"吞吐 {_tp_str} | ETA {eta_text}"
                        )
                    elif llm_done > 0:
                        stats_text.caption(
                            f"⏱️ {_elapsed_str} | LLM {llm_done}/{llm_total} | ETA {eta_text}"
                        )
                    else:
                        stats_text.caption(f"⏱️ {_elapsed_str} | 规则阶段进行中...")

                new_results = []
                for result in judge_all(
                    samples_to_judge, judge_api_key, judge_base_url,
                    judge_model, on_progress, timeout=judge_timeout,
                    max_workers=judge_concurrency,
                ):
                    new_results.append(result)
                    is_prescreened = result.get("_prescreened", False)
                    is_cached = result.get("_content_cached", False)
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
                            _track = _r.get("evaluation_track", "")
                            _idx = len(new_results)
                            _q = (_r.get('question') or '')[:40]
                            if _track == TRACK_RETRIEVAL:
                                t1 = "✓" if _r.get("retrieval_top1_hit") else "✗"
                                t3 = "✓" if _r.get("retrieval_top3_hit") else "✗"
                                t5 = "✓" if _r.get("retrieval_top5_hit") else "✗"
                                pos = _r.get("hit_evidence_position")
                                pos_str = str(pos) if pos else "无"
                                st.write(
                                    f"✅ [{_idx}] {_q} — "
                                    f"Top1:{t1} | Top3:{t3} | Top5:{t5} | "
                                    f"最早命中位置:{pos_str}{tag}"
                                )
                            elif _track == TRACK_CHUNK_EXACT:
                                ce_status = _r.get("chunk_exact_status", "")
                                if ce_status:
                                    # missing_binding / no_trace / no_retrieval
                                    _status_labels = {
                                        "missing_binding": "缺少绑定（expected_segment_id / expected_content_hash）",
                                        "no_trace": "未关联真实 Langfuse trace",
                                        "no_retrieval": "trace 已关联但无检索结果",
                                    }
                                    _label = _status_labels.get(ce_status, ce_status)
                                    st.warning(f"⚠️ [{_idx}] {_q} — Chunk Exact 不可判定：{_label}{tag}")
                                else:
                                    t1 = "✓" if _r.get("retrieval_top1_hit") else "✗"
                                    t3 = "✓" if _r.get("retrieval_top3_hit") else "✗"
                                    t5 = "✓" if _r.get("retrieval_top5_hit") else "✗"
                                    t10 = "✓" if _r.get("retrieval_top10_hit") else "✗"
                                    pos = _r.get("hit_evidence_position")
                                    seg_id = (_r.get("expected_segment_id") or "")[:12]
                                    if _r.get("retrieval_top10_hit"):
                                        st.write(
                                            f"✅ [{_idx}] {_q} — Chunk Exact | "
                                            f"Top1:{t1} | Top3:{t3} | Top5:{t5} | Top10:{t10} | "
                                            f"首次命中:Top{pos}{tag}"
                                        )
                                    elif _r.get("retrieval_top5_hit"):
                                        st.warning(
                                            f"⚠️ [{_idx}] {_q} — Chunk Exact | "
                                            f"Top5命中但 Top10 边缘 | "
                                            f"Top1:{t1} | Top3:{t3} | Top5:{t5} | Top10:{t10} | "
                                            f"首次命中:Top{pos}{tag}"
                                        )
                                    else:
                                        st.warning(
                                            f"⚠️ [{_idx}] {_q} — Chunk Exact | "
                                            f"Top10未命中 | 目标 chunk:{seg_id}{tag}"
                                        )
                            elif _track == TRACK_STRICT_QA:
                                ans = "✓" if _r.get("answer_correct") else "✗"
                                st.write(
                                    f"✅ [{_idx}] {_q} — Answer:{ans}{tag}"
                                )
                            elif _track == TRACK_GROUNDED_QA:
                                gnd = "✓" if _r.get("answer_correct") else "✗"
                                st.write(
                                    f"✅ [{_idx}] {_q} — 回答有据:{gnd}{tag}"
                                )
                            else:
                                st.write(
                                    f"✅ [{_idx}] {_q} — 不可评测：缺少金标准证据{tag}"
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

            # 完成摘要
            _elapsed = _latest_info.get("elapsed", 0)
            _prescreened = _latest_info.get("prescreened_count", 0)
            _cached = _latest_info.get("cached_count", 0)
            _llm_done = _latest_info.get("llm_done", 0)
            _concurrency = _latest_info.get("concurrency", 1)
            _elapsed_str = _fmt_elapsed(_elapsed)

            eval_status.update(
                label=f"{label}完成 — "
                f"总耗时 {_elapsed_str} | "
                f"样本 {len(new_results)} 条 | "
                f"LLM 调用 {_llm_done} 次 | "
                f"规则判定 {_prescreened} 条 | "
                f"内容复用 {_cached} 条 | "
                f"并发 {_concurrency}"
            )

            stats = {
                "elapsed": _elapsed,
                "llm_done": _llm_done,
                "prescreened_count": _prescreened,
                "cached_count": _cached,
                "concurrency": _concurrency,
            }
            return new_results, stats

        # ---------- 预览优化策略 ----------
        if preview_optimization:
            _opt_mode = st.session_state.get("judge_mode", "incremental")
            _opt_plan = build_judge_plan(filtered_samples, existing_results_map, _opt_mode)
            candidates = _opt_plan["samples"]

            prescreen_results = []   # (sample, prescreen_result)
            need_llm = []
            content_seen = {}        # hash -> sample
            content_dup_count = 0

            for s in candidates:
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

                if prescreen_results:
                    # 按类型分组统计
                    _ps_chunk_exact = sum(1 for s, _ in prescreen_results
                                         if classify_evaluation_track(s) == TRACK_CHUNK_EXACT)
                    _ps_not_eval = sum(1 for s, _ in prescreen_results
                                       if classify_evaluation_track(s) == TRACK_NOT_EVALUABLE)
                    _ps_rule = len(prescreen_results) - _ps_chunk_exact - _ps_not_eval

                    _ps_parts = []
                    if _ps_chunk_exact:
                        _ps_parts.append(f"Chunk Exact 机器判定 {_ps_chunk_exact} 条")
                    if _ps_rule:
                        _ps_parts.append(f"规则预筛选 {_ps_rule} 条")
                    if _ps_not_eval:
                        _ps_parts.append(f"缺少金标准 {_ps_not_eval} 条")
                    st.markdown(f"**{len(prescreen_results)} 条 — 直接判定**（{'，'.join(_ps_parts)}，不需要 LLM）")
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
                    raw_retrieval_chars = sum(len(line) for line in _raw_lines)
                else:
                    raw_retrieval_chars = 0

                # 实际版本：清洗 metadata + 分层截断（build_judge_prompt 内部会自动选模板）
                actual_prompt = build_judge_prompt(sample_preview)
                # 检索正文：用 build_judge_prompt 的内部格式化计算清洗后长度
                from judge import _format_single_result, classify_evaluation_track, get_gold_evidence, TRACK_RETRIEVAL
                if _raw_results:
                    _cleaned_lines = []
                    for i, r in enumerate(_raw_results):
                        _cleaned_lines.append(
                            f"--- 检索结果 {i + 1} ---\n"
                            + _format_single_result(r, i)
                        )
                    cleaned_retrieval_chars = sum(len(line) + 2 for line in _cleaned_lines)  # +2 for \n\n join
                else:
                    cleaned_retrieval_chars = 0

                # 检索正文统计
                if raw_retrieval_chars > 0 and cleaned_retrieval_chars > 0:
                    diff = cleaned_retrieval_chars - raw_retrieval_chars
                    if diff >= 0:
                        ratio_text = f"增加 {diff / raw_retrieval_chars * 100:.0f}%"
                    else:
                        ratio_text = f"节省 {-diff / raw_retrieval_chars * 100:.0f}%"
                    st.caption(
                        f"检索结果正文：原始 {raw_retrieval_chars} 字符"
                        f" → 清洗/截断后 {cleaned_retrieval_chars} 字符"
                        f"（{ratio_text}）"
                    )
                elif raw_retrieval_chars == 0:
                    st.caption("检索结果正文：无检索结果")

                # 最终 Prompt 统计
                st.caption(
                    f"最终 Judge Prompt：{len(actual_prompt)} 字符"
                    f"（含模板、评测查询、金标准证据与格式标签）"
                    f"。策略：去除 metadata 块，分层保留正文 — "
                    f"Top-1: 2000字，Top-2/3: 1200字，Top-4/5: 1000字"
                )
                with st.expander("查看处理后的 prompt 示例"):
                    st.code(actual_prompt, language=None)
            else:
                st.success("所有样本都已被跳过或规则判定，不需要调用 LLM！")

            st.divider()

        # ---------- 快速规则预览（零 LLM，不写入正式结果） ----------
        if preview_rules:
            _preview_mode = st.session_state.get("judge_mode", "incremental")
            _preview_plan = build_judge_plan(filtered_samples, existing_results_map, _preview_mode)
            _preview_candidates = _preview_plan["samples"]

            if not _preview_candidates:
                st.info("没有候选样本可供预览。")
            else:
                from judge import retrieval_rule_judge

                _rule_hits = []       # exact_contains_top1
                _rule_misses = []     # empty_results / no_content
                _rule_pending = []    # 需 LLM
                _rule_preview_rows = []

                for _s in _preview_candidates:
                    _track = _s.get("evaluation_track") or classify_evaluation_track(_s)
                    _s["evaluation_track"] = _track

                    if _track != TRACK_RETRIEVAL:
                        # QA 轨道：用 pre_screen 判断
                        _ps = pre_screen(_s)
                        if _ps is not None:
                            _rule_misses.append((_s, _ps))
                        else:
                            _rule_pending.append(_s)
                        continue

                    # 检索轨道：用 retrieval_rule_judge
                    _rr = retrieval_rule_judge(_s)
                    if _rr is not None:
                        if _rr.get("hit_evidence_position"):
                            _rule_hits.append((_s, _rr))
                        else:
                            _rule_misses.append((_s, _rr))
                        _rule_preview_rows.append({
                            "question_id": _s.get("question_id", ""),
                            "question": (_s.get("question") or "")[:50],
                            "rule_result": _rr.get("_rule_name", ""),
                            "top1": _rr.get("retrieval_top1_hit"),
                            "top3": _rr.get("retrieval_top3_hit"),
                            "top5": _rr.get("retrieval_top5_hit"),
                            "position": _rr.get("hit_evidence_position"),
                            "needs_llm": False,
                        })
                    else:
                        _rule_pending.append(_s)
                        _rule_preview_rows.append({
                            "question_id": _s.get("question_id", ""),
                            "question": (_s.get("question") or "")[:50],
                            "rule_result": "待 LLM",
                            "top1": None, "top3": None, "top5": None,
                            "position": None,
                            "needs_llm": True,
                        })

                st.markdown("---")
                st.markdown("##### 快速规则预览（零 LLM）")
                st.warning("⚠️ 以下为规则预览结果，**不是正式评测指标**。正式评测需运行 Judge 模式。")

                _total = len(_preview_candidates)
                _hit_n = len(_rule_hits)
                _miss_n = len(_rule_misses)
                _pending_n = len(_rule_pending)

                _prev_col1, _prev_col2, _prev_col3 = st.columns(3)
                with _prev_col1:
                    st.metric("规则确认命中", _hit_n)
                with _prev_col2:
                    st.metric("规则确认未命中", _miss_n)
                with _prev_col3:
                    st.metric("待 LLM 语义判断", _pending_n)

                # 规则确认 Top1/3/5 统计（仅检索轨道规则判定结果）
                _ret_rows = [r for r in _rule_preview_rows if r["top1"] is not None]
                if _ret_rows:
                    _t1_hit = sum(1 for r in _ret_rows if r["top1"])
                    _t3_hit = sum(1 for r in _ret_rows if r["top3"])
                    _t5_hit = sum(1 for r in _ret_rows if r["top5"])
                    _n = len(_ret_rows)
                    st.caption(
                        f"规则确认的检索结果（{_n} 条）："
                        f"Top1={_t1_hit}/{_n} | Top3={_t3_hit}/{_n} | Top5={_t5_hit}/{_n}"
                        f"  — 预览，不是正式评测指标"
                    )

                # 命中明细
                if _rule_hits:
                    with st.expander(f"规则确认命中（{_hit_n} 条）", expanded=False):
                        for _s, _rr in _rule_hits[:20]:
                            _q = (_s.get("question") or "")[:50]
                            _pos = _rr.get("hit_evidence_position")
                            _reason = _rr.get("reason", "")
                            st.caption(f"  - `{_q}` | Top{_pos} 命中 | {_reason}")

                # 未命中明细
                if _rule_misses:
                    with st.expander(f"规则确认未命中（{_miss_n} 条）", expanded=False):
                        for _s, _rr in _rule_misses[:20]:
                            _q = (_s.get("question") or "")[:50]
                            _reason = _rr.get("reason", "")
                            st.caption(f"  - `{_q}` | {_reason}")

                # 待 LLM 明细
                if _rule_pending:
                    with st.expander(f"待 LLM 语义判断（{_pending_n} 条）", expanded=False):
                        for _s in _rule_pending[:20]:
                            _q = (_s.get("question") or "")[:50]
                            _hint = _s.get("_rule_hint", {})
                            _hint_str = f" | Top{_hint['matched_rank']} 有完整匹配" if _hint else ""
                            st.caption(f"  - `{_q}`{_hint_str}")

        # ---------- 统一执行入口：根据 judge_mode 执行对应计划 ----------
        _run_mode = st.session_state.get("judge_mode")
        if _run_mode:
            if not judge_api_key:
                st.error("请在上方「API 配置」中输入 API Key")
            elif not judge_model:
                st.error("请在上方「API 配置」中输入 Model 名称")
            else:
                _run_plan = build_judge_plan(filtered_samples, existing_results_map, _run_mode)
                samples_to_judge = _run_plan["samples"]
                if not samples_to_judge:
                    st.info("没有需要评测的样本")
                else:
                    _mode_labels = {
                        "quick_test": "快速测试",
                        "incremental": "增量评测",
                        "retry_failed": "失败样本重试",
                        "force_all": "强制全量评测",
                    }
                    _label = _mode_labels.get(_run_mode, "Judge 评测")
                    _load_existing_for_session()
                    new_results, stats = _run_judge_ui(samples_to_judge, label=_label)
                    if new_results:
                        _, history_name = _merge_and_save(new_results)
                        # 按 run_id 分组写入各 run 的 manifest
                        try:
                            from datetime import datetime
                            from experiment import update_experiment_run
                            _completed_at = datetime.now().isoformat()
                            _batch_id = f"judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                            # 按 run_id 分组样本
                            _run_groups = {}
                            for s in samples_to_judge:
                                _rid = s.get("run_id", "")
                                if _rid:
                                    _run_groups.setdefault(_rid, []).append(s)

                            # 统计每个 run 的新结果
                            _result_by_trace = {r.get("trace_id"): r for r in new_results}

                            _is_multi_run = len(_run_groups) > 1
                            _batch_elapsed = round(stats.get("elapsed", 0), 2)

                            for _rid, _run_samples in _run_groups.items():
                                _run_new_results = [
                                    _result_by_trace[s.get("trace_id")]
                                    for s in _run_samples
                                    if s.get("trace_id") in _result_by_trace
                                ]
                                _run_llm = sum(
                                    1 for r in _run_new_results
                                    if not r.get("_prescreened") and not r.get("_content_cached")
                                )
                                _run_prescreened = sum(
                                    1 for r in _run_new_results if r.get("_prescreened")
                                )
                                _run_cached = sum(
                                    1 for r in _run_new_results if r.get("_content_cached")
                                )
                                _manifest_updates = {
                                    "judge_llm_call_count": _run_llm,
                                    "judge_prescreened_count": _run_prescreened,
                                    "judge_content_cached_count": _run_cached,
                                    "judge_concurrency": stats.get("concurrency", 1),
                                    "judge_completed_at": _completed_at,
                                    "judge_batch_id": _batch_id,
                                    "judge_mode": _run_mode,
                                }
                                if _is_multi_run:
                                    # 跨 run 批次：总耗时是批次级别，不是单 run 独占
                                    _manifest_updates["judge_batch_duration_seconds"] = _batch_elapsed
                                    _manifest_updates["judge_duration_scope"] = "batch"
                                else:
                                    # 单 run 批次：总耗时即该 run 的 Judge 墙钟耗时
                                    _manifest_updates["judge_duration_seconds"] = _batch_elapsed
                                    _manifest_updates["judge_duration_scope"] = "run"
                                try:
                                    update_experiment_run(_rid, _manifest_updates)
                                except Exception:
                                    pass  # 单个 run 写入失败不影响其他

                            # 若跨多个 run，额外记录批次信息到全局缓存（非权威）
                            if len(_run_groups) > 1:
                                _batch_cache = {
                                    "batch_id": _batch_id,
                                    "run_ids": list(_run_groups.keys()),
                                    "timestamp": _completed_at,
                                    "mode": _run_mode,
                                    "sample_count": len(new_results),
                                    "judge_duration_seconds": round(stats.get("elapsed", 0), 2),
                                    "judge_llm_call_count": stats.get("llm_done", 0),
                                    "judge_concurrency": stats.get("concurrency", 1),
                                }
                                _batch_file = JUDGED_DIR / f"judge_batch_{_batch_id}.json"
                                _batch_file.write_text(
                                    json.dumps(_batch_cache, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                        except Exception:
                            pass  # 统计保存失败不影响主流程
                        st.success(f"评测完成！结果已保存到 {JUDGED_FILE.name}，历史快照: {history_name}")
            # 执行完毕后清除模式，避免重复触发
            del st.session_state["judge_mode"]

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

                # 命中分布摘要表（替代逐题柱状图，避免巨大图表）
                if view_valid:
                    st.markdown("**命中分布摘要**")
                    _retrieval_v = [r for r in view_valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
                    if _retrieval_v:
                        _t1h = sum(1 for r in _retrieval_v if r.get("retrieval_top1_hit"))
                        _t3h = sum(1 for r in _retrieval_v if r.get("retrieval_top3_hit"))
                        _t5h = sum(1 for r in _retrieval_v if r.get("retrieval_top5_hit"))
                        _n = len(_retrieval_v)
                        _dist = pd.DataFrame([
                            {"指标": "Top1 命中", "数量": _t1h, "占比": f"{_t1h/_n:.0%}"},
                            {"指标": "Top3 命中", "数量": _t3h, "占比": f"{_t3h/_n:.0%}"},
                            {"指标": "Top5 命中", "数量": _t5h, "占比": f"{_t5h/_n:.0%}"},
                        ])
                        st.dataframe(_dist, use_container_width=True, hide_index=True)
                    _qa_v = [r for r in view_valid if r.get("evaluation_track") in (TRACK_STRICT_QA, TRACK_GROUNDED_QA)]
                    if _qa_v:
                        _qa_ok = sum(1 for r in _qa_v if r.get("answer_correct"))
                        _qa_n = len(_qa_v)
                        st.caption(f"QA 回答正确: {_qa_ok}/{_qa_n} ({_qa_ok/_qa_n:.0%})")

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

                # 评测详情卡（复用分页渲染函数）
                st.subheader("评测详情")
                if not view_all:
                    st.info("当前视图下无评测样本")
                else:
                    render_judge_results_list(view_all, _sample_map, key_prefix="jd_detail")

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

                        # 可视化图表：Top1/Top3/Top5 命中率 + 命中/未命中分布
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
                            st.markdown("**命中分布**")
                            if _sv:
                                _t1_hit = sum(1 for r in _sv if r.get("retrieval_top1_hit"))
                                _t1_miss = len(_sv) - _t1_hit
                                _t3_hit = sum(1 for r in _sv if r.get("retrieval_top3_hit"))
                                _t3_miss = len(_sv) - _t3_hit
                                _t5_hit = sum(1 for r in _sv if r.get("retrieval_top5_hit"))
                                _t5_miss = len(_sv) - _t5_hit
                                _dist_df = pd.DataFrame([
                                    {"指标": "Top1", "命中": _t1_hit, "未命中": _t1_miss},
                                    {"指标": "Top3", "命中": _t3_hit, "未命中": _t3_miss},
                                    {"指标": "Top5", "命中": _t5_hit, "未命中": _t5_miss},
                                ])
                                st.dataframe(_dist_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("无有效评测数据")

                        # 检索评测详情（复用共享分页渲染器）
                        st.markdown("---")
                        st.markdown("##### 检索评测详情")
                        render_judge_results_list(
                            _sa, _sample_map, key_prefix="judge_ret", page_size=20,
                        )
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
    _record_rss("实验看板页")
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
    from judge import (
        compute_metrics, compute_chunk_exact_metrics,
        TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_CHUNK_EXACT,
    )
    from report_export import build_evaluation_html, build_runs_csv, build_failed_samples_csv, build_export_filename
    from optimization_analysis import (
        build_analysis_context, analyze_overview, analyze_failure_groups,
        synthesize_optimization_report, save_analysis_report, get_analysis_config,
    )

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
                        processed_file=_resolve_processed_path()[0],
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
                        processed_file=_resolve_processed_path()[0],
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

    # 合并重复配置按钮
    from experiment import merge_duplicate_configs as _merge_dup, find_duplicate_config_groups as _find_dup
    with st.expander("合并重复配置", expanded=False):
        _dup_preview = _merge_dup(dry_run=True)
        if _dup_preview["groups"] > 0:
            _detail = _dup_preview["details"][0]
            st.caption(
                f"发现 {_dup_preview['groups']} 组重复配置。"
            )
            st.markdown(
                f"- **Canonical 配置**: {_detail['canonical_name']}（`{_detail['canonical_id']}`）\n"
                f"- 待迁移 run 数: {len(_detail['migrated_run_ids'])}\n"
                f"- 将删除的重复配置数: {len(_detail['dup_ids'])}\n"
                f"- 不会合并或删除运行，只会统一其配置归属"
            )
            if st.button("合并重复配置（迁移其运行归属）", key="btn_merge_dup_configs"):
                with st.spinner("正在合并..."):
                    _result = _merge_dup(dry_run=False)
                if _result["validation_failures"]:
                    st.error(
                        f"校验失败，未删除配置: {_result['validation_failures']}"
                    )
                else:
                    st.success(
                        f"合并完成：迁移 {_result['runs_migrated']} 个 run 到 "
                        f"canonical 配置，删除 {_result['configs_deleted']} 个重复配置。"
                    )
                st.rerun()
        else:
            st.caption("无重复配置需要合并。")

    configs = list_config_profiles()

    if not configs:
        st.info("暂无配置方案。在「批量提问」页面创建配置后，将自动记录在此。")
        st.stop()

    # 构建下拉选项
    config_options = []
    for cfg in configs:
        runs_count = len(list_runs_by_config(cfg.get("config_id", "")))
        _cid = cfg.get("config_id", "")
        _cid_suffix = _cid[-8:] if len(_cid) > 8 else _cid
        _created = cfg.get("created_at", "")
        _ts_short = _created[:16].replace("T", " ") if _created else ""
        label = f"{cfg.get('config_name', '未命名')} | {cfg.get('knowledge_base_version', '')} | {runs_count} 次运行 | {_cid_suffix} | {_ts_short}"
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

    # ---------- 编辑配置：动态 key_prefix 防止跨配置状态串扰 ----------
    import hashlib as _hashlib
    _ecfg_key_prefix = f"ecfg_{_hashlib.md5(selected_config_id.encode()).hexdigest()[:12]}"

    # 检测配置切换：清理上一配置的编辑态 session_state
    _ecfg_prev_id = st.session_state.get("_ecfg_form_bound_id")
    if _ecfg_prev_id and _ecfg_prev_id != selected_config_id:
        _old_prefix = f"ecfg_{_hashlib.md5(_ecfg_prev_id.encode()).hexdigest()[:12]}"
        _keys_to_clean = [k for k in st.session_state if k.startswith(_old_prefix)]
        for _k in _keys_to_clean:
            del st.session_state[_k]
        # 同时清理旧的编辑说明
        if "ec_edit_note" in st.session_state:
            del st.session_state["ec_edit_note"]
    # 记录当前表单绑定的 config_id
    st.session_state["_ecfg_form_bound_id"] = selected_config_id

    with st.expander("编辑/查看配置详情", expanded=False):
        with st.form("edit_config_form"):
            st.markdown("**可编辑字段**（核心字段 config_id / created_at 不可修改）")
            ec_values = render_config_form(selected_config, key_prefix=_ecfg_key_prefix)
            ec_note = st.text_input("修改说明（可选）", value="", key="ec_edit_note",
                                    help="简要说明本次修改原因，如：补录 Rerank 配置")
            ec_submit = st.form_submit_button("保存配置修改", type="primary")

        if ec_submit:
            from experiment import update_config_profile_safe
            # 一致性保护：三重校验 config_id
            _form_bound = st.session_state.get("_ecfg_form_bound_id")
            _disk_config = load_config_profile(selected_config_id)
            _disk_id = _disk_config.get("config_id") if _disk_config else None
            if not (_form_bound == selected_config_id == _disk_id):
                st.error(
                    f"配置不一致，保存已阻止。"
                    f" 表单绑定: {_form_bound}, 选择: {selected_config_id}, 磁盘: {_disk_id}"
                    f" 请刷新页面后重试。"
                )
            else:
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
                processed_file=find_processed_for_run(rid),
                judged_file=str(JUDGED_FILE),
                include_judge_results=True,
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
        # 补齐旧版 chunk_exact 缺失的 Top10（需扫描 processed sample 的检索结果）
        _ce_proc_path_str, _ce_proc_mtime = _resolve_processed_path()
        _ce_sample_lookup = _load_sample_lookup(_ce_proc_mtime, _ce_proc_path_str)
        for r in all_judge_results:
            backfill_chunk_exact_topk(r, _ce_sample_lookup)
        cumulative_metrics = compute_metrics(all_judge_results)

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

        retrieval_all = [r for r in valid_all
                         if r.get("evaluation_track") == TRACK_RETRIEVAL
                         and r.get("retrieval_evaluable", True)]
        strict_qa_all = [r for r in valid_all if r.get("evaluation_track") == TRACK_STRICT_QA]
        grounded_qa_all = [r for r in valid_all if r.get("evaluation_track") == TRACK_GROUNDED_QA]
        chunk_exact_all = [r for r in valid_all
                           if r.get("evaluation_track") == TRACK_CHUNK_EXACT]
        chunk_exact_evaluable = [r for r in chunk_exact_all
                                 if r.get("retrieval_evaluable", True) is not False
                                 and r.get("retrieval_top1_hit") is not None]
        chunk_exact_unevaluable = [r for r in chunk_exact_all
                                   if r.get("retrieval_evaluable", True) is False
                                   or r.get("retrieval_top1_hit") is None]

        st.markdown("---")
        st.markdown("**累计 Judge 指标**")
        st.caption("按样本加权汇总（命中总数 / 有效样本数），去重后统计。不同评测轨道不混合。")

        has_any_track = retrieval_all or strict_qa_all or grounded_qa_all or chunk_exact_evaluable
        if not has_any_track:
            st.info("暂无评测数据（无 AI 证据评测题，也无 chunk_exact 可评测样本）")
        else:
            # ── AI 证据评测 ──
            st.markdown("##### AI 证据评测（retrieval）")
            if retrieval_all:
                n = len(retrieval_all)
                t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_all)
                t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_all)
                t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_all)
                ai_col1, ai_col2, ai_col3, ai_col4 = st.columns(4)
                with ai_col1:
                    st.metric("可评测", n)
                with ai_col2:
                    st.metric("Top1", f"{t1}/{n} ({t1/n:.1%})")
                with ai_col3:
                    st.metric("Top3", f"{t3}/{n} ({t3/n:.1%})")
                with ai_col4:
                    st.metric("Top5", f"{t5}/{n} ({t5/n:.1%})")
            else:
                st.info("本配置未包含 AI 证据评测题")

            # ── Chunk Exact 机器判定 ──
            if chunk_exact_all:
                st.markdown("##### Chunk Exact（机器判定）")
                ce_n = len(chunk_exact_evaluable)
                ce_total = len(chunk_exact_all)
                if ce_n > 0:
                    ce_t1 = sum(r.get("retrieval_top1_hit", 0) for r in chunk_exact_evaluable)
                    ce_t3 = sum(r.get("retrieval_top3_hit", 0) for r in chunk_exact_evaluable)
                    ce_t5 = sum(r.get("retrieval_top5_hit", 0) for r in chunk_exact_evaluable)
                    ce_t10 = sum(r.get("retrieval_top10_hit", 0) for r in chunk_exact_evaluable)
                    _render_ce_topk(ce_n, ce_t1, ce_t3, ce_t5, ce_t10, ce_total)

                # 不可评测状态统计
                if chunk_exact_unevaluable:
                    ue_status_counts = {}
                    for r in chunk_exact_unevaluable:
                        s = r.get("chunk_exact_status") or r.get("reason", "未知")
                        ue_status_counts[s] = ue_status_counts.get(s, 0) + 1
                    ue_parts = [f"{s}: {c}" for s, c in ue_status_counts.items()]
                    st.caption(f"不可评测 {len(chunk_exact_unevaluable)} 条 — " + "、".join(ue_parts))

            # ── 严格问答 / 合理性问答 ──
            qa_col1, qa_col2 = st.columns(2)
            if strict_qa_all:
                with qa_col1:
                    st.markdown("**严格问答**")
                    n = len(strict_qa_all)
                    acc = sum(r.get("answer_correct", 0) for r in strict_qa_all) / n
                    st.metric("Answer Correctness", f"{acc:.0%}")
                    st.caption(f"有效样本数 n={n}")
            else:
                with qa_col1:
                    st.markdown("**严格问答**")
                    st.info("暂无数据")

            if grounded_qa_all:
                with qa_col2:
                    st.markdown("**合理性问答**")
                    n = len(grounded_qa_all)
                    acc = sum(r.get("answer_correct", 0) for r in grounded_qa_all) / n
                    st.metric("Answer Groundedness", f"{acc:.0%}")
                    st.caption(f"有效样本数 n={n}")
            else:
                with qa_col2:
                    st.markdown("**合理性问答**")
                    st.info("暂无数据")

            # ── 机器判定完成 vs AI Judge 完成 ──
            machine_judged = len(chunk_exact_all)
            ai_judged = len(retrieval_all) + len(strict_qa_all) + len(grounded_qa_all)
            st.caption(f"机器判定完成: {machine_judged} | AI Judge 完成: {ai_judged} | Judge 错误: {len(error_all)}")

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

        # 轨道分布和 chunk_exact 命中位置分布
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Judge 轨道分布**")
            dist_labels = []
            dist_values = []
            if retrieval_all:
                dist_labels.append("检索评测")
                dist_values.append(len(retrieval_all))
            if chunk_exact_evaluable:
                dist_labels.append("Chunk Exact")
                dist_values.append(len(chunk_exact_evaluable))
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
            if chunk_exact_evaluable:
                # Chunk Exact 命中位置分布（互斥分桶，扩展至 Top10）
                st.markdown("**Chunk Exact 命中位置分布**")
                ce_n = len(chunk_exact_evaluable)
                bucket_top1 = 0
                bucket_2_3 = 0
                bucket_4_5 = 0
                bucket_6_10 = 0
                bucket_miss = 0
                for r in chunk_exact_evaluable:
                    pos = r.get("hit_evidence_position")
                    t1 = r.get("retrieval_top1_hit", 0)
                    t10 = r.get("retrieval_top10_hit", 0)
                    if t1:
                        bucket_top1 += 1
                    elif pos is not None and 2 <= pos <= 3:
                        bucket_2_3 += 1
                    elif pos is not None and 4 <= pos <= 5:
                        bucket_4_5 += 1
                    elif pos is not None and 6 <= pos <= 10:
                        bucket_6_10 += 1
                    elif t10:
                        # 旧记录 position=None 但 top10=1（命中在 6-10）
                        bucket_6_10 += 1
                    else:
                        bucket_miss += 1

                buckets = [
                    ("Top1 命中", bucket_top1, "#2ca02c"),
                    ("第2-3位首次命中", bucket_2_3, "#1f77b4"),
                    ("第4-5位首次命中", bucket_4_5, "#ff7f0e"),
                    ("第6-10位首次命中", bucket_6_10, "#9467bd"),
                    ("Top10 未命中", bucket_miss, "#d62728"),
                ]
                # 过滤空桶
                buckets = [(l, v, c) for l, v, c in buckets if v > 0]
                if buckets:
                    fig_ce_hit = go.Figure(data=[go.Bar(
                        y=[b[0] for b in buckets],
                        x=[b[1] for b in buckets],
                        orientation="h",
                        marker_color=[b[2] for b in buckets],
                        text=[f"{b[1]} ({b[1]/ce_n:.0%})" for b in buckets],
                        textposition="auto",
                    )])
                    fig_ce_hit.update_layout(
                        xaxis_title="样本数",
                        height=300, margin=dict(t=20, b=30, l=120),
                    )
                    st.plotly_chart(fig_ce_hit, use_container_width=True, key="cum_ce_hit_pos")
                    st.caption(f"可评测样本数: {ce_n}")
            else:
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

    # ---------- 一键导出评测报告 ----------
    if not config_runs:
        st.markdown("---")
        st.info("当前配置方案下无运行记录，无法导出报告。请先执行批量提问并完成评测。")
    if config_runs:
        st.markdown("---")
        st.markdown("##### 一键导出评测报告")
        _export_cols = st.columns(3)

        # 构建 run_data_list（为三个导出共用）
        _run_data_list = []
        for _i, _run in enumerate(config_runs):
            _rs = _all_run_statuses[_i] if _i < len(_all_run_statuses) else {}
            _jr = _rs.get("judge_results", [])
            _m = compute_metrics(_jr) if _jr else {}
            _run_data_list.append({"run": _run, "run_status": _rs, "metrics": _m})

        # 构建 processed sample lookup（按每个 run 的 provenance 定位 processed file）
        _sample_lookup, _provenance_info = _build_merged_sample_lookup(config_runs)

        _disp_name = selected_config.get('config_name', '未命名')
        _cid = selected_config.get('config_id', '')
        _export_scope = f"配置 {_disp_name}，{len(config_runs)} 次运行"

        with _export_cols[0]:
            # 从 config_snapshot 读取 configured_top_k
            _snapshot = (config_runs[0].get("config_snapshot") or {}) if config_runs else {}
            _configured_top_k = _snapshot.get("top_k", 10) or 10
            _html_bytes = build_evaluation_html(
                selected_config, config_runs, _run_data_list,
                cumulative_metrics, all_judge_results,
                export_scope=_export_scope, sample_lookup=_sample_lookup,
                configured_top_k=_configured_top_k,
            ).encode("utf-8")
            st.download_button(
                label=f"下载 HTML 报告（{_disp_name}）",
                data=_html_bytes,
                file_name=build_export_filename(_disp_name, _cid, "report", "html"),
                mime="text/html",
                use_container_width=True,
                help="自包含 HTML 报告，可在浏览器直接打开并打印为 PDF",
            )
        with _export_cols[1]:
            st.download_button(
                label=f"下载运行汇总 CSV（{_disp_name}）",
                data=build_runs_csv(_run_data_list),
                file_name=build_export_filename(_disp_name, _cid, "runs", "csv"),
                mime="text/csv",
                use_container_width=True,
                help="每个运行一行，含 Top1/3/5 指标",
            )
        with _export_cols[2]:
            _failed_csv = build_failed_samples_csv(all_judge_results, _sample_lookup, selected_config)
            st.download_button(
                label=f"下载未命中样本 CSV（{_disp_name}）",
                data=_failed_csv,
                file_name=build_export_filename(_disp_name, _cid, "failed_samples", "csv"),
                mime="text/csv",
                use_container_width=True,
                help="仅 Top5 未命中的检索样本",
            )

    # ---------- AI 优化分析报告 ----------
    if config_runs and all_judge_results:
        st.markdown("---")
        st.markdown("##### AI 优化分析报告")
        st.caption("基于评测数据调用 LLM 生成知识库优化诊断建议，需要消耗 API 额度")

        _analysis_api_key, _analysis_base_url, _analysis_model = get_analysis_config()

        if not _analysis_api_key:
            st.warning("未配置分析 API。请在 .env 中设置 ANALYSIS_API_KEY 或 JUDGE_API_KEY")
        else:
            _report_cache_key = f"ai_analysis_report_{selected_config_id}"

            if st.button("生成 AI 优化分析", key="btn_gen_ai_analysis", type="primary"):
                with st.status("正在生成 AI 优化分析...", expanded=True) as _ai_status:
                    _ai_progress = st.progress(0, text="构建分析上下文...")
                    _ai_status_text = st.empty()

                    # 阶段 0：构建上下文
                    _ai_status_text.write("正在构建分析上下文...")
                    _ai_context = build_analysis_context(
                        _run_data_list, _sample_lookup, all_judge_results, selected_config,
                    )
                    _ai_progress.progress(0.1, text="上下文构建完成")

                    # 阶段 1：整体概览
                    _ai_status_text.write("阶段 1/3：正在分析总览指标...")
                    try:
                        _ai_stage1 = analyze_overview(
                            _ai_context, _analysis_api_key, _analysis_base_url, _analysis_model,
                        )
                    except Exception as e:
                        _ai_status.update(label="阶段 1 失败", state="error")
                        st.error(f"总览分析失败: {e}")
                        st.stop()
                    _ai_progress.progress(0.35, text="总览分析完成")

                    # 阶段 2：失败诊断（map-reduce）
                    _ai_status_text.write("阶段 2/3：正在分组失败样本...")
                    _stage2_detail = st.empty()

                    def _stage2_progress(phase, detail):
                        if phase == "grouping":
                            _tf = detail["total_failures"]
                            _gc = detail["group_count"]
                            _bc = detail["batch_count"]
                            _ai_status_text.write(
                                f"阶段 2/3：共 {_tf} 条失败，{_gc} 组，{_bc} 个子批次"
                            )
                            _ai_progress.progress(0.38)
                        elif phase == "sub_batch":
                            _bi = detail["batch_index"]
                            _bt = detail["total_batches"]
                            _bs = detail["status"]
                            _pc = detail.get("payload_chars", 0)
                            _status_icon = "✓" if _bs == "ok" else "✗"
                            _ai_status_text.write(
                                f"阶段 2/3：子批次分析 {_bi}/{_bt} {_status_icon}"
                            )
                            _stage2_detail.caption(
                                f"批次 {detail['batch_id']} | "
                                f"payload {_pc} 字符 | "
                                f"状态: {_bs}"
                            )
                            _ai_progress.progress(0.35 + 0.25 * (_bi / _bt))
                        elif phase == "synthesis":
                            if detail["status"] == "started":
                                _ai_status_text.write("阶段 2/3：正在汇总诊断...")
                                _ai_progress.progress(0.62)
                            else:
                                _ai_progress.progress(0.65)
                        elif phase == "done":
                            if detail["status"] == "completed":
                                _tc = detail["total_failures"]
                                _ok = detail["ok_count"]
                                _fc = detail["failed_count"]
                                _stage2_detail.caption(
                                    f"完成: {_tc} 条失败样本, "
                                    f"{_ok} 批成功, {_fc} 批失败"
                                )

                    try:
                        _ai_stage2 = analyze_failure_groups(
                            _ai_context, _analysis_api_key, _analysis_base_url, _analysis_model,
                            progress_callback=_stage2_progress,
                        )
                    except Exception as e:
                        _ai_status.update(label="阶段 2 失败", state="error")
                        st.error(f"失败分析失败: {e}")
                        st.stop()
                    _ai_progress.progress(0.65, text="失败分析完成")

                    # 阶段 3：汇总报告
                    _ai_status_text.write("阶段 3/3：正在生成最终报告...")
                    try:
                        _ai_report_md = synthesize_optimization_report(
                            _ai_stage1, _ai_stage2, _ai_context,
                            _analysis_api_key, _analysis_base_url, _analysis_model,
                        )
                    except Exception as e:
                        _ai_status.update(label="阶段 3 失败", state="error")
                        st.error(f"报告生成失败: {e}")
                        st.stop()
                    _ai_progress.progress(0.9, text="报告生成完成")

                    # 保存到文件
                    _ai_report_path = save_analysis_report(
                        _ai_report_md,
                        selected_config.get("config_name", "unnamed"),
                        REPORTS_DIR,
                    )
                    _ai_progress.progress(1.0, text="完成")
                    _ai_status.update(label="AI 优化分析完成！", state="complete")

                    # 缓存到 session state
                    st.session_state[_report_cache_key] = {
                        "markdown": _ai_report_md,
                        "path": str(_ai_report_path),
                        "filename": _ai_report_path.name,
                        "timestamp": datetime.now().isoformat(),
                    }

            # 显示已缓存的报告
            _ai_cached = st.session_state.get(_report_cache_key)
            if _ai_cached:
                with st.expander("AI 优化分析报告（点击展开）", expanded=True):
                    st.markdown(_ai_cached["markdown"])
                    st.caption(f"生成时间: {_ai_cached['timestamp']}")

                _ai_dl_cols = st.columns(2)
                with _ai_dl_cols[0]:
                    st.download_button(
                        label="下载 AI 优化分析 Markdown",
                        data=_ai_cached["markdown"].encode("utf-8"),
                        file_name=_ai_cached.get("filename", f"ai_analysis_{_ts}.md"),
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with _ai_dl_cols[1]:
                    if st.button("重新生成", key="btn_regenerate_ai_analysis"):
                        del st.session_state[_report_cache_key]
                        st.rerun()

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
                processed_file=find_processed_for_run(run["run_id"]),
                judged_file=str(JUDGED_FILE),
                include_judge_results=False,
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

        # ---------- 运行详情（按需加载单个 run） ----------
        _run_options = {run.get("run_id", ""): run for run in config_runs}
        _run_ids = list(_run_options.keys())

        # 构建 selectbox 选项：run_id | 题集 | Judge 数
        _run_labels = []
        for _rid in _run_ids:
            _r = _run_options[_rid]
            _rs = get_run_status(
                _rid,
                batch_dir=str(BATCH_DIR),
                raw_dir=str(RAW_DIR),
                processed_file=find_processed_for_run(_rid),
                judged_file=str(JUDGED_FILE),
                include_judge_results=False,
            )
            _qs = _rs.get("question_set_name") or _r.get("question_set_name", "") or "旧版题集"
            _jc = _rs.get("judge_count", 0)
            _run_labels.append(f"{_rid} | {_qs} | Judge:{_jc}")

        # 使用 session_state 保持选择
        _detail_key = f"_selected_detail_run_{selected_config.get('config_id', '')}"
        _prev_sel = st.session_state.get(_detail_key, "")

        _sel_idx = 0
        if _prev_sel and _prev_sel in _run_ids:
            _sel_idx = _run_ids.index(_prev_sel)

        _sel_label = st.selectbox(
            "选择运行查看题目明细",
            _run_labels,
            index=_sel_idx,
            key=f"_run_detail_select_{selected_config.get('config_id', '')}",
        )
        _sel_run_id = _run_ids[_run_labels.index(_sel_label)] if _sel_label else ""
        st.session_state[_detail_key] = _sel_run_id

        if _sel_run_id:
            run = _run_options[_sel_run_id]
            run_id = _sel_run_id

            # 按需加载该 run 的完整状态（含 judge_results）
            run_status = get_run_status(
                run_id,
                batch_dir=str(BATCH_DIR),
                raw_dir=str(RAW_DIR),
                processed_file=find_processed_for_run(run_id),
                judged_file=str(JUDGED_FILE),
                include_judge_results=True,
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

            with st.expander(f"{status_icon} {run_id} | 题集: {q_set_name}", expanded=True):
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

                # Judge 运行统计（从 manifest 读取）
                if run.get("judge_completed_at"):
                    _jscope = run.get("judge_duration_scope", "run")
                    if _jscope == "batch":
                        _jd = run.get("judge_batch_duration_seconds", 0)
                        _dur_label = "本批次总耗时"
                    else:
                        _jd = run.get("judge_duration_seconds", 0)
                        _dur_label = "总耗时"
                    _jl = run.get("judge_llm_call_count", 0)
                    _jp = run.get("judge_prescreened_count", 0)
                    _jc = run.get("judge_content_cached_count", 0)
                    _jw = run.get("judge_concurrency", 1)
                    _jm = run.get("judge_mode", "")
                    _ja = run.get("judge_completed_at", "")
                    _m, _s = divmod(int(_jd), 60)
                    _dur_str = f"{_m}分{_s:02d}秒" if _m else f"{_s}秒"

                    st.markdown("**Judge 运行统计**")
                    jstat_col1, jstat_col2, jstat_col3 = st.columns(3)
                    with jstat_col1:
                        st.metric(_dur_label, _dur_str)
                    with jstat_col2:
                        st.metric("LLM 调用", _jl)
                    with jstat_col3:
                        st.metric("并发数", _jw)
                    _detail_parts = [f"规则判定 {_jp} 条", f"内容复用 {_jc} 条"]
                    if _jm:
                        _mode_labels = {
                            "quick_test": "快速测试", "incremental": "增量评测",
                            "retry_failed": "失败重试", "force_all": "强制全量",
                        }
                        _detail_parts.append(f"模式: {_mode_labels.get(_jm, _jm)}")
                    if _jscope == "batch":
                        _detail_parts.append("耗时为跨 run 批次总耗时")
                    if _ja:
                        _detail_parts.append(f"完成于 {_ja[:19].replace('T', ' ')}")
                    st.caption(" | ".join(_detail_parts))

                # Judge 指标
                if judge_count > 0:
                    st.markdown("---")
                    st.markdown("**Judge 评测指标**")

                    judge_results = run_status.get("judge_results", [])
                    # 补齐旧版 chunk_exact 缺失的 Top10
                    _detail_rp_str, _detail_mtime = _resolve_processed_path()
                    _detail_lookup = _load_sample_lookup(_detail_mtime, _detail_rp_str)
                    for r in judge_results:
                        backfill_chunk_exact_topk(r, _detail_lookup)
                    if judge_results:
                        # 计算指标
                        valid_results = [r for r in judge_results if "error" not in r]
                        if valid_results:
                            metrics = compute_metrics(judge_results)

                            # 按轨道分组
                            retrieval_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_RETRIEVAL]
                            strict_qa_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_STRICT_QA]
                            grounded_qa_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_GROUNDED_QA]
                            chunk_exact_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_CHUNK_EXACT]

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

                            if chunk_exact_results:
                                chunk_metrics = compute_chunk_exact_metrics(chunk_exact_results)
                                st.markdown("##### Chunk Exact（独立机器判定，不纳入跨配置检索指标）")
                                ce_col1, ce_col2, ce_col3, ce_col4, ce_col5 = st.columns(5)
                                ce_col1.metric("总题数", chunk_metrics["total_count"])
                                ce_col2.metric("可评测", chunk_metrics["evaluable_count"])
                                ce_col3.metric("缺少绑定", chunk_metrics["missing_binding_count"])
                                ce_col4.metric("无真实 trace", chunk_metrics["no_trace_count"])
                                ce_col5.metric("无 retrieval", chunk_metrics["no_retrieval_count"])

                                if chunk_metrics["formal_usable"]:
                                    top_col1, top_col2, top_col3, top_col4 = st.columns(4)
                                    top_col1.metric("Chunk Exact Top1", f"{chunk_metrics['top1_hit_rate']:.0%}")
                                    top_col2.metric("Chunk Exact Top3", f"{chunk_metrics['top3_hit_rate']:.0%}")
                                    top_col3.metric("Chunk Exact Top5", f"{chunk_metrics['top5_hit_rate']:.0%}")
                                    top_col4.metric("Chunk Exact Top10", f"{chunk_metrics['top10_hit_rate']:.0%}")
                                else:
                                    st.warning(
                                        "当前 chunk_exact Judge 结果不可正式使用：pending 状态不计入 TopK miss。"
                                        "请从 frozen evidence snapshot 重新解析并创建新的 chunk_exact Judge 重试产物。"
                                    )

                # ========== 评测结果可视化 ==========
                if judge_count > 0:
                    judge_results_viz = run_status.get("judge_results", [])
                    if judge_results_viz:
                        valid_viz = [r for r in judge_results_viz if "error" not in r]
                        error_viz = [r for r in judge_results_viz if "error" in r]

                        # 加载当前 run 的 processed samples 构建 sample_map（从缓存过滤）
                        _rp_str, _proc_mtime_local = _resolve_processed_path()
                        _all_lookup = _load_sample_lookup(_proc_mtime_local, _rp_str)
                        # 补齐旧版 chunk_exact 缺失的 Top10
                        for r in judge_results_viz:
                            backfill_chunk_exact_topk(r, _all_lookup)
                        _run_sample_map = {}
                        for _tid, _pobj in _all_lookup.items():
                            _p_run_id = _pobj.get("run_id", "")
                            if not _p_run_id:
                                _p_uid = _pobj.get("user_id", "")
                                if _p_uid.startswith("rag_eval:"):
                                    _p_parts = _p_uid.split(":", 2)
                                    if len(_p_parts) == 3:
                                        _p_run_id = _p_parts[1]
                            if _p_run_id == run_id:
                                _run_sample_map[_tid] = _pobj

                        st.markdown("---")
                        st.markdown("##### 评测结果可视化")

                        # 检索评测轨道
                        retrieval_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_RETRIEVAL]
                        strict_qa_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_STRICT_QA]
                        grounded_qa_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_GROUNDED_QA]
                        chunk_exact_viz = [r for r in valid_viz if r.get("evaluation_track") == TRACK_CHUNK_EXACT]

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
                                st.markdown("**命中分布**")
                                _t1h = sum(1 for r in retrieval_viz if r.get("retrieval_top1_hit"))
                                _t3h = sum(1 for r in retrieval_viz if r.get("retrieval_top3_hit"))
                                _t5h = sum(1 for r in retrieval_viz if r.get("retrieval_top5_hit"))
                                _dist = pd.DataFrame([
                                    {"指标": "Top1 命中", "数量": _t1h, "占比": f"{_t1h/n:.0%}"},
                                    {"指标": "Top3 命中", "数量": _t3h, "占比": f"{_t3h/n:.0%}"},
                                    {"指标": "Top5 命中", "数量": _t5h, "占比": f"{_t5h/n:.0%}"},
                                ])
                                st.dataframe(_dist, use_container_width=True, hide_index=True)
                        elif not chunk_exact_viz:
                            st.info("当前运行无检索评测轨道数据")

                        # -- Chunk Exact 指标 --
                        if chunk_exact_viz:
                            st.markdown("**Chunk Exact（机器判定）**")
                            ce_evaluable = [r for r in chunk_exact_viz
                                            if r.get("retrieval_evaluable", True) is not False
                                            and r.get("retrieval_top1_hit") is not None]
                            ce_unevaluable = [r for r in chunk_exact_viz
                                              if r.get("retrieval_evaluable", True) is False
                                              or r.get("retrieval_top1_hit") is None]
                            ce_n = len(ce_evaluable)
                            ce_total = len(chunk_exact_viz)
                            if ce_n > 0:
                                ce_t1 = sum(r.get("retrieval_top1_hit", 0) for r in ce_evaluable)
                                ce_t3 = sum(r.get("retrieval_top3_hit", 0) for r in ce_evaluable)
                                ce_t5 = sum(r.get("retrieval_top5_hit", 0) for r in ce_evaluable)
                                ce_t10 = sum(r.get("retrieval_top10_hit", 0) for r in ce_evaluable)
                                _render_ce_topk(ce_n, ce_t1, ce_t3, ce_t5, ce_t10, ce_total)

                            if ce_unevaluable:
                                ue_counts = {}
                                for r in ce_unevaluable:
                                    s = r.get("chunk_exact_status") or "未知"
                                    ue_counts[s] = ue_counts.get(s, 0) + 1
                                ue_parts = [f"{s}: {c}" for s, c in ue_counts.items()]
                                st.caption(f"不可评测 {len(ce_unevaluable)} 条 — " + "、".join(ue_parts))

                            # 命中分布（Top1 / Top2-3 / Top4-5 / Top6-10 / Top10 miss）
                            if ce_n > 0:
                                _bucket_top1 = 0
                                _bucket_2_3 = 0
                                _bucket_4_5 = 0
                                _bucket_6_10 = 0
                                _bucket_miss = 0
                                for r in ce_evaluable:
                                    pos = r.get("hit_evidence_position")
                                    if pos is None:
                                        if r.get("retrieval_top10_hit"):
                                            # 有 top10 hit 但无 position（不应发生，防御性处理）
                                            _bucket_miss += 0
                                        else:
                                            _bucket_miss += 1
                                    elif pos <= 1:
                                        _bucket_top1 += 1
                                    elif pos <= 3:
                                        _bucket_2_3 += 1
                                    elif pos <= 5:
                                        _bucket_4_5 += 1
                                    elif pos <= 10:
                                        _bucket_6_10 += 1
                                    else:
                                        _bucket_miss += 1
                                # 对于旧记录：top10=1 但 position=None 的情况归入 Top6-10
                                # （因为旧 judge 只扫 top5，position=None 意味着命中在 6-10）
                                _no_pos_but_top10 = sum(
                                    1 for r in ce_evaluable
                                    if r.get("hit_evidence_position") is None
                                    and r.get("retrieval_top10_hit")
                                )
                                _bucket_6_10 += _no_pos_but_top10
                                _bucket_miss -= _no_pos_but_top10

                                st.markdown("**命中分布**")
                                _hit_dist = pd.DataFrame([
                                    {"区间": "Top1", "数量": _bucket_top1, "占比": f"{_bucket_top1/ce_n:.0%}"},
                                    {"区间": "Top2-3", "数量": _bucket_2_3, "占比": f"{_bucket_2_3/ce_n:.0%}"},
                                    {"区间": "Top4-5", "数量": _bucket_4_5, "占比": f"{_bucket_4_5/ce_n:.0%}"},
                                    {"区间": "Top6-10", "数量": _bucket_6_10, "占比": f"{_bucket_6_10/ce_n:.0%}"},
                                    {"区间": "Top10 miss", "数量": _bucket_miss, "占比": f"{_bucket_miss/ce_n:.0%}"},
                                ])
                                st.dataframe(_hit_dist, use_container_width=True, hide_index=True)

                            # 每条样本摘要
                            with st.expander(f"样本判定详情 ({ce_total} 条)", expanded=False):
                                for i, r in enumerate(ce_evaluable):
                                    seg_id = r.get("expected_segment_id", "")
                                    seg_short = seg_id[:12] + "..." if len(str(seg_id)) > 12 else seg_id
                                    pos = r.get("hit_evidence_position")
                                    t1 = r.get("retrieval_top1_hit", 0)
                                    t10 = r.get("retrieval_top10_hit", 0)
                                    hit_str = f"Top{pos}" if pos else "未命中"
                                    status_str = "✅ 命中" if t1 else "❌ 未命中"
                                    # Top10 状态
                                    if t10:
                                        t10_str = f"Top10 命中（第 {pos} 位）" if pos else "Top10 命中"
                                    else:
                                        t10_str = "Top10 未命中"
                                    st.caption(f"#{i+1} 目标 chunk: {seg_short} | 首次命中: {hit_str} | {status_str} | {t10_str}")

                                    # Top10 未命中时展示实际返回
                                    if not t10:
                                        sample = _run_sample_map.get(r.get("trace_id", ""), {})
                                        ret_results = sample.get("retrieval_results", [])[:10] if sample else []
                                        if ret_results:
                                            with st.expander("实际 Top10 返回", expanded=False):
                                                for j, rr in enumerate(ret_results):
                                                    _sid = rr.get("segment_id", rr.get("document_name", ""))
                                                    _sid_short = str(_sid)[:12] + "..." if len(str(_sid)) > 12 else str(_sid)
                                                    _score = rr.get("score", "")
                                                    _score_str = f"{_score:.4f}" if isinstance(_score, (int, float)) else str(_score)
                                                    st.caption(f"  Top{j+1}: {_sid_short} | score: {_score_str}")

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
                            if chunk_exact_viz:
                                track_labels.append("Chunk Exact")
                                track_values.append(len(chunk_exact_viz))
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

        # ========== 跨配置方案运行对比 ==========
        st.markdown("---")
        with st.expander("跨配置方案运行对比（点击展开）", expanded=False):
            st.markdown("比较不同配置方案下同一题集的检索结果差异。支持跨配置对比，也可用于同配置稳定性诊断。")

            # 读取全局所有 runs（不受当前 config_id 过滤）
            _all_runs = list_experiment_runs()
            _all_configs = {c["config_id"]: c for c in list_config_profiles()}

            # 按 question_set_id 分组（仅 completed runs 且有 question_set_id）
            _global_runs_by_qs = {}
            for _run in _all_runs:
                _qs = _run.get("question_set_id") or ""
                if _qs:
                    _global_runs_by_qs.setdefault(_qs, []).append(_run)

            # 仅保留有 >=2 个 run 的题集
            _eligible_qs = {qs: runs for qs, runs in _global_runs_by_qs.items() if len(runs) >= 2}

            if not _eligible_qs:
                st.info("全局没有相同题集的两次运行可供对比。请先确保至少两个 run 使用了相同的 question_set_id。")
            else:
                # Step 1: 选择题集
                _qs_options = sorted(_eligible_qs.keys())
                _qs_labels = []
                for _qs in _qs_options:
                    _qs_runs = _eligible_qs[_qs]
                    _qs_name = _qs_runs[0].get("question_set_name") or _qs[:40]
                    _config_ids = sorted(set(r.get("config_id", "") for r in _qs_runs))
                    _qs_labels.append(f"{_qs_name} | {len(_qs_runs)} 次运行, {len(_config_ids)} 个配置")

                _sel_qs_label = st.selectbox(
                    "Step 1: 选择题集（仅显示有 ≥2 次运行的题集）",
                    _qs_labels, key="_xcmp_qs",
                )
                _sel_qs = _qs_options[_qs_labels.index(_sel_qs_label)]
                _qs_runs = _eligible_qs[_sel_qs]

                # 按 config_id 分组该题集下的 runs
                _qs_runs_by_cfg = {}
                for _run in _qs_runs:
                    _cid = _run.get("config_id", "")
                    _qs_runs_by_cfg.setdefault(_cid, []).append(_run)

                # 辅助：构建 run label
                def _run_label(r):
                    _cid = r.get("config_id", "")
                    _cfg = _all_configs.get(_cid, {})
                    _cname = _cfg.get("config_name", _cid[:20])
                    _count = r.get("question_count", 0)
                    _time = (r.get("started_at") or "")[:16].replace("T", " ")
                    return f"{r['run_id'][:36]}... | {_cname} | {_time} | {_count}题"

                # Step 2: 选择基准（旧）
                _cfg_options = sorted(_qs_runs_by_cfg.keys())
                _cfg_labels = []
                for _cid in _cfg_options:
                    _cfg = _all_configs.get(_cid, {})
                    _cname = _cfg.get("config_name", _cid[:20])
                    _n = len(_qs_runs_by_cfg[_cid])
                    _cfg_labels.append(f"{_cname} ({_n} 次运行)")

                _diff_col1, _diff_col2 = st.columns(2)
                with _diff_col1:
                    st.markdown("**基准（旧）**")
                    _old_cfg_label = st.selectbox(
                        "配置方案", _cfg_labels, key="_xcmp_old_cfg",
                    )
                    _old_cfg_id = _cfg_options[_cfg_labels.index(_old_cfg_label)]
                    _old_cfg_runs = _qs_runs_by_cfg[_old_cfg_id]
                    _old_run_label = st.selectbox(
                        "运行", [_run_label(r) for r in _old_cfg_runs],
                        key="_xcmp_old_run",
                    )
                    _old_run = _old_cfg_runs[
                        [_run_label(r) for r in _old_cfg_runs].index(_old_run_label)
                    ]

                with _diff_col2:
                    st.markdown("**对比（新）**")
                    _new_cfg_label = st.selectbox(
                        "配置方案", _cfg_labels,
                        index=min(1, len(_cfg_labels) - 1),
                        key="_xcmp_new_cfg",
                    )
                    _new_cfg_id = _cfg_options[_cfg_labels.index(_new_cfg_label)]
                    _new_cfg_runs = _qs_runs_by_cfg[_new_cfg_id]
                    _new_run_label = st.selectbox(
                        "运行", [_run_label(r) for r in _new_cfg_runs],
                        key="_xcmp_new_run",
                    )
                    _new_run = _new_cfg_runs[
                        [_run_label(r) for r in _new_cfg_runs].index(_new_run_label)
                    ]

                # 验证
                _old_rid = _old_run["run_id"]
                _new_rid = _new_run["run_id"]
                _old_qs = _old_run.get("question_set_id", "")
                _new_qs = _new_run.get("question_set_id", "")

                if _old_rid == _new_rid:
                    st.warning("请选择两个不同的运行进行对比。")
                elif _old_qs != _new_qs:
                    st.error(
                        f"两个 run 的 question_set_id 不同，无法对比。"
                        f"旧: `{_old_qs[:30]}`，新: `{_new_qs[:30]}`"
                    )
                else:
                    if st.button("开始对比", key="_xcmp_go"):
                        from retrieval_diff import compare_runs
                        with st.spinner("正在对比检索结果..."):
                            try:
                                _diff = compare_runs(_old_rid, _new_rid)
                                st.session_state["_xcmp_result"] = _diff
                            except Exception as _diff_err:
                                st.error(f"对比失败: {_diff_err}")

                # 显示已有对比结果
                _diff = st.session_state.get("_xcmp_result")
                if _diff and _diff.get("summary"):
                    _s = _diff["summary"]
                    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # 对比元信息
                    st.markdown("---")
                    st.markdown("##### 对比元信息")
                    _meta_rows = [
                        {"项目": "基准 config_id", "值": _diff["old_config"].get("config_id", "N/A")},
                        {"项目": "基准 config_name", "值": _diff["old_config"].get("config_name", "N/A")},
                        {"项目": "基准 run_id", "值": _diff.get("old_run_id", _old_rid)},
                        {"项目": "对比 config_id", "值": _diff["new_config"].get("config_id", "N/A")},
                        {"项目": "对比 config_name", "值": _diff["new_config"].get("config_name", "N/A")},
                        {"项目": "对比 run_id", "值": _diff.get("new_run_id", _new_rid)},
                        {"项目": "question_set_id", "值": _sel_qs},
                    ]
                    st.dataframe(_meta_rows, use_container_width=True, hide_index=True)

                    # TopK Cutoff 分解
                    st.markdown("##### TopK Cutoff 分解")
                    _cutoff = _s.get("cutoff", {})
                    if _cutoff:
                        _cut_rows = []
                        for _K in (1, 3, 5):
                            _cs = _cutoff.get(_K, {})
                            _cut_rows.append({
                                "Cutoff": f"Top{_K}",
                                "旧命中": _cs.get("old_hit_count", 0),
                                "新命中": _cs.get("new_hit_count", 0),
                                "变化": _cs.get("delta", 0),
                                "loss": _cs.get("loss", 0),
                                "evidence_lost": _cs.get("evidence_lost", 0),
                                "ranking_drop": _cs.get("ranking_drop", 0),
                                "gain": _cs.get("gain", 0),
                            })
                        st.dataframe(_cut_rows, use_container_width=True, hide_index=True)

                    st.markdown(f"**诊断**: {_s.get('primary_cause_desc', '')}")

                    # 逐题分类摘要
                    _rows = _diff.get("rows", [])
                    _cat_counts = {}
                    for _r in _rows:
                        _c = _r.get("category", "unknown")
                        _cat_counts[_c] = _cat_counts.get(_c, 0) + 1
                    _cat_parts = [f"{k}={v}" for k, v in _cat_counts.items()]
                    st.caption(f"逐题分类: {' | '.join(_cat_parts)}（共 {len(_rows)} 题）")

                    # 可展开的分类明细
                    for _cat_name, _cat_label in [
                        ("evidence_lost", "evidence_lost 详情"),
                        ("ranking_regression", "ranking_regression 详情"),
                        ("judge_disagreement", "judge_disagreement 详情"),
                    ]:
                        _cat_rows = [r for r in _rows if r.get("category") == _cat_name]
                        if _cat_rows:
                            with st.expander(f"{_cat_label}（{len(_cat_rows)} 条）", expanded=False):
                                for _r in _cat_rows:
                                    _old_r = _r.get("old_rank", "—")
                                    _new_r = _r.get("new_rank", "—")
                                    st.markdown(
                                        f"- **{_r['question_id']}**: {_r.get('question', '')[:60]}"
                                        f"  | 旧 rank={_old_r} → 新 rank={_new_r}"
                                    )

                    # 下载
                    _dl_col1, _dl_col2 = st.columns(2)
                    with _dl_col1:
                        st.download_button(
                            label="下载 CSV",
                            data=_diff["csv_string"].encode("utf-8"),
                            file_name=f"retrieval_diff_{_ts}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with _dl_col2:
                        st.download_button(
                            label="下载 Markdown 报告",
                            data=_diff["markdown"].encode("utf-8"),
                            file_name=f"retrieval_diff_{_ts}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

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
                        processed_file=find_processed_for_run(rid),
                        judged_file=str(JUDGED_FILE),
                        include_judge_results=True,
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
