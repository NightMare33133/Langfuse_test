"""
AI 优化分析报告模块。

读取已完成的评测数据，通过多阶段 LLM 调用生成知识库优化诊断建议。
纯 Python 实现，不依赖 Streamlit。
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from judge import call_llm, compute_metrics, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA
from report_export import build_diagnostic_data, _compute_local_analysis, sanitize_filename_component

# ─── 常量 ────────────────────────────────────────────────────────────────────

_SENSITIVE_KEYS = frozenset({
    "_prompt", "_raw_response", "api_key", "secret_key",
    "cookie", "session_token", "password", "token",
    "observations", "root_input", "root_output",
})

_SENSITIVE_SNAPSHOT_KEYS = frozenset({
    "api_key", "secret_key", "lf_public_key", "lf_secret_key",
    "openai_api_key", "api_keys", "cookie", "session_token", "password", "token",
})

_ABS_PATH_PREFIXES = ("C:\\", "D:\\", "E:\\", "F:\\", "/", "\\\\")

_MAX_FAILURE_SAMPLES_PER_GROUP = 10
_MAX_CONTEXT_CHARS = 120000
_MAX_CONTENT_CHARS = 200
_MIN_GROUP_SAMPLE_COUNT = 3  # 分组样本数阈值：低于此值不生成强结论


# ─── 脱敏 ────────────────────────────────────────────────────────────────────

def sanitize_analysis_payload(data):
    """递归删除敏感字段和绝对路径。

    处理 dict / list / 嵌套结构：
    - 移除 _SENSITIVE_KEYS 中的键
    - 移除 _SENSITIVE_SNAPSHOT_KEYS 中的键（用于 config_snapshot）
    - 将以绝对路径前缀开头的字符串值替换为 "[REDACTED_PATH]"
    - 截断超长字符串到 _MAX_CONTENT_CHARS
    """
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if k in _SENSITIVE_KEYS or k in _SENSITIVE_SNAPSHOT_KEYS:
                continue
            cleaned[k] = sanitize_analysis_payload(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_analysis_payload(item) for item in data]
    elif isinstance(data, str):
        for prefix in _ABS_PATH_PREFIXES:
            if data.startswith(prefix):
                return "[REDACTED_PATH]"
        if len(data) > _MAX_CONTENT_CHARS:
            return data[:_MAX_CONTENT_CHARS] + "...(截断)"
        return data
    else:
        return data


# ─── 配置读取 ─────────────────────────────────────────────────────────────────

def get_analysis_config():
    """读取分析 LLM 配置。优先 ANALYSIS_*，回退 JUDGE_*。

    Returns:
        tuple: (api_key, base_url, model)
    """
    api_key = os.getenv("ANALYSIS_API_KEY") or os.getenv("JUDGE_API_KEY", "")
    base_url = (os.getenv("ANALYSIS_API_BASE")
                or os.getenv("JUDGE_API_BASE", "https://token-plan-cn.xiaomimimo.com/v1"))
    model = os.getenv("ANALYSIS_MODEL") or os.getenv("JUDGE_MODEL", "mimo-v2.5-pro")
    return api_key, base_url, model


# ─── 上下文构建 ───────────────────────────────────────────────────────────────

def _parse_config_values(config_snapshot):
    """从 config_snapshot 解析可读的配置值文本。

    将结构化配置转为 LLM 可直接引用的事实描述，
    避免 LLM 编造"当前值未知"等矛盾表述。
    """
    if not config_snapshot:
        return "config_snapshot 为空，无可用配置参数。"

    lines = []
    # 通用参数
    for key in ("top_k", "retrieval_mode", "retrieval_config", "embedding_model",
                "rerank_model", "knowledge_base_version", "workflow_version"):
        val = config_snapshot.get(key)
        if val is not None and val != "":
            lines.append(f"- {key} = {val}")

    # chunk_strategy 特殊处理：提取可读描述
    cs = config_snapshot.get("chunk_strategy", "")
    if cs:
        lines.append(f"- chunk_strategy = {cs}")
        # 尝试解析常见格式
        cs_str = str(cs)
        size_match = re.search(r'(?:最大块|max[_\s]?size|chunk[_\s]?size)[^\d]*(\d+)', cs_str, re.IGNORECASE)
        overlap_match = re.search(r'(?:重叠|overlap)[^\d]*(\d+)', cs_str, re.IGNORECASE)
        if size_match:
            lines.append(f"  - 解析: 最大块 = {size_match.group(1)}")
        if overlap_match:
            lines.append(f"  - 解析: 重叠 = {overlap_match.group(1)}")

    if not lines:
        return "config_snapshot 中无结构化配置参数。"

    return "\n".join(lines)


def _truncate_list(items, max_items, sort_key=None):
    """确定性截取列表：先排序再取前 N。"""
    if sort_key:
        items = sorted(items, key=sort_key)
    return items[:max_items]


def _truncate_context(context, max_chars=_MAX_CONTEXT_CHARS):
    """渐进缩减上下文直到不超过 max_chars。"""
    def _size():
        return len(json.dumps(context, ensure_ascii=False))

    if _size() <= max_chars:
        return context

    # 第一步：每组失败样本缩减到 5
    for section in ("top5_miss", "sorting_issues"):
        if section in context.get("failures", {}):
            context["failures"][section] = _truncate_list(
                context["failures"][section], 5,
                sort_key=lambda x: (x.get("source_file_name", ""), x.get("trace_id", "")),
            )
    if _size() <= max_chars:
        return context

    # 第二步：gold_evidence 和 judge_reason 截断到 100 字符
    for section in ("top5_miss", "sorting_issues"):
        for record in context.get("failures", {}).get(section, []):
            if len(record.get("gold_evidence", "")) > 100:
                record["gold_evidence"] = record["gold_evidence"][:100] + "..."
            if len(record.get("judge_reason", "")) > 100:
                record["judge_reason"] = record["judge_reason"][:100] + "..."
    if _size() <= max_chars:
        return context

    # 第三步：移除 by_source_format 分组
    context.get("groupings", {}).pop("by_source_format", None)
    if _size() <= max_chars:
        return context

    # 第四步：失败样本缩减到 20
    for section in ("top5_miss", "sorting_issues"):
        if section in context.get("failures", {}):
            context["failures"][section] = _truncate_list(
                context["failures"][section], 20,
                sort_key=lambda x: (x.get("source_file_name", ""), x.get("trace_id", "")),
            )

    return context


def build_analysis_context(run_data_list, sample_lookup, all_judge_results, config):
    """构建结构化分析上下文。

    Args:
        run_data_list: [{"run": dict, "run_status": dict, "metrics": dict}, ...]
        sample_lookup: {trace_id: processed_sample_dict}
        all_judge_results: 去重后的全部 judged results
        config: 配置方案 dict

    Returns:
        dict: 包含 overview, groupings, failures, config_summary, run_summaries, data_quality
    """
    # 聚合指标
    metrics = compute_metrics(all_judge_results) if all_judge_results else {}

    # 诊断数据
    diag = build_diagnostic_data(all_judge_results, sample_lookup, config)

    # 按轨道分组
    valid_results = [r for r in all_judge_results if "error" not in r]
    error_results = [r for r in all_judge_results if "error" in r]
    retrieval_results = [r for r in valid_results
                         if r.get("evaluation_track") == TRACK_RETRIEVAL
                         and r.get("retrieval_evaluable", True)]
    strict_qa_results = [r for r in valid_results
                         if r.get("evaluation_track") == TRACK_STRICT_QA]
    grounded_qa_results = [r for r in valid_results
                           if r.get("evaluation_track") == TRACK_GROUNDED_QA]
    not_evaluable_results = [r for r in valid_results
                             if r.get("evaluation_track") == "not_evaluable"]

    # 分组指标
    groupings = _compute_local_analysis(
        retrieval_results, strict_qa_results, grounded_qa_results,
        error_results, sample_lookup, diag,
    )

    # 运行摘要
    run_summaries = []
    for rd in run_data_list:
        run = rd.get("run", {})
        rs = rd.get("run_status", {})
        m = rd.get("metrics", {})
        run_summaries.append({
            "run_id": run.get("run_id", ""),
            "config_name": (run.get("config_snapshot") or {}).get("config_name", ""),
            "question_count": run.get("question_count", 0),
            "status": run.get("status", ""),
            "started_at": run.get("started_at", ""),
            "question_set_name": rs.get("question_set_name", ""),
            "batch_success": rs.get("batch_success", 0),
            "batch_total": rs.get("batch_total", 0),
            "judge_count": rs.get("judge_count", 0),
            "retrieval_top1_rate": m.get("retrieval_top1_hit_rate"),
            "retrieval_top5_rate": m.get("retrieval_top5_hit_rate"),
        })

    # 数据质量
    no_retrieval_count = sum(
        1 for s in sample_lookup.values()
        if not s.get("retrieval_results")
    )

    # 配置快照（取最新 run 的）
    config_snapshot = {}
    if run_data_list:
        latest_run = run_data_list[-1].get("run", {})
        config_snapshot = latest_run.get("config_snapshot") or {}

    # run_id -> question_set_name 映射（用于归属校验）
    run_id_to_question_set = {}
    for rd in run_data_list:
        run = rd.get("run", {})
        rs = rd.get("run_status", {})
        rid = run.get("run_id", "")
        qsn = rs.get("question_set_name") or run.get("question_set_name", "")
        if rid:
            run_id_to_question_set[rid] = qsn

    # 解析 chunk_strategy 为可读配置事实
    config_values_text = _parse_config_values(config_snapshot)

    # 确定性截取诊断样本
    diag["top5_miss"] = _truncate_list(
        diag["top5_miss"], _MAX_FAILURE_SAMPLES_PER_GROUP,
        sort_key=lambda x: (x.get("source_file_name", ""), x.get("trace_id", "")),
    )
    diag["sorting_issues"] = _truncate_list(
        diag["sorting_issues"], _MAX_FAILURE_SAMPLES_PER_GROUP,
        sort_key=lambda x: (x.get("source_file_name", ""), x.get("trace_id", "")),
    )

    # 截断诊断样本中的检索内容
    for section in ("top5_miss", "sorting_issues"):
        for record in diag.get(section, []):
            for rr in record.get("retrieval_results", []):
                content = rr.get("content", "")
                if len(content) > _MAX_CONTENT_CHARS:
                    rr["content"] = content[:_MAX_CONTENT_CHARS] + "...(截断)"

    total_questions = sum(rd.get("run", {}).get("question_count", 0) for rd in run_data_list)

    context = {
        "overview": {
            **metrics,
            "run_count": len(run_data_list),
            "total_questions": total_questions,
            "retrieval_sample_count": len(retrieval_results),
            "strict_qa_sample_count": len(strict_qa_results),
            "grounded_qa_sample_count": len(grounded_qa_results),
            "not_evaluable_count": len(not_evaluable_results),
            "error_count": len(error_results),
        },
        "groupings": groupings,
        "failures": diag,
        "config_summary": sanitize_analysis_payload(config_snapshot),
        "config_values_text": config_values_text,
        "run_id_to_question_set": run_id_to_question_set,
        "run_summaries": run_summaries,
        "data_quality": {
            "judge_errors": len(error_results),
            "not_evaluable": len(not_evaluable_results),
            "no_retrieval_results": no_retrieval_count,
            "total_processed_samples": len(sample_lookup),
        },
        "generation_timestamp": datetime.now().isoformat(),
    }

    return _truncate_context(context)


# ─── 确定性统计与事实渲染 ────────────────────────────────────────────────────

def compute_precise_stats(context):
    """从 context 确定性计算所有统计数字，不依赖 LLM。

    Returns:
        dict: 包含所有精确计算的统计字段
    """
    ov = context.get("overview", {})
    diag = context.get("failures", {})
    groupings = context.get("groupings", {})

    n = ov.get("retrieval_track_count", 0)
    t1_hit_n = round(ov.get("retrieval_top1_hit_rate", 0) * n) if n else 0
    t3_hit_n = round(ov.get("retrieval_top3_hit_rate", 0) * n) if n else 0
    t5_hit_n = round(ov.get("retrieval_top5_hit_rate", 0) * n) if n else 0
    top5_miss_n = diag.get("total_top5_miss", 0)
    ranking_issue_n = diag.get("total_sorting_issues", 0)

    # 评测轨道检测
    has_qa = ov.get("strict_qa_sample_count", 0) > 0 or ov.get("grounded_qa_sample_count", 0) > 0
    is_retrieval_only = not has_qa

    # 分组统计（带样本数阈值标记）
    def _annotate_groups(groups):
        annotated = []
        for g in groups:
            ag = dict(g)
            ag["sufficient_sample"] = g.get("count", 0) >= _MIN_GROUP_SAMPLE_COUNT
            annotated.append(ag)
        return annotated

    # 从 config_snapshot 提取已确认的配置键
    config_snapshot = context.get("config_summary", {})
    confirmed_config_keys = sorted(config_snapshot.keys()) if config_snapshot else []

    return {
        "retrieval_evaluable_n": n,
        "top1_hit_n": t1_hit_n,
        "top3_hit_n": t3_hit_n,
        "top5_hit_n": t5_hit_n,
        "top5_miss_n": top5_miss_n,
        "ranking_issue_n": ranking_issue_n,
        "top1_miss_n": n - t1_hit_n,
        "top5_hit_rate_pct": f"{ov.get('retrieval_top1_hit_rate', 0) * 100:.1f}" if n else "N/A",
        "top5_miss_rate_pct": f"{top5_miss_n / n * 100:.1f}" if n else "N/A",
        "is_retrieval_only": is_retrieval_only,
        "has_qa": has_qa,
        "qa_tracks_summary": {
            "strict_qa_n": ov.get("strict_qa_sample_count", 0),
            "strict_qa_correct": round(ov.get("strict_qa_answer_rate", 0) * ov.get("strict_qa_sample_count", 0)) if ov.get("strict_qa_sample_count") else 0,
            "grounded_qa_n": ov.get("grounded_qa_sample_count", 0),
            "grounded_qa_grounded": round(ov.get("grounded_qa_answer_rate", 0) * ov.get("grounded_qa_sample_count", 0)) if ov.get("grounded_qa_sample_count") else 0,
        },
        "by_source_file_annotated": _annotate_groups(groupings.get("by_source_file", [])),
        "by_topic_annotated": _annotate_groups(groupings.get("by_topic", [])),
        "by_difficulty_annotated": _annotate_groups(groupings.get("by_difficulty", [])),
        "confirmed_config_keys": confirmed_config_keys,
        "config_values_text": context.get("config_values_text", ""),
        "run_id_to_question_set": context.get("run_id_to_question_set", {}),
    }


def build_facts_section(context, stats):
    """用 Python 渲染"数据事实"Markdown 节，所有数字由 stats 确定性提供。

    Returns:
        str: Markdown 格式的数据事实文本
    """
    ov = context.get("overview", {})
    dq = context.get("data_quality", {})
    rs_list = context.get("run_summaries", [])
    diag = context.get("failures", {})

    lines = []

    # ── 1. 评测总览 ──
    lines.append("## 1. 评测总览\n")
    lines.append("**数据事实（由 Python 计算，不依赖 LLM）：**\n")
    lines.append(f"- 运行数: {ov.get('run_count', 0)}")
    lines.append(f"- 检索评测样本数 (retrieval_evaluable_n): **{stats['retrieval_evaluable_n']}**")
    lines.append(f"- Top1 命中数: **{stats['top1_hit_n']}** / {stats['retrieval_evaluable_n']}")
    lines.append(f"- Top3 命中数: **{stats['top3_hit_n']}** / {stats['retrieval_evaluable_n']}")
    lines.append(f"- Top5 命中数: **{stats['top5_hit_n']}** / {stats['retrieval_evaluable_n']}")
    lines.append(f"- Top5 完全未命中数 (top5_miss_n): **{stats['top5_miss_n']}**")
    lines.append(f"- 排序问题数 (ranking_issue_n，Top1 未中但 Top5 命中): **{stats['ranking_issue_n']}**")

    if stats["is_retrieval_only"]:
        lines.append("")
        lines.append("> 本报告范围为检索评测，不对最终答案质量作出结论。")
    else:
        qa = stats["qa_tracks_summary"]
        if qa["strict_qa_n"]:
            lines.append(f"- 严格问答: {qa['strict_qa_correct']} / {qa['strict_qa_n']}")
        if qa["grounded_qa_n"]:
            lines.append(f"- 合理性问答: {qa['grounded_qa_grounded']} / {qa['grounded_qa_n']}")

    lines.append(f"- Judge 错误: {ov.get('error_count', 0)}")
    lines.append(f"- 不可评测样本: {ov.get('not_evaluable_count', 0)}")
    lines.append("")

    # ── 2. 分析范围与数据质量 ──
    lines.append("## 2. 分析范围与数据质量\n")
    lines.append("**数据事实：**\n")
    for rs in rs_list:
        lines.append(f"- run_id={rs.get('run_id', '?')}: "
                     f"题目数={rs.get('question_count', 0)}, "
                     f"状态={rs.get('status', '?')}, "
                     f"Judge 完成={rs.get('judge_count', 0)}")
    lines.append(f"- 无检索结果样本数: {dq.get('no_retrieval_results', 0)}")
    lines.append(f"- 总处理样本数: {dq.get('total_processed_samples', 0)}")
    lines.append("")

    # ── 3. 按文件/主题的表现 ──
    lines.append("## 3. 按文件/主题的表现\n")
    lines.append("**数据事实：**\n")

    # 按文件（source_file 缺失时显示为"评测源题集/文档: {question_set_name}"）
    file_groups = stats["by_source_file_annotated"]
    if file_groups:
        lines.append("### 按源文件\n")
        lines.append("| 源文件 | 样本数 | Top1 | Top3 | Top5 | 样本充足 |")
        lines.append("|--------|--------|------|------|------|----------|")
        for g in file_groups:
            key = g['key']
            if key == "未记录":
                key = "评测源题集/文档（source_file 缺失）"
            t1r = f"{g['t1_rate'] * 100:.1f}%" if g.get("t1_rate") is not None else "N/A"
            t3r = f"{g['t3_rate'] * 100:.1f}%" if g.get("t3_rate") is not None else "N/A"
            t5r = f"{g['t5_rate'] * 100:.1f}%" if g.get("t5_rate") is not None else "N/A"
            suf = "是" if g["sufficient_sample"] else "否（待观察个例）"
            lines.append(f"| {key} | {g['count']} | {t1r} | {t3r} | {t5r} | {suf} |")
        lines.append("")

    # 按 topic
    topic_groups = stats["by_topic_annotated"]
    if topic_groups:
        lines.append("### 按 Topic\n")
        lines.append("| Topic | 样本数 | Top1 | Top3 | Top5 | 样本充足 |")
        lines.append("|-------|--------|------|------|------|----------|")
        for g in topic_groups:
            t1r = f"{g['t1_rate'] * 100:.1f}%" if g.get("t1_rate") is not None else "N/A"
            t3r = f"{g['t3_rate'] * 100:.1f}%" if g.get("t3_rate") is not None else "N/A"
            t5r = f"{g['t5_rate'] * 100:.1f}%" if g.get("t5_rate") is not None else "N/A"
            suf = "是" if g["sufficient_sample"] else "否（待观察个例）"
            lines.append(f"| {g['key']} | {g['count']} | {t1r} | {t3r} | {t5r} | {suf} |")
        lines.append("")

    # 按难度
    diff_groups = stats["by_difficulty_annotated"]
    if diff_groups:
        lines.append("### 按难度\n")
        lines.append("| 难度 | 样本数 | Top1 | Top3 | Top5 | 样本充足 |")
        lines.append("|------|--------|------|------|------|----------|")
        for g in diff_groups:
            t1r = f"{g['t1_rate'] * 100:.1f}%" if g.get("t1_rate") is not None else "N/A"
            t3r = f"{g['t3_rate'] * 100:.1f}%" if g.get("t3_rate") is not None else "N/A"
            t5r = f"{g['t5_rate'] * 100:.1f}%" if g.get("t5_rate") is not None else "N/A"
            suf = "是" if g["sufficient_sample"] else "否（待观察个例）"
            lines.append(f"| {g['key']} | {g['count']} | {t1r} | {t3r} | {t5r} | {suf} |")
        lines.append("")

    # ── 4. Top5 完全未命中样本清单 ──
    lines.append("## 4. Top5 完全未命中样本清单\n")
    lines.append("**数据事实：**\n")
    miss_list = diag.get("top5_miss", [])
    run_qsn_map = stats.get("run_id_to_question_set", {})
    if miss_list:
        lines.append(f"共 {stats['top5_miss_n']} 条 Top5 完全未命中（上下文中展示了 {len(miss_list)} 条）：\n")
        for d in miss_list:
            rid = d.get('run_id', '?')
            fname = d.get('source_file_name', '')
            if not fname or fname == "未记录":
                qsn = run_qsn_map.get(rid, "")
                fname_display = f"评测源题集/文档: {qsn}" if qsn else "归属信息缺失"
            else:
                fname_display = fname
            lines.append(f"- run_id={rid} | trace_id={d.get('trace_id', '?')} | "
                         f"query={d.get('question', '?')[:60]} | "
                         f"file={fname_display} | topic={d.get('topic', '?')}")
    else:
        lines.append("无 Top5 完全未命中。")
    lines.append("")

    # ── 5. 排序问题样本清单 ──
    lines.append("## 5. 排序问题样本清单\n")
    lines.append("**数据事实：**\n")
    sorting_list = diag.get("sorting_issues", [])
    if sorting_list:
        lines.append(f"共 {stats['ranking_issue_n']} 条排序问题（上下文中展示了 {len(sorting_list)} 条）：\n")
        for d in sorting_list:
            rid = d.get('run_id', '?')
            fname = d.get('source_file_name', '')
            if not fname or fname == "未记录":
                qsn = run_qsn_map.get(rid, "")
                fname_display = f"评测源题集/文档: {qsn}" if qsn else "归属信息缺失"
            else:
                fname_display = fname
            lines.append(f"- run_id={rid} | trace_id={d.get('trace_id', '?')} | "
                         f"query={d.get('question', '?')[:60]} | "
                         f"hit_position={d.get('hit_evidence_position', '?')} | "
                         f"file={fname_display}")
    else:
        lines.append("无排序问题。")
    lines.append("")

    # ── 6. 已确认配置参数 ──
    lines.append("## 6. 已确认配置参数\n")
    lines.append("**数据事实（仅列出 config_snapshot 中实际存在的参数）：**\n")
    config_values_text = stats.get("config_values_text", "")
    if config_values_text:
        lines.append(config_values_text)
    else:
        lines.append("config_snapshot 为空，无可用配置参数。")
    lines.append("")
    lines.append("> 注意：以上为 config_snapshot 中实际记录的值。缺少独立 chunk_size/overlap 字段时，")
    lines.append("> 不得编造具体数值，只能以 chunk_strategy 为基线建议单变量实验。")
    lines.append("")

    return "\n".join(lines)


def build_scope_note(context, stats):
    """生成评测轨道范围说明。"""
    if stats["is_retrieval_only"]:
        return "> **评测轨道说明**: 本次评测范围仅包含检索评测（retrieval），不包含严格问答（strict_qa）或合理性问答（grounded_qa）。本报告不对最终答案质量作出结论，不建议补充 QA 评测作为优化方向。"
    return ""


# ─── 阶段 1：整体概览 ────────────────────────────────────────────────────────

_STAGE1_PROMPT = """\
你是一位 RAG 系统评测分析专家。请基于以下评测数据，识别关键模式和异常区域。

## 输入数据
```json
{context_json}
```

## 分析要求
1. 总结整体检索质量（Top1/3/5 命中率）
2. 识别指标显著低于均值的文件/话题/难度分组
3. 指出跨运行的趋势变化（如有）
4. 标记数据质量问题（缺失样本、评测错误等）

## 输出格式
使用 Markdown，分为：
- **整体评估**（1-3 句话）
- **异常分组清单**（表格：分组维度 | 分组值 | 样本数 | Top1 | Top5 | 偏差说明）
- **数据质量警示**

不要提出优化建议，只描述事实和模式。"""


def analyze_overview(context, api_key, base_url, model, timeout=120):
    """阶段 1：整体概览分析。

    Args:
        context: build_analysis_context 返回的上下文
        api_key, base_url, model: LLM 配置
        timeout: 请求超时秒数

    Returns:
        str: LLM 返回的 Markdown 分析文本
    """
    # 只传 overview + groupings + config_summary，不传 failures
    stage1_context = {
        "overview": context.get("overview", {}),
        "groupings": context.get("groupings", {}),
        "config_summary": context.get("config_summary", {}),
        "run_summaries": context.get("run_summaries", []),
        "data_quality": context.get("data_quality", {}),
    }
    context_json = json.dumps(stage1_context, ensure_ascii=False, indent=2)
    prompt = _STAGE1_PROMPT.format(context_json=context_json)
    return call_llm(prompt, api_key, base_url, model, timeout=timeout)


# ─── 阶段 2：失败诊断（map-reduce） ──────────────────────────────────────────

_SUB_BATCH_MAX_SIZE = 4
_SLIM_GOLD_EVIDENCE_MAX = 160
_SLIM_CONTENT_MAX = 120
_SLIM_MAX_RETRIEVAL_RESULTS = 5

_STAGE2_SUB_PROMPT = """\
你是一位 RAG 检索系统诊断专家。请分析以下失败样本子批次的共性模式。

## 分组上下文
- 失败类型: {failure_type}（top5_miss=Top5 完全未命中, sorting_issue=排序问题）
- 来源标识: {source_key}
- 本组总样本数: {group_total}
- 本批次样本数: {batch_count}

## 本批次样本
```json
{batch_json}
```

## 分析要求
1. 识别本批次样本的共性特征；若无稳定共性，明确写"无稳定共性"
2. 提出可能的根因假设（明确标注为"假设"）
3. 每个判断必须附带审计引用: [run_id=... | trace_id=...]
4. 不得自行统计全局数字，只分析本批次
5. 不得输出统计摘要或数据事实部分

## 输出格式
#### 子批次诊断: {batch_id}
- **共性特征**: ... 或"无稳定共性"
- **根因假设**: ...
- **审计引用**:
  - [run_id=xxx | trace_id=xxx]: 简要说明"""

_STAGE2_REDUCE_PROMPT = """\
你是一位 RAG 系统优化顾问。请基于以下多个子批次的失败诊断结果，生成统一的失败模式分析。

## 全局统计（由 Python 精确计算，不可修改）
- retrieval_evaluable_n: {retrieval_evaluable_n}
- top5_miss_n: {top5_miss_n}
- ranking_issue_n: {ranking_issue_n}

## 各分组样本数
{group_stats_text}

## 子批次诊断结果
{sub_batch_summaries}

## 未完成的子批次（AI 诊断不可用）
{failed_batches_text}

## 分析要求
1. 合并各子批次的诊断，按失败模式优先级排序
2. 样本数 >= 3 的组称为"失败模式"；样本数 < 3 的组标注为"待观察个例"
3. 优先分析 Top5 完全未命中（主要召回问题），其次分析排序问题
4. 识别跨文件/跨题集的共性模式
5. 提出下一轮单变量实验建议
6. 对未完成的子批次，明确标注"AI 诊断不可用"，不得伪装为已分析
7. 每条诊断附审计引用: [run_id=... | trace_id=...]

## 输出格式
使用 Markdown：

### 失败模式优先级排序
（按样本数和严重性排序）

### 跨文件共性
（如有）

### 各失败模式详情
#### 模式 N: [名称]（[召回/排序]）
- **样本数**: X（是否达到阈值 3）
- **共性特征**: ...
- **根因假设**: ...
- **审计引用**: ...

### 待观察个例
（样本数 < 3 的组）

### AI 诊断不可用的样本
（如有未完成批次）

### 下一轮实验建议
（按预期收益排序）"""


def group_failures_for_analysis(failures):
    """将全部失败样本按 (failure_type, source_key) 分组。

    不截断，保留全部失败样本。

    Args:
        failures: dict with "top5_miss" and "sorting_issues" lists

    Returns:
        list of group dicts: {failure_type, source_key, samples, count, trace_ids}
    """
    groups = {}  # (failure_type, source_key) -> group

    for failure_type in ("top5_miss", "sorting_issues"):
        for sample in failures.get(failure_type, []):
            source_key = (sample.get("source_file_name") or "").strip()
            if not source_key:
                source_key = (sample.get("question_set_name") or "").strip()
            if not source_key:
                source_key = "未归类"

            key = (failure_type, source_key)
            if key not in groups:
                groups[key] = {
                    "failure_type": failure_type,
                    "source_key": source_key,
                    "samples": [],
                    "trace_ids": [],
                }
            groups[key]["samples"].append(sample)
            groups[key]["trace_ids"].append(sample.get("trace_id", ""))

    result = []
    for (ft, sk), group in sorted(groups.items()):
        group["count"] = len(group["samples"])
        result.append(group)
    return result


def _slim_sample(sample):
    """将单个失败样本精简为子批次 payload。"""
    slim = {
        "run_id": sample.get("run_id", ""),
        "trace_id": sample.get("trace_id", ""),
        "query": sample.get("retrieval_query", ""),
        "gold_evidence": (sample.get("gold_evidence") or "")[:_SLIM_GOLD_EVIDENCE_MAX],
        "hit_position": sample.get("hit_evidence_position"),
    }
    raw_results = sample.get("retrieval_results") or []
    slim_results = []
    for rr in raw_results[:_SLIM_MAX_RETRIEVAL_RESULTS]:
        slim_results.append({
            "position": rr.get("position"),
            "document_name": rr.get("document_name", ""),
            "content": (rr.get("content") or "")[:_SLIM_CONTENT_MAX],
        })
    slim["retrieval_results"] = slim_results
    return slim


def _split_into_sub_batches(group, max_batch_size=_SUB_BATCH_MAX_SIZE):
    """将一个分组切分为多个子批次，每个子批次 payload 已精简。"""
    samples = group["samples"]
    batches = []
    for i in range(0, len(samples), max_batch_size):
        chunk = samples[i:i + max_batch_size]
        batch_id = f"{group['failure_type']}_{group['source_key']}_{i // max_batch_size}"
        batches.append({
            "batch_id": batch_id,
            "payload": [_slim_sample(s) for s in chunk],
            "trace_ids": [s.get("trace_id", "") for s in chunk],
        })
    return batches


def _analyze_sub_batch(batch_id, batch_payload, group_context, stats,
                       api_key, base_url, model, timeout=120):
    """分析单个子批次，带重试逻辑。"""
    batch_json = json.dumps(batch_payload, ensure_ascii=False, indent=2)
    prompt = _STAGE2_SUB_PROMPT.format(
        failure_type=group_context["failure_type"],
        source_key=group_context["source_key"],
        group_total=group_context["group_total"],
        batch_count=len(batch_payload),
        batch_json=batch_json,
        batch_id=batch_id,
    )
    payload_chars = len(prompt)
    trace_ids = [s.get("trace_id", "") for s in batch_payload]

    try:
        output = call_llm(prompt, api_key, base_url, model, timeout=timeout)
        return {
            "batch_id": batch_id, "status": "ok",
            "output": output, "trace_ids": trace_ids,
            "payload_chars": payload_chars,
        }
    except Exception as e:
        # 重试：进一步截断内容
        try:
            for s in batch_payload:
                for rr in s.get("retrieval_results", []):
                    if len(rr.get("content", "")) > 60:
                        rr["content"] = rr["content"][:60] + "..."
                if len(s.get("gold_evidence", "")) > 80:
                    s["gold_evidence"] = s["gold_evidence"][:80] + "..."
            batch_json_retry = json.dumps(batch_payload, ensure_ascii=False, indent=2)
            prompt_retry = _STAGE2_SUB_PROMPT.format(
                failure_type=group_context["failure_type"],
                source_key=group_context["source_key"],
                group_total=group_context["group_total"],
                batch_count=len(batch_payload),
                batch_json=batch_json_retry,
                batch_id=batch_id,
            )
            output = call_llm(prompt_retry, api_key, base_url, model, timeout=timeout)
            return {
                "batch_id": batch_id, "status": "ok",
                "output": output, "trace_ids": trace_ids,
                "payload_chars": len(prompt_retry),
            }
        except Exception as e2:
            return {
                "batch_id": batch_id, "status": "failed",
                "error": str(e2), "trace_ids": trace_ids,
                "payload_chars": payload_chars,
            }


def _build_python_failure_manifest(failures):
    """Python 生成的完整失败清单（确定性，无 LLM）。"""
    lines = ["## Python 生成的完整失败清单\n"]
    for section, label in [("top5_miss", "Top5 完全未命中"), ("sorting_issues", "排序问题")]:
        items = failures.get(section, [])
        if not items:
            continue
        lines.append(f"### {label}（{len(items)} 条）\n")
        for s in items:
            tid = s.get("trace_id", "")
            rid = s.get("run_id", "")
            q = (s.get("retrieval_query") or s.get("question") or "")[:60]
            gold = (s.get("gold_evidence") or "")[:80]
            hit = s.get("hit_evidence_position")
            lines.append(f"- [{rid} | {tid}] query: {q}")
            lines.append(f"  金标准: {gold} | hit_position: {hit}")
    return "\n".join(lines)


def analyze_failure_groups(context, api_key, base_url, model, timeout=120,
                           progress_callback=None):
    """阶段 2：map-reduce 失败诊断。

    1. Python 确定性分组（不截断）
    2. 每组切分子批次，逐批调用 LLM
    3. 汇总所有子批次结果

    Args:
        context: build_analysis_context 返回的上下文
        api_key, base_url, model: LLM 配置
        timeout: 请求超时秒数
        progress_callback: callable(phase, detail) 进度回调

    Returns:
        str: 完整的 Stage 2 Markdown（含 Python 清单 + AI 诊断）
    """
    stats = compute_precise_stats(context)
    failures = context.get("failures", {})

    # Phase 1: Python 分组
    groups = group_failures_for_analysis(failures)
    total_failures = sum(g["count"] for g in groups)

    # 构建子批次
    all_batches = []
    for group in groups:
        batches = _split_into_sub_batches(group)
        for b in batches:
            b["group_context"] = {
                "failure_type": group["failure_type"],
                "source_key": group["source_key"],
                "group_total": group["count"],
            }
        all_batches.extend(batches)

    if progress_callback:
        progress_callback("grouping", {
            "total_failures": total_failures,
            "group_count": len(groups),
            "batch_count": len(all_batches),
        })

    # Phase 2: 逐批 LLM 分析
    batch_results = []
    for i, batch in enumerate(all_batches):
        result = _analyze_sub_batch(
            batch["batch_id"], batch["payload"],
            batch["group_context"], stats,
            api_key, base_url, model, timeout=timeout,
        )
        batch_results.append(result)
        if progress_callback:
            progress_callback("sub_batch", {
                "batch_index": i + 1,
                "total_batches": len(all_batches),
                "batch_id": batch["batch_id"],
                "status": result["status"],
                "payload_chars": result.get("payload_chars", 0),
            })

    # Phase 3: 汇总
    if progress_callback:
        progress_callback("synthesis", {"status": "started"})

    ok_results = [r for r in batch_results if r["status"] == "ok"]
    failed_results = [r for r in batch_results if r["status"] == "failed"]

    # 构建汇总输入
    sub_batch_summaries = []
    for r in ok_results:
        sub_batch_summaries.append(f"#### {r['batch_id']}\n{r['output']}")
    sub_batch_text = "\n\n".join(sub_batch_summaries) if sub_batch_summaries else "（无成功子批次）"

    group_stats_lines = []
    for g in groups:
        ft_label = "Top5 未命中" if g["failure_type"] == "top5_miss" else "排序问题"
        group_stats_lines.append(f"- {ft_label} / {g['source_key']}: {g['count']} 条")
    group_stats_text = "\n".join(group_stats_lines)

    failed_lines = []
    for r in failed_results:
        failed_lines.append(f"- {r['batch_id']}: {r['error']}（涉及 trace_id: {', '.join(r['trace_ids'][:3])}...）")
    failed_text = "\n".join(failed_lines) if failed_lines else "无"

    # 若全部失败，直接返回 Python 清单 + 错误说明
    if not ok_results:
        manifest = _build_python_failure_manifest(failures)
        ai_section = (
            f"\n\n## AI 诊断\n\n所有 {len(all_batches)} 个子批次均失败，"
            f"AI 诊断不可用。失败原因：\n{failed_text}"
        )
        if progress_callback:
            progress_callback("done", {"status": "all_failed"})
        return manifest + ai_section

    reduce_prompt = _STAGE2_REDUCE_PROMPT.format(
        retrieval_evaluable_n=stats["retrieval_evaluable_n"],
        top5_miss_n=stats["top5_miss_n"],
        ranking_issue_n=stats["ranking_issue_n"],
        group_stats_text=group_stats_text,
        sub_batch_summaries=sub_batch_text,
        failed_batches_text=failed_text,
    )
    ai_diagnosis = call_llm(reduce_prompt, api_key, base_url, model, timeout=timeout)

    if progress_callback:
        progress_callback("synthesis", {"status": "completed"})

    # 组装最终输出
    manifest = _build_python_failure_manifest(failures)
    ai_section = f"\n\n## AI 诊断（基于 {len(ok_results)}/{len(all_batches)} 个子批次分析）\n\n{ai_diagnosis}"

    if progress_callback:
        progress_callback("done", {
            "status": "completed",
            "total_failures": total_failures,
            "batch_count": len(all_batches),
            "ok_count": len(ok_results),
            "failed_count": len(failed_results),
        })

    return manifest + ai_section


# ─── 阶段 3：汇总报告 ────────────────────────────────────────────────────────

_STAGE3_PROMPT = """\
你是一位 RAG 系统优化顾问。请基于以下分析结果，生成"诊断假设"和"建议实验"部分。

## 阶段 1 分析结果（整体概览）
{stage1_output}

## 阶段 2 分析结果（失败诊断）
{stage2_output}

## 已确认配置参数及其实际值
{config_values_text}

## 已确认配置键列表（仅这些参数存在于 config_snapshot 中）
{confirmed_config_keys}

## 统计摘要（由 Python 精确计算，不可修改或复述）
- retrieval_evaluable_n: {retrieval_evaluable_n}
- top1_hit_n: {top1_hit_n}
- top3_hit_n: {top3_hit_n}
- top5_hit_n: {top5_hit_n}
- top5_miss_n: {top5_miss_n}
- ranking_issue_n: {ranking_issue_n}
- is_retrieval_only: {is_retrieval_only}

## run_id 到题集归属映射（用于校验引用归属）
{run_attribution_map}

## 输出要求

请输出以下章节的 Markdown 内容。**不要输出"数据事实"部分**（已由 Python 渲染），只输出"诊断假设"和"建议实验"。

### 你需要输出的章节

#### 7. 整体检索表现诊断
- **诊断假设**: 基于 Top1/3/5 数据的推断（标注为"假设"）
- **建议实验**: 如有改进空间，建议一个可验证的单变量实验

#### 8. 按文件/主题的弱项诊断
- 仅对样本数 >= 3 的分组生成诊断假设
- 样本数 < 3 的分组标注为"待观察个例，数据不足以形成结论"
- 每条附审计引用，且引用的 trace 必须属于该分组对应的 run_id

#### 9. Top5 完全未命中诊断（主要召回问题）
- 按失败模式分组，每组的根因假设
- 优先级最高，因为这是召回缺失
- 每条附 [run_id=... | trace_id=...]，且 trace 必须属于对应 run

#### 10. 排序问题诊断（次要排序问题）
- 排序问题的共性模式
- 优先级低于 Top5 完全未命中
- 每条附 [run_id=... | trace_id=...]

#### 11. 优先级排序的下一轮实验建议
- 按预期收益排序
- **Top5 完全未命中问题的建议必须排在排序问题之前**
- 每条建议只验证一个明确变量
- 只能建议 config_snapshot 中已确认的配置项，或通用 RAG 参数（chunk_size、overlap、top_k、retrieval_mode）
- 不得建议添加 reranker prompt、修改模型架构等无法确认 Dify 支持的操作
- 缺少配置值时，建议"以当前值为基线做单变量实验"，不编造具体数值

#### 12. 局限性与人工验证事项
- 本分析的假设和局限
- 样本量不足的分组

### 关键规则

- **不得输出"数据事实"**，所有数字以"统计摘要"为准，不得复述或重新计算
- 每条诊断假设和建议实验必须附审计引用: `[run_id=... | trace_id=...]`
- **引用归属校验**: 每个引用的 trace_id 必须属于其标注的 run_id。不得使用 run_A 的 trace 作为 run_B 文档的证据。若无法验证归属，写"归属信息缺失"
- 建议实验只验证一个明确变量，不得建议自动修改配置
- **配置值引用规则**: 上方"已确认配置参数"中列出的值即为实际值。当 top_k、retrieval_mode 等已有明确值时，必须引用该值，不得写"当前值未知""如果当前是语义检索"等矛盾表述
- **chunk_strategy 规则**: 若 chunk_strategy 已记录（如"最大块1000、重叠120"），以此为事实基线。缺少独立 chunk_size/overlap 字段时，不得生成"减少20%-30%"等具体百分比，只能建议"以当前 chunk_strategy 为基线，单变量测试更小块或更高重叠"
- **top_k 规则**: 当 top_k 有明确值时，可建议"测试增大 top_k"，但必须明确说明 Top1/3/5 仍按固定指标口径比较，增大 top_k 不改变评测标准
- 不得输出 API Key、Secret Key、绝对路径等敏感信息
- 样本数 < 3 的分组不得称为"失败模式"或"弱项"，只能标注为"待观察个例"
- 当 is_retrieval_only=true 时，不得将 QA=0 当作数据质量问题，不得建议补 QA 评测"""


def synthesize_optimization_report(stage1_output, stage2_output, context,
                                    api_key, base_url, model, timeout=120):
    """阶段 3：汇总生成最终 Markdown 报告。

    "数据事实"部分由 Python 渲染（compute_precise_stats + build_facts_section），
    LLM 只输出"诊断假设"和"建议实验"。

    Args:
        stage1_output: 阶段 1 LLM 输出
        stage2_output: 阶段 2 LLM 输出
        context: build_analysis_context 返回的上下文
        api_key, base_url, model: LLM 配置
        timeout: 请求超时秒数

    Returns:
        str: 最终 Markdown 报告
    """
    stats = compute_precise_stats(context)
    timestamp = context.get("generation_timestamp", datetime.now().isoformat())

    # Python 渲染的事实部分
    facts_md = build_facts_section(context, stats)
    scope_note = build_scope_note(context, stats)

    # LLM 只生成假设和建议
    config_snapshot = context.get("config_summary", {})
    confirmed_keys_str = ", ".join(f"`{k}`" for k in stats["confirmed_config_keys"]) or "（无）"
    config_values_text = stats.get("config_values_text", "无配置参数。")
    run_attribution_map = "\n".join(
        f"- {rid}: {qsn}" for rid, qsn in sorted(stats.get("run_id_to_question_set", {}).items())
    ) or "（无运行记录）"
    prompt = _STAGE3_PROMPT.format(
        stage1_output=stage1_output,
        stage2_output=stage2_output,
        config_values_text=config_values_text,
        confirmed_config_keys=confirmed_keys_str,
        retrieval_evaluable_n=stats["retrieval_evaluable_n"],
        top1_hit_n=stats["top1_hit_n"],
        top3_hit_n=stats["top3_hit_n"],
        top5_hit_n=stats["top5_hit_n"],
        top5_miss_n=stats["top5_miss_n"],
        ranking_issue_n=stats["ranking_issue_n"],
        is_retrieval_only=str(stats["is_retrieval_only"]),
        run_attribution_map=run_attribution_map,
    )
    llm_hypotheses = call_llm(prompt, api_key, base_url, model, timeout=timeout)

    # 组装最终报告
    header = (
        f"# AI 优化分析报告\n\n"
        f"> **AI 诊断建议，不替代人工验证**\n"
        f"> 生成时间: {timestamp}\n"
        f"> 分析模型: {model}\n"
    )
    if scope_note:
        header += f"\n{scope_note}\n"

    report = f"{header}\n---\n\n{facts_md}\n---\n\n{llm_hypotheses}\n"
    return report


# ─── 报告保存 ─────────────────────────────────────────────────────────────────


def build_report_filename(config_name, timestamp=None):
    """生成统一的报告文件名：{config_name}_ai_analysis_{timestamp}.md"""
    safe_name = sanitize_filename_component(config_name)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_ai_analysis_{timestamp}.md"


def save_analysis_report(markdown_content, config_name, reports_dir):
    """保存 Markdown 报告到 data/reports/ 目录。

    Args:
        markdown_content: 完整 Markdown 报告字符串
        config_name: 配置名，用于文件名
        reports_dir: 报告目录路径

    Returns:
        Path: 保存的文件路径
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    filename = build_report_filename(config_name)
    filepath = reports_dir / filename
    filepath.write_text(markdown_content, encoding="utf-8")
    return filepath
