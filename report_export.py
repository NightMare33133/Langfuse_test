"""
RAG 评测报告导出模块。

生成自包含 HTML 报告和 CSV 明细，不依赖 Streamlit。
"""

import csv
import io
import json
import re
from datetime import datetime
from html import escape

from judge import compute_metrics, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE

# 敏感字段黑名单
_SENSITIVE_KEYS = frozenset({
    "_prompt", "_raw_response", "api_key", "secret_key",
    "cookie", "session_token", "password", "token",
    "observations", "root_input", "root_output",
})

# 绝对路径前缀（Windows + Unix）
_ABS_PATH_PREFIXES = ("C:\\", "D:\\", "E:\\", "/", "\\\\")

# 每类诊断样本最大条数
_MAX_DIAGNOSTIC_SAMPLES = 50


def _sanitize_result(r):
    """移除敏感字段，返回安全副本。"""
    return {k: v for k, v in r.items() if k not in _SENSITIVE_KEYS}


def _pct(v):
    """格式化百分比。"""
    return f"{v:.1%}" if v is not None else "N/A"


def _safe_str(v):
    """安全字符串转义。"""
    return escape(str(v)) if v is not None else ""


def _fmt_content(text, max_len=500):
    """格式化检索内容：保留换行，超长时截断并标记。"""
    if not text:
        return "", False
    text = str(text)
    if len(text) <= max_len:
        return text, False
    return text[:max_len] + "...(截断)", True


# ====== 诊断数据构建 ======

def build_diagnostic_data(judge_results, sample_lookup, config=None, max_samples=_MAX_DIAGNOSTIC_SAMPLES):
    """为 Top5 未命中和排序问题样本构建诊断数据。

    从 judged result 通过 trace_id 关联 processed sample，补全检索结果详情。

    Args:
        judge_results: 去重后的全部 judged results
        sample_lookup: {trace_id: processed_sample_dict}
        config: 配置方案 dict
        max_samples: 每类最大条数

    Returns:
        dict: {
            "top5_miss": [...],       # Top5 完全未命中
            "sorting_issues": [...],  # Top1 miss 但 Top3 或 Top5 hit
            "total_top5_miss": int,
            "total_sorting_issues": int,
        }
    """
    config = config or {}
    valid = [r for r in judge_results if "error" not in r]
    retrieval = [r for r in valid
                 if r.get("evaluation_track") == TRACK_RETRIEVAL
                 and r.get("retrieval_evaluable", True)]

    top5_miss = []
    sorting_issues = []

    for r in retrieval:
        t1 = r.get("retrieval_top1_hit", 0)
        t5 = r.get("retrieval_top5_hit", 0)

        if t5 == 0:
            category = "top5_miss"
        elif t1 == 0:
            category = "sorting_issues"
        else:
            continue

        record = _build_one_diagnostic(r, sample_lookup, config)
        if category == "top5_miss":
            top5_miss.append(record)
        else:
            sorting_issues.append(record)

    total_top5 = len(top5_miss)
    total_sorting = len(sorting_issues)
    top5_miss = top5_miss[:max_samples]
    sorting_issues = sorting_issues[:max_samples]

    return {
        "top5_miss": top5_miss,
        "sorting_issues": sorting_issues,
        "total_top5_miss": total_top5,
        "total_sorting_issues": total_sorting,
    }


def _build_one_diagnostic(judge_result, sample_lookup, config):
    """为单个 judged result 构建诊断记录。"""
    tid = judge_result.get("trace_id", "")
    sample = sample_lookup.get(tid)

    base = {
        "trace_id": tid,
        "run_id": judge_result.get("_source_run_id", judge_result.get("run_id", "")),
        "question": judge_result.get("question", ""),
        "evaluation_track": judge_result.get("evaluation_track", ""),
        "hit_evidence_position": judge_result.get("hit_evidence_position"),
        "judge_reason": judge_result.get("reason", ""),
        "config_name": config.get("config_name", ""),
        "config_id": config.get("config_id", ""),
        "knowledge_base_version": config.get("knowledge_base_version", ""),
        "workflow_version": config.get("workflow_version", ""),
        "question_id": judge_result.get("question_id") or "",
        "question_set_id": judge_result.get("question_set_id") or "",
        "topic": judge_result.get("topic") or "",
        "difficulty": judge_result.get("difficulty") or "",
    }

    if sample is None:
        base.update({
            "diagnostic_status": "no_processed_sample",
            "retrieval_query": "",
            "gold_evidence": judge_result.get("source_excerpt") or judge_result.get("reference_answer") or "",
            "retrieval_results": [],
            "retrieval_result_count": 0,
            "final_answer": "",
            "source_format": "",
            "source_file_name": "",
            "evidence_sheet": "",
            "evidence_range": "",
        })
    else:
        gold = (sample.get("source_excerpt") or sample.get("reference_answer")
                or judge_result.get("source_excerpt") or judge_result.get("reference_answer") or "")
        raw_results = sample.get("retrieval_results") or []
        clean_results = []
        for rr in raw_results[:5]:
            clean_results.append({
                "position": rr.get("position"),
                "document_name": rr.get("document_name") or "",
                "score": rr.get("score"),
                "content": rr.get("content") or "",
            })
        base.update({
            "diagnostic_status": "ok",
            "retrieval_query": sample.get("retrieval_query") or sample.get("question") or "",
            "gold_evidence": gold,
            "retrieval_results": clean_results,
            "retrieval_result_count": len(raw_results),
            "final_answer": sample.get("final_answer") or "",
            "source_format": sample.get("source_format") or "",
            "source_file_name": sample.get("source_file_name") or "",
            "evidence_sheet": sample.get("evidence_sheet") or "",
            "evidence_range": sample.get("evidence_range") or "",
        })
        # 回填 sample 侧的扩展字段（judge_result 优先）
        for field in ("question_id", "question_set_id", "topic", "difficulty"):
            if not base.get(field):
                base[field] = sample.get(field) or ""

    return base


# ====== HTML 辅助函数 ======

_SENSITIVE_SNAPSHOT_KEYS = frozenset({
    "api_key", "secret_key", "lf_public_key", "lf_secret_key",
    "openai_api_key", "api_keys", "cookie", "session_token", "password", "token",
})

_ABS_PATH_PREFIXES_HTML = ("C:\\", "D:\\", "E:\\", "F:\\", "/", "\\\\")


def _is_safe_snapshot_value(v):
    """判断快照值是否安全可展示。"""
    s = str(v)
    for prefix in _ABS_PATH_PREFIXES_HTML:
        if s.startswith(prefix):
            return False
    return True


def _render_config_snapshot_table(snapshot):
    """将 config_snapshot 渲染为可读的 HTML 键值表。"""
    if not snapshot:
        return '<p class="section-note">未记录配置快照</p>'

    rows = []
    for k, v in sorted(snapshot.items()):
        if k in _SENSITIVE_SNAPSHOT_KEYS:
            continue
        if not _is_safe_snapshot_value(v):
            continue
        if isinstance(v, dict):
            v_str = json.dumps(v, ensure_ascii=False, indent=2)
        elif isinstance(v, list):
            v_str = ", ".join(str(x) for x in v)
        elif isinstance(v, bool):
            v_str = "是" if v else "否"
        else:
            v_str = str(v) if v is not None else "未记录"
        rows.append(f'<tr><td><strong>{_safe_str(k)}</strong></td><td>{_safe_str(v_str)}</td></tr>')

    if not rows:
        return '<p class="section-note">未记录配置快照</p>'

    return '<table>' + ''.join(rows) + '</table>'


def _compute_local_analysis(retrieval_results, strict_qa_results, grounded_qa_results,
                            error_results, sample_lookup, diag):
    """按 source_file / topic / difficulty / source_format 分组计算指标。

    Returns:
        dict: {
            "by_source_file": [{"key": str, "count": int, ...}, ...],
            "by_topic": [...],
            "by_difficulty": [...],
            "by_source_format": [...],
        }
    """
    def _group_by(key_field):
        groups = {}
        for r in retrieval_results:
            tid = r.get("trace_id", "")
            sample = sample_lookup.get(tid) or {}
            key = sample.get(key_field) or ""
            if not key:
                key = "未记录"
            if key not in groups:
                groups[key] = {"count": 0, "t1_hit": 0, "t3_hit": 0, "t5_hit": 0}
            g = groups[key]
            g["count"] += 1
            g["t1_hit"] += r.get("retrieval_top1_hit", 0)
            g["t3_hit"] += r.get("retrieval_top3_hit", 0)
            g["t5_hit"] += r.get("retrieval_top5_hit", 0)

        for r in strict_qa_results:
            tid = r.get("trace_id", "")
            sample = sample_lookup.get(tid) or {}
            key = sample.get(key_field) or ""
            if not key:
                key = "未记录"
            if key not in groups:
                groups[key] = {"count": 0, "t1_hit": 0, "t3_hit": 0, "t5_hit": 0}
            g = groups[key]
            g.setdefault("sqa_count", 0)
            g.setdefault("sqa_correct", 0)
            g["sqa_count"] += 1
            g["sqa_correct"] += r.get("answer_correct", 0)

        for r in grounded_qa_results:
            tid = r.get("trace_id", "")
            sample = sample_lookup.get(tid) or {}
            key = sample.get(key_field) or ""
            if not key:
                key = "未记录"
            if key not in groups:
                groups[key] = {"count": 0, "t1_hit": 0, "t3_hit": 0, "t5_hit": 0}
            g = groups[key]
            g.setdefault("gqa_count", 0)
            g.setdefault("gqa_grounded", 0)
            g["gqa_count"] += 1
            g["gqa_grounded"] += r.get("answer_correct", 0)

        result = []
        for k in sorted(groups.keys()):
            g = groups[k]
            n = g["count"]
            result.append({
                "key": k,
                "count": n,
                "t1_rate": g["t1_hit"] / n if n > 0 else None,
                "t3_rate": g["t3_hit"] / n if n > 0 else None,
                "t5_rate": g["t5_hit"] / n if n > 0 else None,
                "sqa_count": g.get("sqa_count", 0),
                "sqa_rate": (g["sqa_correct"] / g["sqa_count"]) if g.get("sqa_count") else None,
                "gqa_count": g.get("gqa_count", 0),
                "gqa_rate": (g["gqa_grounded"] / g["gqa_count"]) if g.get("gqa_count") else None,
            })
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    return {
        "by_source_file": _group_by("source_file_name"),
        "by_topic": _group_by("topic"),
        "by_difficulty": _group_by("difficulty"),
        "by_source_format": _group_by("source_format"),
    }


def _render_local_analysis_table(groups, group_label):
    """渲染一个分组分析表。"""
    if not groups:
        return f'<p class="section-note">暂无按{group_label}分组的数据</p>'

    html = '<table>'
    html += f'<tr><th>{group_label}</th><th>样本数</th><th>Top1</th><th>Top3</th><th>Top5</th>'
    html += '<th>严格问答(n)</th><th>合理性问答(n)</th></tr>'
    for g in groups:
        n = g["count"]
        t1 = _pct(g["t1_rate"])
        t3 = _pct(g["t3_rate"])
        t5 = _pct(g["t5_rate"])
        sqa_str = f'{_pct(g["sqa_rate"])} ({g["sqa_count"]})' if g["sqa_count"] else "-"
        gqa_str = f'{_pct(g["gqa_rate"])} ({g["gqa_count"]})' if g["gqa_count"] else "-"
        html += f'<tr><td>{_safe_str(g["key"])}</td><td>{n}</td>'
        html += f'<td>{t1}</td><td>{t3}</td><td>{t5}</td>'
        html += f'<td>{sqa_str}</td><td>{gqa_str}</td></tr>'
    html += '</table>'
    return html


# ====== HTML 报告 ======

def build_evaluation_html(config, config_runs, run_data_list, cumulative_metrics,
                          all_judge_results, export_scope="", sample_lookup=None):
    """生成自包含 HTML 评测报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_name = _safe_str(config.get("config_name", ""))
    config_id = _safe_str(config.get("config_id", ""))
    kb_version = _safe_str(config.get("knowledge_base_version", ""))
    wf_version = _safe_str(config.get("workflow_version", ""))
    sample_lookup = sample_lookup or {}

    # 按轨道分组
    valid_results = [r for r in all_judge_results if "error" not in r]
    error_results = [r for r in all_judge_results if "error" in r]
    retrieval_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_RETRIEVAL
                         and r.get("retrieval_evaluable", True)]
    strict_qa_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_STRICT_QA]
    grounded_qa_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_GROUNDED_QA]
    not_evaluable_results = [r for r in valid_results if r.get("evaluation_track") == TRACK_NOT_EVALUABLE]

    # 诊断数据
    diag = build_diagnostic_data(all_judge_results, sample_lookup, config)

    # 总览统计
    total_questions = sum(rd.get("run", {}).get("question_count", 0) for rd in run_data_list)
    total_batch_success = sum(rd.get("run_status", {}).get("batch_success", 0) for rd in run_data_list)
    total_batch_total = sum(rd.get("run_status", {}).get("batch_total", 0) for rd in run_data_list)
    total_processed = sum(rd.get("run_status", {}).get("processed_count", 0) for rd in run_data_list)
    total_judge = sum(rd.get("run_status", {}).get("judge_count", 0) for rd in run_data_list)

    # 分轨道统计
    track_counts = {
        "retrieval_evaluable": len(retrieval_results),
        "strict_qa": len(strict_qa_results),
        "grounded_qa": len(grounded_qa_results),
        "not_evaluable": len(not_evaluable_results),
    }
    no_retrieval_results_count = sum(
        1 for d in diag["top5_miss"] + diag["sorting_issues"]
        if d.get("diagnostic_status") == "ok" and not d.get("retrieval_results")
    )

    # 构建 HTML
    html_parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 评测报告 - {config_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 30px; }}
  h3 {{ color: #0f3460; margin-top: 20px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
  .meta p {{ margin: 4px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                  gap: 12px; margin: 16px 0; }}
  .metric-card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px;
                  padding: 16px; text-align: center; }}
  .metric-card .value {{ font-size: 1.8em; font-weight: 700; color: #1a1a2e; }}
  .metric-card .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  .hit {{ color: #28a745; font-weight: 600; }}
  .miss {{ color: #dc3545; font-weight: 600; }}
  .warn {{ color: #ffc107; }}
  .section-note {{ color: #888; font-size: 0.85em; font-style: italic; }}
  .diag-card {{ border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 16px 0;
                background: #fff; page-break-inside: avoid; }}
  .diag-card h4 {{ margin: 0 0 8px 0; color: #0f3460; }}
  .diag-meta {{ font-size: 0.85em; color: #666; margin-bottom: 8px; }}
  .diag-meta span {{ margin-right: 16px; }}
  .gold-evidence {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px;
                    padding: 10px; margin: 8px 0; white-space: pre-wrap; font-size: 0.9em; }}
  .retrieval-table {{ margin: 8px 0; }}
  .retrieval-table td {{ font-size: 0.85em; }}
  .retrieval-table .content-cell {{ max-width: 500px; white-space: pre-wrap; word-break: break-all; }}
  .no-result {{ color: #888; font-style: italic; }}
  @media print {{
    body {{ max-width: none; padding: 10px; }}
    .no-print {{ display: none; }}
    .diag-card {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<h1>RAG 评测报告</h1>
<div class="meta">
  <p><strong>配置方案</strong>: {config_name} (<code>{config_id}</code>)</p>
  <p><strong>知识库版本</strong>: {kb_version or '未指定'}</p>
  <p><strong>工作流版本</strong>: {wf_version or '未指定'}</p>
  <p><strong>生成时间</strong>: {now}</p>
  <p><strong>导出范围</strong>: {_safe_str(export_scope) or '当前配置全部运行'}</p>
  <p><strong>数据口径</strong>: 通过 config_id → run_id → processed sample → 真实 Langfuse trace_id → judged result 关联；
     检索命中仅对 evaluation_track=retrieval 的可评测样本计算 Top1/Top3/Top5；QA 轨道单独统计，不与检索命中率混合。</p>
</div>
"""]

    # 1. 总览指标
    html_parts.append(f"""
<h2>1. 总览</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="value">{len(config_runs)}</div><div class="label">运行次数</div></div>
  <div class="metric-card"><div class="value">{total_questions}</div><div class="label">题目总数</div></div>
  <div class="metric-card"><div class="value">{total_batch_success}/{total_batch_total}</div><div class="label">Batch 成功</div></div>
  <div class="metric-card"><div class="value">{total_processed}</div><div class="label">Processed</div></div>
  <div class="metric-card"><div class="value">{total_judge}</div><div class="label">Judge 已评测</div></div>
  <div class="metric-card"><div class="value">{len(error_results)}</div><div class="label">Judge 错误</div></div>
</div>
<table>
<tr><th>轨道</th><th>样本数</th></tr>
<tr><td>retrieval（可评测）</td><td>{track_counts['retrieval_evaluable']}</td></tr>
<tr><td>strict_qa</td><td>{track_counts['strict_qa']}</td></tr>
<tr><td>grounded_qa</td><td>{track_counts['grounded_qa']}</td></tr>
<tr><td>不可评测</td><td>{track_counts['not_evaluable']}</td></tr>
<tr><td>Judge 错误</td><td>{len(error_results)}</td></tr>
<tr><td>无检索结果</td><td>{no_retrieval_results_count}</td></tr>
</table>
""")

    # 2. 配置与运行信息
    html_parts.append("<h2>2. 配置与运行信息</h2>")
    for rd in run_data_list:
        run = rd["run"]
        rs = rd["run_status"]
        rid = _safe_str(run.get("run_id", ""))
        qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
        qsid = _safe_str(rs.get("question_set_id") or run.get("question_set_id", "") or "")
        started = _safe_str(run.get("started_at", ""))[:19]
        status = _safe_str(run.get("status", ""))
        snapshot = run.get("config_snapshot") or {}

        html_parts.append(f'<h3>{rid}</h3><table>')
        html_parts.append(f'<tr><td><strong>配置名</strong></td><td>{_safe_str(snapshot.get("config_name", "") or config_name)}</td></tr>')
        html_parts.append(f'<tr><td><strong>Config ID</strong></td><td>{_safe_str(snapshot.get("config_id", "") or config_id)}</td></tr>')
        html_parts.append(f'<tr><td><strong>知识库版本</strong></td><td>{_safe_str(snapshot.get("knowledge_base_version", "") or "未记录")}</td></tr>')
        html_parts.append(f'<tr><td><strong>工作流版本</strong></td><td>{_safe_str(snapshot.get("workflow_version", "") or "未记录")}</td></tr>')
        html_parts.append(f'<tr><td><strong>题集名称</strong></td><td>{qs}</td></tr>')
        html_parts.append(f'<tr><td><strong>Question Set ID</strong></td><td>{qsid}</td></tr>')
        html_parts.append(f'<tr><td><strong>开始时间</strong></td><td>{started}</td></tr>')
        html_parts.append(f'<tr><td><strong>状态</strong></td><td>{status}</td></tr>')
        html_parts.append('</table>')

        # 完整 config_snapshot
        if snapshot:
            html_parts.append('<details><summary>完整配置快照（config_snapshot）</summary>')
            html_parts.append(_render_config_snapshot_table(snapshot))
            html_parts.append('</details>')
        else:
            html_parts.append('<p class="section-note">该运行未记录配置快照</p>')

    # 3. 全局 Judge 指标
    html_parts.append("<h2>3. 全局 Judge 指标</h2>")
    if retrieval_results:
        n = len(retrieval_results)
        t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_results)
        t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_results)
        t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_results)
        html_parts.append(f"""
<p class="section-note">仅 evaluation_track=retrieval 的可评测样本</p>
<table>
<tr><th>指标</th><th>命中数</th><th>样本数</th><th>命中率</th></tr>
<tr><td>Top1 Hit</td><td>{t1}</td><td>{n}</td><td>{_pct(t1/n)}</td></tr>
<tr><td>Top3 Hit</td><td>{t3}</td><td>{n}</td><td>{_pct(t3/n)}</td></tr>
<tr><td>Top5 Hit</td><td>{t5}</td><td>{n}</td><td>{_pct(t5/n)}</td></tr>
</table>
""")
    else:
        html_parts.append('<p class="section-note">暂无检索评测数据</p>')

    if strict_qa_results:
        n = len(strict_qa_results)
        acc = sum(r.get("answer_correct", 0) for r in strict_qa_results)
        html_parts.append(f'<p>严格问答: 正确 {acc} / 总 {n} = {_pct(acc/n)}</p>')
    if grounded_qa_results:
        n = len(grounded_qa_results)
        gnd = sum(r.get("answer_correct", 0) for r in grounded_qa_results)
        html_parts.append(f'<p>合理性问答: 有据 {gnd} / 总 {n} = {_pct(gnd/n)}</p>')

    # 3.5 局部分析
    html_parts.append("<h2>4. 局部分析</h2>")
    local = _compute_local_analysis(retrieval_results, strict_qa_results, grounded_qa_results,
                                    error_results, sample_lookup, diag)

    html_parts.append("<h3>按源文件</h3>")
    html_parts.append(_render_local_analysis_table(local["by_source_file"], "源文件"))

    html_parts.append("<h3>按 Topic</h3>")
    html_parts.append(_render_local_analysis_table(local["by_topic"], "Topic"))

    html_parts.append("<h3>按难度</h3>")
    html_parts.append(_render_local_analysis_table(local["by_difficulty"], "难度"))

    if local["by_source_format"]:
        html_parts.append("<h3>按文档格式</h3>")
        html_parts.append(_render_local_analysis_table(local["by_source_format"], "格式"))

    # 5. 运行汇总表
    html_parts.append("""
<h2>5. 运行汇总</h2>
<table>
<tr><th>运行 ID</th><th>题集</th><th>题目数</th><th>状态</th><th>Batch</th><th>Processed</th><th>Judge</th>
<th>Top1</th><th>Top3</th><th>Top5</th><th>Top5未命中</th><th>排序问题</th><th>错误数</th></tr>
""")
    for rd in run_data_list:
        run = rd["run"]
        rs = rd["run_status"]
        m = rd.get("metrics") or {}
        rid = _safe_str(run.get("run_id", ""))
        qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
        qc = run.get("question_count", 0)
        status = _safe_str(run.get("status", ""))
        bs = rs.get("batch_success", 0)
        bt = rs.get("batch_total", 0)
        pc = rs.get("processed_count", 0)
        jc = rs.get("judge_count", 0)
        t1 = _pct(m.get("retrieval_top1_hit_rate"))
        t3 = _pct(m.get("retrieval_top3_hit_rate"))
        t5 = _pct(m.get("retrieval_top5_hit_rate"))
        errs = m.get("errors", 0)
        # 从该 run 的 judge_results 计算 miss/sorting
        _run_jr = rs.get("judge_results", [])
        _run_valid = [r for r in _run_jr if "error" not in r]
        _run_ret = [r for r in _run_valid
                    if r.get("evaluation_track") == TRACK_RETRIEVAL
                    and r.get("retrieval_evaluable", True)]
        miss5 = sum(1 for r in _run_ret if r.get("retrieval_top5_hit", 0) == 0)
        sort_issues = sum(1 for r in _run_ret
                          if r.get("retrieval_top1_hit", 0) == 0 and r.get("retrieval_top5_hit", 0) == 1)
        html_parts.append(
            f'<tr><td><code>{rid}</code></td><td>{qs}</td><td>{qc}</td><td>{status}</td>'
            f'<td>{bs}/{bt}</td><td>{pc}</td><td>{jc}</td>'
            f'<td>{t1}</td><td>{t3}</td><td>{t5}</td>'
            f'<td>{miss5}</td><td>{sort_issues}</td><td>{errs}</td></tr>'
        )
    html_parts.append("</table>")

    # 5. 运行详情
    html_parts.append("<h2>6. 运行详情</h2>")
    for rd in run_data_list:
        run = rd["run"]
        rs = rd["run_status"]
        m = rd.get("metrics") or {}
        rid = _safe_str(run.get("run_id", ""))
        qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
        started = _safe_str(run.get("started_at", ""))[:19]
        status = _safe_str(run.get("status", ""))
        batch_file = _safe_str(run.get("batch_results_file", "") or "无")
        raw_file = _safe_str(run.get("raw_results_file", "") or "无")

        # 该 run 的 config_snapshot
        snapshot = run.get("config_snapshot") or {}

        html_parts.append(f"""
<h3>{rid}</h3>
<table>
<tr><td><strong>题集</strong></td><td>{qs}</td></tr>
<tr><td><strong>创建时间</strong></td><td>{started}</td></tr>
<tr><td><strong>状态</strong></td><td>{status}</td></tr>
<tr><td><strong>Batch 文件</strong></td><td><code>{batch_file}</code></td></tr>
<tr><td><strong>Raw 文件</strong></td><td><code>{raw_file}</code></td></tr>
<tr><td><strong>Batch</strong></td><td>{rs.get('batch_success', 0)}/{rs.get('batch_total', 0)}</td></tr>
<tr><td><strong>Processed</strong></td><td>{rs.get('processed_count', 0)}</td></tr>
<tr><td><strong>Judge</strong></td><td>{rs.get('judge_count', 0)}</td></tr>
</table>
""")
        # 展示 config_snapshot
        if snapshot:
            html_parts.append('<details><summary>该次运行配置快照（config_snapshot）</summary>')
            html_parts.append(_render_config_snapshot_table(snapshot))
            html_parts.append('</details>')

        jr = rs.get("judge_results", [])
        if jr:
            valid_jr = [r for r in jr if "error" not in r]
            ret_jr = [r for r in valid_jr if r.get("evaluation_track") == TRACK_RETRIEVAL]
            if ret_jr:
                n = len(ret_jr)
                t1 = sum(r.get("retrieval_top1_hit", 0) for r in ret_jr) / n
                t3 = sum(r.get("retrieval_top3_hit", 0) for r in ret_jr) / n
                t5 = sum(r.get("retrieval_top5_hit", 0) for r in ret_jr) / n
                html_parts.append(f'<p>检索评测 (n={n}): Top1={_pct(t1)} | Top3={_pct(t3)} | Top5={_pct(t5)}</p>')
            sqa_jr = [r for r in valid_jr if r.get("evaluation_track") == TRACK_STRICT_QA]
            if sqa_jr:
                n = len(sqa_jr)
                acc = sum(r.get("answer_correct", 0) for r in sqa_jr) / n
                html_parts.append(f'<p>严格问答 (n={n}): Answer Correctness={_pct(acc)}</p>')
            gqa_jr = [r for r in valid_jr if r.get("evaluation_track") == TRACK_GROUNDED_QA]
            if gqa_jr:
                n = len(gqa_jr)
                acc = sum(r.get("answer_correct", 0) for r in gqa_jr) / n
                html_parts.append(f'<p>合理性问答 (n={n}): Answer Grounded={_pct(acc)}</p>')
            err_jr = [r for r in jr if "error" in r]
            if err_jr:
                html_parts.append(f'<p class="miss">Judge 错误: {len(err_jr)} 条</p>')

    # 6. Top5 完全未命中样本诊断
    html_parts.append("<h2>7. Top5 完全未命中样本诊断</h2>")
    _render_diagnostic_cards(html_parts, diag["top5_miss"], diag["total_top5_miss"],
                             "Top5 未命中（检索结果均未命中金标准）", show_details=True)

    # 7. 排序问题样本
    html_parts.append("<h2>8. 排序问题样本（Top1 未命中但 Top3/Top5 命中）</h2>")
    _render_diagnostic_cards(html_parts, diag["sorting_issues"], diag["total_sorting_issues"],
                             "排序问题（Top1 未命中但更高排名命中，说明相关内容被排到较低位置）",
                             show_details=True)

    # 8. 数据质量
    html_parts.append("<h2>9. 数据质量与可审计信息</h2><ul>")
    html_parts.append(f'<li>Judge 错误结果: <strong>{len(error_results)}</strong> 条</li>')
    html_parts.append(f'<li>不可评测样本（缺少金标准证据）: <strong>{len(not_evaluable_results)}</strong> 条</li>')
    html_parts.append(f'<li>无检索结果的样本: <strong>{no_retrieval_results_count}</strong> 条</li>')

    no_trace = [r for r in all_judge_results if not r.get("trace_id")]
    if no_trace:
        html_parts.append(f'<li class="warn">缺少 trace_id 的结果: <strong>{len(no_trace)}</strong> 条</li>')

    no_sample = sum(1 for d in diag["top5_miss"] if d["diagnostic_status"] == "no_processed_sample")
    no_sample += sum(1 for d in diag["sorting_issues"] if d["diagnostic_status"] == "no_processed_sample")
    if no_sample:
        html_parts.append(f'<li class="warn">未找到对应 processed sample 的诊断样本: <strong>{no_sample}</strong> 条</li>')

    if error_results:
        html_parts.append('</ul><h3>Judge 错误详情（前 10 条）</h3><table>')
        html_parts.append('<tr><th>Trace ID</th><th>错误信息</th></tr>')
        for r in error_results[:10]:
            tid = _safe_str(r.get("trace_id", ""))
            err = _safe_str(str(r.get("error", ""))[:200])
            html_parts.append(f'<tr><td><code>{tid}</code></td><td>{err}</td></tr>')
        html_parts.append('</table>')
    else:
        html_parts.append('</ul>')

    html_parts.append("</body></html>")
    return "".join(html_parts)


def _render_diagnostic_cards(html_parts, records, total_count, empty_msg, show_details=False):
    """渲染诊断卡片列表。show_details=True 时使用 <details> 折叠展示完整信息。"""
    if not records:
        html_parts.append(f'<p class="section-note">无{empty_msg}</p>')
        return

    shown = len(records)
    truncated = total_count > shown
    if truncated:
        html_parts.append(f'<p class="section-note">共 {total_count} 条，显示前 {shown} 条</p>')
    else:
        html_parts.append(f'<p class="section-note">共 {total_count} 条</p>')

    for i, d in enumerate(records, 1):
        rid = _safe_str(d["run_id"])
        tid = _safe_str(d["trace_id"])
        q = _safe_str(d["question"])
        rq = _safe_str(d.get("retrieval_query") or d["question"])
        gold = _safe_str(d["gold_evidence"])
        reason = _safe_str(d["judge_reason"])
        pos = d["hit_evidence_position"]
        pos_str = str(pos) if pos is not None else "null"
        diag_status = d.get("diagnostic_status", "ok")

        html_parts.append(f'<div class="diag-card">')

        if show_details:
            # 使用 <details> 折叠，摘要行只显示问题和 trace_id
            summary = f"#{i} {q}  [Trace: {tid}]"
            html_parts.append(f'<details><summary>{_safe_str(summary)}</summary>')
        else:
            html_parts.append(f'<h4>#{i} {q}</h4>')

        html_parts.append(f'<div class="diag-meta">')
        html_parts.append(f'<span>Run: <code>{rid}</code></span>')
        html_parts.append(f'<span>Trace: <code>{tid}</code></span>')
        html_parts.append(f'<span>Track: {_safe_str(d["evaluation_track"])}</span>')
        html_parts.append(f'<span>Hit Position: {pos_str}</span>')
        if d.get("question_id"):
            html_parts.append(f'<span>Question ID: {_safe_str(d["question_id"])}</span>')
        if d.get("question_set_id"):
            html_parts.append(f'<span>Question Set: {_safe_str(d["question_set_id"])}</span>')
        if d.get("config_name"):
            html_parts.append(f'<span>配置: {_safe_str(d["config_name"])}</span>')
        if d.get("knowledge_base_version"):
            html_parts.append(f'<span>知识库版本: {_safe_str(d["knowledge_base_version"])}</span>')
        if d.get("workflow_version"):
            html_parts.append(f'<span>工作流版本: {_safe_str(d["workflow_version"])}</span>')
        if d.get("source_file_name"):
            html_parts.append(f'<span>源文件: {_safe_str(d["source_file_name"])}</span>')
        if d.get("topic"):
            html_parts.append(f'<span>Topic: {_safe_str(d["topic"])}</span>')
        if d.get("difficulty"):
            html_parts.append(f'<span>难度: {_safe_str(d["difficulty"])}</span>')
        if d.get("source_format"):
            html_parts.append(f'<span>格式: {_safe_str(d["source_format"])}</span>')
        html_parts.append('</div>')

        if diag_status == "no_processed_sample":
            html_parts.append('<p class="warn">诊断数据缺失：未找到对应 processed sample，以下信息可能不完整</p>')

        if rq != q:
            html_parts.append(f'<p><strong>检索查询</strong>: {rq}</p>')

        html_parts.append(f'<p><strong>金标准证据</strong>:</p>')
        html_parts.append(f'<div class="gold-evidence">{gold}</div>')

        # 来源定位信息
        source_parts = []
        if d.get("source_format"):
            source_parts.append(f"格式: {_safe_str(d['source_format'])}")
        if d.get("source_file_name"):
            source_parts.append(f"文件: {_safe_str(d['source_file_name'])}")
        if d.get("evidence_sheet"):
            source_parts.append(f"Sheet: {_safe_str(d['evidence_sheet'])}")
        if d.get("evidence_range"):
            source_parts.append(f"范围: {_safe_str(d['evidence_range'])}")
        if source_parts:
            html_parts.append(f'<p><strong>来源定位</strong>: {" | ".join(source_parts)}</p>')

        if show_details:
            html_parts.append(f'<details><summary>Judge Reason</summary><p>{reason}</p></details>')
        else:
            html_parts.append(f'<p><strong>Judge Reason</strong>: {reason}</p>')

        # 检索结果
        ret_results = d.get("retrieval_results") or []
        if ret_results:
            html_parts.append(f'<p><strong>实际检索结果</strong>（共 {d.get("retrieval_result_count", len(ret_results))} 条，展示前 {len(ret_results)} 条）:</p>')
            html_parts.append('<table class="retrieval-table"><tr><th>#</th><th>文档名</th><th>Score</th><th>Content</th></tr>')
            for rr in ret_results:
                pos_r = rr.get("position", "")
                doc = _safe_str(rr.get("document_name", ""))
                score = rr.get("score", "")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else _safe_str(str(score))
                content_text = rr.get("content") or ""
                content_html = _safe_str(content_text)
                html_parts.append(
                    f'<tr><td>{pos_r}</td><td>{doc}</td><td>{score_str}</td>'
                    f'<td class="content-cell">{content_html}</td></tr>'
                )
            html_parts.append('</table>')
        else:
            html_parts.append('<p class="no-result">未返回检索结果</p>')

        if show_details:
            html_parts.append('</details>')

        html_parts.append('</div>')


# ====== 文件名生成 ======

def sanitize_filename_component(name, max_len=50):
    """将配置名清洗为安全的文件名组成部分。

    规则：
    - 保留中文、英文、数字、空格、-、_、圆括号
    - 替换 Windows 非法字符 < > : " / \\ | ? * 和控制字符为 _
    - 去除末尾空格和句点
    - 限制长度 max_len
    - 空或清洗后为空时回退为 "未命名配置"
    - 不含路径分隔符，无目录穿越风险
    """
    if not name or not isinstance(name, str):
        return "未命名配置"
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name.strip())
    safe = re.sub(r'[^\w\u4e00-\u9fff\s\-\(\)]', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_ ')
    safe = safe.rstrip('.')
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip(' .')
    if not safe or safe.strip('_') == '':
        return "未命名配置"
    return safe


def build_export_filename(config_name, config_id, suffix, extension):
    """生成含配置信息的安全文件名。

    格式: {safe_name}__cfg_{short_id}__{timestamp}.{ext}
    示例: 合同知识库入库_v2_4__cfg_ab12cd34__20260722_101530.html

    Parameters
    ----------
    config_name : str
        配置方案名称，会被 sanitize_filename_component 清洗。
    config_id : str
        完整 config_id，取末 8 字符作为短标识。
    suffix : str
        文件类型标识，如 "report"、"runs"、"failed_samples"。
    extension : str
        文件扩展名（不含点），如 "html"、"csv"。

    Returns
    -------
    str
        安全的文件名。

    Raises
    ------
    ValueError
        config_id 为空时抛出。
    """
    if not config_id or not config_id.strip():
        raise ValueError("config_id 不能为空，无法生成导出文件名")

    safe_name = sanitize_filename_component(config_name)
    # 空格替换为下划线，保持文件名紧凑
    safe_name = safe_name.replace(" ", "_")
    short_id = config_id.strip()[-8:]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}__cfg_{short_id}__{ts}_{suffix}.{extension}"
    # 总长度限制 200 字符
    if len(filename) > 200:
        trim = len(filename) - 200
        safe_name = safe_name[:len(safe_name) - trim]
        filename = f"{safe_name}__cfg_{short_id}__{ts}_{suffix}.{extension}"
    return filename


# ====== CSV 导出 ======

def build_runs_csv(run_data_list):
    """生成运行汇总 CSV。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "run_id", "config_id", "config_name", "knowledge_base_version", "workflow_version",
        "question_set_name", "question_set_id",
        "started_at", "status", "question_count",
        "batch_success", "batch_total", "raw_count", "processed_count", "judge_count",
        "retrieval_track_count", "retrieval_top1_hit_rate", "retrieval_top3_hit_rate", "retrieval_top5_hit_rate",
        "strict_qa_count", "strict_qa_answer_rate",
        "grounded_qa_count", "grounded_qa_answer_rate",
        "top5_miss_count", "sorting_issue_count",
        "errors",
        "config_snapshot_summary",
    ])

    for rd in run_data_list:
        run = rd["run"]
        rs = rd["run_status"]
        m = rd.get("metrics") or {}
        snapshot = run.get("config_snapshot") or {}

        # 计算 miss/sorting
        _run_jr = rs.get("judge_results", [])
        _run_valid = [r for r in _run_jr if "error" not in r]
        _run_ret = [r for r in _run_valid
                    if r.get("evaluation_track") == TRACK_RETRIEVAL
                    and r.get("retrieval_evaluable", True)]
        miss5 = sum(1 for r in _run_ret if r.get("retrieval_top5_hit", 0) == 0)
        sort_issues = sum(1 for r in _run_ret
                          if r.get("retrieval_top1_hit", 0) == 0 and r.get("retrieval_top5_hit", 0) == 1)

        # config_snapshot 可读摘要（排除敏感字段）
        safe_snapshot = {k: v for k, v in snapshot.items()
                         if k not in _SENSITIVE_KEYS and _is_safe_snapshot_value(v)}
        snapshot_summary = "; ".join(f"{k}={v}" for k, v in sorted(safe_snapshot.items()))

        writer.writerow([
            run.get("run_id", ""),
            run.get("config_id", ""),
            snapshot.get("config_name", ""),
            snapshot.get("knowledge_base_version", ""),
            snapshot.get("workflow_version", ""),
            rs.get("question_set_name") or run.get("question_set_name", ""),
            rs.get("question_set_id") or run.get("question_set_id", ""),
            (run.get("started_at") or "")[:19],
            run.get("status", ""),
            run.get("question_count", 0),
            rs.get("batch_success", 0),
            rs.get("batch_total", 0),
            rs.get("raw_count", 0),
            rs.get("processed_count", 0),
            rs.get("judge_count", 0),
            m.get("retrieval_track_count", 0),
            m.get("retrieval_top1_hit_rate"),
            m.get("retrieval_top3_hit_rate"),
            m.get("retrieval_top5_hit_rate"),
            m.get("strict_qa_track_count", 0),
            m.get("strict_qa_answer_rate"),
            m.get("grounded_qa_track_count", 0),
            m.get("grounded_qa_answer_rate"),
            miss5,
            sort_issues,
            m.get("errors", 0),
            snapshot_summary,
        ])

    return output.getvalue().encode("utf-8-sig")


def build_failed_samples_csv(all_judge_results, sample_lookup=None, config=None):
    """生成详细未命中样本 CSV（Top5 未命中 + 排序问题）。

    每行一条样本，展开 Top1-Top5 的检索结果详情。
    """
    sample_lookup = sample_lookup or {}
    diag = build_diagnostic_data(all_judge_results, sample_lookup, config)

    # 合并两类
    all_records = []
    for d in diag["top5_miss"]:
        d["_category"] = "top5_miss"
        all_records.append(d)
    for d in diag["sorting_issues"]:
        d["_category"] = "sorting_issue"
        all_records.append(d)

    truncated = len(all_records) > _MAX_DIAGNOSTIC_SAMPLES
    total_count = len(all_records)
    all_records = all_records[:_MAX_DIAGNOSTIC_SAMPLES]

    # 构建 CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # 表头
    header = [
        "category", "run_id", "trace_id", "config_id", "config_name",
        "question_id", "question_set_id",
        "question", "retrieval_query",
        "gold_evidence",
        "evaluation_track", "hit_evidence_position", "judge_reason",
        "retrieval_result_count", "diagnostic_status",
        "source_format", "source_file_name", "topic", "difficulty",
        "evidence_sheet", "evidence_range",
        "knowledge_base_version", "workflow_version",
    ]
    # 展开 Top1-Top5（不截断 content）
    for i in range(1, 6):
        header.extend([
            f"retrieval_{i}_document_name",
            f"retrieval_{i}_score",
            f"retrieval_{i}_content",
        ])
    writer.writerow(header)

    for d in all_records:
        row = [
            d["_category"],
            d["run_id"],
            d["trace_id"],
            d.get("config_id", ""),
            d.get("config_name", ""),
            d.get("question_id", ""),
            d.get("question_set_id", ""),
            d["question"],
            d.get("retrieval_query") or d["question"],
            d["gold_evidence"],
            d["evaluation_track"],
            d["hit_evidence_position"] if d["hit_evidence_position"] is not None else "",
            d["judge_reason"],
            d.get("retrieval_result_count", 0),
            d.get("diagnostic_status", "ok"),
            d.get("source_format", ""),
            d.get("source_file_name", ""),
            d.get("topic", ""),
            d.get("difficulty", ""),
            d.get("evidence_sheet", ""),
            d.get("evidence_range", ""),
            d.get("knowledge_base_version", ""),
            d.get("workflow_version", ""),
        ]

        ret_results = d.get("retrieval_results") or []
        for i in range(5):
            if i < len(ret_results):
                rr = ret_results[i]
                row.extend([
                    rr.get("document_name", ""),
                    rr.get("score", ""),
                    rr.get("content", ""),
                ])
            else:
                row.extend(["", "", ""])

        writer.writerow(row)

    if truncated:
        output.write(f"\n# 截断说明: 共 {total_count} 条，仅导出前 {_MAX_DIAGNOSTIC_SAMPLES} 条\n")

    return output.getvalue().encode("utf-8-sig")
