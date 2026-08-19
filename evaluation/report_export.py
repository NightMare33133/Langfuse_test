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
from pathlib import Path

from judge import compute_metrics, compute_chunk_exact_metrics, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA, TRACK_NOT_EVALUABLE, TRACK_CHUNK_EXACT, backfill_chunk_exact_topk

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


# 元数据字段：需要从题集 JSONL 回填到报告的字段
_QUESTION_META_FIELDS = (
    "query_style", "retrieval_intent", "target_fact", "target_label",
    "expected_content", "expected_segment_id", "expected_content_hash",
    "document_id", "document_name", "source_position",
    "question_id", "question_set_id",
)


def load_question_set_metadata(question_set_ids=None):
    """加载题集 JSONL，构建 question_set_id + question_id -> 元数据的查找表。

    Args:
        question_set_ids: 要加载的 question_set_id 集合（None=全部）

    Returns:
        dict: {(question_set_id, question_id): {field: value, ...}}
    """
    from question_generator import QUESTIONS_DIR
    lookup = {}
    if not QUESTIONS_DIR.exists():
        return lookup

    for jsonl_path in sorted(QUESTIONS_DIR.glob("questions_*.jsonl")):
        manifest_path = jsonl_path.parent / f"{jsonl_path.stem}_manifest.json"
        # 快速检查 manifest 中的 question_set_id
        if question_set_ids and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                m_qsid = manifest.get("question_set_id", "")
                if m_qsid and m_qsid not in question_set_ids:
                    continue
            except Exception:
                pass

        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        q = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qsid = q.get("question_set_id", "")
                    qid = q.get("question_id", "")
                    if question_set_ids and qsid not in question_set_ids:
                        continue
                    meta = {}
                    for field in _QUESTION_META_FIELDS:
                        val = q.get(field)
                        if val not in (None, ""):
                            meta[field] = val
                    if qsid and qid:
                        lookup[(qsid, qid)] = meta
                    elif qsid:
                        # 按 question 文本回退匹配
                        question_text = q.get("question", "")
                        if question_text:
                            lookup[(qsid, question_text)] = meta
        except Exception:
            continue

    return lookup


def _lookup_question_meta(judge_result, question_meta_lookup):
    """从 judge_result 或题集查找表中获取完整元数据。

    优先级：judge_result 已有字段 > question_meta_lookup > 空字符串
    """
    qsid = judge_result.get("question_set_id", "")
    qid = judge_result.get("question_id", "")
    question_text = judge_result.get("question", "")

    # 从查找表获取
    from_lookup = {}
    if qsid and qid:
        from_lookup = question_meta_lookup.get((qsid, qid), {})
    if not from_lookup and qsid and question_text:
        from_lookup = question_meta_lookup.get((qsid, question_text), {})

    # 合并：judge_result 优先，查找表补充
    result = {}
    for field in _QUESTION_META_FIELDS:
        val = judge_result.get(field)
        if val not in (None, ""):
            result[field] = val
        elif field in from_lookup:
            result[field] = from_lookup[field]
        else:
            result[field] = ""
    return result


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


# ====== Chunk Exact 诊断数据 ======

def _short_id(id_str, n=12):
    """缩短 ID 用于展示，完整值保留在 title 属性中。"""
    if not id_str:
        return ""
    s = str(id_str)
    return s[:n] + "..." if len(s) > n else s


def build_chunk_exact_diagnostic_data(judge_results, sample_lookup, config=None,
                                      max_samples=_MAX_DIAGNOSTIC_SAMPLES,
                                      configured_top_k=10,
                                      knowledge_base_total_chunks=None,
                                      doc_chunk_counts=None,
                                      question_meta_lookup=None):
    """为 chunk_exact 轨道构建诊断数据：Top10 未命中、排序偏后和排序问题。

    诊断拆分：
    - top10_miss: Top10 完全未命中（完全未召回）
    - top5_miss_but_top10_hit: Top5 未命中但 Top10 命中（排序偏后，Top6-10）
    - sorting_issues: Top1 未命中但 Top3/Top5 命中

    Args:
        judge_results: 全部 judged results
        sample_lookup: {trace_id: processed_sample_dict}
        config: 配置方案 dict
        max_samples: 每类最大条数
        configured_top_k: 配置 TopK
        knowledge_base_total_chunks: 知识库总 chunk 数
        doc_chunk_counts: dict[doc_name → chunk_count]
        question_meta_lookup: 题集元数据查找表

    Returns:
        dict: {
            "top10_miss": [...],
            "top5_miss_but_top10_hit": [...],
            "sorting_issues": [...],
            "total_top10_miss": int,
            "total_top5_miss_but_top10_hit": int,
            "total_sorting_issues": int,
        }
    """
    config = config or {}
    # 补齐旧版 chunk_exact 结果缺失的 TopK 字段
    for r in judge_results:
        backfill_chunk_exact_topk(r, sample_lookup)

    valid = [r for r in judge_results if "error" not in r]
    chunk_exact = [r for r in valid
                   if r.get("evaluation_track") == TRACK_CHUNK_EXACT
                   and r.get("retrieval_evaluable", True) is not False
                   and r.get("retrieval_top1_hit") is not None]

    top10_miss = []
    top5_miss_but_top10_hit = []
    sorting_issues = []

    for r in chunk_exact:
        t1 = r.get("retrieval_top1_hit", 0)
        t5 = r.get("retrieval_top5_hit", 0)
        t10 = r.get("retrieval_top10_hit", 0)

        if t10 == 0:
            category = "top10_miss"
        elif t5 == 0:
            # Top5 未命中但 Top10 命中 → 排序偏后
            category = "top5_miss_but_top10_hit"
        elif t1 == 0:
            category = "sorting_issues"
        else:
            continue

        record = _build_chunk_exact_one_diagnostic(
            r, sample_lookup, config,
            configured_top_k=configured_top_k,
            knowledge_base_total_chunks=knowledge_base_total_chunks,
            doc_chunk_counts=doc_chunk_counts,
            question_meta_lookup=question_meta_lookup)
        if category == "top10_miss":
            top10_miss.append(record)
        elif category == "top5_miss_but_top10_hit":
            top5_miss_but_top10_hit.append(record)
        else:
            sorting_issues.append(record)

    return {
        "top10_miss": top10_miss[:max_samples],
        "top5_miss_but_top10_hit": top5_miss_but_top10_hit[:max_samples],
        "sorting_issues": sorting_issues[:max_samples],
        "total_top10_miss": len(top10_miss),
        "total_top5_miss_but_top10_hit": len(top5_miss_but_top10_hit),
        "total_sorting_issues": len(sorting_issues),
    }


def _build_chunk_exact_one_diagnostic(judge_result, sample_lookup, config,
                                      configured_top_k=10,
                                      knowledge_base_total_chunks=None,
                                      doc_chunk_counts=None,
                                      question_meta_lookup=None):
    """为单个 chunk_exact judged result 构建诊断记录。"""
    tid = judge_result.get("trace_id", "")
    sample = sample_lookup.get(tid)

    doc_name = _resolve_doc_name(judge_result, sample, question_meta_lookup)

    base = {
        "trace_id": tid,
        "run_id": judge_result.get("_source_run_id", judge_result.get("run_id", "")),
        "question": judge_result.get("question", ""),
        "question_id": judge_result.get("question_id") or "",
        "question_set_id": judge_result.get("question_set_id") or "",
        "question_set_name": judge_result.get("question_set_name") or "",
        "evaluation_track": TRACK_CHUNK_EXACT,
        "hit_evidence_position": judge_result.get("hit_evidence_position"),
        "expected_segment_id": judge_result.get("expected_segment_id") or "",
        "expected_content_hash": judge_result.get("expected_content_hash") or "",
        "chunk_exact_status": judge_result.get("chunk_exact_status", ""),
        "target_label": judge_result.get("target_label") or "",
        "reason": judge_result.get("reason", ""),
        "config_name": config.get("config_name", ""),
        "config_id": config.get("config_id", ""),
    }

    # 从 sample 获取 retrieval 结果（全部返回，不截断）
    if sample:
        expected_seg_id = judge_result.get("expected_segment_id") or ""

        # ── 多检索支持 ──
        retrieval_calls = sample.get("retrieval_calls") or []
        if retrieval_calls:
            # 多检索模式：收集所有 call 的结果
            all_results = []
            clean_calls = []
            for call in retrieval_calls:
                call_results = call.get("results") or []
                clean_call_results = []
                for rr in call_results:
                    seg_id = rr.get("segment_id") or rr.get("document_name") or ""
                    clean_call_results.append({
                        "position": rr.get("position"),
                        "segment_id": seg_id,
                        "document_name": rr.get("document_name") or "",
                        "score": rr.get("score"),
                        "content": (rr.get("content") or "")[:300],
                        "content_hash": rr.get("content_hash") or "",
                        "is_expected": (seg_id == expected_seg_id) if expected_seg_id else False,
                    })
                    all_results.append({
                        "position": rr.get("position"),
                        "segment_id": seg_id,
                        "document_name": rr.get("document_name") or "",
                        "score": rr.get("score"),
                        "content": (rr.get("content") or "")[:300],
                        "content_hash": rr.get("content_hash") or "",
                        "is_expected": (seg_id == expected_seg_id) if expected_seg_id else False,
                        "_retrieval_call_order": call.get("order"),
                    })
                clean_calls.append({
                    "order": call.get("order"),
                    "observation_id": call.get("observation_id"),
                    "query": call.get("query"),
                    "start_time": call.get("start_time"),
                    "end_time": call.get("end_time"),
                    "latency_ms": call.get("latency_ms"),
                    "results": clean_call_results,
                    "result_count": len(clean_call_results),
                })
            base["retrieval_calls"] = clean_calls
            base["retrieval_call_count"] = len(clean_calls)
            base["retrieval_results"] = all_results  # 平展用于兼容
            base["retrieval_result_count"] = len(all_results)
            base["retrieval_query"] = retrieval_calls[-1].get("query") or sample.get("question") or ""
        else:
            # 单检索模式（向后兼容）
            raw_results = sample.get("retrieval_results") or []
            clean_results = []
            for rr in raw_results:
                seg_id = rr.get("segment_id") or rr.get("document_name") or ""
                clean_results.append({
                    "position": rr.get("position"),
                    "segment_id": seg_id,
                    "document_name": rr.get("document_name") or "",
                    "score": rr.get("score"),
                    "content": (rr.get("content") or "")[:300],
                    "content_hash": rr.get("content_hash") or "",
                    "is_expected": (seg_id == expected_seg_id) if expected_seg_id else False,
                })
            base["retrieval_results"] = clean_results
            base["retrieval_result_count"] = len(raw_results)
            base["retrieval_query"] = sample.get("retrieval_query") or sample.get("question") or ""
            base["retrieval_calls"] = []
            base["retrieval_call_count"] = 0
    else:
        base["retrieval_results"] = []
        base["retrieval_result_count"] = 0
        base["retrieval_query"] = judge_result.get("question", "")
        base["retrieval_calls"] = []
        base["retrieval_call_count"] = 0

    # ── 多检索命中统计 ──
    per_call_hits = judge_result.get("per_call_hits") or []
    if per_call_hits:
        base["per_call_hits"] = per_call_hits
        base["subquery_hit_count"] = judge_result.get("subquery_hit_count", 0)
        base["per_subquery_hit"] = judge_result.get("per_subquery_hit", False)

    # 召回规模信息
    actual_returned = base["retrieval_result_count"]
    effective_k = min(configured_top_k, actual_returned)
    doc_chunks = doc_chunk_counts.get(doc_name) if doc_chunk_counts else None

    if actual_returned >= configured_top_k:
        window_status = "full_window"
        window_reason = ""
    else:
        window_status = "partial_window"
        window_reason = _determine_partial_window_reason(doc_name, configured_top_k, doc_chunk_counts)

    base["configured_top_k"] = configured_top_k
    base["actual_returned_count"] = actual_returned
    base["effective_k"] = effective_k
    base["window_status"] = window_status
    base["window_reason"] = window_reason
    base["target_source_document"] = doc_name
    base["target_source_document_chunk_count"] = doc_chunks
    base["knowledge_base_total_chunks"] = knowledge_base_total_chunks

    return base


def _render_chunk_exact_diagnostic_cards(html_parts, records, total_count, section_title):
    """渲染 chunk_exact 诊断卡片列表。"""
    if not records:
        html_parts.append(f'<p class="section-note">无{section_title}</p>')
        return

    shown = len(records)
    truncated = total_count > shown
    if truncated:
        html_parts.append(f'<p class="section-note">共 {total_count} 条，显示前 {shown} 条</p>')
    else:
        html_parts.append(f'<p class="section-note">共 {total_count} 条</p>')

    html_parts.append('<table><tr><th>#</th><th>Query</th><th>Target Label</th>'
                      '<th>Expected Segment</th><th>首次命中</th><th>状态</th>'
                      '<th>实际返回</th><th>有效 K</th><th>窗口状态</th></tr>')

    for i, d in enumerate(records, 1):
        query = _safe_str(d.get("retrieval_query") or d.get("question", ""))
        target_label = _safe_str(d.get("target_label", ""))
        expected_seg = _safe_str(_short_id(d.get("expected_segment_id", "")))
        expected_seg_full = _safe_str(d.get("expected_segment_id", ""))
        pos = d.get("hit_evidence_position")
        pos_str = f"Top{pos}" if pos is not None else "未命中"
        status = "✅ 命中" if pos is not None and pos <= 5 else "❌ 未命中"

        # 召回规模信息
        actual = d.get("actual_returned_count", d.get("retrieval_result_count", "?"))
        eff_k = d.get("effective_k", "?")
        win_status = d.get("window_status", "")
        win_display = ""
        if win_status == "partial_window":
            reason = d.get("window_reason", "")
            reason_text = {"source_document_has_fewer_chunks": "文档 chunk 不足",
                           "unknown": "原因未知"}.get(reason, reason)
            win_display = f'<span class="warn">⚠️ partial ({reason_text})</span>'
        elif win_status == "full_window":
            win_display = "✅ full"

        html_parts.append(
            f'<tr><td>{i}</td>'
            f'<td title="{_safe_str(d.get("question", ""))}">{query[:60]}</td>'
            f'<td>{target_label}</td>'
            f'<td title="{expected_seg_full}">{expected_seg}</td>'
            f'<td>{pos_str}</td>'
            f'<td>{status}</td>'
            f'<td>{actual}</td>'
            f'<td>{eff_k}</td>'
            f'<td>{win_display}</td></tr>'
        )

    html_parts.append('</table>')

    # 展开详细检索结果（展示全部实际返回结果）
    for i, d in enumerate(records, 1):
        retrieval_calls = d.get("retrieval_calls") or []
        expected_seg = d.get("expected_segment_id", "")

        if retrieval_calls:
            # ── 多检索模式：按 call 展示 ──
            call_count = d.get("retrieval_call_count", len(retrieval_calls))
            total_results = d.get("retrieval_result_count", 0)
            subquery_hit = d.get("subquery_hit_count", 0)
            per_call_hits = d.get("per_call_hits") or []

            html_parts.append(f'<details><summary>#{i} 多检索 ({call_count} 次调用, '
                              f'共 {total_results} 条结果, {subquery_hit}/{call_count} 次命中)</summary>')
            html_parts.append(f'<p class="section-note">Expected: <code>{_safe_str(expected_seg)}</code></p>')

            for call in retrieval_calls:
                call_order = call.get("order", "?")
                call_query = _safe_str(call.get("query") or "(无 query)")
                call_latency = call.get("latency_ms")
                call_result_count = call.get("result_count", 0)
                call_obs_id = _safe_str(_short_id(call.get("observation_id", "")))

                # 查找该 call 的命中状态
                call_hit_info = ""
                for h in per_call_hits:
                    if h.get("order") == call_order:
                        hp = h.get("hit_position")
                        if hp is not None:
                            call_hit_info = f' <span class="hit">✅ 命中 Top{hp}</span>'
                        else:
                            call_hit_info = ' <span class="miss">❌ 未命中</span>'
                        break

                latency_str = f"{call_latency}ms" if call_latency is not None else "N/A"
                html_parts.append(f'<div style="margin:8px 0;padding:8px;background:#f8f9fa;border-radius:4px;">')
                html_parts.append(f'<strong>检索 #{call_order}</strong>: '
                                  f'<code>{call_query[:80]}</code>{call_hit_info}<br>')
                html_parts.append(f'<span class="section-note">'
                                  f'observation: <code>{call_obs_id}</code> | '
                                  f'延迟: {latency_str} | '
                                  f'返回: {call_result_count} 条</span>')

                call_results = call.get("results") or []
                if call_results:
                    html_parts.append('<table class="retrieval-table"><tr><th>Rank</th><th>Segment ID</th>'
                                      '<th>来源文档</th><th>Score</th><th>Content Hash</th><th>内容摘要</th>'
                                      '<th>是否目标</th></tr>')
                    for rr in call_results:
                        seg_id = _safe_str(_short_id(rr.get("segment_id", "")))
                        seg_id_full = _safe_str(rr.get("segment_id", ""))
                        doc_name = _safe_str(rr.get("document_name", "")[:20])
                        score = rr.get("score", "")
                        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else _safe_str(str(score))
                        content_hash = _safe_str(_short_id(rr.get("content_hash", ""), 8))
                        content = rr.get("content", "")
                        content_summary = content[:100] + ("..." if len(content) > 100 else "")
                        is_exp = rr.get("is_expected", False)
                        marker = ' 🎯' if is_exp else ""
                        row_class = ' class="hit"' if is_exp else ""
                        html_parts.append(
                            f'<tr{row_class}><td>{rr.get("position", "")}{marker}</td>'
                            f'<td title="{seg_id_full}">{seg_id}</td>'
                            f'<td>{doc_name}</td>'
                            f'<td>{score_str}</td>'
                            f'<td>{content_hash}</td>'
                            f'<td class="content-cell">{_safe_str(content_summary)}</td>'
                            f'<td>{"✅" if is_exp else ""}</td></tr>'
                        )
                    html_parts.append('</table>')
                html_parts.append('</div>')

            html_parts.append('</details>')
        else:
            # ── 单检索模式（向后兼容） ──
            ret_results = d.get("retrieval_results") or []
            if ret_results:
                query = d.get("retrieval_query") or d.get("question", "")
                actual_count = d.get("actual_returned_count", len(ret_results))
                configured = d.get("configured_top_k", "?")
                window_note = ""
                if d.get("window_status") == "partial_window":
                    window_note = f' <span class="warn">⚠️ 实际返回 {actual_count} 条 &lt; 配置 {configured}</span>'

                html_parts.append(f'<details><summary>#{i} {_safe_str(query[:50])} '
                                  f'— 实际返回 {actual_count} 条{window_note}</summary>')
                html_parts.append(f'<p class="section-note">Expected: <code>{_safe_str(expected_seg)}</code></p>')
                html_parts.append('<table class="retrieval-table"><tr><th>Rank</th><th>Segment ID</th>'
                                  '<th>来源文档</th><th>Score</th><th>Content Hash</th><th>内容摘要</th>'
                                  '<th>是否目标</th></tr>')
                for rr in ret_results:
                    seg_id = _safe_str(_short_id(rr.get("segment_id", "")))
                    seg_id_full = _safe_str(rr.get("segment_id", ""))
                    doc_name = _safe_str(rr.get("document_name", "")[:20])
                    score = rr.get("score", "")
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else _safe_str(str(score))
                    content_hash = _safe_str(_short_id(rr.get("content_hash", ""), 8))
                    content = rr.get("content", "")
                    content_summary = content[:100] + ("..." if len(content) > 100 else "")
                    is_exp = rr.get("is_expected", False)
                    marker = ' 🎯' if is_exp else ""
                    row_class = ' class="hit"' if is_exp else ""
                    html_parts.append(
                        f'<tr{row_class}><td>{rr.get("position", "")}{marker}</td>'
                        f'<td title="{seg_id_full}">{seg_id}</td>'
                        f'<td>{doc_name}</td>'
                        f'<td>{score_str}</td>'
                        f'<td>{content_hash}</td>'
                        f'<td class="content-cell">{_safe_str(content_summary)}</td>'
                        f'<td>{"✅" if is_exp else ""}</td></tr>'
                    )
                html_parts.append('</table></details>')


def _render_chunk_exact_sample_appendix(html_parts, all_judge_results, sample_lookup):
    """渲染 chunk_exact 样本审计附录（按 run 分组，可折叠）。"""
    valid = [r for r in all_judge_results
             if "error" not in r and r.get("evaluation_track") == TRACK_CHUNK_EXACT]
    if not valid:
        return

    # 按 run_id 分组
    by_run = {}
    for r in valid:
        rid = r.get("_source_run_id", r.get("run_id", "unknown"))
        by_run.setdefault(rid, []).append(r)

    html_parts.append('<h2>附录：Chunk Exact 样本明细</h2>')

    for rid, results in sorted(by_run.items()):
        evaluable = [r for r in results
                     if r.get("retrieval_evaluable", True) is not False
                     and r.get("retrieval_top1_hit") is not None]
        unevaluable = [r for r in results if r not in evaluable]

        n = len(evaluable)
        t1 = sum(r.get("retrieval_top1_hit", 0) for r in evaluable) if n else 0
        t3 = sum(r.get("retrieval_top3_hit", 0) for r in evaluable) if n else 0
        t5 = sum(r.get("retrieval_top5_hit", 0) for r in evaluable) if n else 0
        t10 = sum(r.get("retrieval_top10_hit", 0) for r in evaluable) if n else 0

        # 默认展开有问题的 run
        has_issues = any(r.get("retrieval_top1_hit", 0) == 0 for r in evaluable)
        open_attr = " open" if has_issues else ""

        html_parts.append(
            f'<details{open_attr}><summary>'
            f'<strong>{_safe_str(rid)}</strong> — '
            f'可评测 {n}/{len(results)} | '
            f'Top1 {t1}/{n} | Top3 {t3}/{n} | Top5 {t5}/{n} | Top10 {t10}/{n}'
            f'</summary>'
        )

        html_parts.append(
            '<table><tr><th>#</th><th>Query</th><th>Target</th>'
            '<th>Expected Segment</th><th>Hit Rank</th><th>Top1</th><th>Top3</th><th>Top5</th><th>Top10</th><th>Status</th></tr>'
        )

        for i, r in enumerate(results, 1):
            query = _safe_str(r.get("question", ""))[:40]
            target = _safe_str(r.get("target_label", ""))
            expected_seg = _safe_str(_short_id(r.get("expected_segment_id", "")))
            expected_full = _safe_str(r.get("expected_segment_id", ""))
            pos = r.get("hit_evidence_position")
            pos_str = f"Top{pos}" if pos is not None else "—"
            t1v = r.get("retrieval_top1_hit")
            t3v = r.get("retrieval_top3_hit")
            t5v = r.get("retrieval_top5_hit")
            t10v = r.get("retrieval_top10_hit")
            t1s = "✅" if t1v else ("❌" if t1v == 0 else "—")
            t3s = "✅" if t3v else ("❌" if t3v == 0 else "—")
            t5s = "✅" if t5v else ("❌" if t5v == 0 else "—")
            t10s = "✅" if t10v else ("❌" if t10v == 0 else "—")

            ce_status = r.get("chunk_exact_status", "")
            if ce_status:
                status = f"⚠️ {ce_status}"
            elif t1v:
                status = "✅ Top1"
            elif t3v:
                status = "✅ Top3"
            elif t5v:
                status = "✅ Top5"
            elif t10v:
                status = "✅ Top10"
            else:
                status = "❌ 未命中"

            html_parts.append(
                f'<tr><td>{i}</td><td>{query}</td><td>{target}</td>'
                f'<td title="{expected_full}">{expected_seg}</td>'
                f'<td>{pos_str}</td><td>{t1s}</td><td>{t3s}</td><td>{t5s}</td><td>{t10s}</td>'
                f'<td>{status}</td></tr>'
            )

        html_parts.append('</table></details>')


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


# ====== 一致性校验 ======

def validate_report_consistency(all_judge_results, run_data_list, cumulative_metrics):
    """校验报告各处 n 与 TopK 指标是否一致。

    Returns:
        list[str]: 不一致的错误描述列表，空列表表示一致。
    """
    errors = []
    valid = [r for r in all_judge_results if "error" not in r]

    # 全局 chunk_exact
    ce_all = [r for r in valid if r.get("evaluation_track") == TRACK_CHUNK_EXACT]
    ce_eval = [r for r in ce_all
               if r.get("retrieval_evaluable", True) is not False
               and r.get("retrieval_top1_hit") is not None]
    ce_n = len(ce_eval)

    # 累计指标中的 chunk_exact
    cm_ce_n = cumulative_metrics.get("chunk_exact_evaluable_count", 0)
    if ce_n != cm_ce_n:
        errors.append(
            f"Chunk Exact 可评测数不一致：全局统计 {ce_n}，累计指标 {cm_ce_n}"
        )

    # TopK 一致性（命中数不能超过样本数）
    for k_field in ("retrieval_top1_hit", "retrieval_top3_hit", "retrieval_top5_hit", "retrieval_top10_hit"):
        hit_count = sum(r.get(k_field, 0) for r in ce_eval)
        if hit_count > ce_n:
            errors.append(
                f"Chunk Exact {k_field} 命中数 {hit_count} 超过可评测样本数 {ce_n}"
            )

    # 各 run 的 chunk_exact 分母一致性
    for rd in run_data_list:
        rs = rd.get("run_status", {})
        run = rd.get("run", {})
        qsid = rs.get("question_set_id") or run.get("question_set_id", "") or "未知"
        _jr = rs.get("judge_results", [])
        _ce = [r for r in _jr if "error" not in r
               and r.get("evaluation_track") == TRACK_CHUNK_EXACT]
        _ce_eval = [r for r in _ce
                    if r.get("retrieval_evaluable", True) is not False
                    and r.get("retrieval_top1_hit") is not None]
        run_n = len(_ce_eval)
        run_t1 = sum(r.get("retrieval_top1_hit", 0) for r in _ce_eval)
        run_t3 = sum(r.get("retrieval_top3_hit", 0) for r in _ce_eval)
        run_t5 = sum(r.get("retrieval_top5_hit", 0) for r in _ce_eval)
        if run_t1 > run_n:
            errors.append(f"Run {qsid[:12]}: Top1 命中 {run_t1} > 可评测 {run_n}")
        if run_t3 > run_n:
            errors.append(f"Run {qsid[:12]}: Top3 命中 {run_t3} > 可评测 {run_n}")
        if run_t5 > run_n:
            errors.append(f"Run {qsid[:12]}: Top5 命中 {run_t5} > 可评测 {run_n}")

    return errors


# ====== 分层指标 ======

def _build_layered_metrics(chunk_exact_evaluable, sample_lookup,
                           question_meta_lookup=None):
    """按 query_style 和 source document 构建分层指标。

    Args:
        chunk_exact_evaluable: 可评测的 chunk_exact judged results
        sample_lookup: {trace_id: processed_sample}
        question_meta_lookup: {(qsid, qid): meta} 题集元数据查找表

    Returns:
        dict: {
            "by_query_style": {style: {n, t1, t3, t5, t10}},
            "by_doc": {doc_label: {n, t1, t3, t5, t10}},
        }
    """
    question_meta_lookup = question_meta_lookup or {}

    def _group_by(key_fn):
        groups = {}
        for r in chunk_exact_evaluable:
            key = key_fn(r) or "未知"
            groups.setdefault(key, []).append(r)
        result = {}
        for key, items in groups.items():
            n = len(items)
            result[key] = {
                "n": n,
                "t1": sum(r.get("retrieval_top1_hit", 0) for r in items),
                "t3": sum(r.get("retrieval_top3_hit", 0) for r in items),
                "t5": sum(r.get("retrieval_top5_hit", 0) for r in items),
                "t10": sum(r.get("retrieval_top10_hit", 0) for r in items),
            }
        return result

    def _query_style_label(r):
        # 优先从 judged result 读取（新数据可能已有）
        style = r.get("query_style", "")
        if style:
            return style
        # 回退：从题集 JSONL 元数据查找
        meta = _lookup_question_meta(r, question_meta_lookup)
        return meta.get("query_style", "")

    by_style = _group_by(_query_style_label)

    def _doc_label(r):
        tid = r.get("trace_id", "")
        sample = sample_lookup.get(tid, {})
        # 优先级：
        # 1. judged result 的 document_name（chunk_exact 题目绑定的源文档）
        # 2. 题集 JSONL 元数据的 document_name
        # 3. processed sample 的 source_file_name（题集文件名，非源文档）
        # 4. processed sample 的 document_name
        doc_name = r.get("document_name", "")
        if not doc_name:
            meta = _lookup_question_meta(r, question_meta_lookup)
            doc_name = meta.get("document_name", "")
        if not doc_name:
            doc_name = sample.get("source_file_name", "")
        if not doc_name:
            doc_name = sample.get("document_name", "")
        if not doc_name:
            return "未知"
        # 提取文件扩展名
        for ext in (".docx", ".xlsx", ".xls", ".pdf", ".csv", ".txt", ".md"):
            if doc_name.lower().endswith(ext):
                return f"{doc_name} ({ext[1:]})"
        return doc_name

    by_doc = _group_by(_doc_label)

    return {"by_query_style": by_style, "by_doc": by_doc}


def _render_layered_table(groups, group_label, threshold=5):
    """渲染分层指标表。样本数 < threshold 时标注仅供观察。"""
    if not groups:
        return f'<p class="section-note">暂无按{group_label}分组的数据</p>'

    parts = [
        '<table><tr><th>', group_label, '</th><th>n</th><th>Top1</th><th>Top3</th>'
        '<th>Top5</th><th>Top10</th><th>Top1→Top3 排名损失</th></tr>'
    ]
    for key in sorted(groups.keys(), key=lambda k: groups[k]["n"], reverse=True):
        g = groups[key]
        n = g["n"]
        t1, t3, t5, t10 = g["t1"], g["t3"], g["t5"], g["t10"]
        loss = t3 - t1  # Top1→Top3 排名损失
        obs_note = ' <span class="warn">（仅供观察）</span>' if n < threshold else ''
        parts.append(
            f'<tr><td>{_safe_str(key)}{obs_note}</td><td>{n}</td>'
            f'<td>{t1}/{n} ({_pct(t1/n)})</td>'
            f'<td>{t3}/{n} ({_pct(t3/n)})</td>'
            f'<td>{t5}/{n} ({_pct(t5/n)})</td>'
            f'<td>{t10}/{n} ({_pct(t10/n)})</td>'
            f'<td>{loss}</td></tr>'
        )
    parts.append('</table>')
    return "".join(parts)


def _render_recall_overview_section(configured_top_k, knowledge_base_total_chunks,
                                    total_documents, doc_chunk_counts,
                                    recall_stats):
    """渲染召回规模概览 HTML 区块。"""
    parts = []

    # 全局配置信息
    kb_display = knowledge_base_total_chunks if knowledge_base_total_chunks is not None else "未知"
    doc_display = total_documents if total_documents is not None else "未知"

    parts.append('<div class="info-box">')
    parts.append(f'<strong>召回规模配置</strong><br>')
    parts.append(f'配置 TopK: <strong>{configured_top_k}</strong> | '
                 f'知识库总 chunk 数: <strong>{kb_display}</strong> | '
                 f'文档数: <strong>{doc_display}</strong>')
    parts.append('</div>')

    # 每文档 chunk 数量统计
    if doc_chunk_counts:
        parts.append('<details open><summary><strong>各文档 chunk 数量统计</strong></summary>')
        parts.append('<table><tr><th>文档名</th><th>chunk 总数</th></tr>')
        for doc_name, count in sorted(doc_chunk_counts.items()):
            parts.append(f'<tr><td>{_safe_str(doc_name)}</td><td>{count}</td></tr>')
        parts.append('</table></details>')

    # 实际返回数量分布
    dist = recall_stats.get("return_distribution", {})
    if dist:
        parts.append('<details open><summary><strong>实际返回数量分布</strong></summary>')
        parts.append('<table><tr><th>返回数量</th><th>题目数</th></tr>')
        for bucket in _return_distribution_buckets(configured_top_k):
            cnt = dist.get(bucket, 0)
            parts.append(f'<tr><td>{bucket} 条</td><td>{cnt}</td></tr>')
        parts.append('</table></details>')

    # 窗口状态汇总
    full = recall_stats.get("full_window_count", 0)
    partial = recall_stats.get("partial_window_count", 0)
    rate = recall_stats.get("partial_window_rate", 0)
    parts.append(f'<p class="section-note">窗口状态: '
                 f'full_window <strong>{full}</strong> | '
                 f'partial_window <strong>{partial}</strong> '
                 f'({rate:.1%})</p>')

    return "".join(parts)


def _render_doc_level_recall_table(doc_stats, configured_top_k):
    """渲染文档级召回统计表。"""
    if not doc_stats:
        return '<p class="section-note">暂无文档级统计数据</p>'

    parts = [
        '<table><tr><th>文档名</th><th>chunk 总数</th><th>题目数</th>'
        '<th>平均返回数</th><th>窗口受限题数</th>'
        '<th>Top1</th><th>Top3</th><th>Top5</th><th>Top10</th>'
        '<th>Top10 未命中</th><th>窗口受限比例</th></tr>'
    ]

    for d in doc_stats:
        n = d["question_count"]
        chunk_display = d["chunk_count"] if d["chunk_count"] is not None else "未知"
        t1_str = f'{d["t1"]}/{n} ({_pct(d["t1"]/n)})' if n else "N/A"
        t3_str = f'{d["t3"]}/{n} ({_pct(d["t3"]/n)})' if n else "N/A"
        t5_str = f'{d["t5"]}/{n} ({_pct(d["t5"]/n)})' if n else "N/A"
        t10_str = f'{d["t10"]}/{n} ({_pct(d["t10"]/n)})' if n else "N/A"

        parts.append(
            f'<tr><td>{_safe_str(d["doc_name"])}</td>'
            f'<td>{chunk_display}</td>'
            f'<td>{n}</td>'
            f'<td>{d["avg_returned"]}</td>'
            f'<td>{d["partial_count"]}</td>'
            f'<td>{t1_str}</td>'
            f'<td>{t3_str}</td>'
            f'<td>{t5_str}</td>'
            f'<td>{t10_str}</td>'
            f'<td>{d["miss10"]}</td>'
            f'<td>{_pct(d["partial_rate"])}</td></tr>'
        )

    parts.append('</table>')
    parts.append(f'<p class="section-note">配置 TopK = {configured_top_k}；'
                 f'"窗口受限"指实际返回数 &lt; 配置 TopK</p>')
    return "".join(parts)


# ====== 排名诊断 ======

def _build_ranking_diagnostics(chunk_exact_evaluable):
    """构建互斥排名分布。"""
    buckets = {"top1": 0, "top2_3": 0, "top4_5": 0, "top6_10": 0, "top10_miss": 0}
    for r in chunk_exact_evaluable:
        t1 = r.get("retrieval_top1_hit", 0)
        t10 = r.get("retrieval_top10_hit", 0)
        pos = r.get("hit_evidence_position")
        if t1:
            buckets["top1"] += 1
        elif pos is not None and 2 <= pos <= 3:
            buckets["top2_3"] += 1
        elif pos is not None and 4 <= pos <= 5:
            buckets["top4_5"] += 1
        elif pos is not None and 6 <= pos <= 10:
            buckets["top6_10"] += 1
        elif t10:
            buckets["top6_10"] += 1
        else:
            buckets["top10_miss"] += 1
    return buckets


# ====== 召回规模信息 ======

def _resolve_doc_name(judge_result, sample=None, question_meta_lookup=None):
    """解析文档名称，优先级：judge_result > 题集元数据 > sample。"""
    doc_name = judge_result.get("document_name", "")
    if not doc_name and question_meta_lookup:
        meta = _lookup_question_meta(judge_result, question_meta_lookup)
        doc_name = meta.get("document_name", "")
    if not doc_name and sample:
        doc_name = sample.get("source_file_name", "") or sample.get("document_name", "")
    return doc_name or "未知"


def _determine_partial_window_reason(doc_name, configured_top_k, doc_chunk_counts=None):
    """判断 partial_window 原因（保守策略）。

    仅当文档 chunk 总数明确小于 configured_top_k 时才归因为
    "source_document_has_fewer_chunks"，其余情况一律 "unknown"。
    """
    if doc_chunk_counts and doc_name in doc_chunk_counts:
        if doc_chunk_counts[doc_name] < configured_top_k:
            return "source_document_has_fewer_chunks"
    return "unknown"


def _compute_sample_recall_info(judge_result, sample_lookup, configured_top_k=10,
                                knowledge_base_total_chunks=None, doc_chunk_counts=None,
                                question_meta_lookup=None):
    """计算单个 chunk_exact 样本的召回规模信息。

    Returns:
        dict: {
            "actual_returned_count": int,
            "effective_k": int,
            "window_status": "full_window" | "partial_window",
            "window_reason": str,
            "configured_top_k": int,
            "knowledge_base_total_chunks": int | None,
            "target_source_document": str,
            "target_source_document_chunk_count": int | None,
        }
    """
    tid = judge_result.get("trace_id", "")
    sample = sample_lookup.get(tid)
    retrieval_results = (sample.get("retrieval_results") or []) if sample else []
    actual_returned_count = len(retrieval_results)
    effective_k = min(configured_top_k, actual_returned_count)

    doc_name = _resolve_doc_name(judge_result, sample, question_meta_lookup)
    doc_chunks = None
    if doc_chunk_counts and doc_name in doc_chunk_counts:
        doc_chunks = doc_chunk_counts[doc_name]

    if actual_returned_count >= configured_top_k:
        window_status = "full_window"
        window_reason = ""
    else:
        window_status = "partial_window"
        window_reason = _determine_partial_window_reason(doc_name, configured_top_k, doc_chunk_counts)

    return {
        "actual_returned_count": actual_returned_count,
        "effective_k": effective_k,
        "window_status": window_status,
        "window_reason": window_reason,
        "configured_top_k": configured_top_k,
        "knowledge_base_total_chunks": knowledge_base_total_chunks,
        "target_source_document": doc_name,
        "target_source_document_chunk_count": doc_chunks,
    }


def _compute_recall_statistics(chunk_exact_evaluable, sample_lookup, configured_top_k=10,
                               knowledge_base_total_chunks=None, doc_chunk_counts=None,
                               question_meta_lookup=None):
    """计算全局召回规模统计。

    Returns:
        dict: {
            "return_distribution": {bucket_label: count},
            "full_window_count": int,
            "partial_window_count": int,
            "partial_window_rate": float,
            "per_sample_info": list[dict],
        }
    """
    distribution = {}
    for bucket in _return_distribution_buckets(configured_top_k):
        distribution[bucket] = 0

    full_window_count = 0
    partial_window_count = 0
    per_sample_info = []

    for r in chunk_exact_evaluable:
        info = _compute_sample_recall_info(
            r, sample_lookup, configured_top_k,
            knowledge_base_total_chunks, doc_chunk_counts, question_meta_lookup)
        per_sample_info.append(info)
        n = info["actual_returned_count"]
        bucket = _return_count_to_bucket(n, configured_top_k)
        distribution[bucket] = distribution.get(bucket, 0) + 1

        if info["window_status"] == "full_window":
            full_window_count += 1
        else:
            partial_window_count += 1

    total = len(chunk_exact_evaluable) or 1
    return {
        "return_distribution": distribution,
        "full_window_count": full_window_count,
        "partial_window_count": partial_window_count,
        "partial_window_rate": partial_window_count / total,
        "per_sample_info": per_sample_info,
    }


def _return_distribution_buckets(configured_top_k=10):
    """生成返回数量分布的桶标签。"""
    return ["1-3", "4-5", f"6-{configured_top_k - 1}", f"{configured_top_k}+"]


def _return_count_to_bucket(n, configured_top_k=10):
    """将实际返回数映射到分布桶。"""
    if n <= 3:
        return "1-3"
    elif n <= 5:
        return "4-5"
    elif n < configured_top_k:
        return f"6-{configured_top_k - 1}"
    else:
        return f"{configured_top_k}+"


def _compute_doc_level_recall_stats(chunk_exact_evaluable, sample_lookup, configured_top_k=10,
                                    doc_chunk_counts=None, question_meta_lookup=None):
    """按 source document 统计召回规模。

    Returns:
        list[dict]: 按题目数降序排列，每项包含：
            doc_name, chunk_count, question_count, avg_returned,
            partial_count, t1, t3, t5, t10, miss10, partial_rate
    """
    question_meta_lookup = question_meta_lookup or {}
    groups = {}
    for r in chunk_exact_evaluable:
        tid = r.get("trace_id", "")
        sample = sample_lookup.get(tid)
        doc_name = _resolve_doc_name(r, sample, question_meta_lookup)
        groups.setdefault(doc_name, []).append(r)

    stats = []
    for doc_name, items in groups.items():
        n = len(items)
        total_returned = 0
        partial_count = 0
        for r in items:
            tid = r.get("trace_id", "")
            sample = sample_lookup.get(tid)
            ret_count = len(sample.get("retrieval_results") or []) if sample else 0
            total_returned += ret_count
            if ret_count < configured_top_k:
                partial_count += 1

        t1 = sum(r.get("retrieval_top1_hit", 0) for r in items)
        t3 = sum(r.get("retrieval_top3_hit", 0) for r in items)
        t5 = sum(r.get("retrieval_top5_hit", 0) for r in items)
        t10 = sum(r.get("retrieval_top10_hit", 0) for r in items)
        miss10 = sum(1 for r in items if r.get("retrieval_top10_hit", 0) == 0)

        chunk_count = None
        if doc_chunk_counts and doc_name in doc_chunk_counts:
            chunk_count = doc_chunk_counts[doc_name]

        stats.append({
            "doc_name": doc_name,
            "chunk_count": chunk_count,
            "question_count": n,
            "avg_returned": round(total_returned / n, 1) if n else 0,
            "partial_count": partial_count,
            "t1": t1, "t3": t3, "t5": t5, "t10": t10,
            "miss10": miss10,
            "partial_rate": partial_count / n if n else 0,
        })

    stats.sort(key=lambda x: x["question_count"], reverse=True)
    return stats


# ====== 失败样本证据对照 ======

def _build_top1_miss_evidence(chunk_exact_evaluable, sample_lookup,
                               question_meta_lookup=None, max_samples=50):
    """为 Top1 未中样本构建完整证据对照。

    Returns:
        list[dict]: 每项包含完整诊断信息和全部 TopK 返回结果
    """
    question_meta_lookup = question_meta_lookup or {}
    records = []
    for r in chunk_exact_evaluable:
        if r.get("retrieval_top1_hit", 0) == 1:
            continue
        tid = r.get("trace_id", "")
        sample = sample_lookup.get(tid, {})
        pos = r.get("hit_evidence_position")
        t10 = r.get("retrieval_top10_hit", 0)

        # 确定分类
        if not t10:
            category = "Top10 未召回"
        elif pos is not None and 6 <= pos <= 10:
            category = "Top6-10 排序偏后"
        elif pos is not None and 4 <= pos <= 5:
            category = "Top4-5 排序偏后"
        elif pos is not None and 2 <= pos <= 3:
            category = "Top2-3 排序偏后"
        else:
            category = "排序偏后"

        # 从 judge_result + 题集查找表获取完整元数据
        meta = _lookup_question_meta(r, question_meta_lookup)

        # expected_content：优先 judge_result，回退 sample，再回退题集
        expected_content = (r.get("expected_content")
                            or sample.get("expected_content", "")
                            or meta.get("expected_content", ""))
        expected_content_hash = (r.get("expected_content_hash", "")
                                 or meta.get("expected_content_hash", ""))
        expected_segment_id = (r.get("expected_segment_id", "")
                               or meta.get("expected_segment_id", ""))

        # 实际返回结果（全部 TopK，不截断）
        sample_found = bool(tid and tid in sample_lookup)
        retrieval_results = sample.get("retrieval_results", []) if sample_found else []
        actual_returned_count = len(retrieval_results)
        top_results = []
        for i, rr in enumerate(retrieval_results):
            top_results.append({
                "rank": i + 1,
                "segment_id": rr.get("segment_id", ""),
                "document_name": rr.get("document_name", ""),
                "position": rr.get("position"),
                "content": rr.get("content", ""),
                "score": rr.get("score"),
                "is_expected": (rr.get("segment_id", "") == expected_segment_id),
            })

        # 分数比较
        expected_score = None
        top1_score = None
        for tr in top_results:
            if tr["is_expected"]:
                expected_score = tr["score"]
            if tr["rank"] == 1:
                top1_score = tr["score"]

        # score 与排序一致性检查
        score_rank_mismatch = False
        if len(top_results) >= 2 and top_results[0]["score"] is not None and top_results[1]["score"] is not None:
            if top_results[1]["score"] > top_results[0]["score"]:
                score_rank_mismatch = True

        # 元数据缺失标记
        missing_fields = []
        for field in ("query_style", "retrieval_intent", "target_fact", "expected_content"):
            if not meta.get(field):
                missing_fields.append(field)

        # review_label: 历史数据无此字段时默认 unreviewed
        review_label = r.get("review_label", "unreviewed")
        valid_labels = {"unreviewed", "query_ambiguous", "near_neighbor",
                        "chunk_boundary", "ranking_error", "gold_error",
                        "insufficient_evidence"}
        if review_label not in valid_labels:
            review_label = "unreviewed"

        records.append({
            "query": meta.get("query_style") and r.get("question") or r.get("retrieval_query") or r.get("question", ""),
            "query_style": meta.get("query_style", ""),
            "retrieval_intent": meta.get("retrieval_intent", ""),
            "target_fact": meta.get("target_fact", ""),
            "target_label": meta.get("target_label", "") or r.get("target_label", ""),
            "question_id": r.get("question_id", ""),
            "trace_id": tid,
            "expected_segment_id": expected_segment_id,
            "expected_content_hash": expected_content_hash,
            "expected_content": expected_content,
            "expected_doc": meta.get("document_name", "") or r.get("document_name", ""),
            "expected_doc_id": meta.get("document_id", "") or r.get("document_id", ""),
            "expected_source_position": meta.get("source_position", "") or r.get("source_position", ""),
            "hit_position": pos,
            "category": category,
            "review_label": review_label,
            "top_results": top_results,
            "actual_returned_count": actual_returned_count,
            "sample_found": sample_found,
            "expected_score": expected_score,
            "top1_score": top1_score,
            "score_rank_mismatch": score_rank_mismatch,
            "missing_fields": missing_fields,
        })
        if len(records) >= max_samples:
            break
    return records


def _render_top1_miss_evidence(records, total_count, config_top_k=10):
    """渲染 Top1 未中样本证据对照（完整诊断版）。"""
    if not records:
        return '<p class="section-note">无 Top1 未命中样本</p>'

    parts = []
    shown = len(records)
    if total_count > shown:
        parts.append(f'<p class="section-note">共 {total_count} 条 Top1 未命中，显示前 {shown} 条</p>')
    else:
        parts.append(f'<p class="section-note">共 {total_count} 条 Top1 未命中</p>')

    # review_label 计数表
    label_counts = {}
    for rec in records:
        label = rec.get("review_label", "unreviewed")
        label_counts[label] = label_counts.get(label, 0) + 1
    _label_display = {
        "unreviewed": "未审核",
        "query_ambiguous": "查询歧义",
        "near_neighbor": "近邻 chunk",
        "chunk_boundary": "切块边界",
        "ranking_error": "排序异常",
        "gold_error": "金标准错误",
        "insufficient_evidence": "信息不足",
    }
    parts.append('<details open><summary><strong>诊断分类统计</strong></summary>')
    parts.append('<table><tr><th>分类</th><th>数量</th></tr>')
    for label in ("unreviewed", "query_ambiguous", "near_neighbor", "chunk_boundary",
                  "ranking_error", "gold_error", "insufficient_evidence"):
        cnt = label_counts.get(label, 0)
        if cnt > 0:
            display = _label_display.get(label, label)
            parts.append(f'<tr><td>{display}</td><td>{cnt}</td></tr>')
    parts.append('</table></details>')

    for i, rec in enumerate(records, 1):
        parts.append(f'<div class="diag-card">')

        # ── 诊断头 ──
        parts.append(f'<h4>#{i} {_safe_str(rec["query"][:80])}</h4>')
        parts.append(f'<div class="diag-meta">')
        parts.append(f'<span>分类: <strong>{_safe_str(rec["category"])}</strong></span>')
        if rec["query_style"]:
            parts.append(f'<span>query_style: <code>{_safe_str(rec["query_style"])}</code></span>')
        if rec["target_label"]:
            parts.append(f'<span>标签: {_safe_str(rec["target_label"])}</span>')
        _actual = rec["actual_returned_count"]
        _eff_k = min(config_top_k, _actual)
        _win_note = ""
        if _actual < config_top_k:
            _win_note = f' <span class="warn">⚠️ partial_window（有效 K={_eff_k}）</span>'
        parts.append(f'<span>配置 TopK: {config_top_k} | 实际返回: {_actual} | 有效 K: {_eff_k}{_win_note}</span>')
        parts.append(f'</div>')

        # review_label
        _rl = rec.get("review_label", "unreviewed")
        _rl_display = _label_display.get(_rl, _rl)
        parts.append(f'<div class="diag-meta">')
        parts.append(f'<span>诊断分类: <strong>{_safe_str(_rl_display)}</strong> <code>({_safe_str(_rl)})</code></span>')
        parts.append(f'</div>')

        # 可折叠 ID
        parts.append(f'<div class="diag-meta">')
        parts.append(f'<span>question_id: <code>{_safe_str(rec["question_id"] or "缺失")}</code></span>')
        parts.append(f'<span>trace_id: <code>{_safe_str(rec["trace_id"][:16])}…</code></span>')
        parts.append(f'</div>')

        # ── 用户检索意图 ──
        if rec["retrieval_intent"]:
            parts.append(f'<p><strong>用户检索意图 (retrieval_intent)</strong>: {_safe_str(rec["retrieval_intent"])}</p>')
        elif "retrieval_intent" in rec.get("missing_fields", []):
            parts.append(f'<p><strong>用户检索意图</strong>: <span class="warn">缺失</span></p>')

        # ── 证据锚点 ──
        if rec["target_fact"]:
            parts.append(f'<p><strong>标准答案/证据锚点 (target_fact)</strong>: {_safe_str(rec["target_fact"])}</p>')
        elif "target_fact" in rec.get("missing_fields", []):
            parts.append(f'<p><strong>证据锚点</strong>: <span class="warn">缺失</span></p>')

        # ── 紧凑对照区：目标 vs Top1（默认展开） ──
        top_results = rec.get("top_results", [])
        top1_chunk = top_results[0] if top_results else None
        exp_seg = _safe_str(rec["expected_segment_id"])
        exp_hash = _safe_str(rec["expected_content_hash"][:16]) if rec["expected_content_hash"] else "缺失"
        exp_doc = _safe_str(rec["expected_doc"] or "缺失")
        exp_pos = _safe_str(rec["expected_source_position"]) if rec["expected_source_position"] else ""
        exp_content = rec.get("expected_content", "")
        exp_summary = exp_content[:300] + ("..." if len(exp_content) > 300 else "")

        # 只在目标不是 Top1 时展示对照
        target_is_top1 = (top1_chunk and top1_chunk.get("is_expected", False))
        if not target_is_top1:
            parts.append('<div style="display:flex;gap:16px;margin:12px 0;flex-wrap:wrap;">')

            # 左侧：目标 chunk
            parts.append('<div style="flex:1;min-width:300px;border:2px solid #28a745;border-radius:8px;padding:12px;background:#f0fff0;">')
            parts.append('<strong>🎯 目标 chunk</strong>')
            _rank_str = f'Top{rec["hit_position"]}' if rec["hit_position"] else "未命中"
            _score_str = f'{rec["expected_score"]:.4f}' if rec["expected_score"] is not None else "N/A"
            parts.append(f'<div class="diag-meta">rank: <strong>{_rank_str}</strong> | score: {_score_str}'
                         f' | 文档: {exp_doc}'
                         f'{" | pos:" + exp_pos if exp_pos else ""}</div>')
            parts.append(f'<div class="diag-meta">segment: <code title="{exp_seg}">{_safe_str(_short_id(exp_seg))}</code>'
                         f' | hash: <code>{exp_hash}</code></div>')
            if rec["target_fact"]:
                parts.append(f'<p style="margin:6px 0;font-size:0.9em;"><em>{_safe_str(rec["target_fact"])}</em></p>')
            if exp_content:
                parts.append(f'<div class="gold-evidence" style="max-height:150px;overflow:auto;font-size:0.85em;">{_safe_str(exp_summary)}</div>')
            else:
                parts.append(f'<p class="warn" style="font-size:0.85em;">⚠️ expected_content 缺失</p>')
            parts.append('</div>')

            # 右侧：Top1 chunk
            if top1_chunk:
                t1_seg = _safe_str(top1_chunk.get("segment_id", ""))
                t1_doc = _safe_str(top1_chunk.get("document_name", "")[:30])
                t1_score = top1_chunk.get("score")
                t1_score_str = f"{t1_score:.4f}" if isinstance(t1_score, (int, float)) else "N/A"
                t1_content = top1_chunk.get("content", "")
                t1_summary = t1_content[:300] + ("..." if len(t1_content) > 300 else "")

                parts.append('<div style="flex:1;min-width:300px;border:2px solid #dc3545;border-radius:8px;padding:12px;background:#fff5f5;">')
                parts.append(f'<strong>📊 实际 Top1</strong>')
                parts.append(f'<div class="diag-meta">rank: <strong>Top1</strong> | score: {t1_score_str}'
                             f' | 文档: {t1_doc}</div>')
                parts.append(f'<div class="diag-meta">segment: <code title="{t1_seg}">{_safe_str(_short_id(t1_seg))}</code></div>')
                if t1_content:
                    parts.append(f'<div class="gold-evidence" style="max-height:150px;overflow:auto;font-size:0.85em;">{_safe_str(t1_summary)}</div>')
                else:
                    parts.append(f'<p class="no-result" style="font-size:0.85em;">无内容</p>')
                parts.append('</div>')

            parts.append('</div>')  # flex container end

        # ── 目标 chunk 完整信息（始终显示） ──
        parts.append(f'<p><strong>目标 chunk</strong>: '
                     f'segment <code title="{exp_seg}">{_safe_str(_short_id(exp_seg)) or "缺失"}</code>'
                     f' | content_hash: <code>{exp_hash}</code>'
                     f' | 文档: {exp_doc}'
                     f'{" | pos:" + exp_pos if exp_pos else ""}'
                     f' | 命中 rank: <strong>{"Top" + str(rec["hit_position"]) if rec["hit_position"] else "未命中"}</strong>'
                     f'</p>')

        # 分数信息
        if rec["expected_score"] is not None:
            score_parts = [f'目标 score: <strong>{rec["expected_score"]:.4f}</strong>']
            if rec["top1_score"] is not None:
                diff = rec["top1_score"] - rec["expected_score"]
                score_parts.append(f'Top1 score: {rec["top1_score"]:.4f}')
                score_parts.append(f'差值 (Top1-目标): {diff:+.4f}')
            parts.append(f'<p>{" | ".join(score_parts)}</p>')
        elif rec["hit_position"] is not None:
            parts.append(f'<p class="section-note">目标在 TopK 中但无 Dify 返回 score</p>')

        # ── expected_content 展示 ──
        if exp_content:
            summary = exp_content[:500] + ("..." if len(exp_content) > 500 else "")
            parts.append(f'<details><summary><strong>目标证据 (expected_content)</strong> — {len(exp_content)} 字</summary>')
            parts.append(f'<div class="gold-evidence">{_safe_str(exp_content)}</div>')
            parts.append(f'</details>')
            # 折叠状态下的摘要
            parts.append(f'<div class="gold-evidence" style="max-height:120px;overflow:hidden;">{_safe_str(summary)}</div>')
        else:
            parts.append(f'<p><strong>目标证据 (expected_content)</strong>: <span class="warn">⚠️ 缺失 — 无法校验证据完整性</span></p>')

        # 证据完整性校验
        integrity_notes = []
        if not exp_content:
            integrity_notes.append("expected_content 缺失，无法校验证据完整性")
        if not rec["expected_content_hash"]:
            integrity_notes.append("expected_content_hash 缺失")
        if rec.get("missing_fields"):
            integrity_notes.append(f"题集元数据缺失字段: {', '.join(rec['missing_fields'])}")
        if integrity_notes:
            parts.append(f'<p class="section-note">⚠️ 需复核: {"; ".join(integrity_notes)}</p>')

        # ── 实际返回列表（完整 TopK） ──
        if top_results:
            returned = rec["actual_returned_count"]
            short_warning = ""
            if returned < config_top_k:
                short_warning = (f' <span class="warn">⚠️ 实际只返回 {returned} 条，'
                                 f'本次 Top{config_top_k} 为不足窗口口径</span>')

            parts.append(f'<p><strong>实际返回 Top{returned}</strong>:{short_warning}</p>')
            parts.append(f'<p class="section-note">Dify 返回 score（字段排序语义未知，以 rank 为准）</p>')

            if rec.get("score_rank_mismatch"):
                parts.append(f'<p class="warn">⚠️ Top2 score 高于 Top1，score 与返回排序不一致，不可仅凭 score 判断 rerank</p>')

            parts.append('<table class="retrieval-table"><tr><th>Rank</th><th>Segment ID</th>'
                         '<th>来源文档</th><th>Score</th><th>内容摘要</th></tr>')
            for tr in top_results:
                is_exp = tr.get("is_expected", False)
                row_class = ' class="hit"' if is_exp else ""
                marker = ' 🎯 <strong>目标 chunk</strong>' if is_exp else ""
                seg_id = _safe_str(tr["segment_id"])
                seg_short = _safe_str(_short_id(seg_id))
                doc_name = _safe_str(tr.get("document_name", "")[:20])
                score = tr.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                content = tr.get("content", "")
                content_summary = content[:200] + ("..." if len(content) > 200 else "")

                parts.append(
                    f'<tr{row_class}><td>{tr["rank"]}{marker}</td>'
                    f'<td title="{seg_id}"><code>{seg_short}</code></td>'
                    f'<td>{doc_name}</td>'
                    f'<td>{score_str}</td>'
                    f'<td class="content-cell">{_safe_str(content_summary)}</td></tr>'
                )
            parts.append('</table>')

            # 可展开的完整 content
            for tr in top_results:
                if tr.get("content") and len(tr["content"]) > 200:
                    is_exp = tr.get("is_expected", False)
                    label = f'Rank {tr["rank"]}{"（目标 chunk）" if is_exp else ""} 完整内容'
                    parts.append(f'<details><summary>{label} — {len(tr["content"])} 字</summary>')
                    parts.append(f'<div class="gold-evidence">{_safe_str(tr["content"])}</div>')
                    parts.append('</details>')
        elif not rec.get("sample_found", True):
            # Judge 显示命中但 processed sample 未找到 — provenance 关联问题
            parts.append('<p class="warn">⚠️ 报告未找到该 run 的 processed retrieval evidence，无法展示返回列表。'
                         '这是 provenance/export error，不是 Dify retrieval miss。</p>')
        else:
            parts.append('<p class="no-result">processed sample 存在但无检索结果</p>')

        parts.append('</div>')

    return "".join(parts)


# ====== 数据质量旗标 ======

def _build_quality_flags(all_judge_results, sample_lookup, run_data_list):
    """构建数据质量旗标列表。"""
    flags = []
    valid = [r for r in all_judge_results if "error" not in r]
    ce_all = [r for r in valid if r.get("evaluation_track") == TRACK_CHUNK_EXACT]

    # missing_binding
    missing_binding = [r for r in ce_all if r.get("chunk_exact_status") == "missing_binding"]
    if missing_binding:
        flags.append(("warning", f"missing_binding: {len(missing_binding)} 条 chunk_exact 样本缺少 expected_segment_id/content_hash"))

    # no_trace
    no_trace = [r for r in ce_all if r.get("chunk_exact_status") == "no_trace"]
    if no_trace:
        flags.append(("warning", f"no_trace: {len(no_trace)} 条 chunk_exact 样本未关联真实 trace"))

    # no_retrieval
    no_retrieval = [r for r in ce_all if r.get("chunk_exact_status") == "no_retrieval"]
    if no_retrieval:
        flags.append(("warning", f"no_retrieval: {len(no_retrieval)} 条 chunk_exact 样本无检索结果"))

    # 缺少 retrieval_intent / target_fact
    missing_intent = sum(1 for r in ce_all if not r.get("retrieval_intent"))
    missing_fact = sum(1 for r in ce_all if not r.get("target_fact"))
    if missing_intent > 0:
        flags.append(("info", f"缺少 retrieval_intent: {missing_intent}/{len(ce_all)} 条（历史题集可能无此字段）"))
    if missing_fact > 0:
        flags.append(("info", f"缺少 target_fact: {missing_fact}/{len(ce_all)} 条"))

    # 实际返回数少于配置 TopK
    for rd in run_data_list:
        rs = rd.get("run_status", {})
        run = rd.get("run", {})
        qsid = rs.get("question_set_id") or run.get("question_set_id", "")
        _jr = rs.get("judge_results", [])
        short_return = sum(1 for r in _jr
                           if r.get("evaluation_track") == TRACK_CHUNK_EXACT
                           and "error" not in r
                           and len(r.get("retrieval_results", [])) < 5)
        if short_return > 0:
            flags.append(("warning", f"Run {_short_id(qsid)}: {short_return} 条样本实际返回结果少于 5 条"))

    return flags


def _render_quality_flags(flags):
    """渲染数据质量旗标。"""
    if not flags:
        return '<p class="section-note">未发现数据质量问题</p>'
    parts = ['<ul>']
    for level, msg in flags:
        css = ' class="warn"' if level == "warning" else ""
        icon = "⚠️" if level == "warning" else "ℹ️"
        parts.append(f'<li{css}>{icon} {_safe_str(msg)}</li>')
    parts.append('</ul>')
    return "".join(parts)


# ====== AI 分析包 ======

def build_ai_analysis_markdown(config, cumulative_metrics, chunk_exact_evaluable,
                               sample_lookup, layered, ranking_diag, top1_miss_records,
                               top1_miss_total, quality_flags, consistency_errors):
    """生成面向 GPT/LLM 上传的 Markdown 分析包。"""
    lines = []
    config_name = config.get("config_name", "")
    config_id = config.get("config_id", "")

    lines.append("# RAG 评测 AI 分析包")
    lines.append(f"\n配置方案: {config_name} ({config_id})")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 实验口径与配置
    lines.append("\n---\n## 1. 实验口径与配置\n")
    lines.append(f"- question_set_id: {', '.join(r.get('question_set_id', '') for r in chunk_exact_evaluable[:3]) or '未知'}")
    lines.append(f"- question_mode: chunk_exact")
    lines.append(f"- evaluation_type: chunk_exact")
    lines.append(f"- 可评测样本数: {len(chunk_exact_evaluable)}")
    ce_t1 = sum(r.get("retrieval_top1_hit", 0) for r in chunk_exact_evaluable)
    ce_t3 = sum(r.get("retrieval_top3_hit", 0) for r in chunk_exact_evaluable)
    ce_t5 = sum(r.get("retrieval_top5_hit", 0) for r in chunk_exact_evaluable)
    ce_t10 = sum(r.get("retrieval_top10_hit", 0) for r in chunk_exact_evaluable)
    n = len(chunk_exact_evaluable)
    lines.append(f"- Top1: {ce_t1}/{n} ({_pct(ce_t1/n)})")
    lines.append(f"- Top3: {ce_t3}/{n} ({_pct(ce_t3/n)})")
    lines.append(f"- Top5: {ce_t5}/{n} ({_pct(ce_t5/n)})")
    lines.append(f"- Top10: {ce_t10}/{n} ({_pct(ce_t10/n)})")

    if consistency_errors:
        lines.append("\n⚠️ 一致性校验错误:")
        for err in consistency_errors:
            lines.append(f"  - {err}")

    # 2. 分层指标
    lines.append("\n---\n## 2. 分层指标\n")
    lines.append("### 按 query_style")
    lines.append("| query_style | n | Top1 | Top3 | Top5 | Top10 |")
    lines.append("|---|---|---|---|---|---|")
    for style, g in sorted(layered["by_query_style"].items(), key=lambda x: x[1]["n"], reverse=True):
        sn = g["n"]
        obs = " (仅供观察)" if sn < 5 else ""
        lines.append(f"| {style}{obs} | {sn} | {g['t1']}/{sn} ({_pct(g['t1']/sn)}) | {g['t3']}/{sn} ({_pct(g['t3']/sn)}) | {g['t5']}/{sn} ({_pct(g['t5']/sn)}) | {g['t10']}/{sn} ({_pct(g['t10']/sn)}) |")

    lines.append("\n### 按 source document")
    lines.append("| 文档 | n | Top1 | Top3 | Top5 | Top10 |")
    lines.append("|---|---|---|---|---|---|")
    for doc, g in sorted(layered["by_doc"].items(), key=lambda x: x[1]["n"], reverse=True):
        dn = g["n"]
        obs = " (仅供观察)" if dn < 5 else ""
        lines.append(f"| {doc}{obs} | {dn} | {g['t1']}/{dn} ({_pct(g['t1']/dn)}) | {g['t3']}/{dn} ({_pct(g['t3']/dn)}) | {g['t5']}/{dn} ({_pct(g['t5']/dn)}) | {g['t10']}/{dn} ({_pct(g['t10']/dn)}) |")

    # 3. 排名诊断
    lines.append("\n---\n## 3. 排名诊断\n")
    lines.append("| 分桶 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    total = sum(ranking_diag.values()) or 1
    for bucket, label in [("top1", "Top1 命中"), ("top2_3", "第 2-3 位首次命中"),
                          ("top4_5", "第 4-5 位首次命中"), ("top6_10", "第 6-10 位首次命中"),
                          ("top10_miss", "Top10 未命中")]:
        c = ranking_diag[bucket]
        lines.append(f"| {label} | {c} | {_pct(c/total)} |")

    lines.append("\n**指标含义说明：**")
    lines.append("- TopK Hit = 严格命中同一 Dify segment_id / content_hash（机器判定，非语义匹配）")
    lines.append("- Top10 命中但 Top1 未中 → 候选已召回，排序或相近块区分待分析（不自动等于 rerank 故障）")
    lines.append("- Top10 未中 → 优先排查 query、embedding、chunk、候选召回或金标准")
    lines.append("- score 仅为 Dify 返回字段，rank 才是最终排序依据")
    lines.append("- 以上为诊断方向，不是确定因果")

    # 4. Top1 未中样本（完整对照）
    lines.append("\n---\n## 4. Top1 未中样本证据对照\n")
    # 元数据缺失统计
    meta_missing = sum(1 for r in top1_miss_records if r.get("missing_fields"))
    if meta_missing:
        lines.append(f"⚠️ 缺失元数据 {meta_missing} 条（题集 JSONL 中无 query_style/retrieval_intent 字段）\n")
    if top1_miss_records:
        lines.append(f"共 {top1_miss_total} 条 Top1 未命中，展示前 {len(top1_miss_records)} 条\n")
        for i, rec in enumerate(top1_miss_records[:20], 1):
            lines.append(f"### #{i} {rec['query'][:80]}")
            lines.append(f"- 分类: **{rec['category']}**")
            lines.append(f"- question_id: `{rec.get('question_id', '') or '缺失'}`")
            lines.append(f"- trace_id: `{rec.get('trace_id', '')[:16]}…`")
            if rec["query_style"]:
                lines.append(f"- query_style: {rec['query_style']}")
            else:
                lines.append(f"- query_style: 缺失")
            if rec["retrieval_intent"]:
                lines.append(f"- 用户检索意图: {rec['retrieval_intent']}")
            else:
                lines.append(f"- 用户检索意图: 缺失")
            if rec["target_fact"]:
                lines.append(f"- 证据锚点: {rec['target_fact']}")
            else:
                lines.append(f"- 证据锚点: 缺失")
            if rec["target_label"]:
                lines.append(f"- 标签: {rec['target_label']}")
            lines.append(f"- expected segment: `{rec['expected_segment_id'] or '缺失'}`")
            lines.append(f"- expected content_hash: `{(rec['expected_content_hash'] or '缺失')[:16]}`")
            lines.append(f"- 来源文档: {rec['expected_doc'] or '缺失'}"
                         + (f" (pos:{rec['expected_source_position']})" if rec.get('expected_source_position') else ""))
            lines.append(f"- 命中 rank: {'Top' + str(rec['hit_position']) if rec['hit_position'] else '未命中'}")
            if rec["expected_score"] is not None:
                lines.append(f"- 目标 score: {rec['expected_score']:.4f}")
                if rec["top1_score"] is not None:
                    diff = rec["top1_score"] - rec["expected_score"]
                    lines.append(f"- Top1 score: {rec['top1_score']:.4f} | 差值: {diff:+.4f}")
            lines.append(f"- 实际返回: {rec['actual_returned_count']} 条")
            # expected_content 摘要
            exp_content = rec.get("expected_content", "")
            if exp_content:
                lines.append(f"- 目标证据摘要（{len(exp_content)} 字）: {exp_content[:400]}{'...' if len(exp_content) > 400 else ''}")
            else:
                lines.append(f"- 目标证据: ⚠️ 缺失 expected_content")
            if rec.get("missing_fields"):
                lines.append(f"- ⚠️ 需复核: 元数据缺失 {', '.join(rec['missing_fields'])}")
            # 实际返回列表
            if rec["top_results"]:
                returned = rec["actual_returned_count"]
                short_note = ""
                if returned < 10:
                    short_note = f" ⚠️ 实际只返回 {returned} 条，Top10 为不足窗口口径"
                lines.append(f"- 实际返回 Top{returned}{short_note}:")
                for tr in rec["top_results"]:
                    score_str = f"{tr['score']:.4f}" if tr["score"] is not None else "N/A"
                    marker = " 🎯 目标chunk" if tr.get("is_expected") else ""
                    content_preview = tr.get("content", "")[:100]
                    lines.append(f"  - #{tr['rank']}: `{tr['segment_id']}` score={score_str} | {content_preview}{marker}")
                if rec.get("score_rank_mismatch"):
                    lines.append(f"  - ⚠️ Top2 score > Top1，score 与排序不一致")
            elif not rec.get("sample_found", True):
                lines.append(f"- ⚠️ 报告未找到该 run 的 processed retrieval evidence，无法展示返回列表（provenance/export error，非 Dify retrieval miss）")
            lines.append("")
    else:
        lines.append("无 Top1 未命中样本\n")

    # 5. Top10 未中样本
    top10_miss = [r for r in top1_miss_records if r["category"] == "Top10 未召回"]
    lines.append("\n---\n## 5. Top10 未中样本\n")
    if top10_miss:
        lines.append(f"共 {len(top10_miss)} 条 Top10 完全未召回\n")
        for i, rec in enumerate(top10_miss[:10], 1):
            lines.append(f"#{i} `{rec['query'][:50]}` | expected: `{_short_id(rec['expected_segment_id'])}` | doc: {rec['expected_doc']}")
    else:
        lines.append("无 Top10 完全未召回样本\n")

    # 6. 数据质量旗标
    lines.append("\n---\n## 6. 数据质量旗标\n")
    if quality_flags:
        for level, msg in quality_flags:
            icon = "⚠️" if level == "warning" else "ℹ️"
            lines.append(f"- {icon} {msg}")
    else:
        lines.append("未发现数据质量问题")

    # 7. 分析任务说明
    lines.append("\n---\n## 7. 分析任务说明\n")
    lines.append("请基于以上证据，区分以下诊断方向并提出实验建议：")
    lines.append("")
    lines.append("1. **query 质量** — retrieval_intent 是否合理、是否过于具体或过于宽泛")
    lines.append("2. **chunk 边界** — expected chunk 是否包含完整答案、是否因分块导致信息碎片化")
    lines.append("3. **candidate recall** — Top10 未命中说明目标 chunk 未被召回，检查 embedding 或混合检索配置")
    lines.append("4. **rerank 排序** — Top10 命中但 Top1 未中说明召回正确但排序靠后，检查 rerank 模型或相似候选区分")
    lines.append("5. **相似条款竞争** — 多个相似 chunk 竞争排名，检查 disambiguating query_style 的命中率")
    lines.append("")
    lines.append("**不要仅凭 Top1 数值下结论。** 请按影响优先级提出最多 5 条下一步实验建议。")
    lines.append("")
    lines.append("每条建议请包含：问题诊断、影响范围（样本数/占比）、建议操作、预期效果。")

    return "\n".join(lines)


# ====== HTML 报告 ======

def build_evaluation_html(config, config_runs, run_data_list, cumulative_metrics,
                          all_judge_results, export_scope="", sample_lookup=None,
                          provenance_info=None,
                          configured_top_k=None,
                          knowledge_base_total_chunks=None,
                          doc_chunk_counts=None):
    """生成自包含 HTML 评测报告。

    Args:
        configured_top_k: 配置 TopK（None 时从 config_snapshot 读取，默认 10）
        knowledge_base_total_chunks: 知识库总 chunk 数（None 时显示"未知"）
        doc_chunk_counts: dict[doc_name → chunk_count]（None 时显示"未知"）
    """
    # 补齐旧版 chunk_exact 结果缺失的 TopK 字段
    for r in all_judge_results:
        backfill_chunk_exact_topk(r, sample_lookup)

    # 解析 configured_top_k（优先使用传入参数，否则从 config_snapshot 读取）
    if configured_top_k is None:
        snapshot0 = (run_data_list[0].get("run", {}).get("config_snapshot") or {}) if run_data_list else {}
        configured_top_k = snapshot0.get("top_k", 10) or 10
    doc_chunk_counts = doc_chunk_counts or {}

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
    chunk_exact_all = [r for r in valid_results if r.get("evaluation_track") == TRACK_CHUNK_EXACT]
    chunk_exact_evaluable = [r for r in chunk_exact_all
                             if r.get("retrieval_evaluable", True) is not False
                             and r.get("retrieval_top1_hit") is not None]
    chunk_exact_unevaluable = [r for r in chunk_exact_all if r not in chunk_exact_evaluable]

    # 判断是否为纯 chunk_exact 报告
    is_pure_chunk_exact = (
        not retrieval_results
        and not strict_qa_results
        and not grounded_qa_results
        and bool(chunk_exact_evaluable)
    )

    # 检测跨题集
    all_qsid = {r.get("question_set_id", "") for r in valid_results if r.get("question_set_id")}
    is_cross_set = len(all_qsid) > 1

    # 召回规模统计（仅 chunk_exact 轨道）
    recall_stats = {}
    doc_level_recall = []
    if chunk_exact_evaluable:
        # 加载题集元数据（用于文档名解析）
        _qmeta_for_recall = load_question_set_metadata(all_qsid or None)
        recall_stats = _compute_recall_statistics(
            chunk_exact_evaluable, sample_lookup, configured_top_k,
            knowledge_base_total_chunks, doc_chunk_counts, _qmeta_for_recall)
        doc_level_recall = _compute_doc_level_recall_stats(
            chunk_exact_evaluable, sample_lookup, configured_top_k,
            doc_chunk_counts, _qmeta_for_recall)
    total_documents = len(doc_chunk_counts) if doc_chunk_counts else None

    # 诊断数据
    diag = build_diagnostic_data(all_judge_results, sample_lookup, config)
    ce_diag = build_chunk_exact_diagnostic_data(
        all_judge_results, sample_lookup, config,
        configured_top_k=configured_top_k,
        knowledge_base_total_chunks=knowledge_base_total_chunks,
        doc_chunk_counts=doc_chunk_counts,
        question_meta_lookup=_qmeta_for_recall if chunk_exact_evaluable else None)

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
        "chunk_exact": len(chunk_exact_all),
        "chunk_exact_evaluable": len(chunk_exact_evaluable),
    }
    no_retrieval_results_count = sum(
        1 for d in diag["top5_miss"] + diag["sorting_issues"]
        if d.get("diagnostic_status") == "ok" and not d.get("retrieval_results")
    )

    # 报告标题
    if is_pure_chunk_exact:
        report_title = "RAG 评测报告 — Chunk Exact 机器判定（segment_id / content_hash）"
    else:
        report_title = "RAG 评测报告"

    # 构建 HTML
    html_parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{report_title} - {config_name}</title>
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
  tr.hit {{ background: #d4edda !important; }}
  tr.hit td {{ border-color: #28a745; }}
  .warn {{ color: #ffc107; }}
  .info-box {{ background: #e7f3ff; border: 1px solid #b3d7ff; border-radius: 6px;
               padding: 12px 16px; margin: 12px 0; font-size: 0.9em; }}
  .warn-box {{ background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px;
               padding: 12px 16px; margin: 12px 0; font-size: 0.9em; }}
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
<h1>{report_title}</h1>
<div class="meta">
  <p><strong>配置方案</strong>: {config_name} (<code>{config_id}</code>)</p>
  <p><strong>知识库版本</strong>: {kb_version or '未指定'}</p>
  <p><strong>工作流版本</strong>: {wf_version or '未指定'}</p>
  <p><strong>生成时间</strong>: {now}</p>
  <p><strong>导出范围</strong>: {_safe_str(export_scope) or '当前配置全部运行'}</p>
  <p><strong>数据口径</strong>: 通过 config_id → run_id → processed sample → 真实 Langfuse trace_id → judged result 关联；
     检索命中仅对 evaluation_track=retrieval 的可评测样本计算 Top1/Top3/Top5；
     chunk_exact 按 segment_id/content_hash 纯机器判定，单独统计；
     QA 轨道单独统计，不与检索命中率混合；
     missing_binding / no_trace / no_retrieval 不计入 chunk_exact TopK 分母。</p>
</div>
"""]

    # 跨题集警告
    if is_cross_set:
        html_parts.append(
            '<div class="warn-box">⚠️ <strong>跨题集探索汇总</strong>：本报告聚合了多个 question_set_id 的结果，'
            '不作为单一固定题集的横向模型比较基线。</div>'
        )

    # 纯 chunk_exact 说明
    if is_pure_chunk_exact:
        html_parts.append(
            '<div class="info-box">本报告不含 AI 证据 Judge（retrieval）样本；'
            '以下为 <strong>Chunk Exact 机器判定</strong>结果。</div>'
        )

    # 1. 总览指标
    ce_n = len(chunk_exact_evaluable)
    ce_t1 = sum(r.get("retrieval_top1_hit", 0) for r in chunk_exact_evaluable)
    ce_t3 = sum(r.get("retrieval_top3_hit", 0) for r in chunk_exact_evaluable)
    ce_t5 = sum(r.get("retrieval_top5_hit", 0) for r in chunk_exact_evaluable)

    html_parts.append(f"""
<h2>1. 总览</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="value">{len(config_runs)}</div><div class="label">运行次数</div></div>
  <div class="metric-card"><div class="value">{total_questions}</div><div class="label">题目总数</div></div>
  <div class="metric-card"><div class="value">{total_batch_success}/{total_batch_total}</div><div class="label">Batch 成功</div></div>
  <div class="metric-card"><div class="value">{total_processed}</div><div class="label">Processed</div></div>
  <div class="metric-card"><div class="value">{total_judge}</div><div class="label">AI Judge 已评测</div></div>
  <div class="metric-card"><div class="value">{track_counts['chunk_exact']}</div><div class="label">机器判定完成</div></div>
  <div class="metric-card"><div class="value">{len(error_results)}</div><div class="label">Judge 错误</div></div>
</div>
<table>
<tr><th>轨道</th><th>样本数</th></tr>
<tr><td>retrieval（可评测）</td><td>{track_counts['retrieval_evaluable']}</td></tr>
<tr><td>chunk_exact（精确匹配）</td><td>{track_counts['chunk_exact']}</td></tr>
<tr><td>strict_qa</td><td>{track_counts['strict_qa']}</td></tr>
<tr><td>grounded_qa</td><td>{track_counts['grounded_qa']}</td></tr>
<tr><td>不可评测</td><td>{track_counts['not_evaluable']}</td></tr>
<tr><td>Judge 错误</td><td>{len(error_results)}</td></tr>
<tr><td>无检索结果</td><td>{no_retrieval_results_count}</td></tr>
</table>
""")

    # 题集元数据查找表（用于分层指标、证据对照回填 query_style、document_name 等）
    question_meta_lookup = {}

    # chunk_exact 摘要卡片
    if chunk_exact_evaluable:
        ce_t10 = sum(r.get("retrieval_top10_hit", 0) for r in chunk_exact_evaluable)
        # 窗口状态信息
        _fw = recall_stats.get("full_window_count", ce_n)
        _pw = recall_stats.get("partial_window_count", 0)
        _window_note = ""
        if _pw > 0:
            _window_note = f'<br><span class="warn">⚠️ 其中 {_pw} 题实际返回 &lt; {configured_top_k} 条，按实际窗口计算</span>'

        html_parts.append(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="value">{ce_n}/{track_counts['chunk_exact']}</div><div class="label">chunk_exact 可评测</div></div>
  <div class="metric-card"><div class="value">{ce_t1}/{ce_n} ({_pct(ce_t1/ce_n)})</div><div class="label">Top1</div></div>
  <div class="metric-card"><div class="value">{ce_t3}/{ce_n} ({_pct(ce_t3/ce_n)})</div><div class="label">Top3</div></div>
  <div class="metric-card"><div class="value">{ce_t5}/{ce_n} ({_pct(ce_t5/ce_n)})</div><div class="label">Top5</div></div>
  <div class="metric-card"><div class="value">{ce_t10}/{ce_n} ({_pct(ce_t10/ce_n)})</div><div class="label">Top10 (配置={configured_top_k})</div></div>
</div>
<div class="info-box">
  <strong>指标含义说明：</strong><br>
  • <strong>TopK Hit</strong> = 严格命中同一 Dify segment_id / content_hash（机器判定，非语义匹配）<br>
  • <strong>Top10 命中但 Top1 未命中</strong> → 候选已召回，排序或相近块区分待分析（不自动等于 rerank 故障）<br>
  • <strong>Top10 未命中</strong> → 优先排查 query、embedding、chunk、候选召回或金标准<br>
  • <strong>score</strong> 仅为 Dify 返回字段，<strong>rank</strong> 才是最终排序依据<br>
  • full_window: <strong>{_fw}</strong> 题 | partial_window: <strong>{_pw}</strong> 题{_window_note}
</div>
""")
        if chunk_exact_unevaluable:
            ue_counts = {}
            for r in chunk_exact_unevaluable:
                s = r.get("chunk_exact_status") or "未知"
                ue_counts[s] = ue_counts.get(s, 0) + 1
            ue_parts = [f"{s}: {c}" for s, c in ue_counts.items()]
            html_parts.append(
                f'<p class="section-note">不可评测 {len(chunk_exact_unevaluable)} 条 — {"、".join(ue_parts)}</p>'
            )

    # 一致性校验
    consistency_errors = validate_report_consistency(all_judge_results, run_data_list, cumulative_metrics)
    if consistency_errors:
        err_items = "".join(f"<li>❌ {_safe_str(e)}</li>" for e in consistency_errors)
        html_parts.append(
            f'<div class="warn-box"><strong>⚠️ 一致性校验失败</strong><ul>{err_items}</ul></div>'
        )

    # 2. 配置与运行信息
    html_parts.append("<h2>2. 配置与运行信息</h2>")
    # 共享配置显示一次
    if run_data_list:
        snapshot0 = run_data_list[0].get("run", {}).get("config_snapshot") or {}
        if snapshot0:
            html_parts.append('<details><summary>共享配置快照（config_snapshot）</summary>')
            html_parts.append(_render_config_snapshot_table(snapshot0))
            html_parts.append('</details>')

    # 各 run 简表
    html_parts.append(
        '<table><tr><th>运行</th><th>题集</th><th>题集 ID</th><th>题目数</th>'
        '<th>状态</th><th>Batch</th><th>开始时间</th></tr>'
    )
    for rd in run_data_list:
        run = rd["run"]
        rs = rd["run_status"]
        rid = _safe_str(run.get("run_id", ""))
        rid_short = _short_id(rid)
        qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
        qsid = _safe_str(rs.get("question_set_id") or run.get("question_set_id", "") or "")
        qsid_short = _short_id(qsid)
        qc = run.get("question_count", 0)
        status = _safe_str(run.get("status", ""))
        bs = rs.get("batch_success", 0)
        bt = rs.get("batch_total", 0)
        started = _safe_str(run.get("started_at", ""))[:19]
        html_parts.append(
            f'<tr><td title="{rid}"><code>{rid_short}</code></td>'
            f'<td>{qs}</td><td title="{qsid}"><code>{qsid_short}</code></td>'
            f'<td>{qc}</td><td>{status}</td><td>{bs}/{bt}</td><td>{started}</td></tr>'
        )
    html_parts.append('</table>')

    # 2.5 召回规模概览
    if chunk_exact_evaluable:
        html_parts.append("<h2>2.5 召回规模概览</h2>")
        html_parts.append(_render_recall_overview_section(
            configured_top_k, knowledge_base_total_chunks,
            total_documents, doc_chunk_counts, recall_stats))

    # 3. 全局 Judge 指标
    html_parts.append("<h2>3. 全局 Judge 指标</h2>")

    # retrieval 指标
    if retrieval_results:
        n = len(retrieval_results)
        t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval_results)
        t3 = sum(r.get("retrieval_top3_hit", 0) for r in retrieval_results)
        t5 = sum(r.get("retrieval_top5_hit", 0) for r in retrieval_results)
        html_parts.append(f"""
<h3>检索评测（AI Judge）</h3>
<p class="section-note">仅 evaluation_track=retrieval 的可评测样本</p>
<table>
<tr><th>指标</th><th>命中数</th><th>样本数</th><th>命中率</th></tr>
<tr><td>Top1 Hit</td><td>{t1}</td><td>{n}</td><td>{_pct(t1/n)}</td></tr>
<tr><td>Top3 Hit</td><td>{t3}</td><td>{n}</td><td>{_pct(t3/n)}</td></tr>
<tr><td>Top5 Hit</td><td>{t5}</td><td>{n}</td><td>{_pct(t5/n)}</td></tr>
</table>
""")
    elif not is_pure_chunk_exact:
        html_parts.append('<p class="section-note">本报告不含 AI 证据 Judge（retrieval）样本</p>')

    if strict_qa_results:
        n = len(strict_qa_results)
        acc = sum(r.get("answer_correct", 0) for r in strict_qa_results)
        html_parts.append(f'<p>严格问答: 正确 {acc} / 总 {n} = {_pct(acc/n)}</p>')
    if grounded_qa_results:
        n = len(grounded_qa_results)
        gnd = sum(r.get("answer_correct", 0) for r in grounded_qa_results)
        html_parts.append(f'<p>合理性问答: 有据 {gnd} / 总 {n} = {_pct(gnd/n)}</p>')

    # chunk_exact 指标
    if chunk_exact_evaluable:
        n = len(chunk_exact_evaluable)
        t1 = sum(r.get("retrieval_top1_hit", 0) for r in chunk_exact_evaluable)
        t3 = sum(r.get("retrieval_top3_hit", 0) for r in chunk_exact_evaluable)
        t5 = sum(r.get("retrieval_top5_hit", 0) for r in chunk_exact_evaluable)
        t10 = sum(r.get("retrieval_top10_hit", 0) for r in chunk_exact_evaluable)
        html_parts.append(f"""
<h3>Chunk Exact（机器判定）</h3>
<p class="section-note">按 segment_id / content_hash 纯机器判定，分母仅限可评测样本 (n={n})</p>
<table>
<tr><th>指标</th><th>命中数</th><th>样本数</th><th>命中率</th></tr>
<tr><td>Top1 Hit</td><td>{t1}</td><td>{n}</td><td>{_pct(t1/n)}</td></tr>
<tr><td>Top3 Hit</td><td>{t3}</td><td>{n}</td><td>{_pct(t3/n)}</td></tr>
<tr><td>Top5 Hit</td><td>{t5}</td><td>{n}</td><td>{_pct(t5/n)}</td></tr>
<tr><td>Top10 Hit</td><td>{t10}</td><td>{n}</td><td>{_pct(t10/n)}</td></tr>
</table>
""")

        # 命中位置分布（互斥分桶，扩展至 Top10）
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

        html_parts.append(f"""
<h3>命中位置分布</h3>
<table>
<tr><th>分桶</th><th>数量</th><th>占比</th></tr>
<tr><td>Top1 命中</td><td>{bucket_top1}</td><td>{_pct(bucket_top1/n)}</td></tr>
<tr><td>第 2-3 位首次命中</td><td>{bucket_2_3}</td><td>{_pct(bucket_2_3/n)}</td></tr>
<tr><td>第 4-5 位首次命中</td><td>{bucket_4_5}</td><td>{_pct(bucket_4_5/n)}</td></tr>
<tr><td>第 6-10 位首次命中</td><td>{bucket_6_10}</td><td>{_pct(bucket_6_10/n)}</td></tr>
<tr><td>Top10 未命中</td><td>{bucket_miss}</td><td>{_pct(bucket_miss/n)}</td></tr>
</table>
<p class="section-note">分母仅限 chunk_exact 可评测样本 (n={n})，分桶互斥</p>
""")
    elif not chunk_exact_all:
        if not is_pure_chunk_exact:
            html_parts.append('<p class="section-note">暂无 chunk_exact 评测数据</p>')

    # 3.5 局部分析（仅当有 retrieval/QA 数据时展示）
    if retrieval_results or strict_qa_results or grounded_qa_results:
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

    # 5. 运行汇总表（按题集分组）
    # retrieval 汇总
    if retrieval_results or strict_qa_results or grounded_qa_results:
        html_parts.append("""
<h2>5. 运行汇总</h2>
<h3>AI Judge 轨道</h3>
<table>
<tr><th>题集</th><th>题集 ID</th><th>题目数</th><th>状态</th><th>Batch</th><th>Top1</th><th>Top3</th><th>Top5</th><th>Top5未命中</th><th>排序问题</th><th>错误数</th></tr>
""")
        for rd in run_data_list:
            run = rd["run"]
            rs = rd["run_status"]
            m = rd.get("metrics") or {}
            qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
            qsid = _safe_str(rs.get("question_set_id") or run.get("question_set_id", "") or "")
            qsid_short = _short_id(qsid)
            qc = run.get("question_count", 0)
            status = _safe_str(run.get("status", ""))
            bs = rs.get("batch_success", 0)
            bt = rs.get("batch_total", 0)
            t1 = _pct(m.get("retrieval_top1_hit_rate"))
            t3 = _pct(m.get("retrieval_top3_hit_rate"))
            t5 = _pct(m.get("retrieval_top5_hit_rate"))
            errs = m.get("errors", 0)
            _run_jr = rs.get("judge_results", [])
            _run_valid = [r for r in _run_jr if "error" not in r]
            _run_ret = [r for r in _run_valid
                        if r.get("evaluation_track") == TRACK_RETRIEVAL
                        and r.get("retrieval_evaluable", True)]
            miss5 = sum(1 for r in _run_ret if r.get("retrieval_top5_hit", 0) == 0)
            sort_issues = sum(1 for r in _run_ret
                              if r.get("retrieval_top1_hit", 0) == 0 and r.get("retrieval_top5_hit", 0) == 1)
            html_parts.append(
                f'<tr><td>{qs}</td><td title="{qsid}"><code>{qsid_short}</code></td>'
                f'<td>{qc}</td><td>{status}</td><td>{bs}/{bt}</td>'
                f'<td>{t1}</td><td>{t3}</td><td>{t5}</td>'
                f'<td>{miss5}</td><td>{sort_issues}</td><td>{errs}</td></tr>'
            )
        html_parts.append("</table>")

    # chunk_exact 汇总
    if chunk_exact_evaluable:
        html_parts.append("""
<h3>Chunk Exact 轨道</h3>
<table>
<tr><th>题集</th><th>题集 ID</th><th>可评测</th><th>Top1</th><th>Top3</th><th>Top5</th><th>Top10</th><th>Top10未命中</th><th>排序问题</th></tr>
""")
        # 按 question_set_id 分组
        ce_by_qsid = {}
        for r in chunk_exact_evaluable:
            qsid = r.get("question_set_id") or "未知"
            ce_by_qsid.setdefault(qsid, []).append(r)

        for rd in run_data_list:
            run = rd["run"]
            rs = rd["run_status"]
            qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
            qsid = rs.get("question_set_id") or run.get("question_set_id", "") or ""
            qsid_short = _short_id(qsid)

            # 直接从该 run 的 judge_results 计算 chunk_exact 指标
            # （不依赖 per-run metrics dict 中可能缺失的 count 字段）
            _run_jr = rs.get("judge_results", [])
            _run_ce = [r for r in _run_jr if "error" not in r
                       and r.get("evaluation_track") == TRACK_CHUNK_EXACT]
            ce_count = len(_run_ce)
            if ce_count == 0:
                continue

            _run_ce_eval = [r for r in _run_ce
                            if r.get("retrieval_evaluable", True) is not False
                            and r.get("retrieval_top1_hit") is not None]
            ce_eval_n = len(_run_ce_eval)
            ce_t1 = sum(r.get("retrieval_top1_hit", 0) for r in _run_ce_eval)
            ce_t3 = sum(r.get("retrieval_top3_hit", 0) for r in _run_ce_eval)
            ce_t5 = sum(r.get("retrieval_top5_hit", 0) for r in _run_ce_eval)
            ce_t10 = sum(r.get("retrieval_top10_hit", 0) for r in _run_ce_eval)
            ce_miss10 = sum(1 for r in _run_ce_eval
                            if r.get("retrieval_top10_hit", 0) == 0)
            ce_sort = sum(1 for r in _run_ce_eval
                          if r.get("retrieval_top1_hit", 0) == 0
                          and r.get("retrieval_top5_hit", 0) == 1)

            html_parts.append(
                f'<tr><td>{qs}</td><td title="{qsid}"><code>{qsid_short}</code></td>'
                f'<td>{ce_eval_n}/{ce_count}</td>'
                f'<td>{ce_t1}/{ce_eval_n} ({_pct(ce_t1/ce_eval_n) if ce_eval_n else "N/A"})</td>'
                f'<td>{ce_t3}/{ce_eval_n} ({_pct(ce_t3/ce_eval_n) if ce_eval_n else "N/A"})</td>'
                f'<td>{ce_t5}/{ce_eval_n} ({_pct(ce_t5/ce_eval_n) if ce_eval_n else "N/A"})</td>'
                f'<td>{ce_t10}/{ce_eval_n} ({_pct(ce_t10/ce_eval_n) if ce_eval_n else "N/A"})</td>'
                f'<td>{ce_miss10}</td><td>{ce_sort}</td></tr>'
            )
        html_parts.append("</table>")

    # ── 5.5 分析诊断（chunk_exact 专属） ──
    if chunk_exact_evaluable:
        html_parts.append("<h2>5.5 分析诊断</h2>")

        # 实验可比性声明
        all_qsid_set = {r.get("question_set_id", "") for r in chunk_exact_evaluable if r.get("question_set_id")}
        all_modes = {r.get("question_mode", "") for r in chunk_exact_evaluable}
        all_eval_types = {r.get("evaluation_type", "") for r in chunk_exact_evaluable}
        html_parts.append('<div class="info-box">')
        html_parts.append('<strong>实验可比性声明</strong><br>')
        html_parts.append(f'question_set_id: {", ".join(sorted(all_qsid_set)) or "未知"}<br>')
        html_parts.append(f'question_mode: {", ".join(sorted(all_modes)) or "未知"}<br>')
        html_parts.append(f'evaluation_type: {", ".join(sorted(all_eval_types)) or "未知"}<br>')
        html_parts.append(f'可直接比较的 run: 本配置方案下 {len(config_runs)} 次运行<br>')
        html_parts.append('<span class="warn">⚠️ 不同题集、不同 Judge 轨道、不同知识库 snapshot 不得直接比较</span>')
        html_parts.append('</div>')

        # 加载题集元数据（用于分层指标和证据对照回填 query_style、document_name 等）
        all_qsid = {r.get("question_set_id", "") for r in chunk_exact_evaluable if r.get("question_set_id")}
        question_meta_lookup = load_question_set_metadata(all_qsid or None)

        # 分层指标
        layered = _build_layered_metrics(chunk_exact_evaluable, sample_lookup, question_meta_lookup)
        html_parts.append("<h3>按 query_style 分层</h3>")
        html_parts.append(_render_layered_table(layered["by_query_style"], "query_style"))
        html_parts.append("<h3>按 source document 分层</h3>")
        html_parts.append(_render_layered_table(layered["by_doc"], "文档"))

        # 文档级召回统计
        if doc_level_recall:
            html_parts.append("<h3>文档级召回统计</h3>")
            html_parts.append(_render_doc_level_recall_table(doc_level_recall, configured_top_k))

        # 排名诊断
        ranking_diag = _build_ranking_diagnostics(chunk_exact_evaluable)
        rn = sum(ranking_diag.values()) or 1
        html_parts.append("<h3>排名诊断（互斥分布）</h3>")
        html_parts.append('<table><tr><th>分桶</th><th>数量</th><th>占比</th></tr>')
        for bucket, label in [("top1", "Top1 命中"), ("top2_3", "第 2-3 位首次命中"),
                              ("top4_5", "第 4-5 位首次命中"), ("top6_10", "第 6-10 位首次命中"),
                              ("top10_miss", "Top10 未命中")]:
            c = ranking_diag[bucket]
            html_parts.append(f'<tr><td>{label}</td><td>{c}</td><td>{_pct(c/rn)}</td></tr>')
        html_parts.append('</table>')
        html_parts.append(
            '<div class="info-box"><strong>诊断方向说明：</strong><br>'
            '• Top10 命中但 Top1 未中 → 排序/rerank 或相似候选区分问题<br>'
            '• Top10 未中 → 候选召回、query、chunk 或 embedding 问题<br>'
            '• 以上为诊断方向，不是确定因果</div>'
        )

        # Top1 未中样本证据对照
        meta_hit = sum(1 for r in chunk_exact_evaluable
                       if _lookup_question_meta(r, question_meta_lookup).get("query_style"))
        meta_miss = len(chunk_exact_evaluable) - meta_hit

        top1_miss_records = _build_top1_miss_evidence(
            chunk_exact_evaluable, sample_lookup, question_meta_lookup)
        top1_miss_total = sum(1 for r in chunk_exact_evaluable if r.get("retrieval_top1_hit", 0) == 0)
        html_parts.append("<h3>Top1 未中样本证据对照</h3>")
        if meta_miss > 0:
            html_parts.append(
                f'<p class="section-note">ℹ️ 题集元数据回填: {meta_hit}/{len(chunk_exact_evaluable)} 条成功'
                f'，{meta_miss} 条缺失元数据（历史题集可能无 query_style/retrieval_intent 字段）</p>'
            )
        html_parts.append(_render_top1_miss_evidence(top1_miss_records, top1_miss_total))

    # 6. 运行详情
    html_parts.append("<h2>6. 运行详情</h2>")
    for rd in run_data_list:
        run = rd["run"]
        rs = rd["run_status"]
        m = rd.get("metrics") or {}
        rid = _safe_str(run.get("run_id", ""))
        rid_short = _short_id(rid)
        qs = _safe_str(rs.get("question_set_name") or run.get("question_set_name", "") or "旧版题集")
        started = _safe_str(run.get("started_at", ""))[:19]
        status = _safe_str(run.get("status", ""))

        html_parts.append(f'<h3 title="{rid}">{qs} — <code>{rid_short}</code></h3>')
        html_parts.append(f'<p class="section-note">状态: {status} | 开始: {started}</p>')

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

            # chunk_exact 详情
            ce_jr = [r for r in valid_jr if r.get("evaluation_track") == TRACK_CHUNK_EXACT]
            if ce_jr:
                ce_evaluable_jr = [r for r in ce_jr
                                   if r.get("retrieval_evaluable", True) is not False
                                   and r.get("retrieval_top1_hit") is not None]
                ce_unevaluable_jr = [r for r in ce_jr if r not in ce_evaluable_jr]
                n = len(ce_evaluable_jr)
                total_ce = len(ce_jr)
                if n > 0:
                    t1 = sum(r.get("retrieval_top1_hit", 0) for r in ce_evaluable_jr)
                    t3 = sum(r.get("retrieval_top3_hit", 0) for r in ce_evaluable_jr)
                    t5 = sum(r.get("retrieval_top5_hit", 0) for r in ce_evaluable_jr)
                    html_parts.append(
                        f'<p><strong>Chunk Exact（机器判定）</strong> '
                        f'可评测 {n}/{total_ce} | '
                        f'Top1 {t1}/{n} ({_pct(t1/n)}) | '
                        f'Top3 {t3}/{n} ({_pct(t3/n)}) | '
                        f'Top5 {t5}/{n} ({_pct(t5/n)})</p>'
                    )
                if ce_unevaluable_jr:
                    ue_counts = {}
                    for r in ce_unevaluable_jr:
                        s = r.get("chunk_exact_status") or "未知"
                        ue_counts[s] = ue_counts.get(s, 0) + 1
                    ue_parts = [f"{s}: {c}" for s, c in ue_counts.items()]
                    html_parts.append(f'<p class="section-note">不可评测 {len(ce_unevaluable_jr)} 条 — {"、".join(ue_parts)}</p>')

            err_jr = [r for r in jr if "error" in r]
            if err_jr:
                html_parts.append(f'<p class="miss">Judge 错误: {len(err_jr)} 条</p>')

    # 7. Chunk Exact 诊断（Top10 未命中 / 排序偏后 / 排序问题）
    if chunk_exact_evaluable:
        html_parts.append("<h2>7. Chunk Exact 诊断</h2>")

        html_parts.append("<h3>Top10 未命中（完全未召回）</h3>")
        _render_chunk_exact_diagnostic_cards(
            html_parts, ce_diag["top10_miss"], ce_diag["total_top10_miss"],
            "Top10 未命中",
        )

        html_parts.append("<h3>排序偏后（Top5 未命中但 Top10 命中，Top6-10）</h3>")
        _render_chunk_exact_diagnostic_cards(
            html_parts, ce_diag["top5_miss_but_top10_hit"], ce_diag["total_top5_miss_but_top10_hit"],
            "排序偏后",
        )

        html_parts.append("<h3>排序问题（Top1 未命中但 Top3/Top5 命中）</h3>")
        _render_chunk_exact_diagnostic_cards(
            html_parts, ce_diag["sorting_issues"], ce_diag["total_sorting_issues"],
            "排序问题",
        )

    # 8. AI Judge 诊断（retrieval 轨道）
    if retrieval_results:
        html_parts.append("<h2>8. AI Judge 诊断</h2>")

        html_parts.append("<h3>Top5 完全未命中</h3>")
        _render_diagnostic_cards(html_parts, diag["top5_miss"], diag["total_top5_miss"],
                                 "Top5 未命中（检索结果均未命中金标准）", show_details=True)

        html_parts.append("<h3>排序问题（Top1 未命中但 Top3/Top5 命中）</h3>")
        _render_diagnostic_cards(html_parts, diag["sorting_issues"], diag["total_sorting_issues"],
                                 "排序问题（Top1 未命中但更高排名命中，说明相关内容被排到较低位置）",
                                 show_details=True)

    # 9. 数据质量与质量旗标
    quality_flags = _build_quality_flags(all_judge_results, sample_lookup, run_data_list)

    # 计算 processed sample 关联统计
    ce_with_sample = sum(1 for r in chunk_exact_evaluable if sample_lookup.get(r.get("trace_id", "")))
    ce_without_sample = len(chunk_exact_evaluable) - ce_with_sample

    html_parts.append("<h2>9. 数据质量与可审计信息</h2><ul>")
    html_parts.append(f'<li>Judge 错误结果: <strong>{len(error_results)}</strong> 条</li>')
    html_parts.append(f'<li>不可评测样本（缺少金标准证据）: <strong>{len(not_evaluable_results)}</strong> 条</li>')
    html_parts.append(f'<li>无检索结果的样本: <strong>{no_retrieval_results_count}</strong> 条</li>')

    # processed sample 关联统计
    html_parts.append(f'<li>Chunk Exact 成功关联 processed sample: <strong>{ce_with_sample}/{len(chunk_exact_evaluable)}</strong></li>')
    if ce_without_sample > 0:
        html_parts.append(f'<li class="warn">⚠️ 未找到 processed sample 的 chunk_exact 样本: <strong>{ce_without_sample}</strong> 条（provenance 关联失败，非 Dify retrieval miss）</li>')

    # processed source 摘要
    if provenance_info:
        source_paths = provenance_info.get("source_paths", {})
        if source_paths:
            html_parts.append('<li>Processed 来源文件:')
            html_parts.append('<ul>')
            for path_str, count in source_paths.items():
                # 只显示文件名，不泄露完整路径
                fname = Path(path_str).name if path_str else "未知"
                html_parts.append(f'<li><code>{_safe_str(fname)}</code> ({count} samples)</li>')
            html_parts.append('</ul></li>')
        if provenance_info.get("fallback_count", 0) > 0:
            html_parts.append(f'<li class="warn">⚠️ {provenance_info["fallback_count"]} 个 run 未找到 processed 文件（历史 fallback）</li>')

    # chunk_exact 不可评测状态
    if chunk_exact_unevaluable:
        ce_ue_counts = {}
        for r in chunk_exact_unevaluable:
            s = r.get("chunk_exact_status") or "未知"
            ce_ue_counts[s] = ce_ue_counts.get(s, 0) + 1
        for status, count in ce_ue_counts.items():
            html_parts.append(f'<li>chunk_exact 不可评测 ({status}): <strong>{count}</strong> 条</li>')

    no_trace = [r for r in all_judge_results if not r.get("trace_id")]
    if no_trace:
        html_parts.append(f'<li class="warn">缺少 trace_id 的结果: <strong>{len(no_trace)}</strong> 条</li>')

    # 质量旗标
    if quality_flags:
        for level, msg in quality_flags:
            css = ' class="warn"' if level == "warning" else ""
            icon = "⚠️" if level == "warning" else "ℹ️"
            html_parts.append(f'<li{css}>{icon} {_safe_str(msg)}</li>')

    # 一致性校验结果
    if consistency_errors:
        for err in consistency_errors:
            html_parts.append(f'<li class="warn">❌ {_safe_str(err)}</li>')

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

    # 附录：Chunk Exact 样本明细
    if chunk_exact_all:
        _render_chunk_exact_sample_appendix(html_parts, all_judge_results, sample_lookup)

    # AI 分析包下载（嵌入 HTML 中，通过 JS 触发下载）
    if chunk_exact_evaluable:
        _layered = _build_layered_metrics(chunk_exact_evaluable, sample_lookup, question_meta_lookup)
        _ranking_diag = _build_ranking_diagnostics(chunk_exact_evaluable)
        _top1_miss_records = _build_top1_miss_evidence(
            chunk_exact_evaluable, sample_lookup, question_meta_lookup)
        _top1_miss_total = sum(1 for r in chunk_exact_evaluable if r.get("retrieval_top1_hit", 0) == 0)
        _quality_flags = _build_quality_flags(all_judge_results, sample_lookup, run_data_list)
        ai_md = build_ai_analysis_markdown(
            config, cumulative_metrics, chunk_exact_evaluable,
            sample_lookup, _layered, _ranking_diag,
            _top1_miss_records, _top1_miss_total,
            _quality_flags, consistency_errors,
        )
        import base64
        ai_md_b64 = base64.b64encode(ai_md.encode("utf-8")).decode("ascii")
        html_parts.append(f"""
<div class="no-print" style="margin-top:30px; padding:16px; background:#f0f7ff; border:1px solid #b3d7ff; border-radius:8px;">
  <strong>📥 下载 AI 分析包</strong>
  <p style="font-size:0.9em; color:#555;">Markdown 格式，面向 GPT/LLM 上传分析。包含实验口径、分层指标、排名诊断、失败样本对照和分析任务说明。</p>
  <button onclick="downloadAIAnalysis()" style="padding:8px 16px; background:#1a73e8; color:#fff; border:none; border-radius:4px; cursor:pointer;">
    下载 AI 分析包（Markdown）
  </button>
</div>
<script>
function downloadAIAnalysis() {{
  var b64 = "{ai_md_b64}";
  var bytes = atob(b64);
  var blob = new Blob([bytes], {{type: "text/markdown;charset=utf-8"}});
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = "ai_analysis_{_safe_str(config_name)[:20]}.md";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
""")

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
        "chunk_exact_count", "chunk_exact_top1_hit_rate", "chunk_exact_top3_hit_rate", "chunk_exact_top5_hit_rate", "chunk_exact_top10_hit_rate",
        "top10_miss_count", "sorting_issue_count",
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

        # chunk_exact 未命中统计
        _run_ce = [r for r in _run_valid
                   if r.get("evaluation_track") == TRACK_CHUNK_EXACT]
        ce_miss10 = sum(1 for r in _run_ce if r.get("retrieval_top10_hit", 0) == 0)

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
            m.get("chunk_exact_track_count", 0),
            m.get("chunk_exact_top1_hit_rate"),
            m.get("chunk_exact_top3_hit_rate"),
            m.get("chunk_exact_top5_hit_rate"),
            m.get("chunk_exact_top10_hit_rate"),
            ce_miss10,
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


def build_chunk_exact_csv(all_judge_results, sample_lookup=None):
    """生成 chunk_exact 明细 CSV。

    包含每条 chunk_exact 结果的完整信息：绑定、命中、返回结果。
    """
    sample_lookup = sample_lookup or {}
    # 补齐旧版 chunk_exact 结果缺失的 TopK 字段
    for r in all_judge_results:
        backfill_chunk_exact_topk(r, sample_lookup)

    valid = [r for r in all_judge_results
             if "error" not in r and r.get("evaluation_track") == TRACK_CHUNK_EXACT]

    output = io.StringIO()
    writer = csv.writer(output)

    header = [
        "run_id", "question_set_id", "question_id", "query", "evaluation_track",
        "expected_segment_id", "expected_content_hash", "chunk_exact_status",
        "hit_evidence_position", "top1_hit", "top3_hit", "top5_hit", "top10_hit",
        "returned_segment_ids", "retrieval_scores",
    ]
    writer.writerow(header)

    for r in valid:
        tid = r.get("trace_id", "")
        sample = sample_lookup.get(tid, {})
        ret_results = sample.get("retrieval_results", [])[:10] if sample else []

        returned_sids = "; ".join(
            str(rr.get("segment_id", rr.get("document_name", "")))
            for rr in ret_results
        )
        returned_scores = "; ".join(
            f"{rr.get('score', '')}" if isinstance(rr.get("score"), (int, float)) else str(rr.get("score", ""))
            for rr in ret_results
        )

        writer.writerow([
            r.get("run_id", ""),
            r.get("question_set_id", ""),
            r.get("question_id", ""),
            r.get("question", ""),
            r.get("evaluation_track", ""),
            r.get("expected_segment_id", ""),
            r.get("expected_content_hash", ""),
            r.get("chunk_exact_status", ""),
            r.get("hit_evidence_position", "") if r.get("hit_evidence_position") is not None else "",
            r.get("retrieval_top1_hit", ""),
            r.get("retrieval_top3_hit", ""),
            r.get("retrieval_top5_hit", ""),
            r.get("retrieval_top10_hit", ""),
            returned_sids,
            returned_scores,
        ])

    return output.getvalue().encode("utf-8-sig")
