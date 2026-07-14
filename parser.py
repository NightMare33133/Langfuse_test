"""
样本准备模块 — 解析 Dify / Langfuse 记录为结构化样本，回填参考答案和运行元数据。

支持两种 Langfuse 导出格式：
1. Events export: 一行一个 observation，input/output 通常是 JSON 字符串
2. API export: 一行一个 trace 根节点加子 observation，input/output 已解析

核心流程：
- 按 traceId 聚合 observations → 构建结构化样本
- 从题目库回填 reference_answer、source_excerpt、question_mode 等
- 从 user_id (rag_eval:<run_id>:<question_id>) 回填 run_id、question_set_id 等元数据
- 产出 processed samples（使用真实 Langfuse trace_id），供 Judge 评测消费
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)


def safe_json_loads(value):
    if isinstance(value, (dict, list)):
        return value
    if value in (None, "", "null"):
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value



def normalize_timestamp(value):
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo == timezone.utc:
            return dt.isoformat().replace("+00:00", "Z")
        return dt.isoformat()
    except ValueError:
        pass

    for fmt in _TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            if text.endswith("Z"):
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat().replace("+00:00", "Z")
            return dt.isoformat()
        except ValueError:
            continue

    return text



def normalize_observation_row(row):
    """Normalize a raw observation row to a consistent internal format.

    - Parses `input`/`output`/`metadata` from JSON strings to Python objects.
    - Normalizes timestamp strings to ISO-8601 format.
    - Maps TRACE type to SPAN (preserving original in `rawType`).
    - Backfills `sessionId`/`userId`/`traceName` from metadata when missing.

    Returns a new dict; the original `row` is not modified.
    """
    parsed_input = safe_json_loads(row.get("input"))
    parsed_output = safe_json_loads(row.get("output"))
    parsed_metadata = safe_json_loads(row.get("metadata")) or {}

    normalized = dict(row)
    raw_type = row.get("type")

    normalized["input"] = parsed_input
    normalized["output"] = parsed_output
    normalized["metadata"] = parsed_metadata
    normalized["startTime"] = normalize_timestamp(row.get("startTime"))
    normalized["endTime"] = normalize_timestamp(row.get("endTime"))
    normalized["completionStartTime"] = normalize_timestamp(
        row.get("completionStartTime")
    )
    normalized["rawType"] = raw_type
    normalized["isTraceRoot"] = raw_type == "TRACE"

    # API export includes a TRACE root record. We normalize it to SPAN so that
    # downstream code (observation_sort_key, build_trace_sample) treats it like
    # any other observation.  The original type is preserved in `rawType` and
    # flagged via `isTraceRoot` for callers that need to distinguish it.
    if raw_type == "TRACE":
        normalized["type"] = "SPAN"

    if not normalized.get("traceName"):
        normalized["traceName"] = row.get("name")

    if not normalized.get("sessionId"):
        normalized["sessionId"] = (
            parsed_metadata.get("conversation_id")
            or parsed_metadata.get("session_id")
            or parsed_metadata.get("sessionId")
        )

    if not normalized.get("userId"):
        normalized["userId"] = parsed_metadata.get("user_id") or parsed_metadata.get(
            "userId"
        )
        if not normalized.get("userId") and isinstance(parsed_input, dict):
            normalized["userId"] = parsed_input.get("sys.user_id")

    return normalized



def simplify_retrieval_item(item):
    metadata = item.get("metadata", {}) or {}
    return {
        "position": metadata.get("position"),
        "score": metadata.get("score"),
        "document_name": metadata.get("document_name"),
        "segment_id": metadata.get("segment_id"),
        "chunk_id": metadata.get("chunk_id"),
        "node_type": metadata.get("node_type"),
        "title": item.get("title"),
        "content": item.get("content"),
    }



def observation_sort_key(obs):
    root_priority = 0 if obs.get("isTraceRoot") else 1
    return (
        obs.get("startTime") or "",
        root_priority,
        obs.get("name") or "",
        obs.get("id") or "",
    )



def build_trace_sample(trace_id, observations):
    observations = sorted(observations, key=observation_sort_key)

    sample = {
        "trace_id": trace_id,
        "trace_name": None,
        "session_id": None,
        "user_id": None,
        "workflow_run_id": None,
        "question": None,
        "root_input": None,
        "root_output": None,
        "retrieval_query": None,
        "retrieval_results": [],
        "llm_model": None,
        "llm_input": None,
        "llm_output": None,
        "final_answer": None,
        "observations": [],
    }

    for obs in observations:
        parsed_input = obs.get("input")
        parsed_output = obs.get("output")
        parsed_metadata = obs.get("metadata") or {}

        name = obs.get("name")
        node_type = parsed_metadata.get("node_type")
        obs_type = obs.get("type")

        sample["trace_name"] = sample["trace_name"] or obs.get("traceName")
        sample["session_id"] = sample["session_id"] or obs.get("sessionId")
        sample["user_id"] = sample["user_id"] or obs.get("userId")
        sample["workflow_run_id"] = sample["workflow_run_id"] or parsed_metadata.get(
            "workflow_run_id"
        )
        if not sample["workflow_run_id"] and isinstance(parsed_input, dict):
            sample["workflow_run_id"] = parsed_input.get("sys.workflow_run_id")

        sample["observations"].append(
            {
                "id": obs.get("id"),
                "type": obs_type,
                "raw_type": obs.get("rawType"),
                "is_trace_root": obs.get("isTraceRoot", False),
                "name": name,
                "node_type": node_type,
                "start_time": obs.get("startTime"),
                "input": parsed_input,
                "output": parsed_output,
                "metadata": parsed_metadata,
            }
        )

        if isinstance(parsed_input, dict):
            question = parsed_input.get("sys.query") or parsed_input.get("query")
            if question and not sample["question"]:
                sample["question"] = question

        is_root_candidate = (
            obs.get("isTraceRoot")
            or name == "message"
            or (
                obs_type == "SPAN"
                and sample["root_input"] is None
                and isinstance(parsed_output, dict)
                and "answer" in parsed_output
            )
        )
        if is_root_candidate:
            if sample["root_input"] is None and parsed_input is not None:
                sample["root_input"] = parsed_input
            if sample["root_output"] is None and parsed_output is not None:
                sample["root_output"] = parsed_output
            if isinstance(parsed_output, dict) and parsed_output.get("answer"):
                sample["final_answer"] = sample["final_answer"] or parsed_output["answer"]

        if node_type == "knowledge-retrieval" or "??" in (name or ""):
            if isinstance(parsed_input, dict):
                sample["retrieval_query"] = parsed_input.get("query") or sample[
                    "retrieval_query"
                ]
            if isinstance(parsed_output, dict):
                results = parsed_output.get("result") or []
                if isinstance(results, list):
                    sample["retrieval_results"] = [
                        simplify_retrieval_item(item)
                        for item in results
                        if isinstance(item, dict)
                    ]

        if obs.get("rawType") == "GENERATION" or node_type == "llm" or name == "LLM":
            sample["llm_model"] = (
                obs.get("providedModelName")
                or parsed_metadata.get("model")
                or sample["llm_model"]
            )
            sample["llm_input"] = parsed_input
            sample["llm_output"] = parsed_output
            if isinstance(parsed_output, dict) and parsed_output.get("text"):
                sample["final_answer"] = sample["final_answer"] or parsed_output["text"]

        if node_type == "answer" or "??" in (name or ""):
            if isinstance(parsed_output, dict) and parsed_output.get("answer"):
                sample["final_answer"] = parsed_output["answer"]

    return sample



def load_jsonl(path):
    traces = defaultdict(list)
    bad_lines = []

    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_lines.append({"line": idx, "error": str(exc)})
                continue
            trace_id = row.get("traceId")
            if not trace_id:
                bad_lines.append({"line": idx, "error": "missing traceId"})
                continue
            traces[trace_id].append(normalize_observation_row(row))

    return traces, bad_lines



def load_question_index(questions_dir=None):
    """从 data/questions/*.jsonl 加载题目索引，用于回填 reference_answer。

    返回两个 dict：
    - by_id:   {question_id: question_dict}
    - by_text: {question_text: question_dict}  （精确匹配，取最新文件中的版本）

    按文件名排序（时间戳在文件名中），后加载的覆盖先加载的，保证取最新。
    """
    from pathlib import Path as _Path
    if questions_dir is None:
        questions_dir = _Path(__file__).parent / "data" / "questions"
    else:
        questions_dir = _Path(questions_dir)

    by_id = {}
    by_text = {}

    if not questions_dir.exists():
        return by_id, by_text

    for f in sorted(questions_dir.glob("*.jsonl")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        q = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q_text = (q.get("question") or "").strip()
                    if not q_text:
                        continue
                    # 按 question_id 索引
                    qid = q.get("question_id")
                    if qid:
                        by_id[str(qid)] = q
                    # 按 question 文本索引（后加载覆盖先加载）
                    by_text[q_text] = q
        except Exception:
            continue

    return by_id, by_text


_BACKFILL_FIELDS = ("reference_answer", "source_excerpt", "difficulty", "topic", "question_id", "question_mode",
                    "question_set_id", "question_set_name")


def parse_rag_eval_user_id(user_id: str) -> dict:
    """解析 rag_eval:<run_id>:<question_id> 格式的 user_id。

    Returns:
        dict: {"run_id": str, "question_id": str} 或空 dict
    """
    if not user_id or not isinstance(user_id, str):
        return {}

    # 格式: rag_eval:<run_id>:<question_id>
    if user_id.startswith("rag_eval:"):
        parts = user_id.split(":", 2)
        if len(parts) == 3:
            return {
                "run_id": parts[1],
                "question_id": parts[2],
            }

    return {}


def extract_experiment_metadata_from_samples(samples: list) -> dict:
    """从样本中提取实验元数据统计。

    Returns:
        dict: {
            "total_samples": int,
            "identified_runs": dict[str, int],  # run_id -> count
            "identified_count": int,
            "unidentified_count": int,
        }
    """
    stats = {
        "total_samples": len(samples),
        "identified_runs": {},
        "identified_count": 0,
        "unidentified_count": 0,
    }

    for s in samples:
        user_id = s.get("user_id", "")
        parsed = parse_rag_eval_user_id(user_id)
        if parsed.get("run_id"):
            run_id = parsed["run_id"]
            stats["identified_runs"][run_id] = stats["identified_runs"].get(run_id, 0) + 1
            stats["identified_count"] += 1
        else:
            stats["unidentified_count"] += 1

    return stats


def backfill_reference_answers(samples, questions_dir=None):
    """尝试为样本回填 reference_answer 等题目元数据。

    匹配规则：
    1. 从 user_id 解析 run_id/question_id（如果格式为 rag_eval:run_id:question_id）
    2. 按 question_id 精确匹配（如果样本有 question_id）
    3. 按 question 文本精确匹配

    返回 (samples, stats)：
    - samples: 原地修改后的样本列表
    - stats: {"total": N, "backfilled": M, "already_has": K, "run_id_parsed": int}
    """
    by_id, by_text = load_question_index(questions_dir)

    if not by_id and not by_text:
        return samples, {"total": len(samples), "backfilled": 0, "already_has": 0, "run_id_parsed": 0}

    stats = {"total": len(samples), "backfilled": 0, "already_has": 0, "run_id_parsed": 0}

    for s in samples:
        # 从 user_id 解析 run_id/question_id
        user_id = s.get("user_id", "")
        parsed = parse_rag_eval_user_id(user_id)
        if parsed.get("run_id"):
            s["run_id"] = parsed["run_id"]
            stats["run_id_parsed"] += 1
        if parsed.get("question_id") and not s.get("question_id"):
            s["question_id"] = parsed["question_id"]

        # 已有 reference_answer 的跳过
        if (s.get("reference_answer") or "").strip():
            stats["already_has"] += 1
            continue

        matched_q = None

        # 优先按 question_id 匹配
        qid = s.get("question_id")
        if qid and str(qid) in by_id:
            matched_q = by_id[str(qid)]

        # 其次按 question 文本精确匹配
        if not matched_q:
            q_text = (s.get("question") or "").strip()
            if q_text and q_text in by_text:
                matched_q = by_text[q_text]

        if matched_q:
            for field in _BACKFILL_FIELDS:
                val = matched_q.get(field)
                if val and not s.get(field):
                    s[field] = val
            stats["backfilled"] += 1

    return samples, stats


def parse_langfuse_jsonl(input_path):
    """Parse a Langfuse JSONL export file and return samples + summary."""
    input_path = Path(input_path)
    traces, bad_lines = load_jsonl(input_path)
    samples = [build_trace_sample(trace_id, obs) for trace_id, obs in traces.items()]
    samples.sort(key=lambda x: (x.get("question") or "", x["trace_id"]))

    # 回填 reference_answer
    samples, backfill_stats = backfill_reference_answers(samples)

    summary = {
        "input_file": str(input_path),
        "trace_count": len(samples),
        "bad_line_count": len(bad_lines),
        "bad_lines": bad_lines[:20],
        "total_retrieval_results": sum(
            len(s.get("retrieval_results", [])) for s in samples
        ),
        "backfill_stats": backfill_stats,
    }

    return samples, summary



def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def save_results(samples, summary, output_path, summary_path):
    """Save parsed samples and summary to disk."""
    write_jsonl(output_path, samples)

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    full_summary = {**summary, "output_file": str(output_path)}
    summary_path.write_text(
        json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return full_summary
