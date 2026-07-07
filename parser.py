"""
Langfuse JSONL export parser - reusable functions for parsing Langfuse traces.

Supports both of the Langfuse export shapes currently seen in this project:
1. Events export: one observation per line, `input`/`output` often stored as JSON strings.
2. API export: one trace root plus child observations, `input`/`output` already parsed.
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



def parse_langfuse_jsonl(input_path):
    """Parse a Langfuse JSONL export file and return samples + summary."""
    input_path = Path(input_path)
    traces, bad_lines = load_jsonl(input_path)
    samples = [build_trace_sample(trace_id, obs) for trace_id, obs in traces.items()]
    samples.sort(key=lambda x: (x.get("question") or "", x["trace_id"]))

    summary = {
        "input_file": str(input_path),
        "trace_count": len(samples),
        "bad_line_count": len(bad_lines),
        "bad_lines": bad_lines[:20],
        "total_retrieval_results": sum(
            len(s.get("retrieval_results", [])) for s in samples
        ),
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
