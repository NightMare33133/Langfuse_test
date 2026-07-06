"""
Langfuse JSONL export parser - reusable functions for parsing Langfuse traces.

Extracted from main.py to enable reuse by Streamlit app and other consumers.
"""

import json
from collections import defaultdict
from pathlib import Path


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
    return (
        obs.get("startTime") or "",
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
        parsed_input = safe_json_loads(obs.get("input"))
        parsed_output = safe_json_loads(obs.get("output"))
        parsed_metadata = safe_json_loads(obs.get("metadata")) or {}

        name = obs.get("name")
        node_type = parsed_metadata.get("node_type")
        obs_type = obs.get("type")

        sample["trace_name"] = sample["trace_name"] or obs.get("traceName")
        sample["session_id"] = sample["session_id"] or obs.get("sessionId")
        sample["user_id"] = sample["user_id"] or obs.get("userId")
        sample["workflow_run_id"] = sample["workflow_run_id"] or parsed_metadata.get(
            "workflow_run_id"
        )

        sample["observations"].append(
            {
                "id": obs.get("id"),
                "type": obs_type,
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

        if name == "message" and sample["root_input"] is None and parsed_input:
            sample["root_input"] = parsed_input
        if name == "message" and sample["root_output"] is None and parsed_output:
            sample["root_output"] = parsed_output
            if isinstance(parsed_output, dict) and parsed_output.get("answer"):
                sample["final_answer"] = parsed_output["answer"]

        if node_type == "knowledge-retrieval" or "知识" in (name or ""):
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

        if obs_type == "GENERATION" or node_type == "llm" or name == "LLM":
            sample["llm_model"] = obs.get("providedModelName") or sample["llm_model"]
            sample["llm_input"] = parsed_input
            sample["llm_output"] = parsed_output
            if isinstance(parsed_output, dict) and parsed_output.get("text"):
                sample["final_answer"] = sample["final_answer"] or parsed_output["text"]

        if node_type == "answer" or "回复" in (name or ""):
            if isinstance(parsed_output, dict) and parsed_output.get("answer"):
                sample["final_answer"] = parsed_output["answer"]

    return sample


def load_jsonl(path):
    traces = defaultdict(list)
    bad_lines = []

    with path.open("r", encoding="utf-8") as f:
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
            traces[trace_id].append(row)

    return traces, bad_lines


def parse_langfuse_jsonl(input_path):
    """Parse a Langfuse JSONL export file and return samples + summary.

    Args:
        input_path: Path to the Langfuse export JSONL file.

    Returns:
        Tuple of (samples, summary) where samples is a list of trace dicts
        and summary contains parsing statistics.
    """
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
    """Save parsed samples and summary to disk.

    Args:
        samples: List of trace sample dicts.
        summary: Summary dict with parsing statistics.
        output_path: Path for the samples JSONL output.
        summary_path: Path for the summary JSON output.
    """
    write_jsonl(output_path, samples)

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    full_summary = {**summary, "output_file": str(output_path)}
    summary_path.write_text(
        json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return full_summary
