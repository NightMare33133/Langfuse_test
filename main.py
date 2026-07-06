import argparse
import json
from pathlib import Path

from parser import load_jsonl, build_trace_sample, write_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Convert Langfuse exported JSONL into one-sample-per-trace JSONL."
    )
    parser.add_argument("input", help="Path to Langfuse exported .jsonl")
    parser.add_argument(
        "--output",
        help="Output JSONL path. Default: <input stem>.samples.jsonl",
    )
    parser.add_argument(
        "--summary",
        help="Optional summary JSON path. Default: <input stem>.summary.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(
        ".samples.jsonl"
    )
    summary_path = Path(args.summary) if args.summary else input_path.with_suffix(
        ".summary.json"
    )

    traces, bad_lines = load_jsonl(input_path)
    samples = [build_trace_sample(trace_id, obs) for trace_id, obs in traces.items()]
    samples.sort(key=lambda x: (x.get("question") or "", x["trace_id"]))

    write_jsonl(output_path, samples)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "trace_count": len(samples),
        "bad_line_count": len(bad_lines),
        "bad_lines": bad_lines[:20],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
