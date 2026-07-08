"""
Langfuse API trace fetcher - directly pull traces from Langfuse REST API.

Bypasses the UI export feature (which requires S3 storage) and fetches
traces + observations via the public API, then saves as JSONL compatible
with the existing parser pipeline.

Usage:
    python fetch_traces.py
    python fetch_traces.py --limit 100 --output data/raw/my_traces.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")


def fetch_traces(host, public_key, secret_key, limit=50, page=1):
    """Fetch traces from Langfuse API."""
    url = f"{host.rstrip('/')}/api/public/traces"
    resp = requests.get(
        url,
        auth=(public_key, secret_key),
        params={"limit": limit, "page": page},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"获取 traces 失败: HTTP {resp.status_code}\n{resp.text[:500]}")
    return resp.json()


def fetch_observations(host, public_key, secret_key, trace_id):
    """Fetch all observations for a given trace."""
    url = f"{host.rstrip('/')}/api/public/observations"
    resp = requests.get(
        url,
        auth=(public_key, secret_key),
        params={"traceId": trace_id, "limit": 100},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"获取 observations 失败 (trace={trace_id}): HTTP {resp.status_code}\n{resp.text[:300]}"
        )
    data = resp.json()
    return data.get("data", [])


def fetch_all(host, public_key, secret_key, limit=50, max_pages=20):
    """Fetch traces and their observations, yielding JSONL-compatible rows.

    Each row is an observation dict with traceId injected, matching the
    format expected by parser.py's load_jsonl().
    """
    total_rows = 0
    for page in range(1, max_pages + 1):
        print(f"  拉取 traces page {page}...", end=" ", flush=True)
        data = fetch_traces(host, public_key, secret_key, limit=limit, page=page)
        traces = data.get("data", [])
        if not traces:
            print("无更多数据")
            break

        print(f"获取到 {len(traces)} 条 trace", flush=True)

        for trace in traces:
            trace_id = trace["id"]
            # Yield the trace itself as a "root" row (like the UI export does)
            yield {
                "id": trace_id,
                "traceId": trace_id,
                "type": "TRACE",
                "name": trace.get("name"),
                "startTime": trace.get("timestamp"),
                "endTime": trace.get("timestamp"),
                "input": trace.get("input"),
                "output": trace.get("output"),
                "metadata": trace.get("metadata"),
                "sessionId": trace.get("sessionId"),
                "userId": trace.get("userId"),
                "traceName": trace.get("name"),
                "providedModelName": None,
            }
            total_rows += 1

            # Fetch observations for this trace
            try:
                observations = fetch_observations(host, public_key, secret_key, trace_id)
            except Exception as e:
                print(f"    ⚠️ 获取 observations 失败 (trace={trace_id[:8]}): {e}")
                continue

            for obs in observations:
                yield {
                    "id": obs.get("id"),
                    "traceId": trace_id,
                    "type": obs.get("type"),
                    "name": obs.get("name"),
                    "startTime": obs.get("startTime"),
                    "endTime": obs.get("endTime"),
                    "input": obs.get("input"),
                    "output": obs.get("output"),
                    "metadata": obs.get("metadata"),
                    "sessionId": trace.get("sessionId"),
                    "userId": trace.get("userId"),
                    "traceName": trace.get("name"),
                    "providedModelName": obs.get("model"),
                }
                total_rows += 1

        # If fewer results than limit, we've reached the end
        if len(traces) < limit:
            break

        time.sleep(0.2)  # rate limit courtesy

    print(f"  共获取 {total_rows} 行数据")


def main():
    parser = argparse.ArgumentParser(
        description="从 Langfuse API 拉取 Traces 并保存为 JSONL"
    )
    parser.add_argument(
        "--host", default=LANGFUSE_HOST, help="Langfuse 服务地址"
    )
    parser.add_argument(
        "--public-key", default=PUBLIC_KEY, help="Langfuse Public Key"
    )
    parser.add_argument(
        "--secret-key", default=SECRET_KEY, help="Langfuse Secret Key"
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="每页拉取的 trace 数量"
    )
    parser.add_argument(
        "--max-pages", type=int, default=20, help="最大翻页数"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSONL 路径，默认: data/raw/langfuse_api_export_<时间戳>.jsonl",
    )
    args = parser.parse_args()

    if not args.public_key or not args.secret_key:
        print("❌ 请在 .env 中配置 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY")
        print("   或通过 --public-key / --secret-key 参数传入")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "data" / "raw" / f"langfuse_api_export_{ts}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 从 {args.host} 拉取 Traces...")
    print(f"   每页 {args.limit} 条，最多 {args.max_pages} 页")
    print(f"   输出: {output_path}")
    print()

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in fetch_all(
            args.host, args.public_key, args.secret_key,
            limit=args.limit, max_pages=args.max_pages,
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"\n✅ 完成！共写入 {count} 行到 {output_path}")
    print(f"   下一步: 在 Streamlit 应用中选择该文件并点击「开始解析」")


if __name__ == "__main__":
    main()
