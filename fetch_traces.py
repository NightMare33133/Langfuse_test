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


def fetch_all(host, public_key, secret_key, limit=50, max_pages=20,
              progress_callback=None, max_retries=2):
    """Fetch traces and their observations, yielding JSONL-compatible rows.

    Args:
        host: Langfuse API host URL.
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        limit: Number of traces per page.
        max_pages: Maximum number of pages to fetch.
        progress_callback: Optional callable(phase, traces_fetched, pages_done,
            total_traces, retries). Called after each page completes.
            Phases: "connecting", "fetching", "done".
            total_traces comes from API meta.totalItems (may be None).
        max_retries: Max retry attempts per page on transient failure.

    Yields:
        Row dicts compatible with parser.py's load_jsonl().
    """
    total_rows = 0
    traces_fetched = 0
    pages_done = 0
    total_traces = None  # from API meta, may remain None
    cumulative_retries = 0

    if progress_callback:
        progress_callback("connecting", 0, 0, None, 0)

    for page in range(1, max_pages + 1):
        # --- fetch one page with retry ---
        data = None
        page_retries = 0
        for attempt in range(max_retries + 1):
            try:
                data = fetch_traces(host, public_key, secret_key, limit=limit, page=page)
                break
            except Exception:
                if attempt < max_retries:
                    page_retries += 1
                    cumulative_retries += 1
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise

        # Read meta.totalItems from first successful response
        if total_traces is None and data is not None:
            meta = data.get("meta") or {}
            api_total = meta.get("totalItems")
            if isinstance(api_total, int) and api_total > 0:
                total_traces = api_total

        traces = data.get("data", []) if data else []
        if not traces:
            pages_done = page
            if progress_callback:
                progress_callback("fetching", traces_fetched, pages_done,
                                  total_traces, cumulative_retries)
            break

        for trace in traces:
            trace_id = trace["id"]
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

            try:
                observations = fetch_observations(host, public_key, secret_key, trace_id)
            except Exception:
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

        traces_fetched += len(traces)
        pages_done = page

        if progress_callback:
            progress_callback("fetching", traces_fetched, pages_done,
                              total_traces, cumulative_retries)

        if len(traces) < limit:
            break

        time.sleep(0.2)

    if progress_callback:
        progress_callback("done", traces_fetched, pages_done,
                          total_traces, cumulative_retries)


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

    def _cli_progress(phase, traces, pages, total, retries):
        total_str = f"/{total}" if total else ""
        retry_str = f" (重试 {retries} 次)" if retries else ""
        if phase == "done":
            print(f"  完成: {traces} 条 trace, {pages} 页{retry_str}")
        else:
            print(f"  [{phase}] 已拉取 {traces} 条 trace{total_str}, 第 {pages} 页{retry_str}")

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in fetch_all(
            args.host, args.public_key, args.secret_key,
            limit=args.limit, max_pages=args.max_pages,
            progress_callback=_cli_progress,
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"\n✅ 完成！共写入 {count} 行到 {output_path}")
    print(f"   下一步: 在 Streamlit 应用中选择该文件并点击「开始解析」")


if __name__ == "__main__":
    main()
