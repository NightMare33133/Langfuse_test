"""批量提问模块 — 调用 Dify chat-messages API 逐条提问并收集结果。

输出格式与 parser.py 的 sample 结构兼容，可直接作为"样本列表"和"评测"的输入。
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests

BATCH_DIR = Path(__file__).parent / "data" / "batch"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def call_dify_query(question, api_key, base_url, timeout=60):
    """调用 Dify chat-messages API（blocking 模式）提出单个问题。

    Returns:
        dict with keys: answer, conversation_id, message_id, retriever_resources, raw_response
    Raises:
        RuntimeError on API errors.
    """
    url = base_url.rstrip("/") + "/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": {},
        "query": question,
        "response_mode": "blocking",
        "user": "batch-query",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"请求超时 ({timeout}s): {url}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"连接失败: {url}\n{e}")

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"JSON 解析失败: {resp.text[:500]}")

    answer = data.get("answer", "")
    conversation_id = data.get("conversation_id")
    message_id = data.get("message_id")
    metadata = data.get("metadata") or {}
    retriever_resources = metadata.get("retriever_resources") or []

    return {
        "answer": answer,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "retriever_resources": retriever_resources,
        "raw_response": data,
    }


def _map_retriever_resources(resources):
    """将 Dify 的 retriever_resources 映射到 parser.py 的 retrieval_results 格式。

    Dify 返回格式:
        [{"dataset_name": "...", "document_name": "...", "segment_id": "...", "score": 0.95, "content": "..."}]

    目标格式:
        [{"position": int, "score": float, "document_name": str, "segment_id": str,
          "chunk_id": None, "node_type": None, "title": str, "content": str}]
    """
    results = []
    for i, res in enumerate(resources):
        results.append({
            "position": i + 1,
            "score": res.get("score"),
            "document_name": res.get("document_name"),
            "segment_id": res.get("segment_id"),
            "chunk_id": None,
            "node_type": None,
            "title": res.get("dataset_name") or res.get("document_name"),
            "content": res.get("content"),
        })
    return results


def query_to_sample(question, dify_result, index, timestamp, reference_answer="", source_excerpt=""):
    """将单次提问结果转换为 parser.py 兼容的 sample 结构。

    Args:
        question: 原始问题文本
        dify_result: call_dify_query 的返回值
        index: 问题序号（从 0 开始）
        timestamp: 批次时间戳字符串
        reference_answer: 参考答案（如有）
        source_excerpt: 来源摘录（如有）
    """
    retrieval_results = _map_retriever_resources(dify_result.get("retriever_resources", []))
    retrieval_query = question  # Dify 不单独返回 retrieval_query，用原始问题代替

    sample = {
        "trace_id": f"batch_qa_{index}_{timestamp}",
        "trace_name": "batch_query",
        "session_id": dify_result.get("conversation_id"),
        "user_id": "batch-query",
        "workflow_run_id": None,
        "question": question,
        "root_input": None,
        "root_output": None,
        "retrieval_query": retrieval_query,
        "retrieval_results": retrieval_results,
        "llm_model": "dify-chat",
        "llm_input": None,
        "llm_output": None,
        "final_answer": dify_result.get("answer", ""),
        "observations": [],
    }
    if reference_answer:
        sample["reference_answer"] = reference_answer
    if source_excerpt:
        sample["source_excerpt"] = source_excerpt
    return sample


def run_batch_query(questions, api_key, base_url, timeout=60, delay=1.0):
    """批量提问生成器。逐条调用 Dify API，yield 进度和结果。

    Args:
        questions: 问题列表。每项可以是：
            - str: 纯问题文本
            - dict: 包含 question、reference_answer、source_excerpt 等字段
        api_key: Dify API Key
        base_url: Dify API Base URL (e.g. http://localhost/v1)
        timeout: 单次请求超时秒数
        delay: 每次请求之间的间隔秒数

    Yields:
        (index, total, result_dict)
        result_dict 包含:
          - success: bool
          - question: str
          - sample: dict (仅 success=True 时)
          - raw_response: dict (仅 success=True 时)
          - error: str (仅 success=False 时)
    """
    total = len(questions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, q in enumerate(questions):
        # 支持 str 和 dict 两种输入格式
        if isinstance(q, dict):
            question = (q.get("question") or q.get("query") or "").strip()
            reference_answer = q.get("reference_answer", "")
            source_excerpt = q.get("source_excerpt", "")
        else:
            question = str(q).strip()
            reference_answer = ""
            source_excerpt = ""

        if not question:
            yield i, total, {
                "success": False,
                "question": question,
                "error": "问题为空，已跳过",
            }
            continue

        try:
            dify_result = call_dify_query(question, api_key, base_url, timeout=timeout)
            sample = query_to_sample(
                question, dify_result, i, timestamp,
                reference_answer=reference_answer,
                source_excerpt=source_excerpt,
            )
            yield i, total, {
                "success": True,
                "question": question,
                "sample": sample,
                "raw_response": dify_result.get("raw_response"),
            }
        except Exception as e:
            yield i, total, {
                "success": False,
                "question": question,
                "error": str(e),
            }

        # 请求间隔（最后一条不等待）
        if i < total - 1 and delay > 0:
            time.sleep(delay)


def save_batch_results(results, filename=None):
    """保存批量提问结果到 data/batch/ 目录（JSONL 格式）。

    Args:
        results: list of result dicts (from run_batch_query)
        filename: 可选文件名，默认自动生成带时间戳的文件名

    Returns:
        (output_path, filename)
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{ts}.jsonl"

    output_path = BATCH_DIR / filename
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return output_path, filename


def push_to_raw_dir(results, filename=None):
    """将成功的批量提问结果推送到 data/raw/ 目录，供 parser.py 直接读取。

    只保存 sample 结构（与 Langfuse 导出格式兼容），不保存 raw_response 等调试信息。

    Args:
        results: list of result dicts (from run_batch_query)
        filename: 可选文件名

    Returns:
        (output_path, filename)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_qa_{ts}.jsonl"

    output_path = RAW_DIR / filename
    samples = [r["sample"] for r in results if r.get("success") and r.get("sample")]

    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return output_path, filename


def export_csv_bytes(results):
    """导出批量提问结果为 CSV（UTF-8 with BOM，Excel 友好）。"""
    import io
    import csv

    output = io.StringIO()
    # BOM for Excel
    output.write("\ufeff")

    writer = csv.writer(output)
    writer.writerow(["序号", "问题", "回答", "检索结果数", "状态", "错误信息"])

    for i, r in enumerate(results):
        if r.get("success"):
            sample = r.get("sample", {})
            answer = sample.get("final_answer", "")
            retrieval_count = len(sample.get("retrieval_results", []))
            writer.writerow([i + 1, r.get("question", ""), answer, retrieval_count, "成功", ""])
        else:
            writer.writerow([i + 1, r.get("question", ""), "", 0, "失败", r.get("error", "")])

    return output.getvalue().encode("utf-8-sig")
