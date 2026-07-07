"""
Judge LLM module - evaluates RAG samples using an LLM judge.

Uses OpenAI-compatible chat completions API via requests.
"""

import json
import re
from pathlib import Path

import requests

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "judge_prompt.txt"


def load_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def build_judge_prompt(sample, template=None):
    if template is None:
        template = load_prompt_template()

    retrieval_results = sample.get("retrieval_results") or []
    if retrieval_results:
        lines = []
        for r in retrieval_results:
            title = r.get("title") or "(无标题)"
            content = r.get("content") or "(无内容)"
            score = r.get("score")
            pos = r.get("position")
            prefix = f"[位置{pos}]" if pos is not None else ""
            score_str = f" (score: {score})" if score is not None else ""
            lines.append(f"{prefix}{title}{score_str}: {content}")
        retrieval_text = "\n".join(lines)
    else:
        retrieval_text = "(无检索结果)"

    prompt = template
    prompt = prompt.replace("{question}", sample.get("question") or "(无)")
    prompt = prompt.replace("{retrieval_query}", sample.get("retrieval_query") or "(无)")
    prompt = prompt.replace("{retrieval_results}", retrieval_text)
    prompt = prompt.replace("{final_answer}", sample.get("final_answer") or "(无)")
    return prompt


def parse_judge_response(text):
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"LLM response does not contain JSON: {text[:200]}")
    return json.loads(match.group(0))


def call_llm(prompt, api_key, base_url, model, timeout=30):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"请求超时 ({timeout}s): {url}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"连接失败: {url}\n{e}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} | URL: {url}\nResponse: {resp.text[:1000]}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"JSON 解析失败 | Response: {resp.text[:1000]}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"响应结构异常 | Response: {json.dumps(data, ensure_ascii=False)[:1000]}")


def judge_sample(sample, api_key, base_url, model, prompt_template=None, timeout=60):
    """Judge a single sample. Returns a dict with scores or error info."""
    trace_id = sample.get("trace_id", "unknown")
    result = {"trace_id": trace_id, "question": sample.get("question") or ""}

    try:
        prompt = build_judge_prompt(sample, prompt_template)
        result["_prompt"] = prompt
        response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)
        result["_raw_response"] = response_text
        scores = parse_judge_response(response_text)
        result.update(scores)
    except Exception as e:
        result["error"] = str(e)

    return result


def judge_all(samples, api_key, base_url, model, progress_callback=None, timeout=60):
    """Judge all samples sequentially. Yields results one by one."""
    template = load_prompt_template()
    for i, sample in enumerate(samples):
        result = judge_sample(sample, api_key, base_url, model, template, timeout=timeout)
        if progress_callback:
            progress_callback(i + 1, len(samples), result)
        yield result


def compute_metrics(results):
    """Compute aggregate metrics from judge results."""
    valid = [r for r in results if "error" not in r]
    total = len(results)
    errored = total - len(valid)

    if not valid:
        return {
            "total": total,
            "evaluated": 0,
            "errors": errored,
            "top1_hit_rate": None,
            "top3_hit_rate": None,
            "top5_hit_rate": None,
            "answer_correct_rate": None,
        }

    n = len(valid)
    return {
        "total": total,
        "evaluated": n,
        "errors": errored,
        "top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in valid) / n,
        "top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in valid) / n,
        "top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in valid) / n,
        "answer_correct_rate": sum(r.get("answer_correct", 0) for r in valid) / n,
    }
