"""
Judge LLM module - evaluates RAG samples using an LLM judge.

Uses OpenAI-compatible chat completions API via requests.
"""

import hashlib
import json
import re
from pathlib import Path

import requests


def compute_content_hash(sample):
    """基于 question + retrieval_query + final_answer + reference_answer 生成内容指纹。

    同内容不同 trace 的样本会得到相同 hash，用于去重和缓存复用。
    包含 reference_answer 以确保有/无参考答案的样本不会误合并。
    """
    parts = [
        (sample.get("question") or "").strip(),
        (sample.get("retrieval_query") or "").strip(),
        (sample.get("final_answer") or "").strip(),
        (sample.get("reference_answer") or "").strip(),
    ]
    raw = "\n".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "judge_prompt.txt"
PROMPT_TEMPLATE_WITH_REF_PATH = Path(__file__).parent / "prompts" / "judge_prompt_with_ref.txt"


def load_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def load_prompt_template_with_ref():
    return PROMPT_TEMPLATE_WITH_REF_PATH.read_text(encoding="utf-8").strip()


def _clean_content(text):
    """清洗 content 中的结构噪音，保留对 Judge 有用的正文。

    清洗内容：
    1. <metadata>...</metadata> 块 — 结构化元数据 JSON，对判断 hit 无用
    2. 上下文引用标记（previous_content / next_content 等残留）
    """
    if not text:
        return text
    # 去掉 <metadata>...</metadata> 块（含跨行）
    text = re.sub(r"<metadata>\s*\{[\s\S]*?\}\s*</metadata>\s*", "", text)
    # 去掉可能残留的空行堆积
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 每条检索结果的正文字符上限
# Top-1 对 hit 判断最关键，给更多空间；Top-2~5 适当缩减
_CONTENT_LIMITS = {0: 2000, 1: 1200, 2: 1200, 3: 1000, 4: 1000}


def _format_single_result(r, index, max_content_chars=None):
    """格式化单条检索结果：先清洗噪音，再保留正文。"""
    raw_content = r.get("content") or "(无内容)"
    score = r.get("score")
    pos = r.get("position")
    doc_name = r.get("document_name") or ""

    # 1. 清洗 content 中的 metadata 块
    content = _clean_content(raw_content)

    # 2. 确定本条的字符上限
    limit = max_content_chars or _CONTENT_LIMITS.get(index, 1000)
    if len(content) > limit:
        content = content[:limit] + "...(截断)"

    # 3. 来源标识：只用 document_name，不用 title（实测 title 几乎全为 null）
    source_label = doc_name if doc_name else ""

    # 4. 组装：标注行 + 正文
    parts = []
    pos_tag = f"[{pos}]" if pos is not None else ""
    score_tag = f"(score: {score:.4f})" if score is not None else ""
    tags = " ".join(filter(None, [pos_tag, source_label, score_tag]))
    if tags:
        parts.append(tags)
    parts.append(content)
    return "\n".join(parts)


def build_judge_prompt(sample, template=None, max_content_chars=None):
    """构建 Judge prompt。如果 sample 包含 reference_answer，自动使用含参考答案的模板。

    Args:
        sample: 样本数据字典
        template: 自定义 prompt 模板（可选）
        max_content_chars: 统一字符上限。为 None 时使用分层策略（Top-1: 2000, Top-2~5: 1000~1200）。
    """
    has_ref = bool((sample.get("reference_answer") or "").strip())

    if template is not None:
        pass  # 使用传入的模板
    elif has_ref:
        template = load_prompt_template_with_ref()
    else:
        template = load_prompt_template()

    retrieval_results = sample.get("retrieval_results") or []
    if retrieval_results:
        formatted = []
        for i, r in enumerate(retrieval_results):
            formatted.append(
                f"--- 检索结果 {i + 1} ---\n"
                + _format_single_result(r, i, max_content_chars)
            )
        retrieval_text = "\n\n".join(formatted)
    else:
        retrieval_text = "(无检索结果)"

    prompt = template
    prompt = prompt.replace("{question}", sample.get("question") or "(无)")
    prompt = prompt.replace("{retrieval_query}", sample.get("retrieval_query") or "(无)")
    prompt = prompt.replace("{retrieval_results}", retrieval_text)
    prompt = prompt.replace("{final_answer}", sample.get("final_answer") or "(无)")
    if has_ref:
        prompt = prompt.replace("{reference_answer}", sample.get("reference_answer"))
        prompt = prompt.replace("{source_excerpt}", sample.get("source_excerpt") or "(无)")
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
    has_ref = bool((sample.get("reference_answer") or "").strip())
    result = {
        "trace_id": trace_id,
        "question": sample.get("question") or "",
        "has_reference": has_ref,
    }

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
    """Judge all samples sequentially. Yields results one by one.

    内置规则预筛选和内容级去重：
    - 预筛选：无检索结果/无回答的样本直接出结果，不调 LLM
    - 去重：相同 question+retrieval_query+final_answer 的样本只评一次

    模板选择：不预加载模板，由 build_judge_prompt 根据每个 sample
    是否包含 reference_answer 自动选择对应模板。
    """
    content_cache = {}  # content_hash -> result dict (without trace_id/question)
    total = len(samples)

    for i, sample in enumerate(samples):
        # 规则预筛选
        prescreened = pre_screen(sample)
        if prescreened is not None:
            result = {
                "trace_id": sample.get("trace_id", "unknown"),
                "question": sample.get("question") or "",
                "_prescreened": True,
                **prescreened,
            }
            if progress_callback:
                progress_callback(i + 1, total, result)
            yield result
            continue

        # 内容级去重
        ch = compute_content_hash(sample)
        if ch in content_cache:
            cached = content_cache[ch]
            result = {
                "trace_id": sample.get("trace_id", "unknown"),
                "question": sample.get("question") or "",
                "_content_cached": True,
                **cached,
            }
            if progress_callback:
                progress_callback(i + 1, total, result)
            yield result
            continue

        # 实际调用 LLM（不传 template，由 build_judge_prompt 根据 sample 自动选择）
        result = judge_sample(sample, api_key, base_url, model, timeout=timeout)
        # 缓存成功结果（不含 trace_id/question，因为这些是样本特定的）
        if "error" not in result:
            content_cache[ch] = {
                k: v for k, v in result.items()
                if k not in ("trace_id", "question", "_prompt", "_raw_response")
            }
        if progress_callback:
            progress_callback(i + 1, total, result)
        yield result


def pre_screen(sample):
    """规则预筛选：对结果确定的样本直接返回评分，不需要调用 LLM。

    返回 None 表示无法用规则判定，需要走 LLM。
    返回 dict 表示已有确定结果，可直接使用。
    """
    question = (sample.get("question") or "").strip()
    final_answer = (sample.get("final_answer") or "").strip()
    retrieval_results = sample.get("retrieval_results") or []

    # 无问题 → 无法评测
    if not question:
        return {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                "retrieval_top5_hit": 0, "answer_correct": 0,
                "reason": "规则判定：无用户问题"}

    # 无检索结果 → top 全 0
    no_retrieval = len(retrieval_results) == 0

    # 无最终回答 → answer 错误
    no_answer = not final_answer

    if no_retrieval and no_answer:
        return {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                "retrieval_top5_hit": 0, "answer_correct": 0,
                "reason": "规则判定：无检索结果且无最终回答"}

    if no_retrieval:
        # 有回答但无检索，top 全 0，answer 需要 LLM 判断
        return None

    if no_answer:
        # 有检索但无回答
        return {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                "retrieval_top5_hit": 0, "answer_correct": 0,
                "reason": "规则判定：无最终回答"}

    return None


def compute_metrics(results):
    """Compute aggregate metrics from judge results.

    返回值新增 with_ref_count / without_ref_count，用于 UI 区分评测模式。
    """
    valid = [r for r in results if "error" not in r]
    total = len(results)
    errored = total - len(valid)

    with_ref = [r for r in valid if r.get("has_reference")]
    without_ref = [r for r in valid if not r.get("has_reference")]

    if not valid:
        return {
            "total": total,
            "evaluated": 0,
            "errors": errored,
            "top1_hit_rate": None,
            "top3_hit_rate": None,
            "top5_hit_rate": None,
            "answer_correct_rate": None,
            "with_ref_count": 0,
            "without_ref_count": 0,
            "has_reference_data": False,
            "with_ref_answer_rate": None,
            "without_ref_answer_rate": None,
        }

    n = len(valid)
    with_ref_n = len(with_ref)
    without_ref_n = len(without_ref)

    return {
        "total": total,
        "evaluated": n,
        "errors": errored,
        "top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in valid) / n,
        "top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in valid) / n,
        "top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in valid) / n,
        "answer_correct_rate": sum(r.get("answer_correct", 0) for r in valid) / n,
        "with_ref_count": with_ref_n,
        "without_ref_count": without_ref_n,
        "has_reference_data": with_ref_n > 0,
        "with_ref_answer_rate": sum(r.get("answer_correct", 0) for r in with_ref) / with_ref_n if with_ref_n else None,
        "without_ref_answer_rate": sum(r.get("answer_correct", 0) for r in without_ref) / without_ref_n if without_ref_n else None,
    }
