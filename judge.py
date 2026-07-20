"""
Judge 评测模块 — 使用 LLM 对结构化样本进行自动评分。

使用 OpenAI 兼容的 chat completions API。

评测轨道：
- retrieval（检索评测）：有金标准证据，计算 Top1/Top3/Top5 Hit
- strict_qa（严格问答）：有 reference_answer，评判回答正确性
- grounded_qa（合理性问答）：无参考答案，基于检索内容判断合理性
- not_evaluable（不可评测）：缺少金标准证据的检索评测题

三层优化：结果跳过、内容级去重（compute_content_hash）、规则预筛选（pre_screen）。
"""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def compute_content_hash(sample):
    """基于 question_mode + question + retrieval_query + final_answer + gold_evidence 生成内容指纹。

    同内容不同 trace 的样本会得到相同 hash，用于去重和缓存复用。
    包含 question_mode 和 gold_evidence 以确保不同模式/金标准的样本不会误合并。
    """
    gold_evidence = get_gold_evidence(sample)
    parts = [
        (sample.get("question_mode") or "").strip(),
        (sample.get("question") or "").strip(),
        (sample.get("retrieval_query") or "").strip(),
        (sample.get("final_answer") or "").strip(),
        gold_evidence,
    ]
    raw = "\n".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "judge_prompt.txt"
PROMPT_TEMPLATE_WITH_REF_PATH = Path(__file__).parent / "prompts" / "judge_prompt_with_ref.txt"
PROMPT_TEMPLATE_RETRIEVAL_PATH = Path(__file__).parent / "prompts" / "judge_prompt_retrieval.txt"


def load_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def load_prompt_template_with_ref():
    return PROMPT_TEMPLATE_WITH_REF_PATH.read_text(encoding="utf-8").strip()


def load_prompt_template_retrieval():
    return PROMPT_TEMPLATE_RETRIEVAL_PATH.read_text(encoding="utf-8").strip()


# 评测轨道常量
TRACK_RETRIEVAL = "retrieval"          # 检索评测：有金标准证据，可计算 TopK Hit
TRACK_STRICT_QA = "strict_qa"          # 严格问答：有 reference_answer，可评判回答正确性
TRACK_GROUNDED_QA = "grounded_qa"      # 合理性问答：无参考答案，基于检索内容判断
TRACK_NOT_EVALUABLE = "not_evaluable"  # 不可评测：检索评测题但缺少金标准证据


def classify_evaluation_track(sample):
    """根据 question_mode 和参考信息分类评测轨道。

    Returns:
        str: TRACK_RETRIEVAL / TRACK_STRICT_QA / TRACK_GROUNDED_QA / TRACK_NOT_EVALUABLE
    """
    question_mode = (sample.get("question_mode") or "").strip()
    has_source_excerpt = bool((sample.get("source_excerpt") or "").strip())
    has_reference_answer = bool((sample.get("reference_answer") or "").strip())

    if question_mode == "retrieval":
        # 检索评测题：优先用 source_excerpt，其次用 reference_answer
        if has_source_excerpt:
            return TRACK_RETRIEVAL
        elif has_reference_answer:
            return TRACK_RETRIEVAL  # 有 reference_answer 可作为次级金标准
        else:
            return TRACK_NOT_EVALUABLE  # 缺少金标准，无法可靠计算 Hit
    elif question_mode == "qa":
        # 全流程问答题
        if has_reference_answer:
            return TRACK_STRICT_QA
        else:
            return TRACK_GROUNDED_QA
    else:
        # 旧版/未知模式：按是否有参考答案区分
        if has_reference_answer:
            return TRACK_STRICT_QA
        else:
            return TRACK_GROUNDED_QA


def get_gold_evidence(sample):
    """获取金标准证据，优先 source_excerpt，其次 reference_answer。"""
    source_excerpt = (sample.get("source_excerpt") or "").strip()
    reference_answer = (sample.get("reference_answer") or "").strip()
    return source_excerpt or reference_answer or ""


def build_result_status(result):
    """根据 evaluation_track 构建结果状态显示信息。

    Returns:
        dict: {
            "icon": str,           # 图标
            "title": str,          # 标题文案
            "status": str,         # 状态标识
            "description": str,    # 状态描述
        }
    """
    track = result.get("evaluation_track", "")
    t1 = result.get("retrieval_top1_hit")
    t3 = result.get("retrieval_top3_hit")
    t5 = result.get("retrieval_top5_hit")
    answer_correct = result.get("answer_correct")

    if track == TRACK_RETRIEVAL:
        # 检索评测：显示 TopK 命中状态
        parts = []
        if t1 is not None:
            parts.append(f"Top1 {'命中' if t1 else '未命中'}")
        if t3 is not None:
            parts.append(f"Top3 {'命中' if t3 else '未命中'}")
        if t5 is not None:
            parts.append(f"Top5 {'命中' if t5 else '未命中'}")

        hit_summary = "｜".join(parts) if parts else "无检索结果"
        return {
            "icon": "🔍",
            "title": hit_summary,
            "status": "retrieval",
            "description": "检索命中评测",
        }
    elif track == TRACK_STRICT_QA:
        # 严格问答：显示回答正确性
        if answer_correct:
            return {
                "icon": "✅",
                "title": "回答正确",
                "status": "correct",
                "description": "与参考答案一致",
            }
        else:
            return {
                "icon": "❌",
                "title": "回答错误",
                "status": "incorrect",
                "description": "与参考答案不一致或遗漏关键点",
            }
    elif track == TRACK_GROUNDED_QA:
        # 合理性问答：显示回答有据性
        if answer_correct:
            return {
                "icon": "✅",
                "title": "回答有据",
                "status": "grounded",
                "description": "回答被检索内容支持",
            }
        else:
            return {
                "icon": "⚠️",
                "title": "回答缺乏依据",
                "status": "not_grounded",
                "description": "回答未被检索内容充分支持",
            }
    elif track == TRACK_NOT_EVALUABLE:
        # 不可评测
        return {
            "icon": "⚠️",
            "title": "缺少金标准证据",
            "status": "not_evaluable",
            "description": "无法可靠计算检索命中率",
        }
    else:
        # 未知轨道
        return {
            "icon": "❓",
            "title": "未知评测类型",
            "status": "unknown",
            "description": "",
        }


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


def build_judge_prompt(sample, template=None, max_content_chars=None, evaluation_track=None):
    """构建 Judge prompt。根据评测轨道自动选择合适的模板。

    Args:
        sample: 样本数据字典
        template: 自定义 prompt 模板（可选）
        max_content_chars: 统一字符上限。为 None 时使用分层策略（Top-1: 2000, Top-2~5: 1000~1200）。
        evaluation_track: 评测轨道（可选，如不传则自动判断）
    """
    # 确定评测轨道
    if evaluation_track is None:
        evaluation_track = classify_evaluation_track(sample)

    # 根据评测轨道选择模板
    if template is not None:
        pass  # 使用传入的模板
    elif evaluation_track == TRACK_RETRIEVAL:
        template = load_prompt_template_retrieval()
    elif evaluation_track == TRACK_STRICT_QA:
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

    # 检索评测专用占位符
    if evaluation_track == TRACK_RETRIEVAL:
        gold_evidence = get_gold_evidence(sample)
        prompt = prompt.replace("{gold_evidence}", gold_evidence or "(无金标准证据)")
    else:
        # 严格/合理性问答模板的占位符
        has_ref = bool((sample.get("reference_answer") or "").strip())
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


# retrieval 轨道 position → (top1, top3, top5) 的唯一合法映射
_VALID_POSITION_TOPK = {
    1: (1, 1, 1),
    2: (0, 1, 1),
    3: (0, 1, 1),
    4: (0, 0, 1),
    5: (0, 0, 1),
}


def validate_retrieval_judge_output(scores, num_retrieval_results):
    """严格校验 retrieval Judge 的 LLM 输出。

    校验项：
    - top1/top3/top5 必须是整数 0 或 1
    - hit_evidence_position 必须是 null 或 [1, min(5, num_retrieval_results)] 的整数
    - position 与 TopK 的一致性
    - 丢弃 answer_correct（retrieval 轨道不应有此字段）

    不符合时抛出 ValueError，由调用方捕获并记为 judge error。
    """
    # 基本类型校验（bool 是 int 子类，需显式拒绝）
    for key in ("retrieval_top1_hit", "retrieval_top3_hit", "retrieval_top5_hit"):
        val = scores.get(key)
        if isinstance(val, bool) or val not in (0, 1):
            raise ValueError(
                f"retrieval Judge 输出格式错误：{key} 必须为整数 0 或 1，实际为 {val!r}"
            )

    pos = scores.get("hit_evidence_position")
    max_pos = min(5, num_retrieval_results) if num_retrieval_results > 0 else 0

    # position 值域校验
    if pos is not None:
        if not isinstance(pos, int) or isinstance(pos, bool):
            raise ValueError(
                f"retrieval Judge 输出格式错误：hit_evidence_position 必须为 null 或整数，实际为 {pos!r}"
            )
        if max_pos == 0:
            raise ValueError(
                f"retrieval Judge 输出格式错误：无检索结果时 hit_evidence_position 必须为 null，实际为 {pos}"
            )
        if pos < 1 or pos > max_pos:
            raise ValueError(
                f"retrieval Judge 输出格式错误：hit_evidence_position 越界，"
                f"有效范围 [1, {max_pos}]，实际为 {pos}"
            )

    t1 = scores["retrieval_top1_hit"]
    t3 = scores["retrieval_top3_hit"]
    t5 = scores["retrieval_top5_hit"]

    # position ↔ TopK 一致性校验
    if pos is None:
        if (t1, t3, t5) != (0, 0, 0):
            raise ValueError(
                f"retrieval Judge 输出格式错误：position=null 时 TopK 应为 (0,0,0)，"
                f"实际为 ({t1},{t3},{t5})"
            )
    else:
        expected = _VALID_POSITION_TOPK.get(pos)
        if expected is None:
            raise ValueError(
                f"retrieval Judge 输出格式错误：position={pos} 无合法 TopK 映射"
            )
        if (t1, t3, t5) != expected:
            raise ValueError(
                f"retrieval Judge 输出格式错误：position={pos} 时 TopK 应为 {expected}，"
                f"实际为 ({t1},{t3},{t5})"
            )

    # 丢弃 answer_correct：retrieval 轨道不应混入 QA 指标字段
    scores.pop("answer_correct", None)


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
        err = RuntimeError(
            f"HTTP {resp.status_code} | URL: {url}\nResponse: {resp.text[:1000]}"
        )
        err.status_code = resp.status_code
        err.retry_after = resp.headers.get("Retry-After")
        raise err

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"JSON 解析失败 | Response: {resp.text[:1000]}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"响应结构异常 | Response: {json.dumps(data, ensure_ascii=False)[:1000]}")


def _retry_call_llm(prompt, api_key, base_url, model, timeout=30,
                    max_retries=3, base_delay=1.0):
    """带指数退避的 call_llm 包装。429/5xx/超时/连接失败可重试。"""

    def _is_retryable(exc):
        if not hasattr(exc, 'status_code'):
            msg = str(exc)
            return "请求超时" in msg or "连接失败" in msg
        code = exc.status_code
        return code == 429 or (500 <= code < 600)

    for attempt in range(max_retries + 1):
        try:
            return call_llm(prompt, api_key, base_url, model, timeout=timeout)
        except RuntimeError as exc:
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            retry_after = getattr(exc, 'retry_after', None)
            try:
                ra_delay = float(retry_after) if retry_after else 0
            except (ValueError, TypeError):
                ra_delay = 0
            delay = max(base_delay * (2 ** attempt), ra_delay)
            time.sleep(delay)


def judge_sample(sample, api_key, base_url, model, prompt_template=None, timeout=60):
    """Judge a single sample. Returns a dict with scores or error info."""
    trace_id = sample.get("trace_id", "unknown")
    has_ref = bool((sample.get("reference_answer") or "").strip())
    question_mode = (sample.get("question_mode") or "").strip()
    evaluation_track = classify_evaluation_track(sample)

    result = {
        "trace_id": trace_id,
        "question": sample.get("question") or "",
        "question_mode": question_mode,
        "has_reference": has_ref,
        "evaluation_track": evaluation_track,
        "retrieval_evaluable": evaluation_track == TRACK_RETRIEVAL,
    }

    # 透传元数据字段（不得改变评分逻辑）
    for meta_key in ("run_id", "config_id", "question_id", "question_set_id", "question_set_name"):
        val = sample.get(meta_key)
        if val:
            result[meta_key] = val

    # 不可评测的检索题：缺少金标准证据
    if evaluation_track == TRACK_NOT_EVALUABLE:
        result["retrieval_top1_hit"] = 0
        result["retrieval_top3_hit"] = 0
        result["retrieval_top5_hit"] = 0
        result["answer_correct"] = 0
        result["not_evaluable_reason"] = "检索评测题缺少金标准证据（source_excerpt 和 reference_answer 均为空）"
        result["reason"] = "不可评测：缺少金标准证据，无法可靠计算检索命中率"
        return result

    try:
        prompt = build_judge_prompt(sample, prompt_template, evaluation_track=evaluation_track)
        result["_prompt"] = prompt
        response_text = _retry_call_llm(prompt, api_key, base_url, model, timeout=timeout)
        result["_raw_response"] = response_text
        scores = parse_judge_response(response_text)
        # retrieval 轨道：严格校验 LLM 输出格式
        if evaluation_track == TRACK_RETRIEVAL:
            num_results = len(sample.get("retrieval_results") or [])
            validate_retrieval_judge_output(scores, num_results)
        result.update(scores)
    except Exception as e:
        result["error"] = str(e)

    return result


def judge_all(samples, api_key, base_url, model, progress_callback=None,
              timeout=60, max_workers=1):
    """Judge all samples. Yields results one by one in original order.

    内置规则预筛选和内容级去重：
    - 预筛选：无检索结果/无回答的样本直接出结果，不调 LLM
    - 去重：相同 question+retrieval_query+final_answer 的样本只评一次

    max_workers > 1 时，仅对需要 LLM 调用的样本使用 ThreadPoolExecutor 并行。
    max_workers == 1 时行为与旧串行逻辑完全一致。

    模板选择：根据 evaluation_track 自动选择对应模板。

    progress_callback(done, total, result, info) 在每个样本完成时立即调用：
    - done: 已完成样本数（含预筛、去重、LLM）
    - total: 总样本数
    - result: 本次完成的结果 dict
    - info: {"llm_done", "llm_total", "elapsed", "eta_text", "throughput",
             "prescreened_count", "cached_count", "concurrency"}
    """
    content_cache = {}  # content_hash -> result dict (without trace_id/question)
    total = len(samples)
    start_time = time.monotonic()

    # 元数据字段列表（需要从 sample 透传到 result）
    META_KEYS = ("run_id", "config_id", "question_id", "question_set_id", "question_set_name",
                 "source_file_name", "source_format")

    _CACHE_EXCLUDE = {
        "trace_id", "question", "question_mode",
        "evaluation_track", "retrieval_evaluable",
        "has_reference", "_prompt", "_raw_response",
        *META_KEYS,
    }

    def _build_immediate(sample, extra_flags, sample_meta):
        return {
            "trace_id": sample.get("trace_id", "unknown"),
            "question": sample.get("question") or "",
            "question_mode": (sample.get("question_mode") or "").strip(),
            "evaluation_track": sample["evaluation_track"],
            **extra_flags,
            **sample_meta,
        }

    # --- Phase 1: classify, pre-screen, content dedup (sequential) ---
    immediate_results = {}  # idx -> result
    llm_queue = []          # (idx, sample)
    prescreened_count = 0
    cached_count = 0
    done = 0

    for i, sample in enumerate(samples):
        if "evaluation_track" not in sample:
            sample["evaluation_track"] = classify_evaluation_track(sample)
        sample_meta = {k: sample.get(k) for k in META_KEYS if sample.get(k)}

        prescreened = pre_screen(sample)
        if prescreened is not None:
            result = _build_immediate(
                sample, {"_prescreened": True, **prescreened}, sample_meta)
            immediate_results[i] = result
            prescreened_count += 1
            done += 1
            if progress_callback:
                elapsed = time.monotonic() - start_time
                info = {
                    "llm_done": 0, "llm_total": 0,
                    "elapsed": elapsed, "eta_text": "计算中",
                    "throughput": 0.0,
                    "prescreened_count": prescreened_count,
                    "cached_count": cached_count,
                    "concurrency": max_workers,
                }
                progress_callback(done, total, result, info)
            continue

        ch = compute_content_hash(sample)
        if ch in content_cache:
            cached = content_cache[ch]
            result = _build_immediate(
                sample, {"_content_cached": True, **cached}, sample_meta)
            immediate_results[i] = result
            cached_count += 1
            done += 1
            if progress_callback:
                elapsed = time.monotonic() - start_time
                info = {
                    "llm_done": 0, "llm_total": 0,
                    "elapsed": elapsed, "eta_text": "计算中",
                    "throughput": 0.0,
                    "prescreened_count": prescreened_count,
                    "cached_count": cached_count,
                    "concurrency": max_workers,
                }
                progress_callback(done, total, result, info)
            continue

        llm_queue.append((i, sample))

    llm_total = len(llm_queue)

    # --- Phase 2: LLM calls ---
    llm_results = {}  # idx -> result
    llm_done = 0
    llm_start_time = None
    llm_elapsed_sum = 0.0

    def _make_llm_info():
        """构建当前 LLM 进度 info dict。"""
        elapsed = time.monotonic() - start_time
        if llm_done >= 2 and llm_elapsed_sum > 0:
            avg_time = llm_elapsed_sum / llm_done
            remaining = llm_total - llm_done
            eta_secs = avg_time * remaining
            if eta_secs < 60:
                eta_text = f"约 {int(eta_secs)} 秒"
            else:
                eta_text = f"约 {int(eta_secs // 60)} 分 {int(eta_secs % 60)} 秒"
            throughput = llm_done / llm_elapsed_sum
        elif llm_done > 0:
            eta_text = "计算中"
            throughput = 0.0
        else:
            eta_text = "计算中"
            throughput = 0.0
        return {
            "llm_done": llm_done, "llm_total": llm_total,
            "elapsed": elapsed, "eta_text": eta_text,
            "throughput": throughput,
            "prescreened_count": prescreened_count,
            "cached_count": cached_count,
            "concurrency": max_workers,
        }

    def _on_llm_result(idx, result):
        """LLM 结果完成后的统一处理：缓存、计数、回调。"""
        nonlocal llm_done, llm_start_time, llm_elapsed_sum, done
        ch = compute_content_hash(llm_queue_map[idx])
        if "error" not in result:
            content_cache[ch] = {
                k: v for k, v in result.items() if k not in _CACHE_EXCLUDE
            }
        llm_results[idx] = result
        llm_done += 1
        done += 1
        if llm_start_time is not None:
            llm_elapsed_sum = time.monotonic() - llm_start_time
        if progress_callback:
            progress_callback(done, total, result, _make_llm_info())

    if llm_queue:
        def _worker(idx, sample):
            sample_meta = {k: sample.get(k) for k in META_KEYS if sample.get(k)}
            try:
                result = judge_sample(sample, api_key, base_url, model, timeout=timeout)
            except Exception as e:
                result = {
                    "trace_id": sample.get("trace_id", "unknown"),
                    "question": sample.get("question") or "",
                    "evaluation_track": sample.get("evaluation_track", ""),
                    "error": str(e),
                }
            for k, v in sample_meta.items():
                result.setdefault(k, v)
            return idx, result

        llm_queue_map = {idx: sample for idx, sample in llm_queue}

        if max_workers <= 1:
            for idx, sample in llm_queue:
                ch = compute_content_hash(sample)
                if ch in content_cache:
                    sample_meta = {k: sample.get(k) for k in META_KEYS if sample.get(k)}
                    cached = content_cache[ch]
                    result = {
                        "trace_id": sample.get("trace_id", "unknown"),
                        "question": sample.get("question") or "",
                        "question_mode": (sample.get("question_mode") or "").strip(),
                        "evaluation_track": sample.get("evaluation_track", ""),
                        "_content_cached": True,
                        **cached,
                        **sample_meta,
                    }
                    llm_results[idx] = result
                    llm_done += 1
                    done += 1
                    if progress_callback:
                        progress_callback(done, total, result, _make_llm_info())
                    continue
                if llm_start_time is None:
                    llm_start_time = time.monotonic()
                t0 = time.monotonic()
                _, result = _worker(idx, sample)
                llm_elapsed_sum += time.monotonic() - t0
                if "error" not in result:
                    content_cache[ch] = {
                        k: v for k, v in result.items() if k not in _CACHE_EXCLUDE
                    }
                llm_results[idx] = result
                llm_done += 1
                done += 1
                if progress_callback:
                    progress_callback(done, total, result, _make_llm_info())
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_worker, idx, sample): idx
                    for idx, sample in llm_queue
                }
                llm_start_time = time.monotonic()
                for future in as_completed(futures):
                    idx, result = future.result()
                    _on_llm_result(idx, result)

    # --- Phase 3: yield in original order (no progress callback) ---
    for i in range(total):
        if i in immediate_results:
            yield immediate_results[i]
        elif i in llm_results:
            yield llm_results[i]


def pre_screen(sample):
    """规则预筛选：对结果确定的样本直接返回评分，不需要调用 LLM。

    返回 None 表示无法用规则判定，需要走 LLM。
    返回 dict 表示已有确定结果，可直接使用。
    """
    question = (sample.get("question") or "").strip()
    final_answer = (sample.get("final_answer") or "").strip()
    retrieval_results = sample.get("retrieval_results") or []
    evaluation_track = classify_evaluation_track(sample)

    # 不可评测的检索题：缺少金标准证据
    if evaluation_track == TRACK_NOT_EVALUABLE:
        return {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                "retrieval_top5_hit": 0, "answer_correct": 0,
                "retrieval_evaluable": False,
                "not_evaluable_reason": "检索评测题缺少金标准证据",
                "reason": "不可评测：缺少金标准证据，无法可靠计算检索命中率"}

    # 无问题 → 无法评测
    if not question:
        return {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                "retrieval_top5_hit": 0, "answer_correct": 0,
                "reason": "规则判定：无用户问题"}

    # 无检索结果 → top 全 0
    no_retrieval = len(retrieval_results) == 0

    # 检索轨道：仅当检索结果为空时才预筛；final_answer 为空不影响检索命中判断
    if evaluation_track == TRACK_RETRIEVAL:
        if no_retrieval:
            return {"retrieval_top1_hit": 0, "retrieval_top3_hit": 0,
                    "retrieval_top5_hit": 0, "hit_evidence_position": None,
                    "reason": "规则判定：无检索结果"}
        # 有检索结果 → 交给 LLM 判断，不管 final_answer 是否为空
        return None

    # QA 轨道：保持原有逻辑
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

    按评测轨道分组计算指标，避免混合不同口径。
    """
    valid = [r for r in results if "error" not in r]
    total = len(results)
    errored = total - len(valid)

    # 按评测轨道分组
    retrieval_tracks = [r for r in valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
    strict_qa_tracks = [r for r in valid if r.get("evaluation_track") == TRACK_STRICT_QA]
    grounded_qa_tracks = [r for r in valid if r.get("evaluation_track") == TRACK_GROUNDED_QA]
    not_evaluable_tracks = [r for r in valid if r.get("evaluation_track") == TRACK_NOT_EVALUABLE]

    # 检索评测指标（仅有金标准证据的）
    retrieval_evaluable = [r for r in retrieval_tracks if r.get("retrieval_evaluable", True)]
    retrieval_n = len(retrieval_evaluable)

    # 严格问答指标
    strict_qa_n = len(strict_qa_tracks)

    # 合理性问答指标
    grounded_qa_n = len(grounded_qa_tracks)

    # 不可评测样本数
    not_evaluable_n = len(not_evaluable_tracks)

    # 兼容旧版指标（混合口径，仅供参考）
    valid_n = len(valid)
    with_ref = [r for r in valid if r.get("has_reference")]
    without_ref = [r for r in valid if not r.get("has_reference")]
    with_ref_n = len(with_ref)
    without_ref_n = len(without_ref)

    metrics = {
        "total": total,
        "evaluated": valid_n,
        "errors": errored,

        # 检索评测指标
        "retrieval_track_count": retrieval_n,
        "retrieval_top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in retrieval_evaluable) / retrieval_n if retrieval_n else None,
        "retrieval_top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in retrieval_evaluable) / retrieval_n if retrieval_n else None,
        "retrieval_top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in retrieval_evaluable) / retrieval_n if retrieval_n else None,
        "retrieval_not_evaluable_count": not_evaluable_n,

        # 严格问答指标
        "strict_qa_track_count": strict_qa_n,
        "strict_qa_answer_rate": sum(r.get("answer_correct", 0) for r in strict_qa_tracks) / strict_qa_n if strict_qa_n else None,

        # 合理性问答指标
        "grounded_qa_track_count": grounded_qa_n,
        "grounded_qa_answer_rate": sum(r.get("answer_correct", 0) for r in grounded_qa_tracks) / grounded_qa_n if grounded_qa_n else None,

        # 兼容旧版（混合口径，仅供参考）
        "with_ref_count": with_ref_n,
        "without_ref_count": without_ref_n,
        "has_reference_data": with_ref_n > 0,
        "top1_hit_rate": sum(r.get("retrieval_top1_hit", 0) for r in valid) / valid_n if valid_n else None,
        "top3_hit_rate": sum(r.get("retrieval_top3_hit", 0) for r in valid) / valid_n if valid_n else None,
        "top5_hit_rate": sum(r.get("retrieval_top5_hit", 0) for r in valid) / valid_n if valid_n else None,
        "answer_correct_rate": sum(r.get("answer_correct", 0) for r in valid) / valid_n if valid_n else None,
        "with_ref_answer_rate": sum(r.get("answer_correct", 0) for r in with_ref) / with_ref_n if with_ref_n else None,
        "without_ref_answer_rate": sum(r.get("answer_correct", 0) for r in without_ref) / without_ref_n if without_ref_n else None,
    }

    return metrics
