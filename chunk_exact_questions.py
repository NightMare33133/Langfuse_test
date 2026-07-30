"""
从 Chunk Catalog 创建 chunk_exact 题集。

流程：
1. 从已加载的 Chunk Catalog 过滤候选 chunk
2. 调用 LLM 生成短检索查询（LLM 只输出 candidate_id, retrieval_query, target_label）
3. Fail-closed 校验 candidate_id 必须对应当前候选 chunk
4. 构建 question dict，绑定 expected_segment_id / expected_content_hash / dataset_id / document_id / snapshot_id
5. 通过 question_generator.save_questions() 保存题集

安全规则：
- LLM 不输出 reference_answer、expected segment_id 或证据文本
- API Key 不写入任何输出文件
"""

import json
import random
import re
import uuid
from datetime import datetime

from judge import call_llm
from dify_knowledge import compute_content_hash
from question_generator import save_questions


# ── 候选 chunk 过滤 ──────────────────────────────────────────

# 排除模式：纯标题、页码、签字页、目录等
_EXCLUDE_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s+\S+\s*$"),                    # 纯 Markdown 标题行
    re.compile(r"^\s*第?\s*\d+\s*页?\s*$"),                   # 纯页码
    re.compile(r"^\s*(?:签字|签名|签章|盖章)\s*[：:]\s*$"),     # 签字页
    re.compile(r"^\s*(?:目录|目\s*录)\s*$"),                   # 目录标题
    re.compile(r"^\s*(?:附录|附件)\s*[A-Z\d]?\s*$"),          # 附录标题
    re.compile(r"^\s*\d+(?:\.\d+)*\s+\S{1,10}\s*$"),         # 短编号标题（如 "1.2.3 概述"）
]

MIN_CONTENT_LENGTH = 20  # 最少字符数


def filter_candidate_chunks(catalog, duplicates=None):
    """从 Chunk Catalog 过滤候选 chunk。

    过滤条件：
    - status == "completed"
    - enabled == true
    - content_hash 不在 duplicates 集合中
    - content 长度 >= MIN_CONTENT_LENGTH
    - 不匹配排除模式（纯标题/页码/签字页等）

    Args:
        catalog: build_chunk_catalog() 返回的 list[dict]
        duplicates: detect_duplicates() 返回的 dict（可选）

    Returns:
        (candidates, filter_stats)
        candidates: 过滤后的候选列表
        filter_stats: {"total", "passed", "filtered": {reason: count}}
    """
    dup_hashes = set(duplicates.keys()) if duplicates else set()
    total = len(catalog)
    filtered_reasons = {}
    candidates = []

    for entry in catalog:
        # 状态过滤
        if entry.get("status") != "completed":
            filtered_reasons["status_not_completed"] = filtered_reasons.get("status_not_completed", 0) + 1
            continue

        # 启用状态过滤
        if not entry.get("enabled", True):
            filtered_reasons["disabled"] = filtered_reasons.get("disabled", 0) + 1
            continue

        # 重复过滤
        if entry.get("content_hash") in dup_hashes:
            filtered_reasons["duplicate"] = filtered_reasons.get("duplicate", 0) + 1
            continue

        content = (entry.get("content") or "").strip()

        # 空内容过滤
        if not content:
            filtered_reasons["empty"] = filtered_reasons.get("empty", 0) + 1
            continue

        # 长度过滤
        if len(content) < MIN_CONTENT_LENGTH:
            filtered_reasons["too_short"] = filtered_reasons.get("too_short", 0) + 1
            continue

        # 纯标题/页码/签字页过滤
        if any(pat.match(content) for pat in _EXCLUDE_PATTERNS):
            filtered_reasons["title_or_page"] = filtered_reasons.get("title_or_page", 0) + 1
            continue

        candidates.append(entry)

    stats = {
        "total": total,
        "passed": len(candidates),
        "filtered": filtered_reasons,
    }
    return candidates, stats


def get_candidates_by_documents(candidates, document_ids):
    """按文档 ID 过滤候选 chunk。

    Args:
        candidates: filter_candidate_chunks() 返回的候选列表
        document_ids: 要包含的文档 ID 列表

    Returns:
        过滤后的候选列表
    """
    if not document_ids:
        return candidates
    doc_set = set(document_ids)
    return [c for c in candidates if c.get("document_id", "") in doc_set]


def sample_candidates_random(candidates, num_questions, document_ids=None, seed=None):
    """从候选 chunk 中无重复随机抽取 N 个。

    Args:
        candidates: filter_candidate_chunks() 返回的候选列表
        num_questions: 需要抽取的数量
        document_ids: 可选，限定文档 ID 列表
        seed: 可选随机种子，用于复现

    Returns:
        (sampled, actual_count, capped)
        sampled: 抽取后的候选列表
        actual_count: 实际抽取数量
        capped: bool，是否因候选不足而被截断
    """
    if document_ids:
        candidates = get_candidates_by_documents(candidates, document_ids)

    available = len(candidates)
    if available == 0:
        return [], 0, False

    capped = False
    if num_questions >= available:
        actual_count = available
        capped = True
    else:
        actual_count = num_questions

    rng = random.Random(seed) if seed is not None else random
    sampled = rng.sample(candidates, actual_count)
    return sampled, actual_count, capped


def generate_default_set_name(document_names, mode="random"):
    """生成默认题集名称。

    Args:
        document_names: 文档名称列表
        mode: "random" 或 "manual"

    Returns:
        str: 生成的名称
    """
    today = datetime.now().strftime("%Y%m%d")
    if mode == "manual":
        return f"chunk_exact_{today}"

    if not document_names:
        return f"随机题集-{today}"

    if len(document_names) == 1:
        # 单文档：{原文件名去扩展名}-随机题集-{YYYYMMDD}
        name = document_names[0]
        # 去掉常见扩展名
        for ext in (".txt", ".md", ".docx", ".xlsx", ".pdf", ".csv"):
            if name.lower().endswith(ext):
                name = name[:-len(ext)]
                break
        return f"{name}-随机题集-{today}"
    else:
        # 多文档：随机题集-{文档数量}份文档-{YYYYMMDD}
        return f"随机题集-{len(document_names)}份文档-{today}"


# ── LLM 生成检索查询 ─────────────────────────────────────────

CHUNK_EXACT_PROMPT = """你是 RAG 检索评测出题专家。根据以下候选知识片段，为每个片段生成一条短检索查询。

**你的任务：** 为候选片段生成短检索查询，用于测试 RAG 系统能否准确召回对应的知识片段。

**候选片段列表：**
{candidates_text}

**要求：**
- 为每个候选片段生成 1 条短检索查询
- 查询必须是短语或词组（不是问句），用于 embedding 检索
- 查询应能通过语义检索召回对应片段
- target_label 是该查询的简短标签（3-8 字）

**输出格式（严格 JSON 数组）：**
```json
[
  {{"candidate_id": "候选ID", "retrieval_query": "短检索查询", "target_label": "标签"}},
  ...
]
```

**禁止输出：**
- ❌ reference_answer
- ❌ expected_segment_id
- ❌ 证据文本或原文引用
- ❌ 任何非 JSON 内容

请直接输出 JSON 数组，不要添加其他文字。"""


def _build_candidates_text(candidates, max_content_chars=300):
    """构建送入 LLM 的候选片段文本。"""
    lines = []
    for i, c in enumerate(candidates):
        content = c.get("content", "")
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."
        lines.append(f"候选 {i+1} — ID: {c['segment_id']}\n内容: {content}")
    return "\n\n".join(lines)


def _parse_llm_response(text):
    """解析 LLM 输出的 JSON 数组。"""
    text = text.strip()
    # 尝试提取 JSON 数组
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"LLM 输出不包含 JSON 数组: {text[:200]}")
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {exc}") from exc

    if not isinstance(items, list):
        raise ValueError("LLM 输出不是数组")
    return items


def _validate_candidate_id(item, candidate_ids):
    """Fail-closed 校验 candidate_id。"""
    cid = item.get("candidate_id", "")
    if not cid:
        raise ValueError("LLM 输出缺少 candidate_id")
    if cid not in candidate_ids:
        raise ValueError(f"candidate_id '{cid}' 不在当前候选集中（fail-closed）")
    return cid


CHUNK_EXACT_REQUIRED_FIELDS = [
    "snapshot_id", "document_id", "expected_segment_id", "expected_content_hash",
]


def validate_chunk_exact_question(q):
    """校验 chunk_exact 题目的绑定完整性。

    Returns:
        (ok, errors): ok=True 表示有效，errors 为缺失字段列表
    """
    errors = []
    for field in CHUNK_EXACT_REQUIRED_FIELDS:
        val = (q.get(field) or "").strip() if isinstance(q.get(field), str) else q.get(field)
        if not val:
            errors.append(field)
    return len(errors) == 0, errors


def validate_chunk_exact_set(questions):
    """校验整个 chunk_exact 题集的绑定完整性。

    Returns:
        (valid_questions, invalid_questions)
        invalid_questions 每条附带 _validation_errors 字段
    """
    valid = []
    invalid = []
    for q in questions:
        ok, errors = validate_chunk_exact_question(q)
        if ok:
            valid.append(q)
        else:
            q["_validation_errors"] = errors
            invalid.append(q)
    return valid, invalid


def generate_chunk_exact_questions(
    candidates, api_key, base_url, model,
    num_questions=None, dataset_id="", document_id="", snapshot_id="",
    timeout=60, progress_callback=None,
):
    """从候选 chunk 生成 chunk_exact 题集。

    Args:
        candidates: filter_candidate_chunks() 返回的候选列表
        api_key: LLM API Key
        base_url: LLM API Base URL
        model: LLM 模型名
        num_questions: 生成数量（默认 = len(candidates)）
        dataset_id: 知识库 ID
        document_id: 文档 ID
        snapshot_id: 快照 ID（可选，自动生成）
        timeout: LLM 调用超时
        progress_callback: 进度回调 (done, total, message)

    Returns:
        list[dict]: question dicts，可直接传给 save_questions()
    """
    if not candidates:
        raise ValueError("候选 chunk 列表为空")

    if num_questions is None:
        num_questions = len(candidates)
    num_questions = max(1, min(num_questions, len(candidates)))

    if not snapshot_id:
        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # 构建候选集索引
    candidate_map = {c["segment_id"]: c for c in candidates}
    candidate_ids = set(candidate_map.keys())

    # 构建 prompt
    candidates_text = _build_candidates_text(candidates[:num_questions])
    prompt = CHUNK_EXACT_PROMPT.replace("{candidates_text}", candidates_text)

    if progress_callback:
        progress_callback(0, 1, "正在调用 LLM 生成检索查询...")

    # 调用 LLM
    response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)

    if progress_callback:
        progress_callback(0, 1, "正在解析和校验...")

    # 解析 LLM 输出
    items = _parse_llm_response(response_text)

    # 构建 question dicts
    questions = []
    validation_errors = []

    for item in items:
        try:
            cid = _validate_candidate_id(item, candidate_ids)
            candidate = candidate_map[cid]

            retrieval_query = (item.get("retrieval_query") or "").strip()
            if not retrieval_query:
                raise ValueError(f"candidate_id '{cid}' 缺少 retrieval_query")

            target_label = (item.get("target_label") or "").strip()

            # 构建 question dict（所有字段持久化，供预览/批量提问/机器判定使用）
            _dataset_id = dataset_id or candidate.get("dataset_id", "")
            _document_id = document_id or candidate.get("document_id", "")
            _position = candidate.get("position", "")
            q = {
                "question": retrieval_query,
                "retrieval_query": retrieval_query,
                "question_mode": "chunk_exact",
                "evaluation_type": "chunk_exact",
                "question_id": f"ce_{snapshot_id}_{cid}",
                "target_label": target_label,
                "candidate_id": cid,
                "expected_segment_id": candidate["segment_id"],
                "expected_content_hash": candidate["content_hash"],
                "expected_content": candidate.get("content", "")[:500],
                "dataset_id": _dataset_id,
                "document_id": _document_id,
                "document_name": candidate.get("document_name", ""),
                "snapshot_id": snapshot_id,
                "source_position": _position,
                "source_label": f"doc:{_document_id[:8]} pos:{_position}",
            }
            questions.append(q)

        except ValueError as exc:
            validation_errors.append(str(exc))
            continue

    if not questions:
        raise ValueError(
            f"所有 LLM 输出均校验失败。错误: {'; '.join(validation_errors[:5])}"
        )

    # 截取到目标数量
    questions = questions[:num_questions]

    if progress_callback:
        progress_callback(1, 1, f"生成完成: {len(questions)} 道题")

    return questions


def save_chunk_exact_questions(questions, question_set_name=None,
                               dataset_id="", document_id="", snapshot_id="",
                               selection_mode="manual", selected_document_ids=None,
                               random_seed=None):
    """保存 chunk_exact 题集。

    复用 question_generator.save_questions()，写入 chunk_exact 元数据。
    """
    if not question_set_name:
        ts = datetime.now().strftime("%m%d_%H%M")
        question_set_name = f"chunk_exact_{ts}"

    # 从题目中提取 snapshot_id（如果没有传入）
    if not snapshot_id and questions:
        snapshot_id = questions[0].get("snapshot_id", "")

    output_path, filename, qs_id = save_questions(
        questions,
        question_set_name=question_set_name,
        source_document_name=f"dataset:{dataset_id}" if dataset_id else "",
        question_mode="chunk_exact",
        evaluation_type="chunk_exact",
    )

    # 追加 chunk_exact 专用元数据到 manifest
    manifest_path = output_path.parent / f"{output_path.stem}_manifest.json"
    if manifest_path.exists():
        import json as _json
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation_type"] = "chunk_exact"
        manifest["snapshot_id"] = snapshot_id
        manifest["dataset_id"] = dataset_id
        manifest["document_id"] = document_id
        manifest["selection_mode"] = selection_mode
        if selected_document_ids:
            manifest["selected_document_ids"] = selected_document_ids
        if random_seed is not None:
            manifest["random_seed"] = random_seed
        # 每题的 expected_segment_id 和 expected_content_hash 已在 question dict 中
        manifest_path.write_text(_json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path, filename, qs_id
