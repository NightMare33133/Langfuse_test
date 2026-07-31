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


def generate_default_set_name_for_dataset(dataset_name):
    """生成基于知识库名称的默认题集名称。

    格式：{知识库名称}-chunk_exact-{YYYYMMDD}

    Args:
        dataset_name: 知识库名称

    Returns:
        str: 生成的名称
    """
    today = datetime.now().strftime("%Y%m%d")
    name = (dataset_name or "未知知识库").strip()
    return f"{name}-chunk_exact-{today}"


# ── 多文档联合出题 ──────────────────────────────────────────────


def validate_multi_doc_config(doc_configs):
    """校验多文档出题配置。

    Args:
        doc_configs: list[dict]，每项包含 document_id, document_name,
                     candidates (list), num_questions (int)

    Returns:
        (ok, errors): ok=True 表示全部通过，errors 为错误列表
    """
    errors = []
    active_configs = [dc for dc in doc_configs if dc.get("num_questions", 0) > 0]

    if not active_configs:
        errors.append("没有选择任何文档或所有文档生成题数为 0")
        return False, errors

    for dc in active_configs:
        doc_name = dc.get("document_name", dc.get("document_id", "未知"))
        num = dc.get("num_questions", 0)
        avail = len(dc.get("candidates", []))
        if num > avail:
            errors.append(
                f"文档「{doc_name}」需要 {num} 题，但仅有 {avail} 个可用候选 chunk"
            )

    return len(errors) == 0, errors


def generate_chunk_exact_questions_multi_doc(
    doc_configs, api_key, base_url, model,
    dataset_id="", snapshot_id="",
    timeout=60, progress_callback=None,
):
    """从多个文档的候选 chunk 联合生成 chunk_exact 题集。

    每个文档独立从自身完整可用 chunk catalog 中采样对应数量，
    禁止跨文档抢占、禁止重复 chunk。

    Args:
        doc_configs: list[dict]，每项包含：
            - document_id: 文档 ID
            - document_name: 文档名称
            - candidates: 该文档的候选 chunk 列表（已过滤）
            - num_questions: 该文档要生成的题数
        api_key: LLM API Key
        base_url: LLM API Base URL
        model: LLM 模型名
        dataset_id: 知识库 ID
        snapshot_id: 快照 ID（可选，自动生成）
        timeout: LLM 调用超时
        progress_callback: 进度回调 (done, total, message)

    Returns:
        list[dict]: question dicts
    """
    # 校验配置
    ok, errors = validate_multi_doc_config(doc_configs)
    if not ok:
        raise ValueError("多文档配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

    active_configs = [dc for dc in doc_configs if dc.get("num_questions", 0) > 0]

    if not snapshot_id:
        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    total_questions = sum(dc["num_questions"] for dc in active_configs)

    if progress_callback:
        progress_callback(0, total_questions,
                          f"准备从 {len(active_configs)} 个文档生成 {total_questions} 道题...")

    # 每个文档独立采样
    all_sampled = []
    for dc in active_configs:
        candidates = dc["candidates"]
        num = dc["num_questions"]
        doc_id = dc["document_id"]
        doc_name = dc.get("document_name", "")

        # 使用文档 ID 作为种子的一部分，确保可复现
        sampled, actual_count, capped = sample_candidates_random(
            candidates, num, seed=hash(doc_id) % (2**31)
        )

        # 标记来源文档
        for s in sampled:
            s["_source_document_id"] = doc_id
            s["_source_document_name"] = doc_name

        all_sampled.extend(sampled)

    if not all_sampled:
        raise ValueError("没有可用的候选 chunk")

    # 构建候选集索引
    candidate_map = {c["segment_id"]: c for c in all_sampled}
    candidate_ids = set(candidate_map.keys())

    # 构建 prompt
    candidates_text = _build_candidates_text(all_sampled)
    prompt = CHUNK_EXACT_PROMPT.replace("{candidates_text}", candidates_text)

    if progress_callback:
        progress_callback(0, total_questions, "正在调用 LLM 生成检索查询...")

    # 调用 LLM
    response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)

    if progress_callback:
        progress_callback(0, total_questions, "正在解析和校验...")

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

            # 从候选 chunk 获取文档信息
            _dataset_id = dataset_id or candidate.get("dataset_id", "")
            _document_id = candidate.get("_source_document_id",
                                          candidate.get("document_id", ""))
            _document_name = candidate.get("_source_document_name",
                                            candidate.get("document_name", ""))
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
                "document_name": _document_name,
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
    questions = questions[:total_questions]

    if progress_callback:
        progress_callback(total_questions, total_questions,
                          f"生成完成: {len(questions)} 道题")

    return questions


# ── LLM 生成检索查询 ─────────────────────────────────────────

CHUNK_EXACT_PROMPT = """你是 RAG 检索评测出题专家。根据以下候选知识片段，为每个片段生成一条短检索查询。

**核心目标：** 测试 RAG 系统能否准确检索到目标 chunk（Top1/Top3/Top5 命中率）。每条查询对应一个独立知识点，该知识点可在目标 chunk 内独立表达。

**重要：** 检索评测测的是"RAG 能否从知识库召回正确的 chunk"，不是让 LLM 回答问题。你生成的不是问答题，而是**短检索查询**。

**候选片段列表：**
{candidates_text}

---

## 核心原则

**每条查询必须满足：**
1. **单一概念** — 只对应目标 chunk 中一个明确的、可独立表达的知识点
2. **单 chunk 证据** — 该知识点可从目标 chunk 内完整找到，不需要跨 chunk 聚合
3. **无需推理** — 知识点是原文的直接陈述，不需要归纳、推导或对比
4. **可为近义改写** — 查询可以与原文措辞不同，但只能是单一概念的近义/语义改写

---

## retrieval_query（检索查询）规范

**query 必须是短检索查询，可为词、词组、短语或单一检索意图：**
- ✅ 可以是：名词、名词短语、动宾短语、"修饰语+核心概念"
- ✅ 可以与原文措辞不同（近义改写），但只能是单一概念
- ✅ **优先采用同义词、语序调整或等价表述**，以测试 embedding 的语义检索能力
- ✅ 术语类查询允许（如"RAG 技术框架"），但不要让所有题都只是"XX 定义"
- ❌ **不得是问句** — 禁止出现问号（？/ ?）
- ❌ **禁止问答或推理导向表达** — 不得包含"什么/为何/为什么/如何/是否/哪些/请分析/请说明/分别"等词
- ❌ **禁止多子问题** — 不得包含比较、归因、总结、影响分析、步骤分析等需要跨 chunk 聚合的内容
- ❌ **不得逐字照抄** chunk 中的连续标题或原文作为查询 — 专有名词、术语除外；应通过语义改写生成查询
- ❌ **不得引入文中不存在的信息**

---

## 禁止的查询类型

- ❌ **对比/区别类** — "A 和 B 的区别"（需要跨 chunk 信息）
- ❌ **优缺点分析类** — "XXX 优缺点"（需要多处收集）
- ❌ **原因分析类** — "为什么 XXX"（需要推理）
- ❌ **影响/意义类** — "XXX 影响"（需要综合分析）
- ❌ **开放式/问答类** — "如何理解 XXX"、"XXX 是什么"（问句形式）

---

## target_label 规范

- target_label 仅用于预览展示，**不是查询本体**
- 简短标签（3-8 字），概括查询指向的知识点
- 例如："RAG 框架定义"、"故障通报义务"、"认证宽限期"

---

## 输出格式（严格 JSON 数组）

```json
[
  {{"candidate_id": "候选ID", "retrieval_query": "短检索查询（非问句）", "target_label": "简短标签"}},
  ...
]
```

**禁止输出：**
- ❌ reference_answer — chunk_exact 不使用参考答案
- ❌ expected_segment_id / expected_content_hash — 由系统自动绑定
- ❌ 证据文本或原文引用
- ❌ 任何非 JSON 内容

---

## 自检清单（每条查询必须通过）

- [ ] **非问句** — retrieval_query 中不含问号、不含"什么/为何/为什么/如何/是否/哪些/请分析/请说明/分别"等问答导向词？
- [ ] **语义改写** — retrieval_query 是否采用了同义词/语序调整/等价表述，而非逐字照抄 chunk 中的连续标题或原文？
- [ ] **单一概念** — 是否只对应目标 chunk 中一个知识点？（不是跨 chunk 对比）
- [ ] **单 chunk 可答** — 该知识点是否完全在目标 chunk 内，不需要其他 chunk 信息？

**任何一项不通过，请跳过该候选，不要凑数。**

---

## 示例

### ✅ 正确（同义改写）
```
chunk 内容："RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的技术框架。"
retrieval_query: RAG 技术框架定义
target_label: RAG 定义
说明："RAG 技术框架定义"是对原文的同义概括，非逐字照抄
```

### ✅ 正确（语序调整）
```
chunk 内容："如果供应方未获得ISO9001认证，则享有自协议签署之日起六个月的宽限期。"
retrieval_query: 质量管理体系认证宽限期
target_label: 认证宽限期
说明："质量管理体系认证"是对"ISO9001认证"的同义表述
```

### ❌ 错误（问句）
```
retrieval_query: "RAG 技术框架指的是什么？"
原因：问句形式，含"什么"和问号
```

### ❌ 错误（逐字照抄）
```
chunk 内容："缺陷指一项或多项指定服务不符合要求"
retrieval_query: "缺陷指一项或多项指定服务不符合要求"
原因：逐字照抄原文，应改为"服务缺陷定义"等同义概括
```

### ❌ 错误（跨 chunk 聚合）
```
retrieval_query: "RAG 与传统问答系统区别"
原因：需要分别找到 RAG 和传统问答系统的信息，属于多跳
```

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
                               random_seed=None, doc_question_counts=None):
    """保存 chunk_exact 题集。

    复用 question_generator.save_questions()，写入 chunk_exact 元数据。

    Args:
        questions: 题目列表
        question_set_name: 题集名称
        dataset_id: 知识库 ID
        document_id: 文档 ID（单文档时使用）
        snapshot_id: 快照 ID
        selection_mode: "manual" 或 "random"
        selected_document_ids: 选中的文档 ID 列表
        random_seed: 随机种子
        doc_question_counts: dict[doc_id -> question_count]，多文档各文档题数
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
        if doc_question_counts:
            manifest["doc_question_counts"] = doc_question_counts
        if random_seed is not None:
            manifest["random_seed"] = random_seed
        # 每题的 expected_segment_id 和 expected_content_hash 已在 question dict 中
        manifest_path.write_text(_json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path, filename, qs_id
