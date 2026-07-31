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

import hashlib
import json
import math
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


def _derive_doc_seed(master_seed, document_id):
    """从主种子和文档 ID 派生稳定的整数种子（SHA-256，不使用 Python hash）。

    Args:
        master_seed: 用户指定的主种子（int）
        document_id: 文档 ID

    Returns:
        int: 稳定的 64 位整数种子
    """
    payload = f"{master_seed}:{document_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


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


def sample_candidate_pool(candidates, num_questions, document_id, master_seed):
    """抽取候选池（比 N 大，给 Phase 1 筛选空间）。

    pool_size = min(可用候选数, max(N, ceil(N * 1.5)))

    Args:
        candidates: 该文档的候选 chunk 列表
        num_questions: 目标题数 N
        document_id: 文档 ID（用于种子派生）
        master_seed: 用户主种子

    Returns:
        (pool, pool_size, capped)
        pool: 抽取的候选池
        pool_size: 池大小
        capped: 是否因候选不足而被截断
    """
    available = len(candidates)
    if available == 0:
        return [], 0, False

    target_pool = min(available, max(num_questions, math.ceil(num_questions * 1.5)))
    doc_seed = _derive_doc_seed(master_seed, document_id)
    rng = random.Random(doc_seed)

    capped = target_pool >= available
    pool = rng.sample(candidates, target_pool)
    return pool, target_pool, capped


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
    dataset_id="", snapshot_id="", master_seed=0,
    timeout=60, progress_callback=None,
):
    """从多个文档的候选 chunk 联合生成 chunk_exact 题集（两阶段流程）。

    Phase 1: 按文档调用 LLM 做出题规划（筛选 + query_style + 检索方案）
    Phase 2: 按文档调用 LLM 生成短检索查询（遵从 query_style）

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
        master_seed: 用户主种子（0 表示随机生成）
        timeout: LLM 调用超时
        progress_callback: 进度回调 (done, total, message)

    Returns:
        (questions, doc_stats, actual_seed)
        questions: list[dict]，question dicts
        doc_stats: list[dict]，每文档的统计信息
        actual_seed: 实际使用的主种子（用于元数据保存）
    """
    # 校验配置
    ok, errors = validate_multi_doc_config(doc_configs)
    if not ok:
        raise ValueError("多文档配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

    active_configs = [dc for dc in doc_configs if dc.get("num_questions", 0) > 0]

    if not snapshot_id:
        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # 确定主种子：用户指定或随机生成
    if master_seed and master_seed > 0:
        actual_seed = master_seed
    else:
        actual_seed = random.randint(1, 2**31 - 1)

    total_questions = sum(dc["num_questions"] for dc in active_configs)
    total_docs = len(active_configs)

    if progress_callback:
        progress_callback(0, total_docs * 2,
                          f"准备从 {total_docs} 个文档生成 {total_questions} 道题（两阶段）...")

    all_questions = []
    doc_stats = []  # 每文档的统计信息

    for doc_idx, dc in enumerate(active_configs):
        doc_id = dc["document_id"]
        doc_name = dc.get("document_name", doc_id[:12])
        candidates = dc["candidates"]
        num = dc["num_questions"]

        # ── 抽取候选池（比 N 大，给 Phase 1 筛选空间） ──
        pool, pool_size, pool_capped = sample_candidate_pool(
            candidates, num, doc_id, actual_seed
        )

        # 标记来源文档
        for s in pool:
            s["_source_document_id"] = doc_id
            s["_source_document_name"] = doc_name

        if pool_capped and pool_size < num:
            # 可用候选不足 N，fail-closed
            doc_stats.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "requested": num,
                "candidate_pool": pool_size,
                "phase1_planned": 0,
                "phase2_generated": 0,
                "bound": 0,
                "status": "insufficient_candidates",
                "errors": [f"仅有 {pool_size} 个可用候选，不足 {num} 题"],
                "query_style_counts": {},
            })
            continue

        # 构建候选索引
        candidates_map = {c["segment_id"]: c for c in pool}

        # ── Phase 1: 规划 ──
        if progress_callback:
            progress_callback(doc_idx * 2, total_docs * 2,
                              f"[{doc_idx+1}/{total_docs}] {doc_name[:12]}… Phase 1: 规划中（候选池 {pool_size}）...")

        planned_items, phase1_errors = _phase1_plan_document(
            doc_name, pool, api_key, base_url, model, num, timeout
        )

        # 只有 LLM 调用失败或解析失败才视为 phase1_failed
        _phase1_critical_errors = [e for e in phase1_errors if "LLM 调用失败" in e or "解析失败" in e]
        if _phase1_critical_errors and not planned_items:
            doc_stats.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "requested": num,
                "candidate_pool": pool_size,
                "phase1_planned": 0,
                "phase2_generated": 0,
                "bound": 0,
                "status": "phase1_failed",
                "errors": _phase1_critical_errors,
                "query_style_counts": {},
            })
            continue

        if not planned_items:
            doc_stats.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "requested": num,
                "candidate_pool": pool_size,
                "phase1_planned": 0,
                "phase2_generated": 0,
                "bound": 0,
                "status": "phase1_empty",
                "errors": ["Phase 1 未返回任何规划项"],
                "query_style_counts": {},
            })
            continue

        # ── Phase 2: 生成查询 ──
        if progress_callback:
            progress_callback(doc_idx * 2 + 1, total_docs * 2,
                              f"[{doc_idx+1}/{total_docs}] {doc_name[:12]}… Phase 2: 生成查询（{len(planned_items)} 项）...")

        phase2_questions, phase2_errors = _phase2_generate_document(
            doc_name, planned_items, candidates_map,
            api_key, base_url, model, timeout
        )

        # ── 补充重试：Phase 2 少题时对缺失 candidate 重试一次 ──
        if len(phase2_questions) < len(planned_items) and len(phase2_questions) > 0:
            generated_cids = {q["candidate_id"] for q in phase2_questions}
            missing_planned = [p for p in planned_items if p["candidate_id"] not in generated_cids]
            if missing_planned:
                retry_questions, retry_errors = _phase2_generate_document(
                    doc_name, missing_planned, candidates_map,
                    api_key, base_url, model, timeout
                )
                # 只补充缺失的 candidate_id
                existing_cids = {q["candidate_id"] for q in phase2_questions}
                for rq in retry_questions:
                    if rq["candidate_id"] not in existing_cids:
                        phase2_questions.append(rq)
                        existing_cids.add(rq["candidate_id"])
                phase2_errors.extend([f"[重试] {e}" for e in retry_errors])

        # 构建最终 question dicts（本地绑定）
        bound_questions = []
        for q_data in phase2_questions:
            candidate = q_data.get("_candidate")
            if not candidate:
                continue
            _dataset_id = dataset_id or candidate.get("dataset_id", "")
            _document_id = candidate.get("_source_document_id",
                                          candidate.get("document_id", ""))
            _document_name = candidate.get("_source_document_name",
                                            candidate.get("document_name", ""))
            _position = candidate.get("position", "")
            cid = q_data["candidate_id"]

            q = {
                "question": q_data["retrieval_query"],
                "retrieval_query": q_data["retrieval_query"],
                "question_mode": "chunk_exact",
                "evaluation_type": "chunk_exact",
                "question_id": f"ce_{snapshot_id}_{cid}",
                "target_label": q_data["target_label"],
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
                "query_style": q_data.get("query_style", "semantic"),
                "generation_plan": q_data.get("generation_plan", ""),
                "selection_seed": actual_seed,
            }
            bound_questions.append(q)

        # 统计 query_style 分布
        _style_counts = {}
        for q in bound_questions:
            _s = q.get("query_style", "semantic")
            _style_counts[_s] = _style_counts.get(_s, 0) + 1

        # 判断是否 underfilled
        _status = "ok"
        _doc_errors = list(phase2_errors)
        if len(bound_questions) < num:
            _status = "underfilled"
            _doc_errors.append(f"请求 {num} 题，实际绑定 {len(bound_questions)} 题")

        all_questions.extend(bound_questions)
        doc_stats.append({
            "document_id": doc_id,
            "document_name": doc_name,
            "requested": num,
            "candidate_pool": pool_size,
            "phase1_planned": len(planned_items),
            "phase2_generated": len(phase2_questions),
            "bound": len(bound_questions),
            "status": _status,
            "errors": _doc_errors,
            "query_style_counts": _style_counts,
        })

    if not all_questions:
        # 构建详细错误信息
        failed_docs = [s for s in doc_stats if s["status"] not in ("ok", "underfilled")]
        if failed_docs:
            detail = "; ".join(
                f"{s['document_name']}: {s['status']} ({'; '.join(s['errors'][:2])})"
                for s in failed_docs
            )
            raise ValueError(f"所有文档均生成失败。详情: {detail}")
        raise ValueError("没有可用的候选 chunk")

    # 去重（按 candidate_id）
    seen_cids = set()
    deduped_questions = []
    for q in all_questions:
        cid = q.get("candidate_id", "")
        if cid and cid not in seen_cids:
            seen_cids.add(cid)
            deduped_questions.append(q)
    all_questions = deduped_questions

    # 截取到目标数量
    all_questions = all_questions[:total_questions]

    if progress_callback:
        success_count = sum(1 for s in doc_stats if s["status"] in ("ok", "underfilled"))
        progress_callback(total_docs * 2, total_docs * 2,
                          f"生成完成: {len(all_questions)} 道题（{success_count}/{total_docs} 文档成功）")

    return all_questions, doc_stats, actual_seed


def get_multi_doc_stats_summary(doc_stats):
    """生成多文档出题统计摘要（供 UI 显示）。"""
    lines = []
    for s in doc_stats:
        name = s.get("document_name", "")[:15]
        status = s.get("status", "")
        req = s.get("requested", 0)
        pool = s.get("candidate_pool", 0)
        p1 = s.get("phase1_planned", 0)
        p2 = s.get("phase2_generated", 0)
        bound = s.get("bound", 0)
        styles = s.get("query_style_counts", {})
        style_str = " | ".join(f"{k}:{v}" for k, v in sorted(styles.items())) if styles else ""

        if status == "ok":
            style_note = f" [{style_str}]" if style_str else ""
            lines.append(f"✅ {name}: {req}→池{pool}→计划{p1}→生成{p2}→绑定{bound}{style_note}")
        elif status == "underfilled":
            style_note = f" [{style_str}]" if style_str else ""
            err = (s.get("errors") or [""])[-1][:30]
            lines.append(f"⚠️ {name}: {req}→池{pool}→计划{p1}→生成{p2}→绑定{bound}{style_note} — {err}")
        else:
            err = (s.get("errors") or ["未知错误"])[0][:40]
            lines.append(f"❌ {name}: {status} — {err}")
    return "\n".join(lines)


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


# ── 两阶段出题 ──────────────────────────────────────────────────


def _build_phase1_candidates_text(candidates, max_content_chars=200):
    """构建 Phase 1 规划阶段的候选片段文本（含位置和开头+结尾摘要）。"""
    lines = []
    for i, c in enumerate(candidates):
        content = c.get("content", "")
        position = c.get("position", "")
        # 长 chunk 用开头+结尾摘要
        if len(content) > max_content_chars:
            head = content[:max_content_chars // 2]
            tail = content[-(max_content_chars // 2):]
            summary = f"{head}…{tail}"
        else:
            summary = content
        pos_str = f" (位置:{position})" if position else ""
        lines.append(f"候选 {i+1}{pos_str} — ID: {c['segment_id']}\n内容: {summary}")
    return "\n\n".join(lines)


CHUNK_EXACT_PHASE1_PROMPT = """你是 RAG 检索评测出题规划专家。根据以下候选知识片段，筛选出有独立检索价值的片段，并为每个保留的片段规划结构化检索方案。

**文档名称：** {doc_name}
**需要筛选出的目标题数：** {num_questions}

**候选片段列表：**
{candidates_text}

---

## 你的任务

从候选片段中筛选出**恰好 {num_questions} 个**有独立检索价值的片段，排除以下类型：
1. **纯标题/目录** — 没有实质性知识内容
2. **无独立检索价值** — 无法用单一检索意图定位的片段
3. **明显重复** — 与其他片段表达完全相同的知识点
4. **过短/过泛** — 内容不足以支撑一个独立检索查询

**均衡混合策略：** 尽量覆盖不同条款类型（定义、权利义务、期限、例外、金额/表格字段），不要只选一堆相近定义。

---

## query_style（查询风格）

为每个保留的片段指定一种查询风格：

- **lexical** — 保留合同原术语，测关键词/术语召回能力。适用于专有名词、法条编号、特定定义。
- **semantic** — 同义改写或语序变化，测 embedding 的语义鲁棒性。适用于通用概念、流程描述。
- **disambiguating** — 核心概念加一个必要限定词，测相邻相似条款的区分能力。适用于多处出现类似表述的条款。

尽量三类风格均衡分布。

---

## 输出格式（严格 JSON 数组）

```json
[
  {{
    "candidate_id": "候选ID",
    "query_style": "lexical | semantic | disambiguating",
    "search_intent": "简短检索意图",
    "target_label": "简短标签（3-10字）",
    "must_preserve_terms": ["必须保留在查询中的术语"],
    "plan": "一句话说明该知识点的检索价值和出题策略"
  }},
  ...
]
```

**字段说明：**
- candidate_id: 必须是候选列表中的 ID，不得编造
- query_style: 查询风格（lexical / semantic / disambiguating）
- search_intent: 检索意图概述（非查询本体），用于 Phase 2 生成查询
- target_label: 简短标签，概括查询指向的知识点
- must_preserve_terms: 必须保留在最终查询中的术语列表（lexical 风格通常包含原文术语，semantic 风格可为空）
- plan: 一句话说明出题策略

**禁止输出：**
- ❌ segment_id / content_hash — 系统自动绑定
- ❌ reference_answer — chunk_exact 不使用参考答案
- ❌ 完整检索查询 — Phase 2 负责生成
- ❌ 任何非 JSON 内容

请直接输出 JSON 数组，不要添加其他文字。"""


CHUNK_EXACT_PHASE2_PROMPT = """你是 RAG 检索评测出题专家。根据以下已规划的候选片段及其检索方案，为每个片段生成一条短检索查询。

**文档名称：** {doc_name}

**已规划的候选片段：**
{candidates_text}

---

## 查询风格规则（必须严格遵从每个片段的 query_style）

### lexical（保留原术语）
- 保留合同原文中的关键术语、法条编号、专有名词
- 可以直接使用原文中的核心短语
- 测试关键词/术语召回能力

### semantic（同义改写）
- 必须做自然的等价表达，不得照抄原文
- 使用同义词、语序调整、概括性表述
- 测试 embedding 的语义鲁棒性

### disambiguating（区分性查询）
- 保留核心概念，加上一个必要的限定词
- 限定词来自原文中使该条款区别于其他类似条款的修饰语
- 测试相邻相似条款的区分能力

---

## 通用规则

- ❌ **不得是问句** — 禁止出现问号（？/ ?）
- ❌ **禁止问答或推理导向表达** — 不得包含"什么/为何/为什么/如何/是否/哪些/请分析/请说明/分别"等词
- ❌ **不得把 target_label 拼进 query** — query 是检索意图，label 是展示标签
- ❌ **不得引入文中不存在的信息**
- ✅ 短查询，通常不超过约 20 个汉字或 token；英文专有名词可完整保留
- ✅ 允许中文、英文和双语合同术语
- ✅ **必须保留 must_preserve_terms 中列出的术语**

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
- ❌ 任何非 JSON 内容

请直接输出 JSON 数组，不要添加其他文字。"""


def _build_phase2_candidates_text(planned_items, candidates_map, max_content_chars=300):
    """构建 Phase 2 的候选片段文本（含 query_style、检索意图、must_preserve_terms）。"""
    lines = []
    for i, item in enumerate(planned_items):
        cid = item.get("candidate_id", "")
        c = candidates_map.get(cid, {})
        content = c.get("content", "")
        position = c.get("position", "")
        if len(content) > max_content_chars:
            head = content[:max_content_chars // 2]
            tail = content[-(max_content_chars // 2):]
            content = f"{head}…{tail}"
        style = item.get("query_style", "semantic")
        intent = item.get("search_intent", "")
        label = item.get("target_label", "")
        terms = item.get("must_preserve_terms", [])
        terms_str = ", ".join(terms) if terms else "（无特殊要求）"
        pos_str = f" (位置:{position})" if position else ""
        lines.append(
            f"候选 {i+1}{pos_str} — ID: {cid}\n"
            f"查询风格: {style}\n"
            f"检索意图: {intent}\n"
            f"标签: {label}\n"
            f"必须保留的术语: {terms_str}\n"
            f"内容: {content}"
        )
    return "\n\n".join(lines)


def _phase1_plan_document(doc_name, candidates, api_key, base_url, model,
                           num_questions, timeout=60):
    """Phase 1: 为单个文档做题规划。

    Returns:
        (planned_items, errors)
        planned_items: list[dict]，每项含 candidate_id, query_style, search_intent,
                        target_label, must_preserve_terms, plan
        errors: list[str]
    """
    if not candidates:
        return [], ["候选列表为空"]

    # 构建候选文本（使用较短摘要）
    candidates_text = _build_phase1_candidates_text(candidates)
    prompt = (CHUNK_EXACT_PHASE1_PROMPT
              .replace("{doc_name}", doc_name)
              .replace("{candidates_text}", candidates_text)
              .replace("{num_questions}", str(num_questions)))

    try:
        response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)
    except Exception as exc:
        return [], [f"Phase 1 LLM 调用失败: {exc}"]

    try:
        items = _parse_llm_response(response_text)
    except ValueError as exc:
        return [], [f"Phase 1 解析失败: {exc}"]

    # 校验 candidate_id 并去重
    candidate_ids = {c["segment_id"] for c in candidates}
    valid_items = []
    seen_cids = set()
    errors = []
    for item in items:
        cid = item.get("candidate_id", "")
        if not cid:
            errors.append("Phase 1 输出缺少 candidate_id")
            continue
        if cid not in candidate_ids:
            errors.append(f"Phase 1 candidate_id '{cid}' 不在当前候选集中（已拒绝）")
            continue
        if cid in seen_cids:
            errors.append(f"Phase 1 candidate_id '{cid}' 重复（已拒绝）")
            continue
        seen_cids.add(cid)

        # 校验 query_style
        style = item.get("query_style", "semantic")
        if style not in ("lexical", "semantic", "disambiguating"):
            style = "semantic"

        valid_items.append({
            "candidate_id": cid,
            "query_style": style,
            "search_intent": item.get("search_intent", ""),
            "target_label": item.get("target_label", ""),
            "must_preserve_terms": item.get("must_preserve_terms", []),
            "plan": item.get("plan", ""),
        })

    # 截取到目标题数
    valid_items = valid_items[:num_questions]

    return valid_items, errors


def _phase2_generate_document(doc_name, planned_items, candidates_map,
                               api_key, base_url, model, timeout=60):
    """Phase 2: 为单个文档生成检索查询。

    Args:
        doc_name: 文档名称
        planned_items: Phase 1 输出的规划列表（含 query_style, must_preserve_terms）
        candidates_map: {candidate_id: candidate_dict}
        api_key, base_url, model: LLM 配置
        timeout: 超时秒数

    Returns:
        (questions, errors)
        questions: list[dict]，每项含 candidate_id, retrieval_query, target_label,
                   query_style, generation_plan, _candidate
        errors: list[str]
    """
    if not planned_items:
        return [], ["Phase 1 未返回任何规划项"]

    # 构建候选文本（含 Phase 1 检索方案）
    candidates_text = _build_phase2_candidates_text(planned_items, candidates_map)
    prompt = CHUNK_EXACT_PHASE2_PROMPT.replace("{doc_name}", doc_name).replace("{candidates_text}", candidates_text)

    try:
        response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)
    except Exception as exc:
        return [], [f"Phase 2 LLM 调用失败: {exc}"]

    try:
        items = _parse_llm_response(response_text)
    except ValueError as exc:
        return [], [f"Phase 2 解析失败: {exc}"]

    # 构建 planned_items 索引（含 query_style 等元数据）
    planned_by_id = {}
    for p in planned_items:
        cid = p.get("candidate_id", "")
        if cid:
            planned_by_id[cid] = p

    # 校验并构建 question dicts
    questions = []
    seen_cids = set()
    errors = []
    for item in items:
        cid = item.get("candidate_id", "")
        if not cid:
            errors.append("Phase 2 输出缺少 candidate_id")
            continue
        if cid not in planned_by_id:
            errors.append(f"Phase 2 candidate_id '{cid}' 不在 Phase 1 规划集中（已拒绝）")
            continue
        if cid in seen_cids:
            errors.append(f"Phase 2 candidate_id '{cid}' 重复（已拒绝）")
            continue
        seen_cids.add(cid)

        candidate = candidates_map.get(cid)
        if not candidate:
            errors.append(f"candidate_id '{cid}' 未找到对应候选 chunk")
            continue

        retrieval_query = (item.get("retrieval_query") or "").strip()
        if not retrieval_query:
            errors.append(f"candidate_id '{cid}' 缺少 retrieval_query")
            continue

        target_label = (item.get("target_label") or "").strip()

        # 从 Phase 1 规划中继承元数据
        planned = planned_by_id[cid]
        questions.append({
            "candidate_id": cid,
            "retrieval_query": retrieval_query,
            "target_label": target_label,
            "query_style": planned.get("query_style", "semantic"),
            "generation_plan": planned.get("plan", ""),
            "_candidate": candidate,
        })

    return questions, errors


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
