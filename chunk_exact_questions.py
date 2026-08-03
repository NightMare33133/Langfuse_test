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

# ── Query 校验常量 ──────────────────────────────────────────────

# 中文查询长度限制（字符数）
_QUERY_MIN_LEN_ZH = 3
_QUERY_SOFT_MAX_LEN_ZH = 20  # 软上限：目标长度，不强制拒绝
_QUERY_HARD_MAX_LEN_ZH = 30  # 硬上限：超过则拒绝

# 英文查询长度限制（token 数，按空格分词近似）
_QUERY_MAX_LEN_EN_TOKENS = 15

# 禁止出现的问句/问答导向词
_FORBIDDEN_QUESTION_WORDS = frozenset([
    "什么", "为何", "为什么", "如何", "是否", "哪些", "请分析", "请说明",
    "分别", "what", "why", "how", "whether",
])

# 禁止出现的标点
_FORBIDDEN_PUNCTUATION = frozenset(["？", "?"])

# 同 chunk 中独立知识点的并列模式（正则）
_MULTI_CONCEPT_PATTERNS = [
    re.compile(r"与|和|及|以及|并且|同时|另外|此外|还有"),  # 中文并列
    re.compile(r"\band\b|\bor\b|\bas well as\b|\bin addition\b"),  # 英文并列
    re.compile(r"[、，].*[、，]"),  # 多个顿号/逗号分隔的列表
]


def _count_chars(text):
    """统计有效字符数（去除空格后）。"""
    return len(text.replace(" ", "").replace("　", ""))


def _is_mostly_chinese(text):
    """判断文本是否主要为中文。"""
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    return chinese_chars >= len(text) * 0.3


def _is_mostly_english(text):
    """判断文本是否主要为英文。"""
    ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    return ascii_chars >= len(text) * 0.5


def validate_retrieval_query(query, query_style="semantic", target_label=""):
    """Fail-closed 校验 retrieval_query 质量。

    Rules:
    1. 禁止问句词和问号
    2. 中文查询 3-20 字，英文按 token 计
    3. target_label 非空
    4. 禁止多概念并列（disambiguating 除外，允许一个限定词）

    Returns:
        (ok, errors): ok=True 表示通过，errors 为错误列表
    """
    errors = []
    query = (query or "").strip()

    if not query:
        return False, ["retrieval_query 为空"]

    # target_label 校验
    if not (target_label or "").strip():
        errors.append("target_label 为空")

    # 禁止问句词
    query_lower = query.lower()
    for word in _FORBIDDEN_QUESTION_WORDS:
        if word in query_lower:
            errors.append(f"包含禁止问句词「{word}」")
            break  # 一个就够了

    # 禁止问号
    for punct in _FORBIDDEN_PUNCTUATION:
        if punct in query:
            errors.append(f"包含禁止标点「{punct}」")
            break

    # 长度校验
    if _is_mostly_chinese(query):
        char_count = _count_chars(query)
        if char_count < _QUERY_MIN_LEN_ZH:
            errors.append(f"中文查询过短（{char_count} 字，最少 {_QUERY_MIN_LEN_ZH} 字）")
        elif char_count > _QUERY_HARD_MAX_LEN_ZH:
            errors.append(f"中文查询过长（{char_count} 字，硬上限 {_QUERY_HARD_MAX_LEN_ZH} 字）")
        # 注意：超过软上限(_QUERY_SOFT_MAX_LEN_ZH)但低于硬上限不拒绝
        # 重点拒绝坏结构（问句、列表、多概念），而非单纯长度
    elif _is_mostly_english(query):
        token_count = len(query.split())
        if token_count > _QUERY_MAX_LEN_EN_TOKENS:
            errors.append(f"英文查询过长（{token_count} tokens，最多 {_QUERY_MAX_LEN_EN_TOKENS}）")

    # 多概念并列检测（仅对 semantic 和 lexical 生效）
    if query_style in ("semantic", "lexical"):
        for pat in _MULTI_CONCEPT_PATTERNS:
            # 对于 "与" 类并列，需要检测是否有多个独立概念
            if pat.search(query):
                # 允许 "与" 出现在专有名词中（如 "RAG 与传统问答" 可以是一个概念）
                # 但禁止明显的多概念拼接，如 "A 与 B"、"A、B、C"
                # 简单启发式：如果包含多个逗号/顿号分隔的项目，拒绝
                items = re.split(r"[、，,]", query)
                real_items = [it.strip() for it in items if it.strip() and len(it.strip()) >= 2]
                if len(real_items) >= 3:
                    errors.append(f"疑似多概念列表拼接（{len(real_items)} 项）")
                    break

    return len(errors) == 0, errors


def validate_groundedness(query, content, allowed_synonyms=None):
    """Groundedness 校验：query 中的关键实体必须在 content 中可找到。

    Args:
        query: retrieval_query
        content: candidate chunk 的完整内容
        allowed_synonyms: Phase 1 明确允许的等价同义表达 dict（可选）

    Returns:
        (ok, errors): ok=True 表示通过，errors 为错误列表
    """
    errors = []
    query = (query or "").strip()
    content = (content or "").strip()

    if not query:
        return False, ["retrieval_query 为空"]
    if not content:
        return False, ["content 为空"]

    content_lower = content.lower()
    allowed = allowed_synonyms or {}

    # 提取 query 中的关键实体
    # 英文术语：连续英文单词或缩写（≥2 字符）
    en_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", query)
    # 数字/条款号
    numbers = re.findall(r"\d+(?:\.\d+)*(?:\s*条|\s*款|\s*项|\s*条|\s*号)?", query)

    # 中文：使用滑动窗口提取 2-4 字的中文子串
    zh_chars = re.findall(r"[一-鿿]+", query)
    zh_terms = []
    for phrase in zh_chars:
        if len(phrase) <= 2:
            # 短词直接添加
            zh_terms.append(phrase)
        else:
            # 提取所有 2-4 字的子串
            for window in range(2, min(5, len(phrase) + 1)):
                for i in range(len(phrase) - window + 1):
                    zh_terms.append(phrase[i:i + window])

    all_terms = zh_terms + en_terms + numbers

    # 检查是否有至少一个关键实体在 content 中找到
    found_count = 0
    not_found_terms = []

    for term in all_terms:
        term_lower = term.lower()
        # 在 content 中查找
        if term_lower in content_lower:
            found_count += 1
            continue
        # 在允许的同义词中查找
        if term_lower in {k.lower() for k in allowed.keys()}:
            found_count += 1
            continue
        # 检查是否是允许的同义表达的值
        for syn_key, syn_val in allowed.items():
            if isinstance(syn_val, str) and term_lower in syn_val.lower():
                found_count += 1
                break
            elif isinstance(syn_val, list) and any(term_lower in sv.lower() for sv in syn_val):
                found_count += 1
                break
        else:
            # 术语在 content 中未找到
            if len(term) > 2:
                not_found_terms.append(term)

    # 放宽要求：只要至少有一个关键实体在 content 中找到即可
    # 但如果所有长实体（>2字符）都找不到，则拒绝
    if found_count == 0 and not_found_terms:
        errors.append(f"关键实体均未在 content 中找到: {', '.join(not_found_terms[:3])}")

    return len(errors) == 0, errors


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
                "phase2_first_returned": 0,
                "first_rejected": 0,
                "retry_attempted": 0,
                "retry_recovered": 0,
                "final_bound": 0,
                "binding_failed": 0,
                "status": "insufficient_candidates",
                "errors": [f"仅有 {pool_size} 个可用候选，不足 {num} 题"],
                "query_style_counts": {},
                "rejection_diagnostics": [],
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
                "phase2_first_returned": 0,
                "first_rejected": 0,
                "retry_attempted": 0,
                "retry_recovered": 0,
                "final_bound": 0,
                "binding_failed": 0,
                "status": "phase1_failed",
                "errors": _phase1_critical_errors,
                "query_style_counts": {},
                "rejection_diagnostics": [],
            })
            continue

        if not planned_items:
            doc_stats.append({
                "document_id": doc_id,
                "document_name": doc_name,
                "requested": num,
                "candidate_pool": pool_size,
                "phase1_planned": 0,
                "phase2_first_returned": 0,
                "first_rejected": 0,
                "retry_attempted": 0,
                "retry_recovered": 0,
                "final_bound": 0,
                "binding_failed": 0,
                "status": "phase1_empty",
                "errors": ["Phase 1 未返回任何规划项"],
                "query_style_counts": {},
                "rejection_diagnostics": [],
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

        # ── 统计首轮结果 ──
        first_returned_count = len(phase2_questions)
        passed_questions = [q for q in phase2_questions if q.get("validation_status") == "passed"]
        rejected_questions = [q for q in phase2_questions if q.get("validation_status") != "passed"]
        first_rejected_count = len(rejected_questions)

        # 收集拒绝诊断
        rejection_diagnostics = []
        for rq in rejected_questions:
            rejection_diagnostics.append({
                "candidate_id": rq.get("candidate_id", ""),
                "query": rq.get("retrieval_query", ""),
                "errors": rq.get("validation_errors", []),
            })

        # ── 带错误反馈的重试：Phase 2 少题或校验失败时重试一次 ──
        generated_cids = {q["candidate_id"] for q in passed_questions}
        # 找出需要重试的候选：未生成的 + 校验失败的
        retry_rejected = [q for q in rejected_questions if q.get("candidate_id") not in generated_cids]
        # 补充完全未生成的候选
        for p in planned_items:
            cid = p["candidate_id"]
            if cid not in generated_cids and not any(r["candidate_id"] == cid for r in retry_rejected):
                retry_rejected.append({
                    "candidate_id": cid,
                    "retrieval_query": "",
                    "validation_errors": ["未生成"],
                    "target_fact": p.get("target_fact", ""),
                    "retrieval_intent": p.get("retrieval_intent", ""),
                    "target_label": p.get("target_label", ""),
                    "allowed_modifiers": p.get("allowed_modifiers", []),
                    "forbidden_concepts": p.get("forbidden_concepts", []),
                    "query_style": p.get("query_style", "semantic"),
                })

        retry_attempted_count = len(retry_rejected)
        retry_recovered_count = 0

        if retry_rejected:
            # 使用带错误反馈的重试提示
            retry_text = _build_phase2_retry_text(retry_rejected, candidates_map)
            retry_prompt = PHASE2_RETRY_PROMPT.replace("{doc_name}", doc_name).replace("{retry_items}", retry_text)

            try:
                retry_response = call_llm(retry_prompt, api_key, base_url, model, timeout=timeout)
                retry_items = _parse_llm_response(retry_response)
            except Exception as exc:
                phase2_errors.append(f"[重试] 调用或解析失败: {exc}")
                retry_items = []

            # 校验重试结果
            for item in retry_items:
                cid = item.get("candidate_id", "")
                if not cid or cid in generated_cids:
                    continue
                # 检查是否在重试列表中
                retry_info = next((r for r in retry_rejected if r["candidate_id"] == cid), None)
                if not retry_info:
                    continue

                candidate = candidates_map.get(cid)
                if not candidate:
                    continue

                retrieval_query = (item.get("retrieval_query") or "").strip()
                if not retrieval_query:
                    continue

                target_label = (item.get("target_label") or "").strip()

                # 从 planned_items 获取元数据
                planned = next((p for p in planned_items if p["candidate_id"] == cid), {})
                query_style = planned.get("query_style", "semantic")

                # 校验
                query_ok, query_errors = validate_retrieval_query(retrieval_query, query_style, target_label)
                content = candidate.get("content", "")
                allowed_synonyms = {}
                for term in planned.get("must_preserve_terms", []):
                    allowed_synonyms[term.lower()] = term
                for mod in planned.get("allowed_modifiers", []):
                    allowed_synonyms[mod.lower()] = mod
                ground_ok, ground_errors = validate_groundedness(retrieval_query, content, allowed_synonyms)

                all_errors = query_errors + ground_errors
                validation_status = "passed" if (query_ok and ground_ok) else "rejected"

                if validation_status == "passed":
                    passed_questions.append({
                        "candidate_id": cid,
                        "retrieval_query": retrieval_query,
                        "target_label": target_label,
                        "query_style": query_style,
                        "target_fact": planned.get("target_fact", ""),
                        "retrieval_intent": planned.get("retrieval_intent", ""),
                        "allowed_modifiers": planned.get("allowed_modifiers", []),
                        "forbidden_concepts": planned.get("forbidden_concepts", []),
                        "validation_status": "passed",
                        "validation_errors": [],
                        "generation_plan": planned.get("plan", ""),
                        "_candidate": candidate,
                    })
                    generated_cids.add(cid)
                    retry_recovered_count += 1
                else:
                    # 重试仍失败，记录诊断
                    rejection_diagnostics.append({
                        "candidate_id": cid,
                        "query": retrieval_query,
                        "errors": all_errors,
                        "retry": True,
                    })

        # 使用通过校验的题目替换原始列表
        phase2_questions = passed_questions

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
                "expected_content": candidate.get("content", ""),
                "dataset_id": _dataset_id,
                "document_id": _document_id,
                "document_name": _document_name,
                "snapshot_id": snapshot_id,
                "source_position": _position,
                "source_label": f"doc:{_document_id[:8]} pos:{_position}",
                "query_style": q_data.get("query_style", "semantic"),
                "target_fact": q_data.get("target_fact", ""),
                "retrieval_intent": q_data.get("retrieval_intent", ""),
                "allowed_modifiers": q_data.get("allowed_modifiers", []),
                "forbidden_concepts": q_data.get("forbidden_concepts", []),
                "validation_status": q_data.get("validation_status", "passed"),
                "validation_errors": q_data.get("validation_errors", []),
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
        _final_bound = len(bound_questions)
        _status = "ok"
        _doc_errors = list(phase2_errors)
        if _final_bound < num:
            _status = "underfilled"
            _doc_errors.append(
                f"请求 {num} 题，实际绑定 {_final_bound} 题"
                f"（首轮返回 {first_returned_count}，校验拒绝 {first_rejected_count}，"
                f"重试 {retry_attempted_count}，恢复 {retry_recovered_count}）"
            )

        all_questions.extend(bound_questions)
        doc_stats.append({
            "document_id": doc_id,
            "document_name": doc_name,
            "requested": num,
            "candidate_pool": pool_size,
            "phase1_planned": len(planned_items),
            "phase2_first_returned": first_returned_count,
            "first_rejected": first_rejected_count,
            "retry_attempted": retry_attempted_count,
            "retry_recovered": retry_recovered_count,
            "final_bound": _final_bound,
            "binding_failed": 0,  # chunk_exact 不涉及 segment 绑定失败
            "status": _status,
            "errors": _doc_errors,
            "query_style_counts": _style_counts,
            "rejection_diagnostics": rejection_diagnostics,
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
    """生成多文档出题统计摘要（供 UI 显示）。

    字段说明：
    - requested: 请求题数
    - phase1_planned: Phase 1 规划数
    - phase2_first_returned: LLM 首轮返回数
    - first_rejected: 首轮校验拒绝数
    - retry_attempted: 进入重试的候选数
    - retry_recovered: 重试后恢复的题数
    - final_bound: 最终通过校验的题数
    - binding_failed: segment 绑定失败数（chunk_exact 始终为 0）
    """
    lines = []
    total_requested = 0
    total_bound = 0
    total_rejected = 0
    total_binding_failed = 0

    for s in doc_stats:
        name = s.get("document_name", "")[:15]
        status = s.get("status", "")
        req = s.get("requested", 0)
        pool = s.get("candidate_pool", 0)
        p1 = s.get("phase1_planned", 0)
        p2_first = s.get("phase2_first_returned", s.get("phase2_generated", 0))
        first_rej = s.get("first_rejected", 0)
        retry_att = s.get("retry_attempted", 0)
        retry_rec = s.get("retry_recovered", 0)
        final_bound = s.get("final_bound", s.get("bound", 0))
        binding_fail = s.get("binding_failed", 0)
        styles = s.get("query_style_counts", {})
        style_str = " | ".join(f"{k}:{v}" for k, v in sorted(styles.items())) if styles else ""

        total_requested += req
        total_bound += final_bound
        total_rejected += first_rej
        total_binding_failed += binding_fail

        if status == "ok":
            style_note = f" [{style_str}]" if style_str else ""
            lines.append(
                f"✅ {name}: 请求{req}→池{pool}→计划{p1}→首轮{p2_first}"
                f"→校验拒绝{first_rej}→重试{retry_att}/恢复{retry_rec}"
                f"→最终绑定{final_bound}{style_note}"
            )
        elif status == "underfilled":
            style_note = f" [{style_str}]" if style_str else ""
            lines.append(
                f"⚠️ {name}: 请求{req}→池{pool}→计划{p1}→首轮{p2_first}"
                f"→校验拒绝{first_rej}→重试{retry_att}/恢复{retry_rec}"
                f"→最终绑定{final_bound}{style_note}"
            )
        else:
            err = (s.get("errors") or ["未知错误"])[0][:40]
            lines.append(f"❌ {name}: {status} — {err}")

    # 全局摘要
    lines.append("")
    lines.append(
        f"📊 合计: 请求{total_requested} → 最终绑定{total_bound}"
        f" | 质量校验拒绝{total_rejected}"
        f" | 绑定失败{total_binding_failed}"
    )

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


PHASE1_JSON_REPAIR_PROMPT = """以下文本应为 JSON 数组但解析失败。请仅返回合法的 JSON 数组，保持原有内容不变，仅修复语法错误（如缺少逗号、引号不匹配、多余逗号、未闭合括号等）。

原始文本：
{raw_output}

请直接输出修复后的 JSON 数组，不要添加其他文字。"""


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

## 规划思路：从 target_fact 到 retrieval_intent

**先提取 target_fact，再抽象出 retrieval_intent，最后确定 target_label。**

1. **target_fact（证据事实锚点）**— 从 chunk 中提取的完整事实/规则/定义
   - 可以包含完整事实细节（数值、名称、条件），用于 grounding 和校验
   - 示例："买方依瑞典法律组建，供应商依中国法律组建"

2. **retrieval_intent（用户检索意图）**— 用户一次独立的信息需求，即最终查询想找的"主题"
   - 应为"对象 + 属性/规则/条件/范围/要求"等自然短语
   - 不应复述 target_fact 的所有答案细节
   - 示例："协议双方的注册地"
   - 检验标准：如果用户在搜索引擎输入 retrieval_intent，能否期望找到包含 target_fact 的文档？

3. **target_label（展示标签）**— 仅用于 UI 预览的 3-8 字短标签，不替代 retrieval_intent

---

## target_fact 提取规则

- 必须是 chunk 中的一个独立知识点，不能是整个 chunk 的摘要
- 不能把同 chunk 中的多个规则拼接在一起
- 示例：
  - ✅ "供应商需在协议终止后30天内归还客户数据"
  - ✅ "ISO9001认证宽限期为6个月"
  - ❌ "文件冲突时的优先顺序 框架协议的用途"（两个独立概念拼接）
  - ❌ "IT服务需包含认证、会话管理、访问控制、加密、日志记录"（长列表复述）

---

## retrieval_intent 生成规则

- 必须是自然的、面向用户的检索主题
- 应为"对象 + 属性/规则/条件/范围/要求"等结构的自然短语
- **禁止**将 target_fact 中的答案、数值、地点、动作结果直接平铺组合成检索词
- **禁止**将无共同主题的多个规则合并为一个 retrieval_intent
- **允许**同一信息字段下存在一组对应答案（如"协议双方的注册地"涵盖双方注册地）

**正反例：**
- target_fact："买方依瑞典法律组建，供应商依中国法律组建"
  - ❌ 差："买方瑞典组建 供应商中国组建"（答案关键词直接拼接）
  - ✅ 好："协议双方的注册地"（自然检索主题）

- target_fact："业务合作伙伴不得与竞争者约定价格、折扣、销售条款或划分市场"
  - ❌ 差："业务合作伙伴与竞争者合谋限定价格划分市场"（答案平铺）
  - ✅ 好："业务合作伙伴的反垄断合规要求"（面向用户的检索主题）

---

## query_style（查询风格）

为每个保留的片段指定一种查询风格：

- **lexical** — 保留合同原术语，测关键词/术语召回能力。适用于专有名词、法条编号、特定定义。
- **semantic** — 同义改写或语序变化，测 embedding 的语义鲁棒性。适用于通用概念、流程描述。
- **disambiguating** — 核心概念加一个必要限定词，测相邻相似条款的区分能力。适用于多处出现类似表述的条款。

尽量三类风格均衡分布。

---

## allowed_modifiers 与 forbidden_concepts

- **allowed_modifiers** — 最多 2 个必要限定词，用于限定 target_fact 的范围。例如："认证宽限期" 的限定词可以是 "ISO9001"、"6个月"。
- **forbidden_concepts** — 同 chunk 中但不应混入本题的其他知识点。Phase 2 生成查询时必须避免引入这些概念。

---

## 输出格式（严格 JSON 数组）

```json
[
  {{
    "candidate_id": "候选ID",
    "query_style": "lexical | semantic | disambiguating",
    "target_fact": "完整的原子事实/规则/定义（证据锚点，可含完整细节）",
    "retrieval_intent": "面向用户的自然检索主题（不含答案细节）",
    "target_label": "简短标签（3-8字）",
    "allowed_modifiers": ["限定词1", "限定词2"],
    "forbidden_concepts": ["不应混入的知识点1", "不应混入的知识点2"],
    "must_preserve_terms": ["必须保留在查询中的术语"],
    "plan": "一句话说明该知识点的检索价值和出题策略"
  }},
  ...
]
```

**字段说明：**
- candidate_id: 必须是候选列表中的 ID，不得编造
- query_style: 查询风格（lexical / semantic / disambiguating）
- target_fact: 证据事实锚点，可包含完整事实细节，用于 grounding 和校验
- retrieval_intent: 用户检索意图，面向用户的自然检索主题，是 Phase 2 生成 retrieval_query 的主要依据
- target_label: 简短展示标签
- allowed_modifiers: 最多 2 个必要限定词
- forbidden_concepts: 同 chunk 中但不应混入本题的其他知识点
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

## 核心原则

retrieval_query 以 **retrieval_intent**（用户检索意图）为主要依据生成。

- retrieval_intent 决定查询的"主题"和"方向"
- target_fact 是证据锚点和校验边界，**不要求**在查询中覆盖其所有事实细节
- allowed_modifiers、forbidden_concepts 是边界约束，用于防止偏移

---

## 查询风格规则（必须严格遵从每个片段的 query_style）

### lexical（保留原术语）
- 保留合同原文中的关键术语、法条编号、专有名词
- 可以直接使用原文中的核心短语
- 测试关键词/术语召回能力
- 不允许复制长列表或整句

### semantic（同义改写）
- 必须是自然的名词短语，不得照抄原文
- 使用同义词、语序调整、概括性表述
- 测试 embedding 的语义鲁棒性
- 不必为了"语义改写"而让每一题大幅改写；自然即可

### disambiguating（区分性查询）
- 保留核心概念，加上区分相邻条款所需的一个限定条件
- 限定词来自原文中使该条款区别于其他类似条款的修饰语
- 测试相邻相似条款的区分能力

---

## 核心规则

- ✅ **retrieval_query 以 retrieval_intent 为主生成**
- ✅ 短查询，中文通常 3-20 字；英文术语按 token 计，允许完整保留
- ✅ 必须保留 must_preserve_terms 中列出的术语
- ✅ 必须基于 target_fact 表达，不得偏离到 forbidden_concepts
- ❌ **不得是问句** — 禁止出现问号（？/ ?）
- ❌ **禁止问答词** — 不得包含"什么/为何/为什么/如何/是否/哪些/请分析/请说明/分别"等词
- ❌ **禁止照抄原文** — 不得照抄原文完整句、长列表或答案关键词串
- ❌ **禁止多概念拼接** — 不得将"定义 + 义务"、"限制 + 管辖"等独立概念并列
- ❌ **禁止答案平铺** — 不得将 target_fact 中的答案、数值、地点直接拼接成查询
- ❌ **不得引入文中不存在的信息**
- ❌ **不得把 target_label 拼进 query** — query 是检索意图，label 是展示标签

---

## 正反例（强制风格约束）

### 合同主体信息
- target_fact：买方依瑞典法律组建，供应商依中国法律组建
- retrieval_intent：协议双方的注册地
- ❌ 差：买方瑞典组建 供应商中国组建
- ✅ 好：协议双方的注册地

### 通知条款
- target_fact：电子邮件通知以自动回复或电子日志证明收悉时视为有效送达
- retrieval_intent：电子邮件通知的有效送达条件
- ❌ 差：电子邮件通知送达规则自动回复或系统日志记录
- ✅ 好：电子邮件通知的有效送达条件

### 文件条款
- target_fact：合同文件条款冲突时，按文件清单排列顺序确定优先级
- retrieval_intent：合同文件冲突的优先适用顺序
- ❌ 差：协议文件不一致时优先顺序依据文件清单
- ✅ 好：合同文件冲突的优先适用顺序

### 反垄断条款
- target_fact：业务合作伙伴不得与竞争者约定价格、折扣、销售条款或划分市场
- retrieval_intent：业务合作伙伴的反垄断合规要求
- ❌ 差：业务合作伙伴与竞争者合谋限定价格划分市场
- ✅ 好：业务合作伙伴的反垄断合规要求

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
    """构建 Phase 2 的候选片段文本（含 query_style、target_fact、retrieval_intent、allowed_modifiers、forbidden_concepts）。"""
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
        target_fact = item.get("target_fact", "")
        retrieval_intent = item.get("retrieval_intent", "")
        label = item.get("target_label", "")
        terms = item.get("must_preserve_terms", [])
        terms_str = ", ".join(terms) if terms else "（无特殊要求）"
        modifiers = item.get("allowed_modifiers", [])
        modifiers_str = ", ".join(modifiers) if modifiers else "（无限定词）"
        forbidden = item.get("forbidden_concepts", [])
        forbidden_str = ", ".join(forbidden) if forbidden else "（无禁止概念）"
        pos_str = f" (位置:{position})" if position else ""
        intent_line = f"检索意图: {retrieval_intent}\n" if retrieval_intent else ""
        lines.append(
            f"候选 {i+1}{pos_str} — ID: {cid}\n"
            f"查询风格: {style}\n"
            f"目标事实（证据锚点）: {target_fact}\n"
            f"{intent_line}"
            f"标签: {label}\n"
            f"允许的限定词: {modifiers_str}\n"
            f"禁止混入的概念: {forbidden_str}\n"
            f"必须保留的术语: {terms_str}\n"
            f"内容: {content}"
        )
    return "\n\n".join(lines)


def _phase1_plan_document(doc_name, candidates, api_key, base_url, model,
                           num_questions, timeout=60):
    """Phase 1: 为单个文档做题规划。

    Returns:
        (planned_items, errors)
        planned_items: list[dict]，每项含 candidate_id, query_style, target_fact,
                        retrieval_intent, target_label, allowed_modifiers,
                        forbidden_concepts, must_preserve_terms, plan
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
    except ValueError as parse_exc:
        # JSON 解析失败时，尝试一次修复重试
        repair_prompt = PHASE1_JSON_REPAIR_PROMPT.replace("{raw_output}", response_text[:3000])
        try:
            repaired_text = call_llm(repair_prompt, api_key, base_url, model, timeout=timeout)
            items = _parse_llm_response(repaired_text)
        except Exception as repair_exc:
            return [], [f"Phase 1 解析失败（修复重试也失败）: {parse_exc}"]

    # 构建候选索引（用于 groundedness 校验）
    candidates_by_id = {c["segment_id"]: c for c in candidates}

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

        # 提取并校验 target_fact
        target_fact = (item.get("target_fact") or "").strip()
        if not target_fact:
            # 回退：从 search_intent 提取
            target_fact = (item.get("search_intent") or "").strip()
            if target_fact:
                errors.append(f"Phase 1 candidate_id '{cid}' 缺少 target_fact，使用 search_intent 回退")
            else:
                errors.append(f"Phase 1 candidate_id '{cid}' 缺少 target_fact（已跳过）")
                continue

        # 校验 target_label
        target_label = (item.get("target_label") or "").strip()
        if not target_label:
            errors.append(f"Phase 1 candidate_id '{cid}' 缺少 target_label（已跳过）")
            continue

        # 提取 retrieval_intent（可选，兼容历史数据）
        retrieval_intent = (item.get("retrieval_intent") or "").strip()
        # 回退：若缺少 retrieval_intent 但有 search_intent，使用 search_intent
        if not retrieval_intent:
            retrieval_intent = (item.get("search_intent") or "").strip()

        # 提取 allowed_modifiers（最多 2 个）
        allowed_modifiers = item.get("allowed_modifiers", [])
        if not isinstance(allowed_modifiers, list):
            allowed_modifiers = []
        allowed_modifiers = [str(m).strip() for m in allowed_modifiers if str(m).strip()][:2]

        # 提取 forbidden_concepts
        forbidden_concepts = item.get("forbidden_concepts", [])
        if not isinstance(forbidden_concepts, list):
            forbidden_concepts = []
        forbidden_concepts = [str(f).strip() for f in forbidden_concepts if str(f).strip()]

        valid_items.append({
            "candidate_id": cid,
            "query_style": style,
            "target_fact": target_fact,
            "retrieval_intent": retrieval_intent,
            "target_label": target_label,
            "allowed_modifiers": allowed_modifiers,
            "forbidden_concepts": forbidden_concepts,
            "must_preserve_terms": item.get("must_preserve_terms", []),
            "plan": item.get("plan", ""),
        })

    # 截取到目标题数
    valid_items = valid_items[:num_questions]

    return valid_items, errors


PHASE2_RETRY_PROMPT = """你是 RAG 检索评测出题专家。以下候选的检索查询未通过质量校验，请根据拒绝原因修正。

**文档名称：** {doc_name}

**需要修正的候选：**
{retry_items}

---

## 修正要求

- retrieval_query 以 retrieval_intent（检索意图）为主生成，不要照抄 target_fact 的答案细节
- 每个候选只修正其 retrieval_query，不要改变 candidate_id 或 target_label
- 严格遵守 target_fact、allowed_modifiers、forbidden_concepts 的边界约束
- 不得出现问句词（什么/为何/为什么/如何/是否/哪些）或问号
- 不得多概念拼接、长列表复述或答案关键词平铺
- 中文查询目标 3-20 字，但允许原子事实需要的稍长短语（不超过 30 字）
- 只修正被拒绝的候选，不要添加新候选

---

## 输出格式（严格 JSON 数组）

```json
[
  {{"candidate_id": "候选ID", "retrieval_query": "修正后的短检索查询", "target_label": "简短标签"}},
  ...
]
```

请直接输出 JSON 数组，不要添加其他文字。"""


def _build_phase2_retry_text(rejected_items, candidates_map, max_content_chars=200):
    """构建 Phase 2 重试提示文本（含拒绝原因和上下文）。

    Args:
        rejected_items: 被拒绝的 Phase 2 结果列表（含 validation_errors）
        candidates_map: {candidate_id: candidate_dict}
        max_content_chars: 内容摘要最大字符数

    Returns:
        str: 格式化的重试文本
    """
    lines = []
    for i, item in enumerate(rejected_items):
        cid = item.get("candidate_id", "")
        c = candidates_map.get(cid, {})
        content = c.get("content", "")
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."
        rejected_query = item.get("retrieval_query", "")
        errors = item.get("validation_errors", [])
        errors_str = "; ".join(errors) if errors else "未知原因"
        target_fact = item.get("target_fact", "")
        target_label = item.get("target_label", "")
        modifiers = item.get("allowed_modifiers", [])
        modifiers_str = ", ".join(modifiers) if modifiers else "（无限定词）"
        forbidden = item.get("forbidden_concepts", [])
        forbidden_str = ", ".join(forbidden) if forbidden else "（无禁止概念）"

        retrieval_intent = item.get("retrieval_intent", "")
        intent_line = f"检索意图: {retrieval_intent}\n" if retrieval_intent else ""

        lines.append(
            f"候选 {i+1} — ID: {cid}\n"
            f"上次生成: {rejected_query}\n"
            f"拒绝原因: {errors_str}\n"
            f"目标事实（证据锚点）: {target_fact}\n"
            f"{intent_line}"
            f"标签: {target_label}\n"
            f"允许的限定词: {modifiers_str}\n"
            f"禁止混入的概念: {forbidden_str}\n"
            f"内容: {content}"
        )
    return "\n\n".join(lines)


def _phase2_generate_document(doc_name, planned_items, candidates_map,
                               api_key, base_url, model, timeout=60):
    """Phase 2: 为单个文档生成检索查询。

    Args:
        doc_name: 文档名称
        planned_items: Phase 1 输出的规划列表（含 query_style, target_fact, allowed_modifiers, forbidden_concepts）
        candidates_map: {candidate_id: candidate_dict}
        api_key, base_url, model: LLM 配置
        timeout: 超时秒数

    Returns:
        (questions, errors)
        questions: list[dict]，每项含 candidate_id, retrieval_query, target_label,
                   query_style, target_fact, allowed_modifiers, forbidden_concepts,
                   validation_status, validation_errors, generation_plan, _candidate
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

    # 校验并构建 question dicts（含 fail-closed + groundedness 校验）
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
        query_style = planned.get("query_style", "semantic")

        # ── Fail-closed 校验 ──
        query_ok, query_errors = validate_retrieval_query(
            retrieval_query, query_style, target_label
        )

        # ── Groundedness 校验 ──
        content = candidate.get("content", "")
        # 构建允许的同义词映射（从 Phase 1 的 must_preserve_terms 和 allowed_modifiers）
        allowed_synonyms = {}
        for term in planned.get("must_preserve_terms", []):
            allowed_synonyms[term.lower()] = term
        for mod in planned.get("allowed_modifiers", []):
            allowed_synonyms[mod.lower()] = mod

        ground_ok, ground_errors = validate_groundedness(
            retrieval_query, content, allowed_synonyms
        )

        # 合并校验结果
        all_errors = query_errors + ground_errors
        validation_status = "passed" if (query_ok and ground_ok) else "rejected"
        validation_errors = all_errors

        if validation_status == "rejected":
            errors.append(f"candidate_id '{cid}' 校验失败: {'; '.join(all_errors)}")
            # 标记为 rejected 但仍加入（由调用方决定是否使用）
            # 调用方会通过 validation_status 过滤

        questions.append({
            "candidate_id": cid,
            "retrieval_query": retrieval_query,
            "target_label": target_label,
            "query_style": query_style,
            "target_fact": planned.get("target_fact", ""),
            "retrieval_intent": planned.get("retrieval_intent", ""),
            "allowed_modifiers": planned.get("allowed_modifiers", []),
            "forbidden_concepts": planned.get("forbidden_concepts", []),
            "validation_status": validation_status,
            "validation_errors": validation_errors,
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
                "expected_content": candidate.get("content", ""),
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
                               random_seed=None, doc_question_counts=None,
                               generation_diagnostics=None):
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
        generation_diagnostics: 生成诊断信息（含拒绝详情），写入 manifest
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
        if generation_diagnostics:
            manifest["generation_diagnostics"] = generation_diagnostics
        # 每题的 expected_segment_id 和 expected_content_hash 已在 question dict 中
        manifest_path.write_text(_json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path, filename, qs_id
