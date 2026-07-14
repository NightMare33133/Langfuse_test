"""
Question Generator module - generates evaluation questions from knowledge base files.

Uses OpenAI-compatible chat completions API via the existing call_llm() from judge.py.

Chunking strategy:
  - Markdown files: split by headings (# ## ### ...), sub-split long sections by paragraphs
  - Plain text: split by double-newline paragraphs, group small ones together
  - Each chunk carries section_title + chunk_index for coverage control

Question generation strategy:
  - Proportional allocation: longer chunks get more questions (min 1 per chunk)
  - Per-chunk LLM calls with a chunk-aware prompt
  - Post-generation deduplication by question text similarity
"""

import json
import random
import re
import string
from datetime import datetime
from pathlib import Path

import pandas as pd

from judge import call_llm

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "qgen_prompt.txt"
RETRIEVAL_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "qgen_prompt_retrieval.txt"
QUESTIONS_DIR = Path(__file__).parent / "data" / "questions"

# 出题模式
MODE_RETRIEVAL = "retrieval"  # 检索评测模式
MODE_QA = "qa"                # 全流程问答评测模式
MODE_LABELS = {
    MODE_RETRIEVAL: "检索评测",
    MODE_QA: "全流程问答评测",
}


def generate_question_set_id(name: str = "") -> str:
    """生成唯一 question_set_id。

    格式: qs_<YYYYMMDD_HHMMSSffffff>_<slug>
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"{now.microsecond:06d}"

    if name:
        slug = re.sub(r'[^\w\u4e00-\u9fff]', '_', name.strip())
        slug = re.sub(r'_+', '_', slug).strip('_')[:20]
    else:
        slug = "unnamed"

    return f"qs_{timestamp}_{slug}"


def build_question_set_name(source_filename: str, mode: str) -> str:
    """根据源文件名和出题模式生成默认题集名称。

    例如: IS5010期末复习_检索评测
    """
    # 提取文件名（不含扩展名）
    stem = Path(source_filename).stem if source_filename else "未命名"
    mode_label = MODE_LABELS.get(mode, "未知模式")
    return f"{stem}_{mode_label}"

# Chunking parameters
MAX_CHUNK_CHARS = 3000   # Target max chars per chunk
MIN_CHUNK_CHARS = 200    # Merge chunks smaller than this
MAX_CHUNKS = 20          # Hard cap on number of chunks


# ========== Document Chunking ==========

def _split_markdown_sections(content):
    """Split markdown content by heading lines (# ## ### etc.).

    Returns list of (section_title, section_body) tuples.
    Heading level is stripped from the title for readability.
    """
    lines = content.split("\n")
    sections = []
    current_title = "(前言)"
    current_lines = []

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            # Save previous section
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    return sections


def _split_paragraphs(text):
    """Split text by double newlines (natural paragraph boundaries)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text):
    """Split Chinese/English text by sentence boundaries."""
    # Split on Chinese sentence endings, English periods followed by space, etc.
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def _make_chunk(section_title, text, chunk_index):
    """Create a chunk dict with metadata."""
    return {
        "section_title": section_title,
        "chunk_index": chunk_index,
        "text": text,
        "char_count": len(text),
    }


def _subsplit_long_text(text, section_title, start_index, max_chars=MAX_CHUNK_CHARS):
    """Sub-split a long text into smaller chunks by paragraphs, then sentences.

    Returns list of chunk dicts.
    """
    paragraphs = _split_paragraphs(text)
    chunks = []
    buffer = ""
    idx = start_index

    for para in paragraphs:
        if len(para) > max_chars:
            # Paragraph itself is too long — flush buffer, then split by sentences
            if buffer:
                chunks.append(_make_chunk(section_title, buffer.strip(), idx))
                idx += 1
                buffer = ""
            sentences = _split_sentences(para)
            sent_buf = ""
            for sent in sentences:
                if len(sent_buf) + len(sent) + 1 > max_chars and sent_buf:
                    chunks.append(_make_chunk(section_title, sent_buf.strip(), idx))
                    idx += 1
                    sent_buf = ""
                sent_buf += sent + " "
            if sent_buf.strip():
                chunks.append(_make_chunk(section_title, sent_buf.strip(), idx))
                idx += 1
        elif len(buffer) + len(para) + 2 > max_chars and buffer:
            # Buffer would overflow — flush
            chunks.append(_make_chunk(section_title, buffer.strip(), idx))
            idx += 1
            buffer = para
        else:
            buffer += ("\n\n" if buffer else "") + para

    if buffer.strip():
        chunks.append(_make_chunk(section_title, buffer.strip(), idx))
        idx += 1

    return chunks


def _merge_small_chunks(chunks, min_chars=MIN_CHUNK_CHARS):
    """Merge adjacent chunks that are too small, but only within the same section."""
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        # Only merge if same section AND previous chunk is small
        if prev["char_count"] < min_chars and prev["section_title"] == chunk["section_title"]:
            prev["text"] += "\n\n" + chunk["text"]
            prev["char_count"] = len(prev["text"])
        else:
            merged.append(chunk)

    # Check last chunk — only merge into previous if same section
    if merged and merged[-1]["char_count"] < min_chars and len(merged) > 1:
        last = merged[-1]
        prev = merged[-2]
        if prev["section_title"] == last["section_title"]:
            prev["text"] += "\n\n" + last["text"]
            prev["char_count"] = len(prev["text"])
            merged.pop()

    return merged


def chunk_document(content, max_chars=MAX_CHUNK_CHARS, max_chunks=MAX_CHUNKS):
    """Split document content into semantic chunks.

    Strategy:
      1. Try markdown heading split
      2. For each section, if too long, sub-split by paragraphs
      3. For plain text (no headings found), split by paragraphs directly
      4. Merge tiny chunks, cap total count

    Returns list of chunk dicts with keys:
      - section_title: str
      - chunk_index: int
      - text: str
      - char_count: int
    """
    # Step 1: Try markdown split
    sections = _split_markdown_sections(content)

    # If only 1 section found and it's the default "(前言)", treat as plain text
    is_plain = len(sections) == 1 and sections[0][0] == "(前言)"

    raw_chunks = []
    idx = 0

    if is_plain:
        # Plain text: split by paragraphs
        paragraphs = _split_paragraphs(content)
        buffer = ""
        for para in paragraphs:
            if len(para) > max_chars:
                if buffer:
                    raw_chunks.append(_make_chunk("(正文)", buffer.strip(), idx))
                    idx += 1
                    buffer = ""
                # Sub-split this long paragraph
                sub = _subsplit_long_text(para, "(正文)", idx, max_chars)
                raw_chunks.extend(sub)
                idx += len(sub)
            elif len(buffer) + len(para) + 2 > max_chars and buffer:
                raw_chunks.append(_make_chunk("(正文)", buffer.strip(), idx))
                idx += 1
                buffer = para
            else:
                buffer += ("\n\n" if buffer else "") + para
        if buffer.strip():
            raw_chunks.append(_make_chunk("(正文)", buffer.strip(), idx))
    else:
        # Markdown: process each section
        for title, body in sections:
            body = body.strip()
            if not body:
                continue
            if len(body) <= max_chars:
                raw_chunks.append(_make_chunk(title, body, idx))
                idx += 1
            else:
                sub = _subsplit_long_text(body, title, idx, max_chars)
                raw_chunks.extend(sub)
                idx += len(sub)

    # Step 2: Merge small chunks
    chunks = _merge_small_chunks(raw_chunks)

    # Step 3: Re-index and cap
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    if len(chunks) > max_chunks:
        # Merge excess chunks into neighbors
        while len(chunks) > max_chunks:
            # Find smallest chunk and merge into its neighbor
            smallest = min(range(len(chunks)), key=lambda i: chunks[i]["char_count"])
            if smallest < len(chunks) - 1:
                chunks[smallest]["text"] += "\n\n" + chunks[smallest + 1]["text"]
                chunks[smallest]["char_count"] = len(chunks[smallest]["text"])
                chunks.pop(smallest + 1)
            elif smallest > 0:
                chunks[smallest - 1]["text"] += "\n\n" + chunks[smallest]["text"]
                chunks[smallest - 1]["char_count"] = len(chunks[smallest - 1]["text"])
                chunks.pop(smallest)
            else:
                break
        for i, c in enumerate(chunks):
            c["chunk_index"] = i

    return chunks


# ========== Question Allocation ==========

def allocate_questions(chunks, total_questions):
    """Allocate question count to each chunk proportionally to char_count.

    Uses the "largest remainder" method for fair distribution:
      1. Calculate raw proportional share for each chunk
      2. Floor each value (minimum 1)
      3. Distribute remaining questions to chunks with the largest fractional remainders

    This ensures smaller chunks don't get disproportionately many questions.
    """
    if not chunks:
        return []

    n = len(chunks)
    if n >= total_questions:
        # More chunks than questions — give 1 to each, starting from beginning
        allocation = [1] * total_questions + [0] * (n - total_questions)
        return allocation

    total_chars = sum(c["char_count"] for c in chunks)
    if total_chars == 0:
        base = max(1, total_questions // n)
        return [base] * n

    # Raw proportional share
    raw = [(c["char_count"] / total_chars) * total_questions for c in chunks]

    # Floor with minimum 1
    allocation = [max(1, int(r)) for r in raw]

    # Distribute remaining by largest remainder
    diff = total_questions - sum(allocation)
    if diff > 0:
        remainders = [(r - int(r), i) for i, r in enumerate(raw)]
        remainders.sort(key=lambda x: x[0], reverse=True)
        for j in range(diff):
            allocation[remainders[j][1]] += 1
    elif diff < 0:
        # Over-allocated — remove from chunks with smallest remainder (and >1)
        remainders = [(r - int(r), i) for i, r in enumerate(raw)]
        remainders.sort(key=lambda x: x[0])
        for j in range(-diff):
            idx = remainders[j][1]
            if allocation[idx] > 1:
                allocation[idx] -= 1
            else:
                # Find next chunk with >1
                for k in range(j + 1, len(remainders)):
                    alt = remainders[k][1]
                    if allocation[alt] > 1:
                        allocation[alt] -= 1
                        break

    return allocation


# ========== Prompt Building ==========

def load_qgen_prompt_template(mode=MODE_QA):
    """Load prompt template based on mode.

    Args:
        mode: MODE_RETRIEVAL for retrieval testing, MODE_QA for full QA evaluation
    """
    if mode == MODE_RETRIEVAL:
        return RETRIEVAL_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def build_qgen_prompt(content, num_questions=5, difficulty="混合", topic_hint="",
                      section_title=None, chunk_context=None, mode=MODE_QA):
    """Build prompt for a single chunk.

    Args:
        content: The chunk text
        num_questions: How many questions to generate for this chunk
        difficulty: Difficulty preference
        topic_hint: Optional topic direction
        section_title: Name of the section this chunk belongs to
        chunk_context: Brief description of document structure for context
        mode: MODE_RETRIEVAL for retrieval testing, MODE_QA for full QA evaluation
    """
    template = load_qgen_prompt_template(mode)

    topic_hint_section = ""
    if topic_hint:
        topic_hint_section = f"- 主题方向：{topic_hint}"

    section_context = ""
    if section_title:
        section_context = f"\n当前章节：「{section_title}」"
        if chunk_context:
            section_context += f"\n文档整体结构：{chunk_context}"

    # 根据当前 chunk 分配到的题目数，动态调整出题要求
    if mode == MODE_RETRIEVAL:
        # 检索模式：强调质量优先，允许减少数量
        if num_questions <= 1:
            coverage_instruction = "- 当前片段只需生成 1 道题，请聚焦于该片段中最核心的单跳检索题"
        else:
            coverage_instruction = f"- 当前片段目标生成 {num_questions} 道单跳检索题，如果适合的知识点不足，减少数量即可，不要凑数"
    else:
        # 问答模式：正常覆盖
        if num_questions <= 1:
            coverage_instruction = "- 当前片段只需生成 1 道题，请聚焦于该片段中最核心、最有考查价值的知识点"
        else:
            coverage_instruction = f"- 当前片段需生成 {num_questions} 道题，如果涉及多个知识点，尽量覆盖不同知识点出题"

    prompt = template.replace("{content}", content)
    prompt = prompt.replace("{num_questions}", str(num_questions))
    prompt = prompt.replace("{difficulty}", difficulty)
    prompt = prompt.replace("{topic_hint_section}", topic_hint_section)
    prompt = prompt.replace("{section_context}", section_context)
    prompt = prompt.replace("{coverage_instruction}", coverage_instruction)
    return prompt


# ========== Response Parsing ==========

def parse_qgen_response(text):
    """Parse LLM response as a JSON array of question objects.

    Handles cases where the LLM wraps JSON in markdown code blocks.
    """
    text = text.strip()

    # Strip markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try to find JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"LLM response does not contain JSON array: {text[:300]}")

    questions = json.loads(match.group(0))

    if not isinstance(questions, list):
        raise ValueError("LLM response is not a JSON array")

    # Validate and normalize each question
    normalized = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        normalized.append({
            "question": q.get("question", ""),
            "reference_answer": q.get("reference_answer", ""),
            "source_excerpt": q.get("source_excerpt", ""),
            "difficulty": q.get("difficulty", ""),
            "topic": q.get("topic", ""),
        })

    if not normalized:
        raise ValueError("No valid questions parsed from LLM response")

    return normalized


# ========== Deduplication ==========

def _normalize_for_dedup(text):
    """Normalize question text for comparison."""
    return re.sub(r"\s+", "", text.strip().lower())


def deduplicate_questions(questions):
    """Remove near-duplicate questions, keeping the first occurrence.

    Also removes questions with empty question text.
    """
    seen = set()
    unique = []
    for q in questions:
        qtext = q.get("question", "").strip()
        if not qtext:
            continue
        key = _normalize_for_dedup(qtext)
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ========== Strategy Selection ==========

def choose_strategy(content):
    """根据文档特征自动选择生成策略。

    规则（可读、非黑箱）：
      - 字符数 < 3000 → fast（文档很短，一次调用足够）
      - 3000 ≤ 字符数 < 15000 且 markdown section 数 ≤ 3 → fast（结构简单）
      - 3000 ≤ 字符数 < 15000 且 markdown section 数 > 3 → balanced（有结构，适度覆盖）
      - 15000 ≤ 字符数 ≤ 50000 → balanced（中等长度，平衡速度和覆盖）
      - 字符数 > 50000 → deep（长文档需要完整覆盖）

    Returns:
        "fast" | "balanced" | "deep"
    """
    char_count = len(content)
    sections = _split_markdown_sections(content)
    is_plain = len(sections) == 1 and sections[0][0] == "(前言)"
    section_count = 0 if is_plain else len(sections)

    if char_count < 3000:
        return "fast"
    if char_count < 15000:
        return "fast" if section_count <= 3 else "balanced"
    if char_count <= 50000:
        return "balanced"
    return "deep"


# ========== Per-Chunk Generation Helper (shared by all strategies) ==========

def _generate_from_chunks(chunks, num_questions, difficulty, topic_hint,
                          api_key, base_url, model, timeout, progress_callback,
                          mode=MODE_QA):
    """对已切分的 chunks 执行 allocate → per-chunk LLM call → dedup → 多样性裁剪。

    所有策略共用此函数，仅 chunks 的来源和参数不同。
    """
    if not chunks:
        raise ValueError("文档内容为空，无法生成题目")

    allocation = allocate_questions(chunks, num_questions)

    all_titles = list(dict.fromkeys(c["section_title"] for c in chunks))
    chunk_context = "、".join(all_titles[:10])
    if len(all_titles) > 10:
        chunk_context += f"等共{len(all_titles)}个章节"

    all_questions = []
    for i, (chunk, n_questions) in enumerate(zip(chunks, allocation)):
        if n_questions <= 0:
            continue

        if progress_callback:
            progress_callback(i, len(chunks), chunk["section_title"])

        prompt = build_qgen_prompt(
            chunk["text"],
            num_questions=n_questions,
            difficulty=difficulty,
            topic_hint=topic_hint,
            section_title=chunk["section_title"],
            chunk_context=chunk_context,
            mode=mode,
        )

        try:
            response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)
            questions = parse_qgen_response(response_text)
            for q in questions:
                q["source_section"] = chunk["section_title"]
                q["chunk_index"] = chunk["chunk_index"]
                q["question_mode"] = mode  # 透传出题模式
            all_questions.extend(questions)
        except Exception as e:
            print(f"  ⚠️ 章节「{chunk['section_title']}」出题失败: {e}")
            continue

    if not all_questions:
        raise ValueError("所有章节均出题失败，请检查 API 配置或文档内容")

    return _deduplicate_and_trim(all_questions, num_questions)


def _deduplicate_and_trim(all_questions, num_questions):
    """去重 + 多样性裁剪，供所有策略共用。"""
    unique_questions = deduplicate_questions(all_questions)

    if len(unique_questions) > num_questions:
        by_section = {}
        for q in unique_questions:
            sec = q.get("source_section", "")
            by_section.setdefault(sec, []).append(q)

        diversified = []
        section_keys = list(by_section.keys())
        si = 0
        while len(diversified) < num_questions:
            if not any(by_section.values()):
                break
            sec = section_keys[si % len(section_keys)]
            if by_section[sec]:
                diversified.append(by_section[sec].pop(0))
            si += 1
            if si > len(section_keys) * num_questions:
                break
        unique_questions = diversified

    for q in unique_questions:
        q.pop("source_section", None)
        q.pop("chunk_index", None)

    return unique_questions


# ========== Strategy Implementations ==========

# Fast 模式：截取文档前部内容，1 次 LLM 调用
_FAST_MAX_CHARS = 6000

def _generate_fast(content, num_questions, difficulty, topic_hint,
                   api_key, base_url, model, timeout, progress_callback, mode=MODE_QA):
    """极速模式 — 只调用 1 次 LLM。

    如果文档有 ≥3 个 markdown section，取前 3 个 section 合并；
    否则截取前 _FAST_MAX_CHARS 字符。
    """
    sections = _split_markdown_sections(content)
    is_plain = len(sections) == 1 and sections[0][0] == "(前言)"

    if not is_plain and len(sections) >= 3:
        # 取前 3 个 section，保留标题
        parts = []
        for title, body in sections[:3]:
            body = body.strip()
            if body:
                parts.append(f"## {title}\n\n{body}")
        text = "\n\n".join(parts)
        section_title = "、".join(t for t, _ in sections[:3])
    else:
        text = content[:_FAST_MAX_CHARS]
        section_title = "文档前部"

    chunk = _make_chunk(section_title, text, 0)

    return _generate_from_chunks(
        [chunk], num_questions, difficulty, topic_hint,
        api_key, base_url, model, timeout, progress_callback, mode=mode,
    )


# Balanced 模式：适度切分，3~5 次 LLM 调用
_BALANCED_MAX_CHARS = 6000
_BALANCED_MAX_CHUNKS = 5

def _generate_balanced(content, num_questions, difficulty, topic_hint,
                       api_key, base_url, model, timeout, progress_callback, mode=MODE_QA):
    """标准模式 — 控制在 3~5 次 LLM 调用。

    使用更大的 chunk 尺寸和更少的 chunk 上限，平衡速度和覆盖。
    """
    chunks = chunk_document(content, max_chars=_BALANCED_MAX_CHARS, max_chunks=_BALANCED_MAX_CHUNKS)
    return _generate_from_chunks(
        chunks, num_questions, difficulty, topic_hint,
        api_key, base_url, model, timeout, progress_callback, mode=mode,
    )


# Deep 模式：完整切分，当前逻辑
def _generate_deep(content, num_questions, difficulty, topic_hint,
                   api_key, base_url, model, timeout, progress_callback, mode=MODE_QA):
    """深度模式 — 完整切分，覆盖最全面。

    使用默认 chunk_document 参数（max_chars=3000, max_chunks=20），
    每个 chunk 单独调用 LLM，最后去重汇总。
    """
    chunks = chunk_document(content)
    return _generate_from_chunks(
        chunks, num_questions, difficulty, topic_hint,
        api_key, base_url, model, timeout, progress_callback, mode=mode,
    )


# ========== Main Entry Point ==========

_STRATEGY_MAP = {
    "fast": _generate_fast,
    "balanced": _generate_balanced,
    "deep": _generate_deep,
}

STRATEGY_LABELS = {
    "fast": "极速",
    "balanced": "标准",
    "deep": "深度",
    "auto": "自动",
}


def generate_questions(content, api_key, base_url, model,
                       num_questions=5, difficulty="混合",
                       topic_hint="", timeout=120,
                       progress_callback=None, strategy="auto", mode=MODE_QA):
    """Generate questions from content using LLM.

    Args:
        content: Full document text
        api_key, base_url, model: LLM API config
        num_questions: Total number of questions to generate
        difficulty: Difficulty preference
        topic_hint: Optional topic direction
        timeout: LLM request timeout in seconds
        progress_callback: Optional callback(chunk_index, total_chunks, chunk_title)
        strategy: "auto" | "fast" | "balanced" | "deep"
        mode: MODE_RETRIEVAL for retrieval testing, MODE_QA for full QA evaluation

    Returns:
        List of question dicts (deduplicated).
    """
    # 自动模式：根据文档特征选择策略
    if strategy == "auto":
        strategy = choose_strategy(content)

    gen_fn = _STRATEGY_MAP.get(strategy)
    if gen_fn is None:
        raise ValueError(f"未知策略: {strategy}，可选: {list(_STRATEGY_MAP.keys())}")

    return gen_fn(
        content, num_questions, difficulty, topic_hint,
        api_key, base_url, model, timeout, progress_callback, mode=mode,
    )


# ========== Save / Export ==========

def save_questions(questions, filename=None, question_set_id=None,
                   question_set_name=None, source_document_name=None,
                   question_mode=None):
    """Save questions to JSONL file in data/questions/.

    Args:
        questions: 题目列表
        filename: 可选文件名，默认自动生成
        question_set_id: 题集 ID（可选，自动生成）
        question_set_name: 题集名称（可选）
        source_document_name: 源文档名称（可选）
        question_mode: 出题模式（可选）

    Returns:
        (output_path, filename, question_set_id)
    """
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 生成 question_set_id
    if question_set_id is None:
        question_set_id = generate_question_set_id(question_set_name or "")

    # 生成文件名
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if question_set_name:
            slug = re.sub(r'[^\w\u4e00-\u9fff]', '_', question_set_name.strip())
            slug = re.sub(r'_+', '_', slug).strip('_')[:20]
            filename = f"questions_{slug}_{ts}.jsonl"
        else:
            filename = f"questions_{ts}.jsonl"

    output_path = QUESTIONS_DIR / filename

    # 确保文件名唯一
    if output_path.exists():
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        stem = output_path.stem
        output_path = QUESTIONS_DIR / f"{stem}_{suffix}.jsonl"
        filename = output_path.name

    # 为每道题添加题集字段
    for q in questions:
        q["question_set_id"] = question_set_id
        if question_set_name:
            q["question_set_name"] = question_set_name
        if source_document_name:
            q["source_document_name"] = source_document_name

    # 写入文件
    with output_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 验证文件是否写入成功
    if not output_path.exists():
        raise IOError(f"文件写入失败: {output_path}")

    # 保存题集 manifest
    manifest = {
        "question_set_id": question_set_id,
        "question_set_name": question_set_name or "未命名题集",
        "question_mode": question_mode or "",
        "source_document_name": source_document_name or "",
        "question_count": len(questions),
        "created_at": datetime.now().isoformat(),
        "filename": output_path.name,
    }
    manifest_path = QUESTIONS_DIR / f"{output_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[save_questions] 已保存 {len(questions)} 道题目到: {output_path}")
    print(f"[save_questions] question_set_id: {question_set_id}")

    return output_path, filename, question_set_id


def export_csv_bytes(questions):
    """Export questions as CSV bytes (UTF-8 with BOM, Excel-friendly)."""
    df = pd.DataFrame(questions)
    return df.to_csv(index=False).encode("utf-8-sig")


def export_json_bytes(questions):
    """Export questions as formatted JSON bytes."""
    return json.dumps(questions, ensure_ascii=False, indent=2).encode("utf-8")
