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
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from judge import call_llm

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "qgen_prompt.txt"
QUESTIONS_DIR = Path(__file__).parent / "data" / "questions"

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

    - Every chunk gets at least 1 question
    - Longer chunks get more
    - Total sums to exactly total_questions
    """
    if not chunks:
        return []

    n = len(chunks)
    if n >= total_questions:
        # More chunks than questions — give 1 to each, starting from beginning
        allocation = [1] * total_questions + [0] * (n - total_questions)
        return allocation

    # Proportional allocation with minimum 1
    total_chars = sum(c["char_count"] for c in chunks)
    if total_chars == 0:
        return [max(1, total_questions // n)] * n

    raw = [(c["char_count"] / total_chars) * total_questions for c in chunks]
    allocation = [max(1, int(r)) for r in raw]

    # Adjust to match total_questions
    diff = total_questions - sum(allocation)
    if diff > 0:
        # Need more — add to largest chunks
        order = sorted(range(n), key=lambda i: chunks[i]["char_count"], reverse=True)
        for i in order:
            if diff <= 0:
                break
            allocation[i] += 1
            diff -= 1
    elif diff < 0:
        # Need fewer — remove from chunks with >1
        order = sorted(range(n), key=lambda i: chunks[i]["char_count"])
        for i in order:
            if diff >= 0:
                break
            if allocation[i] > 1:
                allocation[i] -= 1
                diff += 1

    return allocation


# ========== Prompt Building ==========

def load_qgen_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def build_qgen_prompt(content, num_questions=5, difficulty="混合", topic_hint="",
                      section_title=None, chunk_context=None):
    """Build prompt for a single chunk.

    Args:
        content: The chunk text
        num_questions: How many questions to generate for this chunk
        difficulty: Difficulty preference
        topic_hint: Optional topic direction
        section_title: Name of the section this chunk belongs to
        chunk_context: Brief description of document structure for context
    """
    template = load_qgen_prompt_template()

    topic_hint_section = ""
    if topic_hint:
        topic_hint_section = f"- 主题方向：{topic_hint}"

    section_context = ""
    if section_title:
        section_context = f"\n当前章节：「{section_title}」"
        if chunk_context:
            section_context += f"\n文档整体结构：{chunk_context}"

    prompt = template.replace("{content}", content)
    prompt = prompt.replace("{num_questions}", str(num_questions))
    prompt = prompt.replace("{difficulty}", difficulty)
    prompt = prompt.replace("{topic_hint_section}", topic_hint_section)
    prompt = prompt.replace("{section_context}", section_context)
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


# ========== Main Entry Point ==========

def generate_questions(content, api_key, base_url, model,
                       num_questions=5, difficulty="混合",
                       topic_hint="", timeout=120,
                       progress_callback=None):
    """Generate questions from content using LLM with smart chunking.

    Args:
        content: Full document text
        api_key, base_url, model: LLM API config
        num_questions: Total number of questions to generate
        difficulty: Difficulty preference
        topic_hint: Optional topic direction
        timeout: LLM request timeout in seconds
        progress_callback: Optional callback(chunk_index, total_chunks, chunk_title)

    Returns:
        List of question dicts (deduplicated).
    """
    # Step 1: Chunk the document
    chunks = chunk_document(content)

    if not chunks:
        raise ValueError("文档内容为空，无法生成题目")

    # Step 2: Allocate questions per chunk
    allocation = allocate_questions(chunks, num_questions)

    # Step 3: Build chunk context string (list of all section titles)
    all_titles = list(dict.fromkeys(c["section_title"] for c in chunks))
    chunk_context = "、".join(all_titles[:10])
    if len(all_titles) > 10:
        chunk_context += f"等共{len(all_titles)}个章节"

    # Step 4: Generate questions per chunk
    all_questions = []
    template = load_qgen_prompt_template()

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
        )

        try:
            response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)
            questions = parse_qgen_response(response_text)
            # Tag each question with source section
            for q in questions:
                q["source_section"] = chunk["section_title"]
                q["chunk_index"] = chunk["chunk_index"]
            all_questions.extend(questions)
        except Exception as e:
            # Log but continue with other chunks
            print(f"  ⚠️ 章节「{chunk['section_title']}」出题失败: {e}")
            continue

    if not all_questions:
        raise ValueError("所有章节均出题失败，请检查 API 配置或文档内容")

    # Step 5: Deduplicate
    unique_questions = deduplicate_questions(all_questions)

    # Step 6: Trim to requested count (keep diverse topics)
    if len(unique_questions) > num_questions:
        # Round-robin from different sections to maximize diversity
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
            # Safety: break if all empty
            if si > len(section_keys) * num_questions:
                break
        unique_questions = diversified

    # Remove internal tracking fields from output
    for q in unique_questions:
        q.pop("source_section", None)
        q.pop("chunk_index", None)

    return unique_questions


# ========== Save / Export ==========

def save_questions(questions, filename=None):
    """Save questions to JSONL file in data/questions/.

    Returns (output_path, filename).
    """
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"questions_{ts}.jsonl"

    output_path = QUESTIONS_DIR / filename
    with output_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return output_path, filename


def export_csv_bytes(questions):
    """Export questions as CSV bytes (UTF-8 with BOM, Excel-friendly)."""
    df = pd.DataFrame(questions)
    return df.to_csv(index=False).encode("utf-8-sig")


def export_json_bytes(questions):
    """Export questions as formatted JSON bytes."""
    return json.dumps(questions, ensure_ascii=False, indent=2).encode("utf-8")
