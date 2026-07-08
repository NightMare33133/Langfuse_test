"""
Question Generator module - generates evaluation questions from knowledge base files.

Uses OpenAI-compatible chat completions API via the existing call_llm() from judge.py.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from judge import call_llm

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "qgen_prompt.txt"
QUESTIONS_DIR = Path(__file__).parent / "data" / "questions"


def load_qgen_prompt_template():
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def truncate_content(content, max_chars=8000):
    """Truncate file content to fit within LLM context limits.

    Approximately 8000 chars ≈ 2000 tokens for Chinese text.
    """
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n...(内容过长，已截断)"


def build_qgen_prompt(content, num_questions=5, difficulty="混合", topic_hint=""):
    template = load_qgen_prompt_template()

    topic_hint_section = ""
    if topic_hint:
        topic_hint_section = f"- 主题方向：{topic_hint}"

    prompt = template.replace("{content}", content)
    prompt = prompt.replace("{num_questions}", str(num_questions))
    prompt = prompt.replace("{difficulty}", difficulty)
    prompt = prompt.replace("{topic_hint_section}", topic_hint_section)
    return prompt


def parse_qgen_response(text):
    """Parse LLM response as a JSON array of question objects.

    Handles cases where the LLM wraps JSON in markdown code blocks.
    """
    text = text.strip()

    # Strip markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try to find JSON array
    import re
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


def generate_questions(content, api_key, base_url, model,
                       num_questions=5, difficulty="混合",
                       topic_hint="", timeout=120):
    """Generate questions from content using LLM.

    Returns a list of question dicts.
    """
    truncated = truncate_content(content)
    prompt = build_qgen_prompt(truncated, num_questions, difficulty, topic_hint)

    response_text = call_llm(prompt, api_key, base_url, model, timeout=timeout)
    questions = parse_qgen_response(response_text)
    return questions


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
