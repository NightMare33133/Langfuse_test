"""
验证脚本：测试题目模式（question_mode）的保存和历史加载正确性。

测试内容：
a. 保存 10 道 retrieval 题后，JSONL 有 10 行且每行 question_mode == "retrieval"
b. 历史加载同一文件后得到 10 道题，并识别为检索评测题
c. 保存和历史加载 qa 模式题目后，识别为全流程问答题
d. 没有 question_mode 的旧 JSONL 仍能加载

不调用真实 LLM、Dify 或 Langfuse API。
使用临时目录，不写入项目真实 data/questions。
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# 导入项目模块
from question_generator import (
    save_questions, QUESTIONS_DIR,
    MODE_RETRIEVAL, MODE_QA, MODE_LABELS
)


def create_test_questions(mode, count=10):
    """创建测试题目。"""
    questions = []
    for i in range(count):
        q = {
            "question": f"测试问题 {i+1}（{MODE_LABELS.get(mode, mode)}模式）",
            "reference_answer": f"测试答案 {i+1}",
            "source_excerpt": f"测试摘录 {i+1}",
            "difficulty": "基础" if mode == MODE_RETRIEVAL else "混合",
            "topic": f"测试主题 {i+1}",
            "question_mode": mode,
        }
        questions.append(q)
    return questions


def test_save_and_verify(mode, count=10, temp_dir=None):
    """测试保存并验证文件内容。使用临时目录。"""
    print(f"\n{'='*60}")
    print(f"测试保存 {count} 道 {MODE_LABELS.get(mode, mode)} 题目")
    print(f"{'='*60}")

    # 创建测试题目
    questions = create_test_questions(mode, count)

    # 使用临时目录保存
    with patch('question_generator.QUESTIONS_DIR', temp_dir):
        output_path, fname = save_questions(questions)
    print(f"保存到: {output_path}")

    # 验证文件存在
    assert output_path.exists(), f"文件不存在: {output_path}"
    print(f"[OK] 文件存在")

    # 验证文件行数
    with output_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == count, f"文件行数不正确: 期望 {count}，实际 {len(lines)}"
    print(f"[OK] 文件行数正确: {len(lines)} 行")

    # 验证每行的 question_mode
    for i, line in enumerate(lines):
        obj = json.loads(line.strip())
        assert obj.get("question_mode") == mode, \
            f"第 {i+1} 行 question_mode 不正确: 期望 '{mode}'，实际 '{obj.get('question_mode')}'"
    print(f"[OK] 每行 question_mode 都是 '{mode}'")

    return output_path


def test_history_load(filepath, expected_mode, expected_count):
    """测试历史加载功能。"""
    print(f"\n{'='*60}")
    print(f"测试历史加载: {filepath.name}")
    print(f"期望: {expected_count} 道 {MODE_LABELS.get(expected_mode, expected_mode)} 题目")
    print(f"{'='*60}")

    # 读取文件
    raw_lines = filepath.read_text(encoding="utf-8").strip().split("\n")
    questions_list = []
    for line in raw_lines:
        try:
            obj = json.loads(line)
            q = obj.get("question") or obj.get("query") or ""
            if q.strip():
                item = {"question": q.strip()}
                if obj.get("reference_answer"):
                    item["reference_answer"] = obj["reference_answer"]
                if obj.get("source_excerpt"):
                    item["source_excerpt"] = obj["source_excerpt"]
                if obj.get("question_mode"):
                    item["question_mode"] = obj["question_mode"]
                questions_list.append(item)
        except json.JSONDecodeError:
            continue

    # 验证题目数量
    assert len(questions_list) == expected_count, \
        f"题目数量不正确: 期望 {expected_count}，实际 {len(questions_list)}"
    print(f"[OK] 加载题目数量正确: {len(questions_list)} 道")

    # 统计 question_mode 分布
    mode_counts = {MODE_RETRIEVAL: 0, MODE_QA: 0, "unknown": 0}
    for q in questions_list:
        qm = q.get("question_mode", "")
        if qm == MODE_RETRIEVAL:
            mode_counts[MODE_RETRIEVAL] += 1
        elif qm == MODE_QA:
            mode_counts[MODE_QA] += 1
        else:
            mode_counts["unknown"] += 1

    print(f"模式分布: 检索评测={mode_counts[MODE_RETRIEVAL]}, 全流程问答={mode_counts[MODE_QA]}, 未知={mode_counts['unknown']}")

    # 验证模式识别
    if expected_mode == MODE_RETRIEVAL:
        assert mode_counts[MODE_RETRIEVAL] == expected_count, \
            f"检索评测题数量不正确: 期望 {expected_count}，实际 {mode_counts[MODE_RETRIEVAL]}"
        assert mode_counts[MODE_QA] == 0, f"不应有全流程问答题"
        mode_tag = "[检索评测题]"
    elif expected_mode == MODE_QA:
        assert mode_counts[MODE_QA] == expected_count, \
            f"全流程问答题数量不正确: 期望 {expected_count}，实际 {mode_counts[MODE_QA]}"
        assert mode_counts[MODE_RETRIEVAL] == 0, f"不应有检索评测题"
        mode_tag = "[全流程问答题]"
    else:
        assert mode_counts["unknown"] == expected_count, \
            f"未知模式题目数量不正确: 期望 {expected_count}，实际 {mode_counts['unknown']}"
        mode_tag = "[旧版/未知模式]"

    print(f"[OK] 模式识别正确: {mode_tag}")

    return questions_list


def test_old_format(temp_dir=None):
    """测试没有 question_mode 的旧 JSONL 格式。使用临时目录。"""
    print(f"\n{'='*60}")
    print(f"测试旧格式（无 question_mode）")
    print(f"{'='*60}")

    # 创建旧格式题目（没有 question_mode）
    questions = []
    for i in range(5):
        q = {
            "question": f"旧格式问题 {i+1}",
            "reference_answer": f"旧格式答案 {i+1}",
            "source_excerpt": f"旧格式摘录 {i+1}",
            "difficulty": "基础",
            "topic": f"旧格式主题 {i+1}",
            # 注意：没有 question_mode 字段
        }
        questions.append(q)

    # 保存到临时目录
    temp_path = temp_dir / "old_format.jsonl"
    with temp_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"保存到: {temp_path}")

    # 验证加载
    loaded_questions = test_history_load(temp_path, "unknown", 5)

    # 验证模式显示为"旧版/未知模式"
    mode_counts = {MODE_RETRIEVAL: 0, MODE_QA: 0, "unknown": 0}
    for q in loaded_questions:
        qm = q.get("question_mode", "")
        if qm == MODE_RETRIEVAL:
            mode_counts[MODE_RETRIEVAL] += 1
        elif qm == MODE_QA:
            mode_counts[MODE_QA] += 1
        else:
            mode_counts["unknown"] += 1

    assert mode_counts["unknown"] == 5, f"旧格式应全部识别为未知模式"
    print(f"[OK] 旧格式正确识别为 [旧版/未知模式]")


def main():
    """运行所有测试。使用临时目录，不写入项目真实目录。"""
    print("=" * 60)
    print("题目模式（question_mode）验证测试")
    print("=" * 60)

    # 使用临时目录
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"使用临时目录: {temp_dir}")

        # 测试 a: 保存 10 道 retrieval 题
        retrieval_path = test_save_and_verify(MODE_RETRIEVAL, 10, temp_dir)

        # 测试 b: 历史加载 retrieval 题
        test_history_load(retrieval_path, MODE_RETRIEVAL, 10)

        # 测试 c: 保存和历史加载 qa 题
        qa_path = test_save_and_verify(MODE_QA, 10, temp_dir)
        test_history_load(qa_path, MODE_QA, 10)

        # 测试 d: 旧格式 JSONL
        test_old_format(temp_dir)

    print(f"\n{'='*60}")
    print("[OK] 所有测试通过！临时目录已自动清理。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
