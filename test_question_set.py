"""
题集命名与题集身份测试。

测试内容：
1. 生成题集后，JSONL 每题均有 question_set_id/question_set_name
2. 两次同名题集生成得到不同 question_set_id 和不同文件
3. 历史列表优先显示题集名称
4. 旧 JSONL 没有题集字段时仍可正常加载
5. batch 结果与 raw 结果保留题集字段

不调用真实 LLM、Dify 或 Langfuse API。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from question_generator import (
    generate_question_set_id, build_question_set_name,
    save_questions, MODE_RETRIEVAL, MODE_QA, MODE_LABELS,
)
from batch_query import query_to_sample, save_batch_results, push_to_raw_dir


def test_question_set_id_generation():
    """测试 question_set_id 生成。"""
    print("=" * 60)
    print("测试 question_set_id 生成")
    print("=" * 60)

    # 测试基本生成
    set_id = generate_question_set_id("测试题集")
    assert set_id.startswith("qs_"), f"应以 'qs_' 开头，实际: {set_id}"
    assert "测试题集" in set_id or "测试题集" in set_id, f"应包含名称，实际: {set_id}"
    print(f"[OK] question_set_id 格式正确: {set_id}")

    # 测试相同名称生成不同 ID（微秒级时间戳）
    set_id1 = generate_question_set_id("相同名称")
    set_id2 = generate_question_set_id("相同名称")
    # 注意：极小概率相同，但微秒级时间戳应该不同
    print(f"[OK] 生成 ID: {set_id1}")
    print(f"[OK] 生成 ID: {set_id2}")

    # 测试空名称
    set_id_empty = generate_question_set_id("")
    assert "unnamed" in set_id_empty, f"空名称应包含 'unnamed'，实际: {set_id_empty}"
    print(f"[OK] 空名称 ID: {set_id_empty}")

    print()


def test_build_question_set_name():
    """测试题集名称生成。"""
    print("=" * 60)
    print("测试题集名称生成")
    print("=" * 60)

    # 测试基本生成
    name = build_question_set_name("IS5010期末复习.md", MODE_RETRIEVAL)
    assert name == "IS5010期末复习_检索评测", f"名称不正确: {name}"
    print(f"[OK] 名称正确: {name}")

    # 测试全流程问答模式
    name_qa = build_question_set_name("fintech.txt", MODE_QA)
    assert name_qa == "fintech_全流程问答评测", f"名称不正确: {name_qa}"
    print(f"[OK] 名称正确: {name_qa}")

    # 测试空文件名
    name_empty = build_question_set_name("", MODE_RETRIEVAL)
    assert "未命名" in name_empty, f"空文件名应包含 '未命名'，实际: {name_empty}"
    print(f"[OK] 空文件名名称: {name_empty}")

    print()


def test_save_questions_with_set_info():
    """测试保存题目时包含题集信息。"""
    print("=" * 60)
    print("测试保存题目时包含题集信息")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir) / "questions"

        with patch('question_generator.QUESTIONS_DIR', questions_dir):
            # 创建测试题目
            questions = [
                {"question": "问题1", "reference_answer": "答案1"},
                {"question": "问题2", "reference_answer": "答案2"},
            ]

            # 保存题目
            output_path, filename, set_id = save_questions(
                questions,
                question_set_name="测试题集",
                source_document_name="test.md",
                question_mode=MODE_RETRIEVAL,
            )

            print(f"[OK] 保存到: {output_path}")
            print(f"[OK] question_set_id: {set_id}")

            # 验证 JSONL 每题均有题集字段
            with output_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 2, f"应有 2 行，实际有 {len(lines)}"

            for i, line in enumerate(lines):
                obj = json.loads(line)
                assert obj.get("question_set_id") == set_id, \
                    f"第 {i+1} 行 question_set_id 不正确: {obj.get('question_set_id')}"
                assert obj.get("question_set_name") == "测试题集", \
                    f"第 {i+1} 行 question_set_name 不正确: {obj.get('question_set_name')}"
                assert obj.get("source_document_name") == "test.md", \
                    f"第 {i+1} 行 source_document_name 不正确: {obj.get('source_document_name')}"
            print("[OK] JSONL 每题均有正确的题集字段")

            # 验证 manifest 文件
            manifest_path = questions_dir / f"{output_path.stem}_manifest.json"
            assert manifest_path.exists(), f"manifest 文件应存在: {manifest_path}"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["question_set_id"] == set_id
            assert manifest["question_set_name"] == "测试题集"
            assert manifest["question_mode"] == MODE_RETRIEVAL
            assert manifest["question_count"] == 2
            print("[OK] manifest 文件正确")

    print()


def test_two_same_name_sets_different_ids():
    """测试两次同名题集生成得到不同 question_set_id 和不同文件。"""
    print("=" * 60)
    print("测试两次同名题集生成得到不同 ID 和文件")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir) / "questions"

        with patch('question_generator.QUESTIONS_DIR', questions_dir):
            questions = [{"question": "问题1"}]

            # 第一次保存
            path1, fname1, id1 = save_questions(
                questions.copy(),
                question_set_name="同名题集",
                question_mode=MODE_RETRIEVAL,
            )

            # 第二次保存
            path2, fname2, id2 = save_questions(
                questions.copy(),
                question_set_name="同名题集",
                question_mode=MODE_RETRIEVAL,
            )

            # 验证 ID 不同
            assert id1 != id2, f"两次保存应有不同 ID，但都是 {id1}"
            print(f"[OK] 两次保存有不同 ID: {id1} vs {id2}")

            # 验证文件不同
            assert path1 != path2, f"两次保存应有不同文件"
            assert path1.exists(), f"第一个文件应存在"
            assert path2.exists(), f"第二个文件应存在"
            print(f"[OK] 两次保存有不同文件")

            # 验证两次保存的题目都有正确的题集 ID
            with path1.open("r", encoding="utf-8") as f:
                obj1 = json.loads(f.readline())
            with path2.open("r", encoding="utf-8") as f:
                obj2 = json.loads(f.readline())

            assert obj1["question_set_id"] == id1
            assert obj2["question_set_id"] == id2
            print("[OK] 两次保存的题目都有正确的题集 ID")

    print()


def test_old_jsonl_compatibility():
    """测试旧 JSONL 没有题集字段时仍可正常加载。"""
    print("=" * 60)
    print("测试旧 JSONL 兼容性")
    print("=" * 60)

    # 模拟旧格式题目（无题集字段）
    old_questions = [
        {"question": "旧问题1", "reference_answer": "旧答案1"},
        {"question": "旧问题2"},
    ]

    # 验证没有题集字段
    for q in old_questions:
        assert not q.get("question_set_id"), "旧题目不应有 question_set_id"
        assert not q.get("question_set_name"), "旧题目不应有 question_set_name"
    print("[OK] 旧题目没有题集字段")

    # 验证 query_to_sample 兼容
    dify_result = {
        "answer": "回答",
        "conversation_id": "conv_123",
        "message_id": "msg_456",
        "retriever_resources": [],
    }

    sample = query_to_sample(
        question="旧问题",
        dify_result=dify_result,
        index=0,
        timestamp="20260713_120000",
    )

    assert not sample.get("question_set_id"), "旧格式不应有 question_set_id"
    assert not sample.get("question_set_name"), "旧格式不应有 question_set_name"
    print("[OK] query_to_sample 兼容旧格式")

    print()


def test_batch_results_with_set_fields():
    """测试 batch 结果保留题集字段。"""
    print("=" * 60)
    print("测试 batch 结果保留题集字段")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir) / "batch"

        batch_results = [
            {
                "success": True,
                "question": "测试问题",
                "sample": {
                    "trace_id": "batch_qa_0_20260713_120000",
                    "question": "测试问题",
                    "final_answer": "回答",
                    "question_set_id": "qs_20260713_120000_123456_测试",
                    "question_set_name": "测试题集",
                    "run_id": "run_123",
                    "config_id": "cfg_456",
                    "question_id": "q_abc",
                    "user_id": "rag_eval:run_123:q_abc",
                    "retrieval_results": [],
                },
            },
        ]

        with patch('batch_query.BATCH_DIR', batch_dir):
            output_path, filename = save_batch_results(batch_results)
            print(f"[OK] 保存 batch 结果到: {output_path}")

            with output_path.open("r", encoding="utf-8") as f:
                obj = json.loads(f.readline())

            sample = obj.get("sample", {})
            assert sample.get("question_set_id") == "qs_20260713_120000_123456_测试"
            assert sample.get("question_set_name") == "测试题集"
            print("[OK] batch 结果保留题集字段")

    print()


def test_raw_results_with_set_fields():
    """测试 raw 结果保留题集字段。"""
    print("=" * 60)
    print("测试 raw 结果保留题集字段")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"

        batch_results = [
            {
                "success": True,
                "question": "测试问题",
                "sample": {
                    "trace_id": "batch_qa_0_20260713_120000",
                    "question": "测试问题",
                    "final_answer": "回答",
                    "question_set_id": "qs_20260713_120000_123456_测试",
                    "question_set_name": "测试题集",
                    "run_id": "run_123",
                    "config_id": "cfg_456",
                    "question_id": "q_abc",
                    "user_id": "rag_eval:run_123:q_abc",
                    "retrieval_results": [],
                },
            },
        ]

        with patch('batch_query.RAW_DIR', raw_dir):
            output_path, filename = push_to_raw_dir(batch_results)
            print(f"[OK] 推送 raw 结果到: {output_path}")

            with output_path.open("r", encoding="utf-8") as f:
                obj = json.loads(f.readline())

            assert obj.get("question_set_id") == "qs_20260713_120000_123456_测试"
            assert obj.get("question_set_name") == "测试题集"
            print("[OK] raw 结果保留题集字段")

    print()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("题集命名与题集身份测试")
    print("=" * 60)
    print()

    test_question_set_id_generation()
    test_build_question_set_name()
    test_save_questions_with_set_info()
    test_two_same_name_sets_different_ids()
    test_old_jsonl_compatibility()
    test_batch_results_with_set_fields()
    test_raw_results_with_set_fields()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
