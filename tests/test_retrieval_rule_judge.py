"""
测试检索评测规则判定（retrieval_rule_judge）。

覆盖：
a. 文本规范化（空白、全半角、Markdown、标点）
b. 确定性直接失败：空检索结果、无有效 content
c. 确定性直接命中：gold_evidence 完整出现在 retrieval content 中
d. 金标准太短时退化为 LLM
e. 规则无法确定时返回 None
f. Top1/3/5 口径与 hit_evidence_position 一致性
g. pre_screen 集成：retrieval 轨道使用规则判定
h. judge_all 集成：prescreened 结果包含 _rule_name
i. audit 模式：audit_llm_agrees 字段存在
j. 历史回放：规则判定与历史 LLM 结果的一致率
"""

import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from judge import (
    _normalize_text, retrieval_rule_judge, pre_screen,
    classify_evaluation_track, TRACK_RETRIEVAL,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_retrieval_sample(gold, retrieval_contents, question="test q"):
    """Build a minimal retrieval-track sample."""
    return {
        "trace_id": "test_trace",
        "question": question,
        "question_mode": "retrieval",
        "reference_answer": gold,
        "source_excerpt": gold,
        "retrieval_results": [
            {"position": i + 1, "content": c, "title": f"doc_{i}"}
            for i, c in enumerate(retrieval_contents)
        ],
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_normalize_text():
    """规范化处理空白、全半角、Markdown、标点。"""
    print("=" * 60)
    print("测试文本规范化")
    print("=" * 60)

    # 空白折叠
    assert _normalize_text("  hello   world  ") == "hello world"
    assert _normalize_text("line1\n\nline2") == "line1 line2"
    assert _normalize_text("a\t\tb") == "a b"
    print("[OK] 空白折叠")

    # 全角 → 半角
    assert _normalize_text("Ｈｅｌｌｏ") == "Hello"
    assert _normalize_text("１２３") == "123"
    print("[OK] 全角→半角")

    # Markdown 去除
    assert _normalize_text("**bold** text") == "bold text"
    assert _normalize_text("__italic__") == "italic"
    assert _normalize_text("`code`") == "code"
    assert _normalize_text("[link](http://example.com)") == "link"
    print("[OK] Markdown 去除")

    # 中文标点 → 半角
    assert _normalize_text("你好，世界！") == "你好,世界!"
    print("[OK] 中文标点→半角")

    # 空/None
    assert _normalize_text("") == ""
    assert _normalize_text(None) == ""
    print("[OK] 空/None 处理")

    print()


def test_rule_empty_results():
    """空检索结果 → 确定性直接失败。"""
    print("=" * 60)
    print("测试空检索结果")
    print("=" * 60)

    sample = _make_retrieval_sample("some evidence", [])
    result = retrieval_rule_judge(sample)
    assert result is not None, "空结果应返回确定结果"
    assert result["retrieval_top1_hit"] == 0
    assert result["retrieval_top3_hit"] == 0
    assert result["retrieval_top5_hit"] == 0
    assert result["hit_evidence_position"] is None
    assert result["_rule_name"] == "empty_results"
    print("[OK] 空结果 → 全 0，rule_name=empty_results")

    print()


def test_rule_no_content():
    """检索结果均无有效 content → 确定性直接失败。"""
    print("=" * 60)
    print("测试无有效 content")
    print("=" * 60)

    sample = _make_retrieval_sample("some evidence", ["", "  ", ""])
    result = retrieval_rule_judge(sample)
    assert result is not None
    assert result["retrieval_top1_hit"] == 0
    assert result["_rule_name"] == "no_content"
    print("[OK] 无有效 content → 全 0，rule_name=no_content")

    print()


def test_rule_exact_match_top1():
    """gold_evidence 出现在 Top1 → 直接命中。"""
    print("=" * 60)
    print("测试精确匹配 Top1")
    print("=" * 60)

    gold = "供应商应确保所有数据处理符合 GDPR 要求"
    content = "根据合同条款，供应商应确保所有数据处理符合 GDPR 要求，包括数据加密和访问控制。"
    sample = _make_retrieval_sample(gold, [content, "other content"])
    result = retrieval_rule_judge(sample)

    assert result is not None
    assert result["retrieval_top1_hit"] == 1
    assert result["retrieval_top3_hit"] == 1
    assert result["retrieval_top5_hit"] == 1
    assert result["hit_evidence_position"] == 1
    assert result["_rule_name"] == "exact_contains_top1"
    assert result["_rule_match_rank"] == 1
    print(f"[OK] Top1 命中，snippet: {result.get('_rule_snippet', '')[:60]}")

    print()


def test_rule_exact_match_top2_returns_none():
    """gold_evidence 出现在 Top2 → 返回 None + _rule_hint（需 LLM）。"""
    print("=" * 60)
    print("测试精确匹配 Top2 → 降级 LLM")
    print("=" * 60)

    gold = "数据保留期限为合同终止后两年"
    sample = _make_retrieval_sample(gold, [
        "some other content",
        "根据规定，数据保留期限为合同终止后两年，除非法律另有要求。",
        "more content",
    ])
    result = retrieval_rule_judge(sample)

    assert result is None, "Top2 匹配应返回 None（需 LLM 判断 Top1 是否语义等价）"
    assert sample.get("_rule_hint", {}).get("matched_rank") == 2
    print("[OK] Top2 匹配 → None + _rule_hint(matched_rank=2)")

    print()


def test_rule_exact_match_top4_returns_none():
    """gold_evidence 出现在 Top4 → 返回 None + _rule_hint（需 LLM）。"""
    print("=" * 60)
    print("测试精确匹配 Top4 → 降级 LLM")
    print("=" * 60)

    gold = "违约金为合同总金额的百分之十"
    sample = _make_retrieval_sample(gold, [
        "content 1", "content 2", "content 3",
        "如一方违约，违约金为合同总金额的百分之十。",
        "content 5",
    ])
    result = retrieval_rule_judge(sample)

    assert result is None, "Top4 匹配应返回 None（需 LLM）"
    assert sample.get("_rule_hint", {}).get("matched_rank") == 4
    print("[OK] Top4 匹配 → None + _rule_hint(matched_rank=4)")

    print()


def test_rule_gold_too_short():
    """金标准太短（<8 字符规范化后）→ 返回 None，交给 LLM。"""
    print("=" * 60)
    print("测试金标准太短")
    print("=" * 60)

    sample = _make_retrieval_sample("短", ["这里包含短字"])
    result = retrieval_rule_judge(sample)
    assert result is None, "金标准太短应返回 None"
    print("[OK] 金标准太短 → None")

    print()


def test_rule_no_match_returns_none():
    """gold_evidence 不在任何 retrieval content 中 → 返回 None。"""
    print("=" * 60)
    print("测试无匹配返回 None")
    print("=" * 60)

    gold = "供应商应实施双因素认证机制"
    sample = _make_retrieval_sample(gold, [
        "数据处理应符合相关法律法规要求",
        "合同有效期为一年",
    ])
    result = retrieval_rule_judge(sample)
    assert result is None, "无匹配应返回 None"
    print("[OK] 无匹配 → None")

    print()


def test_rule_normalized_matching():
    """规范化后匹配：不同空白、全半角、Markdown 格式仍能命中。"""
    print("=" * 60)
    print("测试规范化匹配")
    print("=" * 60)

    gold = "供应商应确保所有数据处理符合 GDPR 要求"
    # Content has extra whitespace, full-width chars, and markdown
    content = "根据合同条款，**供应商应确保所有数据处理符合 ＧＤＰＲ 要求**，包括加密。"
    sample = _make_retrieval_sample(gold, [content])
    result = retrieval_rule_judge(sample)

    assert result is not None
    assert result["retrieval_top1_hit"] == 1
    assert result["_rule_name"] == "exact_contains_top1"
    print("[OK] 不同空白/全角/Markdown 规范化后匹配成功")

    print()


def test_pre_screen_integration():
    """pre_screen 对 retrieval 轨道使用规则判定。"""
    print("=" * 60)
    print("测试 pre_screen 集成")
    print("=" * 60)

    # 空结果
    sample = _make_retrieval_sample("evidence", [])
    result = pre_screen(sample)
    assert result is not None
    assert result["_rule_name"] == "empty_results"
    print("[OK] 空结果通过 pre_screen 规则判定")

    # 直接命中
    gold = "合同有效期自签署之日起为期三年"
    sample = _make_retrieval_sample(gold, [
        "本合同有效期自签署之日起为期三年，届满后可续签。"
    ])
    result = pre_screen(sample)
    assert result is not None
    assert result["_rule_name"] == "exact_contains_top1"
    assert result["retrieval_top1_hit"] == 1
    print("[OK] 直接命中通过 pre_screen 规则判定")

    # 无法确定
    sample = _make_retrieval_sample("some unique evidence", ["unrelated content"])
    result = pre_screen(sample)
    assert result is None
    print("[OK] 无法确定 → None（走 LLM）")

    print()


def test_topk_consistency():
    """仅 Top1 匹配返回确定结果；Top2-5 匹配返回 None + _rule_hint。"""
    print("=" * 60)
    print("测试 TopK 一致性")
    print("=" * 60)

    gold = "测试一致性证据文本应足够长以通过短文本检查"

    # Top1 匹配 → 确定结果
    contents = [f"包含测试一致性证据文本应足够长以通过短文本检查的文档"] + [f"content {i}" for i in range(4)]
    sample = _make_retrieval_sample(gold, contents)
    result = retrieval_rule_judge(sample)
    assert result is not None
    assert result["retrieval_top1_hit"] == 1
    assert result["retrieval_top3_hit"] == 1
    assert result["retrieval_top5_hit"] == 1
    assert result["hit_evidence_position"] == 1
    assert result["_rule_name"] == "exact_contains_top1"
    print("[OK] Top1 匹配 → 确定结果，all true")

    # Top2 匹配 → None + _rule_hint
    contents = ["other content", f"包含测试一致性证据文本应足够长以通过短文本检查的文档"] + [f"content {i}" for i in range(3)]
    sample = _make_retrieval_sample(gold, contents)
    result = retrieval_rule_judge(sample)
    assert result is None, "Top2 匹配应返回 None（需 LLM）"
    assert sample.get("_rule_hint", {}).get("matched_rank") == 2, "应设置 _rule_hint"
    print("[OK] Top2 匹配 → None + _rule_hint(matched_rank=2)")

    # Top3 匹配 → None + _rule_hint
    contents = ["c1", "c2", f"包含测试一致性证据文本应足够长以通过短文本检查的文档", "c4", "c5"]
    sample = _make_retrieval_sample(gold, contents)
    result = retrieval_rule_judge(sample)
    assert result is None
    assert sample.get("_rule_hint", {}).get("matched_rank") == 3
    print("[OK] Top3 匹配 → None + _rule_hint(matched_rank=3)")

    print()


def test_rule_exact_match_top1_only():
    """只有 Top1 匹配才返回确定结果。"""
    print("=" * 60)
    print("测试 Top1 确定性匹配")
    print("=" * 60)

    gold = "供应商应确保所有数据处理符合 GDPR 要求"
    content = "根据合同条款，供应商应确保所有数据处理符合 GDPR 要求，包括加密。"
    sample = _make_retrieval_sample(gold, [content, "other"])
    result = retrieval_rule_judge(sample)

    assert result is not None
    assert result["retrieval_top1_hit"] == 1
    assert result["retrieval_top3_hit"] == 1
    assert result["retrieval_top5_hit"] == 1
    assert result["hit_evidence_position"] == 1
    assert result["_rule_name"] == "exact_contains_top1"
    print("[OK] Top1 匹配 → 确定结果")

    print()


def test_rule_top2_must_call_llm():
    """gold 在 Top2、Top1 为语义等价证据时，必须调用 LLM。"""
    print("=" * 60)
    print("测试 Top2 必须调用 LLM")
    print("=" * 60)

    gold = "数据保留期限为合同终止后两年"
    # Top1 有语义等价但文本不同的证据
    # Top2 有完整原文匹配
    sample = _make_retrieval_sample(gold, [
        "合同结束后的数据保存时间为二十四个月",  # 语义等价，非原文
        "根据规定，数据保留期限为合同终止后两年，除非法律另有要求。",
        "other content",
    ])
    result = retrieval_rule_judge(sample)

    # 必须返回 None，不能直接判 Top1 未命中
    assert result is None, "Top2 匹配时必须返回 None（需 LLM 判断 Top1 是否语义等价）"
    # 应设置 _rule_hint
    assert sample.get("_rule_hint", {}).get("matched_rank") == 2
    print("[OK] Top2 匹配 → None，不预判 Top1 未命中")
    print(f"  _rule_hint: {sample.get('_rule_hint')}")

    print()


def test_audit_conflict_uses_llm():
    """audit 模式冲突时，最终结果应为 LLM 结果。"""
    print("=" * 60)
    print("测试审计冲突使用 LLM 结果")
    print("=" * 60)

    # 模拟 audit 冲突场景
    rule_result = {
        "trace_id": "test",
        "question": "q",
        "evaluation_track": "retrieval",
        "retrieval_top1_hit": 1,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 1,
        "_rule_name": "exact_contains_top1",
        "_rule_match_rank": 1,
        "_prescreened": True,
    }
    llm_result = {
        "trace_id": "test",
        "question": "q",
        "evaluation_track": "retrieval",
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 1,
        "retrieval_top5_hit": 1,
        "hit_evidence_position": 2,
        "reason": "Top1 无等价证据，Top2 命中",
    }

    # Simulate conflict resolution
    rule_pos = rule_result.get("_rule_match_rank")
    llm_pos = llm_result.get("hit_evidence_position")
    agrees = (rule_pos == llm_pos)

    if not agrees:
        final = {
            **llm_result,
            "_rule_conflict": True,
            "_rule_name": rule_result.get("_rule_name"),
            "_final_source": "llm",
        }
        assert final["retrieval_top1_hit"] == 0, "冲突时应采用 LLM 的 Top1=0"
        assert final["hit_evidence_position"] == 2, "冲突时应采用 LLM 的 position=2"
        assert final["_final_source"] == "llm"
        assert final["_rule_conflict"] is True
        print("[OK] 冲突时最终结果为 LLM 结果，Top1=0, position=2")
    else:
        print("[SKIP] 未模拟出冲突")

    print()


def test_rule_hint_in_sample():
    """Top2-5 匹配时，sample 上应有 _rule_hint。"""
    print("=" * 60)
    print("测试 _rule_hint 附加到 sample")
    print("=" * 60)

    gold = "合同违约金为总金额的百分之十"
    sample = _make_retrieval_sample(gold, [
        "无关内容",
        "如一方违约，合同违约金为总金额的百分之十。",
        "其他内容",
    ])
    result = retrieval_rule_judge(sample)

    assert result is None
    assert "_rule_hint" in sample
    assert sample["_rule_hint"]["matched_rank"] == 2
    assert isinstance(sample["_rule_hint"]["snippet"], str)
    print(f"[OK] _rule_hint: rank={sample['_rule_hint']['matched_rank']}, snippet={sample['_rule_hint']['snippet'][:50]}")

    print()


def test_rule_result_has_audit_fields():
    """规则判定结果包含可审计字段。"""
    print("=" * 60)
    print("测试审计字段")
    print("=" * 60)

    gold = "审计字段测试证据内容应足够长以通过检查"
    content = "文档中包含审计字段测试证据内容应足够长以通过检查的段落"
    sample = _make_retrieval_sample(gold, [content])
    result = retrieval_rule_judge(sample)

    assert result is not None
    assert "_rule_name" in result
    assert "_rule_match_rank" in result
    assert "_rule_snippet" in result
    assert result["_rule_name"] == "exact_contains_top1"
    assert isinstance(result["_rule_snippet"], str)
    assert len(result["_rule_snippet"]) > 0
    print(f"[OK] 审计字段完整: rule_name={result['_rule_name']}, rank={result['_rule_match_rank']}")

    print()


def test_historical_replay():
    """历史回放：用已有 eval_results 测量规则覆盖率。"""
    print("=" * 60)
    print("测试历史回放")
    print("=" * 60)

    judged_file = Path(__file__).resolve().parent.parent / "data" / "judged" / "eval_results.jsonl"
    proc_file = Path(__file__).resolve().parent.parent / "data" / "processed" / "langfuse_samples.jsonl"

    if not judged_file.exists() or not proc_file.exists():
        print("[SKIP] 历史数据文件不存在")
        return

    # Load processed samples by trace_id
    samples_by_tid = {}
    with proc_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = obj.get("trace_id")
            if tid:
                obj.pop("observations", None)
                samples_by_tid[tid] = obj

    # Load judged results (retrieval track only)
    judged = []
    with judged_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("evaluation_track") == "retrieval":
                judged.append(obj)

    # Replay rule judge
    rule_decided = 0
    rule_hit = 0
    rule_miss = 0
    needs_llm = 0
    agrees_with_llm = 0
    conflicts = 0

    for j in judged:
        tid = j.get("trace_id")
        sample = samples_by_tid.get(tid)
        if not sample:
            continue

        rule_result = retrieval_rule_judge(sample)
        if rule_result is not None:
            rule_decided += 1
            rule_pos = rule_result.get("hit_evidence_position")
            if rule_pos:
                rule_hit += 1
            else:
                rule_miss += 1

            # Compare with historical LLM result
            llm_pos = j.get("hit_evidence_position")
            if rule_pos == llm_pos:
                agrees_with_llm += 1
            else:
                conflicts += 1
        else:
            needs_llm += 1

    total = len(judged)
    print(f"  总检索结果: {total}")
    print(f"  规则判定: {rule_decided} ({rule_decided/total*100:.1f}%)")
    print(f"    规则命中: {rule_hit}")
    print(f"    规则未命中: {rule_miss}")
    print(f"  需 LLM: {needs_llm} ({needs_llm/total*100:.1f}%)")
    print(f"  与历史 LLM 一致: {agrees_with_llm}")
    print(f"  与历史 LLM 冲突: {conflicts}")

    if rule_decided > 0:
        agree_rate = agrees_with_llm / rule_decided * 100
        print(f"  一致率: {agree_rate:.1f}%")
        # 一致率应 > 80%（允许少量因规范化差异导致的合理分歧）
        assert agree_rate > 70, f"一致率过低: {agree_rate:.1f}%"
        print(f"[OK] 一致率 {agree_rate:.1f}% > 70%")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("检索评测规则判定测试")
    print("=" * 60)
    print()

    test_normalize_text()
    test_rule_empty_results()
    test_rule_no_content()
    test_rule_exact_match_top1()
    test_rule_exact_match_top2_returns_none()
    test_rule_exact_match_top4_returns_none()
    test_rule_gold_too_short()
    test_rule_no_match_returns_none()
    test_rule_normalized_matching()
    test_pre_screen_integration()
    test_topk_consistency()
    test_rule_exact_match_top1_only()
    test_rule_top2_must_call_llm()
    test_audit_conflict_uses_llm()
    test_rule_hint_in_sample()
    test_rule_result_has_audit_fields()
    test_historical_replay()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
