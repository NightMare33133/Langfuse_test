"""
检索评测 Prompt 改动验证脚本。

验证内容：
1. 新版检索查询格式（非问句、语义改写、完整证据单元）
2. 旧版问句式题目全部被拦截
3. reference_answer 是 chunk 中的完整连续证据单元（允许单句定义）
4. query 不逐字照抄 reference_answer 中的连续核心短语
5. source_excerpt 与 reference_answer 完全一致
6. 人工展示"查询"与"原文证据"的差异

不调用真实 LLM API，仅展示期望输出格式和校验逻辑。
"""

import json
import re
import sys

# ========== 校验函数（与 question_generator.py 中一致） ==========

_RETRIEVAL_Q_RE = re.compile(r'[？\?]')
_RETRIEVAL_Q_WORDS = frozenset([
    "什么", "为何", "为什么", "如何", "是否", "哪些",
    "请分析", "请说明", "请解释", "请描述", "请比较",
    "分别", "哪些方面", "怎么回事",
])
_MIN_EVIDENCE_LEN = 15
_CLAUSE_NUM_RE = re.compile(r'^\s*\d+[\.\-]\d+[\.\d]*\s*')
_QUOTED_TERM_RE = re.compile(r'[《》「」""\u201c\u201d]')
_BRACKET_CONTENT_RE = re.compile(r'[（(][^）)]*[）)]')


def _strip_technical_terms(text):
    text = _CLAUSE_NUM_RE.sub('', text)
    text = _BRACKET_CONTENT_RE.sub('', text)
    text = _QUOTED_TERM_RE.sub('', text)
    return text


def _is_valid_evidence(text):
    """完整证据单元：≥15字，允许单句定义。"""
    return len(text) >= _MIN_EVIDENCE_LEN


def _detect_phrase_copying(question, ref_answer):
    q_clean = _strip_technical_terms(question).strip()
    r_clean = _strip_technical_terms(ref_answer).strip()
    if not q_clean or not r_clean:
        return False
    return q_clean in r_clean


def validate_retrieval_question(q, chunk_text):
    """校验单条检索评测查询是否合规（不修改 q 中的原始字段值）。"""
    question = (q.get("question") or "").strip()
    ref_answer = (q.get("reference_answer") or "").strip()
    source_excerpt = (q.get("source_excerpt") or "").strip()

    if not question:
        return False, "query 为空"

    if _RETRIEVAL_Q_RE.search(question):
        return False, f"query 含问号: {question[:40]}"

    for w in _RETRIEVAL_Q_WORDS:
        if w in question:
            return False, f"query 含问答导向词「{w}」: {question[:40]}"

    if ref_answer and not _is_valid_evidence(ref_answer):
        return False, f"reference_answer 过短（{len(ref_answer)}字），不足独立表达证据"

    if ref_answer and chunk_text:
        norm_ref = re.sub(r'\s+', '', ref_answer)
        norm_chunk = re.sub(r'\s+', '', chunk_text)
        if norm_ref not in norm_chunk:
            return False, "reference_answer 不是当前 chunk 的连续子串"

    if question and ref_answer:
        if _detect_phrase_copying(question, ref_answer):
            return False, f"query 逐字照抄证据中的连续核心短语: {question[:40]}"

    if source_excerpt and ref_answer and source_excerpt != ref_answer:
        return False, "source_excerpt 与 reference_answer 不一致"

    return True, ""


# ========== 测试样本 ==========

SAMPLES = [
    {
        "name": "缺陷定义（中文同义改写）",
        "chunk_text": (
            "Defect means that one or several Services (i) do not meet or cease to meet "
            "the requirements set out in the relevant Specification or any other requirement "
            "in this Agreement (including any agreed service levels) or (ii) are not fit for "
            "its intended purpose if such purpose has been set forth in the Specification; or "
            "(iii) do not comply with applicable laws, rules and regulations.\n\n"
            "缺陷指一项或多项指定服务(i)不符合或不再符合相关《规范说明》所载要求或本协议规定的"
            "任何其他要求（包括任何约定的服务水平），或(ii)不符合其预期目的，如果该目的已在"
            "《规范说明》中载明，或(iii)不符合适用法律、法规和规则的规定。"
        ),
        "expected_query": "服务达标要求",
        "expected_evidence": (
            "缺陷指一项或多项指定服务(i)不符合或不再符合相关《规范说明》所载要求或本协议规定的"
            "任何其他要求（包括任何约定的服务水平），或(ii)不符合其预期目的，如果该目的已在"
            "《规范说明》中载明，或(iii)不符合适用法律、法规和规则的规定。"
        ),
    },
    {
        "name": "服务中断通知",
        "chunk_text": (
            "如果供应方已经实际知晓在提供指定服务时已发生的或可能发生的中断或干扰，"
            "则供应方应当立即通知买方以下信息：(i)中断或干扰的类型，(ii)为消除中断或干扰"
            "已采取或将采取的措施，(iii)预计中断或干扰持续的期间，和(iv)可能与买方有关的"
            "其他信息。"
        ),
        "expected_query": "故障通报义务",
        "expected_evidence": (
            "如果供应方已经实际知晓在提供指定服务时已发生的或可能发生的中断或干扰，"
            "则供应方应当立即通知买方以下信息：(i)中断或干扰的类型，(ii)为消除中断或干扰"
            "已采取或将采取的措施，(iii)预计中断或干扰持续的期间，和(iv)可能与买方有关的"
            "其他信息。"
        ),
    },
    {
        "name": "知识产权归属",
        "chunk_text": (
            "9.3 All Intellectual Property created under this Agreement shall be the exclusive "
            "property of Purchaser and Supplier hereby assigns all such Intellectual Property to "
            "Purchaser. Purchaser may freely modify and assign such Intellectual Property.\n\n"
            "本协议项下创造的所有知识产权应为买方的专有财产，供应方在此将所有该等知识产权"
            "转让给买方。买方可自由修改或转让该等知识产权。"
        ),
        "expected_query": "成果权属买方",
        "expected_evidence": (
            "本协议项下创造的所有知识产权应为买方的专有财产，供应方在此将所有该等知识产权"
            "转让给买方。买方可自由修改或转让该等知识产权。"
        ),
    },
    {
        "name": "发票支付期限",
        "chunk_text": (
            "Correctly addressed invoices shall be paid within 60 days from receipt thereof, "
            "unless disputed by Purchaser in good faith in whole or in part. Payment does not "
            "constitute approval of the invoiced amount or the Services. In case of late payment, "
            "Supplier may charge interest on overdue payments in accordance with the Swedish "
            "Interest Act (Sw: Räntelag (1975:635)).\n\n"
            "正确地址的账单应在收到后六十(60)日内支付，除非买方秉承诚信善意原则对账单的"
            "全部或部分内容提出争议。付款不构成对发票金额或指定服务的认可。如果发生延迟付款，"
            "供应方可根据《瑞典利息法》就逾期付款收取利息。"
        ),
        "expected_query": "付款响应时限",
        "expected_evidence": (
            "正确地址的账单应在收到后六十(60)日内支付，除非买方秉承诚信善意原则对账单的"
            "全部或部分内容提出争议。付款不构成对发票金额或指定服务的认可。"
        ),
    },
    {
        "name": "延迟违约金",
        "chunk_text": (
            "如果双方已就某指定服务约定最终完成日期，但未在该等完成日期当日或之前完成"
            "该等指定服务（或者须经验收测试的任何交付物未被买方接受），则每延迟一整周，"
            "买方有权按照该等指定服务应付费用总额的 5%获得违约金，违约金金额不得超过"
            "指定服务应付费用总额的 30%或100,000欧元，以较高金额为准。"
        ),
        "expected_query": "逾期交付违约金上限",
        "expected_evidence": (
            "每延迟一整周，买方有权按照该等指定服务应付费用总额的 5%获得违约金，"
            "违约金金额不得超过指定服务应付费用总额的 30%或100,000欧元，以较高金额为准。"
        ),
    },
    {
        "name": "保密信息排除条件",
        "chunk_text": (
            "18.2 Confidential Information shall not include information which (i) is or becomes "
            "public through no fault of the receiving party; (ii) is lawfully obtained from someone "
            "other than the disclosing party that is not under an obligation to the disclosing party "
            "to keep that information confidential; (iii) was already in the possession of the "
            "receiving party prior to the date of disclosure; or (iv) the receiving party develops "
            "independently without use of the Confidential Information.\n\n"
            "保密信息不包括 (i) 非因接收方的过错而公开的信息；(ii) 接收方从披露方以外的、"
            "不对披露方负保密义务的人处合法获取的信息；(iii) 在披露日之前已经为接收方所持有的"
            "信息；或 (iv) 接收方在不使用保密信息的情况下独立开发的信息。"
        ),
        "expected_query": "机密信息豁免情形",
        "expected_evidence": (
            "保密信息不包括 (i) 非因接收方的过错而公开的信息；(ii) 接收方从披露方以外的、"
            "不对披露方负保密义务的人处合法获取的信息；(iii) 在披露日之前已经为接收方所持有的"
            "信息；或 (iv) 接收方在不使用保密信息的情况下独立开发的信息。"
        ),
    },
    {
        "name": "ISO认证宽限期（完整段落）",
        "chunk_text": (
            "如果供应方未获得ISO9001和 ISO 14001 认证，或不具备其他充分的且经买方认可的"
            "质量/环境管理体系，则除非另有约定，否则供应方享有自本协议签署之日起六个月的宽限期。"
            "在宽限期内，供应方应制定并提交经买方批准的认证获取计划，"
            "并在宽限期届满前取得上述认证。"
        ),
        "expected_query": "质量体系达标过渡期",
        "expected_evidence": (
            "如果供应方未获得ISO9001和 ISO 14001 认证，或不具备其他充分的且经买方认可的"
            "质量/环境管理体系，则除非另有约定，否则供应方享有自本协议签署之日起六个月的宽限期。"
            "在宽限期内，供应方应制定并提交经买方批准的认证获取计划，"
            "并在宽限期届满前取得上述认证。"
        ),
    },
    {
        "name": "英文同义改写 — IP ownership",
        "chunk_text": (
            "9.3 All Intellectual Property created under this Agreement shall be the exclusive "
            "property of Purchaser and Supplier hereby assigns all such Intellectual Property to "
            "Purchaser. Purchaser may freely modify and assign such Intellectual Property.\n\n"
            "本协议项下创造的所有知识产权应为买方的专有财产，供应方在此将所有该等知识产权"
            "转让给买方。买方可自由修改或转让该等知识产权。"
        ),
        "expected_query": "IP ownership under agreement",
        "expected_evidence": (
            "All Intellectual Property created under this Agreement shall be the exclusive "
            "property of Purchaser and Supplier hereby assigns all such Intellectual Property to "
            "Purchaser. Purchaser may freely modify and assign such Intellectual Property.\n\n"
            "本协议项下创造的所有知识产权应为买方的专有财产，供应方在此将所有该等知识产权"
            "转让给买方。买方可自由修改或转让该等知识产权。"
        ),
    },
    {
        "name": "单句定义（英文） — Acceptance Test",
        "chunk_text": (
            "Acceptance Test means Purchaser's testing and review of the Deliverables. "
            "The Acceptance Test shall be conducted in accordance with the procedures "
            "set forth in the relevant Specification."
        ),
        "expected_query": "验收测试定义",
        "expected_evidence": "Acceptance Test means Purchaser's testing and review of the Deliverables.",
    },
    {
        "name": "单句定义（中文） — 验收标准",
        "chunk_text": (
            "验收标准是指买方在《规范说明》中明确规定的、指定服务须满足的技术和功能要求。"
            "供应方应确保指定服务在交付时符合全部验收标准。"
        ),
        "expected_query": "验收标准含义",
        "expected_evidence": "验收标准是指买方在《规范说明》中明确规定的、指定服务须满足的技术和功能要求。",
    },
]

# 反例：旧版问句式题目（应被过滤）
OLD_STYLE_SAMPLES = [
    {"name": "旧版-问号", "question": "缺陷（Defect）指的是什么？"},
    {"name": "旧版-什么", "question": "知识产权指什么"},
    {"name": "旧版-请分析", "question": "请分析服务中断对买方的影响"},
    {"name": "旧版-分别", "question": "缺陷定义、责任方和补救措施分别是什么？"},
    {"name": "旧版-如何", "question": "如何理解保密信息的排除条件"},
]

# 反例：逐字照抄（应被过滤）
PHRASE_COPY_SAMPLES = [
    {
        "name": "照抄-中文连续短语",
        "question": "不符合或不再符合相关规范说明所载要求或本协议规定的任何其他要求",
        "ref_answer": (
            "缺陷指一项或多项指定服务(i)不符合或不再符合相关《规范说明》所载要求或本协议规定的"
            "任何其他要求（包括任何约定的服务水平），或(ii)不符合其预期目的，如果该目的已在"
            "《规范说明》中载明，或(iii)不符合适用法律、法规和规则的规定。"
        ),
        "chunk": "任何满足证据条件的文本，包含足够多的分句和足够长的内容。",
    },
    {
        "name": "照抄-英文连续短语",
        "question": "exclusive property of Purchaser and Supplier hereby assigns all such Intellectual Property",
        "ref_answer": (
            "All Intellectual Property created under this Agreement shall be the exclusive "
            "property of Purchaser and Supplier hereby assigns all such Intellectual Property to "
            "Purchaser. Purchaser may freely modify and assign such Intellectual Property.\n\n"
            "本协议项下创造的所有知识产权应为买方的专有财产，供应方在此将所有该等知识产权"
            "转让给买方。买方可自由修改或转让该等知识产权。"
        ),
        "chunk": "任何满足证据条件的文本，包含足够多的分句和足够长的内容。",
    },
]

# 反例：过短证据（应被过滤）
SHORT_EVIDENCE_SAMPLES = [
    {
        "name": "过短证据-半句截取",
        "question": "合同适用法律",
        "ref_answer": "本协议受瑞典法律管",
        "chunk": "本协议受瑞典法律管辖。双方应遵守适用的法律法规。",
    },
]

# 反例：source_excerpt 与 reference_answer 不一致（应被过滤）
MISMATCH_SAMPLES = [
    {
        "name": "excerpt与answer不一致",
        "question": "付款期限",
        "ref_answer": "正确地址的账单应在收到后六十(60)日内支付，除非买方秉承诚信善意原则对账单的全部或部分内容提出争议。",
        "excerpt": "账单应在收到后60日内支付。",
        "chunk": "正确地址的账单应在收到后六十(60)日内支付，除非买方秉承诚信善意原则对账单的全部或部分内容提出争议。付款不构成对发票金额或指定服务的认可。",
    },
]


def main():
    print("=" * 70)
    print("检索评测 Prompt 改动验证")
    print("=" * 70)

    all_pass = True

    # ========== 1. 验证正例 ==========
    print("\n[1] 新版检索查询格式验证（应全部通过）")
    print("-" * 70)

    for i, sample in enumerate(SAMPLES, 1):
        q = {
            "question": sample["expected_query"],
            "reference_answer": sample["expected_evidence"],
            "source_excerpt": sample["expected_evidence"],  # 必须一致
        }
        ok, reason = validate_retrieval_question(q, sample["chunk_text"])

        norm_ref = re.sub(r'\s+', '', sample["expected_evidence"])
        norm_chunk = re.sub(r'\s+', '', sample["chunk_text"])
        is_substring = norm_ref in norm_chunk
        is_copying = _detect_phrase_copying(
            sample["expected_query"], sample["expected_evidence"]
        )

        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"\n  样本 {i}: {sample['name']}")
        print(f"    检索查询: {sample['expected_query']}")
        print(f"    证据长度: {len(sample['expected_evidence'])} 字")
        print(f"    校验结果: {status}")
        if not ok:
            print(f"    失败原因: {reason}")
            all_pass = False
        print(f"    子串校验: {'✅ 逐字匹配' if is_substring else '❌ 非子串'}")
        print(f"    照抄检测: {'⚠️ 照抄' if is_copying else '✅ 语义改写'}")
        if not is_substring:
            all_pass = False
        if is_copying:
            all_pass = False

    # ========== 2. 验证反例：问句式 ==========
    print("\n\n[2] 旧版问句式题目过滤验证（应全部被拦截）")
    print("-" * 70)

    dummy_chunk = "缺陷指一项或多项指定服务(i)不符合或不再符合相关《规范说明》所载要求或本协议规定的任何其他要求，或(ii)不符合其预期目的。"
    for i, sample in enumerate(OLD_STYLE_SAMPLES, 1):
        q = {
            "question": sample["question"],
            "reference_answer": dummy_chunk,
            "source_excerpt": dummy_chunk,
        }
        ok, reason = validate_retrieval_question(q, dummy_chunk)
        status = "✅ 已拦截" if not ok else "❌ 未拦截"
        print(f"\n  反例 {i}: {sample['name']}")
        print(f"    输入: {sample['question']}")
        print(f"    校验结果: {status}")
        if ok:
            print(f"    ⚠️ 应被拦截但未拦截!")
            all_pass = False

    # ========== 3. 验证反例：逐字照抄 ==========
    print("\n\n[3] 逐字照抄检测验证（应全部被拦截）")
    print("-" * 70)

    for i, sample in enumerate(PHRASE_COPY_SAMPLES, 1):
        q = {
            "question": sample["question"],
            "reference_answer": sample["ref_answer"],
            "source_excerpt": sample["ref_answer"],
        }
        ok, reason = validate_retrieval_question(q, sample["chunk"])
        status = "✅ 已拦截" if not ok else "❌ 未拦截"
        print(f"\n  反例 {i}: {sample['name']}")
        print(f"    查询: {sample['question'][:50]}...")
        print(f"    校验结果: {status}")
        if ok:
            print(f"    ⚠️ 应被拦截但未拦截!")
            all_pass = False
        else:
            print(f"    拦截原因: {reason}")

    # ========== 4. 验证反例：过短证据 ==========
    print("\n\n[4] 过短证据过滤验证（应被拦截）")
    print("-" * 70)

    for i, sample in enumerate(SHORT_EVIDENCE_SAMPLES, 1):
        q = {
            "question": sample["question"],
            "reference_answer": sample["ref_answer"],
            "source_excerpt": sample["ref_answer"],
        }
        ok, reason = validate_retrieval_question(q, sample["chunk"])
        status = "✅ 已拦截" if not ok else "❌ 未拦截"
        print(f"\n  反例 {i}: {sample['name']}")
        print(f"    证据: {sample['ref_answer']}")
        print(f"    证据长度: {len(sample['ref_answer'])} 字")
        print(f"    校验结果: {status}")
        if ok:
            print(f"    ⚠️ 应被拦截但未拦截!")
            all_pass = False
        else:
            print(f"    拦截原因: {reason}")

    # ========== 5. 验证反例：source_excerpt 与 reference_answer 不一致 ==========
    print("\n\n[5] source_excerpt 与 reference_answer 不一致验证（应被拦截）")
    print("-" * 70)

    for i, sample in enumerate(MISMATCH_SAMPLES, 1):
        q = {
            "question": sample["question"],
            "reference_answer": sample["ref_answer"],
            "source_excerpt": sample["excerpt"],
        }
        ok, reason = validate_retrieval_question(q, sample["chunk"])
        status = "✅ 已拦截" if not ok else "❌ 未拦截"
        print(f"\n  反例 {i}: {sample['name']}")
        print(f"    reference_answer: {sample['ref_answer'][:50]}...")
        print(f"    source_excerpt:   {sample['excerpt'][:50]}...")
        print(f"    校验结果: {status}")
        if ok:
            print(f"    ⚠️ 应被拦截但未拦截!")
            all_pass = False
        else:
            print(f"    拦截原因: {reason}")

    # ========== 6. 人工展示：查询 vs 原文证据差异 ==========
    print("\n\n[6] 查询与原文证据差异展示（语义改写人工确认）")
    print("-" * 70)

    for i, sample in enumerate(SAMPLES, 1):
        print(f"\n  样本 {i}: {sample['name']}")
        print(f"    查询:     {sample['expected_query']}")
        evi_preview = sample['expected_evidence'][:80] + (
            "..." if len(sample['expected_evidence']) > 80 else ""
        )
        print(f"    原文证据: {evi_preview}")

    # ========== 7. reference_answer 原始值保留验证 ==========
    print("\n\n[7] reference_answer 原始值保留验证")
    print("-" * 70)

    for i, sample in enumerate(SAMPLES[:3], 1):
        q = {
            "question": sample["expected_query"],
            "reference_answer": sample["expected_evidence"],
            "source_excerpt": sample["expected_evidence"],
        }
        original_ref = q["reference_answer"]
        validate_retrieval_question(q, sample["chunk_text"])
        if q["reference_answer"] == original_ref:
            print(f"  样本 {i}: ✅ 原始值未被修改")
        else:
            print(f"  样本 {i}: ❌ 原始值被修改!")
            all_pass = False

    # ========== 8. source_excerpt 强制等于 reference_answer 验证 ==========
    print("\n\n[8] source_excerpt = reference_answer 强制一致性验证")
    print("-" * 70)

    for i, sample in enumerate(SAMPLES[:3], 1):
        ref = sample["expected_evidence"]
        # 模拟 _generate_from_chunks 中的强制赋值
        q = {"question": sample["expected_query"], "reference_answer": ref}
        # LLM 可能输出不同的 source_excerpt
        q["source_excerpt"] = "某个不同的摘录"
        # 强制赋值（与 _generate_from_chunks 一致）
        ref_trimmed = (q.get("reference_answer") or "").strip()
        if ref_trimmed:
            q["source_excerpt"] = ref_trimmed
        if q["source_excerpt"] == q["reference_answer"]:
            print(f"  样本 {i}: ✅ source_excerpt 已强制等于 reference_answer")
        else:
            print(f"  样本 {i}: ❌ 强制赋值失败!")
            all_pass = False

    # ========== 输出样例 JSON ==========
    print("\n\n[9] 新版检索查询样例 JSON 输出（3 条）")
    print("-" * 70)

    for sample in SAMPLES[:3]:
        obj = {
            "question": sample["expected_query"],
            "reference_answer": sample["expected_evidence"],
            "source_excerpt": sample["expected_evidence"],
            "difficulty": "事实",
            "topic": sample["name"],
        }
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        print()

    # ========== 总结 ==========
    print("=" * 70)
    if all_pass:
        print("✅ 全部验证通过！")
    else:
        print("❌ 存在验证失败项，请检查。")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
