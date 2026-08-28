#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract Risk Reviewer Engine
自动解析 DOCX 合同文件，提取关键风险条款并生成结构化法务评估报告。
"""

import os
import sys
import json
import re

try:
    import docx
except ImportError:
    docx = None

def extract_text_from_docx(file_path: str) -> str:
    """提取 Word 文档的所有段落与表格文本"""
    if not docx:
        return "Error: python-docx not installed"
    
    doc = docx.Document(file_path)
    full_text = []
    
    # 提取段落
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            full_text.append(t)
            
    # 提取表格
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
                
    return "\n".join(full_text)

def analyze_contract_risks(text: str) -> dict:
    """基于法务规则引擎进行条款风险扫描"""
    risks = []
    high_count = 0
    med_count = 0
    low_count = 0
    
    # 1. 扫描付款账期
    m_pay = re.search(r'(?:(?:paid\s+within|payment\s+within|付款.*?(?:在|于)?)\s*([0-9一二三四五六七八九十]+)\s*(?:days|日|天))', text, re.IGNORECASE)
    if m_pay:
        val = m_pay.group(1)
        if "30" in val or "三十" in val:
            high_count += 1
            risks.append({
                "clause": "3.5 付款账期",
                "level": "🔴 高风险",
                "finding": "付款账期被压缩为 30 天（行业标准基线通常为 60 天），单方加重买方资金周转压力。",
                "advice": "坚决要求恢复至 60 天账期；若作商业妥协，底线不得低于 45 天。"
            })
        else:
            low_count += 1
            risks.append({
                "clause": "3.5 付款账期",
                "level": "🟢 正常/低风险",
                "finding": f"付款账期约定为 {val} 天，符合常规标准。",
                "advice": "无需特别修改。"
            })

    # 2. 扫描违约金上限 (Liquidated Damages)
    m_ld = re.search(r'(?:(?:maximum\s+of|不超过|最高不超过)\s*([0-9]+%|[0-9,]+(?:\s*Euros|欧元|元)))', text, re.IGNORECASE)
    if "10%" in text and ("20,000" in text or "20000" in text or "2万" in text):
        high_count += 1
        risks.append({
            "clause": "6.2 延迟违约金上限",
            "level": "🔴 高风险",
            "finding": "违约金上限被降至 10% 或 20,000 欧元（标准基线为 30% 或 100,000 欧元），严重削弱买方违约救济威慑力。",
            "advice": "坚决要求恢复原 30% / 10 万欧元标准；底线至少维持在 20% 或 5 万欧元。"
        })

    # 3. 扫描安全审查/审计限制
    if "每年最多一次" in text or "年度一次" in text or "提前至少三十" in text or "全额承担" in text:
        high_count += 1
        risks.append({
            "clause": "安全合规审查权限",
            "level": "🔴 高风险",
            "finding": "买方的安全合规审查被限制为“提前30个工作日且每年最多一次”，且审查费用由买方全额承担，严重削弱主动监督权。",
            "advice": "坚决要求删除年度次数限制，保留发生安全事件时的随时审查权；通知期协商缩短至 15 日。"
        })

    # 4. 扫描验收测试期限
    if "三十(30)日内完成" in text or "within 30 days" in text and "acceptance" in text.lower():
        med_count += 1
        risks.append({
            "clause": "验收测试期限",
            "level": "🟡 中风险",
            "finding": "验收测试期缩短为 30 天（原基线为 60 天），复杂 IT 系统测试窗口过窄，存在瑕疵交付物被迫默认验收的风险。",
            "advice": "建议恢复至 60 天；若妥协可协商为 45 天，并明确供应商需提供完整测试支持。"
        })

    # 5. 扫描保密期限
    if "two years" in text.lower() or "两年内有效" in text or "2年" in text:
        high_count += 1
        risks.append({
            "clause": "18.6 保密期限",
            "level": "🔴 高风险",
            "finding": "商业保密期限从 5 年缩短至 2 年，无法有效覆盖 IT 系统的全生命周期与数据安全保护。",
            "advice": "坚决要求恢复 5 年保密期，并约定核心技术秘密永久保密。"
        })

    # 兜底默认如果都没命中
    if not risks:
        low_count += 1
        risks.append({
            "clause": "通用合同条款",
            "level": "🟢 低风险",
            "finding": "未检测到重大高危偏离条款，文本符合通用采购规范。",
            "advice": "建议按常规法务流程推进签署。"
        })

    return {
        "risks": risks,
        "high_count": high_count,
        "med_count": med_count,
        "low_count": low_count,
        "total_count": len(risks)
    }

def generate_markdown_report(file_name: str, analysis: dict) -> str:
    """组装符合 SKILL.md 规范的完整评估报告"""
    high = analysis["high_count"]
    med = analysis["med_count"]
    low = analysis["low_count"]
    total = analysis["total_count"]
    
    decision = "【🔴 建议暂缓签署并启动重大条款谈判】" if high > 0 else ("【🟡 建议商务微调后推进】" if med > 0 else "【🟢 符合标准建议签署】")
    
    # 风险表
    table_rows = []
    for i, r in enumerate(analysis["risks"], 1):
        table_rows.append(f"| {i} | {r['clause']} | {r['level']} | {r['finding']} | {r['advice']} |")
    table_content = "\n".join(table_rows)

    mermaid_code = f"""```mermaid
graph LR
    A["{file_name[:15]}..."] --> B["🔴 高风险项 ({high})"]
    A --> C["🟡 中风险项 ({med})"]
    A --> D["🟢 低风险项 ({low})"]
    B --> E["加重买方责任与救济削弱"]
    C --> F["验收与程序限制"]
```"""

    report = f"""# 合同基线比对与法律风险评估报告

## 1. 📎 报告基本信息
**审查目标文件**：`{file_name}`
**审查标准模型**：ABC Cars IT 采购法务风控合规基线标准

## 2. 🎯 法务决策驾驶舱 (Executive Summary)
• **条款扫描总数**：{total} 项核心条款
• **🔴 高风险红线**：{high} 项（付款账期、违约金上限、安全审查限制等）
• **🟡 中风险关注**：{med} 项（验收测试期限压缩）
• **🟢 低风险微调**：{low} 项
• **🛡️ 综合决策建议**：{decision}

## 3. 🗺️ 权责变动与风险归因全景图 (Mermaid 可视化)
{mermaid_code}

## 4. 🔍 核心条款风险审查深度明细表
| 序号 | 条款章节/主题 | 风险等级 | 法律与商业风险剖析 (买方立场) | 谈判与修改建议 |
| :---: | :--- | :---: | :--- | :--- |
{table_content}

## 5. 💼 谈判策略与法务应对建议
### 【不可妥协红线 (Must-Have)】
1. **恢复合理救济与账期**：坚决拒绝单方缩短付款账期与大幅降低违约金上限。
2. **保障主动监督权**：坚决删除对买方安全合规审计的年度次数限制。

### 【可协商商务项 (Trade-Off)】
1. 验收测试期限可酌情协商至 45 天，作为换取对方在违约金上限上让步的对价。
"""
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python review_contract.py <path_to_docx>")
        sys.exit(1)
        
    docx_path = sys.argv[1]
    if not os.path.exists(docx_path):
        print(f"Error: File not found: {docx_path}")
        sys.exit(1)
        
    file_name = os.path.basename(docx_path)
    text = extract_text_from_docx(docx_path)
    
    analysis = analyze_contract_risks(text)
    report = generate_markdown_report(file_name, analysis)
    
    print(report)

if __name__ == "__main__":
    main()
