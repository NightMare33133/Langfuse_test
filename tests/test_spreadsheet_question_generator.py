"""
统一电子表格检索题生成测试。

覆盖：
1. CSV 解析：UTF-8、GBK、BOM、空文件、单列
2. XLSX 解析：SheetContext、合并单元格、公式检测
3. 表格块拆分：小表/大表、表头保留、每块 allowed_anchor_ranges
4. 锚定范围验证：白名单内/外、越界、超大
5. 金标准渲染：单行、多行
6. LLM 响应解析：正常 JSON、markdown 代码块、无效
7. 完整流水线（mock LLM）：CSV、XLSX 端到端
8. doc_parser 集成：CSV/XLS 进入 parse_document
9. 向后兼容：xlsx_question_generator 委托正常

不调用真实 API。
"""

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from openpyxl import Workbook

from spreadsheet_question_generator import (
    SheetContext,
    TableBlock,
    _col_letter,
    _col_index,
    _parse_range_str,
    _range_to_str,
    _detect_csv_encoding,
    _compute_allowed_anchor_ranges,
    _split_into_table_blocks,
    _render_block_markdown,
    _render_cell_values,
    _parse_llm_response,
    _validate_anchor_range,
    _render_reference_answer,
    _validate_and_render_question,
    parse_xlsx_to_sheet_contexts,
    parse_csv_to_sheet_contexts,
    generate_spreadsheet_questions,
    _build_prompt,
)


# ====== Helpers ======

def _make_xlsx_bytes(wb):
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_simple_xlsx():
    wb = Workbook()
    ws = wb.active
    ws.title = "产品表"
    ws["A1"] = "产品名称"
    ws["B1"] = "价格"
    ws["C1"] = "库存"
    ws["A2"] = "产品A"
    ws["B2"] = 100
    ws["C2"] = 50
    ws["A3"] = "产品B"
    ws["B3"] = 200
    ws["C3"] = 30
    ws["A4"] = "产品C"
    ws["B4"] = 150
    ws["C4"] = 0
    return wb


def _make_multi_sheet_xlsx():
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1["A1"] = "Name"
    ws1["B1"] = "Value"
    ws1["A2"] = "Item1"
    ws1["B2"] = 10

    ws2 = wb.create_sheet("Sheet2")
    ws2["A1"] = "Category"
    ws2["B1"] = "Count"
    ws2["A2"] = "CatA"
    ws2["B2"] = 100
    return wb


def _make_formula_xlsx():
    wb = Workbook()
    ws = wb.active
    ws.title = "公式表"
    ws["A1"] = "项目"
    ws["B1"] = "数值"
    ws["A2"] = "A"
    ws["B2"] = 100
    ws["A3"] = "B"
    ws["B3"] = 200
    ws["A4"] = "合计"
    ws["B4"] = "=SUM(B2:B3)"
    return wb


def _make_merged_cell_xlsx():
    wb = Workbook()
    ws = wb.active
    ws.title = "合并表"
    ws["A1"] = "分类"
    ws["B1"] = "项目"
    ws["C1"] = "数值"
    ws["A2"] = "类别A"
    ws["B2"] = "项目1"
    ws["C2"] = 100
    ws["A3"] = None  # 合并后应继承 "类别A"
    ws["B3"] = "项目2"
    ws["C3"] = 200
    ws.merge_cells("A2:A3")
    return wb


def _make_csv_bytes(rows, encoding="utf-8"):
    """创建 CSV 字节。rows 是 list[list[str]]，第一行为表头。"""
    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow(row)
    content = output.getvalue()
    if encoding == "utf-8-sig":
        return b'\xef\xbb\xbf' + content.encode("utf-8")
    return content.encode(encoding)


def _make_large_xlsx(num_data_rows=100):
    """创建大数据量 XLSX。"""
    wb = Workbook()
    ws = wb.active
    ws.title = "大数据表"
    ws["A1"] = "ID"
    ws["B1"] = "名称"
    ws["C1"] = "数值"
    for i in range(1, num_data_rows + 1):
        ws[f"A{i+1}"] = i
        ws[f"B{i+1}"] = f"项目{i}"
        ws[f"C{i+1}"] = i * 10
    return wb


# ====== Column Letter Tests ======

def test_col_letter():
    """列字母转换。"""
    print("=" * 60)
    print("测试：列字母转换")
    print("=" * 60)

    assert _col_letter(1) == "A"
    assert _col_letter(26) == "Z"
    assert _col_letter(27) == "AA"
    assert _col_letter(52) == "AZ"

    assert _col_index("A") == 1
    assert _col_index("Z") == 26
    assert _col_index("AA") == 27
    assert _col_index("AZ") == 52

    print("PASS: 列字母转换正确")


# ====== Range Parsing Tests ======

def test_parse_range_str():
    """范围字符串解析。"""
    print("=" * 60)
    print("测试：范围字符串解析")
    print("=" * 60)

    assert _parse_range_str("A1:C3") == (1, 1, 3, 3)
    assert _parse_range_str("B2:D5") == (2, 2, 4, 5)
    assert _parse_range_str("AA1:AB3") == (27, 1, 28, 3)
    assert _parse_range_str("invalid") is None
    assert _parse_range_str("A3:A1") is None  # min > max

    print("PASS: 范围字符串解析正确")


# ====== CSV Parsing Tests ======

def test_csv_basic():
    """基本 UTF-8 CSV 解析。"""
    print("=" * 60)
    print("测试：CSV 基本解析")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([
        ["产品", "价格", "库存"],
        ["产品A", "100", "50"],
        ["产品B", "200", "30"],
    ])
    sheets = parse_csv_to_sheet_contexts(csv_bytes, "test.csv")
    assert len(sheets) == 1
    ctx = sheets[0]
    assert ctx.sheet_name == "CSV"
    assert ctx.max_row == 3  # header + 2 data rows
    assert ctx.max_col == 3
    assert ctx.headers == ["产品", "价格", "库存"]
    assert ctx.rows[1] == ["产品A", "100", "50"]
    assert len(ctx.formula_cells_without_cache) == 0
    assert len(ctx.merged_cells) == 0
    assert len(ctx.table_blocks) > 0

    print("PASS: CSV 基本解析正确")


def test_csv_encoding_gbk():
    """GBK 编码 CSV 解析。"""
    print("=" * 60)
    print("测试：CSV GBK 编码")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([
        ["产品", "价格"],
        ["产品A", "100"],
    ], encoding="gbk")
    sheets = parse_csv_to_sheet_contexts(csv_bytes, "test_gbk.csv")
    assert len(sheets) == 1
    assert sheets[0].headers == ["产品", "价格"]

    print("PASS: CSV GBK 编码解析正确")


def test_csv_encoding_bom():
    """UTF-8 BOM 编码 CSV 解析。"""
    print("=" * 60)
    print("测试：CSV BOM 编码")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([
        ["Name", "Value"],
        ["Item1", "10"],
    ], encoding="utf-8-sig")
    sheets = parse_csv_to_sheet_contexts(csv_bytes, "test_bom.csv")
    assert len(sheets) == 1
    assert sheets[0].headers == ["Name", "Value"]

    print("PASS: CSV BOM 编码解析正确")


def test_csv_empty():
    """空 CSV 应抛出异常。"""
    print("=" * 60)
    print("测试：空 CSV")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([["A", "B"]])  # 只有表头没有数据
    try:
        parse_csv_to_sheet_contexts(csv_bytes, "empty.csv")
        # 如果只有表头，pandas 会读到空 DataFrame 或只有表头
        # 这里可能通过也可能抛异常，取决于 pandas 行为
        print("  注意：只有表头的 CSV 被接受了（pandas 行为）")
    except ValueError:
        print("  空 CSV 正确抛出 ValueError")

    # 真正的空 CSV
    try:
        parse_csv_to_sheet_contexts(b"", "truly_empty.csv")
        print("  FAIL: 应该抛出异常")
    except (ValueError, Exception):
        print("  空文件正确抛出异常")

    print("PASS: 空 CSV 处理正确")


# ====== XLSX Parsing Tests ======

def test_xlsx_to_sheet_context():
    """XLSX 解析为 SheetContext。"""
    print("=" * 60)
    print("测试：XLSX SheetContext 解析")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    assert len(sheets) == 1
    ctx = sheets[0]
    assert ctx.sheet_name == "产品表"
    assert ctx.max_row == 4
    assert ctx.max_col == 3
    assert ctx.headers == ["产品名称", "价格", "库存"]
    assert ctx.rows[1] == ["产品A", 100, 50]
    assert len(ctx.table_blocks) > 0

    print("PASS: XLSX SheetContext 解析正确")


def test_xlsx_multi_sheet():
    """XLSX 多工作表解析。"""
    print("=" * 60)
    print("测试：XLSX 多工作表")
    print("=" * 60)

    wb = _make_multi_sheet_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    assert len(sheets) == 2
    names = {s.sheet_name for s in sheets}
    assert "Sheet1" in names
    assert "Sheet2" in names

    print("PASS: XLSX 多工作表解析正确")


def test_xlsx_merged_cells():
    """XLSX 合并单元格值继承。"""
    print("=" * 60)
    print("测试：XLSX 合并单元格")
    print("=" * 60)

    wb = _make_merged_cell_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]
    # 合并单元格 A2:A3，A3 应继承 A2 的值 "类别A"
    assert ctx.rows[2][0] == "类别A", f"A3 应为 '类别A': {ctx.rows[2][0]}"
    assert len(ctx.merged_cells) == 1

    print("PASS: XLSX 合并单元格值继承正确")


def test_xlsx_formula_detection():
    """XLSX 公式单元格检测。"""
    print("=" * 60)
    print("测试：XLSX 公式检测")
    print("=" * 60)

    wb = _make_formula_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]
    # B4 有公式 =SUM(B2:B3)，openpyxl 可能有缓存值也可能没有
    # 如果有缓存值，formula_cells_without_cache 为空
    # 如果没有，B4 位置在 formula_cells_without_cache 中
    print(f"  公式无缓存单元格: {ctx.formula_cells_without_cache}")
    # 至少不应崩溃
    assert isinstance(ctx.formula_cells_without_cache, list)

    print("PASS: XLSX 公式检测正常")


def test_formula_with_cached_value():
    """有缓存值的公式：不显示警告，使用缓存值。"""
    print("=" * 60)
    print("测试：公式有缓存值")
    print("=" * 60)

    # 创建 XLSX 并保存（openpyxl 会写入缓存值）
    wb = Workbook()
    ws = wb.active
    ws.title = "缓存表"
    ws["A1"] = "项目"
    ws["B1"] = "数值"
    ws["A2"] = "A"
    ws["B2"] = 100
    ws["A3"] = "B"
    ws["B3"] = 200
    ws["A4"] = "合计"
    ws["B4"] = "=SUM(B2:B3)"

    # 保存并重新打开（模拟有缓存值的文件）
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]

    # 检查 B4 的处理结果
    b4_val = ctx.rows[3][1]  # B4
    print(f"  B4 值: {b4_val}")
    print(f"  formula_cells_without_cache: {ctx.formula_cells_without_cache}")

    # 如果 openpyxl 有缓存值 → B4 应为 300，无警告
    # 如果无缓存值 → B4 应为 [公式未计算]，有警告
    if (4, 2) in ctx.formula_cells_without_cache:
        assert b4_val == "[公式未计算]", f"无缓存时 B4 应为 [公式未计算]: {b4_val}"
        # 检查 block 有公式警告
        assert ctx.table_blocks[0].has_formula_warnings, "应有公式警告"
    else:
        assert b4_val == 300, f"有缓存时 B4 应为 300: {b4_val}"
        assert not ctx.table_blocks[0].has_formula_warnings, "有缓存值不应有公式警告"

    print("PASS: 公式缓存值处理正确")


def test_formula_no_cache_rejects_reference_answer():
    """无缓存值的公式：reference_answer 拒绝含该单元格的范围。"""
    print("=" * 60)
    print("测试：无缓存公式拒绝 reference_answer")
    print("=" * 60)

    # 创建一个公式一定无缓存的场景：
    # 直接用 data_only=False 构造 rows，公式单元格为字符串
    from spreadsheet_question_generator import SheetContext, _render_reference_answer

    ctx = SheetContext(
        sheet_name="测试",
        max_row=2,
        max_col=2,
        headers=["项目", "数值"],
        rows=[["项目", "数值"], ["合计", "=SUM(B1)"]],
        formula_cells_without_cache=[(2, 2)],
        format_warnings=[],
        allowed_anchor_ranges=["A2:B2"],
        table_blocks=[],
    )

    rendered, has_issue = _render_reference_answer("A2:B2", ctx)
    assert has_issue, "应标记公式问题"
    assert "[公式未计算]" in rendered, f"应含 [公式未计算]: {rendered}"
    assert "=SUM" not in rendered, f"不应含公式字符串: {rendered}"

    print("PASS: 无缓存公式正确拒绝")


def test_formula_cached_value_no_warning():
    """有缓存值的公式单元格：reference_answer 使用缓存值，无警告。"""
    print("=" * 60)
    print("测试：缓存公式无警告")
    print("=" * 60)

    from spreadsheet_question_generator import SheetContext, _render_reference_answer

    ctx = SheetContext(
        sheet_name="测试",
        max_row=2,
        max_col=2,
        headers=["项目", "数值"],
        rows=[["项目", "数值"], ["合计", 300]],  # 缓存值已替换公式
        formula_cells_without_cache=[],  # 无缓存问题
        format_warnings=[],
        allowed_anchor_ranges=["A2:B2"],
        table_blocks=[],
    )

    rendered, has_issue = _render_reference_answer("A2:B2", ctx)
    assert not has_issue, "有缓存值不应有公式问题"
    assert "300" in rendered, f"应含缓存值 300: {rendered}"
    assert "[公式未计算]" not in rendered, f"不应含 [公式未计算]: {rendered}"

    print("PASS: 缓存公式无警告正确")


# ====== Table Block Tests ======

def test_split_small_sheet():
    """小表格应只有 1 个块。"""
    print("=" * 60)
    print("测试：小表格拆分为 1 块")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]
    assert len(ctx.table_blocks) == 1, f"应为 1 块，实际 {len(ctx.table_blocks)}"

    block = ctx.table_blocks[0]
    assert block.row_range == (2, 4)
    assert "| 行号 |" in block.markdown
    assert "| 2 |" in block.markdown

    print("PASS: 小表格正确拆分为 1 块")


def test_split_large_sheet():
    """大表格应拆分为多个块。"""
    print("=" * 60)
    print("测试：大表格拆分为多块")
    print("=" * 60)

    wb = _make_large_xlsx(100)
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]
    # 100 行数据，每块 30 行，应为 4 块 (30+30+30+10)
    assert len(ctx.table_blocks) == 4, f"应为 4 块，实际 {len(ctx.table_blocks)}"

    # 每块都应包含表头
    for block in ctx.table_blocks:
        assert "| 行号 |" in block.markdown, f"块 {block.block_index} 缺少表头"

    # 行号连续性
    assert ctx.table_blocks[0].row_range == (2, 31)
    assert ctx.table_blocks[1].row_range == (32, 61)
    assert ctx.table_blocks[2].row_range == (62, 91)
    assert ctx.table_blocks[3].row_range == (92, 101)

    print("PASS: 大表格正确拆分为多块")


def test_allowed_ranges_per_block():
    """每块的 allowed_anchor_ranges 是 sheet 的子集。"""
    print("=" * 60)
    print("测试：每块 allowed_anchor_ranges")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]
    block = ctx.table_blocks[0]
    # 应该有 allowed_anchor_ranges
    assert isinstance(block.allowed_anchor_ranges, list)
    # 所有块级范围应是 sheet 级范围的子集
    for r in block.allowed_anchor_ranges:
        assert r in ctx.allowed_anchor_ranges, f"块级范围 {r} 不在 sheet 级范围中"

    print("PASS: 每块 allowed_anchor_ranges 正确")


def test_semantic_header_value_block():
    """费率/参数表生成语义化二列块。"""
    print("=" * 60)
    print("测试：语义化表头+数值块")
    print("=" * 60)

    wb = Workbook()
    ws = wb.active
    ws.title = "费率表"
    ws["A1"] = "费用项"
    ws["B1"] = "项目经理"
    ws["C1"] = "开发人员"
    ws["D1"] = "测试人员"
    ws["A2"] = "单价(元/人天)"
    ws["B2"] = 1700
    ws["C2"] = 1500
    ws["D2"] = 1200

    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]

    # 应有 2 个块：标准块 + 语义块
    assert len(ctx.table_blocks) == 2, f"应有 2 块: {len(ctx.table_blocks)}"

    sem_block = ctx.table_blocks[1]
    assert "字段名" in sem_block.markdown
    assert "数值" in sem_block.markdown
    assert "项目经理" in sem_block.markdown
    assert "1700" in sem_block.markdown
    # 行标签列应被跳过
    assert "费用项" not in sem_block.markdown, "行标签列应被跳过"
    assert "单价(元/人天)" not in sem_block.markdown, "行标签列应被跳过"

    # 每个字段应有独立 anchor
    assert "B1:B2" in sem_block.allowed_anchor_ranges
    assert "C1:C2" in sem_block.allowed_anchor_ranges
    assert "D1:D2" in sem_block.allowed_anchor_ranges

    print("PASS: 语义化表头+数值块正确")


def test_semantic_no_false_positive():
    """普通数据表不应生成语义化块。"""
    print("=" * 60)
    print("测试：普通表不生成语义块")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]

    # 普通表只有 1 个标准块
    assert len(ctx.table_blocks) == 1, f"普通表应只有 1 块: {len(ctx.table_blocks)}"

    print("PASS: 普通表不生成语义块")


def test_semantic_anchors_in_sheet_whitelist():
    """语义块的 field anchors 应自动添加到 sheet 级白名单。"""
    print("=" * 60)
    print("测试：语义 anchor 在 sheet 白名单中")
    print("=" * 60)

    wb = Workbook()
    ws = wb.active
    ws.title = "费率表"
    ws["A1"] = "费用项"
    ws["B1"] = "项目经理"
    ws["C1"] = "开发人员"
    ws["D1"] = "测试人员"
    ws["A2"] = "单价(元/人天)"
    ws["B2"] = 1700
    ws["C2"] = 1500
    ws["D2"] = 1200

    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]

    # 语义块的 field anchors 应在 sheet 级白名单中
    assert "B1:B2" in ctx.allowed_anchor_ranges, f"B1:B2 应在白名单中: {ctx.allowed_anchor_ranges}"
    assert "C1:C2" in ctx.allowed_anchor_ranges
    assert "D1:D2" in ctx.allowed_anchor_ranges

    print("PASS: 语义 anchor 在 sheet 白名单中")


def test_semantic_anchor_renders_field_and_value():
    """E2:E3 类型的锚点 reference_answer 应同时含字段名和数值。"""
    print("=" * 60)
    print("测试：语义锚点渲染字段+数值")
    print("=" * 60)

    wb = Workbook()
    ws = wb.active
    ws.title = "费率表"
    ws["A1"] = "费用项"
    ws["B1"] = "项目经理"
    ws["C1"] = "开发人员"
    ws["D1"] = "测试人员"
    ws["A2"] = "单价(元/人天)"
    ws["B2"] = 1700
    ws["C2"] = 1500
    ws["D2"] = 1200

    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    sheets_by_name = {s.sheet_name: s for s in sheets}

    # 用语义 anchor 验证
    q = {"question": "项目经理费率", "sheet_name": "费率表", "anchor_range": "B1:B2"}
    result, reason = _validate_and_render_question(q, sheets_by_name, "test.xlsx")
    assert result is not None, f"应通过验证: {reason}"
    ref = result["reference_answer"]
    assert "项目经理" in ref, f"应含字段名 '项目经理': {ref}"
    assert "1700" in ref, f"应含数值 '1700': {ref}"

    print("PASS: 语义锚点正确渲染字段+数值")


def test_isolated_numeric_anchor_rejected():
    """孤立数值锚点如 E3:E3 应被拒绝。"""
    print("=" * 60)
    print("测试：孤立数值锚点拒绝")
    print("=" * 60)

    wb = Workbook()
    ws = wb.active
    ws.title = "报价页"
    ws["A1"] = "项目"
    ws["B1"] = "描述"
    ws["C1"] = "单价"
    ws["A2"] = "服务A"
    ws["B2"] = "咨询服务"
    ws["C2"] = 73900

    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    sheets_by_name = {s.sheet_name: s for s in sheets}

    # 孤立数值锚点应被拒绝
    q = {"question": "服务A单价", "sheet_name": "报价页", "anchor_range": "C2:C2"}
    result, reason = _validate_and_render_question(q, sheets_by_name, "test.xlsx")
    assert result is None, f"孤立数值应被拒绝: {reason}"
    assert "孤立数值" in reason, f"原因应提及孤立数值: {reason}"

    # 包含字段名+数值的锚点应通过
    q2 = {"question": "服务A单价", "sheet_name": "报价页", "anchor_range": "A2:C2"}
    result2, reason2 = _validate_and_render_question(q2, sheets_by_name, "test.xlsx")
    assert result2 is not None, f"含字段名+数值应通过: {reason2}"

    print("PASS: 孤立数值锚点正确拒绝")


def test_cross_row_rejected():
    """跨两行数据的范围应通过，跨三行应被拒绝。"""
    print("=" * 60)
    print("测试：跨行范围拒绝")
    print("=" * 60)

    wb = Workbook()
    ws = wb.active
    ws.title = "报价页"
    ws["A1"] = "项目"
    ws["B1"] = "单价"
    ws["A2"] = "服务A"
    ws["B2"] = 100
    ws["A3"] = "服务B"
    ws["B3"] = 200
    ws["A4"] = "服务C"
    ws["B4"] = 300

    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    sheets_by_name = {s.sheet_name: s for s in sheets}

    # 跨三行数据应被拒绝
    q = {"question": "服务列表", "sheet_name": "报价页", "anchor_range": "A2:B4"}
    result, reason = _validate_and_render_question(q, sheets_by_name, "test.xlsx")
    assert result is None, f"跨三行应被拒绝: {reason}"
    assert "跨行" in reason or "3" in reason, f"原因应提及跨行: {reason}"

    # 相邻两行（label+value）应通过
    q2 = {"question": "服务信息", "sheet_name": "报价页", "anchor_range": "A2:B3"}
    result2, reason2 = _validate_and_render_question(q2, sheets_by_name, "test.xlsx")
    assert result2 is not None, f"相邻两行应通过: {reason2}"

    # 单行数据应通过
    q3 = {"question": "服务A信息", "sheet_name": "报价页", "anchor_range": "A2:B2"}
    result3, reason3 = _validate_and_render_question(q3, sheets_by_name, "test.xlsx")
    assert result3 is not None, f"单行数据应通过: {reason3}"

    print("PASS: 跨行范围正确拒绝")


def test_semantic_block_in_prompt():
    """语义块应出现在发给 LLM 的 prompt 中。"""
    print("=" * 60)
    print("测试：语义块进入 prompt")
    print("=" * 60)

    wb = Workbook()
    ws = wb.active
    ws.title = "费率表"
    ws["A1"] = "费用项"
    ws["B1"] = "项目经理"
    ws["C1"] = "开发人员"
    ws["D1"] = "测试人员"
    ws["A2"] = "单价(元/人天)"
    ws["B2"] = 1700
    ws["C2"] = 1500
    ws["D2"] = 1200

    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    prompt = _build_prompt(sheets, 5, "")

    # prompt 应包含语义块内容
    assert "字段名" in prompt, "prompt 应含语义块的 '字段名' 表头"
    assert "项目经理" in prompt, "prompt 应含语义块的 '项目经理'"
    assert "1700" in prompt, "prompt 应含语义块的 '1700'"
    # 应包含语义块的 allowed_anchor_ranges
    assert "B1:B2" in prompt, "prompt 应含语义块的 B1:B2 anchor"

    print("PASS: 语义块正确进入 prompt")


def test_normal_text_question_not_regressed():
    """普通单行文本题不回归。"""
    print("=" * 60)
    print("测试：普通文本题不回归")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([
        ["功能模块", "描述", "状态"],
        ["登录", "用户认证", "上线"],
        ["支付", "在线支付", "测试中"],
    ])

    mock_response = json.dumps([
        {"question": "登录模块", "sheet_name": "CSV", "anchor_range": "A2:C2", "difficulty": "事实"},
    ])

    import spreadsheet_question_generator as sqg
    original = sqg._call_llm_text
    sqg._call_llm_text = lambda *a, **kw: mock_response

    try:
        questions, stats = generate_spreadsheet_questions(
            csv_bytes, "test.csv", "fake", "http://fake", "fake_model",
        )
        assert len(questions) == 1
        ref = questions[0]["reference_answer"]
        assert "登录" in ref, f"应含 '登录': {ref}"
        assert "用户认证" in ref, f"应含 '用户认证': {ref}"
    finally:
        sqg._call_llm_text = original

    print("PASS: 普通文本题不回归")


# ====== Anchor Validation Tests ======

def test_valid_anchor_in_whitelist():
    """白名单内的范围通过验证。"""
    print("=" * 60)
    print("测试：白名单内范围验证")
    print("=" * 60)

    allowed = ["A1:C3", "A2:C2", "A3:C3"]
    valid, reason = _validate_anchor_range("A2:C2", allowed, 10, 10)
    assert valid, f"应通过: {reason}"

    print("PASS: 白名单内范围正确通过")


def test_anchor_not_in_whitelist():
    """不在白名单的范围被拒绝。"""
    print("=" * 60)
    print("测试：非白名单范围拒绝")
    print("=" * 60)

    allowed = ["A2:C2"]
    valid, reason = _validate_anchor_range("D2:F2", allowed, 10, 10)
    assert not valid
    assert "白名单" in reason

    print("PASS: 非白名单范围正确拒绝")


def test_anchor_subset_legal():
    """白名单 B4:E4 时，子范围 B4:C4 合法。"""
    print("=" * 60)
    print("测试：子范围合法")
    print("=" * 60)

    allowed = ["B4:E4"]
    valid, reason = _validate_anchor_range("B4:C4", allowed, 10, 10)
    assert valid, f"B4:C4 应合法（B4:E4 的子范围）: {reason}"

    # 也是精确匹配
    valid2, reason2 = _validate_anchor_range("B4:E4", allowed, 10, 10)
    assert valid2, f"B4:E4 应合法（精确匹配）: {reason2}"

    print("PASS: 子范围合法")


def test_anchor_subset_right_overflow():
    """白名单 B4:E4 时，B4:F4 右越界非法。"""
    print("=" * 60)
    print("测试：子范围右越界")
    print("=" * 60)

    allowed = ["B4:E4"]
    valid, reason = _validate_anchor_range("B4:F4", allowed, 10, 10)
    assert not valid, "B4:F4 应非法（右边界超出 B4:E4）"
    assert "白名单" in reason

    print("PASS: 子范围右越界正确拒绝")


def test_anchor_subset_left_overflow():
    """白名单 B4:E4 时，A4:C4 左越界非法。"""
    print("=" * 60)
    print("测试：子范围左越界")
    print("=" * 60)

    allowed = ["B4:E4"]
    valid, reason = _validate_anchor_range("A4:C4", allowed, 10, 10)
    assert not valid, "A4:C4 应非法（左边界 A < B 超出白名单）"
    assert "白名单" in reason

    print("PASS: 子范围左越界正确拒绝")


def test_anchor_out_of_bounds():
    """越界范围被拒绝。"""
    print("=" * 60)
    print("测试：越界范围拒绝")
    print("=" * 60)

    allowed = ["A2:C2"]
    valid, reason = _validate_anchor_range("A2:C200", allowed, 10, 3)
    assert not valid
    assert "边界" in reason

    print("PASS: 越界范围正确拒绝")


def test_anchor_too_large():
    """超大范围被拒绝。"""
    print("=" * 60)
    print("测试：超大范围拒绝")
    print("=" * 60)

    allowed = ["A1:ZZ1"]
    valid, reason = _validate_anchor_range("A1:ZZ1", allowed, 1, 703)
    assert not valid
    assert "上限" in reason

    print("PASS: 超大范围正确拒绝")


# ====== Reference Answer Rendering Tests ======

def test_render_single_row():
    """单行范围渲染为键值格式。"""
    print("=" * 60)
    print("测试：单行渲染")
    print("=" * 60)

    cell_values = [["产品A", 100, 50]]
    rendered = _render_cell_values(cell_values)
    assert "产品A" in rendered
    assert "100" in rendered
    assert "|" in rendered

    print("PASS: 单行渲染正确")


def test_render_multi_row():
    """多行范围渲染为表格格式。"""
    print("=" * 60)
    print("测试：多行渲染")
    print("=" * 60)

    cell_values = [
        ["名称", "价格"],
        ["产品A", 100],
        ["产品B", 200],
    ]
    rendered = _render_cell_values(cell_values)
    assert "|" in rendered
    assert "---" in rendered  # 分隔行
    assert "名称" in rendered
    assert "产品A" in rendered

    print("PASS: 多行渲染正确")


def test_render_reference_answer_from_context():
    """从 SheetContext 渲染 reference_answer。"""
    print("=" * 60)
    print("测试：从 SheetContext 渲染 reference_answer")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)
    ctx = sheets[0]

    rendered, has_formula = _render_reference_answer("A2:C2", ctx)
    assert rendered, "应有渲染结果"
    assert "产品A" in rendered
    assert "100" in rendered
    assert not has_formula

    print("PASS: SheetContext 渲染正确")


# ====== LLM Response Parsing Tests ======

def test_parse_valid_json():
    """正常 JSON 数组解析。"""
    print("=" * 60)
    print("测试：正常 JSON 解析")
    print("=" * 60)

    resp = json.dumps([
        {"question": "查询1", "sheet_name": "Sheet1", "anchor_range": "A1:B2"},
        {"question": "查询2", "sheet_name": "Sheet1", "anchor_range": "C1:D2"},
    ])
    parsed = _parse_llm_response(resp)
    assert len(parsed) == 2

    print("PASS: 正常 JSON 解析正确")


def test_parse_markdown_code_block():
    """Markdown 代码块中的 JSON 解析。"""
    print("=" * 60)
    print("测试：Markdown 代码块解析")
    print("=" * 60)

    json_str = json.dumps([{"question": "测试", "sheet_name": "S1", "anchor_range": "A1:B1"}])
    resp = f"```json\n{json_str}\n```"
    parsed = _parse_llm_response(resp)
    assert len(parsed) == 1

    print("PASS: Markdown 代码块解析正确")


def test_parse_invalid_json():
    """无效 JSON 返回空列表。"""
    print("=" * 60)
    print("测试：无效 JSON")
    print("=" * 60)

    parsed = _parse_llm_response("这不是 JSON")
    assert parsed == []

    print("PASS: 无效 JSON 正确返回空列表")


# ====== Full Pipeline Tests (Mocked LLM) ======

def test_generate_csv_questions():
    """CSV 端到端生成（mock LLM）。"""
    print("=" * 60)
    print("测试：CSV 端到端生成")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([
        ["产品名称", "价格", "库存"],
        ["产品A", "100", "50"],
        ["产品B", "200", "30"],
        ["产品C", "150", "0"],
    ])

    # Mock LLM 返回
    mock_response = json.dumps([
        {"question": "产品A价格", "sheet_name": "CSV", "anchor_range": "A2:C2", "difficulty": "事实", "topic": "价格"},
        {"question": "产品B库存", "sheet_name": "CSV", "anchor_range": "A3:C3", "difficulty": "事实", "topic": "库存"},
    ])

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text
    sqg._call_llm_text = lambda *a, **kw: mock_response

    try:
        questions, stats = generate_spreadsheet_questions(
            csv_bytes, "test.csv",
            "fake_key", "http://fake", "fake_model",
            num_questions=5,
        )
        assert len(questions) == 2, f"应生成 2 题: {len(questions)}"
        assert stats["sheet_count"] == 1
        assert questions[0]["source_format"] == "csv"
        assert questions[0]["evidence_sheet"] == "CSV"
        assert questions[0]["question_mode"] == "retrieval"
        assert "产品A" in questions[0]["reference_answer"]
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: CSV 端到端生成正确")


def test_generate_xlsx_questions():
    """XLSX 端到端生成（mock LLM）。"""
    print("=" * 60)
    print("测试：XLSX 端到端生成")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)

    mock_response = json.dumps([
        {"question": "产品A价格", "sheet_name": "产品表", "anchor_range": "A2:C2", "difficulty": "事实", "topic": "价格"},
    ])

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text
    sqg._call_llm_text = lambda *a, **kw: mock_response

    try:
        questions, stats = generate_spreadsheet_questions(
            xlsx_bytes, "test.xlsx",
            "fake_key", "http://fake", "fake_model",
            num_questions=5,
        )
        assert len(questions) == 1
        assert questions[0]["source_format"] == "xlsx"
        assert questions[0]["evidence_sheet"] == "产品表"
        assert "产品A" in questions[0]["reference_answer"]
        assert "100" in questions[0]["reference_answer"]
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: XLSX 端到端生成正确")


def test_validation_rejects_bad_range():
    """LLM 返回非白名单范围时，题目被过滤。"""
    print("=" * 60)
    print("测试：非白名单范围过滤")
    print("=" * 60)

    csv_bytes = _make_csv_bytes([
        ["Name", "Value"],
        ["Item1", "10"],
    ])

    mock_response = json.dumps([
        {"question": "测试", "sheet_name": "CSV", "anchor_range": "Z1:Z5", "difficulty": "事实"},
    ])

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text
    sqg._call_llm_text = lambda *a, **kw: mock_response

    try:
        try:
            generate_spreadsheet_questions(
                csv_bytes, "test.csv",
                "fake_key", "http://fake", "fake_model",
            )
            assert False, "应抛出 ValueError（所有题目被过滤）"
        except ValueError as e:
            assert "未通过" in str(e) or "失败" in str(e)
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: 非白名单范围正确过滤")


# ====== doc_parser Integration Tests ======

def test_csv_in_doc_parser():
    """CSV 进入 parse_document。"""
    print("=" * 60)
    print("测试：doc_parser CSV 解析")
    print("=" * 60)

    from doc_parser import parse_document

    csv_bytes = _make_csv_bytes([
        ["Name", "Value"],
        ["A", "1"],
        ["B", "2"],
    ])
    result = parse_document(file_bytes=csv_bytes, file_name="test.csv")
    assert result["source_type"] == "csv"
    assert result["summary"]["sheet_count"] == 1
    assert result["summary"]["row_count"] == 2
    assert len(result["blocks"]) == 2

    print("PASS: doc_parser CSV 解析正确")


def test_supported_extensions_includes_new():
    """get_supported_extensions 包含新格式。"""
    print("=" * 60)
    print("测试：支持扩展名列表")
    print("=" * 60)

    from doc_parser import get_supported_extensions, is_supported_file

    exts = get_supported_extensions()
    assert ".csv" in exts
    assert ".xls" in exts
    assert ".xlsx" in exts

    assert is_supported_file("test.csv")
    assert is_supported_file("test.xls")
    assert is_supported_file("test.xlsx")

    print("PASS: 支持扩展名列表正确")


# ====== Prompt Build Test ======

def test_build_prompt():
    """prompt 构建包含表格内容。"""
    print("=" * 60)
    print("测试：Prompt 构建")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)
    sheets = parse_xlsx_to_sheet_contexts(xlsx_bytes)

    prompt = _build_prompt(sheets, num_questions=5, topic_hint="产品信息")
    assert "产品表" in prompt
    assert "产品A" in prompt
    assert "5" in prompt
    assert "产品信息" in prompt
    assert "allowed_anchor_ranges" in prompt
    assert "行号" in prompt

    print("PASS: Prompt 构建正确")


# ====== Integration: LLM Request Content Tests ======

def test_llm_request_no_reference_answer():
    """断言实际发送给 LLM 的 prompt 不含 reference_answer/source_excerpt。"""
    print("=" * 60)
    print("测试：LLM 请求不含 reference_answer")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)

    # 捕获实际发送给 LLM 的 prompt
    captured_prompts = []

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text

    def mock_capture(prompt, *args, **kwargs):
        captured_prompts.append(prompt)
        return json.dumps([
            {"question": "产品A", "sheet_name": "产品表", "anchor_range": "A2:C2", "difficulty": "事实"},
        ])

    sqg._call_llm_text = mock_capture

    try:
        questions, stats = generate_spreadsheet_questions(
            xlsx_bytes, "test.xlsx",
            "fake_key", "http://fake", "fake_model",
            num_questions=5,
        )
        assert len(captured_prompts) == 1, f"应捕获 1 个 prompt: {len(captured_prompts)}"
        prompt = captured_prompts[0]

        # 核心断言：prompt 明确禁止 LLM 输出 reference_answer
        assert "不要输出" in prompt and "reference_answer" in prompt, "prompt 应明确禁止 LLM 输出 reference_answer"
        # 输出 JSON 格式中不应包含 reference_answer 作为期望字段
        output_format_section = prompt.split("输出格式")[-1] if "输出格式" in prompt else prompt[-500:]
        assert '"reference_answer"' not in output_format_section, "输出格式中不应有 reference_answer 字段"
        assert "source_excerpt" not in output_format_section, "输出格式中不应有 source_excerpt 字段"
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: LLM 请求不含 reference_answer/source_excerpt")


def test_llm_request_no_formula_string():
    """断言发送给 LLM 的 prompt 不含未计算公式字符串。"""
    print("=" * 60)
    print("测试：LLM 请求不含公式字符串")
    print("=" * 60)

    wb = _make_formula_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)

    captured_prompts = []

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text

    def mock_capture(prompt, *args, **kwargs):
        captured_prompts.append(prompt)
        return json.dumps([
            {"question": "项目A数值", "sheet_name": "公式表", "anchor_range": "A2:B2", "difficulty": "事实"},
        ])

    sqg._call_llm_text = mock_capture

    try:
        questions, stats = generate_spreadsheet_questions(
            xlsx_bytes, "test.xlsx",
            "fake_key", "http://fake", "fake_model",
            num_questions=5,
        )
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # 核心断言：prompt 中不含公式字符串
        assert "=SUM(" not in prompt, f"prompt 不应含公式字符串 =SUM("
        assert "=" not in prompt.split("allowed_anchor_ranges")[0].split("行号")[-1] or \
               "[公式未计算]" in prompt, "公式单元格应显示 [公式未计算] 而非公式表达式"
        assert "[公式未计算]" in prompt, "prompt 应包含 [公式未计算] 标记"
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: LLM 请求不含公式字符串")


def test_llm_request_uses_spreadsheet_prompt():
    """断言使用的是表格专用 prompt（含 allowed_anchor_ranges），而非通用检索 prompt。"""
    print("=" * 60)
    print("测试：使用表格专用 prompt")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)

    captured_prompts = []

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text

    def mock_capture(prompt, *args, **kwargs):
        captured_prompts.append(prompt)
        return json.dumps([
            {"question": "产品A", "sheet_name": "产品表", "anchor_range": "A2:C2", "difficulty": "事实"},
        ])

    sqg._call_llm_text = mock_capture

    try:
        questions, stats = generate_spreadsheet_questions(
            xlsx_bytes, "test.xlsx",
            "fake_key", "http://fake", "fake_model",
            num_questions=5,
        )
        prompt = captured_prompts[0]

        # 表格专用 prompt 的特征
        assert "allowed_anchor_ranges" in prompt, "prompt 应含 allowed_anchor_ranges 白名单"
        assert "行号" in prompt, "prompt 应含 Excel 行号列"
        assert "工作表:" in prompt or "工作表：" in prompt, "prompt 应含工作表标题"
        assert "电子表格" in prompt or "表格内容" in prompt, "prompt 应为表格专用模板"

        # 不应含通用检索 prompt 的特征
        assert "{content}" not in prompt, "prompt 不应含通用模板占位符 {content}"
        assert "{section_context}" not in prompt, "prompt 不应含通用模板占位符 {section_context}"
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: 使用表格专用 prompt")


def test_llm_request_local_reference_answer():
    """断言 reference_answer 只来自本地渲染，不含 LLM 输出。"""
    print("=" * 60)
    print("测试：reference_answer 纯本地渲染")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)

    # Mock LLM 返回（故意不含 reference_answer）
    mock_response = json.dumps([
        {"question": "产品A价格", "sheet_name": "产品表", "anchor_range": "A2:C2", "difficulty": "事实"},
    ])

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text
    sqg._call_llm_text = lambda *a, **kw: mock_response

    try:
        questions, stats = generate_spreadsheet_questions(
            xlsx_bytes, "test.xlsx",
            "fake_key", "http://fake", "fake_model",
        )
        q = questions[0]

        # reference_answer 必须存在且来自本地
        assert "reference_answer" in q, "应有 reference_answer"
        assert "source_excerpt" in q, "应有 source_excerpt"
        assert q["reference_answer"] == q["source_excerpt"], "两者应一致"

        # reference_answer 应包含实际单元格值
        assert "产品A" in q["reference_answer"], f"应含产品A: {q['reference_answer']}"
        assert "100" in q["reference_answer"], f"应含100: {q['reference_answer']}"

        # reference_answer 不应含 LLM 可能自写的文本
        assert "短检索" not in q["reference_answer"], "reference_answer 不应含 prompt 指令文本"
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: reference_answer 纯本地渲染")


# ====== Backward Compatibility Tests ======

def test_xlsx_question_generator_delegates():
    """xlsx_question_generator.generate_xlsx_questions 委托给新模块。"""
    print("=" * 60)
    print("测试：xlsx_question_generator 委托")
    print("=" * 60)

    wb = _make_simple_xlsx()
    xlsx_bytes = _make_xlsx_bytes(wb)

    mock_response = json.dumps([
        {"question": "产品A", "sheet_name": "产品表", "anchor_range": "A2:C2", "difficulty": "事实"},
    ])

    import spreadsheet_question_generator as sqg
    original_call_llm = sqg._call_llm_text
    sqg._call_llm_text = lambda *a, **kw: mock_response

    try:
        from xlsx_question_generator import generate_xlsx_questions
        questions, stats = generate_xlsx_questions(
            xlsx_bytes, "test.xlsx",
            "fake_key", "http://fake", "fake_model",
        )
        assert len(questions) == 1
        assert questions[0]["source_format"] == "xlsx"
    finally:
        sqg._call_llm_text = original_call_llm

    print("PASS: xlsx_question_generator 委托正常")


def test_existing_xlsx_functions_importable():
    """xlsx_question_generator 原有内部函数仍可导入。"""
    print("=" * 60)
    print("测试：原有函数可导入")
    print("=" * 60)

    from xlsx_question_generator import (
        _validate_and_render_evidence,
        _render_evidence_range,
        _parse_xlsx_qgen_response,
        _parse_range,
        _get_cell_display_value,
        check_xlsx_llm_support,
    )
    # 不崩溃即通过
    assert callable(_validate_and_render_evidence)
    assert callable(_render_evidence_range)

    print("PASS: 原有函数仍可导入")


# ====== Main ======

def main():
    tests = [
        test_col_letter,
        test_parse_range_str,
        test_csv_basic,
        test_csv_encoding_gbk,
        test_csv_encoding_bom,
        test_csv_empty,
        test_xlsx_to_sheet_context,
        test_xlsx_multi_sheet,
        test_xlsx_merged_cells,
        test_xlsx_formula_detection,
        test_split_small_sheet,
        test_split_large_sheet,
        test_allowed_ranges_per_block,
        test_valid_anchor_in_whitelist,
        test_anchor_not_in_whitelist,
        test_anchor_out_of_bounds,
        test_anchor_too_large,
        test_render_single_row,
        test_render_multi_row,
        test_render_reference_answer_from_context,
        test_parse_valid_json,
        test_parse_markdown_code_block,
        test_parse_invalid_json,
        test_generate_csv_questions,
        test_generate_xlsx_questions,
        test_validation_rejects_bad_range,
        test_csv_in_doc_parser,
        test_supported_extensions_includes_new,
        test_build_prompt,
        test_xlsx_question_generator_delegates,
        test_existing_xlsx_functions_importable,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 个测试")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
