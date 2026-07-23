"""
统一电子表格检索题生成模块。

支持 XLSX、XLS、CSV 三种格式，统一架构：
1. 本地解析表格 → SheetContext 结构化对象
2. 渲染带锚点的 Markdown 表格块 → 发给 LLM
3. LLM 只返回题目 + sheet_name + anchor_range（不输出 reference_answer）
4. 本地按 anchor_range 从原始数据重新渲染金标准证据

核心原则：
- 表格源文件是唯一事实来源
- LLM 不生成、不决定 reference_answer
- reference_answer 必须由本地原始数据按 anchor 渲染得到
"""

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from question_generator import deduplicate_questions, MODE_RETRIEVAL

# ─── 常量 ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "qgen_prompt_spreadsheet_retrieval.txt"
_MAX_BLOCK_ROWS = 30       # 单个表格块最大行数
_MAX_EVIDENCE_ROWS = 20    # 单个证据范围最大行数
_MAX_EVIDENCE_COLS = 15    # 单个证据范围最大列数


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class SheetContext:
    sheet_name: str
    max_row: int
    max_col: int
    headers: list                           # 列标题字符串列表
    rows: list                              # 二维数组，rows[0] = Excel 第 1 行
    merged_cells: list = field(default_factory=list)
    formula_cells_without_cache: list = field(default_factory=list)
    format_warnings: list = field(default_factory=list)
    allowed_anchor_ranges: list = field(default_factory=list)
    table_blocks: list = field(default_factory=list)


@dataclass
class TableBlock:
    block_index: int
    markdown: str
    row_range: tuple                        # (start_row, end_row) 含两端，1-indexed
    col_range: tuple                        # (start_col, end_col) 含两端，1-indexed
    allowed_anchor_ranges: list
    has_formula_warnings: bool


# ─── 列字母转换（独立于 openpyxl） ────────────────────────────────────────────

def _col_letter(n):
    """将 1-indexed 列号转换为 Excel 列字母（1→A, 26→Z, 27→AA）。"""
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _col_index(letter):
    """将 Excel 列字母转换为 1-indexed 列号（A→1, Z→26, AA→27）。"""
    result = 0
    for ch in letter.upper():
        result = result * 26 + (ord(ch) - 64)
    return result


def _parse_range_str(range_str):
    """解析 'A1:C5' 格式的范围字符串，返回 (min_col, min_row, max_col, max_row)，1-indexed。

    Returns None if invalid.
    """
    range_str = range_str.strip().upper()
    match = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', range_str)
    if not match:
        return None
    try:
        min_col = _col_index(match.group(1))
        min_row = int(match.group(2))
        max_col = _col_index(match.group(3))
        max_row = int(match.group(4))
        if min_row < 1 or min_col < 1 or max_row < min_row or max_col < min_col:
            return None
        return (min_col, min_row, max_col, max_row)
    except (ValueError, Exception):
        return None


def _range_to_str(min_col, min_row, max_col, max_row):
    """将行列范围转为 Excel 范围字符串。"""
    return f"{_col_letter(min_col)}{min_row}:{_col_letter(max_col)}{max_row}"


# ─── CSV 编码探测 ─────────────────────────────────────────────────────────────

def _detect_csv_encoding(file_bytes):
    """依次尝试 UTF-8-sig、UTF-8、GBK，最后回退 charset_normalizer。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            file_bytes.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    try:
        import charset_normalizer
        result = charset_normalizer.from_bytes(file_bytes).best()
        if result:
            return result.encoding
    except Exception:
        pass
    return "utf-8"


# ─── 格式解析器 ───────────────────────────────────────────────────────────────

def parse_xlsx_to_sheet_contexts(file_bytes):
    """解析 XLSX 文件为 SheetContext 列表。

    使用 openpyxl 双重打开：data_only=False 获取公式，data_only=True 获取缓存值。
    支持合并单元格填充、公式检测。
    """
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=False)
    try:
        wb_cached = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception:
        wb_cached = None

    contexts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        ws_cached = wb_cached[sheet_name] if wb_cached and sheet_name in wb_cached.sheetnames else None

        max_row = ws.max_row or 0
        max_col = ws.max_column or 0
        if max_row < 1 or max_col < 1:
            continue

        # 读取原始数据（含公式文本）
        rows = []
        for r in range(1, max_row + 1):
            row_vals = []
            for c in range(1, max_col + 1):
                row_vals.append(ws.cell(row=r, column=c).value)
            rows.append(row_vals)

        # 处理合并单元格：将左上角的值填充到范围内所有单元格
        merged_cells = []
        for merge_range in ws.merged_cells.ranges:
            min_col, min_row, max_col_m, max_row_m = (
                merge_range.min_col, merge_range.min_row,
                merge_range.max_col, merge_range.max_row,
            )
            merged_cells.append((min_row, min_col, max_row_m, max_col_m))
            top_left_val = rows[min_row - 1][min_col - 1]
            for r in range(min_row, max_row_m + 1):
                for c in range(min_col, max_col_m + 1):
                    if r != min_row or c != min_col:
                        rows[r - 1][c - 1] = top_left_val

        # 检测公式单元格并获取缓存值
        formula_cells_without_cache = []
        for r in range(1, max_row + 1):
            for c in range(1, max_col + 1):
                val = rows[r - 1][c - 1]
                if isinstance(val, str) and val.startswith("="):
                    # 有公式，尝试从 cached 版本获取缓存值
                    cached_val = None
                    if ws_cached:
                        cached_val = ws_cached.cell(row=r, column=c).value
                    if cached_val is not None:
                        rows[r - 1][c - 1] = cached_val
                    else:
                        rows[r - 1][c - 1] = "[公式未计算]"
                        formula_cells_without_cache.append((r, c))

        # 表头：第 1 行
        headers = [str(v).strip() if v is not None else f"列{_col_letter(c)}" for c, v in enumerate(rows[0], 1)]

        # 计算允许的锚定范围
        allowed = _compute_allowed_anchor_ranges(rows, max_row, max_col)

        ctx = SheetContext(
            sheet_name=sheet_name,
            max_row=max_row,
            max_col=max_col,
            headers=headers,
            rows=rows,
            merged_cells=merged_cells,
            formula_cells_without_cache=formula_cells_without_cache,
            format_warnings=[],
            allowed_anchor_ranges=allowed,
        )
        _split_into_table_blocks(ctx)
        contexts.append(ctx)

    wb.close()
    if wb_cached:
        wb_cached.close()
    return contexts


def parse_xls_to_sheet_contexts(file_bytes):
    """解析 XLS 文件为 SheetContext 列表。

    使用 pandas + xlrd。不支持公式检测和合并单元格。
    """
    try:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine='xlrd', dtype=str)
    except ImportError:
        raise ValueError("XLS 格式需要安装 xlrd 库。请运行: pip install xlrd")
    except Exception as e:
        raise ValueError(f"无法解析 XLS 文件: {e}")

    contexts = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df = df.fillna("")
        headers = [str(c) for c in df.columns.tolist()]
        rows = [headers] + [[str(v) for v in row] for row in df.values.tolist()]
        max_row = len(rows)
        max_col = len(headers)

        allowed = _compute_allowed_anchor_ranges(rows, max_row, max_col)

        ctx = SheetContext(
            sheet_name=str(sheet_name),
            max_row=max_row,
            max_col=max_col,
            headers=headers,
            rows=rows,
            merged_cells=[],
            formula_cells_without_cache=[],
            format_warnings=["XLS 格式不保留公式信息，单元格显示值可能为缓存计算结果"],
            allowed_anchor_ranges=allowed,
        )
        _split_into_table_blocks(ctx)
        contexts.append(ctx)

    return contexts


def parse_csv_to_sheet_contexts(file_bytes, file_name=""):
    """解析 CSV 文件为 SheetContext 列表（单工作表 "CSV"）。

    支持 UTF-8、UTF-8 BOM、GBK 编码探测。
    """
    encoding = _detect_csv_encoding(file_bytes)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"无法解析 CSV 文件 (编码 {encoding}): {e}")

    if df.empty:
        raise ValueError("CSV 文件为空或无有效数据")

    headers = [str(c) for c in df.columns.tolist()]
    rows = [headers] + [[str(v) for v in row] for row in df.values.tolist()]
    max_row = len(rows)
    max_col = len(headers)

    allowed = _compute_allowed_anchor_ranges(rows, max_row, max_col)

    ctx = SheetContext(
        sheet_name="CSV",
        max_row=max_row,
        max_col=max_col,
        headers=headers,
        rows=rows,
        merged_cells=[],
        formula_cells_without_cache=[],
        format_warnings=[],
        allowed_anchor_ranges=allowed,
    )
    _split_into_table_blocks(ctx)
    return [ctx]


# ─── 锚定范围计算 ─────────────────────────────────────────────────────────────

def _compute_allowed_anchor_ranges(rows, max_row, max_col):
    """扫描网格，找出所有连续非空矩形区域，返回允许的锚定范围列表。

    策略：逐行扫描，识别连续非空列组，输出每个连续区域的范围字符串。
    每个数据行（第 2 行起）单独作为一个允许范围，加上表头+数据的组合范围。
    """
    if max_row < 2 or max_col < 1:
        return []

    allowed = []

    # 每个数据行（第 2 行起）单独作为一个允许范围
    for r in range(2, max_row + 1):
        # 找该行的连续非空列组
        non_empty_cols = []
        for c in range(1, max_col + 1):
            val = rows[r - 1][c - 1] if c - 1 < len(rows[r - 1]) else None
            if val is not None and str(val).strip():
                non_empty_cols.append(c)

        if non_empty_cols:
            # 找连续段
            start = non_empty_cols[0]
            prev = non_empty_cols[0]
            for col in non_empty_cols[1:]:
                if col == prev + 1:
                    prev = col
                else:
                    allowed.append(_range_to_str(start, r, prev, r))
                    start = col
                    prev = col
            allowed.append(_range_to_str(start, r, prev, r))

    # 表头+数据的组合范围（整个连续非空区域）
    for c_start in range(1, max_col + 1):
        # 找连续非空列组
        if not any(
            rows[r][c_start - 1] is not None and str(rows[r][c_start - 1]).strip()
            for r in range(min(2, max_row), max_row)
        ):
            continue
        c_end = c_start
        while c_end + 1 <= max_col and any(
            rows[r][c_end] is not None and str(rows[r][c_end]).strip()
            for r in range(min(2, max_row), max_row)
        ):
            c_end += 1
        # 找行范围
        r_start = 1
        r_end = max_row
        range_str = _range_to_str(c_start, r_start, c_end, r_end)
        if range_str not in allowed:
            allowed.append(range_str)
        c_start = c_end + 1

    # 去重保持顺序
    seen = set()
    unique = []
    for r in allowed:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


# ─── 表格块拆分 ───────────────────────────────────────────────────────────────

def _split_into_table_blocks(sheet_ctx):
    """将 SheetContext 的行拆分为多个 TableBlock，每块最多 _MAX_BLOCK_ROWS 行数据。

    对于表头行+数值行的费率/参数表模式，额外生成语义化二列块。
    """
    max_row = sheet_ctx.max_row
    max_col = sheet_ctx.max_col
    headers = sheet_ctx.headers

    if max_row <= 1:
        # 只有表头，1 个块
        block = TableBlock(
            block_index=0,
            markdown=_render_block_markdown(sheet_ctx.rows[:1], headers, 1, 1),
            row_range=(1, 1),
            col_range=(1, max_col),
            allowed_anchor_ranges=[],
            has_formula_warnings=False,
        )
        sheet_ctx.table_blocks = [block]
        return

    blocks = []
    block_idx = 0
    # 数据从第 2 行开始，每 _MAX_BLOCK_ROWS 行一块
    data_start = 2
    while data_start <= max_row:
        data_end = min(data_start + _MAX_BLOCK_ROWS - 1, max_row)
        # 包含表头行
        block_rows = [sheet_ctx.rows[0]] + sheet_ctx.rows[data_start - 1:data_end]

        # 计算该块的 allowed_anchor_ranges 子集
        block_allowed = []
        for r_str in sheet_ctx.allowed_anchor_ranges:
            bounds = _parse_range_str(r_str)
            if bounds is None:
                continue
            _, r_min, _, r_max = bounds
            if r_min >= data_start and r_max <= data_end:
                block_allowed.append(r_str)

        has_formula = any(
            data_start <= r <= data_end
            for r, _ in sheet_ctx.formula_cells_without_cache
        )

        block = TableBlock(
            block_index=block_idx,
            markdown=_render_block_markdown(block_rows, headers, data_start, data_end),
            row_range=(data_start, data_end),
            col_range=(1, max_col),
            allowed_anchor_ranges=block_allowed,
            has_formula_warnings=has_formula,
        )
        blocks.append(block)
        block_idx += 1

        # 检测表头行+数值行模式，生成语义化二列块
        data_rows = sheet_ctx.rows[data_start - 1:data_end]
        hv_pairs = _detect_header_value_row_pairs(data_rows, headers)
        for label_idx, value_idx in hv_pairs:
            if label_idx == -1:
                # headers 作为标签行
                excel_label_row = 1  # Excel 第 1 行是表头
                excel_value_row = data_start + value_idx
                pair_rows = [headers, data_rows[value_idx]]
            else:
                excel_label_row = data_start + label_idx
                excel_value_row = data_start + value_idx
                pair_rows = [data_rows[label_idx], data_rows[value_idx]]
            semantic_md, field_anchors = _render_semantic_block(
                pair_rows, headers, excel_label_row, sheet_ctx
            )
            if semantic_md and field_anchors:
                # 每个字段-数值对的 anchor 都是合法的
                pair_allowed = [anchor for _, anchor in field_anchors]
                # 加上整行范围
                pair_range = f"{_col_letter(1)}{excel_label_row}:{_col_letter(max_col)}{excel_value_row}"
                pair_allowed.append(pair_range)

                sem_block = TableBlock(
                    block_index=block_idx,
                    markdown=semantic_md,
                    row_range=(excel_label_row, excel_value_row),
                    col_range=(1, max_col),
                    allowed_anchor_ranges=pair_allowed,
                    has_formula_warnings=has_formula,
                )
                blocks.append(sem_block)
                block_idx += 1

        data_start = data_end + 1

    sheet_ctx.table_blocks = blocks

    # 将语义块的 field anchors 添加到 sheet 级白名单，确保验证通过
    for block in blocks:
        if block.allowed_anchor_ranges:
            for anchor in block.allowed_anchor_ranges:
                if anchor not in sheet_ctx.allowed_anchor_ranges:
                    sheet_ctx.allowed_anchor_ranges.append(anchor)


def _render_block_markdown(rows, headers, data_start_row, data_end_row):
    """渲染一个表格块为 Markdown，带 Excel 行号列。"""
    lines = []

    # 表头行
    header_cells = [str(h) if h else "" for h in headers]
    lines.append("| 行号 | " + " | ".join(header_cells) + " |")
    lines.append("|---:|" + "|".join(["---"] * len(header_cells)) + "|")

    # 数据行（第 2 个元素起是数据行，对应 Excel 行 data_start_row）
    for i, row in enumerate(rows[1:], start=0):
        excel_row = data_start_row + i
        cells = []
        for v in row:
            if v is None:
                cells.append("")
            else:
                cells.append(str(v))
        # 补齐列数
        while len(cells) < len(header_cells):
            cells.append("")
        lines.append(f"| {excel_row} | " + " | ".join(cells[:len(header_cells)]) + " |")

    return "\n".join(lines)


# ─── 语义化渲染：表头行+数值行 费率/参数表 ─────────────────────────────────────

def _is_numeric_value(val):
    """判断值是否为数值（数字、百分比、货币等）。"""
    if val is None:
        return False
    s = str(val).strip()
    if not s:
        return False
    # 纯数字
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        pass
    # 百分比
    if s.endswith("%") and s[:-1].replace(".", "").replace(",", "").isdigit():
        return True
    # 货币
    for prefix in ("¥", "$", "€", "￥"):
        if s.startswith(prefix):
            try:
                float(s[len(prefix):].replace(",", ""))
                return True
            except ValueError:
                pass
    return False


def _detect_header_value_row_pairs(rows, headers=None):
    """检测表头行+数值行的费率/参数表模式。

    模式：某行是字段名（文本），下一行是对应数值（数字）。
    常见于：费率表、参数表、配置表等。
    也检测 Excel 表头（headers 参数）与第一行数据的配对。

    Returns:
        list of (label_row_idx, value_row_idx) — 0-indexed in rows array
        其中 label_row_idx=-1 表示使用 headers 作为标签行
    """
    if len(rows) < 1:
        return []

    pairs = []

    # 检查 headers + rows[0] 是否构成表头+数值对（费率/参数表模式）
    # 严格条件：数据行中只有第一列是文本（行标签），其余列全部是数值
    if headers and len(rows) >= 1:
        row_value = rows[0]
        total_cols = max(len(headers), len(row_value))
        # 统计数据行中的文本列和数值列
        text_val_cols = []
        numeric_val_cols = []
        for c in range(total_cols):
            v = row_value[c] if c < len(row_value) else None
            if v is not None and str(v).strip():
                if _is_numeric_value(v):
                    numeric_val_cols.append(c)
                else:
                    text_val_cols.append(c)
        # 条件：只有 1 个文本列（行标签）+ 至少 2 个数值列
        # 且文本列对应的 header 也是文本
        # 且只有 1 行数据（单行费率表；多行由内部行对检测处理）
        if (len(text_val_cols) == 1 and len(numeric_val_cols) >= 2
                and len(rows) == 1):
            label_col = text_val_cols[0]
            h = headers[label_col] if label_col < len(headers) else None
            if h and str(h).strip() and not _is_numeric_value(h):
                pairs.append((-1, 0))

    # 检查 rows 内部的相邻行对
    i = 0
    while i < len(rows) - 1:
        # 跳过已作为 value 被配对的行
        if pairs and pairs[-1][1] == i:
            i += 1
            continue

        row_label = rows[i]
        row_value = rows[i + 1]
        total_cols = max(len(row_label), len(row_value))

        # 统计 label 行和 value 行的文本/数值列
        text_label = []
        text_val = []
        numeric_val = []
        for c in range(total_cols):
            lv = row_label[c] if c < len(row_label) else None
            vv = row_value[c] if c < len(row_value) else None
            if lv is not None and str(lv).strip() and not _is_numeric_value(lv):
                text_label.append(c)
            if isinstance(vv, str) and vv == "[公式未计算]":
                continue
            if vv is not None and str(vv).strip():
                if _is_numeric_value(vv):
                    numeric_val.append(c)
                else:
                    text_val.append(c)

        matched = False

        # 模式 1：value 行只有 1 个文本列（行标签）+ 至少 2 个数值列
        if len(text_val) == 1 and len(numeric_val) >= 2:
            label_col = text_val[0]
            lv = row_label[label_col] if label_col < len(row_label) else None
            if lv is not None and str(lv).strip() and not _is_numeric_value(lv):
                label_val_text = str(lv).strip()
                repeat_count = sum(
                    1 for r in rows
                    if label_col < len(r) and str(r[label_col]).strip() == label_val_text
                )
                if repeat_count >= 2:
                    matched = True

        # 模式 2：label 行文本列远多于 value 行，且 value 行有足够数值列
        # 例如：label 行 12 个文本列，value 行 3 个文本列 + 10 个数值列
        if not matched and len(text_label) > len(text_val) + 3 and len(numeric_val) >= 3:
            # 且 label 行的文本列在 value 行中大部分变为数值
            converted = sum(1 for c in text_label if c in numeric_val)
            if converted >= 3:
                matched = True

        if matched:
            pairs.append((i, i + 1))
            i += 2
            continue
        i += 1

    return pairs


def _render_semantic_block(rows, headers, data_start_row, sheet_ctx):
    """将表头行+数值行渲染为规范化二列 Markdown 表格。

    输入：rows[0] 是字段名行，rows[1] 是数值行。
    输出：| 字段名 | 数值 | 格式的 Markdown 表格。

    同时返回每个字段-数值对的 anchor_range（如 E2:E3）。
    如果第一列是行标签（非数值），自动跳过该列。
    """
    label_row = rows[0]
    value_row = rows[1]
    total_cols = max(len(label_row), len(value_row))

    # 检测第一列是否为行标签（标签行的第一列是文本，数值行的第一列也是文本/非数值）
    start_col = 0
    if total_cols > 1:
        label_first = label_row[0] if len(label_row) > 0 else None
        value_first = value_row[0] if len(value_row) > 0 else None
        if (label_first is not None and str(label_first).strip() and
                not _is_numeric_value(label_first) and
                value_first is not None and str(value_first).strip() and
                not _is_numeric_value(value_first)):
            start_col = 1  # 跳过第一列（行标签列）

    pairs = []  # (field_name, value, col_idx)
    for c in range(start_col, total_cols):
        label = label_row[c] if c < len(label_row) else None
        value = value_row[c] if c < len(value_row) else None
        # 只包含 label 是文本且 value 是数值的配对（真正的字段-数值对）
        # 跳过 label 和 value 都是文本的情况（如 "功能模块 | 功能模块"）
        if (label is not None and str(label).strip() and not _is_numeric_value(label)
                and value is not None and _is_numeric_value(value)):
            pairs.append((str(label).strip(), value, c))

    if not pairs:
        return None, []

    # 渲染二列 Markdown
    lines = []
    lines.append("| 字段名 | 数值 |")
    lines.append("|---|---|")
    for field_name, value, _ in pairs:
        value_str = str(value) if value is not None else ""
        if isinstance(value, str) and value == "[公式未计算]":
            value_str = "[公式未计算]"
        lines.append(f"| {field_name} | {value_str} |")

    # 生成每个字段-数值对的 anchor_range
    label_excel_row = data_start_row
    value_excel_row = data_start_row + 1
    field_anchors = []
    for field_name, value, col_idx in pairs:
        col_letter = _col_letter(col_idx + 1)
        # 单列锚定：label_row:value_row
        anchor = f"{col_letter}{label_excel_row}:{col_letter}{value_excel_row}"
        field_anchors.append((field_name, anchor))

    return "\n".join(lines), field_anchors


# ─── Prompt 构建 ──────────────────────────────────────────────────────────────

def _build_prompt(sheets, num_questions, topic_hint=""):
    """构建发给 LLM 的完整 prompt。"""
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()

    topic_hint_section = ""
    if topic_hint:
        topic_hint_section = f"- 主题方向：{topic_hint}"

    # 渲染所有表格块
    block_texts = []
    for sheet in sheets:
        for block in sheet.table_blocks:
            formula_warn = ""
            if block.has_formula_warnings:
                formula_warn = "\n\n⚠️ 本块包含公式单元格（无缓存计算值），请避免引用这些单元格作为核心证据。"
            block_text = (
                f"### 工作表: {sheet.sheet_name} — 表格块 {block.block_index + 1} "
                f"(行 {block.row_range[0]}-{block.row_range[1]})\n\n"
                f"{block.markdown}\n\n"
                f"allowed_anchor_ranges: {json.dumps(block.allowed_anchor_ranges, ensure_ascii=False)}"
                f"{formula_warn}"
            )
            block_texts.append(block_text)

    table_blocks_text = "\n\n---\n\n".join(block_texts)

    prompt = template.replace("{table_blocks_text}", table_blocks_text)
    prompt = prompt.replace("{num_questions}", str(num_questions))
    prompt = prompt.replace("{topic_hint_section}", topic_hint_section)
    return prompt


# ─── LLM 响应解析 ─────────────────────────────────────────────────────────────

def _parse_llm_response(text):
    """解析 LLM 返回的 JSON 数组。"""
    text = text.strip()

    # 去除 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines).strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 数组
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


# ─── 锚定范围验证 ─────────────────────────────────────────────────────────────

def _validate_anchor_range(anchor_range, allowed_ranges, max_row, max_col, sheet_ctx=None):
    """验证锚定范围是否合法。

    规则：
    1. 范围必须是某个白名单范围的子集
    2. 单行范围：直接通过
    3. 跨行范围：只允许表头(row 1) + 紧邻的一行数据(row 2)
       - 不允许跨越多条业务数据行
       - 语义块的 field anchors（如 B1:B2）自动满足此规则

    Returns (is_valid, reason).
    """
    bounds = _parse_range_str(anchor_range)
    if bounds is None:
        return False, f"无法解析范围 '{anchor_range}'"

    min_col, min_row, max_col_r, max_row_r = bounds

    if min_row < 1 or min_col < 1:
        return False, f"范围 {anchor_range} 起始位置越界"
    if max_row_r > max_row or max_col_r > max_col:
        return False, f"范围 {anchor_range} 超出工作表边界 (max_row={max_row}, max_col={max_col})"

    rows = max_row_r - min_row + 1
    cols = max_col_r - min_col + 1
    if rows > _MAX_EVIDENCE_ROWS:
        return False, f"证据范围行数 ({rows}) 超过上限 {_MAX_EVIDENCE_ROWS}"
    if cols > _MAX_EVIDENCE_COLS:
        return False, f"证据范围列数 ({cols}) 超过上限 {_MAX_EVIDENCE_COLS}"

    # 跨行范围收紧：只允许相邻两行（label 行 + 紧邻的 value 行）
    # 不允许跨越多条业务数据行
    if rows > 1:
        if rows != 2:
            return False, (
                f"跨行范围 {anchor_range} 不合法："
                f"只允许相邻两行（字段名行 + 数值行），"
                f"不允许跨越 {rows} 行"
            )

    # 检查白名单：范围必须是某个白名单范围的子集
    is_subset = False
    for allowed in allowed_ranges:
        allowed_bounds = _parse_range_str(allowed)
        if allowed_bounds is None:
            continue
        a_min_col, a_min_row, a_max_col, a_max_row = allowed_bounds
        if (min_col >= a_min_col and max_col_r <= a_max_col and
                min_row >= a_min_row and max_row_r <= a_max_row):
            is_subset = True
            break
    if not is_subset:
        return False, f"范围 '{anchor_range}' 不在白名单范围内"

    # 孤立数值检测：如果范围只覆盖一个或多个纯数值单元格，拒绝
    if sheet_ctx is not None:
        all_numeric = True
        has_any_value = False
        for r in range(min_row, max_row_r + 1):
            for c in range(min_col, max_col_r + 1):
                val = sheet_ctx.rows[r - 1][c - 1] if c - 1 < len(sheet_ctx.rows[r - 1]) else None
                if val is not None and str(val).strip():
                    has_any_value = True
                    if not _is_numeric_value(val):
                        all_numeric = False
                        break
            if not all_numeric:
                break
        if has_any_value and all_numeric:
            return False, (
                f"范围 {anchor_range} 只包含孤立数值，"
                f"数值类证据必须同时包含字段名称和对应数值"
            )

    return True, ""


# ─── 金标准渲染 ───────────────────────────────────────────────────────────────

def _render_reference_answer(anchor_range, sheet_ctx):
    """从 SheetContext 的 rows 数据中读取指定范围，渲染为 Markdown。

    Returns (rendered_text, has_formula_issue).
    """
    bounds = _parse_range_str(anchor_range)
    if bounds is None:
        return "", True

    min_col, min_row, max_col, max_row = bounds

    cell_values = []
    has_formula_issue = False
    for r in range(min_row, max_row + 1):
        row_vals = []
        for c in range(min_col, max_col + 1):
            val = sheet_ctx.rows[r - 1][c - 1] if c - 1 < len(sheet_ctx.rows[r - 1]) else None
            # 检查是否为公式单元格无缓存值
            if (r, c) in sheet_ctx.formula_cells_without_cache:
                has_formula_issue = True
                row_vals.append("[公式未计算]")
            elif isinstance(val, str) and val == "[公式未计算]":
                has_formula_issue = True
                row_vals.append(val)
            elif val is None:
                row_vals.append(None)
            else:
                row_vals.append(val)
        cell_values.append(row_vals)

    # 检查非空
    non_empty = sum(1 for row in cell_values for v in row if v is not None and str(v).strip())
    if non_empty == 0:
        return "", True

    rendered = _render_cell_values(cell_values)
    return rendered, has_formula_issue


def _render_cell_values(cell_values):
    """将二维单元格值数组渲染为 Markdown。"""
    if not cell_values:
        return ""

    # 单行：键值对格式
    if len(cell_values) == 1:
        parts = [str(v) for v in cell_values[0] if v is not None]
        return " | ".join(parts) if parts else ""

    # 多行：表格格式
    lines = []
    header = cell_values[0]
    header_cells = [str(v) if v is not None else "" for v in header]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

    for row in cell_values[1:]:
        row_cells = [str(v) if v is not None else "" for v in row]
        while len(row_cells) < len(header_cells):
            row_cells.append("")
        lines.append("| " + " | ".join(row_cells[:len(header_cells)]) + " |")

    return "\n".join(lines)


# ─── 题目验证与渲染 ───────────────────────────────────────────────────────────

def _validate_and_render_question(raw_q, sheets_by_name, file_name):
    """验证 LLM 返回的单条题目，渲染金标准证据。

    Returns (validated_dict_or_None, rejection_reason).
    """
    q_text = (raw_q.get("question") or "").strip()
    sheet_name = (raw_q.get("sheet_name") or "").strip()
    anchor_range = (raw_q.get("anchor_range") or "").strip()

    if not q_text:
        return None, "query 为空"
    if not sheet_name:
        return None, "sheet_name 为空"
    if not anchor_range:
        return None, "anchor_range 为空"

    # 查找工作表
    sheet = sheets_by_name.get(sheet_name)
    if sheet is None:
        return None, f"工作表 '{sheet_name}' 不存在，可用: {list(sheets_by_name.keys())}"

    # 验证范围
    valid, reason = _validate_anchor_range(
        anchor_range, sheet.allowed_anchor_ranges, sheet.max_row, sheet.max_col,
        sheet_ctx=sheet,
    )
    if not valid:
        return None, reason

    # 渲染金标准
    rendered, has_formula_issue = _render_reference_answer(anchor_range, sheet)
    if not rendered:
        return None, "证据范围为空"
    if has_formula_issue:
        return None, "证据范围包含公式单元格（无缓存计算值）"

    # 组装验证后的题目
    validated = {
        "question": q_text,
        "reference_answer": rendered,
        "source_excerpt": rendered,
        "sheet_name": sheet_name,
        "anchor_range": anchor_range,
        "evidence_sheet": sheet_name,       # 向后兼容
        "evidence_range": anchor_range,     # 向后兼容
        "source_format": _detect_format(file_name),
        "source_file_name": file_name,
        "difficulty": raw_q.get("difficulty", "事实"),
        "topic": raw_q.get("topic", ""),
    }

    return validated, ""


def _detect_format(file_name):
    """从文件名推断格式。"""
    ext = Path(file_name).suffix.lower()
    if ext == ".xlsx":
        return "xlsx"
    elif ext == ".xls":
        return "xls"
    elif ext == ".csv":
        return "csv"
    return "unknown"


# ─── LLM 调用 ─────────────────────────────────────────────────────────────────

def _call_llm_text(prompt, api_key, base_url, model, timeout=120):
    """标准 OpenAI 兼容 chat completion 请求（纯文本，无文件附件）。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f"请求超时 ({timeout}s): {url}")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"连接失败: {url}\n{e}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} | URL: {url}\nResponse: {resp.text[:1000]}"
        )

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"JSON 解析失败 | Response: {resp.text[:1000]}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"响应结构异常 | Response: {json.dumps(data, ensure_ascii=False)[:1000]}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def generate_spreadsheet_questions(file_bytes, file_name, api_key, base_url, model,
                                   num_questions=5, difficulty="混合", topic_hint="",
                                   timeout=120, progress_callback=None, mode="retrieval"):
    """统一电子表格检索题生成。

    Args:
        file_bytes: 文件原始字节
        file_name: 文件名（用于推断格式）
        api_key, base_url, model: LLM 配置
        num_questions: 目标题数
        difficulty: 难度偏好
        topic_hint: 主题方向
        timeout: LLM 超时秒数
        progress_callback: 进度回调 (step, total, description)
        mode: 题目模式 ("retrieval" 或 "qa")

    Returns:
        tuple: (questions_list, stats_dict)
    """
    if progress_callback:
        progress_callback(0, 4, "解析表格文件")

    # 1. 检测格式并解析
    ext = Path(file_name).suffix.lower()
    if ext == ".xlsx":
        sheets = parse_xlsx_to_sheet_contexts(file_bytes)
    elif ext == ".xls":
        sheets = parse_xls_to_sheet_contexts(file_bytes)
    elif ext == ".csv":
        sheets = parse_csv_to_sheet_contexts(file_bytes, file_name)
    else:
        raise ValueError(f"不支持的表格格式: {ext}")

    if not sheets:
        raise ValueError(f"文件 {file_name} 中未提取到任何工作表数据")

    sheets_by_name = {s.sheet_name: s for s in sheets}
    total_blocks = sum(len(s.table_blocks) for s in sheets)
    total_formula_warnings = sum(len(s.formula_cells_without_cache) for s in sheets)
    format_warnings = []
    for s in sheets:
        format_warnings.extend(s.format_warnings)

    if progress_callback:
        progress_callback(1, 4, "构建 LLM Prompt")

    # 2. 构建 prompt
    prompt = _build_prompt(sheets, num_questions, topic_hint)

    if progress_callback:
        progress_callback(2, 4, "调用 LLM 生成检索查询")

    # 3. 调用 LLM（标准文本 API，不发送文件）
    response_text = _call_llm_text(prompt, api_key, base_url, model, timeout=timeout)

    # 4. 解析响应
    raw_questions = _parse_llm_response(response_text)
    raw_count = len(raw_questions)

    if progress_callback:
        progress_callback(3, 4, "验证锚定范围并渲染金标准")

    # 5. 验证并渲染
    valid_questions = []
    validation_eliminated = 0
    for q in raw_questions:
        validated, reason = _validate_and_render_question(q, sheets_by_name, file_name)
        if validated:
            validated["question_mode"] = MODE_RETRIEVAL
            valid_questions.append(validated)
        else:
            validation_eliminated += 1
            print(f"  ⚠️ 锚定校验不通过（{reason}），已过滤")

    # 6. 去重
    unique_questions = deduplicate_questions(valid_questions)
    dedup_eliminated = len(valid_questions) - len(unique_questions)

    # 7. 裁剪到目标数
    if len(unique_questions) > num_questions:
        unique_questions = unique_questions[:num_questions]

    if not unique_questions:
        raise ValueError("出题失败：所有查询均未通过锚定校验")

    stats = {
        "raw_count": raw_count,
        "validation_eliminated": validation_eliminated,
        "dedup_eliminated": dedup_eliminated,
        "final_count": len(unique_questions),
        "target": num_questions,
        "sheet_count": len(sheets),
        "block_count": total_blocks,
        "formula_warnings": total_formula_warnings,
        "format_warnings": format_warnings,
    }
    return unique_questions, stats
