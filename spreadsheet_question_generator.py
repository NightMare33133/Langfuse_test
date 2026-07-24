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
    header_context_range: str = None        # 表头上下文范围（如 B2:D2），用于价格题


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

    # 检测"标准表头 + 单条业务数据行"模式（价格题语义块）
    # 真正的字段名在 row 2（sheet_ctx.rows[1]），不是 row 1 的通用列名
    field_header_row_idx = 1  # 字段名行在 sheet_ctx.rows 中的索引
    field_headers = sheet_ctx.rows[field_header_row_idx]  # 真正的字段名行
    all_rows_from_field_header = sheet_ctx.rows[field_header_row_idx:]  # 从字段名行开始
    hb_pairs = _detect_header_business_row_pairs(all_rows_from_field_header, field_headers)
    field_header_excel_row = field_header_row_idx + 1  # Excel 中的字段名行号 = 2

    # 找出表头中的文本列（用于 header_context_range）
    text_cols = [c for c, h in enumerate(field_headers) if h and str(h).strip() and not _is_numeric_value(h)]
    header_context_range = None
    if text_cols:
        min_tc = min(text_cols) + 1
        max_tc = max(text_cols) + 1
        header_context_range = f"{_col_letter(min_tc)}{field_header_excel_row}:{_col_letter(max_tc)}{field_header_excel_row}"

    for _, biz_idx in hb_pairs:
        biz_row = all_rows_from_field_header[biz_idx]
        biz_excel_row = field_header_excel_row + biz_idx  # 转为 Excel 行号

        # 业务行锚点：仅包含业务行本身（如 B4:D4）
        if text_cols:
            min_c = min(text_cols) + 1
            max_c = max(text_cols) + 1
            biz_anchor = f"{_col_letter(min_c)}{biz_excel_row}:{_col_letter(max_c)}{biz_excel_row}"
        else:
            continue

        # 渲染语义块 Markdown（表头+业务行，仅用于 LLM 理解）
        hb_md = _render_header_business_markdown(field_headers, biz_row, text_cols)

        if hb_md:
            hb_block = TableBlock(
                block_index=block_idx,
                markdown=hb_md,
                row_range=(biz_excel_row, biz_excel_row),
                col_range=(1, max_col),
                allowed_anchor_ranges=[biz_anchor],
                has_formula_warnings=False,
                header_context_range=header_context_range,
            )
            blocks.append(hb_block)
            block_idx += 1

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

        # 模式 2：label 行文本列明显多于 value 行，且 value 行有足够数值列
        # 例如：label 行 12 个文本列，value 行 3 个文本列 + 10 个数值列
        if not matched and len(text_label) >= len(text_val) + 2 and len(numeric_val) >= 3:
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


def _detect_header_business_row_pairs(rows, headers):
    """检测"标准表头 + 单条业务数据行"模式。

    适用于报价表等结构：
    - Row 2: 字段表头（功能模块, 产品功能, 未税价, 项目经理, ...）
    - Row 3: 费率数值行（B/C/D 为空，E-M 有数值）
    - Row 4+: 业务数据行（B/C 有文本，D 有价格数值，E-M 有人数）

    rows 参数是 sheet_ctx.rows[1:]（即从第 2 行开始的数据）。
    rows[0] 是表头行，rows[1] 是第一行数据（可能是费率行），rows[2]+ 是业务行。

    返回：(header_row_idx, business_row_idx) 对列表，0-indexed in rows
    header_row_idx=-1 表示使用 headers 作为表头行。

    识别条件：
    - 业务行有至少 1 个文本列 + 至少 1 个数值列
    - 业务行的所有非空列都在表头有对应字段名
    - 业务行之前有间隔行（空行或数值行，或表头行本身）
    """
    if not headers or len(rows) < 3:
        return []

    # 识别表头中有字段名的列
    header_cols = set()
    for c, h in enumerate(headers):
        if h and str(h).strip():
            header_cols.add(c)

    if not header_cols:
        return []

    # 检查是否存在"间隔行"（表头与业务数据之间的数值行或空行）
    # 间隔行特征：
    # 1. 大部分列为空或纯数值（如费率行）
    # 2. 部分列重复表头文本、部分列有数值（如报价表的费率行）
    # 3. 表头文本列在该行中重复出现，且有独立数值列
    has_gap = False
    if len(rows) >= 3:
        row1 = rows[1]  # 表头后的第一行
        row1_non_empty = sum(1 for v in row1 if v is not None and str(v).strip())
        row1_numeric = sum(1 for v in row1 if v is not None and _is_numeric_value(v))
        # 统计表头有字段名的列中，该行有多少重复表头文本
        same_as_header = sum(
            1 for c in header_cols
            if c < len(row1) and row1[c] is not None and headers[c] is not None
            and str(row1[c]).strip() == str(headers[c]).strip()
        )
        # 统计表头有字段名的列中，该行有多少为空
        empty_in_header = sum(
            1 for c in header_cols
            if c >= len(row1) or row1[c] is None or not str(row1[c]).strip()
        )
        # 间隔行条件（满足任一即可）：
        # a) 大部分为空
        # b) 纯数值
        # c) 部分重复表头+部分数值（至少 2 个重复 + 至少 2 个数值）
        # d) 表头列中至少 20% 为空（如费率行的 B/C/D 列）
        has_gap = (
            (row1_non_empty < len(header_cols) * 0.5)
            or (row1_non_empty > 0 and row1_numeric == row1_non_empty)
            or (same_as_header >= 2 and row1_numeric >= 2)
            or (empty_in_header >= max(2, len(header_cols) * 0.2))
        )

    if not has_gap:
        return []

    # 找到第一个间隔行之后的所有业务行
    # 间隔行是 rows[1]（已确认），业务行从 rows[2] 开始
    pairs = []
    for i in range(2, len(rows)):
        row = rows[i]
        row_text = set()
        row_num = set()
        for c, v in enumerate(row):
            if v is None or not str(v).strip():
                continue
            if _is_numeric_value(v):
                row_num.add(c)
            else:
                row_text.add(c)

        # 条件：有文本列 + 有数值列，且所有非空列都在表头中有字段名
        non_empty = row_text | row_num
        if row_text and row_num and non_empty.issubset(header_cols):
            pairs.append((-1, i))

    return pairs


def _render_header_business_block(headers, business_row, header_excel_row, business_excel_row, sheet_ctx):
    """渲染"表头 + 单条业务数据行"为语义块（保留用于向后兼容）。"""
    text_cols = [c for c, h in enumerate(headers) if h and str(h).strip() and not _is_numeric_value(h)]
    if not text_cols:
        return None, []

    pairs = []
    for c in text_cols:
        field_name = str(headers[c]).strip()
        value = business_row[c] if c < len(business_row) else None
        if value is not None and str(value).strip():
            pairs.append((field_name, value, c))

    if not pairs:
        return None, []

    lines = ["| 字段 | 值 |", "|---|---|"]
    for field_name, value, _ in pairs:
        lines.append(f"| {field_name} | {value} |")

    field_anchors = []
    for field_name, value, c in pairs:
        col_letter = _col_letter(c + 1)
        anchor = f"{col_letter}{header_excel_row}:{col_letter}{business_excel_row}"
        field_anchors.append((field_name, anchor))

    return "\n".join(lines), field_anchors


def _render_header_business_markdown(headers, business_row, text_cols):
    """渲染"表头 + 业务行"Markdown（仅用于 LLM 理解，不用于锚定）。

    格式：
    | 字段 | 值 |
    |---|---|
    | 功能模块 | CICD工具规范... |
    | 产品功能 | 集成发布流水线梳理 |
    | 未税价（元） | 73900 |
    """
    pairs = []
    for c in text_cols:
        field_name = str(headers[c]).strip() if c < len(headers) and headers[c] else ""
        value = business_row[c] if c < len(business_row) else None
        if field_name and value is not None and str(value).strip():
            pairs.append((field_name, value))

    if not pairs:
        return None

    lines = ["| 字段 | 值 |", "|---|---|"]
    for field_name, value in pairs:
        lines.append(f"| {field_name} | {value} |")
    return "\n".join(lines)


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


# ─── 题意-锚点一致性校验 ──────────────────────────────────────────────────────

# 数值/角色相关关键词
_NUMERIC_KEYWORDS = (
    "未税价", "价格", "报价", "费率", "投入", "人月", "人数", "工期",
    "比例", "金额", "配置", "人力", "开发", "工时",
)
_ROLE_KEYWORDS = (
    "项目经理", "研发经理", "DevOps专家", "DevOps工程师",
    "前端工程师", "后端工程师", "BA", "测试", "SRE工程师", "SRE",
)
_AGGREGATE_KEYWORDS = (
    "各角色", "各模块", "所有", "明细", "汇总", "总计", "配置清单",
    "人力配置", "开发投入",
)
_PRICE_KEYWORDS = ("未税价", "价格", "报价")


def _extract_semantic_field_names(sheets):
    """从所有语义块中提取字段名集合。"""
    field_names = set()
    for sheet in sheets:
        for block in sheet.table_blocks:
            if block.block_index > 0:  # 语义块
                for line in block.markdown.split("\n"):
                    if line.startswith("|") and "字段名" not in line and "---" not in line:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        if parts:
                            field_names.add(parts[0])
    return field_names


def _extract_semantic_anchors(sheets):
    """从所有语义块中提取 field anchor 集合。

    包括：
    - 费率表垂直锚点（如 E2:E3，2行）
    - 业务行锚点（如 B4:D4，1行，来自有 header_context_range 的语义块）
    """
    anchors = set()
    for sheet in sheets:
        for block in sheet.table_blocks:
            if block.block_index > 0:  # 语义块
                for anchor in block.allowed_anchor_ranges:
                    bounds = _parse_range_str(anchor)
                    if bounds:
                        min_col, min_row, max_col_r, max_row_r = bounds
                        row_count = max_row_r - min_row + 1
                        # 2 行：字段名+数值（如 E2:E3）
                        # 1 行 + 有 header_context_range：业务行锚点（如 B4:D4）
                        if row_count >= 2 or block.header_context_range:
                            anchors.add(anchor)
    return anchors


def _is_standalone_role_mention(text, role):
    """检查角色名是否在文本中作为独立实体出现（非子串）。

    使用排除法：如果角色名前面紧接特定前缀字符（构成复合词），则不是独立提及。
    """
    # 常见的会与角色名构成复合词的前缀
    _PREFIX_CHARS = set('自动化半全手动智能')
    idx = text.find(role)
    while idx >= 0:
        # 检查前一个字符是否是会构成复合词的前缀
        if idx > 0 and text[idx - 1] in _PREFIX_CHARS:
            idx = text.find(role, idx + 1)
            continue
        return True
    return False


def _validate_question_anchor_consistency(question, semantic_field_names, semantic_anchors, sheets_by_name):
    """校验题意与锚点的一致性。

    Returns (is_valid, reason).
    """
    q_text = (question.get("question") or "").strip()
    anchor_range = (question.get("anchor_range") or "").strip()
    sheet_name = (question.get("sheet_name") or "").strip()

    q_lower = q_text.lower()

    # 1. 检测聚合型题目
    for kw in _AGGREGATE_KEYWORDS:
        if kw in q_text:
            return False, f"聚合型题目（含'{kw}'），每题只能考一个知识点"

    # 2. 检测数值/角色相关题目
    is_numeric_question = any(kw in q_text for kw in _NUMERIC_KEYWORDS)
    is_role_question = any(kw in q_text for kw in _ROLE_KEYWORDS)

    if (is_numeric_question or is_role_question) and semantic_anchors:
        # 2a. anchor 必须是语义块来源的子集
        anchor_bounds = _parse_range_str(anchor_range)
        is_in_semantic = anchor_range in semantic_anchors
        if not is_in_semantic and anchor_bounds:
            # 检查是否是某个语义 anchor 的子集
            a_min_col, a_min_row, a_max_col, a_max_row = anchor_bounds
            for sa in semantic_anchors:
                sa_bounds = _parse_range_str(sa)
                if sa_bounds:
                    s_min_col, s_min_row, s_max_col, s_max_row = sa_bounds
                    if (a_min_col >= s_min_col and a_max_col <= s_max_col and
                            a_min_row >= s_min_row and a_max_row <= s_max_row):
                        is_in_semantic = True
                        break
        if not is_in_semantic:
            return False, (
                f"数值/角色题'{q_text}'的 anchor '{anchor_range}' "
                f"不在语义块中，数值类题只能使用语义块的字段名+数值锚点"
            )

        # 2b. 问题中出现具体角色名时，anchor 对应的字段名必须包含该角色
        # 只匹配独立的角色名（前后是标点、空格或字符串边界），避免"自动化测试"误匹配"测试"
        for role in _ROLE_KEYWORDS:
            if _is_standalone_role_mention(q_text, role):
                bounds = _parse_range_str(anchor_range)
                if bounds and sheet_name in sheets_by_name:
                    sheet = sheets_by_name[sheet_name]
                    min_col, min_row, _, _ = bounds
                    field_in_anchor = str(sheet.rows[min_row - 1][min_col - 1] or "").strip()
                    if role not in field_in_anchor:
                        return False, (
                            f"问题含角色'{role}'但 anchor '{anchor_range}' "
                            f"对应字段'{field_in_anchor}'，角色不匹配"
                        )
                break

        # 2c. 价格题：检查表头上下文是否包含价格字段列
        for kw in _PRICE_KEYWORDS:
            if kw in q_text:
                # 查找该锚点对应的语义块的 header_context_range
                header_ctx = None
                if sheet_name in sheets_by_name:
                    sheet = sheets_by_name[sheet_name]
                    anchor_bounds = _parse_range_str(anchor_range)
                    for block in sheet.table_blocks:
                        if not block.header_context_range:
                            continue
                        # 检查 anchor 是否在该块的 allowed_anchor_ranges 中（精确或子集）
                        for allowed in block.allowed_anchor_ranges:
                            if anchor_range == allowed:
                                header_ctx = block.header_context_range
                                break
                            allowed_bounds = _parse_range_str(allowed)
                            if anchor_bounds and allowed_bounds:
                                a_min_col, a_min_row, a_max_col, a_max_row = anchor_bounds
                                s_min_col, s_min_row, s_max_col, s_max_row = allowed_bounds
                                if (a_min_col >= s_min_col and a_max_col <= s_max_col and
                                        a_min_row >= s_min_row and a_max_row <= s_max_row):
                                    header_ctx = block.header_context_range
                                    break
                        if header_ctx:
                            break

                if header_ctx:
                    # 双源模型：检查表头上下文是否包含价格字段
                    h_bounds = _parse_range_str(header_ctx)
                    if h_bounds and sheet_name in sheets_by_name:
                        sheet = sheets_by_name[sheet_name]
                        h_min_col, h_min_row, h_max_col, _ = h_bounds
                        has_price_field = False
                        for c in range(h_min_col, h_max_col + 1):
                            field = str(sheet.rows[h_min_row - 1][c - 1] or "").strip()
                            if any(pk in field for pk in _PRICE_KEYWORDS):
                                has_price_field = True
                                break
                        if not has_price_field:
                            return False, (
                                f"价格题'{q_text}'的表头上下文不包含价格字段列"
                            )
                else:
                    # 非双源模型：检查 anchor 本身是否包含价格字段
                    bounds = _parse_range_str(anchor_range)
                    if bounds and sheet_name in sheets_by_name:
                        sheet = sheets_by_name[sheet_name]
                        min_col, min_row, max_col_r, _ = bounds
                        has_price_field = False
                        for c in range(min_col, max_col_r + 1):
                            field = str(sheet.rows[min_row - 1][c - 1] or "").strip()
                            if any(pk in field for pk in _PRICE_KEYWORDS):
                                has_price_field = True
                                break
                        if not has_price_field:
                            return False, (
                                f"价格题'{q_text}'的 anchor '{anchor_range}' "
                                f"不包含价格字段列"
                            )
                break

    return True, ""


# ─── 候选锚点构建 ─────────────────────────────────────────────────────────────

def _build_candidate_anchors(sheets):
    """为 LLM 构建合法候选锚点清单，按类别分组。

    Returns:
        dict: {
            "rate_anchors": [{"anchor": "E2:E3", "field": "项目经理", "value": "1700"}, ...],
            "price_anchors": [{"anchor": "D4:D4", "header_ctx": "B2:M2", "product": "集成发布流水线梳理"}, ...],
            "text_anchors": [{"anchor": "B4:C4", "content": "CICD工具规范 | 集成发布流水线梳理"}, ...],
        }
    """
    rate_anchors = []
    price_anchors = []
    text_anchors = []

    for sheet in sheets:
        for block in sheet.table_blocks:
            if block.block_index == 0:
                continue  # 跳过标准块

            if block.header_context_range:
                # 双源模型：业务行锚点
                # 收集本块所有单行锚点的行号和列范围，构建完整业务行锚点
                row_anchors = {}  # row_num -> (min_col, max_col)
                for anchor in block.allowed_anchor_ranges:
                    bounds = _parse_range_str(anchor)
                    if not bounds:
                        continue
                    min_col, min_row, max_col, max_row = bounds
                    if max_row != min_row:
                        continue  # 只取单行锚点
                    row_num = min_row
                    if row_num not in row_anchors:
                        row_anchors[row_num] = (min_col, max_col)
                    else:
                        old_min, old_max = row_anchors[row_num]
                        row_anchors[row_num] = (min(min_col, old_min), max(max_col, old_max))

                h_bounds = _parse_range_str(block.header_context_range)
                if not h_bounds:
                    continue
                h_min_col, h_min_row, h_max_col, _ = h_bounds

                for row_num, (rc_min, rc_max) in row_anchors.items():
                    # 构建完整业务行锚点：覆盖 header_context_range 的全部列
                    biz_anchor = f"{_col_letter(h_min_col)}{row_num}:{_col_letter(h_max_col)}{row_num}"
                    # 读取业务行数据
                    row_data = sheet.rows[row_num - 1] if row_num - 1 < len(sheet.rows) else []
                    # 找价格字段
                    price_field = None
                    for c in range(h_min_col, h_max_col + 1):
                        h_val = sheet.rows[h_min_row - 1][c - 1] if c - 1 < len(sheet.rows[h_min_row - 1]) else None
                        if h_val and any(pk in str(h_val) for pk in _PRICE_KEYWORDS):
                            price_field = str(h_val).strip()
                            break
                    # 找功能模块和产品功能
                    func_module = ""
                    product_func = ""
                    for c in range(h_min_col, h_max_col + 1):
                        h_val = sheet.rows[h_min_row - 1][c - 1] if c - 1 < len(sheet.rows[h_min_row - 1]) else None
                        if not h_val:
                            continue
                        h_str = str(h_val).strip()
                        if "功能模块" in h_str:
                            v = row_data[c - 1] if c - 1 < len(row_data) else None
                            func_module = str(v).strip() if v else ""
                        elif "产品功能" in h_str:
                            v = row_data[c - 1] if c - 1 < len(row_data) else None
                            product_func = str(v).strip() if v else ""

                    if product_func:
                        price_anchors.append({
                            "anchor": biz_anchor,
                            "header_ctx": block.header_context_range,
                            "product": product_func,
                            "module": func_module,
                        })
            else:
                # 费率表语义块
                for anchor in block.allowed_anchor_ranges:
                    bounds = _parse_range_str(anchor)
                    if not bounds:
                        continue
                    min_col, min_row, max_col, max_row = bounds
                    row_count = max_row - min_row + 1
                    if row_count != 2:
                        continue
                    # 读取字段名和值
                    field_name = ""
                    value = ""
                    for r in range(min_row, max_row + 1):
                        for c in range(min_col, max_col + 1):
                            val = sheet.rows[r - 1][c - 1] if c - 1 < len(sheet.rows[r - 1]) else None
                            if val is None:
                                continue
                            val_str = str(val).strip()
                            if not val_str or val_str == "[公式未计算]":
                                continue
                            if _is_numeric_value(val):
                                value = val_str
                            else:
                                field_name = val_str
                    if field_name and value:
                        rate_anchors.append({
                            "anchor": anchor,
                            "field": field_name,
                            "value": value,
                        })

        # 文本锚点：从标准块中找功能归属
        for block in sheet.table_blocks:
            if block.block_index != 0:
                continue
            for anchor in block.allowed_anchor_ranges:
                bounds = _parse_range_str(anchor)
                if not bounds:
                    continue
                min_col, min_row, max_col, max_row = bounds
                if max_row != min_row:
                    continue
                # 检查是否包含文本列
                text_cols = []
                for c in range(min_col, max_col + 1):
                    val = sheet.rows[min_row - 1][c - 1] if c - 1 < len(sheet.rows[min_row - 1]) else None
                    if val and not _is_numeric_value(val) and str(val).strip():
                        text_cols.append(str(val).strip())
                if len(text_cols) >= 2:
                    text_anchors.append({
                        "anchor": anchor,
                        "content": " | ".join(text_cols[:3]),
                    })

    return {
        "rate_anchors": rate_anchors,
        "price_anchors": price_anchors,
        "text_anchors": text_anchors,
    }


# ─── Prompt 构建 ──────────────────────────────────────────────────────────────

def _build_prompt(sheets, num_questions, topic_hint="", candidate_anchors=None):
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
            header_ctx_info = ""
            if block.header_context_range:
                header_ctx_info = f"\nheader_context_range: {block.header_context_range}"
            block_text = (
                f"### 工作表: {sheet.sheet_name} — 表格块 {block.block_index + 1} "
                f"(行 {block.row_range[0]}-{block.row_range[1]})\n\n"
                f"{block.markdown}\n\n"
                f"allowed_anchor_ranges: {json.dumps(block.allowed_anchor_ranges, ensure_ascii=False)}"
                f"{header_ctx_info}"
                f"{formula_warn}"
            )
            block_texts.append(block_text)

    table_blocks_text = "\n\n---\n\n".join(block_texts)

    # 构建候选锚点清单
    candidate_text = ""
    if candidate_anchors:
        parts = []
        if candidate_anchors.get("rate_anchors"):
            lines = ["**角色费率锚点**（anchor_range 必须精确使用这些值）："]
            for a in candidate_anchors["rate_anchors"]:
                lines.append(f"- `{a['anchor']}` → {a['field']}：{a['value']}")
            parts.append("\n".join(lines))

        if candidate_anchors.get("price_anchors"):
            lines = ["**业务功能未税价锚点**（anchor_range 使用单行业务行，表头上下文自动附带）："]
            for a in candidate_anchors["price_anchors"]:
                lines.append(f"- `{a['anchor']}` → {a['product']}（{a.get('module', '')}）")
            parts.append("\n".join(lines))

        if candidate_anchors.get("text_anchors"):
            lines = ["**功能归属文本锚点**（可用于模块/功能名称类检索题）："]
            for a in candidate_anchors["text_anchors"][:20]:  # 限制显示数量
                lines.append(f"- `{a['anchor']}` → {a['content']}")
            parts.append("\n".join(lines))

        if parts:
            candidate_text = "\n\n---\n\n## 合法候选锚点清单\n\n" + "\n\n".join(parts)

    prompt = template.replace("{table_blocks_text}", table_blocks_text)
    prompt = prompt.replace("{num_questions}", str(num_questions))
    prompt = prompt.replace("{topic_hint_section}", topic_hint_section)
    prompt = prompt.replace("{candidate_anchors_section}", candidate_text)
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

    # 跨行范围收紧：
    # - 2 行：允许（子集检查即可，如字段名+数值、表头+业务行）
    # - 3+ 行：必须在白名单中明确出现（来自语义块）
    if rows >= 3 and anchor_range not in allowed_ranges:
        return False, (
            f"跨行范围 {anchor_range} 不在白名单中，"
            f"3行以上跨行锚点必须来自语义块"
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
    # 双源模型锚点也不例外——D4:D4 等孤立价格数值必须继续拒绝
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
    """将二维单元格值数组渲染为纯文本格式（不含 Markdown 分隔符）。

    单行：值1 | 值2 | 值3
    多行（首行是表头）：字段1：值1；字段2：值2；...
    """
    if not cell_values:
        return ""

    # 单行：值用 | 分隔
    if len(cell_values) == 1:
        parts = [str(v) for v in cell_values[0] if v is not None and str(v).strip()]
        return " | ".join(parts) if parts else ""

    # 多行：首行是表头，后续行是数据 → 键值对格式
    header = cell_values[0]
    lines = []
    for row in cell_values[1:]:
        pairs = []
        for i, val in enumerate(row):
            if val is None or not str(val).strip():
                continue
            h = header[i] if i < len(header) and header[i] is not None else f"列{i+1}"
            pairs.append(f"{str(h).strip()}：{str(val).strip()}")
        if pairs:
            lines.append("；".join(pairs))

    return "\n".join(lines) if lines else ""


def _render_dual_source_reference(anchor_range, header_context_range, sheet_ctx):
    """双源渲染：表头上下文 + 业务行数据。

    按列对齐：只渲染 anchor 范围内每列对应的字段名和值。
    例如 anchor=D4:D4, header_ctx=B2:M2 → 只渲染 D 列：未税价（元）：73900

    Returns (rendered_text, has_formula_issue).
    """
    # 解析表头上下文范围（如 B2:M2）
    h_bounds = _parse_range_str(header_context_range)
    if h_bounds is None:
        return "", True
    h_min_col, h_min_row, h_max_col, h_max_row = h_bounds

    # 解析业务行范围（如 D4:D4）
    b_bounds = _parse_range_str(anchor_range)
    if b_bounds is None:
        return "", True
    b_min_col, b_min_row, b_max_col, b_max_row = b_bounds

    has_formula_issue = False
    lines = []

    # 按列对齐：遍历 anchor 的每一列，找对应的表头字段名
    for c in range(b_min_col, b_max_col + 1):
        # 读取表头字段名（从 header_context_range 中对应列）
        if h_min_col <= c <= h_max_col:
            fname = sheet_ctx.rows[h_min_row - 1][c - 1] if c - 1 < len(sheet_ctx.rows[h_min_row - 1]) else None
            fname = str(fname).strip() if fname else ""
        else:
            fname = ""

        # 读取业务行值
        val = sheet_ctx.rows[b_min_row - 1][c - 1] if c - 1 < len(sheet_ctx.rows[b_min_row - 1]) else None
        if (b_min_row, c) in sheet_ctx.formula_cells_without_cache:
            has_formula_issue = True
            fval = "[公式未计算]"
        elif isinstance(val, str) and val == "[公式未计算]":
            has_formula_issue = True
            fval = val
        elif val is None:
            fval = ""
        else:
            fval = str(val)

        if fname and fval:
            lines.append(f"{fname}：{fval}")

    rendered = "\n".join(lines)
    if not rendered.strip():
        return "", True

    return rendered, has_formula_issue


# ─── 题目验证与渲染 ───────────────────────────────────────────────────────────

def _validate_and_render_question(raw_q, sheets_by_name, file_name):
    """验证 LLM 返回的单条题目，渲染金标准证据。

    支持双源模型：
    - 业务行锚点（如 B4:D4）：唯一业务事实来源
    - 表头上下文（如 B2:D2）：字段语义来源，从语义块元数据获取

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

    # 查找该锚点对应的语义块，获取 header_context_range（支持子集匹配）
    header_context_range = None
    anchor_bounds = _parse_range_str(anchor_range)
    for block in sheet.table_blocks:
        if not block.header_context_range:
            continue
        for allowed in block.allowed_anchor_ranges:
            if anchor_range == allowed:
                header_context_range = block.header_context_range
                break
            allowed_bounds = _parse_range_str(allowed)
            if anchor_bounds and allowed_bounds:
                a_min_col, a_min_row, a_max_col, a_max_row = anchor_bounds
                s_min_col, s_min_row, s_max_col, s_max_row = allowed_bounds
                if (a_min_col >= s_min_col and a_max_col <= s_max_col and
                        a_min_row >= s_min_row and a_max_row <= s_max_row):
                    header_context_range = block.header_context_range
                    break
        if header_context_range:
            break

    # 渲染金标准
    if header_context_range:
        # 双源渲染：表头上下文 + 业务行
        rendered, has_formula_issue = _render_dual_source_reference(
            anchor_range, header_context_range, sheet
        )
    else:
        # 单源渲染：仅业务行
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

    # 添加表头上下文（如有）
    if header_context_range:
        validated["header_context_range"] = header_context_range

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

def _categorize_rejection(reason):
    """将拒绝原因归类为统计类别。"""
    if not reason:
        return "unknown"
    r = reason.lower()
    if "json" in r or "解析" in r or "格式" in r:
        return "json_format"
    if "白名单" in r or "不在" in r:
        return "whitelist"
    if "聚合" in r:
        return "aggregate"
    if "不匹配" in r or "角色" in r:
        return "role_mismatch"
    if "孤立数值" in r:
        return "isolated_numeric"
    if "价格" in r or "价格字段" in r:
        return "price_field"
    if "跨行" in r:
        return "cross_row"
    if "空" in r or "为空" in r:
        return "empty"
    if "公式" in r:
        return "formula"
    if "重复" in r:
        return "duplicate"
    return "other"


def _validate_single_question(q, sheets_by_name, file_name,
                               semantic_field_names, semantic_anchors):
    """验证单条题目，返回 (validated_dict_or_None, rejection_category, reason)。"""
    validated, reason = _validate_and_render_question(q, sheets_by_name, file_name)
    if not validated:
        return None, _categorize_rejection(reason), reason

    consistent, consistency_reason = _validate_question_anchor_consistency(
        validated, semantic_field_names, semantic_anchors, sheets_by_name,
    )
    if not consistent:
        return None, _categorize_rejection(consistency_reason), consistency_reason

    validated["question_mode"] = MODE_RETRIEVAL
    return validated, None, None


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
        progress_callback(0, 5, "解析表格文件")

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

    semantic_field_names = _extract_semantic_field_names(sheets)
    semantic_anchors = _extract_semantic_anchors(sheets)

    if progress_callback:
        progress_callback(1, 5, "构建候选锚点和 LLM Prompt")

    # 2. 构建候选锚点清单
    candidate_anchors = _build_candidate_anchors(sheets)

    # 3. 构建 prompt（含候选锚点）
    prompt = _build_prompt(sheets, num_questions, topic_hint, candidate_anchors)

    if progress_callback:
        progress_callback(2, 5, "调用 LLM 生成检索查询")

    # 4. 首次 LLM 调用
    response_text = _call_llm_text(prompt, api_key, base_url, model, timeout=timeout)
    raw_questions = _parse_llm_response(response_text)
    first_raw_count = len(raw_questions)

    if progress_callback:
        progress_callback(3, 5, "验证并渲染金标准")

    # 5. 首次验证
    rejection_stats = {}
    valid_questions = []
    used_anchors = set()

    for q in raw_questions:
        validated, category, reason = _validate_single_question(
            q, sheets_by_name, file_name, semantic_field_names, semantic_anchors,
        )
        if validated:
            valid_questions.append(validated)
            used_anchors.add(validated.get("anchor_range", ""))
        else:
            rejection_stats[category] = rejection_stats.get(category, 0) + 1

    # 6. 首次去重
    unique_questions = deduplicate_questions(valid_questions)
    first_valid_count = len(unique_questions)

    # 7. 补充调用（如果有效题数 < 目标数且仍有未使用的候选锚点）
    supplement_count = 0
    supplement_valid = 0
    if len(unique_questions) < num_questions:
        remaining = num_questions - len(unique_questions)
        # 找出未使用的候选锚点
        unused_candidates = []
        for cat in ("rate_anchors", "price_anchors", "text_anchors"):
            for a in candidate_anchors.get(cat, []):
                if a["anchor"] not in used_anchors:
                    unused_candidates.append(a)

        if unused_candidates and remaining > 0:
            if progress_callback:
                progress_callback(4, 5, f"补充调用（还需 {remaining} 条）")

            # 构建补充 prompt
            used_anchors_str = ", ".join(f"`{a}`" for a in sorted(used_anchors))
            unused_str = "\n".join(
                f"- `{a['anchor']}` → {a.get('field', '') or a.get('product', '') or a.get('content', '')}"
                for a in unused_candidates[:remaining * 2]
            )
            supplement_prompt = (
                f"已生成的题目使用了以下锚点，不得重复：\n{used_anchors_str}\n\n"
                f"还需生成 {remaining} 条不重复的检索查询。"
                f"以下候选锚点尚未使用，请从中选择：\n{unused_str}\n\n"
                f"请严格输出 JSON 数组，每个元素包含 question, sheet_name, anchor_range, difficulty, topic。"
            )

            try:
                supp_response = _call_llm_text(supplement_prompt, api_key, base_url, model, timeout=timeout)
                supp_raw = _parse_llm_response(supp_response)
                supplement_count = len(supp_raw)

                for q in supp_raw:
                    if len(unique_questions) >= num_questions:
                        break
                    validated, category, reason = _validate_single_question(
                        q, sheets_by_name, file_name, semantic_field_names, semantic_anchors,
                    )
                    if validated and validated.get("anchor_range", "") not in used_anchors:
                        unique_questions.append(validated)
                        used_anchors.add(validated["anchor_range"])
                        supplement_valid += 1
                    elif not validated:
                        rejection_stats[category] = rejection_stats.get(category, 0) + 1
            except Exception as e:
                print(f"  ⚠️ 补充调用失败: {e}")

    # 8. 最终裁剪
    if len(unique_questions) > num_questions:
        unique_questions = unique_questions[:num_questions]

    if not unique_questions:
        raise ValueError("出题失败：所有查询均未通过锚定校验")

    stats = {
        "target": num_questions,
        "first_raw_count": first_raw_count,
        "first_valid_count": first_valid_count,
        "rejection_stats": rejection_stats,
        "supplement_count": supplement_count,
        "supplement_valid": supplement_valid,
        "final_count": len(unique_questions),
        "sheet_count": len(sheets),
        "block_count": total_blocks,
        "formula_warnings": total_formula_warnings,
        "format_warnings": format_warnings,
        "candidate_counts": {
            "rate_anchors": len(candidate_anchors.get("rate_anchors", [])),
            "price_anchors": len(candidate_anchors.get("price_anchors", [])),
            "text_anchors": len(candidate_anchors.get("text_anchors", [])),
        },
    }
    return unique_questions, stats
