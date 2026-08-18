"""
页面导航状态回归测试。

验证：
1. PAGES 包含 7 个页面
2. 默认页面为"知识库探索"
3. 每个页面有对应的 render_* 函数定义
4. 导航使用 _active_page 作为 session_state key
5. 调度代码基于 selected_page 条件渲染
6. 不再有 st.tabs 或 with tab_ 块
7. 材料入库的导入为惰性（在函数体内）
"""

import ast
import sys
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _read_source() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def _parse_ast() -> ast.Module:
    return ast.parse(_read_source())


# ── 测试导航常量 ─────────────────────────────────────────────


def test_pages_constant_has_seven_entries():
    """PAGES 列表应包含 7 个页面。"""
    tree = _parse_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PAGES":
                    if isinstance(node.value, ast.List):
                        assert len(node.value.elts) == 7
                        return
    pytest.fail("未找到 PAGES 常量定义")


def test_default_page_is_kb_explorer():
    """DEFAULT_PAGE 应为"知识库探索"。"""
    tree = _parse_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_PAGE":
                    if isinstance(node.value, ast.Constant):
                        assert node.value.value == "知识库探索"
                        return
    pytest.fail("未找到 DEFAULT_PAGE 常量定义")


def test_render_functions_defined():
    """每个页面应有对应的 render_* 函数定义。"""
    source = _read_source()
    expected_funcs = [
        "def render_kb_explorer():",
        "def render_question_gen():",
        "def render_batch_query():",
        "def render_sample_prep():",
        "def render_judge():",
        "def render_dashboard():",
        "def render_ingestion():",
    ]
    for func_def in expected_funcs:
        assert func_def in source, f"未找到 {func_def}"


# ── 测试导航结构 ────────────────────────────────────────────


def test_active_page_key():
    """导航应使用 _active_page 作为 session_state key。"""
    source = _read_source()
    assert 'key="_active_page"' in source


def test_dispatch_uses_selected_page():
    """调度代码应基于 selected_page 条件渲染。"""
    source = _read_source()
    assert 'if selected_page == "知识库探索"' in source
    assert 'elif selected_page == "材料入库"' in source


def test_no_st_tabs_remaining():
    """不应再有 st.tabs 调用。"""
    source = _read_source()
    assert "st.tabs([" not in source


def test_no_with_tab_remaining():
    """不应再有 with tab_x: 块。"""
    source = _read_source()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("with tab_") and stripped.endswith(":"):
            if "#" not in stripped.split("with")[0]:
                pytest.fail(f"发现残留的 with tab_ 块: {stripped}")


def test_segmented_control_used():
    """应使用 st.segmented_control 替代 st.tabs。"""
    source = _read_source()
    assert "st.segmented_control(" in source


# ── 测试惰性导入 ────────────────────────────────────────────


def test_ingestion_imports_lazy():
    """材料入库的 dify_ingestion 导入应在函数体内，不在模块顶层。"""
    source = _read_source()
    lines = source.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def render_ingestion"):
            in_function = True
        if "from dify_ingestion import" in stripped:
            assert in_function, (
                "from dify_ingestion import 不应在模块顶层，"
                "应在 render_ingestion() 函数体内"
            )
            return
    # dify_ingestion 在整个文件中不存在也是可接受的
