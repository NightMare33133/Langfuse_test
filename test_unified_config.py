"""
统一配置 schema 测试。

测试内容：
1. 新建配置后 JSON 同时含有编辑页所需的全部字段
2. 旧格式配置可正常加载、显示和编辑
3. 批量提问创建的配置，在实验版本编辑页字段和值完全一致
4. 在实验版本编辑后的配置，在批量提问历史配置摘要中同步体现
5. 核心字段不可由任意表单修改
6. 不调用任何外部 API
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from experiment import (
    CONFIG_FIELD_SCHEMA, CONFIG_CORE_FIELDS, CONFIG_EDITABLE_FIELDS,
    CONFIG_REQUIRED_FIELDS, CONFIG_OPTIONAL_FIELDS,
    create_config_profile, load_config_profile,
    update_config_profile_safe,
    get_config_summary, get_config_display_value,
)


def test_schema_completeness():
    """Schema 包含所有可编辑字段。"""
    print("=" * 60)
    print("测试 Schema 完整性")
    print("=" * 60)

    schema_keys = {f[0] for f in CONFIG_FIELD_SCHEMA}
    assert schema_keys == CONFIG_EDITABLE_FIELDS, \
        f"Schema 字段与 CONFIG_EDITABLE_FIELDS 不一致: {schema_keys ^ CONFIG_EDITABLE_FIELDS}"

    # 必填字段在 schema 中标记为 required
    for f in CONFIG_FIELD_SCHEMA:
        key, label, required, widget, placeholder, help_text = f
        if key in CONFIG_REQUIRED_FIELDS:
            assert required, f"{key} 应为必填"
        else:
            assert not required, f"{key} 应为可选"

    print(f"  Schema 字段数: {len(CONFIG_FIELD_SCHEMA)}")
    print(f"  必填: {len(CONFIG_REQUIRED_FIELDS)} - {CONFIG_REQUIRED_FIELDS}")
    print(f"  可选: {len(CONFIG_OPTIONAL_FIELDS)}")
    print("[OK] Schema 完整性正确")

    print()


def test_create_with_all_fields():
    """新建配置后 JSON 含有所有可编辑字段。"""
    print("=" * 60)
    print("测试新建配置含所有字段")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                workflow_version="wf_v1",
                source_description="IS5010 期末复习",
                chunk_strategy="按章节切分",
                embedding_model="bge-large-zh",
                retrieval_mode="hybrid",
                retrieval_config="top_k=5",
                top_k=5,
                rerank_model="bge-reranker",
                changed_variable="chunk_size 优化",
                notes="测试备注",
            )

            # 验证 JSON 包含所有字段
            loaded = load_config_profile(config["config_id"])
            assert loaded is not None

            for key, label, required, widget, placeholder, help_text in CONFIG_FIELD_SCHEMA:
                assert key in loaded, f"缺少字段: {key} ({label})"
                if required:
                    assert loaded[key], f"必填字段 {key} 为空"

            # 验证值
            assert loaded["config_name"] == "测试配置"
            assert loaded["knowledge_base_version"] == "v1"
            assert loaded["workflow_version"] == "wf_v1"
            assert loaded["top_k"] == 5
            assert loaded["rerank_model"] == "bge-reranker"
            assert loaded["embedding_model"] == "bge-large-zh"
            assert loaded["retrieval_mode"] == "hybrid"
            assert loaded["source_description"] == "IS5010 期末复习"
            print("[OK] 新建配置 JSON 包含所有字段且值正确")

    print()


def test_old_config_loads():
    """旧格式配置可正常加载和显示。"""
    print("=" * 60)
    print("测试旧格式配置兼容")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        config_dir.mkdir()

        # 模拟旧配置（只有老字段）
        old_config = {
            "config_id": "cfg_old",
            "config_name": "旧配置",
            "knowledge_base_version": "old_kb",
            "workflow_version": "",
            "changed_variable": "",
            "retrieval_config": "",
            "notes": "",
            "created_at": "2026-07-08T00:00:00",
        }
        (config_dir / "cfg_old.json").write_text(
            json.dumps(old_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            loaded = load_config_profile("cfg_old")
            assert loaded is not None

            # 缺失的新字段显示为"未记录"
            assert get_config_display_value(loaded, "retrieval_mode") == "未记录"
            assert get_config_display_value(loaded, "top_k") == "未记录"
            assert get_config_display_value(loaded, "embedding_model") == "未记录"
            assert get_config_display_value(loaded, "rerank_model") == "未记录"
            assert get_config_display_value(loaded, "chunk_strategy") == "未记录"

            # 有值字段正常显示
            assert get_config_display_value(loaded, "config_name") == "旧配置"
            assert get_config_display_value(loaded, "knowledge_base_version") == "old_kb"

            # 编辑旧配置（补录新字段）
            updated = update_config_profile_safe("cfg_old", {
                "top_k": 5,
                "rerank_model": "bge-reranker",
                "embedding_model": "bge-large-zh",
            }, edit_note="补录配置")

            assert updated["config_id"] == "cfg_old", "config_id 不应变"
            assert updated["top_k"] == 5
            assert updated["rerank_model"] == "bge-reranker"
            assert updated["embedding_model"] == "bge-large-zh"
            # 旧字段保留
            assert updated["config_name"] == "旧配置"
            assert updated["knowledge_base_version"] == "old_kb"
            print("[OK] 旧配置可加载、显示、编辑新字段")

    print()


def test_consistency_between_tabs():
    """批量提问创建的配置与实验版本编辑页字段一致。"""
    print("=" * 60)
    print("测试跨 Tab 字段一致性")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            # 模拟批量提问创建（使用统一 schema 的字段）
            batch_values = {
                "config_name": "批量创建配置",
                "knowledge_base_version": "kb_v2",
                "workflow_version": "wf_v2",
                "source_description": "批量来源",
                "chunk_strategy": "512 tokens",
                "embedding_model": "bge-large",
                "retrieval_mode": "hybrid",
                "retrieval_config": "top_k=3",
                "top_k": 3,
                "rerank_model": "bge-reranker",
                "changed_variable": "测试改动",
                "notes": "批量备注",
            }
            # collect_config_updates 过滤空值
            filtered = {k: v for k, v in batch_values.items()
                        if (isinstance(v, (int, float)) and v > 0) or (isinstance(v, str) and v.strip())}
            config = create_config_profile(**filtered)
            loaded = load_config_profile(config["config_id"])

            # 模拟实验版本编辑页读取（同一字段、同一值）
            for key, label, required, widget, placeholder, help_text in CONFIG_FIELD_SCHEMA:
                display_val = get_config_display_value(loaded, key)
                expected = batch_values.get(key)
                if expected is None or str(expected).strip() == "":
                    assert display_val == "未记录", f"{key}: 期望'未记录', 实际'{display_val}'"
                else:
                    assert display_val == str(expected), \
                        f"{key}: 期望'{expected}', 实际'{display_val}'"

            print("[OK] 批量创建的配置在编辑页显示一致")

            # 模拟实验版本编辑（更新部分字段）
            updated = update_config_profile_safe(config["config_id"], {
                "top_k": 10,
                "notes": "编辑后备注",
            }, edit_note="修改 Top K")

            # 模拟批量提问"使用历史配置"读取
            summary = get_config_summary(updated)
            assert "批量创建配置" in summary
            assert "kb_v2" in summary
            assert "top_k=10" in summary
            assert updated["notes"] == "编辑后备注"
            # 未修改字段保留
            assert updated["retrieval_mode"] == "hybrid"
            assert updated["rerank_model"] == "bge-reranker"
            print("[OK] 编辑后的配置在历史摘要中同步体现")

    print()


def test_core_fields_protected():
    """核心字段不可由任意表单修改。"""
    print("=" * 60)
    print("测试核心字段保护")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", config_dir):
            config = create_config_profile("测试", "v1", "wf1")
            original_id = config["config_id"]
            original_created = config["created_at"]

            # 尝试通过 update_config_profile_safe 修改核心字段
            updated = update_config_profile_safe(original_id, {
                "config_id": "hacked",
                "created_at": "2099-01-01",
                "config_name": "合法修改",
            })

            assert updated["config_id"] == original_id
            assert updated["created_at"] == original_created
            assert updated["config_name"] == "合法修改"
            print("[OK] 核心字段不可修改，可编辑字段正常更新")

    print()


def test_summary_generation():
    """配置摘要生成正确。"""
    print("=" * 60)
    print("测试摘要生成")
    print("=" * 60)

    config = {
        "config_name": "测试配置",
        "knowledge_base_version": "v1",
        "retrieval_mode": "hybrid",
        "top_k": 5,
        "rerank_model": "bge-reranker",
    }
    summary = get_config_summary(config)
    assert "测试配置" in summary
    assert "v1" in summary
    assert "hybrid" in summary
    assert "top_k=5" in summary
    assert "rerank=bge-reranker" in summary
    print(f"  摘要: {summary}")
    print("[OK] 摘要生成正确")

    # 无额外字段时
    config_minimal = {"config_name": "最小配置", "knowledge_base_version": "v2"}
    summary_min = get_config_summary(config_minimal)
    assert "最小配置" in summary_min
    assert "v2" in summary_min
    print(f"  最小摘要: {summary_min}")
    print("[OK] 最小配置摘要正确")

    print()


def main():
    print("=" * 60)
    print("统一配置 Schema 测试")
    print("=" * 60)
    print()

    test_schema_completeness()
    test_create_with_all_fields()
    test_old_config_loads()
    test_consistency_between_tabs()
    test_core_fields_protected()
    test_summary_generation()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
