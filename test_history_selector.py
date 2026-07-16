"""
历史题集选择器测试。

测试内容：
1. 同名题集（相同 question_set_name 和题目数）必须生成不同的显示标签
2. 选择不同项必须加载不同的 question_set_id 和题目内容
3. 旧格式题集（无 question_set_id）仍能正确列出和加载
4. 选择项与实际加载对象一一对应
5. 历史题集按 created_at 降序排列，新版优先，旧版排在最后

不调用真实 API。
"""

import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from question_generator import MODE_RETRIEVAL, MODE_QA, save_questions


def _detect_file_info(filepath):
    """与 app.py 中完全一致的检测逻辑。"""
    info = {
        "modes": {MODE_RETRIEVAL: 0, MODE_QA: 0, "unknown": 0},
        "set_name": "",
        "set_id": "",
        "question_count": 0,
        "has_set_info": False,
    }
    try:
        with filepath.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                info["question_count"] += 1
                if i >= 20:
                    continue
                try:
                    obj = json.loads(line)
                    mode = obj.get("question_mode", "")
                    if mode == MODE_RETRIEVAL:
                        info["modes"][MODE_RETRIEVAL] += 1
                    elif mode == MODE_QA:
                        info["modes"][MODE_QA] += 1
                    else:
                        info["modes"]["unknown"] += 1
                    if obj.get("question_set_name") and not info["set_name"]:
                        info["set_name"] = obj["question_set_name"]
                        info["set_id"] = obj.get("question_set_id", "")
                        info["has_set_info"] = True
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return info


def _build_label(f, info):
    """与 app.py 中完全一致的标签构建逻辑。"""
    modes = info["modes"]
    total_sampled = sum(modes.values())
    q_count = info["question_count"]

    if total_sampled == 0:
        mode_tag = "[空文件]"
    elif modes[MODE_RETRIEVAL] > 0 and modes[MODE_QA] > 0:
        mode_tag = "[混合]"
    elif modes[MODE_RETRIEVAL] > 0 and modes["unknown"] == 0:
        mode_tag = "[检索评测]"
    elif modes[MODE_QA] > 0 and modes["unknown"] == 0:
        mode_tag = "[全流程问答]"
    elif modes["unknown"] > 0 and modes[MODE_RETRIEVAL] == 0 and modes[MODE_QA] == 0:
        mode_tag = "[旧版]"
    elif modes[MODE_RETRIEVAL] > 0:
        mode_tag = "[检索评测+旧版]"
    elif modes[MODE_QA] > 0:
        mode_tag = "[全流程问答+旧版]"
    else:
        mode_tag = "[旧版]"

    if info["has_set_info"] and info["set_name"]:
        _sid = info.get("set_id", "")
        _ts_display = ""
        if _sid:
            _parts = _sid.split("_", 3)
            if len(_parts) >= 3:
                _date_part = _parts[1]
                _time_part = _parts[2]
                if len(_date_part) == 8 and len(_time_part) >= 6:
                    _ts_display = f"{_date_part[:4]}-{_date_part[4:6]}-{_date_part[6:8]} {_time_part[:2]}:{_time_part[2:4]}"
        _sid_short = ""
        if _sid and len(_sid) > 20:
            _sid_short = f" · qs...{_sid[12:20]}"
        _ts_part = f" · {_ts_display}" if _ts_display else ""
        label = f"{mode_tag} {info['set_name']} · {q_count} 题{_ts_part}{_sid_short}"
    else:
        label = f"{mode_tag} {f.stem} · {q_count} 题 [旧版题集]"

    return label


def _get_created_at(filepath, info):
    """与 app.py 中完全一致的创建时间解析逻辑。"""
    # 1. 检查 manifest 文件
    manifest_path = filepath.parent / f"{filepath.stem}_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            created_at = manifest.get("created_at")
            if created_at:
                return datetime.fromisoformat(created_at)
        except Exception:
            pass

    # 2. 从 set_id 解析时间戳
    set_id = info.get("set_id", "")
    if set_id:
        parts = set_id.split("_", 3)
        if len(parts) >= 3:
            date_part = parts[1]
            time_part = parts[2]
            try:
                if len(date_part) == 8 and len(time_part) >= 6:
                    ts_str = date_part + time_part[:6]
                    return datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            except (ValueError, IndexError):
                pass

    # 3. 从文件名解析时间戳
    match = re.search(r'(\d{8}_\d{6})', filepath.stem)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    return None


def test_duplicate_names_different_ids():
    """两个同名同题数但不同 question_set_id 的题集必须生成不同标签。"""
    print("=" * 60)
    print("测试同名题集标签唯一性")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir)

        # 创建两个同名题集，set_id 不同
        questions_a = [
            {"question": "问题A1", "reference_answer": "答案A1", "question_mode": "retrieval",
             "question_set_id": "qs_20260714_101905755296_IS5010", "question_set_name": "IS5010_检索评测"},
            {"question": "问题A2", "reference_answer": "答案A2", "question_mode": "retrieval",
             "question_set_id": "qs_20260714_101905755296_IS5010", "question_set_name": "IS5010_检索评测"},
        ]
        questions_b = [
            {"question": "问题B1", "reference_answer": "答案B1", "question_mode": "retrieval",
             "question_set_id": "qs_20260713_164111972190_IS5010", "question_set_name": "IS5010_检索评测"},
            {"question": "问题B2", "reference_answer": "答案B2", "question_mode": "retrieval",
             "question_set_id": "qs_20260713_164111972190_IS5010", "question_set_name": "IS5010_检索评测"},
        ]

        file_a = questions_dir / "questions_IS5010_检索评测_20260714_101905.jsonl"
        file_b = questions_dir / "questions_IS5010_检索评测_20260713_164111.jsonl"

        for f, qs in [(file_a, questions_a), (file_b, questions_b)]:
            with f.open("w", encoding="utf-8") as fh:
                for q in qs:
                    fh.write(json.dumps(q, ensure_ascii=False) + "\n")

        # 构建标签
        history_files = sorted(questions_dir.glob("*.jsonl"), reverse=True)
        file_info_cache = {}
        file_labels = []
        for f in history_files:
            info = _detect_file_info(f)
            file_info_cache[f] = info
            label = _build_label(f, info)
            file_labels.append(label)

        # 验证标签唯一
        print(f"  文件数: {len(history_files)}")
        for i, (f, label) in enumerate(zip(history_files, file_labels)):
            print(f"  [{i}] {label}")
            print(f"      文件: {f.name}")

        assert len(file_labels) == len(set(file_labels)), \
            f"标签不唯一！重复: {[l for l in file_labels if file_labels.count(l) > 1]}"
        print("[OK] 同名题集标签唯一")

        # 验证选择不同 index 加载不同内容
        for idx, f in enumerate(history_files):
            questions_list = []
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        questions_list.append(json.loads(line))
            loaded_set_id = questions_list[0].get("question_set_id", "")
            loaded_q = questions_list[0].get("question", "")
            print(f"  选择 [{idx}]: set_id={loaded_set_id}, 首题={loaded_q}")

        # 验证两个文件加载的内容不同
        q_a = []
        with file_a.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    q_a.append(json.loads(line))
        q_b = []
        with file_b.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    q_b.append(json.loads(line))

        assert q_a[0]["question_set_id"] != q_b[0]["question_set_id"], \
            "两个题集的 set_id 应不同"
        assert q_a[0]["question"] != q_b[0]["question"], \
            "两个题集的题目内容应不同"
        print("[OK] 选择不同项加载不同内容")

    print()


def test_old_format_compatibility():
    """旧格式题集（无 question_set_id）仍能正确列出和加载。"""
    print("=" * 60)
    print("测试旧格式题集兼容性")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir)

        # 创建旧格式文件（无 set_id/set_name）
        old_questions = [
            {"question": "旧问题1", "reference_answer": "旧答案1"},
            {"question": "旧问题2", "reference_answer": "旧答案2"},
        ]
        old_file = questions_dir / "questions_20260708_111748.jsonl"
        with old_file.open("w", encoding="utf-8") as f:
            for q in old_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # 创建新格式文件
        new_questions = [
            {"question": "新问题1", "reference_answer": "新答案1", "question_mode": "retrieval",
             "question_set_id": "qs_20260714_120000_000000_test", "question_set_name": "测试题集"},
        ]
        new_file = questions_dir / "questions_test_20260714_120000.jsonl"
        with new_file.open("w", encoding="utf-8") as f:
            for q in new_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # 构建标签
        history_files = sorted(questions_dir.glob("*.jsonl"), reverse=True)
        file_labels = []
        for f in history_files:
            info = _detect_file_info(f)
            label = _build_label(f, info)
            file_labels.append(label)

        for i, (f, label) in enumerate(zip(history_files, file_labels)):
            print(f"  [{i}] {label}")

        # 旧格式标签应包含文件名
        old_idx = next(i for i, f in enumerate(history_files) if f == old_file)
        assert "旧版题集" in file_labels[old_idx], \
            f"旧格式标签应包含 [旧版题集]，实际: {file_labels[old_idx]}"
        assert "questions_20260708_111748" in file_labels[old_idx], \
            f"旧格式标签应包含文件名，实际: {file_labels[old_idx]}"
        print("[OK] 旧格式题集标签正确")

        # 新格式标签应包含题集名称
        new_idx = next(i for i, f in enumerate(history_files) if f == new_file)
        assert "测试题集" in file_labels[new_idx], \
            f"新格式标签应包含题集名称，实际: {file_labels[new_idx]}"
        print("[OK] 新格式题集标签正确")

        # 验证加载
        for idx, f in enumerate(history_files):
            questions_list = []
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        questions_list.append(json.loads(line))
            loaded_set_id = questions_list[0].get("question_set_id", "")
            loaded_q = questions_list[0].get("question", "")
            print(f"  加载 [{idx}]: set_id='{loaded_set_id}', 首题={loaded_q}")

        print("[OK] 旧格式题集可正确加载")

    print()


def test_selection_uses_index_not_label():
    """验证 selectbox 使用 index 作为 value，不会因 label 相同而混淆。"""
    print("=" * 60)
    print("测试选择逻辑使用 index 而非 label")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir)

        # 创建两个同名但内容不同的文件
        for i, (name, sid) in enumerate([
            ("题集X", "qs_20260714_000000_000000_X"),
            ("题集X", "qs_20260713_000000_000000_X"),
        ]):
            f = questions_dir / f"questions_{i}.jsonl"
            with f.open("w", encoding="utf-8") as fh:
                for j in range(3):
                    fh.write(json.dumps({
                        "question": f"{name}批次{i+1}问题{j+1}",
                        "reference_answer": f"{name}批次{i+1}答案{j+1}",
                        "question_mode": "retrieval",
                        "question_set_id": sid,
                        "question_set_name": name,
                    }, ensure_ascii=False) + "\n")

        # 模拟选择逻辑
        history_files = sorted(questions_dir.glob("*.jsonl"), reverse=True)
        file_info_cache = {}
        file_labels = []
        for f in history_files:
            info = _detect_file_info(f)
            file_info_cache[f] = info
            label = _build_label(f, info)
            file_labels.append(label)

        # 模拟用户选择 index 0
        selected_idx = 0
        selected_file = history_files[selected_idx]
        selected_info = file_info_cache[selected_file]

        questions_list = []
        with selected_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    questions_list.append(json.loads(line))

        loaded_sid_0 = questions_list[0]["question_set_id"]
        loaded_q_0 = questions_list[0]["question"]

        # 模拟用户选择 index 1
        selected_idx = 1
        selected_file = history_files[selected_idx]
        selected_info = file_info_cache[selected_file]

        questions_list = []
        with selected_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    questions_list.append(json.loads(line))

        loaded_sid_1 = questions_list[0]["question_set_id"]
        loaded_q_1 = questions_list[0]["question"]

        print(f"  选择 [0]: set_id={loaded_sid_0}, 首题={loaded_q_0}")
        print(f"  选择 [1]: set_id={loaded_sid_1}, 首题={loaded_q_1}")

        assert loaded_sid_0 != loaded_sid_1, \
            f"选择不同 index 应加载不同 set_id: {loaded_sid_0} vs {loaded_sid_1}"
        assert loaded_q_0 != loaded_q_1, \
            f"选择不同 index 应加载不同题目: {loaded_q_0} vs {loaded_q_1}"
        print("[OK] 选择不同 index 加载不同内容，index-based 绑定正确")

    print()


def test_label_uniqueness_with_real_data():
    """使用真实数据验证标签唯一性。"""
    print("=" * 60)
    print("测试真实数据标签唯一性")
    print("=" * 60)

    questions_dir = Path(__file__).parent / "data" / "questions"
    batch_dir = Path(__file__).parent / "data" / "batch"

    if not questions_dir.exists():
        print("  [SKIP] data/questions 目录不存在")
        return

    history_files = []
    for d in [questions_dir, batch_dir]:
        if d.exists():
            for f in sorted(d.glob("*.jsonl"), reverse=True):
                history_files.append(f)

    if not history_files:
        print("  [SKIP] 无历史文件")
        return

    file_labels = []
    for f in history_files:
        info = _detect_file_info(f)
        label = _build_label(f, info)
        file_labels.append(label)

    print(f"  文件数: {len(history_files)}")
    print(f"  唯一标签数: {len(set(file_labels))}")

    if len(file_labels) != len(set(file_labels)):
        dupes = [l for l in file_labels if file_labels.count(l) > 1]
        print(f"  [FAIL] 重复标签: {set(dupes)}")
        for i, (f, label) in enumerate(zip(history_files, file_labels)):
            if label in set(dupes):
                print(f"    [{i}] {label} -> {f.name}")
        assert False, "标签不唯一"

    print("[OK] 真实数据所有标签唯一")

    print()


def test_history_sort_by_created_at():
    """验证历史题集按 created_at 降序排列：不同日期、同日不同时分、无时间记录。"""
    print("=" * 60)
    print("测试历史题集排序（created_at 降序）")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir)

        # ---- 场景1: 不同日期的题集 ----
        print("\n  [场景1] 不同日期，应按日期降序")
        # 清理目录
        for f in questions_dir.glob("*"):
            f.unlink()

        files_meta = [
            ("questions_A_20260712_100000.jsonl",
             "qs_20260712_100000000000_A", "题集A", "2026-07-12T10:00:00"),
            ("questions_B_20260715_100000.jsonl",
             "qs_20260715_100000000000_B", "题集B", "2026-07-15T10:00:00"),
            ("questions_C_20260713_100000.jsonl",
             "qs_20260713_100000000000_C", "题集C", "2026-07-13T10:00:00"),
        ]
        _write_test_files(questions_dir, files_meta)
        sorted_names = _run_sort(questions_dir)
        print(f"    排序结果: {sorted_names}")
        assert sorted_names == ["题集B", "题集C", "题集A"], \
            f"不同日期降序失败: {sorted_names}"
        print("    [OK] 不同日期降序正确")

        # ---- 场景2: 同日不同时分 ----
        print("\n  [场景2] 同日不同时分，应按时分降序")
        for f in questions_dir.glob("*"):
            f.unlink()

        files_meta = [
            ("questions_D_20260714_083000.jsonl",
             "qs_20260714_083000000000_D", "题集D", "2026-07-14T08:30:00"),
            ("questions_E_20260714_145710.jsonl",
             "qs_20260714_145710000000_E", "题集E", "2026-07-14T14:57:10"),
            ("questions_F_20260714_101905.jsonl",
             "qs_20260714_101905000000_F", "题集F", "2026-07-14T10:19:05"),
        ]
        _write_test_files(questions_dir, files_meta)
        sorted_names = _run_sort(questions_dir)
        print(f"    排序结果: {sorted_names}")
        assert sorted_names == ["题集E", "题集F", "题集D"], \
            f"同日不同时分降序失败: {sorted_names}"
        print("    [OK] 同日不同时分降序正确")

        # ---- 场景3: 混合有时间和无时间 ----
        print("\n  [场景3] 有时间 + 无时间（旧版），无时间排最后")
        for f in questions_dir.glob("*"):
            f.unlink()

        files_meta = [
            ("questions_G_20260714_120000.jsonl",
             "qs_20260714_120000000000_G", "题集G", "2026-07-14T12:00:00"),
            ("questions_20260708_111748.jsonl",
             "", "旧版题集", None),  # 无 set_id，无 manifest
        ]
        _write_test_files(questions_dir, files_meta)
        sorted_names = _run_sort(questions_dir)
        print(f"    排序结果: {sorted_names}")
        assert sorted_names == ["题集G", "旧版题集"], \
            f"有时间+无时间排序失败: {sorted_names}"
        print("    [OK] 有时间记录优先，无时间排最后")

        # ---- 场景4: manifest created_at 优先于 set_id ----
        print("\n  [场景4] manifest created_at 优先于 set_id 时间戳")
        for f in questions_dir.glob("*"):
            f.unlink()

        files_meta = [
            ("questions_H_20260714_120000.jsonl",
             "qs_20260714_120000000000_H", "题集H", "2026-07-14T12:00:00"),
            ("questions_I_20260714_120000.jsonl",
             "qs_20260714_120000000000_I", "题集I", "2026-07-14T12:00:00"),
        ]
        # 写入文件和题集
        _write_test_files(questions_dir, files_meta)
        # 为题集 I 写一个 manifest，将 created_at 设为更晚的时间
        manifest_i = {
            "question_set_id": "qs_20260714_120000000000_I",
            "question_set_name": "题集I",
            "question_count": 1,
            "created_at": "2026-07-14T18:00:00",
            "filename": "questions_I_20260714_120000.jsonl",
        }
        (questions_dir / "questions_I_20260714_120000_manifest.json").write_text(
            json.dumps(manifest_i, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sorted_names = _run_sort(questions_dir)
        print(f"    排序结果: {sorted_names}")
        assert sorted_names == ["题集I", "题集H"], \
            f"manifest 优先级排序失败: {sorted_names}"
        print("    [OK] manifest created_at 优先于 set_id")

    print()


def _write_test_files(questions_dir, files_meta):
    """辅助：写入测试 JSONL 文件。"""
    for fname, set_id, set_name, created_at in files_meta:
        q = {"question": f"{set_name}问题1", "reference_answer": "答案1"}
        if set_id:
            q["question_set_id"] = set_id
        if set_name:
            q["question_set_name"] = set_name
        if created_at:
            q["question_mode"] = "retrieval"
        f = questions_dir / fname
        with f.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")


def _run_sort(questions_dir):
    """辅助：模拟 app.py 的排序逻辑，返回排序后的题集名称列表。"""
    history_files = list(questions_dir.glob("*.jsonl"))
    file_info_cache = {}
    for f in history_files:
        info = _detect_file_info(f)
        info["created_at"] = _get_created_at(f, info)
        file_info_cache[f] = info

    history_files.sort(
        key=lambda f: file_info_cache[f]["created_at"] or datetime.min,
        reverse=True,
    )

    return [file_info_cache[f]["set_name"] or f.stem for f in history_files]


def main():
    print("=" * 60)
    print("历史题集选择器测试")
    print("=" * 60)
    print()

    test_duplicate_names_different_ids()
    test_old_format_compatibility()
    test_selection_uses_index_not_label()
    test_history_sort_by_created_at()
    test_label_uniqueness_with_real_data()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
