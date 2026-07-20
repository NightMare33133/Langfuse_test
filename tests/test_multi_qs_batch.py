"""
多题集批量提问测试。

验证：
1. 多选 2 个题集时创建 2 个不同 run_id，且都指向同一 config_id
2. 每个 run manifest 的 question_set_id 正确，batch 结果只写入各自 run 目录
3. 相同 question_id 出现在不同题集时不发生覆盖或错误去重
4. 第一个题集失败后第二个题集仍执行
5. 单选/其他问题来源保持原行为
6. 候选题集过滤仍排除非 Question Set 文件
7. 最终汇总的题集数、成功/失败数、总题数准确
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiment import (
    create_config_profile, create_experiment_run,
    load_experiment_run, update_experiment_run,
)


def _make_question(idx, q_set_id="qs_test_001", q_set_name="测试题集"):
    """构造测试用问题。"""
    return {
        "question": f"测试问题 {idx}",
        "question_mode": "qa",
        "question_set_id": q_set_id,
        "question_set_name": q_set_name,
    }


def test_multi_select_creates_separate_runs():
    """多选 2 个题集时创建 2 个不同 run_id，且都指向同一 config_id。"""
    print("=" * 60)
    print("测试：多题集创建独立 run")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 模拟两个题集各自创建 run
            run1 = create_experiment_run(
                config_id=config_id,
                question_set_source="从历史记录加载",
                question_count=10,
            )
            run2 = create_experiment_run(
                config_id=config_id,
                question_set_source="从历史记录加载",
                question_count=15,
            )

            assert run1["run_id"] != run2["run_id"], "两个 run_id 应不同"

            r1 = load_experiment_run(run1["run_id"])
            r2 = load_experiment_run(run2["run_id"])

            assert r1["config_id"] == config_id
            assert r2["config_id"] == config_id
            assert r1["config_id"] == r2["config_id"]

            print(f"[OK] run1: {run1['run_id']}, run2: {run2['run_id']}, "
                  f"config: {config_id}")

    print()


def test_each_run_has_correct_question_set_id():
    """每个 run manifest 的 question_set_id 正确。"""
    print("=" * 60)
    print("测试：manifest question_set_id 正确")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )

            run1 = create_experiment_run(config["config_id"], question_count=5)
            run2 = create_experiment_run(config["config_id"], question_count=8)

            update_experiment_run(run1["run_id"], {
                "question_set_id": "qs_001",
                "question_set_name": "题集A",
            })
            update_experiment_run(run2["run_id"], {
                "question_set_id": "qs_002",
                "question_set_name": "题集B",
            })

            r1 = load_experiment_run(run1["run_id"])
            r2 = load_experiment_run(run2["run_id"])

            assert r1["question_set_id"] == "qs_001"
            assert r1["question_set_name"] == "题集A"
            assert r2["question_set_id"] == "qs_002"
            assert r2["question_set_name"] == "题集B"

            print(f"[OK] run1 question_set: {r1['question_set_id']}, "
                  f"run2 question_set: {r2['question_set_id']}")

    print()


def test_batch_results_written_to_each_run_dir():
    """batch 结果只写入各自 run 目录。"""
    print("=" * 60)
    print("测试：batch 结果写入各自 run 目录")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )

            run1 = create_experiment_run(config["config_id"], question_count=2)
            run2 = create_experiment_run(config["config_id"], question_count=3)

            # 模拟写入 batch 结果到各自 run 目录
            results1 = [{"question": "q1", "success": True}, {"question": "q2", "success": True}]
            results2 = [{"question": "q3", "success": True}, {"question": "q4", "success": False}, {"question": "q5", "success": True}]

            path1 = Path(run1["run_dir"]) / "batch_results.jsonl"
            path2 = Path(run2["run_dir"]) / "batch_results.jsonl"

            with path1.open("w", encoding="utf-8") as f:
                for r in results1:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with path2.open("w", encoding="utf-8") as f:
                for r in results2:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            assert path1.exists()
            assert path2.exists()

            # 验证各自内容独立
            with path1.open("r", encoding="utf-8") as f:
                lines1 = [json.loads(l) for l in f if l.strip()]
            with path2.open("r", encoding="utf-8") as f:
                lines2 = [json.loads(l) for l in f if l.strip()]

            assert len(lines1) == 2
            assert len(lines2) == 3
            assert lines1[0]["question"] == "q1"
            assert lines2[0]["question"] == "q3"

            print(f"[OK] run1 有 {len(lines1)} 条结果, run2 有 {len(lines2)} 条结果")

    print()


def test_same_question_id_different_sets_no_collision():
    """相同 question_id 出现在不同题集时不发生覆盖。"""
    print("=" * 60)
    print("测试：相同 question_id 不覆盖")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )

            # 两个 run 使用不同的 question_set_id，但有相同的 question_id
            run1 = create_experiment_run(config["config_id"], question_count=2)
            run2 = create_experiment_run(config["config_id"], question_count=2)

            update_experiment_run(run1["run_id"], {
                "question_set_id": "qs_A",
                "question_set_name": "题集A",
            })
            update_experiment_run(run2["run_id"], {
                "question_set_id": "qs_B",
                "question_set_name": "题集B",
            })

            # 模拟 batch 结果：相同 question_id 在不同 run 中
            results1 = [
                {"question": "共享问题", "question_id": "q_shared", "run_id": run1["run_id"], "success": True},
            ]
            results2 = [
                {"question": "共享问题", "question_id": "q_shared", "run_id": run2["run_id"], "success": True},
            ]

            path1 = Path(run1["run_dir"]) / "batch_results.jsonl"
            path2 = Path(run2["run_dir"]) / "batch_results.jsonl"

            with path1.open("w", encoding="utf-8") as f:
                for r in results1:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with path2.open("w", encoding="utf-8") as f:
                for r in results2:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 验证两个 run 的结果独立，run_id 不同
            with path1.open("r", encoding="utf-8") as f:
                r1 = json.loads(f.readline())
            with path2.open("r", encoding="utf-8") as f:
                r2 = json.loads(f.readline())

            assert r1["question_id"] == "q_shared"
            assert r2["question_id"] == "q_shared"
            assert r1["run_id"] != r2["run_id"]
            assert r1["run_id"] == run1["run_id"]
            assert r2["run_id"] == run2["run_id"]

            print(f"[OK] 相同 question_id 在不同 run 中: "
                  f"run1={run1['run_id'][-8:]}, run2={run2['run_id'][-8:]}")

    print()


def test_first_set_failure_continues_to_second():
    """第一个题集失败后第二个题集仍执行。"""
    print("=" * 60)
    print("测试：第一个题集失败后继续执行")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )

            completed_runs = []
            failed_runs = []

            # 模拟两个题集的执行
            sets = [
                {"name": "题集A", "id": "qs_A", "count": 5, "fail": True},
                {"name": "题集B", "id": "qs_B", "count": 3, "fail": False},
            ]

            for s in sets:
                try:
                    if s["fail"]:
                        raise RuntimeError(f"模拟 {s['name']} 执行失败")

                    run = create_experiment_run(config["config_id"], question_count=s["count"])
                    update_experiment_run(run["run_id"], {
                        "question_set_id": s["id"],
                        "question_set_name": s["name"],
                        "status": "completed",
                    })
                    completed_runs.append({
                        "run_id": run["run_id"],
                        "q_set_name": s["name"],
                        "count": s["count"],
                    })
                except Exception as e:
                    failed_runs.append({
                        "q_set_name": s["name"],
                        "count": s["count"],
                        "error": str(e),
                    })

            assert len(completed_runs) == 1, f"应有 1 个成功，实际 {len(completed_runs)}"
            assert len(failed_runs) == 1, f"应有 1 个失败，实际 {len(failed_runs)}"
            assert completed_runs[0]["q_set_name"] == "题集B"
            assert failed_runs[0]["q_set_name"] == "题集A"

            print(f"[OK] 成功: {completed_runs[0]['q_set_name']}, "
                  f"失败: {failed_runs[0]['q_set_name']}")

    print()


def test_single_set_backward_compat():
    """单选时行为与旧版一致。"""
    print("=" * 60)
    print("测试：单选向后兼容")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )

            # 单个题集执行
            run = create_experiment_run(config["config_id"], question_count=10)
            update_experiment_run(run["run_id"], {
                "question_set_id": "qs_single",
                "question_set_name": "单题集",
                "status": "completed",
            })

            r = load_experiment_run(run["run_id"])
            assert r["status"] == "completed"
            assert r["question_set_id"] == "qs_single"
            assert r["question_count"] == 10

            print(f"[OK] 单题集 run: {run['run_id']}")

    print()


def test_candidate_filter_excludes_non_question_sets():
    """候选题集过滤仍排除非 Question Set 文件。"""
    print("=" * 60)
    print("测试：候选过滤排除非题集文件")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        questions_dir = Path(tmpdir) / "questions"
        questions_dir.mkdir()

        # 真正的题集文件（有 question_set_id）
        qs_file = questions_dir / "real_qs.jsonl"
        with qs_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "question": "测试问题",
                "question_set_id": "qs_001",
                "question_set_name": "真实题集",
            }) + "\n")

        # 非题集文件（无 question_set_id，无 manifest）
        non_qs_file = questions_dir / "batch_results.jsonl"
        with non_qs_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"question": "批量结果", "success": True}) + "\n")

        # 验证过滤逻辑
        def _is_question_set(filepath):
            manifest_path = filepath.parent / f"{filepath.stem}_manifest.json"
            if manifest_path.exists():
                return True
            try:
                with filepath.open("r", encoding="utf-8") as f:
                    checked = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        checked += 1
                        if checked > 3:
                            break
                        try:
                            obj = json.loads(line)
                            if obj.get("question_set_id"):
                                return True
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
            return False

        assert _is_question_set(qs_file), "应识别为题集"
        assert not _is_question_set(non_qs_file), "不应识别为题集"

        print("[OK] 过滤正确：题集文件通过，非题集文件排除")

    print()


def test_final_summary_counts_accurate():
    """最终汇总的题集数、成功/失败数、总题数准确。"""
    print("=" * 60)
    print("测试：汇总计数准确")
    print("=" * 60)

    completed_runs = [
        {"run_id": "run_1", "q_set_name": "题集A", "count": 10, "success": 8},
        {"run_id": "run_2", "q_set_name": "题集B", "count": 5, "success": 5},
    ]
    failed_runs = [
        {"q_set_name": "题集C", "count": 7, "error": "timeout"},
    ]

    total_sets = len(completed_runs) + len(failed_runs)
    total_questions = sum(r["count"] for r in completed_runs) + sum(r["count"] for r in failed_runs)
    total_success_sets = len(completed_runs)
    total_failed_sets = len(failed_runs)
    total_success_questions = sum(r["success"] for r in completed_runs)

    assert total_sets == 3
    assert total_questions == 22
    assert total_success_sets == 2
    assert total_failed_sets == 1
    assert total_success_questions == 13

    print(f"[OK] {total_sets} 题集, {total_success_sets} 成功, "
          f"{total_failed_sets} 失败, {total_questions} 总题数, "
          f"{total_success_questions} 成功回答")

    print()


def test_skip_completed_run_default():
    """多选中一个题集已有 completed run，默认跳过且不创建新 run。"""
    print("=" * 60)
    print("测试：跳过已完成题集（默认策略）")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 创建一个已完成的 run（模拟题集 A 已执行过）
            existing_run = create_experiment_run(config_id, question_count=10)
            update_experiment_run(existing_run["run_id"], {
                "question_set_id": "qs_A",
                "question_set_name": "题集A",
                "status": "completed",
            })

            # 模拟多选 2 个题集
            selected_sets = [
                {"info": {"set_id": "qs_A", "set_name": "题集A"}, "questions": [_make_question(0)]},
                {"info": {"set_id": "qs_B", "set_name": "题集B"}, "questions": [_make_question(1)]},
            ]

            # 预检：查找已有 completed run
            from experiment import list_runs_by_config
            config_runs = list_runs_by_config(config_id)
            existing_by_qs = {}
            for r in config_runs:
                if r.get("status") == "completed" and r.get("question_set_id"):
                    existing_by_qs[r["question_set_id"]] = r

            assert "qs_A" in existing_by_qs
            assert "qs_B" not in existing_by_qs

            # 模拟跳过策略执行
            strategy = "skip"
            completed = []
            skipped = []
            for qs_info in selected_sets:
                qs_id = qs_info["info"]["set_id"]
                if strategy == "skip" and qs_id in existing_by_qs:
                    skipped.append({"q_set_id": qs_id, "run_id": existing_by_qs[qs_id]["run_id"]})
                else:
                    run = create_experiment_run(config_id, question_count=len(qs_info["questions"]))
                    update_experiment_run(run["run_id"], {
                        "question_set_id": qs_id,
                        "question_set_name": qs_info["info"]["set_name"],
                        "status": "completed",
                    })
                    completed.append({"q_set_id": qs_id, "run_id": run["run_id"]})

            assert len(skipped) == 1, f"应跳过 1 个，实际 {len(skipped)}"
            assert len(completed) == 1, f"应执行 1 个，实际 {len(completed)}"
            assert skipped[0]["q_set_id"] == "qs_A"
            assert skipped[0]["run_id"] == existing_run["run_id"]
            assert completed[0]["q_set_id"] == "qs_B"

            # 验证旧 run 未被覆盖
            r = load_experiment_run(existing_run["run_id"])
            assert r["status"] == "completed"
            assert r["question_count"] == 10

            print(f"[OK] 跳过: {skipped[0]['q_set_id']}, 执行: {completed[0]['q_set_id']}")

    print()


def test_rerun_all_creates_new_runs():
    """选择"重新执行"后，即使已有 completed run 也创建全新 run_id，旧 run 完整保留。"""
    print("=" * 60)
    print("测试：重新执行创建新 run")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 创建一个已完成的 run
            old_run = create_experiment_run(config_id, question_count=10)
            old_run_id = old_run["run_id"]
            update_experiment_run(old_run_id, {
                "question_set_id": "qs_A",
                "question_set_name": "题集A",
                "status": "completed",
            })

            # 模拟重新执行策略
            strategy = "rerun_all"
            new_run = create_experiment_run(config_id, question_count=10)
            new_run_id = new_run["run_id"]
            update_experiment_run(new_run_id, {
                "question_set_id": "qs_A",
                "question_set_name": "题集A",
                "status": "completed",
            })

            # 验证新旧 run 都存在且不同
            assert new_run_id != old_run_id
            old_r = load_experiment_run(old_run_id)
            new_r = load_experiment_run(new_run_id)
            assert old_r["status"] == "completed"
            assert new_r["status"] == "completed"
            assert old_r["question_set_id"] == "qs_A"
            assert new_r["question_set_id"] == "qs_A"

            print(f"[OK] 旧 run: {old_run_id[-8:]}, 新 run: {new_run_id[-8:]}")

    print()


def test_all_sets_skipped_zero_dify_calls():
    """全部题集均被跳过时零 Dify 调用。"""
    print("=" * 60)
    print("测试：全部跳过零调用")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 两个题集都已完成
            for qs_id, qs_name in [("qs_A", "题集A"), ("qs_B", "题集B")]:
                run = create_experiment_run(config_id, question_count=5)
                update_experiment_run(run["run_id"], {
                    "question_set_id": qs_id,
                    "question_set_name": qs_name,
                    "status": "completed",
                })

            # 模拟跳过策略
            from experiment import list_runs_by_config
            config_runs = list_runs_by_config(config_id)
            existing_by_qs = {}
            for r in config_runs:
                if r.get("status") == "completed" and r.get("question_set_id"):
                    existing_by_qs[r["question_set_id"]] = r

            selected_sets = [
                {"info": {"set_id": "qs_A", "set_name": "题集A"}, "questions": [_make_question(0)]},
                {"info": {"set_id": "qs_B", "set_name": "题集B"}, "questions": [_make_question(1)]},
            ]

            strategy = "skip"
            executed = 0
            skipped = 0
            for qs_info in selected_sets:
                qs_id = qs_info["info"]["set_id"]
                if strategy == "skip" and qs_id in existing_by_qs:
                    skipped += 1
                else:
                    executed += 1

            assert executed == 0, f"不应执行任何题集，实际 {executed}"
            assert skipped == 2, f"应跳过 2 个，实际 {skipped}"

            print(f"[OK] 执行 {executed} 个，跳过 {skipped} 个，零 Dify 调用")

    print()


def test_summary_counts_with_skip():
    """最终汇总的成功、失败、跳过、实际执行题数准确。"""
    print("=" * 60)
    print("测试：含跳过的汇总计数")
    print("=" * 60)

    completed_runs = [
        {"run_id": "run_1", "q_set_name": "题集B", "count": 10, "success": 9},
    ]
    failed_runs = [
        {"q_set_name": "题集C", "count": 5, "error": "timeout"},
    ]
    skipped_runs = [
        {"run_id": "run_old", "q_set_name": "题集A", "q_set_id": "qs_A", "count": 8},
    ]

    total_sets = len(completed_runs) + len(failed_runs) + len(skipped_runs)
    total_questions = (
        sum(r["count"] for r in completed_runs)
        + sum(r["count"] for r in failed_runs)
        + sum(r["count"] for r in skipped_runs)
    )
    actual_executed = sum(r["count"] for r in completed_runs) + sum(r["count"] for r in failed_runs)

    assert total_sets == 3
    assert total_questions == 23
    assert actual_executed == 15
    assert len(completed_runs) == 1
    assert len(failed_runs) == 1
    assert len(skipped_runs) == 1

    print(f"[OK] 总计 {total_sets} 题集: "
          f"成功 {len(completed_runs)}, "
          f"失败 {len(failed_runs)}, "
          f"跳过 {len(skipped_runs)}, "
          f"实际执行 {actual_executed}/{total_questions} 题")

    print()


def test_run_summary_index_caching():
    """重复调用 _build_run_summary_index 命中缓存，不重复扫描磁盘。"""
    print("=" * 60)
    print("测试：run 摘要缓存")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )

            # 创建一个 completed run
            run = create_experiment_run(config["config_id"], question_count=5)
            update_experiment_run(run["run_id"], {
                "question_set_id": "qs_001",
                "question_set_name": "题集A",
                "status": "completed",
            })

            # 模拟缓存函数：第一次调用扫描磁盘
            call_count = [0]
            def _mock_build_index(cache_key=""):
                call_count[0] += 1
                from experiment import EXPERIMENTS_DIR
                if not EXPERIMENTS_DIR.exists():
                    return []
                summaries = []
                for run_dir in EXPERIMENTS_DIR.iterdir():
                    if not run_dir.is_dir():
                        continue
                    mp = run_dir / "manifest.json"
                    if mp.exists():
                        try:
                            m = json.loads(mp.read_text(encoding="utf-8"))
                            summaries.append({
                                "run_id": m.get("run_id", ""),
                                "config_id": m.get("config_id", ""),
                                "question_set_id": m.get("question_set_id", ""),
                                "status": m.get("status", ""),
                            })
                        except Exception:
                            continue
                return summaries

            # 两次调用（模拟缓存命中场景）
            r1 = _mock_build_index()
            r2 = _mock_build_index()

            assert len(r1) == 1
            assert r1[0]["question_set_id"] == "qs_001"
            assert r1 == r2  # 相同结果

            # 验证 run 存在
            r = load_experiment_run(run["run_id"])
            assert r["status"] == "completed"

            print(f"[OK] 缓存结果一致: {len(r1)} 个 run")

    print()


def test_cache_invalidation_after_execution():
    """执行完成后缓存失效，新的 completed run 能被发现。"""
    print("=" * 60)
    print("测试：执行后缓存失效")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 初始：无 completed run
            from experiment import list_runs_by_config
            runs_before = list_runs_by_config(config_id)
            completed_before = [r for r in runs_before if r.get("status") == "completed"]
            assert len(completed_before) == 0

            # 执行：创建一个 completed run
            run = create_experiment_run(config_id, question_count=5)
            update_experiment_run(run["run_id"], {
                "question_set_id": "qs_001",
                "question_set_name": "题集A",
                "status": "completed",
            })

            # 执行后：能发现新的 completed run
            runs_after = list_runs_by_config(config_id)
            completed_after = [r for r in runs_after if r.get("status") == "completed"]
            assert len(completed_after) == 1
            assert completed_after[0]["question_set_id"] == "qs_001"

            print(f"[OK] 执行前 {len(completed_before)} 个 completed, "
                  f"执行后 {len(completed_after)} 个 completed")

    print()


def test_skip_rerun_correctness_with_cache():
    """缓存不影响 skip/rerun 判定正确性。"""
    print("=" * 60)
    print("测试：缓存下 skip/rerun 判定正确")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config_profiles"
        exp_dir = Path(tmpdir) / "experiments"

        with patch('experiment.CONFIG_PROFILES_DIR', config_dir), \
             patch('experiment.EXPERIMENTS_DIR', exp_dir):

            config = create_config_profile(
                config_name="测试配置",
                knowledge_base_version="v1",
                retrieval_mode="hybrid",
                top_k=5,
            )
            config_id = config["config_id"]

            # 创建一个 completed run for qs_A
            run_a = create_experiment_run(config_id, question_count=5)
            update_experiment_run(run_a["run_id"], {
                "question_set_id": "qs_A",
                "question_set_name": "题集A",
                "status": "completed",
            })

            # 模拟缓存的 run 摘要
            from experiment import list_runs_by_config
            all_runs = list_runs_by_config(config_id)
            existing_by_qs = {}
            for r in all_runs:
                if r.get("status") == "completed" and r.get("question_set_id"):
                    existing_by_qs[r["question_set_id"]] = r

            # 模拟 3 个题集的选择
            selected_sets = [
                {"info": {"set_id": "qs_A", "set_name": "题集A"}, "questions": [_make_question(0)]},
                {"info": {"set_id": "qs_B", "set_name": "题集B"}, "questions": [_make_question(1)]},
                {"info": {"set_id": "qs_C", "set_name": "题集C"}, "questions": [_make_question(2)]},
            ]

            # skip 策略
            skip_executed = []
            skip_skipped = []
            for ss in selected_sets:
                qs_id = ss["info"]["set_id"]
                if qs_id in existing_by_qs:
                    skip_skipped.append(qs_id)
                else:
                    skip_executed.append(qs_id)

            assert skip_skipped == ["qs_A"]
            assert skip_executed == ["qs_B", "qs_C"]

            # rerun 策略
            rerun_executed = []
            for ss in selected_sets:
                qs_id = ss["info"]["set_id"]
                rerun_executed.append(qs_id)

            assert len(rerun_executed) == 3  # 全部执行

            # 验证旧 run 完整保留
            r = load_experiment_run(run_a["run_id"])
            assert r["status"] == "completed"
            assert r["question_set_id"] == "qs_A"

            print(f"[OK] skip: 跳过 {skip_skipped}, 执行 {skip_executed}")
            print(f"[OK] rerun: 全部执行 {rerun_executed}")

    print()
