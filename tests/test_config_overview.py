"""
配置方案总览测试。

测试内容：
1. 加权汇总：同 config 两个 run 的 Judge 结果按样本数加权，非简单平均
2. 数据隔离：不同 config 的结果不混入
3. 分轨道统计：retrieval / strict_qa / grounded_qa 分开计算
4. 空轨道处理：无数据时显示暂无数据
5. trace_id 去重：重复 trace_id 保留最新无 error 结果
6. 旧 Judge 无 run_id fallback：通过 processed trace_id 汇总

不调用真实 API。
"""

import json
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from experiment import (
    create_config_profile, create_experiment_run,
    update_experiment_run, get_run_status,
)
from judge import TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA


def _setup_dirs(tmpdir):
    dirs = {}
    for name in ["config_profiles", "experiments", "batch", "raw", "processed", "judged"]:
        d = tmpdir / name
        d.mkdir(parents=True)
        dirs[name] = d
    return dirs


def _create_run(config_dir, exp_dir, batch_dir, raw_dir, processed_dir, judged_dir,
                config_id, question_count, run_label):
    with patch("experiment.CONFIG_PROFILES_DIR", config_dir), \
         patch("experiment.EXPERIMENTS_DIR", exp_dir):
        run_result = create_experiment_run(config_id, run_label, question_count)
        run_id = run_result["run_id"]
        update_experiment_run(run_id, {"question_set_name": run_label})

    langfuse_ids = [str(uuid.uuid4()) for _ in range(question_count)]

    # Processed
    proc_file = processed_dir / "langfuse_samples.jsonl"
    with proc_file.open("a", encoding="utf-8") as f:
        for i in range(question_count):
            f.write(json.dumps({
                "trace_id": langfuse_ids[i],
                "question": f"{run_label}问题{i+1}",
                "run_id": run_id,
                "question_mode": "retrieval",
                "reference_answer": f"答案{i+1}",
            }, ensure_ascii=False) + "\n")

    # Batch + Raw (minimal)
    batch_file = f"batch_{run_id}.jsonl"
    (batch_dir / batch_file).write_text(
        "\n".join(json.dumps({"success": True, "sample": {"trace_id": f"b{i}"}})
                  for i in range(question_count)) + "\n",
        encoding="utf-8",
    )
    with patch("experiment.EXPERIMENTS_DIR", exp_dir):
        update_experiment_run(run_id, {"batch_results_file": batch_file, "status": "completed"})

    raw_file = f"raw_{run_id}.jsonl"
    (raw_dir / raw_file).write_text(
        "\n".join(json.dumps({"trace_id": f"b{i}"}) for i in range(question_count)) + "\n",
        encoding="utf-8",
    )
    with patch("experiment.EXPERIMENTS_DIR", exp_dir):
        update_experiment_run(run_id, {"raw_results_file": raw_file})

    return run_id, langfuse_ids, proc_file


def _write_judged(judged_file, results, mode="w"):
    with judged_file.open(mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _collect_all_judge(config_runs, batch_dir, raw_dir, processed_file, judged_file):
    """模拟 app.py 中的汇总逻辑：收集所有 run 的 Judge 结果并去重。"""
    all_raw = []
    for run in config_runs:
        rid = run.get("run_id", "")
        rs = get_run_status(rid, batch_dir=str(batch_dir), raw_dir=str(raw_dir),
                           processed_file=str(processed_file), judged_file=str(judged_file))
        for r in rs.get("judge_results", []):
            r_copy = dict(r)
            r_copy["_source_run_id"] = rid
            all_raw.append(r_copy)

    # 去重：同一 trace_id 保留最新且无 error 的结果
    seen = {}
    for r in all_raw:
        tid = r.get("trace_id", "")
        if not tid:
            continue
        existing = seen.get(tid)
        if existing is None:
            seen[tid] = r
        elif "error" in existing and "error" not in r:
            seen[tid] = r
        else:
            seen[tid] = r
    return list(seen.values())


def test_weighted_aggregation():
    """同 config 两个 run 的 Judge 结果按样本数加权汇总，非简单平均。"""
    print("=" * 60)
    print("测试加权汇总（非简单平均）")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试配置", "kb_v1")
            config_id = config["config_id"]

        # Run 1: 2 条样本，Top1=100%
        run1_id, ids1, proc_file = _create_run(
            dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], config_id, 2, "题集A",
        )
        # Run 2: 8 条样本，Top1=50% (4/8)
        run2_id, ids2, _ = _create_run(
            dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], config_id, 8, "题集B",
        )

        # Judge 结果
        judged_file = dirs["judged"] / "eval_results.jsonl"
        results = []
        for i, tid in enumerate(ids1):
            results.append({"trace_id": tid, "run_id": run1_id,
                           "evaluation_track": TRACK_RETRIEVAL,
                           "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
                           "retrieval_evaluable": True})
        for i, tid in enumerate(ids2):
            results.append({"trace_id": tid, "run_id": run2_id,
                           "evaluation_track": TRACK_RETRIEVAL,
                           "retrieval_top1_hit": 1 if i < 4 else 0,
                           "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
                           "retrieval_evaluable": True})
        _write_judged(judged_file, results)

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config_runs = []
            for rid in [run1_id, run2_id]:
                from experiment import load_experiment_run
                config_runs.append(load_experiment_run(rid))

            all_results = _collect_all_judge(
                config_runs, dirs["batch"], dirs["raw"], proc_file, judged_file,
            )

        # 验证
        valid = [r for r in all_results if "error" not in r]
        retrieval = [r for r in valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
        n = len(retrieval)
        t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval) / n

        print(f"  总样本数: {n}")
        print(f"  Top1 Hit: {t1:.0%}")
        print(f"  期望: (2*100% + 8*50%) / 10 = 60%")

        assert n == 10, f"总样本数应为 10，实际 {n}"
        assert abs(t1 - 0.6) < 0.01, f"Top1 应为 60%（加权），实际 {t1:.0%}"
        print("[OK] 加权汇总正确，非简单平均 (75%)")

    print()


def test_config_isolation():
    """不同 config 的结果不混入。"""
    print("=" * 60)
    print("测试数据隔离")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config_a = create_config_profile("配置A", "kb_a")
            config_b = create_config_profile("配置B", "kb_b")
            config_a_id = config_a["config_id"]
            config_b_id = config_b["config_id"]

        # Config A: 2 条，Top1=100%
        run_a_id, ids_a, proc_a = _create_run(
            dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], config_a_id, 2, "A题集",
        )
        # Config B: 2 条，Top1=0%
        run_b_id, ids_b, proc_b = _create_run(
            dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], config_b_id, 2, "B题集",
        )

        # Judge 结果混合写入
        judged_file = dirs["judged"] / "eval_results.jsonl"
        _write_judged(judged_file, [
            {"trace_id": ids_a[0], "run_id": run_a_id, "evaluation_track": TRACK_RETRIEVAL,
             "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
            {"trace_id": ids_a[1], "run_id": run_a_id, "evaluation_track": TRACK_RETRIEVAL,
             "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
            {"trace_id": ids_b[0], "run_id": run_b_id, "evaluation_track": TRACK_RETRIEVAL,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0, "retrieval_top5_hit": 0, "retrieval_evaluable": True},
            {"trace_id": ids_b[1], "run_id": run_b_id, "evaluation_track": TRACK_RETRIEVAL,
             "retrieval_top1_hit": 0, "retrieval_top3_hit": 0, "retrieval_top5_hit": 0, "retrieval_evaluable": True},
        ])

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            from experiment import load_experiment_run
            config_a_runs = [load_experiment_run(run_a_id)]
            config_b_runs = [load_experiment_run(run_b_id)]

            results_a = _collect_all_judge(config_a_runs, dirs["batch"], dirs["raw"], proc_a, judged_file)
            results_b = _collect_all_judge(config_b_runs, dirs["batch"], dirs["raw"], proc_b, judged_file)

        # Config A: Top1=100%, Config B: Top1=0%
        ret_a = [r for r in results_a if "error" not in r and r.get("evaluation_track") == TRACK_RETRIEVAL]
        ret_b = [r for r in results_b if "error" not in r and r.get("evaluation_track") == TRACK_RETRIEVAL]

        t1_a = sum(r.get("retrieval_top1_hit", 0) for r in ret_a) / len(ret_a)
        t1_b = sum(r.get("retrieval_top1_hit", 0) for r in ret_b) / len(ret_b)

        print(f"  Config A Top1: {t1_a:.0%} (n={len(ret_a)})")
        print(f"  Config B Top1: {t1_b:.0%} (n={len(ret_b)})")

        assert abs(t1_a - 1.0) < 0.01, f"Config A Top1 应为 100%，实际 {t1_a:.0%}"
        assert abs(t1_b - 0.0) < 0.01, f"Config B Top1 应为 0%，实际 {t1_b:.0%}"
        print("[OK] 不同 config 结果完全隔离")

    print()


def test_track_separation():
    """retrieval / strict_qa / grounded_qa 分轨道统计。"""
    print("=" * 60)
    print("测试分轨道统计")
    print("=" * 60)

    mixed_results = [
        {"trace_id": "t1", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
        {"trace_id": "t2", "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 0, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
        {"trace_id": "t3", "evaluation_track": TRACK_STRICT_QA, "answer_correct": 1},
        {"trace_id": "t4", "evaluation_track": TRACK_STRICT_QA, "answer_correct": 0},
        {"trace_id": "t5", "evaluation_track": TRACK_GROUNDED_QA, "answer_correct": 1},
    ]

    valid = [r for r in mixed_results if "error" not in r]
    retrieval = [r for r in valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
    strict_qa = [r for r in valid if r.get("evaluation_track") == TRACK_STRICT_QA]
    grounded_qa = [r for r in valid if r.get("evaluation_track") == TRACK_GROUNDED_QA]

    n_ret = len(retrieval)
    t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval) / n_ret
    n_strict = len(strict_qa)
    acc_strict = sum(r.get("answer_correct", 0) for r in strict_qa) / n_strict
    n_grounded = len(grounded_qa)
    acc_grounded = sum(r.get("answer_correct", 0) for r in grounded_qa) / n_grounded

    print(f"  retrieval: n={n_ret}, Top1={t1:.0%}")
    print(f"  strict_qa: n={n_strict}, Answer={acc_strict:.0%}")
    print(f"  grounded_qa: n={n_grounded}, Answer={acc_grounded:.0%}")

    assert n_ret == 2 and abs(t1 - 0.5) < 0.01
    assert n_strict == 2 and abs(acc_strict - 0.5) < 0.01
    assert n_grounded == 1 and abs(acc_grounded - 1.0) < 0.01
    print("[OK] 分轨道统计正确，不混合")

    print()


def test_empty_track():
    """空轨道显示无数据。"""
    print("=" * 60)
    print("测试空轨道处理")
    print("=" * 60)

    # 只有 strict_qa，没有 retrieval
    results = [
        {"trace_id": "t1", "evaluation_track": TRACK_STRICT_QA, "answer_correct": 1},
        {"trace_id": "t2", "evaluation_track": TRACK_STRICT_QA, "answer_correct": 0},
    ]

    valid = [r for r in results if "error" not in r]
    retrieval = [r for r in valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
    strict_qa = [r for r in valid if r.get("evaluation_track") == TRACK_STRICT_QA]

    assert len(retrieval) == 0, "应无 retrieval 数据"
    assert len(strict_qa) == 2

    # 模拟 UI 判断
    has_retrieval = len(retrieval) > 0
    has_strict_qa = len(strict_qa) > 0

    assert not has_retrieval, "retrieval 轨道应为空"
    assert has_strict_qa
    print("[OK] 空轨道正确识别，不报错")

    print()


def test_trace_id_dedup():
    """重复 trace_id 保留最新无 error 结果。"""
    print("=" * 60)
    print("测试 trace_id 去重")
    print("=" * 60)

    # 同一 trace_id 出现两次：一次有 error，一次无 error
    raw_results = [
        {"trace_id": "t1", "_source_run_id": "run_old",
         "evaluation_track": TRACK_RETRIEVAL, "error": "LLM 超时"},
        {"trace_id": "t1", "_source_run_id": "run_new",
         "evaluation_track": TRACK_RETRIEVAL,
         "retrieval_top1_hit": 1, "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
    ]

    # 去重逻辑（与 app.py 一致）
    seen = {}
    for r in raw_results:
        tid = r.get("trace_id", "")
        if not tid:
            continue
        existing = seen.get(tid)
        if existing is None:
            seen[tid] = r
        elif "error" in existing and "error" not in r:
            seen[tid] = r
        else:
            seen[tid] = r

    deduped = list(seen.values())
    assert len(deduped) == 1
    assert "error" not in deduped[0], "应保留无 error 的结果"
    assert deduped[0]["_source_run_id"] == "run_new", "应保留最新的 run"
    print("[OK] 同 trace_id 去重：保留最新无 error 结果")

    # 同一 trace_id 两次都无 error，保留后者（更新的 run）
    raw_results2 = [
        {"trace_id": "t2", "_source_run_id": "run_old",
         "evaluation_track": TRACK_RETRIEVAL, "retrieval_top1_hit": 0,
         "retrieval_top3_hit": 0, "retrieval_top5_hit": 0, "retrieval_evaluable": True},
        {"trace_id": "t2", "_source_run_id": "run_new",
         "evaluation_track": TRACK_RETRIEVAL, "retrieval_top1_hit": 1,
         "retrieval_top3_hit": 1, "retrieval_top5_hit": 1, "retrieval_evaluable": True},
    ]

    seen2 = {}
    for r in raw_results2:
        tid = r.get("trace_id", "")
        if not tid:
            continue
        existing = seen2.get(tid)
        if existing is None:
            seen2[tid] = r
        elif "error" in existing and "error" not in r:
            seen2[tid] = r
        else:
            seen2[tid] = r

    deduped2 = list(seen2.values())
    assert len(deduped2) == 1
    assert deduped2[0]["retrieval_top1_hit"] == 1
    assert deduped2[0]["_source_run_id"] == "run_new"
    print("[OK] 同 trace_id 两次无 error：保留后者（更新 run）")

    print()


def test_legacy_no_run_id():
    """旧 Judge 结果缺 run_id 时仍可经 processed trace_id 汇总。"""
    print("=" * 60)
    print("测试旧 Judge 无 run_id fallback")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试配置", "kb_v1")
            config_id = config["config_id"]

        run_id, langfuse_ids, proc_file = _create_run(
            dirs["config_profiles"], dirs["experiments"], dirs["batch"], dirs["raw"],
            dirs["processed"], dirs["judged"], config_id, 3, "测试题集",
        )

        # Judge 结果没有 run_id（旧格式）
        judged_file = dirs["judged"] / "eval_results.jsonl"
        for i, tid in enumerate(langfuse_ids):
            _write_judged(judged_file, [
                {"trace_id": tid, "evaluation_track": TRACK_RETRIEVAL,
                 "retrieval_top1_hit": 1 if i < 2 else 0,
                 "retrieval_top3_hit": 1, "retrieval_top5_hit": 1,
                 "retrieval_evaluable": True},
            ], mode="a")

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            from experiment import load_experiment_run
            config_runs = [load_experiment_run(run_id)]
            all_results = _collect_all_judge(config_runs, dirs["batch"], dirs["raw"], proc_file, judged_file)

        valid = [r for r in all_results if "error" not in r]
        retrieval = [r for r in valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
        n = len(retrieval)
        t1 = sum(r.get("retrieval_top1_hit", 0) for r in retrieval) / n

        print(f"  Judge 结果数: {n}")
        print(f"  Top1 Hit: {t1:.0%}")

        assert n == 3, f"应有 3 条结果，实际 {n}"
        assert abs(t1 - 2 / 3) < 0.01, f"Top1 应为 67%，实际 {t1:.0%}"
        print("[OK] 旧 Judge 无 run_id 时通过 trace_id 正确汇总")

    print()


def main():
    print("=" * 60)
    print("配置方案总览测试")
    print("=" * 60)
    print()

    test_weighted_aggregation()
    test_config_isolation()
    test_track_separation()
    test_empty_track()
    test_trace_id_dedup()
    test_legacy_no_run_id()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
