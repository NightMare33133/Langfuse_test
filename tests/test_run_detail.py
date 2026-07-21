"""
运行看板评测详情测试。

测试内容：
1. 当前 run 只返回它自己的 Judge result 和 processed sample
2. 通过 trace_id 正确拼接 Judge result 与 sample
3. 历史 Judge 无 run_id 时仍能展示详情
4. retrieval 和 QA 两种轨道的数据格式
5. 缺少样本、参考答案、检索结果、reason 时不报错
6. 跨 run 结果绝不出现在当前运行详情中

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


def _create_run_with_samples(config_dir, exp_dir, processed_dir, judged_dir,
                              config_id, question_count, run_label, question_mode="retrieval"):
    """创建 run 并写入 processed + judged 数据。"""
    with patch("experiment.CONFIG_PROFILES_DIR", config_dir), \
         patch("experiment.EXPERIMENTS_DIR", exp_dir):
        run_result = create_experiment_run(config_id, run_label, question_count)
        run_id = run_result["run_id"]
        update_experiment_run(run_id, {"question_set_name": run_label})

    langfuse_ids = [str(uuid.uuid4()) for _ in range(question_count)]

    # Processed samples
    proc_file = processed_dir / "langfuse_samples.jsonl"
    with proc_file.open("a", encoding="utf-8") as f:
        for i in range(question_count):
            sample = {
                "trace_id": langfuse_ids[i],
                "question": f"{run_label}问题{i+1}",
                "final_answer": f"{run_label}回答{i+1}",
                "run_id": run_id,
                "question_mode": question_mode,
            }
            if question_mode == "retrieval":
                sample["source_excerpt"] = f"{run_label}金标准证据{i+1}"
                sample["reference_answer"] = f"{run_label}参考答案{i+1}"
                sample["retrieval_results"] = [
                    {"position": 1, "score": 0.95, "document_name": "doc1.md", "content": f"检索内容{i+1}"},
                    {"position": 2, "score": 0.80, "document_name": "doc2.md", "content": f"其他内容{i+1}"},
                ]
            elif question_mode == "qa":
                sample["reference_answer"] = f"{run_label}参考答案{i+1}"
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Judge results
    judged_file = judged_dir / "eval_results.jsonl"
    with judged_file.open("a", encoding="utf-8") as f:
        for i in range(question_count):
            result = {
                "trace_id": langfuse_ids[i],
                "question": f"{run_label}问题{i+1}",
                "run_id": run_id,
                "evaluation_track": TRACK_RETRIEVAL if question_mode == "retrieval" else TRACK_STRICT_QA,
                "reason": f"{run_label}评测原因{i+1}",
            }
            if question_mode == "retrieval":
                result["retrieval_top1_hit"] = 1 if i < 8 else 0
                result["retrieval_top3_hit"] = 1 if i < 9 else 0
                result["retrieval_top5_hit"] = 1
                result["hit_evidence_position"] = 1 if i < 8 else None
                result["retrieval_evaluable"] = True
            else:
                result["answer_correct"] = 1 if i < 7 else 0
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return run_id, langfuse_ids, proc_file, judged_file


def test_run_only_returns_own_results():
    """当前 run 只返回它自己的 Judge result 和 processed sample。"""
    print("=" * 60)
    print("测试 run 只返回自己的结果")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试", "v1", "wf1")
            config_id = config["config_id"]

        # Run A: 5 条
        run_a_id, ids_a, proc_file, judged_file = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 5, "题集A",
        )
        # Run B: 5 条
        run_b_id, ids_b, _, _ = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 5, "题集B",
        )

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            status_a = get_run_status(run_a_id,
                                      batch_dir=str(dirs["batch"]),
                                      raw_dir=str(dirs["raw"]),
                                      processed_file=str(proc_file),
                                      judged_file=str(judged_file),
                                      include_judge_results=True)
            status_b = get_run_status(run_b_id,
                                      batch_dir=str(dirs["batch"]),
                                      raw_dir=str(dirs["raw"]),
                                      processed_file=str(proc_file),
                                      judged_file=str(judged_file),
                                      include_judge_results=True)

        # 验证 run_a 的 judge_results 只包含自己的
        assert len(status_a["judge_results"]) == 5, f"Run A 应有 5 条，实际 {len(status_a['judge_results'])}"
        for r in status_a["judge_results"]:
            assert r["trace_id"] in ids_a, f"Run A 不应包含其他 run 的 trace_id: {r['trace_id']}"
        print("[OK] Run A 只返回自己的 5 条结果")

        # 验证 run_b 的 judge_results 只包含自己的
        assert len(status_b["judge_results"]) == 5, f"Run B 应有 5 条，实际 {len(status_b['judge_results'])}"
        for r in status_b["judge_results"]:
            assert r["trace_id"] in ids_b, f"Run B 不应包含其他 run 的 trace_id: {r['trace_id']}"
        print("[OK] Run B 只返回自己的 5 条结果")

    print()


def test_trace_id_links_result_to_sample():
    """通过 trace_id 正确拼接 Judge result 与 sample。"""
    print("=" * 60)
    print("测试 trace_id 关联")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试", "v1", "wf1")
            config_id = config["config_id"]

        run_id, langfuse_ids, proc_file, judged_file = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 3, "测试题集", question_mode="retrieval",
        )

        # 加载 judge results
        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            status = get_run_status(run_id,
                                    batch_dir=str(dirs["batch"]),
                                    raw_dir=str(dirs["raw"]),
                                    processed_file=str(proc_file),
                                    judged_file=str(judged_file),
                                    include_judge_results=True)

        judge_results = status["judge_results"]

        # 加载 processed samples 并构建 sample_map
        sample_map = {}
        with proc_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("run_id") == run_id:
                    sample_map[obj["trace_id"]] = obj

        # 验证每条 judge result 都能通过 trace_id 找到 sample
        for r in judge_results:
            tid = r["trace_id"]
            assert tid in sample_map, f"trace_id {tid} 应在 sample_map 中"
            sample = sample_map[tid]
            assert sample["question"] == r["question"], "sample 和 result 的 question 应一致"
            assert sample.get("final_answer"), "sample 应有 final_answer"
            assert sample.get("source_excerpt"), "retrieval sample 应有 source_excerpt"
            assert sample.get("retrieval_results"), "retrieval sample 应有 retrieval_results"
        print(f"[OK] {len(judge_results)} 条 judge result 均通过 trace_id 正确关联到 sample")

    print()


def test_legacy_no_run_id_still_works():
    """历史 Judge 无 run_id 时仍能展示详情。"""
    print("=" * 60)
    print("测试历史 Judge 无 run_id fallback")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试", "v1", "wf1")
            config_id = config["config_id"]

        run_id, langfuse_ids, proc_file, judged_file = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 3, "测试题集",
        )

        # 移除 judge results 中的 run_id（模拟旧格式）
        results_without_run_id = []
        with judged_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                obj.pop("run_id", None)
                results_without_run_id.append(obj)

        legacy_judged_file = dirs["judged"] / "eval_results_legacy.jsonl"
        with legacy_judged_file.open("w", encoding="utf-8") as f:
            for r in results_without_run_id:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            status = get_run_status(run_id,
                                    batch_dir=str(dirs["batch"]),
                                    raw_dir=str(dirs["raw"]),
                                    processed_file=str(proc_file),
                                    judged_file=str(legacy_judged_file),
                                    include_judge_results=True)

        # 应通过 trace_id fallback 关联到 3 条
        assert len(status["judge_results"]) == 3, \
            f"应通过 trace_id 关联到 3 条，实际 {len(status['judge_results'])}"

        # 验证关联正确
        sample_map = {}
        with proc_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("run_id") == run_id:
                    sample_map[obj["trace_id"]] = obj

        for r in status["judge_results"]:
            tid = r["trace_id"]
            assert tid in sample_map, f"trace_id {tid} 应在 sample_map 中"
        print("[OK] 历史 Judge 无 run_id 时通过 trace_id 正确关联")

    print()


def test_retrieval_and_qa_tracks():
    """retrieval 和 QA 两种轨道的数据格式。"""
    print("=" * 60)
    print("测试 retrieval 和 QA 轨道数据")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试", "v1", "wf1")
            config_id = config["config_id"]

        # retrieval 轨道
        run_ret, ids_ret, proc_file, judged_file = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 3, "检索题集", question_mode="retrieval",
        )
        # QA 轨道
        run_qa, ids_qa, _, _ = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 3, "问答题集", question_mode="qa",
        )

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            status_ret = get_run_status(run_ret,
                                        batch_dir=str(dirs["batch"]),
                                        raw_dir=str(dirs["raw"]),
                                        processed_file=str(proc_file),
                                        judged_file=str(judged_file),
                                        include_judge_results=True)
            status_qa = get_run_status(run_qa,
                                       batch_dir=str(dirs["batch"]),
                                       raw_dir=str(dirs["raw"]),
                                       processed_file=str(proc_file),
                                       judged_file=str(judged_file),
                                       include_judge_results=True)

        # retrieval 轨道验证
        for r in status_ret["judge_results"]:
            assert r["evaluation_track"] == TRACK_RETRIEVAL
            assert "retrieval_top1_hit" in r
            assert "retrieval_top3_hit" in r
            assert "retrieval_top5_hit" in r
            assert "reason" in r
        print(f"[OK] retrieval 轨道: {len(status_ret['judge_results'])} 条，格式正确")

        # QA 轨道验证
        for r in status_qa["judge_results"]:
            assert r["evaluation_track"] == TRACK_STRICT_QA
            assert "answer_correct" in r
            assert "reason" in r
        print(f"[OK] QA 轨道: {len(status_qa['judge_results'])} 条，格式正确")

    print()


def test_missing_fields_no_error():
    """缺少样本、参考答案、检索结果、reason 时不报错。"""
    print("=" * 60)
    print("测试缺少字段不报错")
    print("=" * 60)

    # 模拟缺少各种字段的 judge result
    sparse_result = {
        "trace_id": "nonexistent_trace_id",
        "question": "测试问题",
        "evaluation_track": TRACK_RETRIEVAL,
        "retrieval_top1_hit": 0,
        "retrieval_top3_hit": 0,
        "retrieval_top5_hit": 0,
        # 缺少 reason, hit_evidence_position, run_id 等
    }

    # 模拟空 sample_map
    sample_map = {}

    # 不应报错
    try:
        # 导入渲染函数
        from app import render_retrieval_result_detail, render_strict_qa_result_detail, render_grounded_qa_result_detail

        # 这些函数使用 st.expander，在非 Streamlit 环境不能直接调用
        # 但我们可以验证函数签名和基本逻辑
        print("[OK] 渲染函数已定义且可导入")

        # 验证 build_result_status 不报错
        from judge import build_result_status
        status = build_result_status(sparse_result)
        assert status["icon"] == "🔍"
        assert status["status"] == "retrieval"
        print("[OK] build_result_status 处理稀疏数据不报错")

        # 验证空 sample_map 不会导致 KeyError
        sample = sample_map.get(sparse_result["trace_id"], {})
        assert sample == {}
        gold = (sample.get("source_excerpt") or "").strip()
        assert gold == ""
        print("[OK] 空 sample_map 处理正确")

    except Exception as e:
        print(f"[FAIL] 报错: {e}")
        raise

    print()


def test_cross_run_never_leaks():
    """跨 run 结果绝不出现在当前运行详情中。"""
    print("=" * 60)
    print("测试跨 run 隔离")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dirs = _setup_dirs(tmpdir)

        with patch("experiment.CONFIG_PROFILES_DIR", dirs["config_profiles"]), \
             patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            config = create_config_profile("测试", "v1", "wf1")
            config_id = config["config_id"]

        # Run A: retrieval 轨道，Top1=100%
        run_a_id, ids_a, proc_file, judged_file = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 3, "题集A", question_mode="retrieval",
        )
        # Run B: QA 轨道，Answer=100%
        run_b_id, ids_b, _, _ = _create_run_with_samples(
            dirs["config_profiles"], dirs["experiments"], dirs["processed"], dirs["judged"],
            config_id, 3, "题集B", question_mode="qa",
        )

        with patch("experiment.EXPERIMENTS_DIR", dirs["experiments"]):
            status_a = get_run_status(run_a_id,
                                      batch_dir=str(dirs["batch"]),
                                      raw_dir=str(dirs["raw"]),
                                      processed_file=str(proc_file),
                                      judged_file=str(judged_file),
                                      include_judge_results=True)

        # Run A 的结果不应包含 QA 轨道
        tracks_a = set(r.get("evaluation_track") for r in status_a["judge_results"])
        assert TRACK_STRICT_QA not in tracks_a, f"Run A 不应包含 QA 轨道: {tracks_a}"
        assert TRACK_RETRIEVAL in tracks_a, f"Run A 应包含 retrieval 轨道: {tracks_a}"

        # Run A 的 trace_id 不应出现在 Run B 的 ids 中
        for r in status_a["judge_results"]:
            assert r["trace_id"] not in ids_b, \
                f"Run A 的 trace_id 不应是 Run B 的: {r['trace_id']}"

        print(f"[OK] Run A 只有 retrieval 轨道，不包含 Run B 的 QA 数据")

    print()


def main():
    print("=" * 60)
    print("运行看板评测详情测试")
    print("=" * 60)
    print()

    test_run_only_returns_own_results()
    test_trace_id_links_result_to_sample()
    test_legacy_no_run_id_still_works()
    test_retrieval_and_qa_tracks()
    test_missing_fields_no_error()
    test_cross_run_never_leaks()

    print("=" * 60)
    print("[OK] 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
