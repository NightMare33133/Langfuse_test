"""
实验版本管理模块 - RAG 配置方案 + 测试运行记录。

数据模型：
1. config_profile: 可复用的 RAG 配置方案
2. experiment_run: 每次批量提问的运行记录

目录结构：
- data/config_profiles/<config_id>.json
- data/experiments/<run_id>/manifest.json
"""

import json
import random
import re
import string
from datetime import datetime
from pathlib import Path

CONFIG_PROFILES_DIR = Path(__file__).parent / "data" / "config_profiles"
EXPERIMENTS_DIR = Path(__file__).parent / "data" / "experiments"


# ========== ID 生成 ==========

def _generate_id(prefix: str, name: str = "") -> str:
    """生成唯一 ID。

    格式: <prefix>_<YYYYMMDD_HHMMSS_microseconds>_<slug>[_<suffix>]
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"

    if name:
        slug = re.sub(r'[^\w\u4e00-\u9fff]', '_', name.strip())
        slug = re.sub(r'_+', '_', slug).strip('_')[:20]
    else:
        slug = "unnamed"

    return f"{prefix}_{timestamp}_{slug}"


def _ensure_unique_file(base_path: Path) -> Path:
    """确保文件路径唯一。如果已存在，追加随机后缀。"""
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    while base_path.exists():
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        base_path = parent / f"{stem}_{rand}{suffix}"

    return base_path


def _ensure_unique_dir(base_id: str, parent_dir: Path) -> tuple:
    """确保目录唯一。如果已存在，追加随机后缀。

    Returns:
        (unique_id, unique_dir)
    """
    dir_path = parent_dir / base_id
    if not dir_path.exists():
        return base_id, dir_path

    while dir_path.exists():
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        new_id = f"{base_id}_{rand}"
        dir_path = parent_dir / new_id

    return new_id, dir_path


# ========== Config Profile 管理 ==========

def create_config_profile(
    config_name: str,
    knowledge_base_version: str,
    workflow_version: str = "",
    changed_variable: str = "",
    retrieval_config: str = "",
    notes: str = "",
) -> dict:
    """创建新的配置方案。

    Returns:
        dict: 配置信息，包含 config_id 等
    """
    config_id = _generate_id("cfg", config_name)
    CONFIG_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    config_path = _ensure_unique_file(CONFIG_PROFILES_DIR / f"{config_id}.json")

    config = {
        "config_id": config_path.stem,
        "config_name": config_name,
        "knowledge_base_version": knowledge_base_version,
        "workflow_version": workflow_version,
        "changed_variable": changed_variable,
        "retrieval_config": retrieval_config,
        "notes": notes,
        "created_at": datetime.now().isoformat(),
    }

    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def load_config_profile(config_id: str) -> dict:
    """加载配置方案。"""
    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def update_config_profile(config_id: str, updates: dict) -> dict:
    """更新配置方案。"""
    config = load_config_profile(config_id)
    if config is None:
        raise ValueError(f"配置方案不存在: {config_id}")

    config.update(updates)

    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def list_config_profiles() -> list:
    """列出所有配置方案。"""
    if not CONFIG_PROFILES_DIR.exists():
        return []

    configs = []
    for config_path in sorted(CONFIG_PROFILES_DIR.glob("*.json"), reverse=True):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configs.append(config)
        except (json.JSONDecodeError, IOError):
            continue

    return configs


# ========== Experiment Run 管理 ==========

def create_experiment_run(
    config_id: str,
    question_set_source: str = "",
    question_count: int = 0,
) -> dict:
    """创建新的测试运行。

    Args:
        config_id: 关联的配置方案 ID
        question_set_source: 题目来源
        question_count: 题目数量

    Returns:
        dict: 运行信息，包含 run_id, manifest 等
    """
    # 加载配置快照
    config = load_config_profile(config_id)
    if config is None:
        raise ValueError(f"配置方案不存在: {config_id}")

    # 生成 run_id
    run_id = _generate_id("run", config.get("config_name", ""))

    # 确保目录唯一
    run_id, run_dir = _ensure_unique_dir(run_id, EXPERIMENTS_DIR)

    # 创建运行目录
    run_dir.mkdir(parents=True, exist_ok=False)

    # 构建 manifest（包含配置快照）
    manifest = {
        "run_id": run_id,
        "config_id": config_id,
        "config_snapshot": config,  # 配置快照，防止配置修改影响历史记录
        "question_set_source": question_set_source,
        "question_count": question_count,
        "started_at": datetime.now().isoformat(),
        "batch_results_file": None,
        "raw_results_file": None,
        "status": "created",
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "manifest": manifest,
    }


def load_experiment_run(run_id: str) -> dict:
    """加载运行记录。"""
    manifest_path = EXPERIMENTS_DIR / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def update_experiment_run(run_id: str, updates: dict) -> dict:
    """更新运行记录。"""
    manifest = load_experiment_run(run_id)
    if manifest is None:
        raise ValueError(f"运行记录不存在: {run_id}")

    manifest.update(updates)

    manifest_path = EXPERIMENTS_DIR / run_id / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def list_experiment_runs() -> list:
    """列出所有运行记录。"""
    if not EXPERIMENTS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                runs.append(manifest)
            except (json.JSONDecodeError, IOError):
                continue

    return runs


def list_runs_by_config(config_id: str) -> list:
    """列出指定配置方案的所有运行记录。"""
    all_runs = list_experiment_runs()
    return [r for r in all_runs if r.get("config_id") == config_id]


# ========== 工具函数 ==========

def ensure_question_id(question: dict) -> dict:
    """确保问题有 question_id。如果没有，生成一个。"""
    if not question.get("question_id"):
        import hashlib
        q_text = (question.get("question") or "").strip()
        if q_text:
            qid = hashlib.md5(q_text.encode("utf-8")).hexdigest()[:12]
        else:
            qid = datetime.now().strftime("%H%M%S%f")[:12]
        question["question_id"] = qid
    return question


def build_dify_user_field(run_id: str, question_id: str) -> str:
    """构建 Dify API 调用的 user 字段。

    格式: rag_eval:<run_id>:<question_id>
    """
    return f"rag_eval:{run_id}:{question_id}"
