"""
运行看板模块 - RAG 配置方案 + 测试运行记录。

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


# ========== 配置字段统一 Schema ==========

# A. 系统核心字段：不可在 UI 编辑
CONFIG_CORE_FIELDS = {"config_id", "created_at"}
RUN_CORE_FIELDS = {"run_id", "config_id", "question_set_id", "question_set_name",
                   "question_set_source", "batch_results_file", "raw_results_file",
                   "started_at", "status"}

# 配置字段 schema：key -> (label, required, widget, placeholder, help)
# widget: "text" | "textarea" | "number" | "select"
CONFIG_FIELD_SCHEMA = [
    ("config_name",          "配置名称",       True,  "text",     "例如：chunk_size 优化测试", "必填"),
    ("knowledge_base_version","知识库版本",    True,  "text",     "例如：fintech_kb_v2",       "必填，可填写自由文本"),
    ("workflow_version",     "工作流版本",     True,  "text",     "例如：chatflow_v2",         "必填"),
    ("source_description",   "文档/数据来源",  False, "text",     "例如：IS5010 期末复习 MD",  ""),
    ("chunk_strategy",       "分块策略",       False, "text",     "例如：按章节切分 / 512 tokens", ""),
    ("embedding_model",      "Embedding 模型", False, "text",     "例如：bge-large-zh-v1.5",   ""),
    ("retrieval_mode",       "检索模式",       False, "text",     "例如：hybrid / semantic",    ""),
    ("retrieval_config",     "检索配置说明",   False, "text",     "例如：hybrid / top_k=5 / reranker=on", ""),
    ("top_k",                "Top K",          False, "number",   "例如：5",                   "整数"),
    ("rerank_model",         "Rerank 模型",    False, "text",     "例如：bge-reranker-v2-m3",  ""),
    ("changed_variable",     "本次改动",       False, "text",     "例如：chunk_size: 1000 -> 500", ""),
    ("notes",                "备注",           False, "textarea", "其他需要记录的信息",         ""),
]

# B+C 可编辑字段集合（从 schema 自动生成）
CONFIG_REQUIRED_FIELDS = {f[0] for f in CONFIG_FIELD_SCHEMA if f[2]}
CONFIG_OPTIONAL_FIELDS = {f[0] for f in CONFIG_FIELD_SCHEMA if not f[2]}
CONFIG_EDITABLE_FIELDS = CONFIG_REQUIRED_FIELDS | CONFIG_OPTIONAL_FIELDS


def get_config_summary(config: dict) -> str:
    """生成配置摘要文本，用于列表显示。"""
    parts = []
    parts.append(config.get("config_name", "未命名"))
    parts.append(config.get("knowledge_base_version", "") or "未记录")
    mode = config.get("retrieval_mode", "")
    topk = config.get("top_k", "")
    rerank = config.get("rerank_model", "")
    extras = []
    if mode:
        extras.append(mode)
    if topk:
        extras.append(f"top_k={topk}")
    if rerank:
        extras.append(f"rerank={rerank}")
    if extras:
        parts.append(" / ".join(extras))
    return " | ".join(parts)


def get_config_display_value(config: dict, key: str) -> str:
    """获取配置字段的显示值，空值返回'未记录'。"""
    val = config.get(key)
    if val is None or str(val).strip() == "":
        return "未记录"
    return str(val)


def _protect_core_fields(existing: dict, updates: dict, core_fields: set) -> dict:
    """从 updates 中移除核心字段，返回安全的更新字典。"""
    safe = {}
    for k, v in updates.items():
        if k in core_fields:
            continue  # 跳过核心字段
        safe[k] = v
    return safe


# ========== Config Profile 管理 ==========

def create_config_profile(
    config_name: str,
    knowledge_base_version: str,
    workflow_version: str = "",
    changed_variable: str = "",
    retrieval_config: str = "",
    notes: str = "",
    **kwargs,
) -> dict:
    """创建新的配置方案。

    必填：config_name, knowledge_base_version, workflow_version
    其余均为可选，允许为空。

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

    # 可选实验字段：只写入非空值，缺失字段不自动填充默认值
    for field in CONFIG_OPTIONAL_FIELDS:
        val = kwargs.get(field)
        if val is not None and str(val).strip():
            config[field] = val

    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def load_config_profile(config_id: str) -> dict:
    """加载配置方案。"""
    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def update_config_profile(config_id: str, updates: dict) -> dict:
    """更新配置方案（内部使用，不校验字段）。"""
    config = load_config_profile(config_id)
    if config is None:
        raise ValueError(f"配置方案不存在: {config_id}")

    config.update(updates)

    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def update_config_profile_safe(config_id: str, updates: dict, edit_note: str = "") -> dict:
    """安全编辑配置方案：仅允许编辑描述性字段，保护核心字段。

    Args:
        config_id: 配置方案 ID
        updates: 要更新的字段（核心字段会被自动忽略）
        edit_note: 可选的修改说明

    Returns:
        dict: 更新后的配置
    """
    config = load_config_profile(config_id)
    if config is None:
        raise ValueError(f"配置方案不存在: {config_id}")

    # 只保留可编辑字段
    safe_updates = _protect_core_fields(config, updates, CONFIG_CORE_FIELDS)
    config.update(safe_updates)
    config["updated_at"] = datetime.now().isoformat()
    if edit_note:
        config["edit_note"] = edit_note

    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def list_config_profiles(include_archived: bool = False) -> list:
    """列出配置方案。

    Args:
        include_archived: 是否包含已归档的重复配置。默认 False。
    """
    if not CONFIG_PROFILES_DIR.exists():
        return []

    configs = []
    for config_path in sorted(CONFIG_PROFILES_DIR.glob("*.json"), reverse=True):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not include_archived and config.get("archived_duplicate"):
                continue
            configs.append(config)
        except (json.JSONDecodeError, IOError):
            continue

    return configs


# ========== 配置去重与指纹 ==========

# 指纹计算中排除的字段（非业务字段或敏感字段）
_FP_EXCLUDE_KEYS = frozenset({
    "config_id", "created_at", "updated_at", "edit_note", "archived_duplicate",
    "config_snapshot", "run_id", "question_set_id", "question_set_name",
    # 敏感字段
    "api_key", "secret_key", "lf_public_key", "lf_secret_key",
    "openai_api_key", "api_keys", "cookie", "session_token", "password", "token",
})


def config_fingerprint(config: dict) -> str:
    """计算配置的业务内容指纹。

    排除 config_id、created_at 等非业务字段和所有敏感字段。
    空值字段不参与指纹计算。
    字典 key 顺序无关。
    """
    import hashlib
    biz = {}
    for k, v in sorted(config.items()):
        if k in _FP_EXCLUDE_KEYS:
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        # 数值统一为字符串
        if isinstance(v, (int, float)):
            biz[k] = str(v)
        else:
            biz[k] = str(v).strip()
    raw = json.dumps(biz, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def find_duplicate_config_groups(configs: list = None) -> dict:
    """按 fingerprint 分组找出重复配置。

    Returns:
        dict: {fingerprint: [config_dict, ...]} 只包含有重复的组（>=2 个）。
    """
    if configs is None:
        configs = list_config_profiles()

    groups = {}
    for cfg in configs:
        if cfg.get("archived_duplicate"):
            continue  # 已归档的不参与分组
        fp = config_fingerprint(cfg)
        groups.setdefault(fp, []).append(cfg)

    return {fp: cfgs for fp, cfgs in groups.items() if len(cfgs) >= 2}


def find_canonical_config(fingerprint: str, configs: list = None) -> dict:
    """找到 fingerprint 对应的 canonical（最早创建）配置。"""
    if configs is None:
        configs = list_config_profiles()

    candidates = [
        cfg for cfg in configs
        if not cfg.get("archived_duplicate") and config_fingerprint(cfg) == fingerprint
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.get("created_at", ""))


def cleanup_duplicate_configs(dry_run: bool = True) -> dict:
    """清理重复配置：将非 canonical 的标记为 archived_duplicate。

    Args:
        dry_run: True 时只分析不实际修改。

    Returns:
        dict: {
            "groups": int,           # 重复组数
            "canonical": int,        # 保留的 canonical 数
            "archived": int,         # 已归档数（之前已标记）
            "newly_archived": int,   # 本次新归档数
            "deleted": int,          # 安全删除数（无 run 引用的非 canonical）
            "details": [{fingerprint, canonical_id, archived_ids, deleted_ids}],
        }
    """
    configs = list_config_profiles()
    groups = find_duplicate_config_groups(configs)

    # 收集所有被 run 引用的 config_id
    referenced_ids = set()
    if EXPERIMENTS_DIR.exists():
        for run_dir in EXPERIMENTS_DIR.iterdir():
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    cid = manifest.get("config_id")
                    if cid:
                        referenced_ids.add(cid)
                    # config_snapshot 中也可能有 config_id
                    snap = manifest.get("config_snapshot") or {}
                    cid2 = snap.get("config_id")
                    if cid2:
                        referenced_ids.add(cid2)
                except (json.JSONDecodeError, IOError):
                    continue

    result = {
        "groups": len(groups),
        "canonical": 0,
        "archived": 0,
        "newly_archived": 0,
        "deleted": 0,
        "details": [],
    }

    for fp, cfgs in groups.items():
        # 按创建时间排序，最早的为 canonical
        sorted_cfgs = sorted(cfgs, key=lambda c: c.get("created_at", ""))
        canonical = sorted_cfgs[0]
        duplicates = sorted_cfgs[1:]
        result["canonical"] += 1

        detail = {
            "fingerprint": fp,
            "canonical_id": canonical["config_id"],
            "canonical_name": canonical.get("config_name", ""),
            "archived_ids": [],
            "already_archived_ids": [],
            "deleted_ids": [],
        }

        for dup in duplicates:
            dup_id = dup["config_id"]
            if dup.get("archived_duplicate"):
                result["archived"] += 1
                detail["already_archived_ids"].append(dup_id)
                continue

            if dup_id in referenced_ids:
                # 被 run 引用，只能归档
                if not dry_run:
                    _mark_config_archived(dup_id)
                result["newly_archived"] += 1
                detail["archived_ids"].append(dup_id)
            else:
                # 无 run 引用，可以安全删除
                if not dry_run:
                    _delete_config(dup_id)
                result["deleted"] += 1
                detail["deleted_ids"].append(dup_id)

        result["details"].append(detail)

    return result


def merge_duplicate_configs(dry_run: bool = True) -> dict:
    """合并重复配置：将非 canonical 配置关联的 run 重新绑定到 canonical，然后删除重复配置。

    流程：
    1. 按 fingerprint 分组，最早创建者为 canonical。
    2. 找出所有 manifest 中 config_id 属于非 canonical 的 run。
    3. 将这些 run 的 manifest config_id 更新为 canonical config_id。
    4. 校验每个迁移后的 run 仍可正常读取。
    5. 全部成功后删除非 canonical 配置文件。

    Args:
        dry_run: True 时只分析不实际修改。

    Returns:
        dict: {
            "groups": int,
            "runs_migrated": int,
            "configs_deleted": int,
            "validation_failures": list[str],
            "details": [{canonical_id, canonical_name, dup_ids, migrated_run_ids, deleted_config_ids, errors}],
        }
    """
    all_configs = list_config_profiles(include_archived=True)
    groups = find_duplicate_config_groups(all_configs)

    # 建立 config_id -> config 映射
    config_map = {c["config_id"]: c for c in all_configs}

    # 扫描所有 run，建立 config_id -> [run_id, ...] 映射
    config_to_runs = {}
    if EXPERIMENTS_DIR.exists():
        for run_dir in EXPERIMENTS_DIR.iterdir():
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    cid = manifest.get("config_id")
                    if cid:
                        config_to_runs.setdefault(cid, []).append(manifest.get("run_id", run_dir.name))
                except (json.JSONDecodeError, IOError):
                    continue

    result = {
        "groups": len(groups),
        "runs_migrated": 0,
        "configs_deleted": 0,
        "validation_failures": [],
        "details": [],
    }

    for fp, cfgs in groups.items():
        sorted_cfgs = sorted(cfgs, key=lambda c: c.get("created_at", ""))
        canonical = sorted_cfgs[0]
        canonical_id = canonical["config_id"]
        duplicates = sorted_cfgs[1:]

        detail = {
            "canonical_id": canonical_id,
            "canonical_name": canonical.get("config_name", ""),
            "dup_ids": [],
            "migrated_run_ids": [],
            "deleted_config_ids": [],
            "errors": [],
        }

        for dup in duplicates:
            dup_id = dup["config_id"]
            detail["dup_ids"].append(dup_id)

            # 找出引用此重复配置的 run
            run_ids = config_to_runs.get(dup_id, [])

            if not dry_run:
                # 迁移每个 run
                for run_id in run_ids:
                    try:
                        update_experiment_run(run_id, {"config_id": canonical_id})
                        result["runs_migrated"] += 1
                        detail["migrated_run_ids"].append(run_id)
                    except Exception as e:
                        detail["errors"].append(f"run {run_id}: {e}")

            else:
                result["runs_migrated"] += len(run_ids)
                detail["migrated_run_ids"].extend(run_ids)

        # 校验：确认所有迁移后的 run 仍可读取
        if not dry_run and not detail["errors"]:
            all_ok = True
            for run_id in detail["migrated_run_ids"]:
                m = load_experiment_run(run_id)
                if m is None:
                    detail["errors"].append(f"校验失败: run {run_id} 无法加载")
                    result["validation_failures"].append(run_id)
                    all_ok = False
                elif m.get("config_id") != canonical_id:
                    detail["errors"].append(
                        f"校验失败: run {run_id} config_id={m.get('config_id')} != {canonical_id}"
                    )
                    result["validation_failures"].append(run_id)
                    all_ok = False

            # 全部校验通过后才删除重复配置
            if all_ok:
                for dup in duplicates:
                    dup_id = dup["config_id"]
                    # 再次确认无 run 引用此 config
                    remaining = config_to_runs.get(dup_id, [])
                    # 迁移后应该没有 run 引用它了，但再检查一次
                    still_referenced = False
                    for rid in remaining:
                        m = load_experiment_run(rid)
                        if m and m.get("config_id") == dup_id:
                            still_referenced = True
                            break
                    if not still_referenced:
                        _delete_config(dup_id)
                        result["configs_deleted"] += 1
                        detail["deleted_config_ids"].append(dup_id)
                    else:
                        detail["errors"].append(f"跳过删除: {dup_id} 仍有 run 引用")

        result["details"].append(detail)

    return result
    """将配置标记为 archived_duplicate。"""
    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["archived_duplicate"] = True
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    except (json.JSONDecodeError, IOError):
        pass


def _delete_config(config_id: str):
    """安全删除配置文件（仅在确认无 run 引用后调用）。"""
    config_path = CONFIG_PROFILES_DIR / f"{config_id}.json"
    if config_path.exists():
        config_path.unlink()


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
    """更新运行记录（内部使用，不校验字段）。"""
    manifest = load_experiment_run(run_id)
    if manifest is None:
        raise ValueError(f"运行记录不存在: {run_id}")

    manifest.update(updates)

    manifest_path = EXPERIMENTS_DIR / run_id / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def update_run_snapshot(run_id: str, snapshot_updates: dict, edit_note: str = "") -> dict:
    """安全编辑某次运行的配置快照：仅编辑描述性字段，保护核心关联字段。

    保存前将旧快照存入 snapshot_edit_history 以便追溯。

    Args:
        run_id: 运行 ID
        snapshot_updates: 要更新的 config_snapshot 字段（核心字段会被忽略）
        edit_note: 可选的修改说明

    Returns:
        dict: 更新后的 manifest
    """
    manifest = load_experiment_run(run_id)
    if manifest is None:
        raise ValueError(f"运行记录不存在: {run_id}")

    snapshot = dict(manifest.get("config_snapshot", {}))

    # 保存修改前快照
    if "snapshot_edit_history" not in manifest:
        manifest["snapshot_edit_history"] = []
    manifest["snapshot_edit_history"].append({
        "before": snapshot.copy(),
        "edited_at": datetime.now().isoformat(),
        "edit_note": edit_note or "手动修正配置记录",
    })

    # 只保留可编辑字段，跳过核心字段
    safe_updates = _protect_core_fields(snapshot, snapshot_updates, CONFIG_CORE_FIELDS)
    snapshot.update(safe_updates)
    snapshot["snapshot_updated_at"] = datetime.now().isoformat()
    if edit_note:
        snapshot["snapshot_edit_note"] = edit_note

    manifest["config_snapshot"] = snapshot

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


def parse_rag_eval_user_id(user_id: str) -> dict:
    """解析 rag_eval:<run_id>:<question_id> 格式的 user_id。

    Returns:
        dict: {"run_id": str, "question_id": str} 或空 dict
    """
    if not user_id or not isinstance(user_id, str):
        return {}

    if user_id.startswith("rag_eval:"):
        parts = user_id.split(":", 2)
        if len(parts) == 3:
            return {
                "run_id": parts[1],
                "question_id": parts[2],
            }

    return {}


def backfill_manifest_from_batch(run_id: str, batch_dir=None) -> bool:
    """从 batch 文件回填 manifest 中的题集信息。

    Returns:
        bool: 是否成功回填
    """
    from pathlib import Path

    manifest = load_experiment_run(run_id)
    if manifest is None:
        return False

    # 如果已有 question_set_id，跳过
    if manifest.get("question_set_id"):
        return False

    batch_file = manifest.get("batch_results_file")
    if not batch_file:
        return False

    # 优先查找 run 目录，fallback 到全局 batch 目录
    batch_path = None
    run_local = EXPERIMENTS_DIR / run_id / batch_file
    if run_local.exists():
        batch_path = run_local
    elif batch_dir:
        global_path = Path(batch_dir) / batch_file
        if global_path.exists():
            batch_path = global_path
    if batch_path is None:
        return False

    # 从 batch 文件读取第一条成功记录的题集信息
    try:
        with batch_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sample = obj.get("sample", {})
                if sample.get("question_set_id"):
                    updates = {
                        "question_set_id": sample["question_set_id"],
                        "question_set_name": sample.get("question_set_name", ""),
                    }
                    if sample.get("run_id"):
                        updates["run_id_in_sample"] = sample["run_id"]
                    if sample.get("config_id"):
                        updates["config_id_in_sample"] = sample["config_id"]
                    update_experiment_run(run_id, updates)
                    return True
    except (json.JSONDecodeError, IOError):
        pass

    return False


def migrate_judged_results(processed_file=None, judged_file=None, backup=True) -> dict:
    """为 judged results 回填 run_id、config_id、question_set_id 等元数据。

    通过 processed sample 的 trace_id 匹配 judged results。

    Args:
        processed_file: processed samples JSONL 文件路径
        judged_file: judged results JSONL 文件路径
        backup: 是否在迁移前创建备份

    Returns:
        dict: {"migrated": int, "backup_path": str}
    """
    from pathlib import Path
    from datetime import datetime

    if not processed_file or not judged_file:
        return {"migrated": 0, "backup_path": ""}

    processed_path = Path(processed_file)
    judged_path = Path(judged_file)

    if not processed_path.exists() or not judged_path.exists():
        return {"migrated": 0, "backup_path": ""}

    # 构建 trace_id -> 元数据映射（从 processed samples）
    trace_metadata = {}
    try:
        with processed_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = obj.get("trace_id", "")
                if tid and obj.get("run_id"):
                    trace_metadata[tid] = {
                        "run_id": obj.get("run_id", ""),
                        "config_id": obj.get("config_id", ""),
                        "question_id": obj.get("question_id", ""),
                        "question_set_id": obj.get("question_set_id", ""),
                        "question_set_name": obj.get("question_set_name", ""),
                    }
    except (json.JSONDecodeError, IOError):
        return {"migrated": 0, "backup_path": ""}

    if not trace_metadata:
        return {"migrated": 0, "backup_path": ""}

    # 创建备份
    backup_path = ""
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = str(judged_path.parent / f"{judged_path.stem}_backup_{ts}.jsonl")
        import shutil
        shutil.copy2(judged_path, backup_path)

    # 读取并更新 judged results
    updated_results = []
    migrated_count = 0
    try:
        with judged_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = obj.get("trace_id", "")

                # 如果已有 run_id，跳过
                if obj.get("run_id"):
                    updated_results.append(obj)
                    continue

                # 从 trace_metadata 回填
                if tid in trace_metadata:
                    meta = trace_metadata[tid]
                    obj["run_id"] = meta["run_id"]
                    if meta["config_id"]:
                        obj["config_id"] = meta["config_id"]
                    if meta["question_id"]:
                        obj["question_id"] = meta["question_id"]
                    if meta["question_set_id"]:
                        obj["question_set_id"] = meta["question_set_id"]
                    if meta["question_set_name"]:
                        obj["question_set_name"] = meta["question_set_name"]
                    migrated_count += 1

                updated_results.append(obj)
    except (json.JSONDecodeError, IOError):
        return {"migrated": 0, "backup_path": backup_path}

    # 写回文件
    with judged_path.open("w", encoding="utf-8") as f:
        for obj in updated_results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {"migrated": migrated_count, "backup_path": backup_path}


def migrate_processed_samples(processed_file=None, experiments_dir=None, backup=True) -> dict:
    """为 processed samples 回填 config_id、question_set_id 等元数据。

    通过 run_id 读取 manifest，回填缺失的字段。

    Args:
        processed_file: processed samples JSONL 文件路径
        experiments_dir: 实验目录路径
        backup: 是否在迁移前创建备份

    Returns:
        dict: {"migrated": int, "backup_path": str}
    """
    from pathlib import Path
    from datetime import datetime

    if not processed_file:
        return {"migrated": 0, "backup_path": ""}

    processed_path = Path(processed_file)
    if not processed_path.exists():
        return {"migrated": 0, "backup_path": ""}

    # 创建备份
    backup_path = ""
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = str(processed_path.parent / f"{processed_path.stem}_backup_{ts}.jsonl")
        import shutil
        shutil.copy2(processed_path, backup_path)

    # 读取并更新 processed samples
    updated_samples = []
    migrated_count = 0
    manifest_cache = {}

    try:
        with processed_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)

                # 获取 run_id
                run_id = obj.get("run_id", "")
                if not run_id:
                    user_id = obj.get("user_id", "")
                    if user_id.startswith("rag_eval:"):
                        parts = user_id.split(":", 2)
                        if len(parts) == 3:
                            run_id = parts[1]
                            obj["run_id"] = run_id

                # 如果有 run_id 但缺少 config_id 等字段，从 manifest 回填
                if run_id and not obj.get("config_id"):
                    # 缓存 manifest
                    if run_id not in manifest_cache:
                        manifest_cache[run_id] = load_experiment_run(run_id)

                    manifest = manifest_cache[run_id]
                    if manifest:
                        if not obj.get("config_id"):
                            obj["config_id"] = manifest.get("config_id", "")
                        if not obj.get("question_set_id"):
                            obj["question_set_id"] = manifest.get("question_set_id", "")
                        if not obj.get("question_set_name"):
                            obj["question_set_name"] = manifest.get("question_set_name", "")
                        migrated_count += 1

                updated_samples.append(obj)
    except (json.JSONDecodeError, IOError):
        return {"migrated": 0, "backup_path": backup_path}

    # 写回文件
    with processed_path.open("w", encoding="utf-8") as f:
        for obj in updated_samples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {"migrated": migrated_count, "backup_path": backup_path}


def get_run_status(run_id: str, batch_dir=None, raw_dir=None,
                   processed_file=None, judged_file=None,
                   include_judge_results=False) -> dict:
    """获取运行状态统计。

    Args:
        include_judge_results: If True, include full judge_results list in the
            return dict. Default False to reduce memory usage when only counts
            are needed.

    Returns:
        dict: {
            "batch_success": int, "batch_total": int,
            "raw_count": int, "processed_count": int,
            "judge_count": int, "question_count": int,
            "question_set_id": str, "question_set_name": str,
            "judge_results": list (empty if include_judge_results=False),
        }
    """
    from pathlib import Path

    manifest = load_experiment_run(run_id)
    if manifest is None:
        return {}

    question_count = manifest.get("question_count", 0)
    question_set_id = manifest.get("question_set_id", "")
    question_set_name = manifest.get("question_set_name", "")

    # Batch 状态（优先查找 run 目录，fallback 到全局 batch 目录）
    batch_success = 0
    batch_total = 0
    batch_file = manifest.get("batch_results_file")
    batch_path = None
    if batch_file:
        run_local = EXPERIMENTS_DIR / run_id / batch_file
        if run_local.exists():
            batch_path = run_local
        elif batch_dir:
            global_path = Path(batch_dir) / batch_file
            if global_path.exists():
                batch_path = global_path
    if batch_path is not None:
        try:
            with batch_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    batch_total += 1
                    if obj.get("success"):
                        batch_success += 1
                    # 从 batch 中提取题集信息（如果 manifest 没有）
                    if not question_set_id:
                        sample = obj.get("sample", {})
                        if sample.get("question_set_id"):
                            question_set_id = sample["question_set_id"]
                            question_set_name = sample.get("question_set_name", "")
        except (json.JSONDecodeError, IOError):
            pass

    # Raw 状态
    raw_count = 0
    raw_file = manifest.get("raw_results_file")
    if raw_file and raw_dir:
        raw_path = Path(raw_dir) / raw_file
        if raw_path.exists():
            try:
                with raw_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        raw_count += 1
                        # 从 raw 中提取题集信息（如果 manifest 没有）
                        if not question_set_id:
                            obj = json.loads(line)
                            if obj.get("question_set_id"):
                                question_set_id = obj["question_set_id"]
                                question_set_name = obj.get("question_set_name", "")
            except (json.JSONDecodeError, IOError):
                pass

    # Processed 状态（从 processed 文件中按 run_id 统计）
    processed_count = 0
    if processed_file and Path(processed_file).exists():
        try:
            with Path(processed_file).open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    # 通过 run_id 或 user_id 中的 run_id 关联
                    sample_run_id = obj.get("run_id", "")
                    if not sample_run_id:
                        user_id = obj.get("user_id", "")
                        if user_id.startswith("rag_eval:"):
                            parts = user_id.split(":", 2)
                            if len(parts) == 3:
                                sample_run_id = parts[1]
                    if sample_run_id == run_id:
                        processed_count += 1
        except (json.JSONDecodeError, IOError):
            pass

    # Judge 状态（通过 processed sample 的 trace_id 匹配 judged results）
    judge_count = 0
    judge_results_for_run = []

    # 第一步：从 processed samples 找出当前 run 的所有真实 Langfuse trace_id
    processed_trace_ids = set()
    if processed_file and Path(processed_file).exists():
        try:
            with Path(processed_file).open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    # 通过 run_id 或 user_id 中的 run_id 关联
                    sample_run_id = obj.get("run_id", "")
                    if not sample_run_id:
                        user_id = obj.get("user_id", "")
                        if user_id.startswith("rag_eval:"):
                            parts = user_id.split(":", 2)
                            if len(parts) == 3:
                                sample_run_id = parts[1]
                    if sample_run_id == run_id:
                        tid = obj.get("trace_id", "")
                        if tid:
                            processed_trace_ids.add(tid)
        except (json.JSONDecodeError, IOError):
            pass

    # 第二步：用 processed trace_id 匹配 judged results
    if judged_file and Path(judged_file).exists() and processed_trace_ids:
        try:
            with Path(judged_file).open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    tid = obj.get("trace_id", "")
                    # 通过 run_id 关联（新格式）
                    if obj.get("run_id") == run_id:
                        judge_count += 1
                        judge_results_for_run.append(obj)
                    # 通过 processed trace_id 关联（真实 Langfuse UUID）
                    elif tid in processed_trace_ids:
                        judge_count += 1
                        judge_results_for_run.append(obj)
        except (json.JSONDecodeError, IOError):
            pass

    return {
        "batch_success": batch_success,
        "batch_total": batch_total,
        "raw_count": raw_count,
        "processed_count": processed_count,
        "judge_count": judge_count,
        "question_count": question_count,
        "question_set_id": question_set_id,
        "question_set_name": question_set_name,
        "judge_results": judge_results_for_run if include_judge_results else [],
    }


def get_judge_metrics_by_run(judge_results: list, run_id: str) -> dict:
    """按 run_id 过滤 Judge 结果并计算指标。

    不创建新的评分公式，复用现有 compute_metrics。
    """
    from judge import compute_metrics, TRACK_RETRIEVAL, TRACK_STRICT_QA, TRACK_GROUNDED_QA

    # 过滤属于该 run 的结果（严格按 run_id 匹配）
    run_results = []
    for r in judge_results:
        if r.get("run_id") == run_id:
            run_results.append(r)

    if not run_results:
        return None

    # 复用现有 compute_metrics
    metrics = compute_metrics(run_results)

    # 添加按轨道分组的指标
    valid = [r for r in run_results if "error" not in r]
    retrieval_results = [r for r in valid if r.get("evaluation_track") == TRACK_RETRIEVAL]
    strict_qa_results = [r for r in valid if r.get("evaluation_track") == TRACK_STRICT_QA]
    grounded_qa_results = [r for r in valid if r.get("evaluation_track") == TRACK_GROUNDED_QA]

    metrics["retrieval_count"] = len(retrieval_results)
    metrics["strict_qa_count"] = len(strict_qa_results)
    metrics["grounded_qa_count"] = len(grounded_qa_results)

    return metrics
