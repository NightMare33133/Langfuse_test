"""
Langfuse 项目管理模块 — 按 Project 管理 trace 同步，不按 API Key。

功能：
- 项目识别（host + public_key → stable project_id）
- 项目注册表（registry.json）
- 增量同步（fromTimestamp 游标）
- trace_id 去重
- 项目统计
- 旧导出文件清理候选

存储结构：
  data/langfuse_projects/<project_id>/
    registry.json          — 项目元数据 + 同步游标
    traces.jsonl.gz        — 累积 trace 存储（gzip，按 trace_id 去重）
    snapshots/             — 用户显式创建的 frozen snapshot

安全规则：
- API Key 绝不写入 registry、日志或报告
- 仅保存 key_masked 和 key_source
"""

import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECTS_DIR = Path(__file__).parent / "data" / "langfuse_projects"
RAW_DIR = Path(__file__).parent / "data" / "raw"
DATA_DIR = Path(__file__).parent / "data"


# ── 项目识别 ─────────────────────────────────────────────────


def generate_project_id(host: str, public_key: str) -> str:
    """从 host + public_key 生成稳定的 project_id。

    使用 SHA-256 前 16 位，确保同一项目跨 Key 识别。
    """
    normalized_host = host.strip().rstrip("/").lower()
    raw = f"{normalized_host}|{public_key.strip()}"
    return "proj_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def identify_project(host: str, public_key: str, secret_key: str,
                     timeout: int = 10) -> dict:
    """识别 Langfuse 项目。

    调用 API 测试连接并获取项目信息。

    Returns:
        {"project_id", "project_name", "host", "key_masked", "total_traces"}

    Raises:
        RuntimeError: 连接失败或认证错误
    """
    host = host.strip().rstrip("/")
    url = f"{host}/api/public/traces"
    try:
        resp = requests.get(
            url, auth=(public_key.strip(), secret_key.strip()),
            params={"limit": 1}, timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"连接失败: {type(e).__name__}: {e}") from e

    if resp.status_code == 401:
        raise RuntimeError("认证失败，请检查 Public Key 和 Secret Key")
    if resp.status_code == 403:
        raise RuntimeError("权限不足，请检查 Key 权限")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError("API 响应解析失败")

    total_traces = data.get("meta", {}).get("totalItems", 0)
    project_id = generate_project_id(host, public_key)

    return {
        "project_id": project_id,
        "project_name": f"Langfuse@{host.split('//')[-1].split(':')[0]}",
        "host": host,
        "key_masked": _mask_key(public_key),
        "total_traces": total_traces,
    }


def _mask_key(key: str) -> str:
    if not key or len(key) < 12:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


# ── 项目注册表 ───────────────────────────────────────────────


def _project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def _registry_path(project_id: str) -> Path:
    return _project_dir(project_id) / "registry.json"


def _traces_path(project_id: str) -> Path:
    """TRACE 行存储（索引，用于去重和列表）。"""
    return _project_dir(project_id) / "traces.jsonl.gz"


def _obs_path(project_id: str) -> Path:
    """OBSERVATION 行存储（证据，用于解析 retrieval）。"""
    return _project_dir(project_id) / "observations.jsonl.gz"


def list_projects() -> list:
    """列出所有已注册项目。"""
    if not PROJECTS_DIR.exists():
        return []
    projects = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        reg_path = d / "registry.json"
        if d.is_dir() and reg_path.exists():
            try:
                reg = json.loads(reg_path.read_text(encoding="utf-8"))
                reg["_dir"] = str(d)
                reg["_traces_size"] = _traces_path(d.name).stat().st_size if _traces_path(d.name).exists() else 0
                projects.append(reg)
            except Exception:
                continue
    return projects


def load_project(project_id: str) -> dict | None:
    """加载项目注册信息。"""
    rp = _registry_path(project_id)
    if not rp.exists():
        return None
    return json.loads(rp.read_text(encoding="utf-8"))


def register_project(project_id: str, project_name: str, host: str,
                     key_masked: str) -> dict:
    """注册新项目（或更新已有项目）。"""
    pdir = _project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "snapshots").mkdir(exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    existing = load_project(project_id)

    if existing:
        existing["project_name"] = project_name
        existing["host"] = host
        existing["key_masked"] = key_masked
        existing["updated_at"] = now
        _save_registry(project_id, existing)
        return existing

    registry = {
        "project_id": project_id,
        "project_name": project_name,
        "host": host,
        "key_masked": key_masked,
        "last_sync_at": None,
        "last_trace_timestamp": None,
        "total_traces_synced": 0,
        "snapshot_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _save_registry(project_id, registry)
    return registry


def _save_registry(project_id: str, registry: dict):
    rp = _registry_path(project_id)
    rp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


# ── trace_id 去重 ───────────────────────────────────────────


def load_existing_trace_ids(project_id: str) -> set:
    """从 traces.jsonl.gz 加载已存储的 trace_id（仅 TRACE 类型行）。

    注意：不收集 observation 的 id，否则会导致 observation 被去重跳过。
    """
    tp = _traces_path(project_id)
    if not tp.exists():
        return set()
    trace_ids = set()
    try:
        with gzip.open(tp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    # 仅收集 TRACE 类型的 id，不收集 observation 的 traceId
                    if row.get("type") == "TRACE":
                        tid = row.get("id") or row.get("traceId")
                        if tid:
                            trace_ids.add(tid)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return trace_ids


def load_existing_obs_ids(project_id: str) -> set:
    """从 observations.jsonl.gz 加载已存储的 observation id。"""
    op = _obs_path(project_id)
    if not op.exists():
        return set()
    obs_ids = set()
    try:
        with gzip.open(op, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    oid = row.get("id")
                    if oid:
                        obs_ids.add(oid)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return obs_ids


def append_traces(project_id: str, rows: list[dict],
                  existing_ids: set = None) -> tuple[int, int]:
    """追加 TRACE 行到 traces.jsonl.gz（按 trace id 去重）。

    仅处理 type=="TRACE" 的行。observation 行由 append_observations 处理。

    Args:
        project_id: 项目 ID
        rows: 要追加的行列表（混合 TRACE 和 OBSERVATION）
        existing_ids: 已有 trace_id 集合（可选，传入避免重复加载）

    Returns:
        (appended_count, skipped_count)
    """
    if existing_ids is None:
        existing_ids = load_existing_trace_ids(project_id)

    tp = _traces_path(project_id)
    tp.parent.mkdir(parents=True, exist_ok=True)

    appended = 0
    skipped = 0
    with gzip.open(tp, "at", encoding="utf-8") as f:
        for row in rows:
            if row.get("type") != "TRACE":
                continue  # 只处理 TRACE 行
            tid = row.get("id") or row.get("traceId")
            if tid and tid in existing_ids:
                skipped += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if tid:
                existing_ids.add(tid)
            appended += 1

    return appended, skipped


def append_observations(project_id: str, rows: list[dict],
                        existing_obs_ids: set = None) -> tuple[int, int]:
    """追加 OBSERVATION 行到 observations.jsonl.gz（按 observation id 去重）。

    Args:
        project_id: 项目 ID
        rows: 要追加的行列表（混合 TRACE 和 OBSERVATION）
        existing_obs_ids: 已有 observation id 集合（可选）

    Returns:
        (appended_count, skipped_count)
    """
    if existing_obs_ids is None:
        existing_obs_ids = load_existing_obs_ids(project_id)

    op = _obs_path(project_id)
    op.parent.mkdir(parents=True, exist_ok=True)

    appended = 0
    skipped = 0
    with gzip.open(op, "at", encoding="utf-8") as f:
        for row in rows:
            if row.get("type") == "TRACE":
                continue  # 只处理非 TRACE 行（OBSERVATION / GENERATION / SPAN 等）
            oid = row.get("id")
            if oid and oid in existing_obs_ids:
                skipped += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if oid:
                existing_obs_ids.add(oid)
            appended += 1

    return appended, skipped


# ── 增量同步 ─────────────────────────────────────────────────


def incremental_sync(project_id: str, host: str, public_key: str,
                     secret_key: str, limit: int = 50, max_pages: int = 50,
                     from_timestamp: str = None,
                     progress_callback=None,
                     max_retries: int = 2,
                     force_full: bool = False) -> dict:
    """增量同步 trace 到项目存储。

    Args:
        project_id: 项目 ID
        host: Langfuse host
        public_key: Public Key
        secret_key: Secret Key
        limit: 每页 trace 数
        max_pages: 最大页数
        from_timestamp: 起始时间（ISO8601），None 则从上次游标继续
        progress_callback: (phase, new_traces, skipped, pages, total)
        max_retries: 每页最大重试
        force_full: True 时忽略 last_trace_timestamp 游标，强制全量拉取

    Returns:
        {"new_traces", "skipped", "new_observations", "skipped_observations",
         "pages", "last_timestamp", "elapsed"}
    """
    registry = load_project(project_id)
    if not registry:
        proj_info = identify_project(host, public_key, secret_key)
        registry = register_project(
            project_id, proj_info["project_name"],
            host, proj_info["key_masked"],
        )

    # 确定起始时间
    if force_full:
        from_timestamp = from_timestamp  # 显式传入则用，否则 None = 不过滤
    elif from_timestamp is None:
        from_timestamp = registry.get("last_trace_timestamp")

    existing_ids = load_existing_trace_ids(project_id)
    existing_obs_ids = load_existing_obs_ids(project_id)

    total_new = 0
    total_skipped = 0
    total_obs_new = 0
    total_obs_skipped = 0
    pages_done = 0
    max_ts = from_timestamp
    t0 = time.time()

    if progress_callback:
        progress_callback("connecting", 0, 0, 0, None)

    host = host.strip().rstrip("/")
    cumulative_retries = 0

    for page in range(1, max_pages + 1):
        # 带重试的单页请求
        data = None
        for attempt in range(max_retries + 1):
            try:
                params = {"limit": limit, "page": page}
                if from_timestamp:
                    params["fromTimestamp"] = from_timestamp
                resp = requests.get(
                    f"{host}/api/public/traces",
                    auth=(public_key.strip(), secret_key.strip()),
                    params=params, timeout=30,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                break
            except Exception:
                if attempt < max_retries:
                    cumulative_retries += 1
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise

        traces = data.get("data", []) if data else []
        if not traces:
            pages_done = page
            break

        # 构建行并追加
        rows = []
        page_max_ts = None
        for trace in traces:
            trace_id = trace["id"]
            ts = trace.get("timestamp", "")
            if ts and (page_max_ts is None or ts > page_max_ts):
                page_max_ts = ts

            rows.append({
                "id": trace_id,
                "traceId": trace_id,
                "type": "TRACE",
                "name": trace.get("name"),
                "startTime": trace.get("timestamp"),
                "endTime": trace.get("timestamp"),
                "input": trace.get("input"),
                "output": trace.get("output"),
                "metadata": trace.get("metadata"),
                "sessionId": trace.get("sessionId"),
                "userId": trace.get("userId"),
                "traceName": trace.get("name"),
                "providedModelName": None,
            })

            # 获取 observations
            try:
                obs_resp = requests.get(
                    f"{host}/api/public/observations",
                    auth=(public_key.strip(), secret_key.strip()),
                    params={"traceId": trace_id, "limit": 100},
                    timeout=30,
                )
                if obs_resp.status_code == 200:
                    observations = obs_resp.json().get("data", [])
                    for obs in observations:
                        rows.append({
                            "id": obs.get("id"),
                            "traceId": trace_id,
                            "type": obs.get("type"),
                            "name": obs.get("name"),
                            "startTime": obs.get("startTime"),
                            "endTime": obs.get("endTime"),
                            "input": obs.get("input"),
                            "output": obs.get("output"),
                            "metadata": obs.get("metadata"),
                            "sessionId": trace.get("sessionId"),
                            "userId": trace.get("userId"),
                            "traceName": trace.get("name"),
                            "providedModelName": obs.get("model"),
                        })
            except Exception:
                pass

        # 分离存储：TRACE 行 → traces.jsonl.gz，OBSERVATION 行 → observations.jsonl.gz
        appended, skipped = append_traces(project_id, rows, existing_ids)
        total_new += appended
        total_skipped += skipped

        obs_appended, obs_skipped = append_observations(project_id, rows, existing_obs_ids)
        total_obs_new += obs_appended
        total_obs_skipped += obs_skipped

        if page_max_ts and (max_ts is None or page_max_ts > max_ts):
            max_ts = page_max_ts

        pages_done = page

        if progress_callback:
            total_meta = (data.get("meta") or {}).get("totalItems")
            progress_callback("syncing", total_new, total_skipped, pages_done, total_meta)

        if len(traces) < limit:
            break
        time.sleep(0.2)

    elapsed = time.time() - t0

    # 更新 checkpoint（仅成功后）
    if total_new > 0 or total_skipped > 0:
        now = datetime.now(timezone.utc).isoformat()
        registry["last_sync_at"] = now
        if max_ts:
            registry["last_trace_timestamp"] = max_ts
        registry["total_traces_synced"] = len(existing_ids)
        registry["total_observations_synced"] = len(existing_obs_ids)
        registry["updated_at"] = now
        _save_registry(project_id, registry)

    # 仅当有新增数据时更新快照（逻辑引用，不复制文件）
    snapshot_meta = None
    snapshot_created = False
    if total_new > 0 or total_obs_new > 0:
        try:
            snapshot_meta = _update_current_snapshot(project_id)
            snapshot_created = True
        except Exception:
            pass

    if progress_callback:
        progress_callback("done", total_new, total_skipped, pages_done, None)

    result = {
        "new_traces": total_new,
        "skipped": total_skipped,
        "new_observations": total_obs_new,
        "skipped_observations": total_obs_skipped,
        "pages": pages_done,
        "last_timestamp": max_ts,
        "elapsed": elapsed,
        "snapshot_created": snapshot_created,
    }
    if snapshot_meta:
        result["snapshot_id"] = snapshot_meta.get("snapshot_id", "")
    return result


# ── 回填历史 Observation ─────────────────────────────────────


def backfill_observations(project_id: str, host: str, public_key: str,
                          secret_key: str, limit_per_trace: int = 100,
                          progress_callback=None,
                          max_retries: int = 2) -> dict:
    """为本地已有但缺少 observation 的 trace 回填 observation 数据。

    遍历 traces.jsonl.gz 中的 trace_id，检查 observations.jsonl.gz 中
    是否已有该 trace 的 observation。对缺失的调用 API 补充。

    Args:
        project_id: 项目 ID
        host: Langfuse host
        public_key: Public Key
        secret_key: Secret Key
        limit_per_trace: 每个 trace 最多拉取的 observation 数
        progress_callback: (phase, done, total, new_obs, errors)
        max_retries: 每次请求最大重试

    Returns:
        {"total_traces", "traces_with_obs_before", "traces_backfilled",
         "new_observations", "errors", "elapsed"}
    """
    host = host.strip().rstrip("/")
    existing_trace_ids = load_existing_trace_ids(project_id)
    existing_obs_ids = load_existing_obs_ids(project_id)

    total_traces = len(existing_trace_ids)
    if total_traces == 0:
        return {"total_traces": 0, "traces_with_obs_before": 0,
                "traces_backfilled": 0, "new_observations": 0,
                "errors": 0, "elapsed": 0}

    # 统计哪些 trace 已有 observation（通过 traceId 关联）
    traces_with_obs = set()
    op = _obs_path(project_id)
    if op.exists():
        try:
            with gzip.open(op, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        tid = row.get("traceId")
                        if tid:
                            traces_with_obs.add(tid)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    traces_needing_backfill = existing_trace_ids - traces_with_obs
    traces_needing_list = sorted(traces_needing_backfill)
    total_to_backfill = len(traces_needing_list)

    if progress_callback:
        progress_callback("starting", 0, total_to_backfill, 0, 0)

    t0 = time.time()
    new_obs_total = 0
    error_count = 0
    done = 0

    obs_path = _obs_path(project_id)
    obs_path.parent.mkdir(parents=True, exist_ok=True)

    for trace_id in traces_needing_list:
        # 获取 observations
        rows = []
        for attempt in range(max_retries + 1):
            try:
                obs_resp = requests.get(
                    f"{host}/api/public/observations",
                    auth=(public_key.strip(), secret_key.strip()),
                    params={"traceId": trace_id, "limit": limit_per_trace},
                    timeout=30,
                )
                if obs_resp.status_code == 200:
                    observations = obs_resp.json().get("data", [])
                    for obs in observations:
                        rows.append({
                            "id": obs.get("id"),
                            "traceId": trace_id,
                            "type": obs.get("type"),
                            "name": obs.get("name"),
                            "startTime": obs.get("startTime"),
                            "endTime": obs.get("endTime"),
                            "input": obs.get("input"),
                            "output": obs.get("output"),
                            "metadata": obs.get("metadata"),
                            "sessionId": None,
                            "userId": None,
                            "traceName": None,
                            "providedModelName": obs.get("model"),
                        })
                    break
                elif obs_resp.status_code == 401:
                    raise RuntimeError("认证失败，请检查 Key")
                else:
                    raise RuntimeError(f"HTTP {obs_resp.status_code}")
            except Exception:
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    error_count += 1

        # 按 observation id 去重写入
        if rows:
            obs_appended, _ = append_observations(project_id, rows, existing_obs_ids)
            new_obs_total += obs_appended

        done += 1
        if progress_callback:
            progress_callback("backfilling", done, total_to_backfill,
                              new_obs_total, error_count)

        time.sleep(0.1)  # 避免过快请求

    elapsed = time.time() - t0

    # 更新 registry
    registry = load_project(project_id)
    if registry:
        registry["total_observations_synced"] = len(existing_obs_ids)
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_registry(project_id, registry)

    if progress_callback:
        progress_callback("done", done, total_to_backfill, new_obs_total, error_count)

    return {
        "total_traces": total_traces,
        "traces_with_obs_before": len(traces_with_obs),
        "traces_backfilled": done,
        "new_observations": new_obs_total,
        "errors": error_count,
        "elapsed": elapsed,
    }


def get_observation_coverage(project_id: str) -> dict:
    """获取 observation 覆盖率统计。

    Returns:
        {"total_traces", "traces_with_obs", "coverage_pct"}
    """
    trace_ids = load_existing_trace_ids(project_id)
    total = len(trace_ids)
    if total == 0:
        return {"total_traces": 0, "traces_with_obs": 0, "coverage_pct": 0}

    op = _obs_path(project_id)
    traces_with_obs = set()
    if op.exists():
        try:
            with gzip.open(op, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        tid = row.get("traceId")
                        if tid:
                            traces_with_obs.add(tid)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    covered = len(traces_with_obs & trace_ids)
    return {
        "total_traces": total,
        "traces_with_obs": covered,
        "coverage_pct": round(covered / total * 100, 1) if total > 0 else 0,
    }


# ── 统计与清理 ───────────────────────────────────────────────


def get_project_stats(project_id: str) -> dict:
    """获取项目统计信息。"""
    registry = load_project(project_id)
    if not registry:
        return {}

    tp = _traces_path(project_id)
    file_size = tp.stat().st_size if tp.exists() else 0

    op = _obs_path(project_id)
    obs_file_size = op.stat().st_size if op.exists() else 0

    snap_dir = _project_dir(project_id) / "snapshots"
    snapshot_count = len(list(snap_dir.glob("*.jsonl.gz"))) if snap_dir.exists() else 0

    return {
        "project_id": project_id,
        "project_name": registry.get("project_name", ""),
        "host": registry.get("host", ""),
        "key_masked": registry.get("key_masked", ""),
        "last_sync_at": registry.get("last_sync_at"),
        "last_trace_timestamp": registry.get("last_trace_timestamp"),
        "total_traces_synced": registry.get("total_traces_synced", 0),
        "total_observations_synced": registry.get("total_observations_synced", 0),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "obs_file_size_bytes": obs_file_size,
        "obs_file_size_mb": round(obs_file_size / (1024 * 1024), 2),
        "snapshot_count": snapshot_count,
    }


def list_cleanup_candidates() -> list[dict]:
    """列出可清理的旧版全量导出文件。

    Returns:
        [{"path", "name", "size_mb", "mtime"}]
    """
    if not RAW_DIR.exists():
        return []
    candidates = []
    for f in RAW_DIR.glob("langfuse_api_export_*.jsonl"):
        stat = f.stat()
        candidates.append({
            "path": str(f),
            "name": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    candidates.sort(key=lambda x: x["path"], reverse=True)
    return candidates


def cleanup_files(file_paths: list[str]) -> tuple[int, int]:
    """删除指定文件。

    Returns:
        (deleted_count, failed_count)
    """
    deleted = 0
    failed = 0
    for fp in file_paths:
        try:
            p = Path(fp)
            if p.exists() and p.is_file():
                p.unlink()
                deleted += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    return deleted, failed


# ── 加载 traces ──────────────────────────────────────────────


def load_project_traces(project_id: str):
    """从 traces.jsonl.gz + observations.jsonl.gz 加载所有行（生成器）。

    先 yield TRACE 行，再 yield OBSERVATION 行。
    """
    tp = _traces_path(project_id)
    op = _obs_path(project_id)
    if not tp.exists():
        return
    try:
        with gzip.open(tp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    if op.exists():
        try:
            with gzip.open(op, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass


# ── 快照管理 ─────────────────────────────────────────────────


def _snapshots_dir(project_id: str) -> Path:
    return _project_dir(project_id) / "snapshots"


def _update_current_snapshot(project_id: str) -> dict:
    """更新/创建当前逻辑快照（不复制文件，仅更新 registry 元数据）。

    逻辑快照指向当前 traces.jsonl.gz + observations.jsonl.gz，
    记录 trace_count、obs_count 和文件大小。
    每次增量同步有新数据时调用。
    """
    tp = _traces_path(project_id)
    if not tp.exists():
        raise RuntimeError("无可用 trace 数据，请先同步")

    op = _obs_path(project_id)
    has_observations = op.exists() and op.stat().st_size > 0

    # 统计 trace 数和 observation 数
    trace_count = _count_lines_by_type(tp, count_type="TRACE")
    obs_count = _count_lines_by_type(op, count_type="non-TRACE") if has_observations else 0

    trace_file_size = tp.stat().st_size
    obs_file_size = op.stat().st_size if has_observations else 0
    total_file_size = trace_file_size + obs_file_size

    now = datetime.now(timezone.utc)
    registry = load_project(project_id)
    if not registry:
        raise RuntimeError("项目不存在")

    current_id = registry.get("current_snapshot_id", "")

    # 查找当前快照条目并更新
    snapshots = registry.get("snapshots", [])
    current_snap = None
    for snap in snapshots:
        if snap.get("snapshot_id") == current_id and snap.get("snapshot_type") == "logical":
            current_snap = snap
            break

    if current_snap:
        # 更新现有逻辑快照
        current_snap["trace_count"] = trace_count
        current_snap["observation_count"] = obs_count
        current_snap["has_observations"] = has_observations
        current_snap["file_size_bytes"] = trace_file_size
        current_snap["trace_file_size_bytes"] = trace_file_size
        current_snap["observation_file_size_bytes"] = obs_file_size
        current_snap["total_file_size_bytes"] = total_file_size
        current_snap["updated_at"] = now.isoformat()
    else:
        # 创建新的逻辑快照
        snapshot_id = f"logical_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        snap_meta = {
            "snapshot_id": snapshot_id,
            "snapshot_type": "logical",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "trace_count": trace_count,
            "observation_count": obs_count,
            "has_observations": has_observations,
            "file_size_bytes": trace_file_size,
            "trace_file_size_bytes": trace_file_size,
            "observation_file_size_bytes": obs_file_size,
            "total_file_size_bytes": total_file_size,
            "parsed": False,
            "parsed_at": None,
        }
        snapshots.append(snap_meta)
        registry["current_snapshot_id"] = snapshot_id
        current_snap = snap_meta

    registry["snapshots"] = snapshots
    registry["snapshot_count"] = len(snapshots)
    _save_registry(project_id, registry)
    return current_snap


def _count_lines_by_type(file_path: Path, count_type: str = "TRACE") -> int:
    """统计 gzip JSONL 文件中指定类型的行数。"""
    if not file_path.exists():
        return 0
    count = 0
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if count_type == "TRACE" and row.get("type") == "TRACE":
                        count += 1
                    elif count_type == "non-TRACE" and row.get("type") != "TRACE":
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        return -1
    return count


def create_frozen_snapshot(project_id: str) -> dict:
    """从当前缓存创建独立快照（复制文件到 snapshots/ 目录）。

    仅在用户明确请求时调用。复制 traces.jsonl.gz 和 observations.jsonl.gz
    到 snapshots/snap_<timestamp>.jsonl.gz，在 registry 中记录快照元数据。

    只有同时拥有 trace 和 observation 副本的 frozen evidence snapshot
    才能作为正式样本准备和 Judge 的来源。仅有 trace 的 frozen snapshot
    是 index-only 存档，不能用于正式评测。

    Returns:
        {"snapshot_id", "snapshot_type": "frozen", ...}
    """
    tp = _traces_path(project_id)
    if not tp.exists():
        raise RuntimeError("无可用 trace 数据，请先同步")

    op = _obs_path(project_id)
    has_observations = op.exists() and op.stat().st_size > 0

    snap_dir = _snapshots_dir(project_id)
    snap_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%d_%H%M%S_%f")
    snapshot_id = f"snap_{ts_str}"
    snap_traces_path = snap_dir / f"{snapshot_id}.jsonl.gz"
    snap_obs_path = snap_dir / f"{snapshot_id}.obs.jsonl.gz"

    # 复制 traces 文件
    import shutil
    shutil.copy2(tp, snap_traces_path)

    # 复制 observations 文件（如果存在）
    obs_file_size = 0
    if has_observations:
        shutil.copy2(op, snap_obs_path)
        obs_file_size = snap_obs_path.stat().st_size

    # 统计 trace 数和 observation 数
    trace_count = 0
    obs_count = 0
    try:
        with gzip.open(tp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("type") == "TRACE":
                        trace_count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        trace_count = -1

    if has_observations:
        try:
            with gzip.open(op, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if row.get("type") != "TRACE":
                            obs_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            obs_count = -1

    trace_file_size = snap_traces_path.stat().st_size
    total_file_size = trace_file_size + obs_file_size

    # 更新 registry
    registry = load_project(project_id)
    if registry:
        snapshots = registry.get("snapshots", [])
        snap_meta = {
            "snapshot_id": snapshot_id,
            "snapshot_type": "frozen",
            "created_at": now.isoformat(),
            "trace_count": trace_count,
            "observation_count": obs_count,
            "has_observations": has_observations,
            "file_size_bytes": trace_file_size,          # 兼容旧字段
            "trace_file_size_bytes": trace_file_size,
            "observation_file_size_bytes": obs_file_size,
            "total_file_size_bytes": total_file_size,
            "parsed": False,
            "parsed_at": None,
        }
        snapshots.append(snap_meta)
        registry["snapshots"] = snapshots
        registry["snapshot_count"] = len(snapshots)
        # 冻结快照不自动设为 current（current 保留给逻辑快照）
        _save_registry(project_id, registry)
        return snap_meta

    return {
        "snapshot_id": snapshot_id,
        "snapshot_type": "frozen",
        "created_at": now.isoformat(),
        "trace_count": trace_count,
        "observation_count": obs_count,
        "has_observations": has_observations,
        "file_size_bytes": trace_file_size,
        "trace_file_size_bytes": trace_file_size,
        "observation_file_size_bytes": obs_file_size,
        "total_file_size_bytes": total_file_size,
        "parsed": False,
    }


def list_snapshots(project_id: str) -> list[dict]:
    """列出项目的所有快照。"""
    registry = load_project(project_id)
    if not registry:
        return []
    return registry.get("snapshots", [])


def get_current_snapshot_id(project_id: str) -> str:
    """获取当前活跃的 snapshot_id。"""
    registry = load_project(project_id)
    if not registry:
        return ""
    return registry.get("current_snapshot_id", "")


def get_snapshot_path(project_id: str, snapshot_id: str) -> Path:
    """获取快照文件路径。逻辑快照返回缓存文件路径。"""
    # 检查是否为逻辑快照
    registry = load_project(project_id)
    if registry:
        for snap in registry.get("snapshots", []):
            if snap.get("snapshot_id") == snapshot_id and snap.get("snapshot_type") == "logical":
                return _traces_path(project_id)
    return _snapshots_dir(project_id) / f"{snapshot_id}.jsonl.gz"


def export_snapshot_as_jsonl(project_id: str, snapshot_id: str,
                             output_path: Path = None) -> Path:
    """将 gzip 快照解压并合并为纯 JSONL 文件（供 parser 使用）。

    合并 traces + observations 为单一 JSONL，与旧版 fetch_traces.py 输出格式兼容。

    Args:
        project_id: 项目 ID
        snapshot_id: 快照 ID
        output_path: 输出路径（可选，默认为快照同目录下 .jsonl）

    Returns:
        解压后的 JSONL 文件路径

    Raises:
        RuntimeError: 快照不是 frozen evidence snapshot
    """
    registry = load_project(project_id)
    snapshot_meta = next(
        (snap for snap in (registry or {}).get("snapshots", [])
         if snap.get("snapshot_id") == snapshot_id),
        None,
    )
    if not snapshot_meta:
        raise RuntimeError(f"快照不存在: {snapshot_id}")
    if snapshot_meta.get("snapshot_type") != "frozen":
        raise RuntimeError(
            f"快照 {snapshot_id} 为 logical snapshot，仅引用当前项目缓存，"
            "不可作为正式样本解析来源。请创建并选择 frozen evidence snapshot。"
        )
    if not snapshot_meta.get("has_observations", False):
        raise RuntimeError(
            f"快照 {snapshot_id} 为 index-only snapshot（不含检索证据），"
            "不可解析为评测样本。请使用 frozen evidence snapshot。"
        )

    snap_path = get_snapshot_path(project_id, snapshot_id)
    if not snap_path.exists():
        raise RuntimeError(f"快照文件不存在: {snapshot_id}")

    # frozen snapshot 的 observation 必须与 trace 文件位于 snapshots/ 目录。
    snap_obs_path = snap_path.parent / f"{snapshot_id}.obs.jsonl.gz"
    has_obs = snap_obs_path.exists() and snap_obs_path.stat().st_size > 0

    if not has_obs:
        raise RuntimeError(
            f"快照 {snapshot_id} 缺少独立 observation 文件，"
            "不是完整 frozen evidence snapshot，不能用于正式解析。"
        )

    if output_path is None:
        output_path = snap_path.with_suffix("")  # 去掉 .gz

    # 合并 traces + observations
    with output_path.open("w", encoding="utf-8") as fout:
        with gzip.open(snap_path, "rt", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
        with gzip.open(snap_obs_path, "rt", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)

    return output_path


def mark_snapshot_parsed(project_id: str, snapshot_id: str):
    """标记快照已解析。"""
    registry = load_project(project_id)
    if not registry:
        return
    now = datetime.now(timezone.utc).isoformat()
    for snap in registry.get("snapshots", []):
        if snap.get("snapshot_id") == snapshot_id:
            snap["parsed"] = True
            snap["parsed_at"] = now
            break
    _save_registry(project_id, registry)


def can_cleanup_snapshot(project_id: str, snapshot_id: str) -> tuple[bool, str]:
    """检查 frozen snapshot 是否具备人工清理前提。

    Returns:
        (can_cleanup, reason)。本函数不执行删除；返回 True 仅表示可由用户
        在完成引用核验后手动处理。
    """
    registry = load_project(project_id)
    if not registry:
        return False, "项目不存在"

    # 当前活跃快照不可清理
    if registry.get("current_snapshot_id") == snapshot_id:
        return False, "当前活跃快照，不可删除"

    for snap in registry.get("snapshots", []):
        if snap.get("snapshot_id") == snapshot_id:
            if snap.get("snapshot_type") != "frozen":
                return False, "logical snapshot 引用当前项目缓存，不可单独清理"
            if not snap.get("parsed"):
                return False, "尚未解析，不可删除"
            trace_path = get_snapshot_path(project_id, snapshot_id)
            if not trace_path.exists():
                return False, "frozen snapshot 缺少 trace 文件，不可清理"
            if snap.get("has_observations", False):
                obs_path = trace_path.parent / f"{snapshot_id}.obs.jsonl.gz"
                if not obs_path.exists():
                    return False, "frozen evidence snapshot 缺少 observation 文件，不可清理"
            references = _find_snapshot_references(project_id, snapshot_id)
            if references:
                return False, f"已被 {len(references)} 个历史产物引用，不可删除"
            return True, "未发现已知引用；人工清理时必须成对处理 trace 与 observation 文件"

    return False, "快照不存在"


def cleanup_old_snapshots(project_id: str, keep: int = 3) -> tuple[int, str]:
    """检查旧 frozen snapshot 的人工清理候选，不删除任何文件。

    本阶段禁止自动删除快照。保留此 API 是为了兼容旧调用方并提供提示，
    返回的 deleted_count 永远为 0。

    Returns:
        (deleted_count, message)
    """
    registry = load_project(project_id)
    if not registry:
        return 0, "项目不存在"

    snapshots = registry.get("snapshots", [])
    current_id = registry.get("current_snapshot_id", "")

    # 按时间排序，最新的在前
    sorted_snaps = sorted(snapshots, key=lambda s: s.get("created_at", ""), reverse=True)

    candidates = []
    for snap in sorted_snaps:
        sid = snap.get("snapshot_id", "")
        if sid == current_id:
            continue
        if snap.get("snapshot_type") != "frozen" or not snap.get("parsed"):
            continue
        can_cleanup, _ = can_cleanup_snapshot(project_id, sid)
        if can_cleanup:
            candidates.append(snap)

    manual_candidates = candidates[keep:]
    return 0, (
        f"未执行自动清理。本次发现 {len(manual_candidates)} 个可由用户人工确认的 "
        "frozen snapshot 候选；当前、未解析或被引用快照均已保留。"
    )


def _find_snapshot_references(project_id: str, snapshot_id: str) -> list[str]:
    """查找已知历史产物中的 snapshot provenance 引用。

    该检查是保守的：任何匹配都阻止清理。历史全局 processed 文件尚未完成
    provenance 迁移时，无法证明其来源的快照也不会被这个函数判定为可自动删除。
    """
    references = []
    snapshot_path = str(get_snapshot_path(project_id, snapshot_id))
    for root in (DATA_DIR / "processed", DATA_DIR / "judged", DATA_DIR / "reports"):
        if not root.exists():
            continue
        for path in root.rglob("*.json*"):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if (
                snapshot_id in text
                or snapshot_path in text
                or (
                    f'"langfuse_project_id": "{project_id}"' in text
                    and f'"langfuse_snapshot_id": "{snapshot_id}"' in text
                )
            ):
                references.append(str(path))
    return references


# ── 可解析数据源 ─────────────────────────────────────────────


def _get_snapshot_sizes(project_id: str, snap: dict) -> dict:
    """获取快照的 trace / obs / 总大小（字节），兼容旧 registry 条目。

    优先从 snap 元数据读取；若缺失则动态 stat 文件。
    逻辑快照从缓存文件读取，冻结快照从 snapshots/ 目录读取。
    """
    sid = snap.get("snapshot_id", "")
    has_obs = snap.get("has_observations", False)
    snap_type = snap.get("snapshot_type", "frozen")

    # 优先用已存储的值
    trace_size = snap.get("trace_file_size_bytes")
    obs_size = snap.get("observation_file_size_bytes")
    total_size = snap.get("total_file_size_bytes")

    # 动态计算缺失值
    if snap_type == "logical":
        # 逻辑快照：从缓存文件读取
        if trace_size is None:
            tp = _traces_path(project_id)
            trace_size = tp.stat().st_size if tp.exists() else 0
        if obs_size is None:
            op = _obs_path(project_id)
            obs_size = op.stat().st_size if op.exists() else 0
    else:
        # 冻结快照：从 snapshots/ 目录读取
        snap_dir = _snapshots_dir(project_id)
        if trace_size is None:
            tp = snap_dir / f"{sid}.jsonl.gz"
            trace_size = tp.stat().st_size if tp.exists() else 0
        if obs_size is None and has_obs:
            op = snap_dir / f"{sid}.obs.jsonl.gz"
            obs_size = op.stat().st_size if op.exists() else 0
        elif obs_size is None:
            obs_size = 0

    if total_size is None:
        total_size = trace_size + obs_size

    return {
        "trace_file_size_bytes": trace_size,
        "observation_file_size_bytes": obs_size,
        "total_file_size_bytes": total_size,
    }


def list_parseable_sources(project_id: str = None) -> list[dict]:
    """列出所有可解析的数据源（项目快照 + 旧版 raw 文件）。

    Returns:
        [{"source_id", "source_type", "label", "path", "size_mb",
          "trace_size_mb", "obs_size_mb", "mtime",
          "parsed", "project_id", "snapshot_id", "has_observations"}]
    """
    sources = []

    # 项目快照
    if project_id:
        for snap in list_snapshots(project_id):
            sid = snap.get("snapshot_id", "")
            snap_type = snap.get("snapshot_type", "frozen")
            has_obs = snap.get("has_observations", False)
            obs_count = snap.get("observation_count", 0)
            trace_count = snap.get("trace_count", 0)
            sizes = _get_snapshot_sizes(project_id, snap)

            # 逻辑快照指向缓存文件，冻结快照指向 snapshots/ 目录
            if snap_type == "logical":
                snap_path = _traces_path(project_id)
                if not snap_path.exists():
                    continue
                type_badge = "📋 当前缓存"
            else:
                snap_path = get_snapshot_path(project_id, sid)
                if not snap_path.exists():
                    continue
                type_badge = "📦 独立快照"

            if snap_type == "logical":
                label = f"{type_badge} — {trace_count} traces（仅同步缓存，不可正式解析）"
                source_type = "logical_snapshot"
            elif has_obs:
                label = f"{type_badge} — {trace_count} traces, {obs_count} obs"
                source_type = "evidence_snapshot"
            else:
                label = f"{type_badge} — {trace_count} traces，仅索引"
                source_type = "index_snapshot"

            total_mb = round(sizes["total_file_size_bytes"] / (1024 * 1024), 2)
            trace_mb = round(sizes["trace_file_size_bytes"] / (1024 * 1024), 2)
            obs_mb = round(sizes["observation_file_size_bytes"] / (1024 * 1024), 2)

            sources.append({
                "source_id": f"{project_id}:{sid}",
                "source_type": source_type,
                "snapshot_type": snap_type,
                "label": label,
                "path": str(snap_path),
                "size_mb": total_mb,
                "trace_size_mb": trace_mb,
                "obs_size_mb": obs_mb,
                "mtime": datetime.fromtimestamp(snap_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "parsed": snap.get("parsed", False),
                "project_id": project_id,
                "snapshot_id": sid,
                "has_observations": has_obs,
                "trace_count": trace_count,
                "observation_count": obs_count,
            })

    # 旧版 raw 文件（天然包含 observations）
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.glob("langfuse_api_export_*.jsonl"), reverse=True):
            stat = f.stat()
            sources.append({
                "source_id": f"legacy:{f.name}",
                "source_type": "legacy_raw",
                "label": f"📁 [旧版] {f.name}",
                "path": str(f),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "parsed": False,
                "project_id": None,
                "snapshot_id": None,
                "has_observations": True,  # 旧版全量导出包含 observations
            })

    return sources
