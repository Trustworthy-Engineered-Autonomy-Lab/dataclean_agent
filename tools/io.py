from __future__ import annotations

import json
import os
import re
import tempfile
import time
import threading
import shutil
import uuid
from pathlib import Path

ROOT = ".dataclean"
TASKS_DIR = "tasks"
STATE = "state.json"
SESSION = "session.json"

__all__ = [
    "ROOT", "TASKS_DIR", "STATE", "SESSION",
    "print_progress", "set_progress_queue",
    "_task_dir", "_state_path", "_migrate_legacy",
    "_load", "_save", "_artifact", "_load_task_spec", "_write_json_atomic",
    "_task_artifact_reference",
    "record_observation", "append_ledger",
    "_reset_vlm_budget_if_new_round",
    "_session_path", "_load_session", "_save_session",
]


_progress_local = threading.local()
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def set_progress_queue(q):
    _progress_local.queue = q


def print_progress(body):
    print(body, flush=True)
    q = getattr(_progress_local, "queue", None)
    if q is not None:
        try:
            q.put(body)
        except Exception:
            pass


def _safe_task_id(branch: str) -> str:
    b = (branch or "exp_1").strip() or "exp_1"
    if not _TASK_ID_RE.fullmatch(b):
        raise ValueError("task_id must contain only ASCII letters, numbers, underscores, and hyphens")
    return b


def _task_dir(workspace_dir, branch: str = "exp_1", create: bool = True) -> Path:
    b = _safe_task_id(branch)
    _migrate_legacy(workspace_dir)
    tasks_root = (Path(workspace_dir).resolve() / ROOT / TASKS_DIR).resolve()
    d = (tasks_root / b).resolve()
    if d.parent != tasks_root:
        raise ValueError("task path escapes workspace task directory")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d

def _state_path(workspace_dir, branch: str = "exp_1"):
    return _task_dir(workspace_dir, branch=branch, create=False) / STATE

def _migrate_legacy(workspace_dir):
    root = Path(workspace_dir) / ROOT
    tasks = root / TASKS_DIR
    if tasks.exists():
        return
    legacy_main = root / STATE
    legacy_exp = root / "experiments"
    legacy_art = root / "artifacts"
    if not legacy_main.exists() and not legacy_exp.exists():
        return

    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{TASKS_DIR}.migrating-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        if legacy_main.exists():
            td = staging / "main"
            td.mkdir()
            shutil.copy2(legacy_main, td / STATE)
            if legacy_art.is_dir():
                shutil.copytree(legacy_art, td / "artifacts")

        if legacy_exp.is_dir():
            for branch_dir in sorted(legacy_exp.iterdir()):
                if not branch_dir.is_dir() or not _TASK_ID_RE.fullmatch(branch_dir.name):
                    continue
                td = staging / branch_dir.name
                td.mkdir(exist_ok=True)
                source_state = branch_dir / STATE
                if source_state.is_file():
                    shutil.copy2(source_state, td / STATE)
                source_artifacts = branch_dir / "artifacts"
                if source_artifacts.is_dir():
                    shutil.copytree(
                        source_artifacts, td / "artifacts", dirs_exist_ok=True
                    )

        # Validate copied JSON state before publishing the migration. Legacy
        # files remain untouched regardless of success or failure.
        for state_path in staging.glob(f"*/{STATE}"):
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Legacy state is not a JSON object: {state_path}")
        os.replace(staging, tasks)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

def _load(workspace_dir, branch: str = "main"):
    p = _state_path(workspace_dir, branch=branch)
    if not p.exists():
        raise ValueError(f"Task '{branch}' is not initialized. Call define_task first.")
    return json.loads(p.read_text())


def _write_json_atomic(path: Path, payload: dict | list):
    """Atomically replace a JSON document so interrupted writes cannot truncate state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _save(workspace_dir, state, branch: str = "main"):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json_atomic(_task_dir(workspace_dir, branch=branch) / STATE, state)

def _artifact(workspace_dir, name, branch: str = "main"):
    d = _task_dir(workspace_dir, branch) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    candidate = (d / str(name)).resolve()
    if candidate.parent != d.resolve():
        raise ValueError("artifact name must be a filename within the task artifact directory")
    return candidate


def _task_artifact_reference(workspace_dir, branch, reference, *, must_exist=True):
    """Resolve a persisted artifact reference without allowing state-path escape."""
    artifact_dir = _artifact(workspace_dir, "placeholder", branch=branch).parent.resolve()
    raw = Path(str(reference))
    if not raw.is_absolute() and raw.parent != Path("."):
        raise ValueError("Relative artifact references must be filenames")
    candidate = raw.resolve() if raw.is_absolute() else (artifact_dir / raw.name).resolve()
    if candidate.parent != artifact_dir:
        raise ValueError("Artifact reference escapes the active task artifact directory")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"Task artifact does not exist: {candidate.name}")
    return candidate


def _load_task_spec(workspace_dir, branch: str = "main"):
    p = _task_dir(workspace_dir, branch=branch, create=False) / "task_spec.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def record_observation(state: dict, stage: str, payload: dict, workspace_dir=None, branch="main"):
    state.setdefault("latest_observation", {})[stage] = payload
    if workspace_dir:
        _auto_write_decision_log(workspace_dir, branch or state.get("branch", "main"), state, stage, payload)


def _auto_write_decision_log(workspace_dir, branch, state, stage, payload):
    log_path = _task_dir(workspace_dir, branch) / "decision_log.md"
    r = state.get("round", 0)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary_str = json.dumps(payload, ensure_ascii=False)
    if len(summary_str) > 200:
        summary_str = summary_str[:200] + "..."
    block = (
        f"### [Round {r}] Stage={stage} · Task={branch} · {now}\n"
        f"- **Observation Summary**: `{summary_str}`\n\n"
        f"---\n\n"
    )
    if not log_path.exists():
        header = (
            f"# Decision Log · Task `{branch}`\n\n"
            f"> Automatically maintained from stage observations; structured action rationales live in state.json decision_trace.\n\n"
            f"---\n\n"
        )
        log_path.write_text(header + block, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(block)


def append_ledger(state: dict, entry: dict):
    state.setdefault("round_ledger", []).append(entry)


def _reset_vlm_budget_if_new_round(state: dict):
    if state.get("vlm_budget_current_round") != state.get("round"):
        state["vlm_budget_current_round"] = state.get("round", 0)
        state["vlm_budget_used_this_round"] = 0


_SESSION_DEFAULT = {
    "goal": "",
    "hypothesis": "",
    "status": "",
    "open_questions": [],
    "learnings": [],
}


def _session_path(workspace_dir, branch: str = "main"):
    return _task_dir(workspace_dir, branch=branch, create=False) / SESSION


def _load_session(workspace_dir, branch: str = "main"):
    p = _session_path(workspace_dir, branch=branch)
    if not p.exists():
        return {k: ("" if isinstance(v, str) else []) for k, v in _SESSION_DEFAULT.items()}
    try:
        s = json.loads(p.read_text())
    except Exception:
        return {k: ("" if isinstance(v, str) else []) for k, v in _SESSION_DEFAULT.items()}
    for k, v in _SESSION_DEFAULT.items():
        if k not in s:
            s[k] = "" if isinstance(v, str) else []
        elif isinstance(v, str) and not isinstance(s[k], str):
            s[k] = str(s[k])
        elif isinstance(v, list) and not isinstance(s[k], list):
            s[k] = []
    return s


def _save_session(workspace_dir, branch: str, session: dict):
    p = _session_path(workspace_dir, branch=branch)
    p.parent.mkdir(parents=True, exist_ok=True)
    session["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json_atomic(p, session)
