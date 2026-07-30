import json
import time
import threading
from pathlib import Path

ROOT = ".dataclean"
TASKS_DIR = "tasks"
STATE = "state.json"
SESSION = "session.json"

__all__ = [
    "ROOT", "TASKS_DIR", "STATE", "SESSION",
    "print_progress", "set_progress_queue",
    "_task_dir", "_state_path", "_migrate_legacy",
    "_load", "_save", "_artifact",
    "record_observation", "append_ledger", "_advance_round",
    "_reset_vlm_budget_if_new_round",
    "_session_path", "_load_session", "_save_session",
]


_progress_queue = None
_progress_lock = threading.Lock()


def set_progress_queue(q):
    global _progress_queue
    with _progress_lock:
        _progress_queue = q


def print_progress(body):
    print(body, flush=True)
    with _progress_lock:
        q = _progress_queue
    if q is not None:
        try:
            q.put(body)
        except Exception:
            pass


def _task_dir(workspace_dir, branch: str = "exp_1") -> Path:
    b = (branch or "exp_1").strip() or "exp_1"
    _migrate_legacy(workspace_dir)
    d = Path(workspace_dir) / ROOT / TASKS_DIR / b
    d.mkdir(parents=True, exist_ok=True)
    return d

def _state_path(workspace_dir, branch: str = "exp_1"):
    b = (branch or "exp_1").strip() or "exp_1"
    _migrate_legacy(workspace_dir)
    return Path(workspace_dir) / ROOT / TASKS_DIR / b / STATE

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

    tasks.mkdir(parents=True, exist_ok=True)

    if legacy_main.exists():
        td = Path(workspace_dir) / ROOT / TASKS_DIR / "main"
        td.mkdir(parents=True, exist_ok=True)
        if legacy_art.exists():
            dest = td / "artifacts"
            dest.mkdir(parents=True, exist_ok=True)
            for f in legacy_art.iterdir():
                if f.is_file():
                    try:
                        f.rename(dest / f.name)
                    except Exception:
                        pass
            try:
                legacy_art.rmdir()
            except Exception:
                pass
        try:
            legacy_main.rename(td / STATE)
        except Exception:
            pass

    if legacy_exp.exists():
        for bdir in sorted(legacy_exp.iterdir()):
            if not bdir.is_dir():
                continue
            td = Path(workspace_dir) / ROOT / TASKS_DIR / bdir.name
            td.mkdir(parents=True, exist_ok=True)
            src_state = bdir / STATE
            if src_state.exists():
                try:
                    src_state.rename(td / STATE)
                except Exception:
                    pass
            src_art = bdir / "artifacts"
            if src_art.exists():
                dest = td / "artifacts"
                dest.mkdir(parents=True, exist_ok=True)
                for f in src_art.iterdir():
                    if f.is_file():
                        try:
                            f.rename(dest / f.name)
                        except Exception:
                            pass
        try:
            legacy_exp.rmdir()
        except Exception:
            pass

def _load(workspace_dir, branch: str = "main"):
    p = _state_path(workspace_dir, branch=branch)
    if not p.exists():
        raise ValueError("Dataset not configured. Call configure_dataset first.")
    return json.loads(p.read_text())

def _save(workspace_dir, state, branch: str = "main"):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _state_path(workspace_dir, branch=branch).write_text(json.dumps(state, ensure_ascii=False, indent=2))

def _artifact(workspace_dir, name, branch: str = "main"):
    d = _task_dir(workspace_dir, branch) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


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
            f"> Automatically maintained by system observation hooks to record auditable D1-D5 decision traces.\n\n"
            f"---\n\n"
        )
        log_path.write_text(header + block, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(block)


def append_ledger(state: dict, entry: dict):
    state.setdefault("round_ledger", []).append(entry)


def _advance_round(state: dict):
    state["round"] = int(state.get("round", 0)) + 1


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
    _migrate_legacy(workspace_dir)
    return Path(workspace_dir) / ROOT / TASKS_DIR / branch / SESSION


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
    p.write_text(json.dumps(session, ensure_ascii=False, indent=2))
