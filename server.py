import json
import os
import sys
from collections import deque
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import database
from agent import Agent
from tools import Tool, optional_dependency_errors
from tools.utils import ROOT, TASKS_DIR, _load, _load_session, _dataset_config, _task_dir, _write_json_atomic
from tools.policies import (describe_task_types, describe_default_pipeline,
                            LEDGER_FIELDS, policies_payload)
from doctor import _module_status

HOST = os.environ.get("DATACLEAN_HOST", "127.0.0.1")
PORT = int(os.environ.get("DATACLEAN_PORT", "8799"))
HERE = Path(__file__).parent
SETTINGS = HERE / "settings.json"
STATIC = HERE / "frontend"
app = FastAPI(title="DataClean Agent")
active_turns = {}

def settings():
    if not SETTINGS.exists():
        _write_json_atomic(SETTINGS, {
            "model": "", "base_url": "", "api_key": "",
            "temperature": 0.0, "seed": None,
            "vlm_model": "", "vlm_base_url": "", "vlm_api_key": "",
        })
    return json.loads(SETTINGS.read_text())

def get_agent():
    s = settings()
    api_key = os.environ.get("DATACLEAN_LLM_API_KEY") or s.get("api_key")
    model = os.environ.get("DATACLEAN_LLM_MODEL") or s.get("model")
    base_url = os.environ.get("DATACLEAN_LLM_BASE_URL") or s.get("base_url", "")
    temperature_raw = os.environ.get("DATACLEAN_LLM_TEMPERATURE", s.get("temperature", 0.0))
    seed_raw = os.environ.get("DATACLEAN_LLM_SEED", s.get("seed"))
    if not model:
        return None
    if not api_key:
        # Local OpenAI-compatible servers commonly ignore authentication, while
        # the OpenAI client still requires a non-empty constructor value.
        api_key = "local-openai-compatible"
    try:
        temperature = None if temperature_raw in (None, "") else float(temperature_raw)
        seed = None if seed_raw in (None, "") else int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(409, f"Invalid Agent reproducibility setting: {exc}")
    if temperature is not None and not 0 <= temperature <= 2:
        raise HTTPException(409, "Agent temperature must be in [0, 2]")
    return Agent(api_key, base_url, model, temperature=temperature, seed=seed)

def workspace():
    p = database.current_workspace_dir()
    if not p:
        raise HTTPException(409, "Open a workspace first")
    return p

def _sanitize_obsolete_runtime_history(message):
    """Keep the audit log intact while removing a retired gate from model context."""
    if message.get("role") not in {"assistant", "tool"}:
        return message
    content = str(message.get("content") or "")
    obsolete_gate = (
        "allow_physical_deploy" in content
        or "Physical deployment requires either task-level preauthorization" in content
        or "Physical deployment is disabled for this task" in content
    )
    deployment_context = any(token in content.lower() for token in (
        "deploy", "deployment", "部署", "物理", "小车"
    ))
    if not (obsolete_gate and deployment_context):
        return message
    cleaned = dict(message)
    cleaned["content"] = (
        "[Obsolete result from a retired deployment-authorization gate. The current runtime "
        "does not use allow_physical_deploy. If the user asks to retry, call "
        "deploy_controller again and use its new result.]"
    )
    return cleaned


def prune_conversation_history(messages: list, keep_last_n_turns: int = 2,
                                recent_msg_cap: int = 4000, old_msg_cap: int = 1200,
                                max_old_turns: int = 8) -> list:
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [
        _sanitize_obsolete_runtime_history(m)
        for m in messages if m.get("role") != "system"
    ]
    if not rest:
        return sys_msgs

    def cap_msg(m, cap):
        c = m.get("content")
        if isinstance(c, str) and len(c) > cap:
            m = dict(m)
            head_len = int(cap * 0.65)
            tail_len = int(cap * 0.30)
            omitted = len(c) - head_len - tail_len
            m["content"] = c[:head_len] + f"\n\n... [Middle omitted {omitted} chars; see state.json] ...\n\n" + c[-tail_len:]
        return m

    turns = []
    cur = None
    for m in rest:
        if m.get("role") == "user":
            if cur:
                turns.append(cur)
            cur = [m]
        else:
            if cur is None:
                cur = []
            cur.append(m)
    if cur:
        turns.append(cur)
    if not turns:
        return sys_msgs + rest

    recent = turns[-keep_last_n_turns:]
    older = turns[:-keep_last_n_turns]
    dropped = 0
    if len(older) > max_old_turns:
        dropped = len(older) - max_old_turns
        older = older[-max_old_turns:]

    out = []
    for t in older:
        for m in t:
            if m.get("role") == "tool":
                m2 = dict(m)
                m2["content"] = "[Early tool output collapsed to save context; see round_ledger in state.json]"
                out.append(m2)
            else:
                out.append(cap_msg(m, old_msg_cap))
    for t in recent:
        for m in t:
            out.append(cap_msg(m, recent_msg_cap))

    result = sys_msgs + out
    if dropped > 0:
        result = [{
            "role": "user",
            "content": "[Notice] [Collapsed earlier {} turns (total {} turns); see <round_ledger> and <context>]".format(
                dropped, len(turns))
        }] + result
    return result


def _load_ledger(workspace_dir, task_id):
    try:
        st = _load(workspace_dir, branch=task_id)
        return (st or {}).get("round_ledger") or []
    except Exception:
        return []


def _format_ledger(ledger, max_rounds=3):
    if not ledger:
        return ""
    _RESERVED = {"stage", "round", "kpi"}
    by_round = {}
    for e in ledger:
        by_round.setdefault(e.get("round", "?"), []).append(e)
    def _rkey(r):
        return (0, r) if isinstance(r, (int, float)) else (1, str(r))
    rounds = sorted(by_round.keys(), key=_rkey)
    truncated = len(rounds) > max_rounds
    if truncated:
        rounds = rounds[-max_rounds:]
    lines = ["<round_ledger>  (Operational trace only; held-out ground truth is not exposed)"]
    if truncated:
        lines.append("[Showing recent {} rounds KPI; see state.json for full ledger]".format(max_rounds))
    for r in rounds:
        parts = []
        for e in by_round[r]:
            stage = e.get("stage")
            if e.get("kpi") is not None:
                parts.append(str(e["kpi"]))
            elif stage in LEDGER_FIELDS:
                parts.append("{} ".format(stage) + " ".join(
                    "{}={}".format(label, e.get(field))
                    for field, label in LEDGER_FIELDS[stage]
                    if e.get(field) is not None))
            else:
                extra = " ".join("{}={}".format(k, e[k])
                                 for k in e if k not in _RESERVED)
                if extra:
                    parts.append("{} {}".format(stage, extra))
        if parts:
            lines.append("round{}: ".format(r) + "; ".join(parts))
    lines.append("</round_ledger>")
    return "\n".join(lines)

def build_context(p, task_id):
    active_branch = task_id or "exp_1"
    try:
        observation = json.loads(
            Tool.get("get_pipeline_state").run(workspace_dir=p, branch=active_branch)
        )
    except Exception as exc:
        observation = {"branch": active_branch, "state_error": str(exc)}
    return "<context>\n" + json.dumps(observation, ensure_ascii=False, indent=2) + "\n</context>"

def prompt(path):
    return """You are the adaptive decision-maker for an autonomous-driving data-curation research system.
Workspace: {path}.

Mission: test whether observation-conditioned agent decisions outperform preregistered fixed scripts under the same data, tools, and budgets.

Research contract:
- The conversation is the task-control interface. Infer the current goal from the user's latest message together with relevant dialogue history; do not replace that interaction with a fixed pipeline or an internal mode switch. If the user explicitly requests one operation (for example, train the detector for 15 epochs), perform that operation, report its result, and stop. If the user explicitly requests a multi-step outcome (for example, complete one cleaning round), continue observing and deciding until that outcome is reached, blocked, or requires clarification. If the intended boundary is materially ambiguous, ask the user before changing state.
- Never infer downstream work merely because it is the conventional next pipeline stage. Conversely, when the user has clearly requested an end-to-end outcome, do not stop after each tool and ask for ritual approval. The action sequence must be driven by the conversational goal and current evidence, not by a hard-coded checklist.
- The agent is unsupervised. Never infer or request hidden evaluation labels, AUC, precision, recall, F1, purity, or other held-out results when making decisions.
- Distinguish user-directed task data composition from data cleaning. If the user explicitly requests per-source counts or caps (for example, one named source at 8000 and every other source at 500), call configure_task_dataset before any experiment episode or training. A source name explicitly supplied by the user may be used only to configure that requested view; never treat its semantic name as evaluation evidence. Do not reinterpret the request as a desired clean output and do not start detector training. If execution has already begun, explain that a new task is required.
- Task lifecycle is DRAFT → LOCKED → RUNNING → COMPLETED. Dataset design changes are allowed only in DRAFT. A stop=true decision is terminal and makes the task read-only; further experimentation requires a new task.
- In round t, D_t is the immutable round input. The canonical dependency path can produce C_t through detector, scoring, partition, and resolution, while controller training/deployment are optional. This is an available experiment graph, not a checklist: do not call a stage merely to complete the pipeline. Observation-only diagnosis, sensitivity analysis, bounded ablations, early stopping, and replanning are valid whenever the task specification and tool dependencies permit them.
- A round advances only through commit_round. clean_only commits D_(t+1)=C_t; deploy_collect_merge commits D_(t+1)=C_t union N_t after collected data is transferred.
- eval_controller returns a unique deployment_run_id. Transfer only that exact run with transfer_eval_results; never guess a "latest" file. The resulting N_t is a task-local CollectionArtifact and never becomes part of the workspace BaseDataset. Pass CollectionArtifact IDs to commit_round, not dataset source names.
- Respect evaluation_visibility. In heldout_only tasks, do not seek or infer deployment CTE; it is reserved for final comparison. In online_feedback tasks, returned CTE is valid observable feedback.
- Adaptive mode: choose actions from observable evidence and provide concise rationales. Fixed-baseline mode: tools apply the preregistered fixed_policy through the same action interface.
- Adaptive experiment protocol: freely inspect evidence and directly execute a justified legal action. Use propose_experiment_episode only when a bounded multi-action investigation or ablation genuinely benefits from an explicit hypothesis, scope, and success/replan signals; it is optional and must never become a ritual before ordinary work.
- Execute at most one consequential action per model step so you can observe its result. When an optional episode is active, stay within its scope and assess it at a milestone, failure, exhausted allowance, triggered replan condition, or useful checkpoint—not after every routine action.
- partition without a threshold and all state/statistical inspection calls are observation-only. If new read-only evidence invalidates an episode before any consequential action, withdraw_experiment_episode records why. If a restart leaves an action executing, use reconcile_interrupted_action and never blindly retry a possibly state-changing action.
- Do not reveal private chain-of-thought. The protocol fields should contain concise, experiment-auditable evidence and tradeoffs only.
- When the evidence supports a round-level continue/stop decision, record it explicitly with assess_stopping. Never change constraints after execution begins; a changed preregistered design requires a new task, while tactical choices inside the declared independent variables remain adaptive.
- Treat task pipeline, experimental variable, hard resource/safety constraints, and dataset fingerprints in <context> as authoritative. Do not change an experimental variable unless the user explicitly changes the task design.
- Physical deployment/evaluation is controlled through the conversation, as in the original lab implementation. When the user asks to deploy or retry deployment, call the physical tool; do not refuse based on allow_physical_deploy or historical assistant claims about that obsolete gate. A retry in a new user message is a new instruction and may repeat identical tool arguments. A tool failure or unresolved VLM result is not evidence that a sample is anomalous.
- Treat a structured tool result with status=failed/error/cancelled/rejected as a failed action. Inspect and replan; never describe it as completed.
- Within one user turn, never retry a failed operation by changing only rationale/request_basis. In a later turn, an explicit user retry instruction is new authority: call the tool again even with identical operational arguments instead of refusing from conversation history.
- Every Agent tool result uses {ok,status,code,data,error,retryable,state_changed,artifact_refs}. Read domain output from data; use top-level ok/status as the only execution-success signal.

Working style: communicate naturally and concisely in English; inspect current state before consequential actions; report evidence rather than invented metrics; avoid ritualistic restatement of every stage.""".replace("{path}", str(path))


@app.get("/api/workspace")
def get_workspace(): 
    return {"current": database.current_workspace()}


@app.get("/api/health")
def health():
    """Deployment preflight visible to both the UI and lab operators."""
    required_actions = {
        "configure_dataset", "configure_task_dataset", "define_task",
        "get_pipeline_state", "assess_stopping",
        "train_detector", "score_and_fit", "partition", "resolve", "evaluate",
        "train_controller", "deploy_controller", "eval_controller",
        "transfer_eval_results", "commit_round",
    }
    registered = set(Tool._registry)
    missing = sorted(required_actions - registered)
    module_status = _module_status()
    missing_modules = sorted(
        name for name, status in module_status.items() if not status.get("ok")
    )
    settings_error = None
    try:
        model = os.environ.get("DATACLEAN_LLM_MODEL") or settings().get("model")
    except Exception as exc:
        model = None
        settings_error = f"{type(exc).__name__}: {exc}"
    return {
        "ok": (
            not missing and not missing_modules
            and sys.version_info >= (3, 9) and settings_error is None
        ),
        "python": sys.version.split()[0],
        "python_supported": sys.version_info >= (3, 9),
        "missing_required_tools": missing,
        "missing_required_modules": missing_modules,
        "module_status": module_status,
        "dependency_import_errors": optional_dependency_errors(),
        "agent_model_configured": bool(model),
        "settings_error": settings_error,
        "workspace_open": database.current_workspace_dir() is not None,
    }

@app.post("/api/workspace")
async def open_workspace(request: Request):
    body = await request.json()
    p = Path(body.get("path", "")).expanduser()
    if not p.is_absolute() or not p.is_dir(): 
        raise HTTPException(400, "path must be an existing absolute directory")
    if active_turns: 
        raise HTTPException(409, "a turn is active")
    database.open_workspace(p)
    return {"current": database.current_workspace()}

@app.get("/api/settings")
def get_settings():
    s = settings()
    return {"model": s.get("model", ""), "base_url": s.get("base_url", ""), "api_key_set": bool(s.get("api_key")),
            "temperature": s.get("temperature", 0.0), "seed": s.get("seed"),
            "vlm_model": s.get("vlm_model", ""), "vlm_base_url": s.get("vlm_base_url", ""),
            "vlm_api_key_set": bool(s.get("vlm_api_key"))}

@app.post("/api/settings")
async def save_settings(request: Request):
    data = await request.json()
    old = settings()
    keys = ("model", "base_url", "api_key", "vlm_model", "vlm_base_url", "vlm_api_key")
    old.update({k: data[k].strip() for k in keys if k in data and isinstance(data[k], str)})
    try:
        if "temperature" in data:
            value = data["temperature"]
            old["temperature"] = None if value in (None, "") else float(value)
            if old["temperature"] is not None and not 0 <= old["temperature"] <= 2:
                raise HTTPException(400, "temperature must be in [0, 2]")
        if "seed" in data:
            value = data["seed"]
            old["seed"] = None if value in (None, "") else int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid reproducibility setting: {exc}")
    _write_json_atomic(SETTINGS, old)
    return {"ok": True}

@app.post("/api/configure")
async def configure(request: Request):
    p = workspace()
    data = await request.json()
    tool = Tool.get("configure_dataset")
    kwargs = {"workspace_dir": p}
    for k in ("dataset_path", "dataset_id", "include_sources", "exclude_sources", "max_per_source"):
        if k in data:
            kwargs[k] = data[k]
    try:
        result = json.loads(tool.run(**kwargs))
    except Exception as exc:
        raise HTTPException(400, str(exc))
    initialized, initialization_errors = [], {}
    if result.get("configured"):
        tasks_root = Path(p) / ROOT / TASKS_DIR
        for task_dir in sorted(tasks_root.iterdir()) if tasks_root.is_dir() else []:
            if not task_dir.is_dir():
                continue
            try:
                task_state = _load(p, branch=task_dir.name)
                if (
                    task_state.get("task_status", "DRAFT") == "DRAFT"
                    and not task_state.get("round_input_dataset")
                ):
                    Tool.get("configure_dataset").run(
                        workspace_dir=p, branch=task_dir.name
                    )
                    initialized.append(task_dir.name)
            except Exception as exc:
                initialization_errors[task_dir.name] = str(exc)
    result["initialized_d0_tasks"] = initialized
    result["d0_initialization_errors"] = initialization_errors
    return result

@app.get("/api/state")
def state():
    p = workspace()
    try:
        return json.loads(Tool.get("get_pipeline_state").run(workspace_dir=p))
    except ValueError as e:
        raise HTTPException(409, str(e))

@app.get("/api/dataset")
def dataset_status():
    """Workspace-level dataset resource status (independent of any task)."""
    p = workspace()
    try:
        reg = _dataset_config(p)
        return {"configured": True, "dataset_id": reg.get("dataset_id"),
                "dataset_mode": reg.get("dataset_mode"), "raw_samples": reg.get("raw_samples"),
                "source_composition": reg.get("source_composition"),
                "sources": [s.get("name") for s in reg.get("sources", [])],
                "dataset_path": reg.get("dataset_path")}
    except Exception as exc:
        return {"configured": False, "error": str(exc)}

@app.get("/api/tasks")
def tasks():
    p = workspace()
    return json.loads(Tool.get("list_tasks").run(workspace_dir=p))

@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    p = workspace()
    if task_id in active_turns:
        raise HTTPException(409, "Cannot delete a task while its agent turn is active")
    import shutil
    try:
        td = _task_dir(p, task_id, create=False)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if td.exists():
        shutil.rmtree(td)
    database.delete_task_chat(task_id)
    return {"ok": True}

@app.get("/api/policies")
def policies():
    """Single source of truth for stages, stage labels, policies, the default
    pipeline, and built-in task types. UI + prompt derive from this."""
    return policies_payload()

@app.post("/api/tasks")
async def create_task(request: Request):
    p = workspace()
    data = await request.json()
    task_id = (data.get("task_id") or "").strip()
    if not task_id:
        raise HTTPException(400, "task_id 必填")
    try:
        raw = Tool.get("define_task").run(
            workspace_dir=p,
            task_id=task_id,
            description=data.get("description", ""),
            independent_variable=data.get("independent_variable", ""),
            variants=data.get("variants") or [],
            baseline=data.get("baseline", ""),
            metrics=data.get("metrics") or [],
            seeds=int(data.get("seeds", 1) or 1),
            budget=data.get("budget"),
            depends_on=data.get("depends_on", ""),
            hypothesis=data.get("hypothesis", ""),
            constraints=data.get("constraints"),
            pipeline=data.get("pipeline"),
            vlm=data.get("vlm"),
            execution_mode=data.get("execution_mode", "adaptive_agent"),
            fixed_policy=data.get("fixed_policy"),
            experimental_controls=data.get("experimental_controls"),
            transition_policy=data.get("transition_policy", "clean_only"),
            dataset_subset=data.get("dataset_subset"),
            evaluation_visibility=data.get("evaluation_visibility"),
            paired_with=data.get("paired_with", ""),
        )
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    p = workspace()
    try:
        td = _task_dir(p, task_id, create=False)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not td.is_dir():
        raise HTTPException(404, "task not found: " + task_id)
    spec, state, log = {}, None, ""
    if (td / "task_spec.json").exists():
        spec = json.loads((td / "task_spec.json").read_text())
    if (td / "state.json").exists():
        state = json.loads((td / "state.json").read_text())
    if (td / "decision_log.md").exists():
        log = (td / "decision_log.md").read_text()
    session = {}
    sp = td / "session.json"
    if sp.exists():
        try:
            session = json.loads(sp.read_text())
        except Exception:
            session = {}
    artifacts = []
    art_dir = td / "artifacts"
    if art_dir.is_dir():
        for f in sorted(art_dir.iterdir()):
            if f.is_file():
                artifacts.append({"name": f.name, "size": f.stat().st_size})
    protocol = {}
    protocol_path = td / "agent_protocol.json"
    if protocol_path.exists():
        try:
            protocol = json.loads(protocol_path.read_text())
        except Exception:
            protocol = {"error": "agent_protocol.json is unreadable"}
    trajectory_path = td / "agent_trajectory.jsonl"
    trajectory_events = 0
    if trajectory_path.exists():
        with trajectory_path.open(encoding="utf-8") as handle:
            trajectory_events = sum(1 for line in handle if line.strip())
    return {"task_id": task_id, "spec": spec, "state": state, "decision_log": log,
            "session": session, "artifacts": artifacts, "agent_protocol": protocol,
            "trajectory_events": trajectory_events}


@app.get("/api/tasks/{task_id}/trajectory")
def task_trajectory(task_id: str, limit: int = 200):
    """User-visible audit trajectory; it contains no held-out evaluation metrics."""
    p = workspace()
    try:
        path = _task_dir(p, task_id, create=False) / "agent_trajectory.jsonl"
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not path.exists():
        return {"task_id": task_id, "events": [], "total_events": 0, "truncated": False}
    if not 1 <= limit <= 2000:
        raise HTTPException(400, "limit must be in [1, 2000]")
    recent_lines = deque(maxlen=limit)
    total_events = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                total_events += 1
                recent_lines.append((line_number, line))
    events = []
    for line_number, line in recent_lines:
        try:
            events.append(json.loads(line))
        except Exception:
            events.append({"event": "unreadable", "line": line_number})
    return {
        "task_id": task_id,
        "events": events,
        "total_events": total_events,
        "truncated": total_events > len(events),
    }

@app.get("/api/tasks/{task_id}/artifact/{name}")
def task_artifact(task_id: str, name: str):
    p = workspace()
    try:
        art_dir = (_task_dir(p, task_id, create=False) / "artifacts").resolve()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if ".." in name or "/" in name:
        raise HTTPException(400, "invalid artifact name")
    fp = (art_dir / name).resolve()
    if not fp.is_relative_to(art_dir) or not fp.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(fp)

@app.get("/api/tasks/{task_id}/messages")
def task_messages(task_id: str):
    p = workspace()
    td = _task_dir(p, task_id, create=False)
    if not (td / "state.json").is_file():
        raise HTTPException(404, "task not found: " + task_id)
    c = database.get_task_chat(task_id)
    return {"messages": c.messages()}


@app.post("/api/tasks/{task_id}/evaluate")
async def evaluate_task(task_id: str, request: Request):
    """User-controlled hidden evaluation; this tool is not exposed to the agent."""
    p = workspace()
    try:
        td = _task_dir(p, task_id, create=False)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not (td / "state.json").is_file():
        raise HTTPException(404, "task not found: " + task_id)
    try:
        evaluator = Tool.get("evaluate")
    except KeyError:
        raise HTTPException(409, "Evaluation dependencies are not installed")
    body = await request.json()
    try:
        return json.loads(evaluator.run(
            workspace_dir=p,
            branch=task_id,
            target=body.get("target", "cleandata"),
            threshold_mode=body.get("threshold_mode", "partition"),
            detector_id=body.get("detector_id"),
        ))
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.get("/api/chats")
def chats(): 
    workspace()
    return [c.__dict__ for c in database.Chat.all()]

@app.post("/api/chats")
def create_chat(): 
    workspace()
    return database.Chat.create().__dict__

@app.get("/api/chats/{chat_id}")
def read_chat(chat_id: int):
    workspace()
    c = database.Chat.get(chat_id)
    if not c: 
        raise HTTPException(404, "chat not found")
    return {"messages": c.messages()}

@app.post("/api/chats/{chat_id}/stop")
def stop(chat_id: int):
    t = active_turns.get(chat_id)
    if t: 
        t.interrupt()
    return {"ok": True}


@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str):
    t = active_turns.get(task_id)
    if t:
        t.interrupt()
    return {"ok": True}

@app.post("/api/chats/{chat_id}")
async def chat(chat_id: int, request: Request):
    p = workspace()
    a = get_agent()
    if not a: 
        raise HTTPException(409, "Configure an OpenAI-compatible model in settings.json or the UI")
    body = await request.json()
    text = body.get("message", "").strip()
    if not text: 
        raise HTTPException(400, "message is required")
    task_id = (body.get("task_id") or "").strip()
    if not task_id:
        tasks_list = tasks()
        if not tasks_list:
            raise HTTPException(409, "Create a task before starting an agent turn")
        task_id = tasks_list[0]["task_id"]
    try:
        task_dir = _task_dir(p, task_id, create=False)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not (task_dir / "state.json").is_file():
        raise HTTPException(404, "task not found: " + task_id)
    c = database.get_task_chat(task_id)  # 每个任务独立线程，切换即隔离

    def generate():
        with c.lock():
            c.append("user", text)
            print(f"[chat] task={task_id!r} chat_id={chat_id}", flush=True)

            ledger_block = _format_ledger(_load_ledger(p, task_id))
            system_content = build_context(p, task_id) + "\n\n" + prompt(p)
            if ledger_block:
                system_content += "\n\n" + ledger_block
            pruned_history = prune_conversation_history(c.messages())

            def run_streaming(history):
                payload = [{"role": "system", "content": system_content}] + history
                turn = a.run_turn(payload, Tool, {
                    "workspace_dir": p,
                    "branch": task_id,
                    "user_request_text": text,
                })
                active_turns[task_id] = turn
                active_turns[chat_id] = turn  # backwards-compatible stop endpoint
                try:
                    for ev in turn:
                        yield ev
                finally:
                    active_turns.pop(task_id, None)
                    active_turns.pop(chat_id, None)
                return turn

            def consume(gen):
                while True:
                    try:
                        ev = next(gen)
                    except StopIteration as si:
                        return si.value
                    yield json.dumps(ev) + "\n"

            gen = run_streaming(pruned_history)
            turn = yield from consume(gen)

            try:
                for msg in turn.messages:
                    c.append(msg["role"], msg.get("content"), msg.get("tool_calls"), msg.get("tool_call_id"))
                yield json.dumps({"type": "done", "usage": turn.usage, "stopped": turn.stopped}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "text": str(e)}) + "\n"
            finally:
                active_turns.pop(task_id, None)
                active_turns.pop(chat_id, None)

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/")
def index(): 
    return FileResponse(STATIC / "index.html")

app.mount("/", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
