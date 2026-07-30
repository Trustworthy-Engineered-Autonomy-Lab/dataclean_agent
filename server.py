import json
import re
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import database
from agent import Agent
from tools import Tool
from tools.utils import ROOT, TASKS_DIR, _load, _load_session, _dataset_config
from tools.policies import (describe_task_types, describe_default_pipeline,
                            LEDGER_FIELDS, policies_payload)

HOST, PORT = "127.0.0.1", 8766
HERE = Path(__file__).parent
SETTINGS = HERE / "settings.json"
STATIC = HERE / "frontend"
app = FastAPI(title="DataClean Agent")
active_turns = {}

def settings():
    if not SETTINGS.exists():
        SETTINGS.write_text(json.dumps({
            "model": "", "base_url": "", "api_key": "",
            "vlm_model": "", "vlm_base_url": "", "vlm_api_key": "",
        }, indent=2))
    return json.loads(SETTINGS.read_text())

def get_agent():
    s = settings()
    if not (s.get("api_key") and s.get("model")):
        return None
    return Agent(s["api_key"], s.get("base_url", ""), s["model"])

def workspace():
    p = database.current_workspace_dir()
    if not p:
        raise HTTPException(409, "Open a workspace first")
    return p

def prune_conversation_history(messages: list, keep_last_n_turns: int = 2,
                                recent_msg_cap: int = 4000, old_msg_cap: int = 1200,
                                max_old_turns: int = 8) -> list:
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
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
            "role": "system",
            "content": "[Collapsed earlier {} turns (total {} turns); see <round_ledger> and <context>]".format(
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
    lines = ["<round_ledger>  (Cross-round KPI, ground truth in state.json)"]
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
    block = ["<context>"]
    block.append(f"current_task: {active_branch}")
    if task_id:
        spec_path = Path(p) / ROOT / TASKS_DIR / task_id / "task_spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text())
                if spec.get("description") and spec["description"] != active_branch:
                    block.append("task_description: " + spec["description"])
            except Exception:
                pass
    block.append("</context>")
    return "\n".join(block)

def prompt(path):
    return """You are an intelligent AI pair programmer specialized in autonomous driving dataset curation and quality analysis.
Workspace: {path}.

═══ Core Directives ═══
1. Pair Programming Partner: Communicate with the user naturally and professionally in Chinese. Discuss intent, analyze data, and carry out dataset curation tasks collaboratively.
2. Ground Truth & Empirical Evidence: Rely strictly on real state in <context> and execution outputs returned by tools. Never guess or hallucinate metrics.
3. Dialogue-Driven Execution: Focus on the user's natural language requests. Execute appropriate tools (such as dataset configuration, detector training, score_and_partition, train_and_deploy, or run_python) directly when requested.
4. Clean Academic Style: Maintain a clean, concise, academic communication style. Avoid emojis or rigid pre-scripted template options.""".replace("{path}", str(path))

def _final_assistant_text(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
            return m["content"]
    return ""

def _asserts_wrong_branch(text, task_id):
    for pat in (
        r"on (?:the )?([\w\-]+) branch",
        r"active branch is ([\w\-]+)",
        r"位于\s*[「『]?([\w\-]+)[」』]?\s*分支",
        r"我们在\s*[「『]?([\w\-]+)[」』]?\s*分支",
        r"当前[^。]*?分支[是为：:]\s*[「『]?([\w\-]+)[」』]?",
    ):
        for claimed in re.findall(pat, text, re.IGNORECASE):
            claimed = claimed.strip()
            if claimed and claimed != task_id:
                return True
    return False

@app.get("/api/workspace")
def get_workspace(): 
    return {"current": database.current_workspace()}

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
    return {"model": s.get("model", ""), "base_url": s.get("base_url", ""), "api_key": s.get("api_key", ""),
            "vlm_model": s.get("vlm_model", ""), "vlm_base_url": s.get("vlm_base_url", ""),
            "vlm_api_key": s.get("vlm_api_key", "")}

@app.post("/api/settings")
async def save_settings(request: Request):
    data = await request.json()
    old = settings()
    keys = ("model", "base_url", "api_key", "vlm_model", "vlm_base_url", "vlm_api_key")
    old.update({k: data[k].strip() for k in keys if k in data and isinstance(data[k], str)})
    SETTINGS.write_text(json.dumps(old, indent=2))
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
    return json.loads(tool.run(**kwargs))

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
    except Exception:
        return {"configured": False}

@app.get("/api/tasks")
def tasks():
    p = workspace()
    return json.loads(Tool.get("list_tasks").run(workspace_dir=p))

@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    p = workspace()
    import shutil
    td = Path(p) / ROOT / TASKS_DIR / task_id
    if td.exists():
        shutil.rmtree(td, ignore_errors=True)
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
        )
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    p = workspace()
    td = Path(p) / ROOT / TASKS_DIR / task_id
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
    return {"task_id": task_id, "spec": spec, "state": state, "decision_log": log,
            "session": session, "artifacts": artifacts}

@app.get("/api/tasks/{task_id}/artifact/{name}")
def task_artifact(task_id: str, name: str):
    p = workspace()
    art_dir = (Path(p) / ROOT / TASKS_DIR / task_id / "artifacts").resolve()
    if ".." in name or "/" in name:
        raise HTTPException(400, "invalid artifact name")
    fp = (art_dir / name).resolve()
    if not str(fp).startswith(str(art_dir)) or not fp.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(fp)

@app.get("/api/tasks/{task_id}/messages")
def task_messages(task_id: str):
    p = workspace()
    c = database.get_task_chat(task_id)
    return {"messages": c.messages()}

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
        task_id = tasks_list[0]["task_id"] if tasks_list else "exp_1"
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
                turn = a.run_turn(payload, Tool, {"workspace_dir": p, "branch": task_id})
                active_turns[chat_id] = turn
                for ev in turn:
                    yield ev
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

            final_text = _final_assistant_text(turn.messages)
            if final_text and _asserts_wrong_branch(final_text, task_id):
                c.append("system", "[Correction] The active branch in your previous response contradicted current context. Active branch is '"
                                  + task_id + "'. Please answer based strictly on <context>.")
                pruned_history = prune_conversation_history(c.messages())
                gen2 = run_streaming(pruned_history)
                turn = yield from consume(gen2)

            try:
                for msg in turn.messages:
                    c.append(msg["role"], msg.get("content"), msg.get("tool_calls"), msg.get("tool_call_id"))
                yield json.dumps({"type": "done", "usage": turn.usage, "stopped": turn.stopped}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "text": str(e)}) + "\n"
            finally:
                active_turns.pop(chat_id, None)

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/")
def index(): 
    return FileResponse(STATIC / "index.html")

app.mount("/", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)