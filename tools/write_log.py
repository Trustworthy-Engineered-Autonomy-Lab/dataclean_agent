import json
import time
from .base import Tool
from .utils import _load, _save, _d5, _task_dir, _ensure_constraints

class WriteLog(Tool):
    name = "write_log"
    description = "Append an auditable decision log entry (D1-D5) for experimental auditability."
    parameters = {
        "type": "object",
        "properties": {
            "decision_point": {"type": "string", "enum": ["D1", "D2", "D3", "D4", "D5"]},
            "decision": {"type": "string"},
            "rationale": {"type": "string"}
        },
        "required": ["decision_point", "decision", "rationale"]
    }
    
    def run(self, decision_point, decision, rationale, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        gate_warning = None

        if decision_point == "D5":
            gate = _d5(s)
            if decision in ("continue_cleaning", "terminate", "deploy") and decision not in gate["allowed"]:
                gate_warning = gate["reason"]
            if decision == "continue_cleaning":
                s["skip_streak"] = s.get("skip_streak", 0) + 1

        entry = {
            "round": s["round"],
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "point": decision_point,
            "decision": decision,
            "rationale": rationale,
            "gate_warning": gate_warning,
            "branch": branch
        }
        s.setdefault("history", []).append(entry)

        _save(workspace_dir, s, branch=branch)
        _append_decision_log(workspace_dir, branch, entry)
        return json.dumps({"logged": entry, "next_d5_gate": _d5(s)}, ensure_ascii=False)


def _append_decision_log(workspace_dir, branch, entry):
    log_path = _task_dir(workspace_dir, branch) / "decision_log.md"
    forced_note = f" ⚠️ Warning: {entry['gate_warning']}" if entry.get("gate_warning") else ""
    block = (
        f"### [Round {entry['round']}] {entry['point']} · Task={branch} · {entry['time']}{forced_note}\n"
        f"- **Decision**: {entry['decision']}\n"
        f"- **Rationale**: {entry['rationale']}\n\n"
        f"---\n\n"
    )
    if not log_path.exists():
        header = (
            f"# Decision Log · Task `{branch}`\n\n"
            f"> Automatically maintained by write_log to record auditable D1-D5 decision traces.\n\n"
            f"---\n\n"
        )
        log_path.write_text(header + block, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(block)