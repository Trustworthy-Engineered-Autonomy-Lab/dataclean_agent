import json

from .base import Tool
from .decision_policy import effective_action, record_decision
from .utils import _load, _save, record_observation


class AssessStopping(Tool):
    name = "assess_stopping"
    description = (
        "Record an explicit continue/stop decision through the shared adaptive/fixed-policy "
        "interface. Fixed baselines stop only at their preregistered max_rounds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "stop": {"type": "boolean"},
            "rationale": {
                "type": "string",
                "description": "Observation-based reason to stop or continue.",
            },
        },
        "required": ["stop", "rationale"],
    }

    def run(self, stop, rationale, branch="main", workspace_dir=None, **_):
        state = _load(workspace_dir, branch=branch)
        if state.get("task_status") == "COMPLETED":
            raise ValueError("Completed tasks are immutable; create a new task to run another experiment")
        proposed = {"stop": bool(stop)}
        effective, source = effective_action(state, "stopping", proposed)
        if source == "fixed_policy":
            max_rounds = int(effective["max_rounds"])
            stop = int(state.get("round", 0)) >= max_rounds
            effective = {**effective, "stop": stop}
            rationale = rationale or f"Preregistered maximum of {max_rounds} completed rounds"
        else:
            stop = bool(effective.get("stop", stop))
            if not str(rationale).strip():
                raise ValueError("Adaptive stopping decisions require an observation-based rationale")
            max_rounds = (state.get("constraints") or {}).get("max_rounds")
            if max_rounds is not None and int(state.get("round", 0)) >= int(max_rounds):
                stop = True
                effective = {**effective, "stop": True}
                rationale = f"Experimental maximum of {int(max_rounds)} committed rounds reached"
            elif stop and not (state.get("constraints") or {}).get("allow_voluntary_terminate", True):
                raise ValueError("Voluntary termination is disabled by this task's preregistered constraints")

        state["termination_required"] = stop
        state["termination_reason"] = rationale if stop else None
        if stop:
            state["task_status"] = "COMPLETED"
        observation = {
            "round": int(state.get("round", 0)),
            "round_status": state.get("round_status"),
            "deployments": int(state.get("deployments", 0)),
        }
        record_decision(
            state, "stopping", proposed, effective, rationale, source,
            observation=observation,
        )
        summary = {
            "stop": stop, "decision_source": source,
            "task_status": state.get("task_status"), **observation,
        }
        record_observation(state, "assess_stopping", summary, workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, state, branch=branch)
        return json.dumps(summary, ensure_ascii=False)
