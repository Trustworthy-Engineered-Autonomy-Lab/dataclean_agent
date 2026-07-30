__all__ = ["_default_constraints", "_ensure_constraints", "_d5"]


def _default_constraints(branch: str = None) -> dict:
    return {
        "enforce_first_deploy": False,
        "max_skip_streak": None,
        "max_deployments": 10,
        "lock_round0_detector": False,
        "allow_voluntary_terminate": True,
    }


def _ensure_constraints(state: dict, branch: str = None) -> dict:
    if not isinstance(state.get("constraints"), dict) or not state["constraints"]:
        state["constraints"] = _default_constraints(branch)
    return state


def _d5(state):
    c = state.get("constraints") or {}
    if state.get("termination_required"):
        return {"allowed": ["terminate"], "advisory": False,
                "reason": state.get("termination_reason", "deployment termination criterion reached")}
    notes = []
    max_dep = c.get("max_deployments")
    if max_dep is not None:
        remaining = max(0, max_dep - state.get("deployments", 0))
        notes.append(f"Max deployments limit {max_dep}, remaining {remaining}")
    if c.get("locked_threshold") is not None:
        notes.append(f"Threshold locked to {c['locked_threshold']} (fixed experiment arm)")
    if c.get("enforce_first_deploy"):
        notes.append("First round deployment recommended")
    if c.get("max_skip_streak") is not None:
        notes.append(f"Consecutive continue recommended <= {c['max_skip_streak']} rounds")
    return {
        "allowed": ["deploy", "continue_cleaning", "terminate"],
        "advisory": True,
        "reason": "Agent autonomous decision; constraints for reference: " + ("; ".join(notes) if notes else "No extra constraints"),
    }
