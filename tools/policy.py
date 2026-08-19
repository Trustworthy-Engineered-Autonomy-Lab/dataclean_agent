__all__ = ["_default_constraints", "_ensure_constraints", "_d5"]


def _default_constraints(branch: str = None) -> dict:
    return {
        "enforce_first_deploy": False,
        "max_skip_streak": None,
        "max_deployments": 10 if branch == "main" else None,
        "max_collection_images_total": None,
        "max_rounds": None,
        "max_vlm_calls_total": None,
        "max_detector_train_epochs_total": None,
        "max_controller_train_epochs_total": None,
        "lock_round0_detector": False,
        "allow_voluntary_terminate": True,
    }


def _ensure_constraints(state: dict, branch: str = None) -> dict:
    defaults = _default_constraints(branch)
    current = state.get("constraints")
    if not isinstance(current, dict):
        current = {}
    state["constraints"] = {**defaults, **current}
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
    if c.get("enforce_first_deploy"):
        notes.append("First round deployment recommended")
    if c.get("max_skip_streak") is not None:
        notes.append(f"Consecutive continue recommended <= {c['max_skip_streak']} rounds")
    if c.get("max_rounds") is not None:
        notes.append(f"Maximum committed rounds {c['max_rounds']}")
    if c.get("max_collection_images_total") is not None:
        notes.append(
            f"Collection images {state.get('collection_images_budget_used', 0)}/"
            f"{c['max_collection_images_total']}"
        )
    if c.get("max_detector_train_epochs_total") is not None:
        notes.append(
            f"Detector epochs {state.get('detector_train_epochs_used', 0)}/"
            f"{c['max_detector_train_epochs_total']}"
        )
    if c.get("max_controller_train_epochs_total") is not None:
        notes.append(
            f"Controller epochs {state.get('controller_train_epochs_used', 0)}/"
            f"{c['max_controller_train_epochs_total']}"
        )
    return {
        "allowed": ["deploy", "continue_cleaning", "terminate"],
        "advisory": True,
        "reason": "Agent autonomous decision; constraints for reference: " + ("; ".join(notes) if notes else "No extra constraints"),
    }
