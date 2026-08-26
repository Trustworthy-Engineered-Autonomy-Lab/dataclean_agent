__all__ = ["_default_constraints", "_ensure_constraints"]


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
