import json
from .base import Tool
from .utils import _load, _save, _ensure_constraints, _d5

class SetConstraints(Tool):
    name = "set_constraints"
    agent_exposed = False
    description = (
        "Adjust resource constraints before an adaptive task starts. Fixed baselines and tasks "
        "with recorded decisions are immutable; create a new task to change their design."
    )
    parameters = {
        "type": "object",
        "properties": {
            "enforce_first_deploy": {
                "type": "boolean",
                "description": "Whether to enforce deployment at round 0."
            },
            "max_skip_streak": {
                "type": "integer",
                "description": "Max consecutive cleaning rounds without deployment. Use -1 for unlimited."
            },
            "max_deployments": {
                "type": "integer",
                "description": "Maximum deployment limit before hard termination. Use -1 for unlimited."
            },
            "max_rounds": {
                "type": "integer",
                "description": "Maximum number of committed cleaning rounds. Use -1 for unlimited."
            },
            "max_vlm_calls_total": {
                "type": "integer",
                "description": "Total VLM review-call budget across rounds. Use -1 for unlimited."
            },
            "max_collection_images_total": {"type": "integer"},
            "max_detector_train_epochs_total": {"type": "integer"},
            "max_controller_train_epochs_total": {"type": "integer"},
            "lock_round0_detector": {
                "type": "boolean",
                "description": "Whether round 0 detector hyper-parameters are locked."
            },
            "allow_voluntary_terminate": {
                "type": "boolean",
                "description": "Whether voluntary termination is permitted."
            },
        },
        "required": []
    }

    def run(self, enforce_first_deploy=None,
            max_skip_streak=None, max_deployments=None, max_rounds=None,
            max_vlm_calls_total=None, max_detector_train_epochs_total=None,
            max_controller_train_epochs_total=None, lock_round0_detector=None,
            max_collection_images_total=None,
            allow_voluntary_terminate=None,
            branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if s.get("execution_mode") == "fixed_baseline":
            raise ValueError("Fixed-baseline constraints are preregistered and immutable")
        if s.get("decision_trace") or s.get("latest_scores") or int(s.get("round", 0)) > 0:
            raise ValueError("Constraints cannot change after task execution begins; create a new task")
        mapping = {
            "enforce_first_deploy": enforce_first_deploy,
            "max_skip_streak": max_skip_streak,
            "max_deployments": max_deployments,
            "max_rounds": max_rounds,
            "max_vlm_calls_total": max_vlm_calls_total,
            "max_detector_train_epochs_total": max_detector_train_epochs_total,
            "max_controller_train_epochs_total": max_controller_train_epochs_total,
            "max_collection_images_total": max_collection_images_total,
            "lock_round0_detector": lock_round0_detector,
            "allow_voluntary_terminate": allow_voluntary_terminate,
        }
        changed = {}
        for k, v in mapping.items():
            if v is None:
                continue
            if k in (
                "max_skip_streak", "max_deployments", "max_rounds", "max_vlm_calls_total",
                "max_detector_train_epochs_total", "max_controller_train_epochs_total",
                "max_collection_images_total",
            ):
                if not isinstance(v, int):
                    raise ValueError(f"{k} must be an integer")
                if v == -1:
                    v = None
                elif v < 0:
                    raise ValueError(f"{k} must be -1 (unlimited) or non-negative")
            if k in ("enforce_first_deploy", "lock_round0_detector", "allow_voluntary_terminate") and not isinstance(v, bool):
                raise ValueError(f"{k} must be a boolean")
            s["constraints"][k] = v
            changed[k] = v

        if not changed:
            return json.dumps({
                "branch": branch,
                "changed": {},
                "note": "No non-null constraint parameters provided; policy unchanged",
                "constraints": s["constraints"],
                "next_d5_gate": _d5(s)
            }, ensure_ascii=False)

        _save(workspace_dir, s, branch=branch)
        return json.dumps({
            "branch": branch,
            "changed": changed,
            "constraints": s["constraints"],
            "next_d5_gate": _d5(s)
        }, ensure_ascii=False)
