import json
import re
import time
from .base import Tool
from .utils import _save, _task_dir, _default_constraints
from .policies import default_pipeline_for, validate_pipeline

_TASK_ID_RE = re.compile(r"^[\w\-]+$")
_RESET_FIELDS = {
    "round": 0,
    "deployments": 0,
    "skip_streak": 0,
    "vlm_budget_current_round": 0,
    "vlm_budget_used_this_round": 0,
    "history": [],
    "latest_observation": {},
    "latest_scores": None,
    "latest_partition": None,
    "last_deployed_cte": None,
    "best_cte": None,
    "active_detector": None,
    "active_clean_dataset": None,
    "active_controller": None,
    "termination_required": False,
    "termination_reason": None,
    "last_deployment": None,
}

class DefineTask(Tool):
    name = "define_task"
    description = (
        "Register a structured task specification (main pipeline or ablation/comparative experiment) "
        "and create an isolated directory under tasks/<task_id>/."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Unique task ID (alphanumeric, underscores, hyphens only), acting as directory name."
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of the task."
            },
            "independent_variable": {
                "type": "string",
                "description": "Independent variable being studied (e.g. 'threshold_strategy')."
            },
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of variant values/configurations to compare."
            },
            "baseline": {
                "type": "string",
                "description": "Baseline configuration or comparison task ID."
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of evaluation metrics to track."
            },
            "seeds": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of seeds/runs, defaults to 1."
            },
            "budget": {
                "type": "integer",
                "description": "Optional VLM budget cap for this task."
            },
            "depends_on": {
                "type": "string",
                "description": "Optional parent task ID."
            },
            "hypothesis": {
                "type": "string",
                "description": "Optional experimental hypothesis."
            },
            "constraints": {
                "type": "object",
                "description": "Optional constraint overrides for non-main branches."
            },
            "pipeline": {
                "type": "object",
                "description": "Optional declarative pipeline stage-to-policy mapping overrides."
            },
            "vlm": {
                "type": "object",
                "description": "Optional per-task VLM reviewer declaration (model/base_url/api_key)."
            }
        },
        "required": ["task_id"]
    }

    def run(self, task_id, description="", independent_variable="", variants=None,
            baseline="", metrics=None, seeds=1, budget=None, depends_on="", hypothesis="",
            constraints=None, pipeline=None, vlm=None, workspace_dir=None, **_):
        if not _TASK_ID_RE.match(task_id):
            raise ValueError("task_id must contain only letters, numbers, underscores, and hyphens.")

        td = _task_dir(workspace_dir, task_id)

        eff = _default_constraints(task_id)
        if isinstance(constraints, dict):
            for k, v in constraints.items():
                if v is not None:
                    eff[k] = v

        eff_pipeline = default_pipeline_for()
        if isinstance(pipeline, dict):
            eff_pipeline.update({k: v for k, v in pipeline.items() if v is not None})
        ok, errs = validate_pipeline(eff_pipeline)
        if not ok:
            raise ValueError("Pipeline validation failed: " + "; ".join(errs))

        spec = {
            "task_id": task_id,
            "description": description or task_id,
            "independent_variable": independent_variable,
            "variants": variants or [],
            "baseline": baseline,
            "metrics": metrics or [],
            "seeds": int(seeds),
            "budget": budget,
            "depends_on": depends_on,
            "hypothesis": hypothesis,
            "constraints": eff,
            "pipeline": eff_pipeline,
            "vlm": vlm,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (td / "task_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))

        state_path = td / "state.json"
        if not state_path.exists():
            new_state = dict(_RESET_FIELDS)
            new_state["vlm_budget_per_round"] = int(budget) if budget is not None else 500
            new_state["constraints"] = eff
            new_state["pipeline"] = eff_pipeline
            new_state["vlm"] = vlm
            _save(workspace_dir, new_state, branch=task_id)

        return json.dumps({
            "defined": task_id,
            "task_dir": str(td),
            "state_initialized": state_path.exists(),
            "spec": spec
        }, ensure_ascii=False)
