import json
import re
import time
from .base import Tool
from .utils import (
    _dataset_config,
    _dataset_registry_path,
    _default_constraints,
    _load,
    _raw_records,
    _save,
    _task_dir,
    _write_dataset_snapshot,
    _write_json_atomic,
)
from .policies import default_pipeline_for, validate_pipeline
from .decision_policy import DECISION_FIELDS, validate_fixed_policy

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PAIRED_BUDGET_FIELDS = (
    "max_rounds", "max_vlm_calls_total", "max_detector_train_epochs_total",
    "max_controller_train_epochs_total", "max_deployments",
    "max_collection_images_total",
)
_RESET_FIELDS = {
    "schema_version": 2,
    "task_status": "DRAFT",
    "round": 0,
    "round_status": "ready",
    "deployments": 0,
    "collection_images_budget_used": 0,
    "skip_streak": 0,
    "vlm_budget_current_round": 0,
    "vlm_budget_used_this_round": 0,
    "detector_train_epochs_used": 0,
    "controller_train_epochs_used": 0,
    "history": [],
    "latest_observation": {},
    "latest_scores": None,
    "latest_partition": None,
    "last_deployed_cte": None,
    "best_cte": None,
    "active_detector": None,
    "active_clean_dataset": None,
    "previous_clean_dataset": None,
    "round_input_dataset": None,
    "round_input_count": None,
    "round_input_fingerprint": None,
    "pending_collection_ids": [],
    "consumed_collection_ids": [],
    "unconsumed_collections": [],
    "deployment_runs": [],
    "round_history": [],
    "active_controller": None,
    "termination_required": False,
    "termination_reason": None,
    "last_deployment": None,
}

class DefineTask(Tool):
    name = "define_task"
    agent_exposed = False
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
                "description": "Optional VLM API-call cap per round."
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
            "execution_mode": {
                "type": "string",
                "enum": ["adaptive_agent", "fixed_baseline"],
                "description": "Adaptive agent decisions or a preregistered fixed baseline policy.",
            },
            "fixed_policy": {
                "type": "object",
                "description": "Required for fixed_baseline; fixed actions such as threshold_rule, VLM settings, detector strategy, and stopping rule.",
            },
            "experimental_controls": {
                "type": "object",
                "description": "Only the experimental variables that must stay fixed in an adaptive arm, keyed by decision name."
            },
            "transition_policy": {
                "type": "string",
                "enum": ["clean_only", "deploy_collect_merge"],
                "description": "Default cross-round data transition; the adaptive agent may justify changing it unless locked by the experiment.",
            },
            "dataset_subset": {
                "type": "object",
                "description": "Optional task-scoped include_sources/exclude_sources/max_per_source used to freeze D_0."
            },
            "vlm": {
                "type": "object",
                "description": "Optional per-task VLM reviewer declaration (model/base_url only; secrets stay in settings/environment)."
            },
            "evaluation_visibility": {
                "type": "string",
                "enum": ["online_feedback", "heldout_only"],
                "description": "Whether deployment CTE is visible to the Agent or retained only for final evaluation."
            },
            "paired_with": {
                "type": "string",
                "description": "Optional comparison task ID whose D_0, hard budgets, and evaluation visibility must match."
            }
        },
        "required": ["task_id"]
    }

    def run(self, task_id, description="", independent_variable="", variants=None,
            baseline="", metrics=None, seeds=1, budget=None, depends_on="", hypothesis="",
            constraints=None, pipeline=None, vlm=None, execution_mode="adaptive_agent",
            fixed_policy=None, experimental_controls=None, transition_policy="clean_only",
            dataset_subset=None, evaluation_visibility=None,
            paired_with="", workspace_dir=None, **_):
        if not _TASK_ID_RE.match(task_id):
            raise ValueError("task_id must contain only letters, numbers, underscores, and hyphens.")

        td = _task_dir(workspace_dir, task_id, create=False)
        state_path = td / "state.json"
        spec_path = td / "task_spec.json"
        if state_path.exists() or spec_path.exists():
            raise ValueError(
                f"Task '{task_id}' already exists. Create a new task ID so its preregistered spec "
                "cannot silently diverge from existing state."
            )

        paired_state = None
        if paired_with:
            if not _TASK_ID_RE.fullmatch(str(paired_with)) or paired_with == task_id:
                raise ValueError("paired_with must name a different valid task ID")
            paired_state = _load(workspace_dir, branch=paired_with)
            if not paired_state.get("round_input_fingerprint"):
                raise ValueError("paired_with task has no frozen D_0 fingerprint")

            # Omitted fairness controls inherit from the reference arm. Explicit
            # differences are still rejected below.
            if dataset_subset is None:
                dataset_subset = dict(paired_state.get("dataset_subset") or {})
            inherited_constraints = dict(constraints or {})
            for key in _PAIRED_BUDGET_FIELDS:
                if key not in inherited_constraints:
                    inherited_constraints[key] = (paired_state.get("constraints") or {}).get(key)
            constraints = inherited_constraints
            if budget is None:
                budget = int(paired_state.get("vlm_budget_per_round", 500))
            if evaluation_visibility is None:
                evaluation_visibility = paired_state.get("evaluation_visibility", "heldout_only")

        if evaluation_visibility is None:
            evaluation_visibility = "heldout_only"

        if execution_mode not in ("adaptive_agent", "fixed_baseline"):
            raise ValueError("execution_mode must be adaptive_agent or fixed_baseline")
        if evaluation_visibility not in ("online_feedback", "heldout_only"):
            raise ValueError("evaluation_visibility must be online_feedback or heldout_only")
        if not isinstance(seeds, int) or isinstance(seeds, bool) or seeds < 1:
            raise ValueError("seeds must be >= 1")
        if budget is not None and (
            not isinstance(budget, int) or isinstance(budget, bool) or budget < 0
        ):
            raise ValueError("budget must be a non-negative integer")
        if execution_mode == "fixed_baseline" and not isinstance(fixed_policy, dict):
            raise ValueError("fixed_baseline requires a fixed_policy object")
        if execution_mode == "fixed_baseline":
            policy_errors = validate_fixed_policy(fixed_policy)
            if policy_errors:
                raise ValueError("Invalid fixed_policy: " + "; ".join(policy_errors))
            fixed_transition = fixed_policy["transition"]["transition_policy"]
            if fixed_transition != transition_policy:
                raise ValueError(
                    "transition_policy must match fixed_policy.transition.transition_policy"
                )
            if (
                fixed_policy["resolve"]["resolution_policy"] in ("vlm", "inspect_only")
                and budget is not None
                and budget < int(fixed_policy["resolve"]["budget"])
            ):
                raise ValueError("Per-round VLM budget is below the fixed resolve action budget")
        if transition_policy not in ("clean_only", "deploy_collect_merge"):
            raise ValueError("Unknown transition_policy")
        if experimental_controls is not None and not isinstance(experimental_controls, dict):
            raise ValueError("experimental_controls must be an object")
        if isinstance(experimental_controls, dict):
            unknown_controls = sorted(set(experimental_controls) - set(DECISION_FIELDS))
            if unknown_controls:
                raise ValueError(f"Unknown experimental control decisions: {unknown_controls}")
            if not all(isinstance(v, dict) for v in experimental_controls.values()):
                raise ValueError("Each experimental control must be an object")
        if dataset_subset is not None and not isinstance(dataset_subset, dict):
            raise ValueError("dataset_subset must be an object")
        if isinstance(vlm, dict):
            vlm = {k: vlm[k] for k in ("model", "base_url") if vlm.get(k)}
        constraint_defaults = _default_constraints(task_id)
        if isinstance(constraints, dict):
            unknown_constraints = sorted(set(constraints) - set(constraint_defaults))
            if unknown_constraints:
                raise ValueError(f"Unknown constraints: {unknown_constraints}")
            for key in (
                "max_skip_streak", "max_deployments", "max_rounds", "max_vlm_calls_total",
                "max_detector_train_epochs_total", "max_controller_train_epochs_total",
                "max_collection_images_total",
            ):
                value = constraints.get(key)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                ):
                    raise ValueError(f"constraints.{key} must be a non-negative integer or null")
            for key in ("enforce_first_deploy", "lock_round0_detector", "allow_voluntary_terminate"):
                value = constraints.get(key)
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"constraints.{key} must be boolean")
            if (
                transition_policy == "deploy_collect_merge"
                and constraints.get("max_deployments") == 0
            ):
                raise ValueError(
                    "deploy_collect_merge conflicts with max_deployments=0"
                )

        # Validate the task-scoped view before writing either spec or state.  A
        # malformed subset is an experimental-design error, not the same thing
        # as a workspace whose dataset simply has not been configured yet.
        try:
            registry = _dataset_config(workspace_dir)
        except ValueError:
            if _dataset_registry_path(workspace_dir).exists():
                raise
            registry = None
        if paired_state is not None and registry is None:
            raise ValueError("Cannot create a paired task without the configured BaseDataset")
        normalized_subset = dataset_subset or {}
        if registry is not None and dataset_subset:
            from .configure_dataset import ConfigureDataset
            normalized_subset = ConfigureDataset._normalize_subset(
                registry,
                dataset_subset.get("include_sources"),
                dataset_subset.get("exclude_sources"),
                dataset_subset.get("max_per_source"),
            )

        eff = constraint_defaults
        if isinstance(constraints, dict):
            for k, v in constraints.items():
                if v is not None:
                    eff[k] = v
        if execution_mode == "fixed_baseline":
            fixed_rounds = int(fixed_policy["stopping"]["max_rounds"])
            if isinstance(constraints, dict) and constraints.get("max_rounds") not in (None, fixed_rounds):
                raise ValueError("constraints.max_rounds conflicts with fixed_policy.stopping.max_rounds")
            eff["max_rounds"] = fixed_rounds
            detector_strategy = fixed_policy["detector"]["strategy"]
            detector_rounds = 1 if detector_strategy == "retrain_first_then_reuse" else fixed_rounds
            required_detector_epochs = int(fixed_policy["detector"]["epochs"]) * detector_rounds
            required_controller_epochs = int(fixed_policy["controller"]["epochs"]) * fixed_rounds
            if eff["max_detector_train_epochs_total"] is None:
                eff["max_detector_train_epochs_total"] = required_detector_epochs
            elif int(eff["max_detector_train_epochs_total"]) < required_detector_epochs:
                raise ValueError("Detector epoch cap is below the fixed policy's preregistered schedule")
            if eff["max_controller_train_epochs_total"] is None:
                eff["max_controller_train_epochs_total"] = required_controller_epochs
            elif int(eff["max_controller_train_epochs_total"]) < required_controller_epochs:
                raise ValueError("Controller epoch cap is below the fixed policy's preregistered schedule")

        if paired_state is not None:
            mismatched = [
                key for key in _PAIRED_BUDGET_FIELDS
                if eff.get(key) != (paired_state.get("constraints") or {}).get(key)
            ]
            if mismatched:
                raise ValueError(
                    "Paired tasks must use identical hard budgets; mismatched: "
                    + ", ".join(mismatched)
                )
            if evaluation_visibility != paired_state.get("evaluation_visibility", "heldout_only"):
                raise ValueError("Paired tasks must use identical evaluation_visibility")
            requested_vlm_budget = int(budget) if budget is not None else 500
            if requested_vlm_budget != int(paired_state.get("vlm_budget_per_round", 500)):
                raise ValueError("Paired tasks must use identical per-round VLM budget")

        eff_pipeline = default_pipeline_for()
        if isinstance(pipeline, dict):
            eff_pipeline.update({k: v for k, v in pipeline.items() if v is not None})
        if execution_mode == "fixed_baseline":
            implied = {
                "train_detector": fixed_policy["detector"]["strategy"],
                "resolve": fixed_policy["resolve"]["resolution_policy"],
                "transition": fixed_policy["transition"]["transition_policy"],
            }
            if isinstance(pipeline, dict):
                conflicts = [
                    stage for stage, value in implied.items()
                    if pipeline.get(stage) is not None and pipeline.get(stage) != value
                ]
                if conflicts:
                    raise ValueError(f"pipeline conflicts with fixed_policy at stages: {conflicts}")
            eff_pipeline.update(implied)
        eff_pipeline["transition"] = transition_policy
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
            "execution_mode": execution_mode,
            "fixed_policy": fixed_policy or {},
            "experimental_controls": experimental_controls or {},
            "transition_policy": transition_policy,
            "dataset_subset": normalized_subset,
            "vlm": vlm,
            "evaluation_visibility": evaluation_visibility,
            "paired_with": paired_with or None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        td.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(spec_path, spec)

        new_state = dict(_RESET_FIELDS)
        # Avoid sharing nested mutable defaults between task instances.
        new_state["history"] = []
        new_state["round_history"] = []
        new_state["latest_observation"] = {}
        new_state["pending_collection_ids"] = []
        new_state["vlm_budget_per_round"] = int(budget) if budget is not None else 500
        new_state["constraints"] = eff
        new_state["pipeline"] = eff_pipeline
        new_state["execution_mode"] = execution_mode
        new_state["fixed_policy"] = fixed_policy or {}
        new_state["experimental_controls"] = experimental_controls or {}
        new_state["default_transition_policy"] = transition_policy
        new_state["dataset_subset"] = normalized_subset
        new_state["vlm"] = vlm
        new_state["evaluation_visibility"] = evaluation_visibility
        new_state["paired_with"] = paired_with or None
        _save(workspace_dir, new_state, branch=task_id)

        # Freeze D_0 immediately when a workspace dataset is already configured.
        if registry is not None:
            try:
                records = _raw_records(workspace_dir, branch=task_id)
                input_path, payload = _write_dataset_snapshot(
                    workspace_dir,
                    task_id,
                    "input_r0.json",
                    records,
                    round_index=0,
                    role="round_input",
                    parents=["workspace_dataset"],
                    metadata={"task_id": task_id},
                )
                new_state["round_input_dataset"] = str(input_path)
                new_state["round_input_count"] = len(records)
                new_state["round_input_fingerprint"] = payload["fingerprint"]
                if (
                    paired_state is not None
                    and payload["fingerprint"] != paired_state.get("round_input_fingerprint")
                ):
                    raise ValueError(
                        "Paired tasks must freeze the exact same D_0 fingerprint"
                    )
                if paired_state is not None:
                    new_state["paired_d0_fingerprint"] = payload["fingerprint"]
                _save(workspace_dir, new_state, branch=task_id)
            except Exception:
                # Avoid leaving a half-defined experimental task if materializing
                # D_0 fails after the preflight validation.
                import shutil
                shutil.rmtree(td)
                raise

        return json.dumps({
            "defined": task_id,
            "task_dir": str(td),
            "state_initialized": True,
            "spec": spec
        }, ensure_ascii=False)
