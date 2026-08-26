import json
from .base import Tool
from .utils import (
    _anonymize_source_name,
    _dataset_config,
    _load,
    _load_task_spec,
    _records,
)
from .agent_protocol import public_protocol_state


_ALWAYS_HIDDEN_METRICS = {
    "auc", "precision", "recall", "f1", "purity", "retention_purity",
    "normal_retention", "anomaly_leak", "confusion_matrix", "ground_truth_provenance",
}
_CTE_METRICS = {
    "cte_mean", "real_cte_mean", "real_cte_std", "real_cte_rmse",
    "real_cte_signed_mean", "best_cte", "last_deployed_cte",
}


def _agent_visible_projection(value, *, hide_cte):
    if isinstance(value, list):
        return [_agent_visible_projection(item, hide_cte=hide_cte) for item in value]
    if not isinstance(value, dict):
        return value
    hidden = _ALWAYS_HIDDEN_METRICS | (_CTE_METRICS if hide_cte else set())
    return {
        key: _agent_visible_projection(item, hide_cte=hide_cte)
        for key, item in value.items()
        if key not in hidden and key not in {"evaluate", "allow_physical_deploy"}
    }

class PipelineState(Tool):
    name = "get_pipeline_state"
    description = "Read compact structured pipeline observation state for active task branch."
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def run(self, branch="main", workspace_dir=None, **_):
        ds = None
        dataset_error = None
        try:
            ds = _dataset_config(workspace_dir)
        except Exception as exc:
            dataset_error = str(exc)
            ds = None

        try:
            s = _load(workspace_dir, branch=branch)
        except Exception:
            return json.dumps({
                "branch": branch,
                "task_created": False,
                "dataset_configured": bool(ds and ds.get("sources")),
                "dataset_error": dataset_error,
                "raw_samples": (ds or {}).get("raw_samples"),
            }, ensure_ascii=False)

        spec = _load_task_spec(workspace_dir, branch=branch)
        hide_cte = s.get("evaluation_visibility", "online_feedback") == "heldout_only"
        active_samples = s.get("round_input_count")
        active_comp = None
        round_input_error = None
        if ds:
            try:
                recs = _records(workspace_dir, branch=branch)
                if active_samples is None:
                    active_samples = len(recs)
                src_map = {}
                for r in recs:
                    src = _anonymize_source_name(r["source"])
                    src_map[src] = src_map.get(src, 0) + 1
                active_comp = src_map
            except Exception as exc:
                round_input_error = str(exc)

        raw_sources = [_anonymize_source_name(x.get("name")) for x in (ds or {}).get("sources", [])] if ds else []
        raw_comp = (ds or {}).get("source_composition")
        if raw_comp and isinstance(raw_comp, dict):
            raw_comp = {_anonymize_source_name(k): v for k, v in raw_comp.items()}
        internal_task_status = s.get("task_status", "DRAFT")
        dataset_editable = internal_task_status == "DRAFT"
        execution_status = (
            "completed"
            if internal_task_status == "COMPLETED"
            else ("not_started" if dataset_editable else "started")
        )

        out = {
            "state_schema_version": s.get("schema_version", 1),
            "branch": branch,
            "task_created": True,
            "task": {
                "description": spec.get("description", branch),
                "hypothesis": spec.get("hypothesis", ""),
                "independent_variable": spec.get("independent_variable", ""),
                "execution_mode": s.get("execution_mode", spec.get("execution_mode", "adaptive_agent")),
                "pipeline": s.get("pipeline") or spec.get("pipeline") or {},
                "experimental_controls": s.get("experimental_controls") or {},
                "fixed_policy": s.get("fixed_policy") if s.get("execution_mode") == "fixed_baseline" else None,
                "default_transition_policy": s.get("default_transition_policy", spec.get("transition_policy", "clean_only")),
            },
            "round": s.get("round", 0),
            # DRAFT/LOCKED/RUNNING are internal transaction states. Expose the
            # user-relevant capability instead so the model cannot invent a
            # manual task-locking ritual.
            "dataset_configuration": {
                "status": "editable" if dataset_editable else "frozen",
                "editable": dataset_editable,
                "freezes_automatically_on_first_experimental_action": True,
            },
            "execution_status": execution_status,
            "task_completed": internal_task_status == "COMPLETED",
            "round_status": s.get("round_status", "ready"),
            "deployments": s.get("deployments", 0),
            "active_detector": s.get("active_detector"),
            "active_controller": {
                "id": (s.get("active_controller") or {}).get("id"),
                "trained_on_round": (s.get("active_controller") or {}).get("round"),
                "metrics": (s.get("active_controller") or {}).get("metrics"),
            } if s.get("active_controller") else None,
            "round_input": {
                "artifact": (s.get("round_input_dataset") or "").split("/")[-1] or None,
                "count": active_samples if active_samples is not None else (ds or {}).get("raw_samples"),
                "fingerprint": s.get("round_input_fingerprint"),
                "error": round_input_error,
            },
            "clean_output": {
                "artifact": (s.get("active_clean_dataset") or "").split("/")[-1] or None,
                "count": s.get("clean_count"),
            },
            "dataset_error": dataset_error,
            "active_samples": active_samples if active_samples is not None else (ds or {}).get("raw_samples"),
            "anonymous_sources": (
                sorted(active_comp or {}) if s.get("round_input_dataset") else raw_sources
            ),
            "anonymous_source_composition": (
                active_comp if s.get("round_input_dataset") else raw_comp
            ),
            "constraints": _agent_visible_projection(
                s.get("constraints") or {}, hide_cte=hide_cte
            ),
            "latest_observation": _agent_visible_projection(
                s.get("latest_observation") or {}, hide_cte=hide_cte
            ),
            "recent_decisions": _agent_visible_projection(
                (s.get("decision_trace") or [])[-8:], hide_cte=hide_cte
            ),
            "recent_rounds": _agent_visible_projection(
                (s.get("round_history") or [])[-3:], hide_cte=hide_cte
            ),
            "pending_collection_ids": s.get("pending_collection_ids") or [],
            "recent_deployment_runs": [
                {
                    "deployment_run_id": run.get("deployment_run_id"),
                    "collection_id": run.get("collection_id"),
                    "round": run.get("round"),
                    "status": run.get("status"),
                    "controller_id": run.get("controller_id"),
                    "n_images_target": run.get("n_images_target"),
                }
                for run in (s.get("deployment_runs") or [])[-5:]
            ],
            "evaluation_visibility": s.get("evaluation_visibility", "online_feedback"),
            "vlm_budget": {
                "per_round": s.get("vlm_budget_per_round"),
                "used_this_round": s.get("vlm_budget_used_this_round", 0),
                "used_total": s.get("vlm_calls_total", 0),
            },
            "training_budget": {
                "detector_epochs_used": s.get("detector_train_epochs_used", 0),
                "controller_epochs_used": s.get("controller_train_epochs_used", 0),
                "detector_epochs_cap": (s.get("constraints") or {}).get("max_detector_train_epochs_total"),
                "controller_epochs_cap": (s.get("constraints") or {}).get("max_controller_train_epochs_total"),
            },
            "deployment_budget": {
                "evaluations_used": s.get("deployments", 0),
                "evaluations_cap": (s.get("constraints") or {}).get("max_deployments"),
                "collection_images_used": s.get("collection_images_budget_used", 0),
                "collection_images_cap": (
                    s.get("constraints") or {}
                ).get("max_collection_images_total"),
            },
            "agent_protocol": public_protocol_state(workspace_dir, branch),
        }
        return json.dumps(out, ensure_ascii=False)
