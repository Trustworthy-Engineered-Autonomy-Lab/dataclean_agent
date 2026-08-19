"""Deterministic, LLM-free driver for preregistered fixed-baseline tasks.

The driver intentionally calls the same tools used by the adaptive arm.  Each
tool resolves its action from ``fixed_policy`` so proposed CLI defaults cannot
silently change the baseline.
"""

import argparse
import json
from pathlib import Path

from tools import Tool  # Imports and registers the production tool set.
from tools.utils import _load, _save


def _run(tool_name, workspace, task_id, **kwargs):
    raw = Tool.get(tool_name).run(
        workspace_dir=workspace,
        branch=task_id,
        **kwargs,
    )
    return json.loads(raw)


def run_fixed_round(workspace, task_id, train_controller=True, commit=True):
    workspace = str(Path(workspace).resolve())
    state = _load(workspace, branch=task_id)
    if state.get("execution_mode") != "fixed_baseline":
        raise ValueError("baseline_runner only accepts fixed_baseline tasks")
    if not state.get("round_input_dataset") or not state.get("round_input_fingerprint"):
        raise ValueError("Fixed-baseline task has no frozen D_t; configure D_0 first")
    if state.get("task_status", "DRAFT") == "DRAFT":
        state["task_status"] = "RUNNING"
        _save(workspace, state, branch=task_id)

    outputs = {}
    stopping = _run(
        "assess_stopping", workspace, task_id, stop=False,
        rationale="Preregistered fixed-baseline stopping check",
    )
    outputs["stopping"] = stopping
    if stopping["stop"]:
        return outputs

    state = _load(workspace, branch=task_id)
    status = state.get("round_status", "ready")
    fixed = state["fixed_policy"]

    if status == "ready":
        action = fixed["detector"]
        outputs["detector"] = _run(
            "train_detector", workspace, task_id,
            lambda_value=action.get("lambda_value", 1.0),
            steer_lambda=action.get("steer_lambda", 1.0),
            strategy=action.get("strategy", "retrain"),
            rationale="Preregistered fixed detector policy",
        )
        status = "detector_ready"
    if status == "detector_ready":
        outputs["score"] = _run(
            "score_and_fit", workspace, task_id, alpha=0.5,
            rationale="Preregistered fixed score policy",
        )
        status = "scored"
    if status == "scored":
        outputs["partition_analysis"] = _run("partition", workspace, task_id)
        outputs["partition"] = _run(
            "partition", workspace, task_id, threshold=0.0,
            rationale="Preregistered fixed partition policy",
        )
        status = "partitioned"
    if status == "partitioned":
        outputs["resolve"] = _run(
            "resolve", workspace, task_id, resolution_policy="auto_keep",
            rationale="Preregistered fixed resolution policy",
        )
        status = _load(workspace, branch=task_id).get("round_status")

    if status != "resolved":
        outputs["awaiting"] = status
        return outputs

    latest_state = _load(workspace, branch=task_id)
    controller = latest_state.get("active_controller") or {}
    if train_controller and controller.get("round") != int(latest_state.get("round", 0)):
        outputs["controller"] = _run(
            "train_controller", workspace, task_id,
            rationale="Preregistered fixed controller policy",
        )
    if commit:
        if (
            fixed["transition"]["transition_policy"] == "deploy_collect_merge"
            and not (_load(workspace, branch=task_id).get("pending_collection_ids") or [])
        ):
            outputs["awaiting"] = "exact_deployment_collection"
            outputs["next_required_tools"] = [
                "deploy_controller", "eval_controller", "transfer_eval_results"
            ]
            return outputs
        outputs["commit"] = _run(
            "commit_round", workspace, task_id,
            transition_policy=fixed["transition"]["transition_policy"],
            rationale="Preregistered fixed transition policy",
        )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace")
    parser.add_argument("task_id")
    parser.add_argument("--skip-controller", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args()
    result = run_fixed_round(
        args.workspace,
        args.task_id,
        train_controller=not args.skip_controller,
        commit=not args.no_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
