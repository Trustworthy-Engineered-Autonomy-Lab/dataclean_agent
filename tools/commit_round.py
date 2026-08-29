import json
from pathlib import Path

from .base import Tool
from .decision_policy import effective_action, record_decision
from .utils import (
    _load,
    _load_dataset_snapshot,
    _save,
    _write_dataset_snapshot,
    append_ledger,
    record_observation,
    _task_artifact_reference,
)
from .collections import load_collection


TRANSITION_POLICIES = ("clean_only", "deploy_collect_merge")


class CommitRound(Tool):
    name = "commit_round"
    description = (
        "Commit the next immutable round-input dataset and only then advance round. "
        "clean_only sets D_(t+1)=C_t; deploy_collect_merge sets D_(t+1)=C_t union newly collected N_t."
    )
    parameters = {
        "type": "object",
        "properties": {
            "transition_policy": {
                "type": "string",
                "enum": list(TRANSITION_POLICIES),
                "description": "How the next round input is formed.",
            },
            "collection_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For deploy_collect_merge, exact task-local CollectionArtifact IDs. Defaults to pending collection IDs.",
            },
            "rationale": {
                "type": "string",
                "description": "Concise observation-based reason for choosing this transition.",
            },
        },
        "required": ["transition_policy", "rationale"],
    }

    def run(
        self,
        transition_policy,
        rationale,
        collection_ids=None,
        branch="main",
        workspace_dir=None,
        **_,
    ):
        state = _load(workspace_dir, branch=branch)
        if state.get("termination_required"):
            raise ValueError(
                "Task is completed by a terminal stop decision; create a new task"
            )
        if state.get("round_status") != "resolved":
            raise ValueError("commit_round requires a resolved current round")
        proposed = {"transition_policy": transition_policy}
        effective, source = effective_action(state, "transition", proposed)
        transition_policy = effective.get("transition_policy")
        if transition_policy not in TRANSITION_POLICIES:
            raise ValueError(f"Unknown transition_policy: {transition_policy}")
        if source.startswith("agent") and not str(rationale).strip():
            raise ValueError("Adaptive transition decisions require an observation-based rationale")
        if source == "fixed_policy" and not str(rationale).strip():
            rationale = "Preregistered fixed baseline transition policy"
        max_rounds = (state.get("constraints") or {}).get("max_rounds")
        if max_rounds is not None and int(state.get("round", 0)) >= int(max_rounds):
            raise ValueError(f"Maximum round limit reached ({max_rounds})")
        clean_ref = state.get("active_clean_dataset")
        if not clean_ref:
            raise ValueError("Current round has no committed clean dataset; run resolve first")
        clean_payload = _load_dataset_snapshot(
            _task_artifact_reference(workspace_dir, branch, clean_ref)
        )
        current_round = int(state.get("round", 0))
        if (
            clean_payload.get("task_id") != branch
            or clean_payload.get("role") != "clean"
            or int(clean_payload.get("round", -1)) != current_round
        ):
            raise ValueError("Active clean snapshot identity, role, or round is invalid")
        clean_records = clean_payload["records"]
        if not clean_records:
            raise ValueError("Refusing to commit an empty clean dataset")

        collection_records = []
        selected_collection_ids = []
        collection_parents = []
        if transition_policy == "deploy_collect_merge":
            requested = collection_ids or state.get("pending_collection_ids") or []
            selected_collection_ids = list(dict.fromkeys(str(x) for x in requested))
            if not selected_collection_ids:
                raise ValueError(
                    "deploy_collect_merge requires transferred collected data; "
                    "run transfer_eval_results or provide collection_ids"
                )
            for collection_id in selected_collection_ids:
                artifact = load_collection(workspace_dir, branch, collection_id)
                if int(artifact.get("round", -1)) != int(state.get("round", 0)):
                    raise ValueError(
                        f"Collection {collection_id} belongs to round {artifact.get('round')}, "
                        f"not current round {state.get('round', 0)}"
                    )
                collection_records.extend(artifact["records"])
                collection_parents.append({
                    "collection_id": collection_id,
                    "anonymous_source": artifact["anonymous_source"],
                    "deployment_run_id": artifact.get("deployment_run_id"),
                    "fingerprint": artifact.get("fingerprint"),
                    "count": artifact.get("count"),
                })

        merged = []
        seen = {}
        duplicates = 0
        for record in clean_records + collection_records:
            prior = seen.get(record["id"])
            if prior is None:
                merged.append(record)
                seen[record["id"]] = record
            elif prior == record:
                duplicates += 1
            else:
                raise ValueError(f"Conflicting records share sample ID: {record['id']}")

        successful_runs = [
            run for run in (state.get("deployment_runs") or [])
            if int(run.get("round", -1)) == current_round
            and run.get("status") in {"completed", "transferred"}
        ]
        constraints = state.get("constraints") or {}
        if current_round == 0 and constraints.get("enforce_first_deploy") and not successful_runs:
            raise ValueError("Round 0 requires a successful physical evaluation before commit")
        next_skip_streak = 0 if successful_runs else int(state.get("skip_streak", 0)) + 1
        max_skip_streak = constraints.get("max_skip_streak")
        if max_skip_streak is not None and next_skip_streak > int(max_skip_streak):
            raise ValueError(
                f"Maximum consecutive rounds without physical evaluation exceeded ({max_skip_streak})"
            )
        next_round = current_round + 1
        parents = [Path(clean_ref).name]
        parents.extend(selected_collection_ids)
        input_path, snapshot = _write_dataset_snapshot(
            workspace_dir,
            branch,
            f"input_r{next_round}.json",
            merged,
            round_index=next_round,
            role="round_input",
            parents=parents,
            metadata={
                "transition_policy": transition_policy,
                "clean_count": len(clean_records),
                "collected_count": len(collection_records),
                "collection_artifacts": collection_parents,
                "deduplicated_count": duplicates,
                "rationale": rationale,
            },
        )

        completed = {
            "round": current_round,
            "input_dataset": state.get("round_input_dataset"),
            "input_fingerprint": state.get("round_input_fingerprint"),
            "clean_dataset": clean_ref,
            "scores_artifact": state.get("latest_scores"),
            "clean_count": len(clean_records),
            "transition_policy": transition_policy,
            "collection_ids": selected_collection_ids,
            "collection_artifacts": collection_parents,
            "collected_count": len(collection_records),
            "deduplicated_count": duplicates,
            "next_input_dataset": str(input_path),
            "next_input_count": len(merged),
            "rationale": rationale,
            "observations": {
                key: value
                for key, value in (state.get("latest_observation") or {}).items()
                if key in {
                    "train_detector", "score_and_fit", "partition", "resolve",
                    "train_controller", "eval_controller", "transfer_eval_results", "evaluate",
                }
            },
        }
        decision_entry = record_decision(
            state,
            "transition",
            proposed,
            effective,
            rationale,
            source,
            observation={
                "clean_count": len(clean_records),
                "collected_count": len(collection_records),
                "deduplicated_count": duplicates,
            },
        )
        state.setdefault("round_history", []).append(completed)
        append_ledger(state, {"stage": "commit_round", **completed})
        record_observation(
            state,
            "commit_round",
            completed,
            workspace_dir=workspace_dir,
            branch=branch,
            decision=decision_entry,
        )

        state["round"] = next_round
        state["skip_streak"] = next_skip_streak
        state["round_input_dataset"] = str(input_path)
        state["round_input_fingerprint"] = snapshot["fingerprint"]
        state["round_input_count"] = len(merged)
        state["previous_clean_dataset"] = clean_ref
        state["active_clean_dataset"] = None
        state["clean_count"] = None
        state["latest_scores"] = None
        state["score_round"] = None
        state["score_detector_id"] = None
        state.pop("score_alpha", None)
        state["score_contract"] = None
        state["latest_partition"] = None
        state["latest_observation"] = {}
        state["vlm_budget_current_round"] = next_round
        state["vlm_budget_used_this_round"] = 0
        pending = list(state.get("pending_collection_ids") or [])
        unconsumed = [x for x in pending if x not in set(selected_collection_ids)]
        if transition_policy == "clean_only":
            unconsumed = pending
        if unconsumed:
            state.setdefault("unconsumed_collections", []).append({
                "round": current_round,
                "collection_ids": unconsumed,
                "disposition": "not_merged",
            })
        state.setdefault("consumed_collection_ids", []).extend(selected_collection_ids)
        state["pending_collection_ids"] = []
        state.pop("pending_collected_sources", None)
        state["round_status"] = "ready"
        _save(workspace_dir, state, branch=branch)

        return json.dumps(
            {
                "committed": True,
                "completed_round": current_round,
                "next_round": next_round,
                "transition_policy": transition_policy,
                "clean_count": len(clean_records),
                "collected_count": len(collection_records),
                "collection_ids": selected_collection_ids,
                "unconsumed_collection_ids": unconsumed,
                "deduplicated_count": duplicates,
                "next_input_count": len(merged),
                "next_input_dataset": input_path.name,
                "next_input_fingerprint": snapshot["fingerprint"],
            },
            ensure_ascii=False,
        )
