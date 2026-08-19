import json
from pathlib import Path

from .base import Tool
from .decision_policy import effective_action, record_decision
from .policies import POLICIES
from .utils import (
    _artifact,
    _load,
    _load_dataset_snapshot,
    _save,
    _write_dataset_snapshot,
    _write_json_atomic,
    record_observation,
    _task_artifact_reference,
)
from .vlm_review import run_vlm_review


_RESOLUTION_POLICIES = tuple(POLICIES["resolve"])


class Resolve(Tool):
    name = "resolve"
    description = (
        "Resolve the current partition into immutable C_t. auto_keep keeps only the detector keep region; "
        "vlm adds gray samples explicitly accepted by the reviewer; inspect_only audits gray without producing C_t. "
        "Unreviewed or failed VLM samples are quarantined, never mislabeled as VLM rejects."
    )
    parameters = {
        "type": "object",
        "properties": {
            "resolution_policy": {"type": "string", "enum": list(_RESOLUTION_POLICIES)},
            "budget": {"type": "integer", "minimum": 1},
            "sampling_strategy": {
                "type": "string",
                "enum": ["pollution_defense", "rare_behavior_recovery", "information_gain", "verification"],
            },
            "accept_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "rationale": {
                "type": "string",
                "description": "Observation-based reason for the adaptive resolution choice.",
            },
            "vlm_model": {"type": "string"},
            "vlm_base_url": {"type": "string"},
        },
        "required": ["resolution_policy", "rationale"],
    }

    def run(
        self,
        resolution_policy="vlm",
        budget=200,
        sampling_strategy="pollution_defense",
        accept_confidence="high",
        rationale="",
        branch="main",
        workspace_dir=None,
        vlm_model=None,
        vlm_base_url=None,
        cancel_event=None,
        **_,
    ):
        state = _load(workspace_dir, branch=branch)
        partition = state.get("latest_partition")
        if not partition:
            raise ValueError("Run partition before resolve")
        if state.get("round_status") != "partitioned":
            raise ValueError("Resolve requires an applied current-round partition")
        if state.get("active_clean_dataset"):
            raise ValueError("This round already has a committed clean dataset")

        proposed = {
            "resolution_policy": resolution_policy,
            "budget": int(budget),
            "sampling_strategy": sampling_strategy,
            "accept_confidence": accept_confidence,
        }
        effective, source = effective_action(state, "resolve", proposed)
        resolution_policy = effective.get("resolution_policy", resolution_policy)
        budget = int(effective.get("budget", budget))
        sampling_strategy = effective.get("sampling_strategy", sampling_strategy)
        accept_confidence = effective.get("accept_confidence", accept_confidence)
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if sampling_strategy not in (
            "pollution_defense", "rare_behavior_recovery", "information_gain", "verification"
        ):
            raise ValueError("Unknown sampling_strategy")
        if accept_confidence not in ("low", "medium", "high"):
            raise ValueError("Unknown accept_confidence")
        if resolution_policy not in _RESOLUTION_POLICIES:
            raise ValueError(f"Unknown resolution_policy: {resolution_policy}")
        if source.startswith("agent") and not rationale.strip():
            raise ValueError("Adaptive resolve decisions require an observation-based rationale")
        if source == "fixed_policy" and not rationale.strip():
            rationale = "Preregistered fixed baseline resolution policy"

        scores_ref = partition.get("scores_artifact") or state.get("latest_scores")
        scored = json.loads(
            _task_artifact_reference(workspace_dir, branch, scores_ref).read_text()
        )
        score_map = {r["id"]: r for r in scored}
        keep = [score_map[i] for i in partition.get("keep_ids", []) if i in score_map]
        gray = [score_map[i] for i in partition.get("gray_ids", []) if i in score_map]
        detector_discard_ids = {
            i for i in partition.get("discard_ids", []) if i in score_map
        }

        review = {
            "accepted": [], "rejected": [], "unresolved": [], "call_failed": [],
            "reviewed": [], "selected": 0, "api_calls": 0,
        }
        if resolution_policy in ("vlm", "inspect_only") and gray:
            review = run_vlm_review(
                workspace_dir,
                state,
                branch,
                budget,
                sampling_strategy,
                accept_confidence,
                vlm_model,
                vlm_base_url,
                None,
                cancel_event,
            )

        reviewed_ids = {r["sample_id"] for r in review["reviewed"]}
        accepted_ids = {r["id"] for r in review["accepted"]}
        rejected_ids = {r["id"] for r in review["rejected"]}
        unresolved_ids = {r["id"] for r in review["unresolved"]}
        call_failed_ids = {r["id"] for r in review["call_failed"]}
        unreviewed_ids = (
            set() if resolution_policy == "auto_keep"
            else {r["id"] for r in gray} - reviewed_ids
        )
        quarantined_ids = sorted(unresolved_ids | call_failed_ids | unreviewed_ids)

        decision_observation = {
            "keep_count": len(keep),
            "gray_count": len(gray),
            "detector_discard_count": len(detector_discard_ids),
            "vlm_selected": review["selected"],
            "vlm_accepted": len(accepted_ids),
            "vlm_rejected": len(rejected_ids),
            "vlm_unresolved": len(unresolved_ids),
            "vlm_call_failed": len(call_failed_ids),
            "quarantined": len(quarantined_ids),
        }
        record_decision(
            state,
            "resolve",
            proposed,
            effective,
            rationale,
            source,
            observation=decision_observation,
        )

        round_index = int(state.get("round", 0))
        if resolution_policy == "inspect_only":
            summary = {
                "policy": resolution_policy,
                **decision_observation,
                "clean_count": None,
                "clean_dataset_id": None,
            }
            record_observation(state, "resolve", summary, workspace_dir=workspace_dir, branch=branch)
            _save(workspace_dir, state, branch=branch)
            return json.dumps(summary, ensure_ascii=False)

        accepted = review["accepted"] if resolution_policy == "vlm" else []
        clean = keep + accepted
        if not clean:
            raise ValueError("Resolution produced an empty clean dataset")
        clean_path, clean_payload = _write_dataset_snapshot(
            workspace_dir,
            branch,
            f"clean_r{round_index}.json",
            clean,
            round_index=round_index,
            role="clean",
            parents=[Path(state.get("round_input_dataset") or "workspace_dataset").name],
            metadata={
                "resolution_policy": resolution_policy,
                "partition_threshold": partition.get("threshold"),
                "gray_upper_threshold": partition.get("gray_upper_threshold"),
                "decision_source": source,
                "rationale": rationale,
            },
        )
        quarantine_path = _artifact(workspace_dir, f"quarantine_r{round_index}.json", branch=branch)
        _write_json_atomic(
            quarantine_path,
            {
                "round": round_index,
                "unreviewed_ids": sorted(unreviewed_ids),
                "unresolved_ids": sorted(unresolved_ids),
                "call_failed_ids": sorted(call_failed_ids),
                "vlm_rejected_ids": sorted(rejected_ids),
                "detector_discard_ids": sorted(detector_discard_ids),
                "policy_discarded_gray_ids": sorted({r["id"] for r in gray}) if resolution_policy == "auto_keep" else [],
            },
        )

        state["active_clean_dataset"] = str(clean_path)
        state["clean_count"] = len(clean)
        state["round_status"] = "resolved"
        summary = {
            "policy": resolution_policy,
            "decision_source": source,
            "keep_count": len(keep),
            "gray_count": len(gray),
            "detector_discard_count": len(detector_discard_ids),
            "vlm_selected": review["selected"],
            "vlm_api_calls": review["api_calls"],
            "vlm_accepted": len(accepted_ids),
            "vlm_rejected": len(rejected_ids),
            "vlm_unresolved": len(unresolved_ids),
            "vlm_call_failed": len(call_failed_ids),
            "quarantined_count": len(quarantined_ids),
            "policy_discarded_gray_count": len(gray) if resolution_policy == "auto_keep" else 0,
            "clean_count": len(clean),
            "clean_dataset_id": clean_path.name,
            "clean_fingerprint": clean_payload["fingerprint"],
            "quarantine_artifact": quarantine_path.name,
        }
        record_observation(state, "resolve", summary, workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, state, branch=branch)
        return json.dumps(summary, ensure_ascii=False)
