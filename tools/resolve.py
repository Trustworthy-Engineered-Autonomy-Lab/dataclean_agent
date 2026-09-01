import json
from pathlib import Path

from .base import Tool
from .decision_policy import effective_action, record_decision
from .detector_contract import normality_scores, require_partition_contract, score_contract
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
DEFAULT_VLM_SAMPLING_STRATEGY = "information_gain"


class Resolve(Tool):
    name = "resolve"
    description = (
        "Resolve the current partition into immutable C_t. auto_keep keeps only the detector keep region; "
        "vlm adds gray samples explicitly accepted by the reviewer; inspect_only audits gray without producing C_t. "
        "Unreviewed or failed VLM samples are quarantined, never mislabeled as VLM rejects. "
        "For vlm/inspect_only, provide a call budget no greater than the fixed 200-call per-round cap "
        "and accept_confidence; sampling_strategy defaults to information_gain and may be overridden "
        "explicitly. auto_keep needs none of these."
    )
    parameters = {
        "type": "object",
        "properties": {
            "resolution_policy": {"type": "string", "enum": list(_RESOLUTION_POLICIES)},
            "budget": {
                "type": "integer", "minimum": 1, "maximum": 200,
                "description": "VLM calls for this review (1-200); the task-level per-round cap is fixed at 200.",
            },
            "sampling_strategy": {
                "type": "string",
                "enum": ["pollution_defense", "rare_behavior_recovery", "information_gain", "verification"],
                "description": "Gray-zone review ordering; omitted values default to information_gain.",
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
        resolution_policy=None,
        budget=None,
        sampling_strategy=DEFAULT_VLM_SAMPLING_STRATEGY,
        accept_confidence=None,
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
        require_partition_contract(partition)
        if state.get("round_status") != "partitioned":
            raise ValueError("Resolve requires an applied current-round partition")
        if state.get("active_clean_dataset"):
            raise ValueError("This round already has a committed clean dataset")

        proposed = {
            "resolution_policy": resolution_policy,
            "budget": None if budget is None else int(budget),
            "sampling_strategy": sampling_strategy,
            "accept_confidence": accept_confidence,
        }
        effective, source = effective_action(state, "resolve", proposed)
        resolution_policy = effective.get("resolution_policy")
        if resolution_policy not in _RESOLUTION_POLICIES:
            raise ValueError(f"Unknown resolution_policy: {resolution_policy}")
        if resolution_policy in ("vlm", "inspect_only"):
            review_fields = ("budget", "sampling_strategy", "accept_confidence")
            missing = [name for name in review_fields if effective.get(name) is None]
            if missing:
                raise ValueError(
                    "VLM resolution requires explicit decisions for: "
                    + ", ".join(missing)
                )
            budget = int(effective["budget"])
            sampling_strategy = effective["sampling_strategy"]
            accept_confidence = effective["accept_confidence"]
            if budget < 1:
                raise ValueError("budget must be >= 1")
            if budget > 200:
                raise ValueError("budget must be <= 200 (fixed per-round VLM cap)")
            if sampling_strategy not in (
                "pollution_defense", "rare_behavior_recovery", "information_gain", "verification"
            ):
                raise ValueError("Unknown sampling_strategy")
            if accept_confidence not in ("low", "medium", "high"):
                raise ValueError("Unknown accept_confidence")
        else:
            budget = None
            sampling_strategy = None
            accept_confidence = None
        effective = {
            **effective,
            "budget": budget,
            "sampling_strategy": sampling_strategy,
            "accept_confidence": accept_confidence,
        }
        if source.startswith("agent") and not rationale.strip():
            raise ValueError("Adaptive resolve decisions require an observation-based rationale")
        if source == "fixed_policy" and not rationale.strip():
            rationale = "Preregistered fixed baseline resolution policy"

        scores_ref = partition.get("scores_artifact") or state.get("latest_scores")
        scored = json.loads(
            _task_artifact_reference(workspace_dir, branch, scores_ref).read_text()
        )
        normality_scores(scored)
        score_map = {r["id"]: r for r in scored}
        keep = [score_map[i] for i in partition.get("keep_ids", []) if i in score_map]
        gray = [score_map[i] for i in partition.get("gray_ids", []) if i in score_map]

        review = {
            "accepted": [], "rejected": [], "unresolved": [], "call_failed": [],
            "model_unresolved": [], "below_accept_confidence": [],
            "technical_failures": [], "reviewed": [], "selected": 0,
            "api_calls": 0, "successful_responses": 0,
            "model_unresolved_count": 0, "below_accept_confidence_count": 0,
            "technical_failure_count": 0, "output_truncated": 0,
            "invalid_responses": 0, "response_status_counts": {},
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
            "score_contract": score_contract(),
            "keep_count": len(keep),
            "gray_count": len(gray),
            "vlm_selected": review["selected"],
            "vlm_api_calls": review["api_calls"],
            "vlm_successful_responses": review.get("successful_responses", 0),
            "vlm_accepted": len(accepted_ids),
            "vlm_rejected": len(rejected_ids),
            "vlm_unresolved": len(unresolved_ids),
            "vlm_model_unresolved": review.get("model_unresolved_count", 0),
            "vlm_below_accept_confidence": review.get("below_accept_confidence_count", 0),
            "vlm_technical_failures": review.get("technical_failure_count", 0),
            "vlm_output_truncated": review.get("output_truncated", 0),
            "vlm_invalid_responses": review.get("invalid_responses", 0),
            "vlm_call_failed": len(call_failed_ids),
            "vlm_response_status_counts": review.get("response_status_counts", {}),
            "vlm_max_tokens": review.get("max_tokens"),
            "vlm_thinking_mode": review.get("thinking_mode"),
            "vlm_prompt_version": review.get("prompt_version"),
            "vlm_review_artifact": review.get("review_artifact"),
            "quarantined": len(quarantined_ids),
        }
        decision_entry = record_decision(
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
            record_observation(
                state,
                "resolve",
                summary,
                workspace_dir=workspace_dir,
                branch=branch,
                decision=decision_entry,
            )
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
                "score_contract": score_contract(),
                "scores_artifact": Path(scores_ref).name,
                "round_input_fingerprint": state.get("round_input_fingerprint"),
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
                "policy_discarded_gray_ids": sorted({r["id"] for r in gray}) if resolution_policy == "auto_keep" else [],
            },
        )

        state["active_clean_dataset"] = str(clean_path)
        state["clean_count"] = len(clean)
        state["round_status"] = "resolved"
        summary = {
            "policy": resolution_policy,
            "decision_source": source,
            **decision_observation,
            "quarantined_count": len(quarantined_ids),
            "policy_discarded_gray_count": len(gray) if resolution_policy == "auto_keep" else 0,
            "clean_count": len(clean),
            "clean_dataset_id": clean_path.name,
            "clean_fingerprint": clean_payload["fingerprint"],
            "quarantine_artifact": quarantine_path.name,
        }
        record_observation(
            state,
            "resolve",
            summary,
            workspace_dir=workspace_dir,
            branch=branch,
            decision=decision_entry,
        )
        _save(workspace_dir, state, branch=branch)
        return json.dumps(summary, ensure_ascii=False)
