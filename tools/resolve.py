import json
from pathlib import Path
from .base import Tool
from .utils import _load, _save, _artifact, record_observation, _advance_round
from .vlm_review import run_vlm_review
from .policies import POLICIES

_RESOLUTION_POLICIES = tuple(POLICIES["resolve"])


class Resolve(Tool):
    name = "resolve"
    description = (
        "Clean dataset assembly stage: generates active_clean_dataset based on partition results and resolution_policy. "
        "policy='auto_keep' skips VLM review (keeps only keep-region samples); "
        "policy='vlm' performs VLM review on gray region (assembles keep + accepted samples); "
        "policy='inspect_only' reviews gray region without producing a new clean set (active_clean_dataset untouched)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "resolution_policy": {
                "type": "string",
                "enum": list(_RESOLUTION_POLICIES),
                "description": "How the gray region is resolved: 'auto_keep' (keep region only, no VLM), 'vlm' (reviews gray with local VLM and emits clean set), 'inspect_only' (reviews gray with VLM without altering clean set)."
            },
            "budget": {
                "type": "integer",
                "minimum": 1,
                "description": "Only used when resolution_policy='vlm' or 'inspect_only': max gray samples to review this round."
            },
            "sampling_strategy": {
                "type": "string",
                "enum": ["pollution_defense", "rare_behavior_recovery", "information_gain", "verification"],
                "description": "Only used when resolution_policy='vlm' or 'inspect_only': gray ranking strategy."
            },
            "accept_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Only used when resolution_policy='vlm' or 'inspect_only': minimum confidence to accept a VLM 'keep'."
            },
            "vlm_model": {
                "type": "string",
                "description": "Optional: VLM reviewer model name. Defaults to task_spec/settings.json configuration."
            },
            "vlm_base_url": {
                "type": "string",
                "description": "Optional: VLM reviewer OpenAI-compatible base URL."
            },
            "vlm_api_key": {
                "type": "string",
                "description": "Optional: VLM reviewer API key."
            }
        },
        "required": ["resolution_policy"]
    }

    def run(self, resolution_policy="vlm", budget=200, sampling_strategy="pollution_defense",
            accept_confidence="high", branch="main", workspace_dir=None,
            vlm_model=None, vlm_base_url=None, vlm_api_key=None, **_):
        if resolution_policy not in ("vlm", "auto_keep", "inspect_only"):
            raise ValueError(f"Unknown resolution_policy: {resolution_policy}")
        s = _load(workspace_dir, branch=branch)
        p = s.get("latest_partition")
        if not p:
            raise ValueError("Must run partition before executing resolve.")
        if p.get("keep") is None:
            raise ValueError("Partition result missing keep region.")

        keep = p["keep"]
        gray = p.get("gray", [])

        if resolution_policy == "inspect_only":
            reviewed_n = 0
            if gray:
                run_vlm_review(workspace_dir, s, branch, budget, sampling_strategy,
                               accept_confidence, vlm_model, vlm_base_url, vlm_api_key)
                reviewed_n = s.get("latest_observation", {}).get("vlm", {}).get("reviewed", 0)
            record_observation(s, "resolve", {
                "policy": "inspect_only",
                "keep_count": len(keep),
                "gray_count": len(gray),
                "gray_discarded": len(gray),
                "vlm_reviewed": reviewed_n,
                "accepted_from_gray": 0,
                "clean_count": None,
                "clean_dataset_id": None,
                "note": "inspection only: gray reviewed, no clean set produced",
            })
            _save(workspace_dir, s, branch=branch)
            return json.dumps({
                "policy": "inspect_only",
                "keep_count": len(keep),
                "gray_count": len(gray),
                "vlm_reviewed": reviewed_n,
                "accepted_from_gray": 0,
                "clean_count": None,
                "review_artifact": s.get("latest_observation", {}).get("vlm", {}).get("review_artifact"),
                "note": "Gray zone reviewed by VLM without outputting clean dataset (active_clean_dataset unchanged); use policy=vlm to output clean dataset"
            }, ensure_ascii=False)

        accepted, reviewed_n = [], 0
        if resolution_policy == "vlm" and gray:
            accepted, _, reviewed_n = run_vlm_review(
                workspace_dir, s, branch, budget, sampling_strategy, accept_confidence,
                vlm_model, vlm_base_url, vlm_api_key)

        clean = keep + accepted
        clean_path = _artifact(workspace_dir, f"clean_r{s['round']+1}.json", branch=branch)
        clean_path.write_text(json.dumps(
            {"ids": [r["id"] for r in clean], "records": clean}, ensure_ascii=False, indent=2))

        s["active_clean_dataset"] = str(clean_path)
        s["clean_count"] = len(clean)
        record_observation(s, "resolve", {
            "policy": resolution_policy,
            "keep_count": len(keep),
            "gray_count": len(gray),
            "gray_discarded": len(gray) - len(accepted),
            "vlm_reviewed": reviewed_n,
            "accepted_from_gray": len(accepted),
            "clean_count": len(clean),
                "clean_dataset_id": clean_path.name,
        })
        _advance_round(s)
        _save(workspace_dir, s, branch=branch)

        return json.dumps({
            "policy": resolution_policy,
            "keep_count": len(keep),
            "gray_count": len(gray),
            "gray_discarded": len(gray) - len(accepted),
            "accepted_from_gray": len(accepted),
            "clean_count": len(clean),
            "clean_dataset_id": clean_path.name,
            "note": "auto_keep: gray region discarded without VLM" if resolution_policy == "auto_keep"
                    else "vlm: gray region reviewed by local VLM"
        }, ensure_ascii=False)
