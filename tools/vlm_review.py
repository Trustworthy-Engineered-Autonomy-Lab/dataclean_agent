import json
import base64
import time
from io import BytesIO
from pathlib import Path
from PIL import Image
from .utils import (
    _artifact, _get_vlm_client, _build_vlm_prompt, _extract_vlm_response,
    _parse_vlm_response, resolve_vlm_config, record_observation,
    _reset_vlm_budget_if_new_round, print_progress, _write_json_atomic, _save,
    VLM_PROMPT_VERSION, vlm_prompt_hash,
)
from .io import _task_artifact_reference
from .detector_contract import normality_scores, require_partition_contract, review_rank


def run_vlm_review(workspace_dir, s, branch, budget, sampling_strategy, accept_confidence,
                   vlm_model=None, vlm_base_url=None, vlm_api_key=None,
                   cancel_event=None):
    p = s.get("latest_partition")
    if not p:
        raise ValueError("Must run partition before executing VLM review.")
    require_partition_contract(p)

    _reset_vlm_budget_if_new_round(s)

    per_round = s.get("vlm_budget_per_round", 500)
    remaining_budget = per_round - s.get("vlm_budget_used_this_round", 0)
    total_limit = (s.get("constraints") or {}).get("max_vlm_calls_total")
    remaining_total = None if total_limit is None else int(total_limit) - int(s.get("vlm_calls_total", 0))

    scores_ref = p.get("scores_artifact") or s.get("latest_scores")
    if not scores_ref:
        raise ValueError("Partition has no score artifact reference")
    scored = json.loads(
        _task_artifact_reference(workspace_dir, branch, scores_ref).read_text()
    )
    scored_map = {r["id"]: r for r in scored}
    normality_scores(scored)
    gray_candidates = [scored_map[i] for i in p.get("gray_ids", []) if i in scored_map]
    if not gray_candidates:
        return {
            "accepted": [], "rejected": [], "unresolved": [], "call_failed": [],
            "model_unresolved": [], "below_accept_confidence": [],
            "technical_failures": [], "reviewed": [], "selected": 0,
            "api_calls": 0, "response_status_counts": {},
        }

    if remaining_budget <= 0 or (remaining_total is not None and remaining_total <= 0):
        raise ValueError("VLM budget exhausted for the current round.")

    n = min(int(budget), remaining_budget, len(gray_candidates))
    if remaining_total is not None:
        n = min(n, remaining_total)

    threshold = float(p.get("threshold", 0.0))

    selected = sorted(
        gray_candidates, key=lambda r: review_rank(r, sampling_strategy, threshold),
    )[:n]
    cfg = resolve_vlm_config(vlm_model, vlm_base_url, vlm_api_key, state_vlm=s.get("vlm"))
    client = _get_vlm_client(cfg)
    levels = {"low": 0, "medium": 1, "high": 2}
    accepted, rejected, unresolved, call_failed_records, reviewed = [], [], [], [], []
    model_unresolved_records, below_confidence_records = [], []
    technical_failure_records = []
    actual_calls = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status_counts = {}
    prompt_hash = vlm_prompt_hash()
    started = time.monotonic()

    total_steps = len(selected)
    step = 0
    last_pct = -10.0
    for idx, r in enumerate(selected):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("VLM review cancelled")
        call_failed = False
        review_status = "success"
        response_meta = {
            "finish_reason": None,
            "content_type": None,
            "has_reasoning_content": False,
        }
        response_text = None
        per_call_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            img_full_path = (Path(workspace_dir).resolve() / r["image"]).resolve()
            if not img_full_path.is_relative_to(Path(workspace_dir).resolve()):
                raise ValueError("sample image escapes workspace")
            with Image.open(img_full_path) as source_img:
                pil_img = source_img.convert("RGB")
                buf = BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            call_failed = True
            review_status = "input_preparation_failed"
            vlm_label, conf, raw_dict = "unresolved", "low", {
                "reasoning": f"VLM input preparation failed: {type(exc).__name__}: {exc}",
                "parse_status": review_status,
            }
        else:
            actual_calls += 1
            # Persist the budget before the external call so cancellation or an
            # endpoint exception cannot make an already attempted call disappear.
            s["vlm_budget_used_this_round"] = int(s.get("vlm_budget_used_this_round", 0)) + 1
            s["vlm_calls_total"] = int(s.get("vlm_calls_total", 0)) + 1
            _save(workspace_dir, s, branch=branch)
            request = dict(
                model=cfg["model"],
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _build_vlm_prompt(r["steering"])},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}],
                temperature=0.1,
                extra_body={"chat_template_kwargs": dict(cfg["chat_template_kwargs"])},
            )
            if cfg["max_tokens"] is not None:
                request["max_tokens"] = cfg["max_tokens"]
            try:
                response = client.chat.completions.create(**request)
            except Exception as exc:
                call_failed = True
                review_status = "endpoint_failed"
                vlm_label, conf, raw_dict = "unresolved", "low", {
                    "reasoning": f"VLM endpoint call failed: {type(exc).__name__}: {exc}",
                    "parse_status": review_status,
                }
            else:
                response_usage = getattr(response, "usage", None)
                if response_usage is not None:
                    for key in usage:
                        value = int(getattr(response_usage, key, 0) or 0)
                        usage[key] += value
                        per_call_usage[key] = value
                response_text, response_meta = _extract_vlm_response(response)
                review_status = response_meta["status"]
                if review_status != "success":
                    vlm_label, conf, raw_dict = "unresolved", "low", {
                        "reasoning": (
                            "VLM output ended before a final response"
                            if review_status == "output_truncated"
                            else f"VLM returned no usable text ({review_status})"
                        ),
                        "parse_status": review_status,
                    }
                else:
                    vlm_label, conf, raw_dict = _parse_vlm_response(response_text)
                    review_status = raw_dict.get("parse_status", "invalid_schema")

        status_counts[review_status] = status_counts.get(review_status, 0) + 1
        valid_response = review_status == "success"
        is_clean = valid_response and vlm_label == "normal"
        accept = is_clean and levels.get(conf, -1) >= levels.get(accept_confidence, 2)
        reject = valid_response and vlm_label == "anomalous"
        if not valid_response:
            technical_failure_records.append(r)
            disposition = "technical_failure"
        elif vlm_label == "unresolved":
            model_unresolved_records.append(r)
            disposition = "model_unresolved"
        elif is_clean and not accept:
            below_confidence_records.append(r)
            disposition = "below_accept_confidence"
        elif accept:
            disposition = "accepted"
        else:
            disposition = "rejected"

        review_record = {
            "sample_id": r["id"], "source": r["source"],
            "label": "keep" if accept else ("discard" if reject else "unresolved"),
            "model_label": vlm_label,
            "confidence": conf,
            "reason": str(raw_dict.get("reasoning", f"{cfg['model']} review assessment")),
            "accepted": accept,
            "disposition": disposition,
            "call_failed": call_failed,
            "review_status": review_status,
            "finish_reason": response_meta.get("finish_reason"),
            "content_type": response_meta.get("content_type"),
            "has_reasoning_content": bool(response_meta.get("has_reasoning_content")),
            "request_max_tokens": cfg["max_tokens"],
            "request_enable_thinking": cfg["chat_template_kwargs"]["enable_thinking"],
            "prompt_version": VLM_PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "token_usage": per_call_usage,
        }
        if response_text is not None and review_status != "success":
            review_record["output_preview"] = response_text[:500]
        reviewed.append(review_record)
        if accept:
            accepted.append(r)
        elif reject:
            rejected.append(r)
        elif call_failed:
            call_failed_records.append(r)
        else:
            unresolved.append(r)

        step += 1
        pct = step / total_steps * 100
        if pct - last_pct >= 10.0 or step == total_steps:
            print_progress(f"[VLM Review] Progress {step}/{total_steps} ({pct:3.0f}%)")
            last_pct = pct

    vlm_review_path = _artifact(workspace_dir, f"vlm_review_r{s['round']}.json", branch=branch)
    _write_json_atomic(vlm_review_path, reviewed)

    diagnostic_summary = {
        "selected": n, "api_calls": actual_calls,
        "successful_responses": status_counts.get("success", 0),
        "accepted_count": len(accepted), "rejected_count": len(rejected),
        "model_unresolved_count": len(model_unresolved_records),
        "below_accept_confidence_count": len(below_confidence_records),
        "technical_failure_count": len(technical_failure_records),
        "output_truncated": status_counts.get("output_truncated", 0),
        "unresolved_total": len(unresolved),
        "call_failed_count": len(call_failed_records),
        "invalid_responses": sum(
            count for status, count in status_counts.items()
            if status not in {"success", "endpoint_failed", "input_preparation_failed"}
        ),
        "response_status_counts": status_counts,
        "used_this_round": s["vlm_budget_used_this_round"],
        "remaining_this_round": per_round - s["vlm_budget_used_this_round"],
        "strategy": sampling_strategy, "review_artifact": vlm_review_path.name,
        "model": cfg["model"], "base_url": cfg["base_url"],
        "temperature": 0.1, "seed": None, "max_tokens": cfg["max_tokens"],
        "thinking_mode": "non_thinking",
        "chat_template_kwargs": dict(cfg["chat_template_kwargs"]),
        "prompt_version": VLM_PROMPT_VERSION,
        "prompt_hash": prompt_hash,
        "token_usage": usage,
        "duration_seconds": round(time.monotonic() - started, 6),
    }
    record_observation(s, "vlm", diagnostic_summary)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "unresolved": unresolved,
        "call_failed": call_failed_records,
        "model_unresolved": model_unresolved_records,
        "below_accept_confidence": below_confidence_records,
        "technical_failures": technical_failure_records,
        "reviewed": reviewed,
        **diagnostic_summary,
    }
