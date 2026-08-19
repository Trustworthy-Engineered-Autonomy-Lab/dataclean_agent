import json
import base64
import time
from io import BytesIO
from pathlib import Path
from PIL import Image
from .utils import (_artifact, _get_vlm_client, _build_vlm_prompt, _parse_vlm_response,
                   resolve_vlm_config, record_observation, _reset_vlm_budget_if_new_round,
                   print_progress, _write_json_atomic, _save)
from .io import _task_artifact_reference


def run_vlm_review(workspace_dir, s, branch, budget, sampling_strategy, accept_confidence,
                   vlm_model=None, vlm_base_url=None, vlm_api_key=None,
                   cancel_event=None):
    p = s.get("latest_partition")
    if not p:
        raise ValueError("Must run partition before executing VLM review.")

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
    gray_candidates = [scored_map[i] for i in p.get("gray_ids", []) if i in scored_map]
    if not gray_candidates:
        return {
            "accepted": [], "rejected": [], "unresolved": [], "call_failed": [],
            "reviewed": [], "selected": 0, "api_calls": 0,
        }

    if remaining_budget <= 0 or (remaining_total is not None and remaining_total <= 0):
        raise ValueError("VLM budget exhausted for the current round.")

    n = min(int(budget), remaining_budget, len(gray_candidates))
    if remaining_total is not None:
        n = min(n, remaining_total)

    threshold = float(p.get("threshold", 0.0))

    def rank(r):
        score = r["anomaly_score"]
        if sampling_strategy == "pollution_defense":
            return -score
        if sampling_strategy == "rare_behavior_recovery":
            return (0 if abs(r["steering"]) >= .35 else 1, abs(score - threshold))
        if sampling_strategy == "information_gain":
            return abs(score - threshold)
        if sampling_strategy == "verification":
            return -abs(score - threshold)
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

    selected = sorted(gray_candidates, key=rank)[:n]
    cfg = resolve_vlm_config(vlm_model, vlm_base_url, vlm_api_key, state_vlm=s.get("vlm"))
    client = _get_vlm_client(cfg)
    levels = {"low": 0, "medium": 1, "high": 2}
    accepted, rejected, unresolved, call_failed_records, reviewed = [], [], [], [], []
    actual_calls = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    started = time.monotonic()

    total_steps = len(selected)
    step = 0
    last_pct = -10.0
    for idx, r in enumerate(selected):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("VLM review cancelled")
        call_failed = False
        try:
            img_full_path = (Path(workspace_dir).resolve() / r["image"]).resolve()
            if not img_full_path.is_relative_to(Path(workspace_dir).resolve()):
                raise ValueError("sample image escapes workspace")
            with Image.open(img_full_path) as source_img:
                pil_img = source_img.convert("RGB")
                buf = BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            actual_calls += 1
            # Persist the budget before the external call so cancellation or an
            # endpoint exception cannot make an already attempted call disappear.
            s["vlm_budget_used_this_round"] = int(s.get("vlm_budget_used_this_round", 0)) + 1
            s["vlm_calls_total"] = int(s.get("vlm_calls_total", 0)) + 1
            _save(workspace_dir, s, branch=branch)
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _build_vlm_prompt(r["steering"])},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}], max_tokens=512, temperature=0.1
            )
            response_usage = getattr(response, "usage", None)
            if response_usage is not None:
                for key in usage:
                    usage[key] += int(getattr(response_usage, key, 0) or 0)
            vlm_label, conf, raw_dict = _parse_vlm_response(response.choices[0].message.content)
        except Exception as e:
            call_failed = True
            vlm_label, conf, raw_dict = "unresolved", "low", {"reasoning": f"VLM endpoint call error: {str(e)}"}

        is_clean = (vlm_label == "normal")
        accept = (not call_failed) and is_clean and levels.get(conf, -1) >= levels.get(accept_confidence, 2)
        reject = (not call_failed) and vlm_label == "anomalous"

        reviewed.append({
            "sample_id": r["id"], "source": r["source"],
            "label": "keep" if accept else ("discard" if reject else "unresolved"),
            "confidence": conf,
            "reason": raw_dict.get("reasoning", f"{cfg['model']} review assessment"),
            "accepted": accept,
            "call_failed": call_failed,
        })
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

    record_observation(s, "vlm", {
        "selected": n, "api_calls": actual_calls,
        "successful_responses": n - len(call_failed_records),
        "accepted": len(accepted), "rejected": len(rejected),
        "unresolved": len(unresolved), "call_failed": len(call_failed_records),
        "used_this_round": s["vlm_budget_used_this_round"],
        "remaining_this_round": per_round - s["vlm_budget_used_this_round"],
        "strategy": sampling_strategy, "review_artifact": vlm_review_path.name,
        "model": cfg["model"], "base_url": cfg["base_url"],
        "temperature": 0.1, "seed": None, "token_usage": usage,
        "duration_seconds": round(time.monotonic() - started, 6),
    })
    return {
        "accepted": accepted,
        "rejected": rejected,
        "unresolved": unresolved,
        "call_failed": call_failed_records,
        "reviewed": reviewed,
        "selected": n,
        "api_calls": actual_calls,
        "model": cfg["model"],
        "token_usage": usage,
        "duration_seconds": round(time.monotonic() - started, 6),
    }
