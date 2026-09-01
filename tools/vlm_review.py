import json
import base64
import hashlib
import os
import re
import shutil
import tempfile
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
from .dataset import _anonymize_source_name, _deanonymize_source_name, _load_dataset_registry
from .io import _task_artifact_reference
from .detector_contract import normality_scores, require_partition_contract, review_rank


def _ground_truth_class(source, workspace_dir):
    """Resolve post-hoc binary truth from the configured source directory only.

    This helper is intentionally used only for the user-facing post-hoc report;
    its result is never placed in Agent observations or the canonical dataset.
    Unmapped collection sources remain unknown rather than being guessed anomalous.
    """
    try:
        registry = _load_dataset_registry(workspace_dir) or {}
        sources = registry.get("sources") or []
        names = {str(item.get("name")) for item in sources if item.get("name")}
    except Exception:
        names = set()
        sources = []
    raw = str(source or "")
    real = _deanonymize_source_name(raw, sources)
    if real == "normal" or raw == "normal" or raw == "src_01":
        return "normal"
    if real in names or raw in names:
        return "anomalous"
    return "unknown"


def _safe_copy_name(sample_id, source, image):
    alias = _anonymize_source_name(str(source or "unknown")) or "unknown"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8]
    suffix = Path(str(image)).suffix.lower() or ".jpg"
    return f"{alias}__{safe_id}_{digest}{suffix}"


def _posthoc_vlm_artifacts(workspace_dir, branch, round_index, reviewed, accepted):
    """Write user-only accepted-image and label-based analysis artifacts."""
    artifact_dir = _artifact(workspace_dir, f"vlm_accepted_r{round_index}", branch=branch)
    if artifact_dir.exists():
        raise FileExistsError(f"VLM accepted-image artifact already exists: {artifact_dir.name}")
    root = artifact_dir.parent
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact_dir.name}-", dir=str(root)))
    image_dir = staging / "images"
    image_dir.mkdir()
    try:
        manifest_records = []
        reviewed_by_id = {str(item.get("sample_id")): item for item in reviewed}
        workspace = Path(workspace_dir).resolve()
        for record in accepted:
            source_path = (workspace / str(record["image"])).resolve()
            if not source_path.is_relative_to(workspace) or not source_path.is_file():
                raise FileNotFoundError(f"Accepted sample image is unavailable: {record.get('id')}")
            filename = _safe_copy_name(record["id"], record.get("source"), record["image"])
            shutil.copy2(source_path, image_dir / filename)
            manifest_records.append({
                "sample_id": record["id"],
                "anonymous_source": _anonymize_source_name(str(record.get("source", ""))),
                "copied_file": f"images/{filename}",
                "model_label": reviewed_by_id.get(str(record["id"]), {}).get("model_label"),
                "confidence": reviewed_by_id.get(str(record["id"]), {}).get("confidence"),
                "accepted": True,
            })
        (staging / "manifest.json").write_text(json.dumps({
            "round": int(round_index), "accepted_count": len(accepted),
            "records": manifest_records,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, artifact_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    outcome = {key: {"sent": 0, "pred_normal": 0, "pred_anomalous": 0,
                     "unresolved": 0, "technical_failure": 0, "accepted": 0}
               for key in ("normal", "anomalous", "unknown")}
    for item in reviewed:
        truth = _ground_truth_class(item.get("source"), workspace_dir)
        row = outcome[truth]
        row["sent"] += 1
        technical = bool(item.get("call_failed")) or item.get("review_status") in {
            "endpoint_failed", "input_preparation_failed", "output_truncated",
            "empty_content", "unsupported_content_type", "invalid_schema", "malformed_json",
        }
        prediction = item.get("model_label")
        if technical:
            row["technical_failure"] += 1
        elif prediction == "normal":
            row["pred_normal"] += 1
        elif prediction == "anomalous":
            row["pred_anomalous"] += 1
        else:
            row["unresolved"] += 1
        if item.get("accepted"):
            row["accepted"] += 1
        if truth == "normal" and prediction == "normal":
            matrix["TP"] += 1
        elif truth == "normal" and prediction == "anomalous":
            matrix["FN"] += 1
        elif truth == "anomalous" and prediction == "normal":
            matrix["FP"] += 1
        elif truth == "anomalous" and prediction == "anomalous":
            matrix["TN"] += 1

    report_path = _artifact(workspace_dir, f"vlm_review_r{round_index}.txt", branch=branch)
    labeled = outcome["normal"]["sent"] + outcome["anomalous"]["sent"]
    valid = sum(matrix.values())
    lines = [
        f"VLM post-hoc analysis report - round {round_index}",
        "Ground truth: source directory 'normal' = normal; other registered source directories = anomalous.",
        "Unmapped/collection sources = unknown and are excluded from TP/FP/FN/TN.",
        "",
        f"Images sent to VLM: {len(reviewed)}",
        f"Images accepted back to C_t: {len(accepted)}",
        f"Labeled images: {labeled}; valid binary predictions: {valid}",
        "",
        "Confusion Matrix (normal is the positive class; unresolved/technical failures excluded):",
        "                         Pred normal   Pred anomalous",
        f"Ground-truth normal          {matrix['TP']:>8}       {matrix['FN']:>8}",
        f"Ground-truth anomalous       {matrix['FP']:>8}       {matrix['TN']:>8}",
        "",
        "Per-ground-truth outcome table:",
        "class       sent  pred_normal  pred_anomalous  unresolved  technical_failure  accepted_to_C_t",
    ]
    for key in ("normal", "anomalous", "unknown"):
        row = outcome[key]
        lines.append(f"{key:<10} {row['sent']:>5} {row['pred_normal']:>12} {row['pred_anomalous']:>15} "
                     f"{row['unresolved']:>11} {row['technical_failure']:>19} {row['accepted']:>16}")
    lines.extend(["", f"Accepted-image artifact: {artifact_dir.name}/", "Manifest: manifest.json"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"accepted_images_artifact": artifact_dir.name,
            "posthoc_report_artifact": report_path.name}


def run_vlm_review(workspace_dir, s, branch, budget, sampling_strategy, accept_confidence,
                   vlm_model=None, vlm_base_url=None, vlm_api_key=None,
                   cancel_event=None):
    p = s.get("latest_partition")
    if not p:
        raise ValueError("Must run partition before executing VLM review.")
    require_partition_contract(p)

    _reset_vlm_budget_if_new_round(s)

    # Enforce the fixed cap even for state files created before this setting
    # was introduced.  A call may still request fewer than 200 samples.
    configured_cap = s.get("vlm_budget_per_round", 200)
    per_round = min(200, int(200 if configured_cap is None else configured_cap))
    s["vlm_budget_per_round"] = per_round
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
    try:
        posthoc_artifacts = _posthoc_vlm_artifacts(
            workspace_dir, branch, int(s["round"]), reviewed, accepted,
        )
    except Exception as exc:
        # The user-only analysis artifact must never invalidate a successful VLM
        # decision or prevent C_t from being materialized.
        posthoc_artifacts = {"accepted_images_artifact": None,
                             "posthoc_report_artifact": None,
                             "posthoc_artifact_error": f"{type(exc).__name__}: {exc}"}

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
        **posthoc_artifacts,
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
