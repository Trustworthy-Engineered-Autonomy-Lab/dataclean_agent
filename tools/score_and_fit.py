import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .base import Tool
from .decision_policy import effective_action, record_decision
from .models import UnifiedCAE, calculate_pcc_tensor
from .utils import _load, _save, _artifact, _records, _quantiles, _combined_score, record_observation, DrivingDataset, print_progress, _dataset_config, _write_json_atomic

class ScoreAndFit(Tool):
    name = "score_and_fit"
    description = "Evaluate dataset using UnifiedCAE detector and calculate anomaly score distribution (mean and std)."
    parameters = {
        "type": "object",
        "properties": {
            "detector_id": {
                "type": "string",
                "description": "Optional specific detector ID to score. Defaults to active_detector."
            },
            "alpha": {
                "type": "number",
                "description": "Optional weighting alpha between reconstruction PCC and steering prediction error (range 0.0-1.0, default 0.5)."
            },
            "rationale": {
                "type": "string",
                "description": "Reason for adaptive score-component weighting."
            },
        },
        "required": []
    }

    def run(self, detector_id=None, alpha=0.5, rationale="", branch="main", workspace_dir=None, cancel_event=None, **_):
        started = time.monotonic()
        s = _load(workspace_dir, branch=branch)
        if s.get("round_status") not in ("detector_ready", "scored"):
            raise ValueError("Scoring requires a detector-ready round and cannot run after partition")
        proposed = {"alpha": float(alpha) if alpha is not None else 0.5}
        effective, decision_source = effective_action(s, "score", proposed)
        alpha = float(effective.get("alpha", proposed["alpha"]))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if decision_source.startswith("agent") and not rationale.strip():
            raise ValueError("Adaptive score decisions require an observation-based rationale")
        if decision_source == "fixed_policy" and not rationale.strip():
            rationale = "Preregistered fixed baseline score policy"
        target_detector = detector_id or s.get("active_detector")
        if not target_detector:
            raise ValueError("No detector_id specified and no active_detector active.")

        ckpt_path = _artifact(workspace_dir, f"{target_detector}.pt", branch=branch)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Detector weights file not found: {ckpt_path}")

        records = _records(workspace_dir, branch=branch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UnifiedCAE().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        dataset = DrivingDataset(workspace_dir, records)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)

        pcc_list, steer_err_list = [], []
        total_steps = len(dataloader)
        step = 0
        last_pct = -10.0
        with torch.no_grad():
            for imgs, steers, _ in dataloader:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Scoring cancelled")
                imgs, steers = imgs.to(device), steers.to(device)
                recon, _, steer_pred = model(imgs)
                pcc_batch = calculate_pcc_tensor(imgs, recon)
                err_batch = (steer_pred - steers).abs().squeeze(1)
                pcc_list.extend(pcc_batch.cpu().numpy().tolist())
                steer_err_list.extend(err_batch.cpu().numpy().tolist())

                step += 1
                pct = step / total_steps * 100
                if pct - last_pct >= 10.0 or step == total_steps:
                    print_progress(f"[Score & Fit] Scoring [{target_detector}] {step}/{total_steps} ({pct:3.0f}%)")
                    last_pct = pct

        pcc, err = np.array(pcc_list), np.array(steer_err_list)
        if not np.all(np.isfinite(pcc)) or not np.all(np.isfinite(err)):
            raise RuntimeError("Detector produced non-finite reconstruction or steering scores")
        alpha_val = float(alpha) if alpha is not None else 0.5
        scores = _combined_score(pcc, err, alpha=alpha_val)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Composite anomaly scoring produced non-finite values")

        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))

        path = _artifact(workspace_dir, f"scores_r{s.get('round', 0)}_{target_detector}.json", branch=branch)
        scored = [{**r, "pcc": round(float(pcc[i]), 6), "steer_error": round(float(err[i]), 6), "anomaly_score": round(float(scores[i]), 6)} for i, r in enumerate(records)]
        _write_json_atomic(path, scored)

        obs = {
            "detector_id_scored": target_detector,
            "alpha": alpha_val,
            "raw_count": len(records),
            "pcc": _quantiles(pcc),
            "anomaly_score": _quantiles(scores),
            "stats": {"mean": round(s_mean, 5), "std": round(s_std, 5)},
            "high_abs_steering_count": int(np.sum(np.abs([r['steering'] for r in records]) >= .35)),
            "score_artifact": path.name,
            "device": str(device),
            "duration_seconds": round(time.monotonic() - started, 6),
        }
        record_decision(
            s,
            "score",
            proposed,
            {"alpha": alpha},
            rationale,
            decision_source,
            observation={"detector_id": target_detector, "round_input_count": len(records)},
        )
        s["latest_scores"] = str(path)
        s["score_round"] = s.get("round", 0)
        s["score_detector_id"] = target_detector
        s["score_alpha"] = alpha_val
        s["round_status"] = "scored"
        record_observation(s, "score_and_fit", obs, workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, s, branch=branch)
        return json.dumps(obs, ensure_ascii=False)
