import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .base import Tool
from .decision_policy import effective_action, record_decision
from .models import IROS2026CAE, calculate_pcc_tensor
from .detector_contract import DETECTOR_ARCHITECTURE, SCORE_CONTRACT_VERSION, score_contract
from .image_contract import (
    IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CONTRACT_VERSION,
)
from .utils import _load, _save, _artifact, _records, _quantiles, record_observation, DrivingDataset, print_progress, _write_json_atomic

class ScoreAndFit(Tool):
    name = "score_and_fit"
    description = (
        "Score D_t with the IROS2026 image+steering CAE. The only decision score is raw "
        "reconstruction PCC in [-1,1], HIGHER = more normal. No alpha, steering error, "
        "within-round normalization or smoothing. Reconstruction MSE is diagnostic only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "detector_id": {
                "type": "string",
                "description": "Optional specific detector ID to score. Defaults to active_detector."
            },
            "rationale": {
                "type": "string",
                "description": "Optional reason for scoring/re-scoring the current data with this detector."
            },
        },
        "required": []
    }

    def run(self, detector_id=None, rationale="", branch="main", workspace_dir=None, cancel_event=None, **_):
        started = time.monotonic()
        s = _load(workspace_dir, branch=branch)
        if s.get("round_status") not in ("detector_ready", "scored"):
            raise ValueError("Scoring requires a detector-ready round and cannot run after partition")
        if _.get("alpha") is not None:
            raise ValueError("alpha was removed: IROS2026 uses raw PCC only; omit alpha")
        proposed = {"method": "pcc"}
        effective, decision_source = effective_action(s, "score", proposed)
        if effective.get("alpha") is not None or effective.get("method") != "pcc":
            raise ValueError("Score policy is incompatible: use method=pcc without alpha")
        if decision_source == "fixed_policy" and not rationale.strip():
            rationale = "Preregistered fixed baseline score policy"
        target_detector = detector_id or s.get("active_detector")
        if not target_detector:
            raise ValueError("No detector_id specified and no active_detector active.")
        detector_meta = s.get("detector") or {}
        if (
            detector_meta.get("id") != target_detector
            or detector_meta.get("architecture") != DETECTOR_ARCHITECTURE
            or detector_meta.get("input_shape") != [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
            or detector_meta.get("input_contract_version") != INPUT_CONTRACT_VERSION
        ):
            raise ValueError(
                "Detector checkpoint is not the IROS2026 224x224 action-conditioned CAE. "
                "Retrain the detector before scoring."
            )

        ckpt_path = _artifact(workspace_dir, f"{target_detector}.pt", branch=branch)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Detector weights file not found: {ckpt_path}")

        records = _records(workspace_dir, branch=branch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IROS2026CAE().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        dataset = DrivingDataset(workspace_dir, records)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)

        pcc_list, mse_list = [], []
        total_steps = len(dataloader)
        step = 0
        last_pct = -10.0
        with torch.no_grad():
            for imgs, steers, _ in dataloader:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Scoring cancelled")
                imgs, steers = imgs.to(device), steers.to(device)
                recon, _ = model(imgs, steers)
                pcc_batch = calculate_pcc_tensor(imgs, recon)
                mse_batch = (recon - imgs).square().mean(dim=(1, 2, 3))
                pcc_list.extend(pcc_batch.cpu().numpy().tolist())
                mse_list.extend(mse_batch.cpu().numpy().tolist())

                step += 1
                pct = step / total_steps * 100
                if pct - last_pct >= 10.0 or step == total_steps:
                    print_progress(f"[Score & Fit] Scoring [{target_detector}] {step}/{total_steps} ({pct:3.0f}%)")
                    last_pct = pct

        pcc, mse = np.array(pcc_list), np.array(mse_list)
        if not len(pcc) or not np.all(np.isfinite(pcc)) or not np.all(np.isfinite(mse)):
            raise RuntimeError("Detector produced empty or non-finite reconstruction scores")
        scores = pcc

        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))

        path = _artifact(workspace_dir, f"scores_r{s.get('round', 0)}_{target_detector}.json", branch=branch)
        scored = []
        for i, record in enumerate(records):
            # C_t can carry previous-round diagnostic fields. Never propagate
            # legacy scores alongside a new score definition.
            clean_record = {k: v for k, v in record.items() if k not in (
                "anomaly_score", "steer_error", "score_alpha",
                "pcc", "normality_score", "reconstruction_mse", "score_contract_version",
            )}
            scored.append({
                **clean_record, "pcc": float(pcc[i]), "normality_score": float(pcc[i]),
                "reconstruction_mse": float(mse[i]),
                "score_contract_version": SCORE_CONTRACT_VERSION,
            })
        _write_json_atomic(path, scored)

        obs = {
            "detector_id_scored": target_detector,
            "architecture": DETECTOR_ARCHITECTURE,
            "score_contract": score_contract(),
            "raw_count": len(records),
            "pcc": _quantiles(pcc),
            "normality_score": _quantiles(scores),
            "reconstruction_mse": _quantiles(mse),
            "stats": {"mean": round(s_mean, 5), "std": round(s_std, 5)},
            "high_abs_steering_count": int(np.sum(np.abs([r['steering'] for r in records]) >= .35)),
            "score_artifact": path.name,
            "input_shape": [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            "input_contract_version": INPUT_CONTRACT_VERSION,
            "device": str(device),
            "duration_seconds": round(time.monotonic() - started, 6),
        }
        decision_entry = record_decision(
            s,
            "score",
            proposed,
            effective,
            rationale,
            decision_source,
            observation={"detector_id": target_detector, "round_input_count": len(records)},
        )
        s["latest_scores"] = str(path)
        s["score_round"] = s.get("round", 0)
        s["score_detector_id"] = target_detector
        s.pop("score_alpha", None)
        s["score_contract"] = score_contract()
        s["round_status"] = "scored"
        record_observation(
            s, "score_and_fit", obs, workspace_dir=workspace_dir,
            branch=branch, decision=decision_entry,
        )
        _save(workspace_dir, s, branch=branch)
        return json.dumps(obs, ensure_ascii=False)
