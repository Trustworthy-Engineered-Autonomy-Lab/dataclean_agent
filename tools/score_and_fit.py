import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .base import Tool
from .models import UnifiedCAE, calculate_pcc_tensor
from .utils import _load, _save, _artifact, _records, _quantiles, _combined_score, record_observation, DrivingDataset, print_progress, _dataset_config

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
            }
        },
        "required": []
    }

    def run(self, detector_id=None, alpha=0.5, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
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
                imgs, steers = imgs.to(device), steers.to(device)
                recon, _, steer_pred = model(imgs, steers)
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
        alpha_val = float(alpha) if alpha is not None else 0.5
        scores = _combined_score(pcc, err, alpha=alpha_val)

        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))

        path = _artifact(workspace_dir, f"scores_{target_detector}.json", branch=branch)
        scored = [{**r, "pcc": round(float(pcc[i]), 6), "steer_error": round(float(err[i]), 6), "anomaly_score": round(float(scores[i]), 6)} for i, r in enumerate(records)]
        path.write_text(json.dumps(scored, ensure_ascii=False, indent=2))

        obs = {
            "detector_id_scored": target_detector,
            "raw_count": len(records),
            "pcc": _quantiles(pcc),
            "anomaly_score": _quantiles(scores),
            "stats": {"mean": round(s_mean, 5), "std": round(s_std, 5)},
            "high_abs_steering_count": int(np.sum(np.abs([r['steering'] for r in records]) >= .35)),
            "score_artifact": path.name
        }
        s["latest_scores"] = str(path)
        record_observation(s, "score_and_fit", obs)
        _save(workspace_dir, s, branch=branch)
        return json.dumps(obs, ensure_ascii=False)