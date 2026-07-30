import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .base import Tool
from .models import UnifiedCAE, calculate_pcc_tensor
from .utils import (_load, _save, _artifact, _records, _quantiles, _combined_score,
                    record_observation, append_ledger, DrivingDataset, print_progress, _ensure_constraints)
from .partition import Partition


class ScoreAndPartition(Tool):
    name = "score_and_partition"
    description = (
        "Evaluate dataset using UnifiedCAE detector, calculate anomaly score distribution, "
        "and optionally partition dataset into keep/gray regions in a single unified step. "
        "Specify threshold or set auto_partition=True to execute partition immediately, "
        "or omit threshold to run candidate analysis and score distribution inspection."
    )
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
            "threshold": {
                "type": "number",
                "description": "Optional partitioning threshold. If specified, partitions dataset into keep/gray regions immediately."
            },
            "auto_partition": {
                "type": "boolean",
                "description": "If true and threshold is omitted, automatically selects optimal candidate threshold (e.g. Otsu/KDE valley) and partitions dataset."
            }
        },
        "required": []
    }

    def run(self, detector_id=None, alpha=0.5, threshold=None, auto_partition=False, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
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
                    print_progress(f"[Score & Partition] Processing [{target_detector}] {step}/{total_steps} ({pct:3.0f}%)")
                    last_pct = pct

        pcc, err = np.array(pcc_list), np.array(steer_err_list)
        alpha_val = float(alpha) if alpha is not None else 0.5
        scores = _combined_score(pcc, err, alpha=alpha_val)

        s_mean, s_std = float(np.mean(scores)), float(np.std(scores))
        path = _artifact(workspace_dir, f"scores_{target_detector}.json", branch=branch)
        scored = [{**r, "pcc": round(float(pcc[i]), 6), "steer_error": round(float(err[i]), 6), "anomaly_score": round(float(scores[i]), 6)} for i, r in enumerate(records)]
        path.write_text(json.dumps(scored, ensure_ascii=False, indent=2))
        s["latest_scores"] = str(path)

        partition_tool = Partition()
        candidates = partition_tool._compute_candidates(scores)
        stats = partition_tool._score_stats(scores)

        chosen_threshold = threshold
        if chosen_threshold is None and auto_partition:
            chosen_threshold = candidates.get("otsu_threshold") or candidates.get("kde_valley_threshold") or candidates.get("mean_plus_std_threshold")

        c = s.get("constraints") or {}
        locked_thr = c.get("locked_threshold")
        if locked_thr is not None and chosen_threshold is None:
            chosen_threshold = float(locked_thr)

        if chosen_threshold is None:
            summary = {
                "mode": "scored_and_analyzed",
                "detector_id": target_detector,
                "n_samples": int(len(records)),
                "candidates": candidates,
                "score_stats": stats,
                "scores_artifact": path.name,
            }
            record_observation(s, "score_and_partition", summary, workspace_dir=workspace_dir, branch=branch)
            _save(workspace_dir, s, branch=branch)
            return json.dumps(summary, ensure_ascii=False)

        thr = float(chosen_threshold)
        keep = [r for r in scored if r["anomaly_score"] < thr]
        gray = [r for r in scored if r["anomaly_score"] >= thr]
        s["latest_partition"] = {
            "threshold": round(thr, 5),
            "keep": keep,
            "gray": gray,
        }
        summary = {
            "mode": "scored_and_partitioned",
            "detector_id": target_detector,
            "threshold_applied": round(thr, 5),
            "keep_count": int(len(keep)),
            "gray_count": int(len(gray)),
            "keep_ratio": round(len(keep) / max(1, len(records)), 5),
            "candidates": candidates,
            "score_stats": stats,
            "scores_artifact": path.name,
        }
        record_observation(s, "score_and_partition", summary, workspace_dir=workspace_dir, branch=branch)
        append_ledger(s, {
            "stage": "score_and_partition",
            "round": s.get("round"),
            "threshold": round(thr, 5),
            "keep": int(len(keep)),
            "gray": int(len(gray)),
        })
        _save(workspace_dir, s, branch=branch)
        return json.dumps(summary, ensure_ascii=False)
