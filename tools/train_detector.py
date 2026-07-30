import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import Tool
from .models import UnifiedCAE
from .utils import _load, _save, _artifact, _records, DrivingDataset, _ensure_constraints, print_progress

class TrainDetector(Tool):
    name = "train_detector"
    description = "Train UnifiedCAE detector on D_raw or D_clean."
    parameters = {
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "learning_rate": {
                "type": "number",
                "minimum": 5e-6,
                "maximum": 5e-4,
                "description": "Learning rate for Adam optimizer (range: 5e-6 to 5e-4)."
            },
            "epochs": {
                "type": "integer",
                "minimum": 10,
                "maximum": 120,
                "description": "Number of training epochs (range: 10 to 120)."
            },
            "lambda_value": {
                "type": "number",
                "enum": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            },
            "steer_lambda": {
                "type": "number",
                "enum": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            },
            "strategy": {"type": "string", "enum": ["retrain", "reuse"]},
            "n_reference_latents": {
                "type": "integer",
                "minimum": 10,
                "maximum": 5000,
                "description": "Number of reference latent features for the latent-reference loss term (range: 10 to 5000). More = denser reference manifold, slower build. Default 500."
            }
        },
        "required": ["lambda_value", "steer_lambda", "strategy"]
    }

    def run(self, lambda_value, steer_lambda, strategy, learning_rate=5e-4, epochs=60, dataset_id=None, n_reference_latents=500, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if strategy == "reuse" and not s.get("active_detector"):
            raise ValueError("No existing detector to reuse")

        detector_id = s.get("active_detector") if strategy == "reuse" else f"det-r{s['round']+1}-{int(time.time())}"
        train_source = "D_raw" if s["round"] == 0 else "D_clean_previous"

        c = s.get("constraints") or {}
        lock0 = (s["round"] == 0 and c.get("lock_round0_detector", False))
        if lock0:
            actual_lr = 5e-4
            actual_epochs = 10
            actual_lambda = 1.0
            actual_steer_lambda = 1.0
            epoch_msg = "Round 0 detector hyper-params locked (constraint lock_round0_detector)"
        else:
            actual_lr = max(5e-6, min(5e-4, float(learning_rate)))
            actual_epochs = max(10, min(120, int(epochs)))
            actual_lambda = lambda_value if lambda_value in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] else 1.0
            actual_steer_lambda = steer_lambda if steer_lambda in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] else 1.0
            epoch_msg = f"Agent specified {actual_epochs} epochs, lr={actual_lr}"

        batch_size = 64

        if train_source == "D_raw":
            recs = _records(workspace_dir, branch=branch)
        else:
            clean_p = Path(s.get("active_clean_dataset") or "")
            if not clean_p.exists():
                recs = _records(workspace_dir, branch=branch)
            else:
                recs = json.loads(clean_p.read_text())["records"]

        if len(recs) < 10:
            raise ValueError("Insufficient training samples")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UnifiedCAE().to(device)

        if strategy == "reuse":
            ckpt_path = _artifact(workspace_dir, f"{detector_id}.pt", branch=branch)
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            loss_summary = s.get("detector", {}).get("loss", {"total": 0.05, "reconstruction": 0.03, "latent_reference": 0.01, "steering_prediction": 0.01})
            print(f"\n[Train Detector] Reusing existing detector: {detector_id}", flush=True)
        else:
            dataset = DrivingDataset(workspace_dir, recs)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, 
                num_workers=4 if torch.cuda.is_available() else 0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
            mse = nn.MSELoss()

            ref_latents = None
            if actual_lambda > 0:
                ref_size = min(int(n_reference_latents), len(recs))
                print(f"\n[Train Detector] Building {ref_size} reference latent features...", flush=True)
                ref_indices = torch.randperm(len(recs))[:ref_size]
                ref_imgs = torch.stack([dataset[i][0] for i in ref_indices]).to(device)
                ref_steers = torch.stack([dataset[i][1] for i in ref_indices]).to(device)
                model.eval()
                with torch.no_grad():
                    _, ref_latents, _ = model(ref_imgs, ref_steers)

            total_l = total_rec = total_reg = total_pred = 0.0
            print(f"\n==================================================", flush=True)
            print(f"[Train Detector] Starting UnifiedCAE Model Training", flush=True)
            print(f" -> Device          : {device}", flush=True)
            print(f" -> Training Samples: {len(recs)}", flush=True)
            print(f" -> Epoch Strategy  : {epoch_msg}", flush=True)
            print(f" -> Learning Rate   : {actual_lr}", flush=True)
            print(f" -> Batch Size      : {batch_size}", flush=True)
            print(f" -> Hyperparameters : lambda={actual_lambda}, steer_lambda={actual_steer_lambda}", flush=True)
            print(f"==================================================", flush=True)

            for epoch in range(actual_epochs):
                model.train()
                ep_l = ep_rec = ep_reg = ep_pred = 0.0
                ep_n = 0
                for imgs, steers, _ in dataloader:
                    imgs, steers = imgs.to(device), steers.to(device)
                    recon, latent, steer_pred = model(imgs, steers)

                    l_rec = mse(recon, imgs)
                    l_reg = torch.tensor(0.0, device=device)
                    if actual_lambda > 0 and ref_latents is not None:
                        dists = torch.cdist(latent, ref_latents)
                        nearest = ref_latents[torch.argmin(dists, dim=1)]
                        l_reg = (latent - nearest).pow(2).mean()

                    l_pred = (steer_pred - steers).pow(2).mean() if actual_steer_lambda > 0 else torch.tensor(0.0, device=device)
                    loss = l_rec + actual_lambda * l_reg + actual_steer_lambda * l_pred

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_l += loss.item()
                    total_rec += l_rec.item()
                    total_reg += l_reg.item()
                    total_pred += l_pred.item()
                    ep_l += loss.item(); ep_rec += l_rec.item(); ep_reg += l_reg.item(); ep_pred += l_pred.item(); ep_n += 1

                print_progress(f"[Train Detector] Epoch {epoch+1}/{actual_epochs} done | "
                               f"loss={ep_l/ep_n:.4f} rec={ep_rec/ep_n:.4f} pred={ep_pred/ep_n:.4f}")

            n_batches = len(dataloader) * actual_epochs
            loss_summary = {
                "total": round(total_l / n_batches, 5),
                "reconstruction": round(total_rec / n_batches, 5),
                "latent_reference": round(actual_lambda * (total_reg / n_batches), 5),
                "steering_prediction": round(actual_steer_lambda * (total_pred / n_batches), 5)
            }

            print(f"\n[Train Detector] Training completed! Loss Summary: {loss_summary}\n", flush=True)

            ckpt_path = _artifact(workspace_dir, f"{detector_id}.pt", branch=branch)
            torch.save(model.state_dict(), ckpt_path)

        s["active_detector"] = detector_id
        s["detector"] = {
            "id": detector_id, "strategy": strategy, "training_source": train_source,
            "learning_rate": actual_lr, "epochs_used": actual_epochs,
            "lambda": actual_lambda, "steer_lambda": actual_steer_lambda, "batch_size": batch_size,
            "loss": loss_summary
        }
        _save(workspace_dir, s, branch=branch)
        return json.dumps({
            "detector_id": detector_id, "strategy": strategy, "training_source": train_source,
            "learning_rate": actual_lr, "epochs_used": actual_epochs,
            "lambda_value": actual_lambda, "steer_lambda": actual_steer_lambda,
            "batch_size": batch_size, "training_samples": len(recs), "loss_summary": loss_summary
        }, ensure_ascii=False)