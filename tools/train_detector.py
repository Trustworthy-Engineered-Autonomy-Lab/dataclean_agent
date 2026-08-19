import json
import os
import time
import uuid
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base import Tool
from .decision_policy import effective_action, record_decision
from .models import UnifiedCAE
from .utils import _load, _save, _artifact, _records, DrivingDataset, _ensure_constraints, print_progress, record_observation

class TrainDetector(Tool):
    name = "train_detector"
    description = "Train or reuse the UnifiedCAE detector for the immutable current-round input D_t."
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
            "strategy": {"type": "string", "enum": ["retrain", "reuse", "retrain_first_then_reuse"]},
            "n_reference_latents": {
                "type": "integer",
                "minimum": 10,
                "maximum": 5000,
                "description": "Number of reference latent features for the latent-reference loss term (range: 10 to 5000). More = denser reference manifold, slower build. Default 500."
            },
            "seed": {"type": "integer", "description": "Training RNG seed."},
            "rationale": {
                "type": "string",
                "description": "Observation-based reason for adaptive detector strategy/hyperparameters."
            },
        },
        "required": ["lambda_value", "steer_lambda", "strategy"]
    }

    def run(self, lambda_value, steer_lambda, strategy, learning_rate=5e-4, epochs=60,
            dataset_id=None, n_reference_latents=500, seed=0, rationale="", branch="main",
            workspace_dir=None, cancel_event=None, **_):
        started = time.monotonic()
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        previous_detector = s.get("active_detector")
        if s.get("round_status", "ready") not in ("ready", "detector_ready"):
            raise ValueError("Detector training is only allowed before scoring the current D_t")
        if s.get("latest_scores"):
            raise ValueError("Current round is already scored; detector changes would invalidate downstream lineage")
        if s.get("round_status") == "resolved":
            raise ValueError("Current round is already resolved; commit it before training the next detector")
        proposed = {
            "strategy": strategy,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "lambda_value": lambda_value,
            "steer_lambda": steer_lambda,
            "n_reference_latents": n_reference_latents,
            "seed": seed,
        }
        effective, decision_source = effective_action(s, "detector", proposed)
        strategy = effective.get("strategy", strategy)
        learning_rate = effective.get("learning_rate", learning_rate)
        epochs = effective.get("epochs", epochs)
        lambda_value = effective.get("lambda_value", lambda_value)
        steer_lambda = effective.get("steer_lambda", steer_lambda)
        n_reference_latents = effective.get("n_reference_latents", n_reference_latents)
        seed = int(effective.get("seed", seed))
        if decision_source.startswith("agent") and not rationale.strip():
            raise ValueError("Adaptive detector decisions require an observation-based rationale")
        if decision_source == "fixed_policy" and not rationale.strip():
            rationale = "Preregistered fixed baseline detector policy"
        if strategy == "retrain_first_then_reuse":
            strategy = "reuse" if s.get("active_detector") else "retrain"
        if strategy not in ("retrain", "reuse"):
            raise ValueError("Unknown detector strategy")
        if strategy == "reuse" and not s.get("active_detector"):
            raise ValueError("No existing detector to reuse")
        effective = {**effective, "applied_strategy": strategy}

        detector_id = (
            s.get("active_detector") if strategy == "reuse"
            else f"det-r{s['round']}-{uuid.uuid4().hex[:12]}"
        )
        train_source = "D_t_round_input"

        c = s.get("constraints") or {}
        lock0 = (s["round"] == 0 and c.get("lock_round0_detector", False))
        if lock0:
            actual_lr = 5e-4
            actual_epochs = 10
            actual_lambda = 1.0
            actual_steer_lambda = 1.0
            epoch_msg = "Round 0 detector hyper-params locked (constraint lock_round0_detector)"
        else:
            actual_lr = float(learning_rate)
            actual_epochs = int(epochs)
            actual_lambda = float(lambda_value)
            actual_steer_lambda = float(steer_lambda)
            allowed_lambdas = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
            if not 5e-6 <= actual_lr <= 5e-4:
                raise ValueError("learning_rate must be in [5e-6, 5e-4]")
            if not 10 <= actual_epochs <= 120:
                raise ValueError("epochs must be in [10, 120]")
            if actual_lambda not in allowed_lambdas or actual_steer_lambda not in allowed_lambdas:
                raise ValueError("lambda_value and steer_lambda must use an allowed value")
            if not 10 <= int(n_reference_latents) <= 5000:
                raise ValueError("n_reference_latents must be in [10, 5000]")
            epoch_msg = f"Agent specified {actual_epochs} epochs, lr={actual_lr}"

        # Compute this only after the lock/default policy has resolved the
        # effective epoch count.  Referencing actual_epochs before this point
        # made every first detector-training call fail at runtime.
        epochs_consumed = actual_epochs if strategy == "retrain" else 0

        batch_size = 64

        recs = _records(workspace_dir, branch=branch)

        if len(recs) < 10:
            raise ValueError("Insufficient training samples")

        if strategy == "retrain":
            used_epochs = int(s.get("detector_train_epochs_used", 0))
            epoch_cap = (s.get("constraints") or {}).get("max_detector_train_epochs_total")
            if epoch_cap is not None and used_epochs + actual_epochs > int(epoch_cap):
                raise ValueError(
                    f"Detector epoch budget exceeded: {used_epochs}+{actual_epochs}>{int(epoch_cap)}"
                )
            # Reserve the full requested budget before training. Interrupted or
            # failed jobs still consumed compute and cannot be retried for free.
            s["detector_train_epochs_used"] = used_epochs + actual_epochs
            _save(workspace_dir, s, branch=branch)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        train_generator = torch.Generator().manual_seed(seed)
        model = UnifiedCAE().to(device)

        if strategy == "reuse":
            ckpt_path = _artifact(workspace_dir, f"{detector_id}.pt", branch=branch)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Cannot reuse missing detector checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            loss_summary = s.get("detector", {}).get("loss") or {}
            print(f"\n[Train Detector] Reusing existing detector: {detector_id}", flush=True)
        else:
            dataset = DrivingDataset(workspace_dir, recs)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, 
                num_workers=4 if torch.cuda.is_available() else 0,
                pin_memory=True if torch.cuda.is_available() else False,
                generator=train_generator,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
            mse = nn.MSELoss()

            ref_latents = None
            ref_indices = None
            warmup_epochs = max(1, actual_epochs // 5)
            if actual_lambda > 0:
                ref_size = min(int(n_reference_latents), len(recs))
                ref_indices = torch.randperm(len(recs), generator=train_generator)[:ref_size]
                print(
                    f"\n[Train Detector] Reference manifold will be frozen after "
                    f"{warmup_epochs} warm-up epochs ({ref_size} samples).",
                    flush=True,
                )

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

            try:
                for epoch in range(actual_epochs):
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Detector training cancelled")
                    if actual_lambda > 0 and ref_latents is None and epoch == warmup_epochs:
                        model.eval()
                        with torch.no_grad():
                            ref_imgs = torch.stack([dataset[int(i)][0] for i in ref_indices]).to(device)
                            _, ref_latents, _ = model(ref_imgs)
                            ref_latents = ref_latents.detach()
                        del ref_imgs
                    model.train()
                    ep_l = ep_rec = ep_reg = ep_pred = 0.0
                    ep_n = 0
                    for imgs, steers, _ in dataloader:
                        if cancel_event is not None and cancel_event.is_set():
                            raise InterruptedError("Detector training cancelled")
                        imgs, steers = imgs.to(device), steers.to(device)
                        recon, latent, steer_pred = model(imgs)

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
                if not all(math.isfinite(value) for value in loss_summary.values()):
                    raise RuntimeError("Detector training produced non-finite loss")

                print(f"\n[Train Detector] Training completed! Loss Summary: {loss_summary}\n", flush=True)

                ckpt_path = _artifact(workspace_dir, f"{detector_id}.pt", branch=branch)
                tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
                try:
                    torch.save(model.state_dict(), tmp_ckpt)
                    os.replace(tmp_ckpt, ckpt_path)
                finally:
                    tmp_ckpt.unlink(missing_ok=True)
            finally:
                del model
                if "ref_latents" in locals():
                    del ref_latents
                if "dataloader" in locals():
                    del dataloader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        s["active_detector"] = detector_id
        s["detector"] = {
            "id": detector_id, "strategy": strategy, "training_source": train_source,
            "round": s.get("round", 0),
            "learning_rate": actual_lr, "epochs_used": epochs_consumed,
            "configured_epochs": actual_epochs,
            "lambda": actual_lambda, "steer_lambda": actual_steer_lambda, "batch_size": batch_size,
            "seed": seed,
            "loss": loss_summary
        }
        s["round_status"] = "detector_ready"
        record_decision(
            s,
            "detector",
            proposed,
            effective,
            rationale,
            decision_source,
            observation={"round_input_count": len(recs), "previous_detector": previous_detector},
        )
        result = {
            "detector_id": detector_id, "strategy": strategy, "training_source": train_source,
            "learning_rate": actual_lr, "epochs_used": epochs_consumed,
            "configured_epochs": actual_epochs,
            "lambda_value": actual_lambda, "steer_lambda": actual_steer_lambda,
            "batch_size": batch_size, "seed": seed,
            "device": str(device),
            "duration_seconds": round(time.monotonic() - started, 6),
            "detector_epochs_used_total": s.get("detector_train_epochs_used", 0),
            "training_samples": len(recs), "loss_summary": loss_summary
        }
        record_observation(s, "train_detector", result, workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, s, branch=branch)
        return json.dumps(result, ensure_ascii=False)
