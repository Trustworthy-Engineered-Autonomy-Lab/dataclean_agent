import json
import os
import time
import uuid
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from .base import Tool
from .decision_policy import effective_action, record_decision
from .models import IROS2026CAE
from .detector_contract import DETECTOR_ARCHITECTURE, score_contract
from .image_contract import (
    IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CONTRACT_VERSION,
)
from .utils import _load, _save, _artifact, _records, DrivingDataset, _ensure_constraints, print_progress, record_observation

class TrainDetector(Tool):
    name = "train_detector"
    description = (
        "Train or reuse the IROS2026 image+steering CAE on current D_t. "
        "Loss is reconstruction MSE + lambda * nearest frozen-reference latent MSE, "
        "active from epoch 1. No steering-prediction head or warm-up. "
        "Reference default: max(1, floor(N/50)); batch default: 256, as in the supplied code. "
        "For retraining, provide learning_rate, epochs, lambda_value and seed; reuse does not require them."
    )
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
                "minimum": 1,
                "maximum": 120,
                "description": "Number of training epochs (range: 1 to 120)."
            },
            "lambda_value": {
                "type": "number",
                "enum": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            },
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 512,
                           "description": "Default 256 (IROS2026); lower explicitly if GPU memory is limited."},
            "strategy": {"type": "string", "enum": ["retrain", "reuse", "retrain_first_then_reuse"]},
            "n_reference_latents": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional reference count override. Omit for the IROS2026 floor(N/50) rule (at least 1)."
            },
            "seed": {"type": "integer", "description": "Training RNG seed."},
            "rationale": {
                "type": "string",
                "description": "Observation-based reason for adaptive detector strategy/hyperparameters."
            },
        },
        "required": ["strategy", "rationale"],
    }

    def run(self, strategy=None, learning_rate=None, epochs=None, lambda_value=None,
            batch_size=None, dataset_id=None, n_reference_latents=None, seed=None,
            rationale="", branch="main",
            workspace_dir=None, cancel_event=None, **_):
        started = time.monotonic()
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if _.get("steer_lambda") is not None:
            raise ValueError("IROS2026 has no steer_lambda or steering prediction loss; remove that argument")
        previous_detector = s.get("active_detector")
        legacy_restart = (
            (s.get("detector") or {}).get("architecture") != DETECTOR_ARCHITECTURE
            and s.get("round_status") in ("scored", "partitioned")
            and not s.get("active_clean_dataset")
        )
        if s.get("round_status", "ready") not in ("ready", "detector_ready") and not legacy_restart:
            raise ValueError("Detector training is only allowed before scoring the current D_t")
        if s.get("latest_scores") and not legacy_restart:
            raise ValueError("Current round is already scored; detector changes would invalidate downstream lineage")
        if s.get("round_status") == "resolved":
            raise ValueError("Current round is already resolved; commit it before training the next detector")
        proposed = {
            "strategy": strategy,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "lambda_value": lambda_value,
            "batch_size": batch_size,
            "n_reference_latents": n_reference_latents,
            "seed": seed,
        }
        effective, decision_source = effective_action(s, "detector", proposed)
        strategy = effective.get("strategy")
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
        if effective.get("steer_lambda") is not None:
            raise ValueError("Legacy detector policy contains steer_lambda; preregister an IROS2026 policy")
        if strategy == "reuse" and (s.get("detector") or {}).get("architecture") != DETECTOR_ARCHITECTURE:
            raise ValueError("Legacy detector checkpoint cannot be reused; select strategy=retrain for IROS2026")

        recs = _records(workspace_dir, branch=branch)
        if len(recs) < 10:
            raise ValueError("Insufficient training samples")

        detector_id = (
            s.get("active_detector") if strategy == "reuse"
            else f"det-r{s['round']}-{uuid.uuid4().hex[:12]}"
        )
        train_source = "D_t_round_input"

        c = s.get("constraints") or {}
        lock0 = (s["round"] == 0 and c.get("lock_round0_detector", False))
        detector_meta = s.get("detector") or {}
        if strategy == "reuse":
            actual_lr = float(detector_meta.get("learning_rate"))
            actual_epochs = int(
                detector_meta.get("configured_epochs", detector_meta.get("epochs_used", 0))
            )
            actual_lambda = float(detector_meta.get("lambda"))
            actual_n_reference_latents = int(detector_meta["n_reference_latents"])
            batch_size = int(detector_meta["batch_size"])
            seed = int(detector_meta.get("seed", 0))
            epoch_msg = "Reusing the existing detector and its recorded training configuration"
        elif lock0:
            actual_lr = 5e-4
            actual_epochs = 10
            actual_lambda = 1.0
            actual_n_reference_latents = max(1, len(recs) // 50)
            batch_size = 256
            seed = 0
            epoch_msg = "Round 0 detector hyper-params locked (constraint lock_round0_detector)"
        else:
            training_fields = (
                "learning_rate", "epochs", "lambda_value", "seed",
            )
            missing = [name for name in training_fields if effective.get(name) is None]
            if missing:
                raise ValueError(
                    "Detector retraining requires explicit decisions for: "
                    + ", ".join(missing)
                )
            actual_lr = float(effective["learning_rate"])
            actual_epochs = int(effective["epochs"])
            actual_lambda = float(effective["lambda_value"])
            reference_count = effective.get("n_reference_latents")
            actual_n_reference_latents = (
                max(1, len(recs) // 50) if reference_count is None else int(reference_count)
            )
            batch_size = 256 if effective.get("batch_size") is None else int(effective["batch_size"])
            seed = int(effective["seed"])
            allowed_lambdas = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
            if not 5e-6 <= actual_lr <= 5e-4:
                raise ValueError("learning_rate must be in [5e-6, 5e-4]")
            if not 1 <= actual_epochs <= 120:
                raise ValueError("epochs must be in [1, 120]")
            if actual_lambda not in allowed_lambdas:
                raise ValueError("lambda_value must use an allowed value")
            if not 1 <= actual_n_reference_latents <= len(recs):
                raise ValueError("n_reference_latents must be between 1 and the dataset size")
            if not 1 <= batch_size <= 512:
                raise ValueError("batch_size must be in [1, 512]")
            epoch_msg = f"Agent specified {actual_epochs} epochs, lr={actual_lr}"

        effective = {
            **effective,
            "applied_strategy": strategy,
            "learning_rate": actual_lr,
            "epochs": actual_epochs,
            "lambda_value": actual_lambda,
            "batch_size": batch_size,
            "n_reference_latents": actual_n_reference_latents,
            "seed": seed,
        }

        # Compute this only after the adaptive/fixed/reuse policy has resolved
        # the effective epoch count.
        epochs_consumed = actual_epochs if strategy == "retrain" else 0

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
        model = IROS2026CAE().to(device)

        if strategy == "reuse":
            detector_meta = s.get("detector") or {}
            expected_shape = [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]
            if (
                detector_meta.get("id") != detector_id
                or detector_meta.get("input_shape") != expected_shape
                or detector_meta.get("input_contract_version") != INPUT_CONTRACT_VERSION
            ):
                raise ValueError(
                    "The requested detector does not declare the current 224x224 input "
                    "contract. Retrain the detector before reuse."
                )
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

            # Match IROS2026: sample 2% of D_t, encode with the INITIAL model
            # in eval mode, then freeze these latents for the entire training.
            # Batch the reference pass so it does not load all RGB images on GPU.
            ref_indices = np.random.RandomState(seed).choice(
                len(recs), size=actual_n_reference_latents, replace=False,
            )
            reference_loader = DataLoader(
                Subset(dataset, ref_indices.tolist()), batch_size=batch_size, shuffle=False,
            )
            model.eval()
            references = []
            with torch.no_grad():
                for ref_imgs, ref_steers, _idx in reference_loader:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Detector reference initialization cancelled")
                    _, ref = model(ref_imgs.to(device), ref_steers.to(device))
                    references.append(ref.detach())
            ref_latents = torch.cat(references)
            del references, reference_loader

            total_l = total_rec = total_reg = 0.0
            print(f"\n==================================================", flush=True)
            print(f"[Train Detector] Starting IROS2026 action-conditioned CAE training", flush=True)
            print(f" -> Device          : {device}", flush=True)
            print(f" -> Training Samples: {len(recs)}", flush=True)
            print(f" -> Epoch Strategy  : {epoch_msg}", flush=True)
            print(f" -> Learning Rate   : {actual_lr}", flush=True)
            print(f" -> Batch Size      : {batch_size}", flush=True)
            print(f" -> Reference loss  : lambda={actual_lambda}, {actual_n_reference_latents} frozen initial latents, active from epoch 1", flush=True)
            print(f"==================================================", flush=True)

            try:
                for epoch in range(actual_epochs):
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Detector training cancelled")
                    model.train()
                    ep_l = ep_rec = ep_reg = 0.0
                    ep_n = 0
                    for imgs, steers, _ in dataloader:
                        if cancel_event is not None and cancel_event.is_set():
                            raise InterruptedError("Detector training cancelled")
                        imgs, steers = imgs.to(device), steers.to(device)
                        recon, latent = model(imgs, steers)

                        l_rec = mse(recon, imgs)
                        l_reg = torch.tensor(0.0, device=device)
                        if actual_lambda > 0 and ref_latents is not None:
                            dists = torch.cdist(latent, ref_latents)
                            nearest = ref_latents[torch.argmin(dists, dim=1)]
                            l_reg = (latent - nearest).pow(2).mean()

                        loss = l_rec + actual_lambda * l_reg
                        if not torch.isfinite(loss):
                            raise RuntimeError("Detector training produced non-finite loss")

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_l += loss.item()
                        total_rec += l_rec.item()
                        total_reg += l_reg.item()
                        ep_l += loss.item(); ep_rec += l_rec.item(); ep_reg += l_reg.item(); ep_n += 1

                    print_progress(f"[Train Detector] Epoch {epoch+1}/{actual_epochs} done | "
                                   f"loss={ep_l/ep_n:.4f} rec={ep_rec/ep_n:.4f} ref={actual_lambda*ep_reg/ep_n:.4f}")

                n_batches = len(dataloader) * actual_epochs
                loss_summary = {
                    "total": round(total_l / n_batches, 5),
                    "reconstruction": round(total_rec / n_batches, 5),
                    "latent_reference": round(actual_lambda * (total_reg / n_batches), 5),
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

        if legacy_restart:
            # Keep old artifacts on disk, but do not let their score direction
            # survive into a newly trained detector's downstream actions.
            for key in ("latest_scores", "latest_partition", "score_round", "score_detector_id", "score_contract"):
                s[key] = None
            for stage in ("score_and_fit", "partition", "resolve", "evaluate"):
                (s.get("latest_observation") or {}).pop(stage, None)
        s.pop("score_alpha", None)
        s["active_detector"] = detector_id
        s["detector"] = {
            "id": detector_id, "strategy": strategy, "training_source": train_source,
            "round": detector_meta.get("round", s.get("round", 0)) if strategy == "reuse" else s.get("round", 0),
            "learning_rate": actual_lr, "epochs_used": epochs_consumed,
            "configured_epochs": actual_epochs,
            "lambda": actual_lambda, "batch_size": batch_size,
            "architecture": DETECTOR_ARCHITECTURE,
            "reference_initialization": "frozen_before_first_optimizer_step",
            "score_contract": score_contract(),
            "n_reference_latents": actual_n_reference_latents, "seed": seed,
            "input_shape": [IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            "input_contract_version": INPUT_CONTRACT_VERSION,
            "loss": loss_summary
        }
        s["round_status"] = "detector_ready"
        decision_entry = record_decision(
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
            "lambda_value": actual_lambda,
            "architecture": DETECTOR_ARCHITECTURE,
            "reference_initialization": "frozen_before_first_optimizer_step",
            "score_contract": score_contract(),
            "n_reference_latents": actual_n_reference_latents,
            "batch_size": batch_size, "seed": seed,
            "device": str(device),
            "duration_seconds": round(time.monotonic() - started, 6),
            "detector_epochs_used_total": s.get("detector_train_epochs_used", 0),
            "training_samples": len(recs), "loss_summary": loss_summary
        }
        record_observation(
            s, "train_detector", result, workspace_dir=workspace_dir,
            branch=branch, decision=decision_entry,
        )
        _save(workspace_dir, s, branch=branch)
        return json.dumps(result, ensure_ascii=False)
