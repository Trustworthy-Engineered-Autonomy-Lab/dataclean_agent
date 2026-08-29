import json
import os
import time
import uuid
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.utils.data import DataLoader, random_split
from .base import Tool
from .decision_policy import effective_action, record_decision
from .models import ControllerCNN, ControllerDeploymentWrapper
from .image_contract import (
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, INPUT_CONTRACT_VERSION,
)
from .utils import (_load, _save, _artifact, record_observation, DrivingDataset,
                    print_progress, _ensure_constraints, _task_artifact_reference,
                    _load_dataset_snapshot)

class TrainController(Tool):
    name = "train_controller"
    description = "Train an NVIDIA-style steering controller on the resolved current-round clean dataset C_t."
    parameters = {
        "type": "object",
        "properties": {
            "clean_dataset_id": {
                "type": "string",
                "description": "Optional clean dataset artifact filename (e.g. 'clean_r1.json'). Defaults to active_clean_dataset."
            },
            "epochs": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Explicitly selected number of training epochs."
            },
            "batch_size": {
                "type": "integer",
                "minimum": 1,
                "maximum": 512,
                "description": "Explicitly selected training batch size."
            },
            "lr": {
                "type": "number",
                "minimum": 1e-6,
                "maximum": 0.01,
                "description": "Explicitly selected Adam learning rate."
            },
            "weight_decay": {
                "type": "number", "minimum": 0.0, "maximum": 0.1,
                "description": "Explicitly selected Adam weight decay.",
            },
            "validation_fraction": {
                "type": "number", "minimum": 0.05, "maximum": 0.4,
                "description": "Explicitly selected validation fraction.",
            },
            "seed": {"type": "integer", "description": "Training and split RNG seed."},
            "rationale": {
                "type": "string",
                "description": "Observation-based reason for the controller training configuration.",
            },
        },
        "required": [
            "epochs", "batch_size", "lr", "weight_decay",
            "validation_fraction", "seed", "rationale",
        ]
    }
    
    def run(self, clean_dataset_id=None, epochs=None, batch_size=None, lr=None,
            validation_fraction=None, seed=None, rationale="", weight_decay=None,
            branch="main", workspace_dir=None, cancel_event=None, **_):
        started = time.monotonic()
        s = _load(workspace_dir, branch=branch) or {}
        _ensure_constraints(s, branch)
        if s.get("round_status") != "resolved":
            raise ValueError("Controller training requires the current round's resolved C_t")
        proposed = {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "validation_fraction": validation_fraction,
            "seed": seed,
        }
        effective, decision_source = effective_action(s, "controller", proposed)
        required_fields = (
            "epochs", "batch_size", "lr", "weight_decay",
            "validation_fraction", "seed",
        )
        missing = [name for name in required_fields if effective.get(name) is None]
        if missing:
            raise ValueError(
                "train_controller requires explicit decisions for: " + ", ".join(missing)
            )
        epochs = int(effective["epochs"])
        batch_size = int(effective["batch_size"])
        lr = float(effective["lr"])
        weight_decay = float(effective["weight_decay"])
        validation_fraction = float(effective["validation_fraction"])
        seed = int(effective["seed"])
        if not 1 <= epochs <= 100:
            raise ValueError("Controller epochs must be in [1, 100]")
        if not 1 <= batch_size <= 512:
            raise ValueError("Controller batch_size must be in [1, 512]")
        if not 1e-6 <= lr <= 0.01:
            raise ValueError("Controller lr must be in [1e-6, 0.01]")
        if not 0.0 <= weight_decay <= 0.1:
            raise ValueError("Controller weight_decay must be in [0, 0.1]")
        if not 0.05 <= validation_fraction <= 0.4:
            raise ValueError("validation_fraction must be in [0.05, 0.4]")
        if decision_source.startswith("agent") and not rationale.strip():
            raise ValueError("Adaptive controller decisions require an observation-based rationale")
        elif decision_source == "fixed_policy" and not rationale.strip():
            rationale = "Preregistered fixed baseline controller policy"
        
        active_clean_ref = s.get("active_clean_dataset")
        if not active_clean_ref:
            raise ValueError("Resolved round has no active C_t snapshot")
        active_clean_path = _task_artifact_reference(
            workspace_dir, branch, active_clean_ref
        )
        clean_path = active_clean_path
        if clean_dataset_id:
            requested_path = _artifact(workspace_dir, clean_dataset_id, branch=branch)
            if requested_path.resolve() != active_clean_path.resolve():
                raise ValueError(
                    "Controller training must use the active current-round C_t; "
                    "prior or unrelated task artifacts are not valid training inputs"
                )

        try:
            clean_payload = _load_dataset_snapshot(clean_path)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            raise ValueError(f"Failed to read clean dataset {clean_path.name}: {e}") from e
        if clean_payload.get("task_id") != branch:
            raise ValueError("Clean dataset task identity does not match the active task")
        if clean_payload.get("role") != "clean":
            raise ValueError("Controller training input must be a clean C_t snapshot")
        if int(clean_payload.get("round", -1)) != int(s.get("round", 0)):
            raise ValueError("Controller training input belongs to a different round")
        data = clean_payload["records"]
        if len(data) < 10:
            raise ValueError(f"Clean dataset contains too few samples ({len(data)} samples; minimum 10 required)")

        used_epochs = int(s.get("controller_train_epochs_used", 0))
        epoch_cap = (s.get("constraints") or {}).get("max_controller_train_epochs_total")
        if epoch_cap is not None and used_epochs + epochs > int(epoch_cap):
            raise ValueError(
                f"Controller epoch budget exceeded: {used_epochs}+{epochs}>{int(epoch_cap)}"
            )
        s["controller_train_epochs_used"] = used_epochs + epochs
        _save(workspace_dir, s, branch=branch)
            
        steer = np.array([r["steering"] for r in data])
        cur_round = int(s.get("round", 0))
        cid = f"ctrl-{branch}-r{cur_round}-{uuid.uuid4().hex[:12]}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dataset = DrivingDataset(workspace_dir, data)
        bs = int(batch_size)
        val_n = max(1, int(round(len(dataset) * validation_fraction)))
        train_n = len(dataset) - val_n
        if train_n < 2:
            raise ValueError("Not enough samples for train/validation split")
        split_gen = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [train_n, val_n], generator=split_gen)
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, generator=split_gen,
                                  num_workers=2 if torch.cuda.is_available() else 0)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                                num_workers=2 if torch.cuda.is_available() else 0)

        model = ControllerCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        criterion = nn.MSELoss()

        num_epochs = int(epochs)
        final_loss = 0.0
        best_val_loss = float("inf")
        best_state = None

        try:
            for ep in range(1, num_epochs + 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Controller training cancelled")
                model.train()
                running_loss = 0.0
                total_batches = len(train_loader)
                for step, (imgs, steers, _) in enumerate(train_loader, 1):
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Controller training cancelled")
                    imgs, steers = imgs.to(device), steers.to(device)
                    optimizer.zero_grad()
                    preds = model(imgs)
                    loss = criterion(preds, steers)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / max(1, total_batches)
                final_loss = epoch_loss
                model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for imgs, steers, _ in val_loader:
                        imgs, steers = imgs.to(device), steers.to(device)
                        val_total += criterion(model(imgs), steers).item()
                val_loss = val_total / max(1, len(val_loader))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print_progress(f"[TrainController] Epoch {ep}/{num_epochs} train={epoch_loss:.5f} val={val_loss:.5f}")

            if best_state is None:
                raise RuntimeError("Controller training produced no checkpoint")
            if not math.isfinite(final_loss) or not math.isfinite(best_val_loss):
                raise RuntimeError("Controller training produced non-finite loss")
            model.load_state_dict(best_state)

            ckpt_path = _artifact(workspace_dir, f"{cid}.pt", branch=branch)
            tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
            try:
                torch.save(model.state_dict(), tmp_ckpt)
                os.replace(tmp_ckpt, ckpt_path)
            finally:
                tmp_ckpt.unlink(missing_ok=True)

            onnx_path = _artifact(workspace_dir, f"{cid}.onnx", branch=branch)
            tmp_onnx = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
            model_cpu = ControllerCNN().cpu()
            model_cpu.load_state_dict(model.state_dict())
            model_cpu.eval()
            deployment_model = ControllerDeploymentWrapper(model_cpu).eval()
            dummy_input = torch.zeros(
                1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, dtype=torch.float32
            )
            try:
                torch.onnx.export(
                    deployment_model,
                    dummy_input,
                    tmp_onnx,
                    export_params=True,
                    opset_version=17,
                    # Keep the lab's TorchScript/opset-17 export path explicit.
                    # Newer torch defaults to Dynamo, which adds onnxscript and
                    # may emit external weights for this temporary filename.
                    dynamo=False,
                    do_constant_folding=True,
                    input_names=["image"],
                    output_names=["steer"]
                )
                onnx.checker.check_model(onnx.load(str(tmp_onnx)))
                os.replace(tmp_onnx, onnx_path)
            finally:
                tmp_onnx.unlink(missing_ok=True)
            print_progress(f"[TrainController] Successfully exported single ONNX model to {onnx_path.name}")
        finally:
            del model
            if "train_loader" in locals():
                del train_loader
            if "val_loader" in locals():
                del val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics = {
            "train_samples": train_n,
            "validation_samples": val_n,
            "steering_mean": round(float(steer.mean()), 5),
            "steering_std": round(float(steer.std()), 5),
            "final_train_mse": round(final_loss, 5),
            "best_validation_mse": round(best_val_loss, 5),
            "seed": seed,
            "device": str(device),
            "duration_seconds": round(time.monotonic() - started, 6),
            "controller_epochs_used_total": s.get("controller_train_epochs_used", 0),
            "weights_pt_artifact": ckpt_path.name,
            "weights_onnx_artifact": onnx_path.name,
            "onnx_input_contract": (
                f"float32 NHWC [N,{IMAGE_HEIGHT},{IMAGE_WIDTH},{IMAGE_CHANNELS}], "
                "pixel range [0,255]"
            ),
            "onnx_input_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
            "input_contract_version": INPUT_CONTRACT_VERSION,
        }
        
        ctrl_info = {
            "id": cid,
            "round": cur_round,
            "quality": 1.0 / (1.0 + best_val_loss),
            "metrics": metrics,
            "weights": str(ckpt_path),
            "weights_onnx": str(onnx_path)
        }
        s["active_controller"] = ctrl_info
        decision_entry = record_decision(
            s, "controller", proposed, effective, rationale, decision_source,
            observation={"clean_count": len(data), "best_validation_mse": round(best_val_loss, 5)},
        )
        record_observation(
            s,
            "train_controller",
            ctrl_info,
            workspace_dir=workspace_dir,
            branch=branch,
            decision=decision_entry,
        )
        _save(workspace_dir, s, branch=branch)
        
        return json.dumps({"controller_id": cid, "metrics": metrics, "branch": branch}, ensure_ascii=False)
