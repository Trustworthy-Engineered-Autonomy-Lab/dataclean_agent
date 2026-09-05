"""Transfer one exact DeploymentRun into a task-local CollectionArtifact."""

import csv
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

from .base import Tool
from .collections import (
    collection_dir, collection_root, discover_source_dir, load_collection,
    write_collection_manifest,
)
try:
    from .teacar import TEACar
except ImportError:  # Core artifact tests remain usable without physical-car extras.
    TEACar = None
from .utils import _load, _save, _ensure_constraints, record_observation, print_progress


def _find_run(state, deployment_run_id):
    matches = [run for run in (state.get("deployment_runs") or [])
               if run.get("deployment_run_id") == deployment_run_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous deployment_run_id: {deployment_run_id}")
    return matches[0]


def _safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive contains unsupported link/device member: {member.name}")
        for member in tar.getmembers():
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Archive file has no readable content: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _cte_metrics(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ValueError("CTE CSV is empty")
    header = [cell.strip().lower().replace("-", "_").replace(" ", "_") for cell in rows[0]]
    names = {"cte_mm", "cross_track_error", "crosstrack_error", "lateral_error"}
    indexes = [i for i, name in enumerate(header) if name in names]
    if indexes:
        column, data_rows = indexes[0], rows[1:]
    elif len(rows[0]) == 1:
        column, data_rows = 0, rows
    else:
        raise ValueError("CTE CSV has no recognized CTE column")
    values = []
    for row in data_rows:
        if len(row) <= column:
            continue
        try:
            value = float(row[column].strip())
            if np.isfinite(value):
                values.append(value)
        except ValueError:
            continue
    if not values:
        raise ValueError("CTE CSV has no numeric values")
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    return {
        "real_cte_mean": round(float(np.mean(absolute)), 5),
        "real_cte_std": round(float(np.std(absolute)), 5),
        "real_cte_rmse": round(float(np.sqrt(np.mean(array ** 2))), 5),
        "real_cte_signed_mean": round(float(np.mean(array)), 5),
        "cte_samples": len(values),
    }


def _apply_transfer_state(state, run, artifact, metrics):
    """Idempotently link an already committed collection into task state."""
    collection_id = artifact["collection_id"]
    alias = artifact["anonymous_source"]
    if run.get("anonymous_source", alias) != alias:
        raise ValueError("DeploymentRun and collection disagree on anonymous source identity")
    run["anonymous_source"] = alias
    run["status"] = "transferred"
    run["collection_fingerprint"] = artifact["fingerprint"]
    run["collection_count"] = artifact["count"]
    pending = state.setdefault("pending_collection_ids", [])
    consumed = set(state.get("consumed_collection_ids") or [])
    if collection_id not in pending and collection_id not in consumed:
        pending.append(collection_id)

    # Physical evaluation is always an observable feedback signal.  Persist
    # Normalize the mode so legacy task state is migrated when it next
    # transfers a result.
    visibility = "online_feedback"
    state["evaluation_visibility"] = visibility
    state["last_deployed_cte"] = metrics["real_cte_mean"]
    current_best = state.get("best_cte")
    state["best_cte"] = metrics["real_cte_mean"] if current_best is None else min(
        current_best, metrics["real_cte_mean"]
    )
    return visibility, metrics


class TransferEvalResults(Tool):
    name = "transfer_eval_results"
    description = (
        "Transfer the exact remote artifacts produced by one DeploymentRun. "
        "Creates an immutable, task-local CollectionArtifact; it never changes BaseDataset."
    )
    parameters = {
        "type": "object",
        "properties": {
            "deployment_run_id": {
                "type": "string",
                "description": "Exact deployment_run_id returned by eval_controller.",
            }
        },
        "required": ["deployment_run_id"],
    }

    def run(self, deployment_run_id, branch="main", workspace_dir=None, **_):
        state = _load(workspace_dir, branch=branch)
        _ensure_constraints(state, branch)
        run = _find_run(state, deployment_run_id)
        if int(run.get("round", -1)) != int(state.get("round", 0)):
            raise ValueError("DeploymentRun does not belong to the active round")
        if run.get("status") not in {"completed", "transferred"}:
            raise ValueError(f"DeploymentRun is not transferable (status={run.get('status')})")
        collection_id = run.get("collection_id")
        if not collection_id:
            raise ValueError("DeploymentRun has no bound collection_id")

        manifest_path = collection_dir(workspace_dir, branch, collection_id) / "collection.json"
        if manifest_path.exists():
            artifact = load_collection(workspace_dir, branch, collection_id)
            metrics = _cte_metrics(collection_dir(workspace_dir, branch, collection_id) / "cte.csv")
            visibility, visible_metrics = _apply_transfer_state(state, run, artifact, metrics)
            result = {
                "status": "success", "idempotent": True,
                "deployment_run_id": deployment_run_id,
                "collection_id": collection_id, "collection_count": artifact["count"],
                "anonymous_source": artifact["anonymous_source"],
                "collection_fingerprint": artifact["fingerprint"],
                "evaluation_visibility": visibility, **visible_metrics,
            }
            record_observation(state, "transfer_eval_results", result,
                               workspace_dir=workspace_dir, branch=branch)
            _save(workspace_dir, state, branch=branch)
            return json.dumps(result, ensure_ascii=False)

        remote_data = run.get("remote_data_path")
        remote_cte = run.get("remote_cte_path")
        if not remote_data or not remote_cte:
            raise ValueError("DeploymentRun lacks exact remote data/CTE paths")

        root = collection_root(workspace_dir, branch)
        staging_root = root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        stage = staging_root / deployment_run_id
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        archive, cte_path, payload_dir = stage / "data.tar.gz", stage / "cte.csv", stage / "payload"
        final_dir = collection_dir(workspace_dir, branch, collection_id)

        print_progress(f"[TransferEvalResults] Fetching exact run {deployment_run_id}...")
        try:
            if TEACar is None:
                raise RuntimeError("Physical transfer requires the optional paramiko dependency")
            teacar = TEACar()
            with teacar as car_client:
                with car_client.open_sftp() as car_sftp:
                    car_sftp.stat(str(remote_data))
                    car_sftp.get(str(remote_data), str(archive))
                with teacar.jump.open_sftp() as jump_sftp:
                    jump_sftp.stat(str(remote_cte))
                    jump_sftp.get(str(remote_cte), str(cte_path))
            _safe_extract(archive, payload_dir)
            staged_source = discover_source_dir(payload_dir)
            metrics = _cte_metrics(cte_path)
            relative_source = staged_source.relative_to(stage)
            if final_dir.exists():
                raise FileExistsError(f"Collection target already exists without manifest: {final_dir}")
            stage.rename(final_dir)
            try:
                artifact = write_collection_manifest(
                    workspace_dir, branch, collection_id, deployment_run_id,
                    final_dir / relative_source, round_index=state.get("round", 0),
                    controller_id=run.get("controller_id"),
                    remote_paths={"data": remote_data, "cte": remote_cte},
                )
            except Exception:
                shutil.rmtree(final_dir)
                raise
        except Exception as exc:
            if stage.exists():
                shutil.rmtree(stage)
            return json.dumps({
                "status": "failed",
                "error": f"Failed to transfer DeploymentRun {deployment_run_id}: {exc}",
                "deployment_run_id": deployment_run_id,
            }, ensure_ascii=False)

        visibility, visible_metrics = _apply_transfer_state(state, run, artifact, metrics)

        result = {
            "status": "success", "deployment_run_id": deployment_run_id,
            "collection_id": collection_id, "collection_count": artifact["count"],
            "anonymous_source": artifact["anonymous_source"],
            "collection_fingerprint": artifact["fingerprint"],
            "evaluation_visibility": visibility, **visible_metrics,
            "message": "Task-local CollectionArtifact created; BaseDataset was not modified.",
        }
        record_observation(state, "transfer_eval_results", result,
                           workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, state, branch=branch)
        return json.dumps(result, ensure_ascii=False)
