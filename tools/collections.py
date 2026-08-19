"""Task-local deployment collection artifacts.

Collected driving data is experimental output, not a new BaseDataset source.  It
therefore lives below one task and can enter a later D_t only through an explicit
collection ID recorded by ``commit_round``.
"""

import csv
import json
import math
import re
import time
from pathlib import Path

from .dataset import _dataset_fingerprint
from .io import _task_dir, _write_json_atomic


_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MANIFEST = "collection.json"


def _safe_id(value, field):
    value = str(value or "").strip()
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} must contain only letters, numbers, underscores, and hyphens")
    return value


def collection_root(workspace_dir, branch, create=True):
    root = _task_dir(workspace_dir, branch=branch, create=create) / "collections"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def collection_dir(workspace_dir, branch, collection_id, create=False):
    collection_id = _safe_id(collection_id, "collection_id")
    root = collection_root(workspace_dir, branch, create=create).resolve()
    path = (root / collection_id).resolve()
    if path.parent != root:
        raise ValueError("collection path escapes the task")
    if create:
        path.mkdir(parents=False, exist_ok=False)
    return path


def discover_source_dir(payload_root):
    payload_root = Path(payload_root).resolve()
    candidates = sorted({
        p.parent.resolve() for p in payload_root.rglob("labels.csv")
        if (p.parent / "images").is_dir()
    })
    if not candidates:
        raise ValueError("Collection contains no images/ + labels.csv source")
    if len(candidates) != 1:
        raise ValueError(
            f"Collection must contain exactly one dataset source; found {len(candidates)}"
        )
    return candidates[0]


def _read_collection_records(workspace_dir, collection_id, source_dir,
                             image_column=0, steering_column=1):
    workspace = Path(workspace_dir).resolve()
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_relative_to(workspace):
        raise ValueError("Collection source must remain inside the workspace")
    labels = source_dir / "labels.csv"
    images = source_dir / "images"
    if not labels.is_file() or not images.is_dir():
        raise ValueError("Collection source requires labels.csv and images/")
    with labels.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if rows and len(rows[0]) > image_column and rows[0][image_column].strip().lower() in {
        "image_filename", "filename", "image"
    }:
        rows = rows[1:]
    records = []
    for index, row in enumerate(rows):
        if len(row) <= max(image_column, steering_column):
            continue
        name = row[image_column].strip()
        name = name if Path(name).suffix else name + ".png"
        try:
            steering = float(row[steering_column])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(steering):
            continue
        image = (images / name).resolve()
        if not image.is_relative_to(images.resolve()) or not image.is_file():
            continue
        records.append({
            "id": f"{collection_id}:{index}",
            "source": collection_id,
            "image": str(image.relative_to(workspace)),
            "steering": max(-1.0, min(1.0, steering)),
        })
    if not records:
        raise ValueError("Collection contains no valid image/steering records")
    return records


def write_collection_manifest(workspace_dir, branch, collection_id,
                              deployment_run_id, source_dir, *, round_index,
                              controller_id=None, remote_paths=None):
    """Validate and finalize an immutable manifest after payload extraction."""
    collection_id = _safe_id(collection_id, "collection_id")
    deployment_run_id = _safe_id(deployment_run_id, "deployment_run_id")
    directory = collection_dir(workspace_dir, branch, collection_id)
    if not directory.is_dir():
        raise ValueError("Collection directory must exist before its manifest is finalized")
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_relative_to(directory.resolve()):
        raise ValueError("Collection payload must remain inside its task-local collection directory")
    path = directory / MANIFEST
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("deployment_run_id") != deployment_run_id:
            raise ValueError("Collection ID already belongs to a different deployment run")
        return existing
    records = _read_collection_records(workspace_dir, collection_id, source_dir)
    payload = {
        "schema_version": 1,
        "role": "deployment_collection",
        "collection_id": collection_id,
        "deployment_run_id": deployment_run_id,
        "task_id": branch,
        "round": int(round_index),
        "controller_id": controller_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(source_dir.relative_to(Path(workspace_dir).resolve())),
        "remote_paths": remote_paths or {},
        "count": len(records),
        "fingerprint": _dataset_fingerprint(records),
        "records": records,
    }
    _write_json_atomic(path, payload)
    return payload


def load_collection(workspace_dir, branch, collection_id):
    path = collection_dir(workspace_dir, branch, collection_id) / MANIFEST
    if not path.is_file():
        raise ValueError(f"Unknown or incomplete task-local collection: {collection_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("collection_id") != collection_id or payload.get("task_id") != branch:
        raise ValueError(f"Collection manifest identity mismatch: {collection_id}")
    records = payload.get("records")
    if not isinstance(records, list) or payload.get("fingerprint") != _dataset_fingerprint(records):
        raise ValueError(f"Collection manifest is corrupt: {collection_id}")
    return payload
