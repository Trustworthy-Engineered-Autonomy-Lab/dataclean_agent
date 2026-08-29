import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
import numpy as np
from PIL import Image
_ML_IMPORT_ERROR = None
try:
    import torch
    from torch.utils.data import Dataset
    from .models import TRANSFORM_224_224
except Exception as exc:  # Dataset configuration/state tools remain usable without ML extras.
    _ML_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    torch = None
    TRANSFORM_224_224 = None
    class Dataset:  # type: ignore[no-redef]
        pass
from .io import ROOT, _artifact, _write_json_atomic

__all__ = [
    "DrivingDataset", "_raw_records", "_write_dataset_snapshot", "_load_dataset_snapshot",
    "_dataset_fingerprint",
    "_records", "_sources_from_manifest", "_discover_datasets", "_relocate_dataset",
    "_read_sources_records",
    "_dataset_registry_path", "_load_dataset_registry", "_save_dataset_registry",
    "_dataset_config", "_anonymize_source_name", "_deanonymize_source_name",
    "_quantiles",
]


_REGISTRY_FIELDS = (
    "dataset_path", "dataset_id", "dataset_mode", "sources",
    "image_column", "steering_column", "raw_samples", "source_composition",
    "max_per_source", "vlm",
)

_ANON_PREDEFINED = {
    "normal": "src_01",
    "erratic": "src_02",
    "hitwall": "src_03",
    "obstacle": "src_04",
    "plastic": "src_05",
}

def _anonymize_source_name(real_name: str) -> str:
    """One stable, idempotent alias used in records, observations and reports.

    Collection IDs are immutable UUID-backed identities, so their digest stays
    the same across retries/rounds without allocating report-local numbers.
    """
    if not real_name:
        return real_name
    if real_name in _ANON_PREDEFINED:
        return _ANON_PREDEFINED[real_name]
    if re.fullmatch(r"src_(?:[0-9]+|[0-9a-f]{8})", real_name):
        return real_name
    if real_name.startswith("car_log_"):
        real_name = "collect_" + real_name[len("car_log_"):]
    # Hash arbitrary semantic source names so labels such as "failure_case"
    # cannot leak into the unsupervised agent context.
    digest = hashlib.sha256(real_name.encode("utf-8")).hexdigest()[:8]
    return f"src_{digest}"

def _deanonymize_source_name(anon_name: str, sources_list: list = None) -> str:
    if not anon_name:
        return anon_name
    for k, v in _ANON_PREDEFINED.items():
        if v == anon_name:
            return k
    if anon_name.startswith("car_log_"):
        candidate = anon_name.replace("car_log_", "collect_")
        if sources_list:
            known = {s.get("name") for s in sources_list}
            if candidate in known:
                return candidate
        return candidate
    if sources_list:
        for s in sources_list:
            real = s.get("name", "")
            if _anonymize_source_name(real) == anon_name or real == anon_name:
                return real
    return anon_name


def _dataset_registry_path(workspace_dir):
    return Path(workspace_dir) / ROOT / "dataset.json"


def _load_dataset_registry(workspace_dir):
    p = _dataset_registry_path(workspace_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Dataset registry is corrupt: {p}") from exc
    from .io import _state_path
    legacy = _state_path(workspace_dir, branch="main")
    if legacy.exists():
        try:
            st = json.loads(legacy.read_text())
        except Exception:
            st = {}
        if st.get("sources"):
            reg = {k: st.get(k) for k in _REGISTRY_FIELDS}
            reg["configured_at"] = st.get("updated_at", "")
            _save_dataset_registry(workspace_dir, reg)
            return reg
    raise ValueError("Dataset not configured. Call configure_dataset first.")


def _save_dataset_registry(workspace_dir, registry):
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = _dataset_registry_path(workspace_dir)
    _write_json_atomic(p, registry)


def _dataset_config(workspace_dir):
    return _load_dataset_registry(workspace_dir)


class DrivingDataset(Dataset):
    def __init__(self, workspace_dir, records):
        self.workspace_dir = Path(workspace_dir)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if torch is None or TRANSFORM_224_224 is None:
            suffix = f" ({_ML_IMPORT_ERROR})" if _ML_IMPORT_ERROR else ""
            raise RuntimeError(
                "PyTorch and torchvision are required for model training/scoring" + suffix
            )
        rec = self.records[idx]
        img_path = (self.workspace_dir.resolve() / rec["image"]).resolve()
        if not img_path.is_relative_to(self.workspace_dir.resolve()):
            raise ValueError(f"Sample image escapes workspace: {rec['image']}")
        img = Image.open(img_path).convert('RGB')
        tensor_img = TRANSFORM_224_224(img)
        steer = torch.tensor([rec["steering"]], dtype=torch.float32)
        return tensor_img, steer, idx


def _safe_child(base: Path, relative, what: str) -> Path:
    base = base.resolve()
    candidate = (base / str(relative)).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"{what} escapes configured dataset directory: {relative}")
    return candidate


def _read_sources_records(base, reg, sources, cap_map, diagnostics=None):
    out = []
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("skipped_rows", 0)
    diagnostics.setdefault("missing_images", 0)
    diagnostics.setdefault("clipped_steering", 0)
    image_column = reg.get("image_column", 0)
    steering_column = reg.get("steering_column", 1)
    for source in sources:
        source_path = _safe_child(base, source["path"], f"source '{source['name']}' path")
        labels = source_path / "labels.csv"
        images = source_path / "images"
        if not labels.exists() or not images.is_dir():
            continue
        cap = cap_map.get(source["name"])
        taken = 0
        with labels.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if rows and len(rows[0]) > image_column and rows[0][image_column].lower() in {"image_filename", "filename", "image"}:
            rows = rows[1:]
        for i, row in enumerate(rows):
            if len(row) <= max(image_column, steering_column):
                diagnostics["skipped_rows"] += 1
                continue
            name = row[image_column].strip()
            name = name if Path(name).suffix else name + ".png"
            try:
                steer = float(row[steering_column])
            except (TypeError, ValueError):
                diagnostics["skipped_rows"] += 1
                continue
            if not math.isfinite(steer):
                diagnostics["skipped_rows"] += 1
                continue
            try:
                image_path = _safe_child(images, name, "image path")
            except ValueError:
                diagnostics["skipped_rows"] += 1
                continue
            if image_path.is_file():
                safe_name = image_path.relative_to(images.resolve())
                clipped = max(-1.0, min(1.0, steer))
                if clipped != steer:
                    diagnostics["clipped_steering"] += 1
                out.append({
                    "id": f"{source['name']}:{i}",
                    "source": source["name"],
                    "image": str(Path(reg["dataset_path"]) / source["path"] / "images" / safe_name),
                    "steering": clipped,
                })
                taken += 1
                if cap is not None and taken >= cap:
                    break
            else:
                diagnostics["missing_images"] += 1
    return out


def _subset_config(state, reg, ignore_subset=False):
    if ignore_subset:
        return list(reg["sources"]), {}
    subset_params = (state or {}).get("dataset_subset") or {}
    sources = list(reg["sources"])
    inc = subset_params.get("include_sources")
    exc = subset_params.get("exclude_sources")
    if inc:
        allow = set(inc)
        sources = [s for s in sources if s["name"] in allow]
    if exc:
        deny = set(exc)
        sources = [s for s in sources if s["name"] not in deny]
    cap_cfg = subset_params.get("max_per_source")
    if cap_cfg is None:
        cap_cfg = reg.get("max_per_source")
    if isinstance(cap_cfg, int):
        cap_map = {s["name"]: cap_cfg for s in sources}
    elif isinstance(cap_cfg, dict):
        cap_map = {str(k): int(v) for k, v in cap_cfg.items()}
    else:
        cap_map = {}
    return sources, cap_map


def _raw_records(workspace_dir, branch="main", ignore_subset=False, diagnostics=None,
                 subset_override=None):
    reg = _load_dataset_registry(workspace_dir)
    base = _safe_child(Path(workspace_dir).resolve(), reg["dataset_path"], "dataset path")
    st = None
    if subset_override is not None:
        st = {"dataset_subset": subset_override}
    elif not ignore_subset and branch:
        from .io import _load
        st = _load(workspace_dir, branch=branch)
    sources, cap_map = _subset_config(st, reg, ignore_subset=ignore_subset)
    out = _read_sources_records(base, reg, sources, cap_map, diagnostics=diagnostics)
    if not out:
        raise ValueError("No valid image_filename,steering rows found")
    return out


def _load_dataset_snapshot(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset snapshot does not exist: {p}")
    payload = json.loads(p.read_text())
    required = {
        "schema_version", "artifact_id", "task_id", "role", "round",
        "count", "ids", "records", "fingerprint",
    }
    missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
    if missing:
        raise ValueError(f"Dataset snapshot is missing required fields {missing}: {p}")
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError(f"Unsupported dataset snapshot schema: {p}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Dataset snapshot is missing records: {p}")
    try:
        normalized_ids = [str(record["id"]) for record in records]
        for record in records:
            if not all(field in record for field in ("id", "source", "image", "steering")):
                raise ValueError("record is missing a canonical field")
            if not math.isfinite(float(record["steering"])):
                raise ValueError("record has non-finite steering")
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"Dataset snapshot contains invalid records: {p}: {exc}") from exc
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError(f"Dataset snapshot contains duplicate IDs: {p}")
    if int(payload["count"]) != len(records):
        raise ValueError(f"Dataset snapshot count does not match records: {p}")
    if list(payload["ids"]) != normalized_ids:
        raise ValueError(f"Dataset snapshot IDs do not match records: {p}")
    if payload["fingerprint"] != _dataset_fingerprint(records):
        raise ValueError(f"Dataset snapshot fingerprint mismatch: {p}")
    return payload


def _write_dataset_snapshot(workspace_dir, branch, name, records, *, round_index, role, parents=None, metadata=None):
    # Dataset lineage contains only canonical sample fields. Derived scores and
    # reviewer outputs belong in their own artifacts and must not leak into D_t.
    normalized = []
    for record in records:
        steering = float(record["steering"])
        if not math.isfinite(steering):
            raise ValueError(f"Dataset snapshot contains non-finite steering: {record.get('id')}")
        normalized.append({
            "id": str(record["id"]),
            "source": str(record["source"]),
            "image": str(record["image"]),
            "steering": steering,
        })
    records = normalized
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset snapshot contains duplicate sample IDs")
    payload = {
        "schema_version": 2,
        "artifact_id": str(name),
        "task_id": str(branch),
        "role": role,
        "round": int(round_index),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parents": parents or [],
        "metadata": metadata or {},
        "count": len(records),
        "ids": ids,
        "records": records,
    }
    payload["fingerprint"] = _dataset_fingerprint(records)
    path = _artifact(workspace_dir, name, branch=branch)
    _write_json_atomic(path, payload)
    return path, payload


def _records(workspace_dir, branch="main", ignore_subset=False):
    """Return the immutable input snapshot for the active round.

    It never guesses a merge from mutable state fields; round transitions are
    committed explicitly by commit_round.
    """
    if not ignore_subset and branch:
        from .io import _load
        st = _load(workspace_dir, branch=branch)
        ref = st.get("round_input_dataset")
        if ref:
            # A broken declared snapshot is corruption; never silently substitute
            # the current physical registry, which would change D_t mid-run.
            snapshot_path = Path(ref).resolve()
            artifact_dir = _artifact(workspace_dir, "placeholder", branch=branch).parent.resolve()
            if snapshot_path.parent != artifact_dir:
                raise ValueError("Round input snapshot escapes the active task artifact directory")
            payload = _load_dataset_snapshot(snapshot_path)
            if payload.get("task_id") != branch or payload.get("role") != "round_input":
                raise ValueError("Round input snapshot identity/role mismatch")
            if int(payload.get("round", -1)) != int(st.get("round", 0)):
                raise ValueError("Round input snapshot belongs to a different round")
            if payload.get("fingerprint") != st.get("round_input_fingerprint"):
                raise ValueError("Round input state fingerprint does not match its snapshot")
            return payload["records"]
        raise ValueError(
            f"Task '{branch}' has no frozen D_t snapshot; configure its D_0 before execution"
        )
    return _raw_records(workspace_dir, branch=branch, ignore_subset=ignore_subset)


def _dataset_fingerprint(records):
    logical_records = [
        {"id": r["id"], "image": r["image"], "steering": float(r["steering"])}
        for r in records
    ]
    return hashlib.sha256(
        json.dumps(logical_records, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sources_from_manifest(workspace_dir, dataset_path):
    ws = Path(workspace_dir).resolve()
    dp = Path(dataset_path)
    if dp.is_absolute():
        try:
            dp = dp.relative_to(ws)
        except ValueError:
            raise ValueError("dataset path escapes workspace")
    base = (ws / dp).resolve()
    if not base.is_relative_to(ws):
        raise ValueError("dataset path escapes workspace")
    manifest = base / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        sources = data.get("sources", [])
        if not sources or not all(isinstance(x, dict) and x.get("name") and x.get("path") for x in sources):
            raise ValueError("manifest.json needs non-empty sources with name and path")
        names = [str(x["name"]) for x in sources]
        if len(names) != len(set(names)):
            raise ValueError("manifest.json source names must be unique")
        if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
            raise ValueError("manifest source names may contain only letters, numbers, dot, underscore, and hyphen")
        normalized_sources = []
        for source in sources:
            source_dir = _safe_child(base, source["path"], f"source '{source['name']}' path")
            normalized_sources.append({**source, "path": str(source_dir.relative_to(base))})
        image_column = int(data.get("image_column", 0))
        steering_column = int(data.get("steering_column", 1))
        if image_column < 0 or steering_column < 0 or image_column == steering_column:
            raise ValueError("image_column and steering_column must be distinct non-negative indexes")
        return base, data.get("dataset_id"), normalized_sources, image_column, steering_column, "manifest"
    if (base / "images").is_dir() and (base / "labels.csv").is_file():
        return base, None, [{"name": "default", "path": "."}], 0, 1, "single_source"
    sources = [{"name": p.name, "path": p.name} for p in sorted(base.iterdir()) if p.is_dir() and (p / "images").is_dir() and (p / "labels.csv").is_file()]
    if not sources:
        raise ValueError("Dataset needs images/ + labels.csv, or manifest.json listing source folders")
    return base, None, sources, 0, 1, "auto_discovered"


def _discover_datasets(workspace_dir):
    ws = Path(workspace_dir).resolve()
    candidates = []
    for child in sorted(ws.iterdir()):
        if not child.is_dir():
            continue
        if child.name == ROOT or child.name.startswith("."):
            continue
        try:
            _sources_from_manifest(workspace_dir, child.name)
            candidates.append(child.name)
        except Exception:
            continue
    return candidates


def _relocate_dataset(workspace_dir, existing):
    cands = _discover_datasets(workspace_dir)
    if not cands:
        return None
    known = {s.get("name") for s in existing.get("sources", []) if s.get("name")}
    for c in cands:
        cp = Path(workspace_dir).resolve() / c
        try:
            subs = {p.name for p in cp.iterdir()
                    if (p / "images").is_dir() and (p / "labels.csv").exists()}
        except Exception:
            continue
        if known and known.issubset(subs):
            return cp
    if len(cands) == 1:
        return Path(workspace_dir).resolve() / cands[0]
    return None


def _quantiles(values):
    a = np.asarray(values, dtype=float)
    return {
        "min": round(float(a.min()), 5),
        "p05": round(float(np.quantile(a, .05)), 5),
        "p25": round(float(np.quantile(a, .25)), 5),
        "median": round(float(np.median(a)), 5),
        "p75": round(float(np.quantile(a, .75)), 5),
        "p95": round(float(np.quantile(a, .95)), 5),
        "max": round(float(a.max()), 5),
        "mean": round(float(a.mean()), 5),
        "std": round(float(a.std()), 5)
    }
