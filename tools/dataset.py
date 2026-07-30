import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .models import TRANSFORM_144_224
from .io import ROOT

__all__ = [
    "DrivingDataset",
    "_records", "_sources_from_manifest", "_discover_datasets", "_relocate_dataset",
    "_dataset_registry_path", "_load_dataset_registry", "_save_dataset_registry",
    "_dataset_config",
    "_quantiles", "_combined_score", "_threshold",
]


_REGISTRY_FIELDS = (
    "dataset_path", "dataset_id", "dataset_mode", "sources",
    "image_column", "steering_column", "raw_samples", "source_composition",
    "max_per_source", "vlm",
)


def _dataset_registry_path(workspace_dir):
    return Path(workspace_dir) / ROOT / "dataset.json"


def _load_dataset_registry(workspace_dir):
    p = _dataset_registry_path(workspace_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
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
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry, ensure_ascii=False, indent=2))


def _dataset_config(workspace_dir):
    return _load_dataset_registry(workspace_dir)


class DrivingDataset(Dataset):
    def __init__(self, workspace_dir, records):
        self.workspace_dir = Path(workspace_dir)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.workspace_dir / rec["image"]
        img = Image.open(img_path).convert('RGB')
        tensor_img = TRANSFORM_144_224(img)
        steer = torch.tensor([rec["steering"]], dtype=torch.float32)
        return tensor_img, steer, idx


def _records(workspace_dir, branch="main", ignore_subset=False):
    reg = _load_dataset_registry(workspace_dir)
    base = Path(workspace_dir) / reg["dataset_path"]
    
    subset_params = None
    if not ignore_subset and branch:
        try:
            from .io import _load
            st = _load(workspace_dir, branch=branch)
            subset_params = st.get("dataset_subset")
        except Exception:
            subset_params = None

    sources = reg["sources"]
    if subset_params and isinstance(subset_params, dict):
        inc = subset_params.get("include_sources")
        exc = subset_params.get("exclude_sources")
        if inc:
            allow = set(inc)
            sources = [s for s in sources if s["name"] in allow]
        if exc:
            deny = set(exc)
            sources = [s for s in sources if s["name"] not in deny]
        cap_cfg = subset_params.get("max_per_source")
    else:
        cap_cfg = None if ignore_subset else reg.get("max_per_source")

    out = []
    image_column = reg.get("image_column", 0)
    steering_column = reg.get("steering_column", 1)
    if isinstance(cap_cfg, int):
        cap_map = {s["name"]: cap_cfg for s in sources}
    elif isinstance(cap_cfg, dict):
        cap_map = {str(k): int(v) for k, v in cap_cfg.items()}
    else:
        cap_map = {}
    for source in sources:
        source_path = base / source["path"]
        labels = source_path / "labels.csv"
        images = source_path / "images"
        if not labels.exists() or not images.is_dir():
            raise ValueError(f"source '{source['name']}' needs images/ and labels.csv")
        cap = cap_map.get(source["name"])
        taken = 0
        with labels.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] and rows[0][image_column].lower() in {"image_filename", "filename", "image"}:
            rows = rows[1:]
        for i, row in enumerate(rows):
            if len(row) <= max(image_column, steering_column):
                continue
            name = row[image_column].strip()
            name = name if Path(name).suffix else name + ".png"
            try:
                steer = float(row[steering_column])
            except ValueError:
                continue
            if (images / name).is_file():
                out.append({
                    "id": f"{source['name']}:{i}",
                    "source": source["name"],
                    "image": str(Path(reg["dataset_path"]) / source["path"] / "images" / name),
                    "steering": max(-1.0, min(1.0, steer))
                })
                taken += 1
                if cap is not None and taken >= cap:
                    break
    if not out:
        raise ValueError("No valid image_filename,steering rows found")
    return out


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
        return base, data.get("dataset_id"), sources, int(data.get("image_column", 0)), int(data.get("steering_column", 1)), "manifest"
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


def _combined_score(pcc, steer_err, alpha: float = 0.5, eps: float = 1e-9):
    p = np.asarray(pcc, dtype=float)
    e = np.asarray(steer_err, dtype=float)

    def _n(x):
        return (x - x.min()) / (x.max() - x.min() + eps)

    return alpha * _n(1.0 - p) + (1.0 - alpha) * _n(e)


def _threshold(scores, k: float = 1.0, eps: float = 1e-9):
    a = np.asarray(scores, dtype=float)
    return float(a.mean()) + float(k) * float(a.std())
