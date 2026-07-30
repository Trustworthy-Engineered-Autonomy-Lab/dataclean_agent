import json
import time
from pathlib import Path
from .base import Tool
from .utils import (_sources_from_manifest, _records, _discover_datasets, _relocate_dataset,
                    _load_dataset_registry, _save_dataset_registry, _dataset_config)

class ConfigureDataset(Tool):
    name = "configure_dataset"
    description = ("Configure workspace-level dataset resource (writes to .dataclean/dataset.json). "
                   "Does not create a task. Supports subset filtering via include_sources/exclude_sources "
                   "and per-source sampling caps via max_per_source. "
                   "Set list_sources=True for read-only discovery of available sources and counts.")
    parameters = {
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Dataset directory (containing images/+labels.csv or manifest.json). Reuses existing if omitted."
            },
            "dataset_id": {
                "type": "string",
                "description": "Optional custom dataset ID."
            },
            "include_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Whitelist of source folders to include. Takes priority if exclude_sources is also provided."
            },
            "exclude_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Blacklist of source folders to exclude."
            },
            "max_per_source": {
                "type": "object",
                "description": "Per-source sampling cap as a JSON object mapping source folder name to max sample count, e.g. {\"normal\": 8000, \"erratic\": 500}."
            },
            "list_sources": {
                "type": "boolean",
                "description": "Read-only mode: returns source list and sample counts for subset planning without modifying registry."
            }
        },
        "required": []
    }

    def run(self, dataset_path=None, dataset_id="default",
            include_sources=None, exclude_sources=None, max_per_source=None,
            list_sources=False, workspace_dir=None, **kwargs):
        existing = None
        try:
            existing = _load_dataset_registry(workspace_dir)
        except Exception:
            existing = None

        if list_sources:
            return json.dumps(self._list_sources_report(workspace_dir, existing), ensure_ascii=False)

        wants_subset = any(v is not None for v in (max_per_source, include_sources, exclude_sources))

        if not dataset_path:
            if existing and existing.get("sources"):
                if wants_subset:
                    _ws = Path(workspace_dir).resolve()
                    stored_path = existing.get("dataset_path")
                    _cand = Path(stored_path) if stored_path else None
                    _dp_abs = (_cand if (_cand and _cand.is_absolute())
                               else (_ws / _cand)) if _cand else None
                    if not (_dp_abs and _dp_abs.exists()):
                        _dp_abs = _relocate_dataset(workspace_dir, existing)
                    if _dp_abs and _dp_abs.exists():
                        dataset_path = str(_dp_abs)
                    else:
                        return json.dumps(self._report(existing, note=(
                            "Reused configured dataset; unable to locate disk path to apply subset parameters."
                            "Pass a valid dataset_path to apply subset parameters.")), ensure_ascii=False)
                else:
                    return json.dumps(self._report(existing, note=(
                        "Reused configured dataset. To rescan or apply subset parameters, "
                        "pass dataset_path or max_per_source/include_sources/exclude_sources."
                        "To inspect current sources, pass list_sources=True.")),
                        ensure_ascii=False)
            else:
                candidates = _discover_datasets(workspace_dir)
                if candidates:
                    return json.dumps({
                        "configured": False,
                        "message": "Dataset not yet configured. Candidates discovered in workspace:",
                        "candidates": candidates,
                        "hint": "Pass list_sources=True to inspect source composition."
                    }, ensure_ascii=False)
                raise ValueError(
                    "No dataset_path provided and no valid dataset directory discovered in workspace "
                    "(must contain images/ and labels.csv, or manifest.json)."
                )

        _ws = Path(workspace_dir).resolve()
        _cand = Path(dataset_path)
        _dp_abs = _cand if _cand.is_absolute() else (_ws / _cand)
        if not _dp_abs.exists():
            candidates = _discover_datasets(workspace_dir)
            return json.dumps({
                "configured": False,
                "error": f"Dataset path does not exist: {_dp_abs} (Workspace: {_ws})",
                "candidates": candidates,
                "message": "Path does not exist. Use one of the candidates below as dataset_path."
            }, ensure_ascii=False)

        base, manifest_id, raw_sources, image_column, steering_column, mode = _sources_from_manifest(workspace_dir, dataset_path)

        reg = {
            "configured": True,
            "dataset_id": manifest_id or dataset_id,
            "dataset_path": str(base.relative_to(_ws)),
            "sources": raw_sources,
            "dataset_mode": mode,
            "image_column": image_column,
            "steering_column": steering_column,
            "vlm": (existing or {}).get("vlm"),
            "configured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_dataset_registry(workspace_dir, reg)

        raw_records = _records(workspace_dir, branch="", ignore_subset=True)
        raw_comp = {s["name"]: sum(r["source"] == s["name"] for r in raw_records) for s in raw_sources}
        reg["raw_samples"] = len(raw_records)
        reg["source_composition"] = raw_comp
        _save_dataset_registry(workspace_dir, reg)

        branch = (kwargs.get("branch") or "").strip()
        if max_per_source is not None:
            norm_cap = {str(k): int(v) for k, v in max_per_source.items()} if isinstance(max_per_source, dict) else int(max_per_source)
        else:
            norm_cap = None

        if branch:
            from .io import _load, _save
            try:
                st = _load(workspace_dir, branch=branch)
            except Exception:
                st = {"round": 0, "deployments": 0}

            st["dataset_subset"] = {
                "include_sources": include_sources,
                "exclude_sources": exclude_sources,
                "max_per_source": norm_cap
            }
            _save(workspace_dir, st, branch=branch)

        records = _records(workspace_dir, branch=branch)
        active_sources = [s["name"] for s in raw_sources]
        if include_sources:
            active_sources = [s for s in active_sources if s in set(include_sources)]
        if exclude_sources:
            active_sources = [s for s in active_sources if s not in set(exclude_sources)]
            
        composition = {name: sum(r["source"] == name for r in records) for name in active_sources}
        
        report_reg = dict(reg)
        report_reg["raw_samples"] = len(records)
        report_reg["source_composition"] = composition
        report_reg["max_per_source"] = norm_cap

        return json.dumps(self._report(report_reg, state="ready"), ensure_ascii=False)

    @staticmethod
    def _report(reg, state=None, note=None):
        out = {
            "configured": True,
            "dataset_id": reg.get("dataset_id"),
            "dataset_mode": reg.get("dataset_mode"),
            "raw_samples": reg.get("raw_samples"),
            "sources": [s.get("name") for s in reg.get("sources", [])],
            "source_composition": reg.get("source_composition"),
            "dataset_path": reg.get("dataset_path"),
        }
        if state:
            out["state"] = state
        if note:
            out["note"] = note
        return out

    @staticmethod
    def _list_sources_report(workspace_dir, existing):
        if existing and existing.get("sources"):
            return {
                "mode": "list_sources",
                "configured": True,
                "dataset_id": existing.get("dataset_id"),
                "dataset_path": existing.get("dataset_path"),
                "sources": [{"name": s.get("name"), "path": s.get("path")}
                            for s in existing.get("sources", [])],
                "source_composition": existing.get("source_composition", {}),
                "max_per_source": existing.get("max_per_source"),
                "note": ("Read-only inspection: source names + counts returned. "
                         "Set include_sources / exclude_sources / max_per_source to apply subset configuration.")
            }
        return {
            "mode": "list_sources",
            "configured": False,
            "candidates": _discover_datasets(workspace_dir),
            "note": ("Dataset not configured yet. Candidate dataset directories listed below. "
                     "Pass one as dataset_path to configure.")
        }
