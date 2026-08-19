import json
import time
from pathlib import Path

from .base import Tool
from .utils import (
    _anonymize_source_name,
    _deanonymize_source_name,
    _dataset_registry_path,
    _dataset_fingerprint,
    _discover_datasets,
    _load,
    _load_dataset_registry,
    _raw_records,
    _read_sources_records,
    _save,
    _save_dataset_registry,
    _sources_from_manifest,
    _write_dataset_snapshot,
)


class ConfigureDataset(Tool):
    name = "configure_dataset"
    agent_exposed = False
    description = (
        "Register and validate the workspace's physical dataset without creating or resetting a task. "
        "When called inside a task, include/exclude/max_per_source are stored only as that task's "
        "round-0 sampling view. Use list_sources=True for read-only discovery."
    )
    parameters = {
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Workspace-relative dataset directory containing images/labels.csv or manifest.json.",
            },
            "dataset_id": {"type": "string", "description": "Optional stable dataset ID."},
            "include_sources": {"type": "array", "items": {"type": "string"}},
            "exclude_sources": {"type": "array", "items": {"type": "string"}},
            "max_per_source": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}},
                ],
                "description": "Task-scoped sampling cap; never changes the physical workspace registry.",
            },
            "list_sources": {"type": "boolean"},
        },
        "required": [],
    }

    def run(
        self,
        dataset_path=None,
        dataset_id="default",
        include_sources=None,
        exclude_sources=None,
        max_per_source=None,
        list_sources=False,
        branch=None,
        workspace_dir=None,
        **_,
    ):
        existing = self._existing(workspace_dir)
        if list_sources:
            return json.dumps(self._list_sources(workspace_dir, existing), ensure_ascii=False)

        if dataset_path:
            registry = self._register_physical_dataset(
                workspace_dir, dataset_path, dataset_id, existing
            )
        elif existing:
            registry = existing
        else:
            candidates = _discover_datasets(workspace_dir)
            if candidates:
                return json.dumps(
                    {
                        "configured": False,
                        "candidates": candidates,
                        "message": "Dataset candidates discovered; choose one as dataset_path.",
                    },
                    ensure_ascii=False,
                )
            raise ValueError("No configured dataset and no dataset_path was provided")

        if branch and all(v is None for v in (include_sources, exclude_sources, max_per_source)):
            try:
                declared = _load(workspace_dir, branch=branch).get("dataset_subset") or {}
                include_sources = declared.get("include_sources")
                exclude_sources = declared.get("exclude_sources")
                max_per_source = declared.get("max_per_source")
            except ValueError:
                pass
        subset = self._normalize_subset(
            registry, include_sources, exclude_sources, max_per_source
        )
        state = None
        if branch:
            state = _load(workspace_dir, branch=branch)
            if not self.task_sampling_is_mutable(state):
                raise ValueError(
                    "Task sampling cannot change after experiment execution has started; "
                    "create a new task for a different dataset view."
                )

        diagnostics = {}
        records = _raw_records(
            workspace_dir,
            branch=branch or "",
            ignore_subset=not bool(branch),
            diagnostics=diagnostics,
            subset_override=subset if branch else None,
        )
        if branch:
            # Never overwrite the snapshot referenced by the old state.  The
            # state pointer is updated only after this content-addressed file is
            # durable, so an interrupted save leaves the previous D_0 intact.
            snapshot_name = f"input_r0_{_dataset_fingerprint(records)[:16]}.json"
            input_path, payload = _write_dataset_snapshot(
                workspace_dir,
                branch,
                snapshot_name,
                records,
                round_index=0,
                role="round_input",
                parents=["workspace_dataset"],
                metadata={"task_subset": subset},
            )
            # State is written only after the requested subset has been fully
            # normalized, materialized, and fingerprinted. A bad request cannot
            # leave dataset_subset changed while D_0 still points at old data.
            state["dataset_subset"] = subset
            state["round_input_dataset"] = str(input_path)
            state["round_input_count"] = len(records)
            state["round_input_fingerprint"] = payload["fingerprint"]
            _save(workspace_dir, state, branch=branch)
        return json.dumps(
            {
                "configured": True,
                "dataset_id": registry.get("dataset_id"),
                "dataset_mode": registry.get("dataset_mode"),
                "dataset_path": registry.get("dataset_path"),
                "physical_samples": registry.get("raw_samples"),
                "active_samples": len(records),
                "sources": [_anonymize_source_name(s["name"]) for s in registry["sources"]],
                "task_subset": self._anonymous_subset(subset) if branch else None,
                "ingestion_diagnostics": diagnostics,
                "state": "ready",
            },
            ensure_ascii=False,
        )

    @staticmethod
    def task_sampling_is_mutable(state):
        """A task's D_0 may be replaced only before any experimental execution.

        define_task eagerly freezes the full registry as D_0 so the task is always
        well formed. That initial snapshot is a default, not evidence that the
        experiment has started.
        """
        if int(state.get("round", 0)) != 0 or state.get("round_status", "ready") != "ready":
            return False
        if state.get("decision_trace") or state.get("round_ledger") or state.get("round_history"):
            return False
        if state.get("latest_scores") or state.get("latest_partition"):
            return False
        if state.get("active_detector") or state.get("active_controller") or state.get("active_clean_dataset"):
            return False
        if int(state.get("detector_train_epochs_used", 0)) or int(state.get("controller_train_epochs_used", 0)):
            return False
        if int(state.get("vlm_calls_total", 0)) or int(state.get("deployments", 0)):
            return False
        return True

    @staticmethod
    def _existing(workspace_dir):
        try:
            return _load_dataset_registry(workspace_dir)
        except ValueError:
            if _dataset_registry_path(workspace_dir).exists():
                raise
            return None

    @staticmethod
    def _register_physical_dataset(workspace_dir, dataset_path, dataset_id, existing):
        ws = Path(workspace_dir).resolve()
        base, manifest_id, sources, image_col, steering_col, mode = _sources_from_manifest(
            workspace_dir, dataset_path
        )
        registry = {
            "schema_version": 2,
            "configured": True,
            "dataset_id": manifest_id or dataset_id,
            "dataset_path": str(base.relative_to(ws)),
            "sources": sources,
            "dataset_mode": mode,
            "image_column": image_col,
            "steering_column": steering_col,
            "vlm": (existing or {}).get("vlm"),
            "configured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        diagnostics = {}
        # Validate entirely in memory and publish the registry once. A process
        # interruption can therefore leave only the old complete registry, not
        # a half-configured dataset.json without counts/diagnostics.
        records = _read_sources_records(
            base, registry, sources, {}, diagnostics=diagnostics
        )
        if not records:
            raise ValueError("No valid image_filename,steering rows found")
        composition = {
            s["name"]: sum(r["source"] == s["name"] for r in records) for s in sources
        }
        registry.update(
            {
                "raw_samples": len(records),
                "source_composition": composition,
                "ingestion_diagnostics": diagnostics,
            }
        )
        _save_dataset_registry(workspace_dir, registry)
        return registry

    @staticmethod
    def _normalize_subset(registry, include_sources, exclude_sources, max_per_source):
        sources = registry.get("sources") or []
        known = {s["name"] for s in sources}

        def names(values):
            if values is None:
                return None
            normalized = [_deanonymize_source_name(str(v), sources) for v in values]
            unknown = sorted(set(normalized) - known)
            if unknown:
                raise ValueError(f"Unknown dataset sources: {unknown}")
            return normalized

        inc = names(include_sources)
        exc = names(exclude_sources)
        if inc and exc and set(inc) & set(exc):
            raise ValueError("A source cannot be both included and excluded")

        caps = max_per_source
        if isinstance(caps, dict):
            if any(not isinstance(v, int) or isinstance(v, bool) for v in caps.values()):
                raise ValueError("max_per_source values must be integers")
            caps = {_deanonymize_source_name(str(k), sources): v for k, v in caps.items()}
            unknown = sorted(set(caps) - known)
            if unknown:
                raise ValueError(f"Unknown max_per_source keys: {unknown}")
            if any(v < 1 for v in caps.values()):
                raise ValueError("max_per_source values must be positive")
        elif caps is not None:
            if not isinstance(caps, int) or isinstance(caps, bool):
                raise ValueError("max_per_source must be an integer")
            if caps < 1:
                raise ValueError("max_per_source must be positive")

        return {"include_sources": inc, "exclude_sources": exc, "max_per_source": caps}

    @staticmethod
    def _anonymous_subset(subset):
        out = dict(subset)
        for field in ("include_sources", "exclude_sources"):
            if out.get(field):
                out[field] = [_anonymize_source_name(x) for x in out[field]]
        if isinstance(out.get("max_per_source"), dict):
            out["max_per_source"] = {
                _anonymize_source_name(k): v for k, v in out["max_per_source"].items()
            }
        return out

    @staticmethod
    def _list_sources(workspace_dir, existing):
        if not existing:
            return {
                "mode": "list_sources",
                "configured": False,
                "candidates": _discover_datasets(workspace_dir),
            }
        composition = existing.get("source_composition") or {}
        return {
            "mode": "list_sources",
            "configured": True,
            "dataset_id": existing.get("dataset_id"),
            "dataset_path": existing.get("dataset_path"),
            "raw_samples": existing.get("raw_samples"),
            "sources": [
                {
                    "name": _anonymize_source_name(s["name"]),
                    "count": composition.get(s["name"], 0),
                }
                for s in existing.get("sources", [])
            ],
            "ingestion_diagnostics": existing.get("ingestion_diagnostics") or {},
        }
