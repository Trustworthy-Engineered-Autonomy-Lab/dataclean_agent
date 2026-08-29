"""User-directed, pre-execution configuration of a task's frozen D_0 view."""

import json
import copy
import re
import time
from pathlib import Path

from .base import Tool
from .configure_dataset import ConfigureDataset
from .agent_protocol import public_protocol_state, WithdrawExperimentEpisode
from .utils import (
    _anonymize_source_name,
    _dataset_config,
    _load,
    _load_task_spec,
    _load_dataset_snapshot,
    _save,
    _task_dir,
    _write_json_atomic,
)


def _is_paired_task(workspace_dir, branch, state):
    if state.get("paired_with"):
        return True
    tasks_root = _task_dir(workspace_dir, branch, create=False).parent
    for task_dir in tasks_root.iterdir() if tasks_root.is_dir() else []:
        spec_path = task_dir / "task_spec.json"
        if not spec_path.is_file():
            continue
        try:
            if json.loads(spec_path.read_text(encoding="utf-8")).get("paired_with") == branch:
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


class ConfigureTaskDataset(Tool):
    name = "configure_task_dataset"
    description = (
        "Apply an explicit user-requested per-source sampling view to the active task before any "
        "training, scoring, partitioning, episode action, or deployment. Example: source_caps "
        "{'normal': 8000} plus default_cap 500 means normal up to 8000 and every other source "
        "up to 500. This configures D_0; it is not a cleaning operation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_caps": {
                "type": "object",
                "additionalProperties": {"type": "integer", "minimum": 1},
                "description": "Explicit caps keyed by a user-provided real or anonymous source name.",
            },
            "default_cap": {
                "type": "integer",
                "minimum": 1,
                "description": "Cap applied to every source not named in source_caps.",
            },
            "include_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional sources to include; omitted means include all sources.",
            },
            "exclude_sources": {
                "type": "array",
                "items": {"type": "string"},
            },
            "request_basis": {
                "type": "string",
                "description": "Concise reference to the user's explicit dataset-composition request.",
            },
        },
        "required": ["request_basis"],
    }

    def run(self, request_basis, source_caps=None, default_cap=None, include_sources=None,
            exclude_sources=None, branch="main", workspace_dir=None,
            agent_metadata=None, user_request_text=None, **_):
        source_caps = source_caps or {}
        include_sources = include_sources or None
        exclude_sources = exclude_sources or None
        if not isinstance(source_caps, dict):
            raise ValueError("source_caps must be an object")
        if not source_caps and default_cap is None and not include_sources and not exclude_sources:
            raise ValueError("Provide source caps, a default cap, or an include/exclude selection")
        if not isinstance(request_basis, str) or not request_basis.strip():
            raise ValueError("request_basis must cite the explicit user request")
        if agent_metadata is not None:
            # ``user_request_text`` is injected by Turn after model arguments are
            # assembled, so the model cannot forge or replace it. Natural-language
            # intent belongs to the Agent; a second keyword parser here only creates
            # brittle false rejections (for example, “设为” vs “设置为”).
            if not isinstance(user_request_text, str) or not user_request_text.strip():
                raise PermissionError(
                    "Agent dataset configuration requires a bound current user turn"
                )
            request_numbers = {
                int(value.replace(",", ""))
                for value in re.findall(r"\d[\d,]*", user_request_text)
            }
            operation_numbers = {int(value) for value in source_caps.values()}
            if default_cap is not None:
                operation_numbers.add(int(default_cap))
            if operation_numbers and not operation_numbers.issubset(request_numbers):
                raise PermissionError(
                    "Dataset quota values must appear in the bound current user turn"
                )
            selected_names = [str(value).lower() for value in (
                list(include_sources or []) + list(exclude_sources or [])
            )]
            request_lower = user_request_text.lower()
            if selected_names and not all(name in request_lower for name in selected_names):
                raise PermissionError(
                    "Included/excluded source names must appear in the bound current user turn"
                )
            request_basis = user_request_text.strip()
        if default_cap is not None and (
            not isinstance(default_cap, int) or isinstance(default_cap, bool) or default_cap < 1
        ):
            raise ValueError("default_cap must be a positive integer")
        for name, value in source_caps.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"source_caps.{name} must be a positive integer")

        state = _load(workspace_dir, branch=branch)
        if state.get("task_status", "DRAFT") != "DRAFT":
            raise ValueError(
                "D_0 can only be configured before experimental execution begins; "
                "create a new task for a different dataset composition"
            )
        if _is_paired_task(workspace_dir, branch, state):
            raise ValueError(
                "D_0 is frozen by a paired comparison; configure the first arm before pairing "
                "or create a new pair"
            )
        if state.get("execution_mode", "adaptive_agent") != "adaptive_agent":
            raise ValueError(
                "A fixed-baseline task's dataset view is preregistered; create a new task to change it"
            )
        if not ConfigureDataset.task_sampling_is_mutable(state):
            raise ValueError(
                "The current task has already started experimental execution; create a new task "
                "for a different D_0 composition"
            )
        protocol = public_protocol_state(workspace_dir, branch)
        # An unchanged failed attempt can legitimately restore DRAFT. Its
        # audit entry remains, but must not override the state/budget/lineage
        # mutability checks above and permanently block dataset correction.
        if protocol.get("executing_action"):
            raise ValueError(
                "Close the current experiment activity and create a new task before changing D_0"
            )
        active_episode = protocol.get("active_episode")
        if active_episode:
            if int(active_episode.get("actions_used", 0)) != 0:
                raise ValueError(
                    "The active episode already executed an action; create a new task before changing D_0"
                )
            WithdrawExperimentEpisode().run(
                reason=(
                    "Superseded before execution by explicit user dataset configuration: "
                    + request_basis.strip()
                ),
                branch=branch,
                workspace_dir=workspace_dir,
            )

        registry = _dataset_config(workspace_dir)
        sources = registry.get("sources") or []
        normalized_explicit = ConfigureDataset._normalize_subset(
            registry, None, None, source_caps
        )["max_per_source"]
        caps = {}
        for source in sources:
            real_name = source["name"]
            if real_name in normalized_explicit:
                caps[real_name] = normalized_explicit[real_name]
            elif default_cap is not None:
                caps[real_name] = int(default_cap)
        if not caps and not include_sources and not exclude_sources:
            raise ValueError("The requested caps did not select any configured source")

        before = {
            "count": state.get("round_input_count"),
            "fingerprint": state.get("round_input_fingerprint"),
            "dataset_subset": state.get("dataset_subset") or {},
        }
        original_state = copy.deepcopy(state)
        original_spec = _load_task_spec(workspace_dir, branch=branch)
        configured = json.loads(ConfigureDataset().run(
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            max_per_source=caps or None,
            branch=branch,
            workspace_dir=workspace_dir,
        ))
        updated = _load(workspace_dir, branch=branch)
        subset = updated.get("dataset_subset") or {}
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        design_event = {
            "time": now,
            "type": "configure_task_dataset",
            "request_basis": request_basis.strip(),
            "before": before,
            "after": {
                "count": updated.get("round_input_count"),
                "fingerprint": updated.get("round_input_fingerprint"),
                "dataset_subset": subset,
            },
        }
        spec_path = _task_dir(workspace_dir, branch, create=False) / "task_spec.json"
        spec = copy.deepcopy(original_spec)
        spec["dataset_subset"] = subset
        spec.setdefault("pre_execution_design_log", []).append(design_event)
        updated.setdefault("design_log", []).append(design_event)
        try:
            # Publish both metadata documents as one logical design update.  If
            # either atomic file replacement fails, restore both previous
            # documents; the newly created content-addressed snapshot is only an
            # unreferenced artifact and cannot change the experiment.
            _write_json_atomic(spec_path, spec)
            _save(workspace_dir, updated, branch=branch)
        except Exception as exc:
            try:
                _write_json_atomic(spec_path, original_spec)
                _save(workspace_dir, original_state, branch=branch)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Dataset design update failed and metadata rollback also failed: "
                    f"update={exc}; rollback={rollback_exc}"
                ) from exc
            raise

        composition = {}
        snapshot_ref = updated.get("round_input_dataset")
        if not snapshot_ref:
            raise RuntimeError("Configured task has no frozen D_0 snapshot reference")
        snapshot = _load_dataset_snapshot(Path(snapshot_ref))
        for record in snapshot.get("records", []):
            source = _anonymize_source_name(record["source"])
            composition[source] = composition.get(source, 0) + 1
        return json.dumps({
            "configured": True,
            "task_id": branch,
            "round_input_count": updated.get("round_input_count"),
            "round_input_fingerprint": updated.get("round_input_fingerprint"),
            "anonymous_source_composition": composition,
            "task_subset": configured.get("task_subset"),
            "request_basis": request_basis.strip(),
            "message": "Task D_0 was re-frozen before execution; no cleaning or training was run.",
        }, ensure_ascii=False)
