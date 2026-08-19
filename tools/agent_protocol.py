"""Episode-based policy protocol for the adaptive experiment Agent.

Pipeline tools enforce legal state transitions, budgets, and safety. An experiment
episode records a local hypothesis and grants a bounded action scope in which the
Agent may gather evidence and adapt freely. This avoids turning every tool call
into a rigid mini-workflow while retaining a hidden-label-safe audit trail.
"""

import hashlib
import json
import os
import time
import uuid

from .base import Tool
from .io import _load, _save, _task_dir, _write_json_atomic


PROTOCOL_SCHEMA_VERSION = 2
PROTOCOL_STATE_FILE = "agent_protocol.json"
TRAJECTORY_FILE = "agent_trajectory.jsonl"

# Observation-only calls remain freely callable. These actions can change
# experiment state or spend a material resource budget.
CONSEQUENTIAL_TOOLS = {
    "train_detector": "detector",
    "score_and_fit": "score",
    "partition": "partition",
    "resolve": "resolve",
    "train_controller": "controller",
    "deploy_controller": "deployment",
    "eval_controller": "evaluation",
    "transfer_eval_results": "collection",
    "commit_round": "transition",
    "assess_stopping": "stopping",
}

MILESTONE_TOOLS = {"commit_round", "deploy_controller"}
_CONTEXT_ARGUMENTS = {"branch", "workspace_dir", "cancel_event"}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(payload):
    raw = payload if isinstance(payload, str) else _canonical(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _protocol_path(workspace_dir, branch):
    return _task_dir(workspace_dir, branch=branch, create=False) / PROTOCOL_STATE_FILE


def _default_protocol_state():
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "active_episode": None,
        "executing_action": None,
        "last_action": None,
        "assessment_required": False,
        "assessment_reasons": [],
    }


def _migrate_v1(payload):
    """Best-effort migration for a v1 proposal surviving a process restart."""
    state = _default_protocol_state()
    pending = payload.get("pending_proposal") or {}
    if pending:
        episode_id = "episode_legacy_" + str(
            pending.get("proposal_id", uuid.uuid4().hex[:8])
        )
        target = pending.get("target_tool")
        state["active_episode"] = {
            "episode_id": episode_id,
            "created_at": pending.get("created_at", _now()),
            "status": "active",
            "round_started": pending.get("round", 0),
            "title": "Migrated single-action proposal",
            "hypothesis": pending.get("diagnosis", "Legacy proposal"),
            "objective": pending.get("expected_observation", "Complete legacy action safely"),
            "evidence": pending.get("evidence", []),
            "permitted_actions": [target] if target in CONSEQUENTIAL_TOOLS else [],
            "max_action_calls": 1,
            "actions_used": 0,
            "actions": [],
            "decision_variables": {},
            "budget_plan": {},
            "success_signals": [pending.get("expected_observation", "Legacy expectation")],
            "replan_triggers": [pending.get("fallback", "Unexpected result")],
            "considered_alternatives": [],
            "rationale": pending.get("rationale", "Migrated from protocol v1"),
            "observation_hash": pending.get("observation_hash"),
            "observation_artifact": pending.get("observation_artifact"),
            "agent_metadata": pending.get("agent_metadata") or {},
        }
        if pending.get("status") == "executing":
            state["executing_action"] = {
                "action_id": "action_legacy_" + uuid.uuid4().hex[:8],
                "episode_id": episode_id,
                "target_tool": target,
                "arguments": (pending.get("selected_action") or {}).get("arguments") or {},
                "started_at": pending.get("authorized_at", _now()),
                "agent_metadata": pending.get("agent_metadata") or {},
            }
    state["last_action"] = payload.get("last_action")
    state["assessment_required"] = bool(payload.get("requires_outcome_assessment"))
    if state["assessment_required"]:
        state["assessment_reasons"] = ["migrated_v1_outcome_pending"]
        if not state.get("active_episode") and state.get("last_action"):
            legacy_action = dict(state["last_action"])
            episode_id = "episode_legacy_" + str(
                legacy_action.get("proposal_id", uuid.uuid4().hex[:8])
            )
            legacy_action["episode_id"] = episode_id
            legacy_action.setdefault("action_id", "action_legacy_" + uuid.uuid4().hex[:8])
            state["last_action"] = legacy_action
            state["active_episode"] = {
                "episode_id": episode_id,
                "created_at": legacy_action.get("completed_at", _now()),
                "status": "active",
                "round_started": 0,
                "title": "Migrated completed v1 action",
                "hypothesis": "Assess the observable outcome of the migrated action.",
                "objective": legacy_action.get("expected_observation", "Close legacy decision"),
                "evidence": [],
                "permitted_actions": [legacy_action.get("target_tool")],
                "max_action_calls": 1,
                "actions_used": 1,
                "actions": [{
                    "action_id": legacy_action["action_id"],
                    "target_tool": legacy_action.get("target_tool"),
                    "status": legacy_action.get("status"),
                    "result_hash": legacy_action.get("result_hash"),
                }],
                "decision_variables": {},
                "budget_plan": {},
                "success_signals": [legacy_action.get("expected_observation", "Legacy expectation")],
                "replan_triggers": ["Legacy result differs from expectation"],
                "considered_alternatives": [],
                "rationale": "Migrated from protocol v1",
                "observation_hash": None,
                "observation_artifact": None,
                "agent_metadata": {},
            }
    return state


def _load_protocol(workspace_dir, branch):
    path = _protocol_path(workspace_dir, branch)
    if not path.exists():
        return _default_protocol_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Agent protocol state is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Agent protocol state must be a JSON object")
    if int(payload.get("schema_version", 1)) < PROTOCOL_SCHEMA_VERSION:
        return _migrate_v1(payload)
    return {**_default_protocol_state(), **payload}


def _save_protocol(workspace_dir, branch, payload):
    payload["schema_version"] = PROTOCOL_SCHEMA_VERSION
    payload["updated_at"] = _now()
    _write_json_atomic(_protocol_path(workspace_dir, branch), payload)


def _append_event(workspace_dir, branch, event):
    path = _task_dir(workspace_dir, branch=branch, create=False) / TRAJECTORY_FILE
    row = {"schema_version": PROTOCOL_SCHEMA_VERSION, "time": _now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _visible_observation(workspace_dir, branch):
    from .pipeline_state import PipelineState
    return json.loads(PipelineState().run(branch=branch, workspace_dir=workspace_dir))


def _snapshot_observation(workspace_dir, branch, observation):
    observation_hash = _digest(observation)
    task_dir = _task_dir(workspace_dir, branch=branch, create=False)
    path = task_dir / "observations" / "agent_visible" / f"{observation_hash}.json"
    if not path.exists():
        _write_json_atomic(path, observation)
    return observation_hash, str(path.relative_to(task_dir))


def _snapshot_action_result(workspace_dir, branch, result_text):
    result_hash = _digest(result_text)
    task_dir = _task_dir(workspace_dir, branch=branch, create=False)
    path = task_dir / "observations" / "action_results" / f"{result_hash}.json"
    if not path.exists():
        try:
            payload = {"result": json.loads(result_text)}
        except Exception:
            payload = {"result": result_text}
        _write_json_atomic(path, payload)
    return result_hash, str(path.relative_to(task_dir))


def _execution_fingerprint(workspace_dir, branch, state):
    """Detect whether a failed action changed durable experiment state/artifacts."""
    marker_fields = (
        "round", "round_status", "active_detector", "active_controller",
        "active_clean_dataset", "latest_scores", "latest_partition",
        "detector_train_epochs_used", "controller_train_epochs_used",
        "vlm_calls_total", "deployments", "pending_collection_ids",
        "termination_required",
    )
    artifact_dir = _task_dir(workspace_dir, branch=branch, create=False) / "artifacts"
    artifacts = []
    if artifact_dir.is_dir():
        artifacts = sorted(
            (path.name, path.stat().st_size)
            for path in artifact_dir.iterdir() if path.is_file()
        )
    return _digest({
        "state": {field: state.get(field) for field in marker_fields},
        "decision_count": len(state.get("decision_trace") or []),
        "round_history_count": len(state.get("round_history") or []),
        "artifacts": artifacts,
    })


def _nonempty_strings(value, field, minimum=1, maximum=None):
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} item(s)")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} items")
    cleaned = [str(item).strip() for item in value]
    if not all(cleaned):
        raise ValueError(f"{field} items must be non-empty")
    return cleaned


def action_requires_episode(tool_name, args=None):
    if tool_name not in CONSEQUENTIAL_TOOLS:
        return False
    if tool_name == "partition" and (args or {}).get("threshold") is None:
        return False
    return True


# Compatibility for integrations using the earlier helper name.
action_requires_proposal = action_requires_episode


def _recent_episode_assessments(workspace_dir, branch, limit=5):
    path = _task_dir(workspace_dir, branch=branch, create=False) / TRAJECTORY_FILE
    if not path.exists():
        return []
    recent = []
    try:
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            event = json.loads(line)
            if event.get("event") != "episode_assessed":
                continue
            recent.append({
                "episode_id": event.get("episode_id"),
                "expectation_status": event.get("expectation_status"),
                "outcome_summary": event.get("outcome_summary"),
                "belief_update": event.get("belief_update"),
                "next_step": event.get("next_step"),
                "post_episode_observation_hash": event.get("post_episode_observation_hash"),
            })
            if len(recent) >= limit:
                break
        recent.reverse()
    except Exception:
        return [{"error": "recent episode trajectory is unreadable"}]
    return recent


def public_protocol_state(workspace_dir, branch):
    state = _load_protocol(workspace_dir, branch)
    episode = state.get("active_episode") or {}
    executing = state.get("executing_action") or {}
    last_action = state.get("last_action") or {}
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "active_episode": {
            "episode_id": episode.get("episode_id"),
            "title": episode.get("title"),
            "hypothesis": episode.get("hypothesis"),
            "objective": episode.get("objective"),
            "permitted_actions": episode.get("permitted_actions"),
            "actions_used": episode.get("actions_used", 0),
            "max_action_calls": episode.get("max_action_calls"),
            "success_signals": episode.get("success_signals"),
            "replan_triggers": episode.get("replan_triggers"),
        } if episode else None,
        "executing_action": {
            "action_id": executing.get("action_id"),
            "target_tool": executing.get("target_tool"),
        } if executing else None,
        "assessment_required": bool(state.get("assessment_required")),
        "assessment_reasons": state.get("assessment_reasons") or [],
        "last_action": {
            "action_id": last_action.get("action_id"),
            "episode_id": last_action.get("episode_id"),
            "target_tool": last_action.get("target_tool"),
            "status": last_action.get("status"),
        } if last_action else None,
        "recent_episode_assessments": _recent_episode_assessments(workspace_dir, branch),
    }


def authorize_action_call(workspace_dir, branch, tool_name, args, agent_metadata=None):
    """Track one adaptive state-changing call.

    Episodes are optional research plans, not an execution permission ritual.  If
    an episode is active its scope and allowance are enforced.  Otherwise the
    action is recorded as a standalone observation-conditioned decision and the
    domain tool remains responsible for lineage, budget, and safety constraints.
    """
    if not action_requires_episode(tool_name, args):
        return {"episode_required": False, "episode_id": None, "action_id": None}
    pipeline_state = _load(workspace_dir, branch=branch)
    if pipeline_state.get("execution_mode", "adaptive_agent") != "adaptive_agent":
        return {"episode_required": False, "episode_id": None, "action_id": None}
    if pipeline_state.get("task_status") == "COMPLETED":
        raise ValueError("Completed tasks are read-only")
    if not pipeline_state.get("round_input_dataset") or not pipeline_state.get(
        "round_input_fingerprint"
    ):
        raise ValueError("The task has no frozen D_t; configure D_0 before experimental actions")

    protocol = _load_protocol(workspace_dir, branch)
    if protocol.get("executing_action"):
        raise ValueError("Reconcile the earlier executing action before another action")
    if protocol.get("assessment_required") and protocol.get("active_episode"):
        reasons = ", ".join(protocol.get("assessment_reasons") or ["episode milestone"])
        raise ValueError(
            f"Assess the active experiment episode before another consequential action ({reasons})"
        )
    episode = protocol.get("active_episode")
    current_round = int(pipeline_state.get("round", 0))
    if episode:
        if episode.get("status") != "active":
            raise ValueError(f"Episode {episode.get('episode_id')} is not active")
        if tool_name not in (episode.get("permitted_actions") or []):
            raise ValueError(
                f"Action '{tool_name}' is outside episode scope {episode.get('permitted_actions')}"
            )
        if int(episode.get("actions_used", 0)) >= int(episode.get("max_action_calls", 0)):
            raise ValueError("Episode action allowance is exhausted; assess or replan the episode")
        if int(episode.get("round_started", current_round)) != current_round:
            raise ValueError("Episode belongs to an earlier round and must be assessed")

    # The first authorized experimental action atomically freezes the task
    # design. Read-only observations and episode proposals do not lock D_0.
    if pipeline_state.get("task_status", "DRAFT") == "DRAFT":
        pipeline_state["task_status"] = "LOCKED"
        pipeline_state["dataset_locked_at"] = _now()
        _save(workspace_dir, pipeline_state, branch=branch)

    episode_metadata = (episode or {}).get("agent_metadata") or {}
    proposal_turn = episode_metadata.get("turn_id")
    action_turn = (agent_metadata or {}).get("turn_id")
    proposal_step = episode_metadata.get("model_step")
    action_step = (agent_metadata or {}).get("model_step")
    if (
        proposal_turn == action_turn
        and proposal_step is not None
        and action_step is not None
        and int(action_step) <= int(proposal_step)
    ):
        raise ValueError("The first episode action must follow episode validation")

    last = protocol.get("last_action") or {}
    last_meta = last.get("agent_metadata") or {}
    if (
        last_meta.get("turn_id") == action_turn
        and last_meta.get("model_step") is not None
        and action_step is not None
        and int(last_meta["model_step"]) == int(action_step)
    ):
        raise ValueError(
            "Only one consequential action is allowed per model step so its result can be observed"
        )

    action_id = "action_" + uuid.uuid4().hex[:16]
    call_args = {k: v for k, v in (args or {}).items() if k not in _CONTEXT_ARGUMENTS}
    observation = _visible_observation(workspace_dir, branch)
    observation_hash, observation_artifact = _snapshot_observation(
        workspace_dir, branch, observation
    )
    executing = {
        "action_id": action_id,
        "episode_id": episode.get("episode_id") if episode else None,
        "execution_style": "episode" if episode else "standalone",
        "target_tool": tool_name,
        "decision_type": CONSEQUENTIAL_TOOLS[tool_name],
        "arguments": call_args,
        "started_at": _now(),
        "round_started": current_round,
        "observation_hash": observation_hash,
        "observation_artifact": observation_artifact,
        "pre_execution_fingerprint": _execution_fingerprint(
            workspace_dir, branch, pipeline_state
        ),
        "agent_metadata": agent_metadata or {},
    }
    protocol["executing_action"] = executing
    _save_protocol(workspace_dir, branch, protocol)
    _append_event(workspace_dir, branch, {"event": "episode_action_started", **executing})
    return {
        "tracked": True,
        "episode_required": bool(episode),
        "episode_id": episode.get("episode_id") if episode else None,
        "action_id": action_id,
    }


def finish_action_call(workspace_dir, branch, authorization, tool_name, result):
    if not authorization or not authorization.get("tracked"):
        return
    protocol = _load_protocol(workspace_dir, branch)
    executing = protocol.get("executing_action") or {}
    if executing.get("action_id") != authorization.get("action_id"):
        raise ValueError("Executing Agent action changed before outcome recording")

    result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    status = "completed"
    parsed = None
    try:
        parsed = json.loads(result_text)
        if isinstance(parsed, dict) and (
            parsed.get("error")
            or parsed.get("cancelled")
            or parsed.get("protocol_rejected")
            or parsed.get("ok") is False
            or str(parsed.get("status", "")).lower() in {
                "failed", "error", "cancelled", "rejected"
            }
        ):
            status = "failed"
    except Exception:
        pass
    domain_result = parsed.get("data") if isinstance(parsed, dict) and isinstance(
        parsed.get("data"), dict
    ) else parsed
    result_hash, result_artifact = _snapshot_action_result(
        workspace_dir, branch, result_text
    )
    summary = result_text if len(result_text) <= 2000 else result_text[:2000] + "... [truncated]"
    last_action = {
        **executing,
        "status": status,
        "completed_at": _now(),
        "result_hash": result_hash,
        "result_artifact": result_artifact,
        "result_summary": summary,
    }
    episode = protocol.get("active_episode")
    if episode:
        episode["actions_used"] = int(episode.get("actions_used", 0)) + 1
        episode.setdefault("actions", []).append({
            "action_id": last_action["action_id"],
            "target_tool": tool_name,
            "status": status,
            "result_hash": result_hash,
        })
        episode["actions"] = episode["actions"][-20:]

    reasons = []
    pipeline_state = _load(workspace_dir, branch=branch)
    if status == "failed":
        if episode:
            reasons.append("action_failed")
        # A schema/precondition failure that changed nothing must not make D_0
        # permanently immutable. Failures that consumed budget or wrote an
        # artifact remain LOCKED and require a new task/design decision.
        unchanged = executing.get("pre_execution_fingerprint") == _execution_fingerprint(
            workspace_dir, branch, pipeline_state
        )
        if unchanged and pipeline_state.get("task_status") == "LOCKED":
            pipeline_state["task_status"] = "DRAFT"
            pipeline_state.pop("dataset_locked_at", None)
            _save(workspace_dir, pipeline_state, branch=branch)
    else:
        if pipeline_state.get("task_status") == "LOCKED":
            pipeline_state["task_status"] = "RUNNING"
            _save(workspace_dir, pipeline_state, branch=branch)
    if episode and tool_name in MILESTONE_TOOLS:
        reasons.append(f"milestone:{tool_name}")
    if episode and tool_name == "assess_stopping" and isinstance(domain_result, dict) and domain_result.get("stop"):
        reasons.append("milestone:stop_decision")
    if episode and episode["actions_used"] >= int(episode.get("max_action_calls", 0)):
        reasons.append("episode_action_allowance_exhausted")
    try:
        new_round = int(_load(workspace_dir, branch=branch).get("round", 0))
        if episode and new_round != int(episode.get("round_started", new_round)):
            reasons.append("round_changed")
    except Exception:
        reasons.append("pipeline_state_unreadable")

    protocol["active_episode"] = episode
    protocol["executing_action"] = None
    protocol["last_action"] = last_action
    protocol["assessment_required"] = bool(reasons)
    protocol["assessment_reasons"] = reasons
    _save_protocol(workspace_dir, branch, protocol)
    _append_event(workspace_dir, branch, {
        "event": "episode_action_completed",
        **last_action,
        "episode_actions_used": episode.get("actions_used") if episode else None,
        "assessment_required": bool(reasons),
        "assessment_reasons": reasons,
    })


def record_turn_completed(workspace_dir, branch, agent_metadata, usage, model_steps, tool_calls, stopped):
    if not workspace_dir or not branch:
        return
    try:
        _append_event(workspace_dir, branch, {
            "event": "agent_turn_completed",
            "agent_metadata": agent_metadata or {},
            "usage": usage or {},
            "model_steps": int(model_steps),
            "tool_calls": int(tool_calls),
            "stopped": bool(stopped),
        })
    except Exception:
        pass


class ProposeExperimentEpisode(Tool):
    name = "propose_experiment_episode"
    description = (
        "Open a bounded adaptive experiment episode with a local hypothesis, evidence, action "
        "scope, budget plan, success signals, and replan triggers. Multiple consequential actions "
        "may run inside the validated scope without per-action proposals."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "hypothesis": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "evidence": {
                "type": "array", "minItems": 1, "maxItems": 12,
                "items": {"type": "string", "minLength": 1},
            },
            "permitted_actions": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {"type": "string", "enum": sorted(CONSEQUENTIAL_TOOLS)},
            },
            "max_action_calls": {"type": "integer", "minimum": 1, "maximum": 20},
            "decision_variables": {"type": "object"},
            "budget_plan": {"type": "object"},
            "success_signals": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {"type": "string", "minLength": 1},
            },
            "replan_triggers": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {"type": "string", "minLength": 1},
            },
            "considered_alternatives": {
                "type": "array", "maxItems": 5,
                "items": {"type": "string", "minLength": 1},
            },
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": [
            "title", "hypothesis", "objective", "evidence", "permitted_actions",
            "max_action_calls", "decision_variables", "budget_plan", "success_signals",
            "replan_triggers", "rationale",
        ],
    }

    def run(self, title, hypothesis, objective, evidence, permitted_actions,
            max_action_calls, decision_variables, budget_plan, success_signals,
            replan_triggers, rationale, considered_alternatives=None, branch="main",
            workspace_dir=None, agent_metadata=None, **_):
        pipeline_state = _load(workspace_dir, branch=branch)
        if pipeline_state.get("execution_mode", "adaptive_agent") != "adaptive_agent":
            raise ValueError("Fixed baselines use preregistered actions, not adaptive episodes")
        for field, value in (
            ("title", title), ("hypothesis", hypothesis), ("objective", objective),
            ("rationale", rationale),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        evidence = _nonempty_strings(evidence, "evidence", 1, 12)
        success_signals = _nonempty_strings(success_signals, "success_signals", 1, 8)
        replan_triggers = _nonempty_strings(replan_triggers, "replan_triggers", 1, 8)
        alternatives = _nonempty_strings(
            considered_alternatives or [], "considered_alternatives", 0, 5
        )
        if not isinstance(permitted_actions, list) or not 1 <= len(permitted_actions) <= 8:
            raise ValueError("permitted_actions must contain between one and eight actions")
        permitted_actions = list(dict.fromkeys(permitted_actions))
        unknown = [name for name in permitted_actions if name not in CONSEQUENTIAL_TOOLS]
        if unknown:
            raise ValueError(f"Unknown consequential actions: {unknown}")
        unavailable = []
        for name in permitted_actions:
            try:
                Tool.get(name)
            except KeyError:
                unavailable.append(name)
        if unavailable:
            raise ValueError(f"Episode action tools unavailable in this installation: {unavailable}")
        if not isinstance(max_action_calls, int) or isinstance(max_action_calls, bool) or not 1 <= max_action_calls <= 20:
            raise ValueError("max_action_calls must be an integer in [1, 20]")
        if not isinstance(decision_variables, dict) or not isinstance(budget_plan, dict):
            raise ValueError("decision_variables and budget_plan must be objects")

        protocol = _load_protocol(workspace_dir, branch)
        if protocol.get("executing_action"):
            raise ValueError("Reconcile the executing action before opening an episode")
        if protocol.get("active_episode"):
            raise ValueError("An experiment episode is already active; assess or withdraw it first")

        observation = _visible_observation(workspace_dir, branch)
        observation_hash, observation_artifact = _snapshot_observation(
            workspace_dir, branch, observation
        )
        episode = {
            "episode_id": "episode_" + uuid.uuid4().hex[:16],
            "created_at": _now(),
            "status": "active",
            "round_started": int(pipeline_state.get("round", 0)),
            "title": title.strip(),
            "hypothesis": hypothesis.strip(),
            "objective": objective.strip(),
            "evidence": evidence,
            "permitted_actions": permitted_actions,
            "max_action_calls": max_action_calls,
            "actions_used": 0,
            "actions": [],
            "decision_variables": decision_variables,
            "budget_plan": budget_plan,
            "success_signals": success_signals,
            "replan_triggers": replan_triggers,
            "considered_alternatives": alternatives,
            "rationale": rationale.strip(),
            "observation_hash": observation_hash,
            "observation_artifact": observation_artifact,
            "agent_metadata": agent_metadata or {},
        }
        protocol["active_episode"] = episode
        protocol["assessment_required"] = False
        protocol["assessment_reasons"] = []
        _save_protocol(workspace_dir, branch, protocol)
        _append_event(workspace_dir, branch, {"event": "episode_opened", **episode})
        return json.dumps({
            "episode_id": episode["episode_id"],
            "status": "active",
            "observation_hash": observation_hash,
            "permitted_actions": permitted_actions,
            "max_action_calls": max_action_calls,
            "next_step": (
                "Observe validation. In later model steps, gather read-only evidence freely and "
                "execute at most one permitted consequential action per step."
            ),
        }, ensure_ascii=False)


class AssessExperimentEpisode(Tool):
    name = "assess_experiment_episode"
    description = (
        "Close the active episode at a milestone, failure, replan trigger, or voluntary evidence "
        "checkpoint. Summarize observable evidence and update the next experimental belief."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expectation_status": {
                "type": "string",
                "enum": ["supported", "partially_supported", "refuted", "inconclusive"],
            },
            "outcome_summary": {"type": "string", "minLength": 1},
            "evidence": {
                "type": "array", "minItems": 1, "maxItems": 12,
                "items": {"type": "string", "minLength": 1},
            },
            "belief_update": {"type": "string", "minLength": 1},
            "next_step": {"type": "string", "minLength": 1},
        },
        "required": [
            "expectation_status", "outcome_summary", "evidence", "belief_update", "next_step"
        ],
    }

    def run(self, expectation_status, outcome_summary, evidence, belief_update, next_step,
            branch="main", workspace_dir=None, agent_metadata=None, **_):
        if expectation_status not in (
            "supported", "partially_supported", "refuted", "inconclusive"
        ):
            raise ValueError("Invalid expectation_status")
        protocol = _load_protocol(workspace_dir, branch)
        episode = protocol.get("active_episode")
        if not episode:
            raise ValueError("There is no active experiment episode to assess")
        if protocol.get("executing_action"):
            raise ValueError("Reconcile the executing action before assessing the episode")
        last_action = protocol.get("last_action") or {}
        last_metadata = last_action.get("agent_metadata") or {}
        if (
            last_action.get("episode_id") == episode.get("episode_id")
            and last_metadata.get("turn_id") == (agent_metadata or {}).get("turn_id")
            and last_metadata.get("model_step") is not None
            and (agent_metadata or {}).get("model_step") is not None
            and int(last_metadata["model_step"]) == int(agent_metadata["model_step"])
        ):
            raise ValueError(
                "Assess the episode in a later model step after observing the action result"
            )
        for field, value in (
            ("outcome_summary", outcome_summary),
            ("belief_update", belief_update),
            ("next_step", next_step),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        evidence = _nonempty_strings(evidence, "evidence", 1, 12)
        observation = _visible_observation(workspace_dir, branch)
        observation_hash, observation_artifact = _snapshot_observation(
            workspace_dir, branch, observation
        )
        assessment = {
            "episode_id": episode["episode_id"],
            "title": episode.get("title"),
            "hypothesis": episode.get("hypothesis"),
            "actions_used": int(episode.get("actions_used", 0)),
            "expectation_status": expectation_status,
            "outcome_summary": outcome_summary.strip(),
            "evidence": evidence,
            "belief_update": belief_update.strip(),
            "next_step": next_step.strip(),
            "assessment_reasons": protocol.get("assessment_reasons") or [],
            "post_episode_observation_hash": observation_hash,
            "post_episode_observation_artifact": observation_artifact,
            "agent_metadata": agent_metadata or {},
        }
        protocol["active_episode"] = None
        protocol["assessment_required"] = False
        protocol["assessment_reasons"] = []
        _save_protocol(workspace_dir, branch, protocol)
        _append_event(workspace_dir, branch, {"event": "episode_assessed", **assessment})
        return json.dumps({
            "episode_id": episode["episode_id"],
            "status": "closed",
            "expectation_status": expectation_status,
            "belief_update": belief_update.strip(),
            "next_step": next_step.strip(),
            "post_episode_observation_hash": observation_hash,
        }, ensure_ascii=False)


class WithdrawExperimentEpisode(Tool):
    name = "withdraw_experiment_episode"
    description = (
        "Withdraw an active episode before any consequential action has run when newer read-only "
        "evidence invalidates its premise. The withdrawal remains in the audit trajectory."
    )
    parameters = {
        "type": "object",
        "properties": {"reason": {"type": "string", "minLength": 1}},
        "required": ["reason"],
    }

    def run(self, reason, branch="main", workspace_dir=None, agent_metadata=None, **_):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Withdrawal reason must be non-empty")
        protocol = _load_protocol(workspace_dir, branch)
        episode = protocol.get("active_episode")
        if not episode or protocol.get("executing_action"):
            raise ValueError("There is no idle active episode to withdraw")
        episode_metadata = episode.get("agent_metadata") or {}
        if (
            episode_metadata.get("turn_id") == (agent_metadata or {}).get("turn_id")
            and episode_metadata.get("model_step") is not None
            and (agent_metadata or {}).get("model_step") is not None
            and int(episode_metadata["model_step"]) == int(agent_metadata["model_step"])
        ):
            raise ValueError("Withdraw the episode only after observing its validation result")
        if int(episode.get("actions_used", 0)) != 0:
            raise ValueError("An episode with executed actions must be assessed, not withdrawn")
        protocol["active_episode"] = None
        protocol["assessment_required"] = False
        protocol["assessment_reasons"] = []
        _save_protocol(workspace_dir, branch, protocol)
        _append_event(workspace_dir, branch, {
            "event": "episode_withdrawn",
            "episode_id": episode["episode_id"],
            "title": episode.get("title"),
            "reason": reason.strip(),
            "agent_metadata": agent_metadata or {},
        })
        return json.dumps({
            "episode_id": episode["episode_id"],
            "status": "withdrawn",
            "reason": reason.strip(),
        }, ensure_ascii=False)


class ReconcileInterruptedAction(Tool):
    name = "reconcile_interrupted_action"
    description = (
        "Recover a protocol state left executing by a process interruption. It never retries the "
        "action; it snapshots current visible state. Episode actions require assessment, while "
        "standalone actions return to observation-driven replanning."
    )
    parameters = {
        "type": "object",
        "properties": {"reason": {"type": "string", "minLength": 1}},
        "required": ["reason"],
    }

    def run(self, reason, branch="main", workspace_dir=None, agent_metadata=None, **_):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Reconciliation reason must be non-empty")
        protocol = _load_protocol(workspace_dir, branch)
        executing = protocol.get("executing_action")
        episode = protocol.get("active_episode")
        if not executing:
            raise ValueError("There is no interrupted executing action to reconcile")
        observation = _visible_observation(workspace_dir, branch)
        observation_hash, observation_artifact = _snapshot_observation(
            workspace_dir, branch, observation
        )
        last_action = {
            **executing,
            "status": "interrupted",
            "completed_at": _now(),
            "result_hash": None,
            "result_artifact": None,
            "result_summary": reason.strip(),
            "reconciliation_observation_hash": observation_hash,
            "reconciliation_observation_artifact": observation_artifact,
        }
        if episode:
            episode["actions_used"] = int(episode.get("actions_used", 0)) + 1
            episode.setdefault("actions", []).append({
                "action_id": executing["action_id"],
                "target_tool": executing["target_tool"],
                "status": "interrupted",
            })
        protocol["active_episode"] = episode
        protocol["executing_action"] = None
        protocol["last_action"] = last_action
        protocol["assessment_required"] = bool(episode)
        protocol["assessment_reasons"] = ["action_interrupted"] if episode else []
        _save_protocol(workspace_dir, branch, protocol)
        _append_event(workspace_dir, branch, {
            "event": "episode_action_interrupted",
            **last_action,
            "agent_metadata": agent_metadata or {},
        })
        return json.dumps({
            "episode_id": executing.get("episode_id"),
            "action_id": executing["action_id"],
            "status": "interrupted",
            "observation_hash": observation_hash,
            "next_step": (
                "Assess the active episode with expectation_status=inconclusive."
                if episode else
                "Inspect current observable state before choosing any retry or replacement action."
            ),
        }, ensure_ascii=False)
