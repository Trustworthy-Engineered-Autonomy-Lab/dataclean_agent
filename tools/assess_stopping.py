import json
import numpy as np

from .base import Tool
from .decision_policy import effective_action, record_decision
from .utils import _load, _save, record_observation


class AssessStopping(Tool):
    name = "assess_stopping"
    description = (
        "Record an explicit continue/stop decision through the shared adaptive/fixed-policy "
        "interface. Fixed baselines stop only at their preregistered max_rounds. "
        "Includes convergence diagnostics: CTE improvement, VLM recovery volume, and threshold stability."
    )
    parameters = {
        "type": "object",
        "properties": {
            "stop": {"type": "boolean"},
            "rationale": {
                "type": "string",
                "description": "Observation-based reason to stop or continue.",
            },
        },
        "required": ["stop", "rationale"],
    }

    @staticmethod
    def _check_convergence(state):
        """Check three convergence signals: CTE improvement, VLM recovery volume, threshold stability.

        Returns dict with convergence status and diagnostics.
        """
        result = {
            "cte_converged": None,
            "recovery_converged": None,
            "threshold_converged": None,
            "cte_history": [],
            "recovery_history": [],
            "threshold_history": [],
            "diagnostics": {},
        }

        # Extract CTE history from deployment_runs
        deployments = state.get("deployment_runs") or []
        cte_values = [float(d.get("cte_mean") or d.get("real_cte_mean") or 0)
                      for d in deployments if d.get("status") == "transferred"]
        result["cte_history"] = cte_values

        # Check CTE convergence: last 2 rounds improvement < 5%
        if len(cte_values) >= 2:
            last_cte = cte_values[-1]
            prev_cte = cte_values[-2]
            if prev_cte > 0:
                cte_improvement = (prev_cte - last_cte) / prev_cte * 100
                result["diagnostics"]["cte_improvement_pct"] = round(cte_improvement, 2)
                result["cte_converged"] = cte_improvement < 5.0
            else:
                result["cte_converged"] = False

        # Extract VLM recovery counts from decision_trace (resolve observations)
        decision_trace = state.get("decision_trace") or []
        resolve_observations = [d.get("observation") for d in decision_trace
                               if d.get("decision") == "resolve" and d.get("observation")]

        # Collect vlm_accepted counts from resolve stage observations
        recovery_counts = []
        for obs in resolve_observations:
            # obs is from resolve tool, contains vlm_accepted count
            if isinstance(obs, dict) and "vlm_accepted" in obs:
                recovery_counts.append(int(obs["vlm_accepted"]))

        # Fallback: check latest resolve observation
        if not recovery_counts:
            latest_resolve = (state.get("latest_observation") or {}).get("resolve") or {}
            if "vlm_accepted" in latest_resolve:
                recovery_counts = [int(latest_resolve["vlm_accepted"])]

        result["recovery_history"] = recovery_counts

        # Check recovery convergence: last 2 rounds change < 5%
        if len(recovery_counts) >= 2:
            last_recovery = recovery_counts[-1]
            prev_recovery = recovery_counts[-2]
            if prev_recovery > 0:
                recovery_change = abs(last_recovery - prev_recovery) / prev_recovery * 100
                result["diagnostics"]["recovery_change_pct"] = round(recovery_change, 2)
                result["recovery_converged"] = recovery_change < 5.0
            else:
                result["recovery_converged"] = recovery_change == 0

        # Extract threshold history from decision_trace (partition decisions)
        partition_decisions = [d for d in decision_trace if d.get("decision") == "partition"]
        thresholds = []
        for dec in partition_decisions:
            effective = dec.get("effective") or {}
            if "threshold" in effective:
                thresholds.append(float(effective["threshold"]))

        # Also include current threshold
        current_partition = state.get("latest_partition") or {}
        if "threshold" in current_partition:
            thresholds.append(float(current_partition["threshold"]))

        result["threshold_history"] = [round(t, 6) for t in thresholds]

        # Check threshold convergence: last 2 rounds relative change < 2%
        if len(thresholds) >= 2:
            last_threshold = thresholds[-1]
            prev_threshold = thresholds[-2]
            if prev_threshold > 0:
                threshold_change = abs(last_threshold - prev_threshold) / prev_threshold * 100
                result["diagnostics"]["threshold_change_pct"] = round(threshold_change, 2)
                result["threshold_converged"] = threshold_change < 2.0
            else:
                result["threshold_converged"] = threshold_change == 0

        # Overall convergence: all three signals converged
        if all(v is not None for v in [result["cte_converged"],
                                       result["recovery_converged"],
                                       result["threshold_converged"]]):
            result["all_converged"] = all([
                result["cte_converged"],
                result["recovery_converged"],
                result["threshold_converged"]
            ])

        return result

    def run(self, stop, rationale, branch="main", workspace_dir=None, **_):
        state = _load(workspace_dir, branch=branch)
        if state.get("task_status") == "COMPLETED":
            raise ValueError("Completed tasks are immutable; create a new task to run another experiment")
        proposed = {"stop": bool(stop)}
        effective, source = effective_action(state, "stopping", proposed)
        if source == "fixed_policy":
            max_rounds = int(effective["max_rounds"])
            stop = int(state.get("round", 0)) >= max_rounds
            effective = {**effective, "stop": stop}
            rationale = rationale or f"Preregistered maximum of {max_rounds} completed rounds"
        else:
            stop = bool(effective.get("stop", stop))
            if not str(rationale).strip():
                raise ValueError("Adaptive stopping decisions require an observation-based rationale")
            max_rounds = (state.get("constraints") or {}).get("max_rounds")
            if max_rounds is not None and int(state.get("round", 0)) >= int(max_rounds):
                stop = True
                effective = {**effective, "stop": True}
                rationale = f"Experimental maximum of {int(max_rounds)} committed rounds reached"
            elif stop and not (state.get("constraints") or {}).get("allow_voluntary_terminate", True):
                raise ValueError("Voluntary termination is disabled by this task's preregistered constraints")

        state["termination_required"] = stop
        state["termination_reason"] = rationale if stop else None
        if stop:
            state["task_status"] = "COMPLETED"

        # Compute convergence diagnostics for adaptive decisions
        convergence = self._check_convergence(state)

        observation = {
            "round": int(state.get("round", 0)),
            "round_status": state.get("round_status"),
            "deployments": int(state.get("deployments", 0)),
        }
        decision_entry = record_decision(
            state, "stopping", proposed, effective, rationale, source,
            observation=observation,
        )
        summary = {
            "stop": stop, "decision_source": source,
            "execution_status": "completed" if stop else "active",
            "convergence_diagnostics": convergence,
            **observation,
        }
        record_observation(
            state,
            "assess_stopping",
            summary,
            workspace_dir=workspace_dir,
            branch=branch,
            decision=decision_entry,
        )
        _save(workspace_dir, state, branch=branch)
        return json.dumps(summary, ensure_ascii=False)
