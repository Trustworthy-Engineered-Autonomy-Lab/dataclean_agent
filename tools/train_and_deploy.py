import json
import time
from pathlib import Path
import numpy as np
from .base import Tool
from .utils import _load, _save, _artifact, record_observation, append_ledger, _ensure_constraints


class TrainAndDeploy(Tool):
    name = "train_and_deploy"
    description = (
        "Train imitation learning controller on clean dataset and immediately deploy & evaluate CTE performance in simulation in a single unified step."
    )
    parameters = {
        "type": "object",
        "properties": {
            "clean_dataset_id": {
                "type": "string",
                "description": "Optional specific clean dataset artifact filename (defaults to active_clean_dataset)."
            },
            "min_improvement": {
                "type": "number",
                "description": "Optional relative improvement threshold to flag CTE stagnation (default 0.05)."
            }
        },
        "required": []
    }

    def run(self, clean_dataset_id=None, min_improvement=0.05, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)

        clean_path = None
        if clean_dataset_id:
            target_artifact = _artifact(workspace_dir, clean_dataset_id, branch=branch)
            if target_artifact.exists():
                clean_path = target_artifact

        if not clean_path and s.get("active_clean_dataset"):
            active_p = Path(s["active_clean_dataset"])
            if active_p.exists():
                clean_path = active_p

        if not clean_path or not clean_path.exists():
            raise ValueError(f"No usable clean dataset found: {clean_dataset_id or s.get('active_clean_dataset')}")

        try:
            data = json.loads(clean_path.read_text()).get("records", [])
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to read clean dataset {clean_path.name}: {e}") from e

        if len(data) < 10:
            raise ValueError(f"Clean dataset contains too few samples ({len(data)} samples; minimum 10 required)")

        steer = np.array([r["steering"] for r in data])
        quality = 1.0 - float(np.mean([r.get("anomaly_score", 0.0) for r in data]))

        cur_round = int(s.get("round", 0))
        cid = f"ctrl-{branch}-r{cur_round+1}-{int(time.time())}"

        ctrl_obj = {
            "id": cid,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clean_dataset": str(clean_path),
            "samples": len(data),
            "steering_mean": round(float(np.mean(steer)), 5),
            "steering_std": round(float(np.std(steer)), 5),
            "quality": round(quality, 5),
            "round": cur_round + 1,
            "branch": branch,
        }
        s["active_controller"] = ctrl_obj

        # Deployment evaluation step
        max_dep = (s.get("constraints") or {}).get("max_deployments")
        if max_dep is not None and s.get("deployments", 0) >= max_dep:
            raise ValueError(f"Maximum deployment limit reached ({max_dep}). Cannot deploy further.")

        d = s.get("deployments", 0) + 1
        cte = max(.05, .74 - .42 * quality + .012 * d)
        prior = s.get("last_deployed_cte")
        improvement = None if prior is None else (prior - cte) / prior

        s["deployments"] = d
        s["skip_streak"] = 0
        s["last_deployed_cte"] = cte
        s["best_cte"] = cte if s.get("best_cte") is None else min(s["best_cte"], cte)

        cap_hit = (max_dep is not None and d >= max_dep)
        min_imp = float(min_improvement) if min_improvement is not None else 0.05
        stagnant = (improvement is not None and improvement < min_imp)
        terminate = cap_hit

        deploy_result = {
            "deployment": d,
            "controller_id": cid,
            "clean_dataset": clean_path.name,
            "clean_samples": len(data),
            "cte_mean": round(cte, 5),
            "cte_std": round(.03 + .01 * (1 - quality), 5),
            "improvement_from_previous": None if improvement is None else round(improvement, 5),
            "termination_required": terminate,
            "reason": "Maximum deployment limit reached" if cap_hit else (f"CTE improvement under {min_imp*100:.1f}%, termination recommended but iteration permitted" if stagnant else "Iteration permitted"),
            "mode": "deterministic_proxy"
        }

        s["last_deployment"] = deploy_result
        s["termination_required"] = terminate
        s["termination_reason"] = deploy_result["reason"] if terminate else None

        record_observation(s, "train_and_deploy", deploy_result, workspace_dir=workspace_dir, branch=branch)
        append_ledger(s, {
            "stage": "train_and_deploy",
            "round": cur_round + 1,
            "controller_id": cid,
            "cte_mean": round(cte, 5),
            "improvement": None if improvement is None else round(improvement, 5)
        })

        _save(workspace_dir, s, branch=branch)
        return json.dumps(deploy_result, ensure_ascii=False)
