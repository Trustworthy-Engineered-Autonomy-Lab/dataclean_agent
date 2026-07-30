import json
from .base import Tool
from .utils import _load, _save, _ensure_constraints

class DeployEval(Tool):
    name = "deploy_eval"
    description = "Deploy and evaluate controller performance in simulation. Increments deployment count and records CTE metrics."
    parameters = {
        "type": "object",
        "properties": {
            "controller_id": {
                "type": "string",
                "description": "Optional specific controller ID to deploy and evaluate. Defaults to active_controller."
            },
            "min_improvement": {
                "type": "number",
                "description": "Optional relative improvement threshold to flag stagnation (default 0.05)."
            }
        },
        "required": []
    }
    
    def run(self, controller_id=None, min_improvement=0.05, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)

        max_dep = (s.get("constraints") or {}).get("max_deployments")
        if max_dep is not None and s["deployments"] >= max_dep:
            raise ValueError(f"Maximum deployment limit reached ({max_dep}). Cannot deploy further.")

        ctrl = s.get("active_controller") or {}
        target_ctrl_id = controller_id or ctrl.get("id")
        
        if not target_ctrl_id:
            raise ValueError("No controller_id specified and no active_controller active.")

        d = s["deployments"] + 1
        ctrl_quality = ctrl.get("quality", 0.5)
        
        cte = max(.05, .74 - .42 * ctrl_quality + .012 * d)
        prior = s.get("last_deployed_cte")
        improvement = None if prior is None else (prior - cte) / prior
        
        s["deployments"] = d
        s["skip_streak"] = 0
        s["last_deployed_cte"] = cte
        s["best_cte"] = cte if s["best_cte"] is None else min(s["best_cte"], cte)
        
        max_dep = (s.get("constraints") or {}).get("max_deployments")
        cap_hit = (max_dep is not None and d >= max_dep)
        min_imp = float(min_improvement) if min_improvement is not None else 0.05
        stagnant = (improvement is not None and improvement < min_imp)
        terminate = cap_hit
        
        result = {
            "deployment": d,
            "controller_id": target_ctrl_id,
            "cte_mean": round(cte, 5),
            "cte_std": round(.03 + .01 * (1 - ctrl_quality), 5),
            "improvement_from_previous": None if improvement is None else round(improvement, 5),
            "termination_required": terminate,
            "reason": "Maximum deployment limit reached" if cap_hit else (f"CTE improvement under {min_imp*100:.1f}%, termination recommended but iteration permitted" if stagnant else "Iteration permitted"),
            "mode": "deterministic_proxy"
        }
        
        s["last_deployment"] = result
        s["termination_required"] = terminate
        s["termination_reason"] = result["reason"] if terminate else None

        _save(workspace_dir, s, branch=branch)
        return json.dumps(result, ensure_ascii=False)