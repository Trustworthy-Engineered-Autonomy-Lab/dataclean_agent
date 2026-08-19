import json
from .base import Tool
from .utils import _load, _save, _ensure_constraints, record_observation

class DeployEval(Tool):
    name = "deploy_eval"
    agent_exposed = False
    description = (
        "Run a deterministic controller-quality smoke-test proxy. This is not a deployment, "
        "does not produce measured CTE, and must not be used as scientific outcome evidence."
    )
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

        ctrl = s.get("active_controller") or {}
        target_ctrl_id = controller_id or ctrl.get("id")
        
        if not target_ctrl_id:
            raise ValueError("No controller_id specified and no active_controller active.")

        d = int(s.get("simulation_runs", 0)) + 1
        ctrl_quality = ctrl.get("quality", 0.5)
        
        cte = max(.05, .74 - .42 * ctrl_quality + .012 * d)
        prior = s.get("last_proxy_cte")
        improvement = None if prior is None else (prior - cte) / prior
        
        s["simulation_runs"] = d
        s["last_proxy_cte"] = cte
        s["best_proxy_cte"] = cte if s.get("best_proxy_cte") is None else min(s["best_proxy_cte"], cte)

        min_imp = float(min_improvement) if min_improvement is not None else 0.05
        stagnant = (improvement is not None and improvement < min_imp)
        
        result = {
            "simulation_run": d,
            "controller_id": target_ctrl_id,
            "proxy_value": round(cte, 5),
            "proxy_improvement": None if improvement is None else round(improvement, 5),
            "proxy_stagnant": stagnant,
            "note": "Synthetic smoke-test only; not measured CTE and not a deployment outcome.",
            "mode": "deterministic_non_scientific_proxy"
        }

        record_observation(s, "simulation_smoke_test", result, workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, s, branch=branch)
        return json.dumps(result, ensure_ascii=False)
