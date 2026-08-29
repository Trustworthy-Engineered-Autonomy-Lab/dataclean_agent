import json
from pathlib import Path
from .base import Tool
from .utils import ROOT, TASKS_DIR, STATE

class ListTasks(Tool):
    name = "list_tasks"
    description = "Enumerate all registered tasks in the workspace with progress summaries."
    parameters = {
        "type": "object",
        "properties": {}
    }

    def run(self, workspace_dir=None, **_):
        root = Path(workspace_dir) / ROOT / TASKS_DIR
        if not root.exists():
            return json.dumps([], ensure_ascii=False)

        out = []
        for td in sorted(root.iterdir()):
            if not td.is_dir():
                continue
            task_id = td.name
            spec_path = td / "task_spec.json"
            state_path = td / STATE
            if not spec_path.exists() and not state_path.exists():
                continue
            spec = {}
            if spec_path.exists():
                try:
                    spec = json.loads(spec_path.read_text())
                except Exception:
                    spec = {}
            summary = {
                "task_id": task_id,
                "description": spec.get("description", ""),
                "execution_mode": spec.get("execution_mode", "adaptive_agent"),
                "transition_policy": spec.get("transition_policy", "deploy_collect_merge"),
                "configured": False,
                "round": None,
                "deployments": None,
                "clean_count": None,
            }
            state_path = td / STATE
            if state_path.exists():
                try:
                    s = json.loads(state_path.read_text())
                    summary["configured"] = True
                    summary["round"] = s.get("round")
                    summary["deployments"] = s.get("deployments")
                    summary["clean_count"] = s.get("clean_count")
                    summary["round_status"] = s.get("round_status")
                    internal_status = s.get("task_status", "DRAFT")
                    summary["dataset_configuration"] = (
                        "editable" if internal_status == "DRAFT" else "frozen"
                    )
                    summary["execution_status"] = (
                        "completed"
                        if internal_status == "COMPLETED"
                        else ("not_started" if internal_status == "DRAFT" else "started")
                    )
                    summary["round_input_count"] = s.get("round_input_count")
                except Exception:
                    pass
            out.append(summary)
        return json.dumps(out, ensure_ascii=False)
