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
                    summary["best_cte"] = s.get("best_cte")
                except Exception:
                    pass
            out.append(summary)
        return json.dumps(out, ensure_ascii=False)
