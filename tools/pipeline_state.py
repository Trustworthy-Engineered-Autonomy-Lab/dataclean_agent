import json
from .base import Tool
from .utils import _load, _dataset_config

class PipelineState(Tool):
    name = "get_pipeline_state"
    description = "Read compact structured pipeline observation state for active task branch."
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def run(self, branch="main", workspace_dir=None, **_):
        ds = None
        try:
            ds = _dataset_config(workspace_dir)
        except Exception:
            ds = None

        try:
            s = _load(workspace_dir, branch=branch)
        except Exception:
            return json.dumps({
                "branch": branch,
                "task_created": False,
                "dataset_configured": bool(ds and ds.get("sources")),
                "raw_samples": (ds or {}).get("raw_samples"),
            }, ensure_ascii=False)

        out = {
            "branch": branch,
            "task_created": True,
            "round": s.get("round", 0),
            "deployments": s.get("deployments", 0),
            "active_detector": s.get("active_detector"),
            "active_clean_dataset": s.get("active_clean_dataset"),
            "dataset_id": (ds or {}).get("dataset_id"),
            "raw_samples": (ds or {}).get("raw_samples"),
            "sources": [x.get("name") for x in (ds or {}).get("sources", [])] if ds else [],
            "source_composition": (ds or {}).get("source_composition"),
        }
        return json.dumps(out, ensure_ascii=False)