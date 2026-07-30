import json
import time
from pathlib import Path
import numpy as np
from .base import Tool
from .utils import _load, _save, _artifact

class TrainController(Tool):
    name = "train_controller"
    description = "Train lightweight image-to-steering controller on specified D_clean dataset."
    parameters = {
        "type": "object",
        "properties": {
            "clean_dataset_id": {
                "type": "string",
                "description": "Optional clean dataset artifact filename (e.g. 'clean_r1.json'). Defaults to active_clean_dataset."
            }
        },
        "required": []
    }
    
    def run(self, clean_dataset_id=None, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch) or {}
        
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
        metrics = {
            "train_samples": len(data),
            "steering_mean": round(float(steer.mean()), 5),
            "steering_std": round(float(steer.std()), 5),
            "proxy_validation_mae": round(max(.025, .22 - .16 * quality), 5)
        }
        
        s["active_controller"] = {"id": cid, "quality": quality, "metrics": metrics}
        _save(workspace_dir, s, branch=branch)
        
        return json.dumps({"controller_id": cid, "metrics": metrics, "branch": branch}, ensure_ascii=False)