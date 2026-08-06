import json
from .base import Tool
from pathlib import Path

from .teacar import TEACar, CONTROLLER_TARGET_DIR

class DeployController(Tool):
    name = "deploy_controller"
    description = "Copy a trained controller file to the physical 1/16 racing car over SFTP/SSH for real-world deployment."
    parameters = {
        "type": "object",
        "properties": {
            "controller_path": {
                "type": "string",
                "description": "Path to the controller file to deploy, resolved relative to the workspace directory."
            }
        },
        "required": ["controller_path"]
    }
    
    def run(self, controller_path: str, workspace_dir=None, **_):

        source_path = (Path(workspace_dir) / controller_path).resolve()

        try:
            with TEACar() as client:
                
                target_path = Path(CONTROLLER_TARGET_DIR) / source_path.name

                with client.open_sftp() as sftp:
                    sftp.put(str(source_path), str(target_path))
                    
        except Exception as e:
            return f"Failed to deploy {controller_path}: {e}"

       
        return f"Deployed {controller_path} to {target_path} on the car"