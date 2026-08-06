import json
from .base import Tool
from .utils import _load, _save, _ensure_constraints
from pathlib import Path
import paramiko

HOSTNAME = "teacar2"
USERNAME = "nvidia"
PASSWORD = "nvidia"
CONTROLLER_TARGET_DIR = f"/home/{USERNAME}/iros_ws/controllers/"

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
            with paramiko.SSHClient() as client:
                client.load_system_host_keys()
                client.set_missing_host_key_policy(paramiko.RejectPolicy())

                client.connect(
                    hostname=HOSTNAME,
                    username=USERNAME,
                    password=PASSWORD,
                )

                target_path = Path(CONTROLLER_TARGET_DIR) / source_path.name

                with client.open_sftp() as sftp:
                    sftp.put(str(source_path), str(target_path))
        except Exception as e:
            return f"Failed to deploy {controller_path}: {e}"

       
        return f"Deployed {controller_path} to {target_path} on the car"