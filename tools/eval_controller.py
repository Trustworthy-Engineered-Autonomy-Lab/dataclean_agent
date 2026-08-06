import json
from .base import Tool
from pathlib import Path
from contextlib import chdir
import paramiko

from .deploy_controller import USERNAME, HOSTNAME, PASSWORD, CONTROLLER_TARGET_DIR

class EvalController(Tool):
    name = "eval_controller"
    description = (
        "Run a deployed controller on the physical 1/16 racing car on a track: "
        "the controller drives steering while a human controls throttle and "
        "decides when to capture each image, at a fixed frequency. "
        "The run ends once n_images images have been collected."
    )
    parameters = {
        "type": "object",
        "properties": {
            "controller_path": {
                "type": "string",
                "description": "Path to the controller file already deployed on the car (e.g. via deploy_controller)."
            },
            "n_images": {
                "type": "integer",
                "description": "Number of images to collect before the evaluation run exits."
            }
        },
        "required": ["controller_path", "n_images"]
    }
    
    def run(self, controller_path: str, n_images: int, workspace_dir=None, **_):

        try:
            with paramiko.SSHClient() as client:
                client.load_system_host_keys()
                client.set_missing_host_key_policy(paramiko.RejectPolicy())

                client.connect(
                    hostname=HOSTNAME,
                    username=USERNAME,
                    password=PASSWORD,
                )

                stdin, stdout, stderr = client.exec_command(
                    "cd ~/iros_ws && "
                    "source ~/iros_ws/devel/setup.bash && "
                    "roslaunch iros_bringup self_evolve_paper.launch "
                    f"model_path:={controller_path} "
                    f"exit_threshold:={n_images}"
                )
                stdout_text = stdout.read().decode()
                stderr_text = stderr.read().decode()
        except Exception as e:
            return json.dumps({
                "stdout": "",
                "stderr": "",
                "result": f"Failed to evaluate {controller_path}: {e}"
            }, ensure_ascii=False)

        return json.dumps({
            "stdout": stdout_text,
            "stderr": stderr_text,
            "result": "Finished evaluating the controller"
        }, ensure_ascii=False)