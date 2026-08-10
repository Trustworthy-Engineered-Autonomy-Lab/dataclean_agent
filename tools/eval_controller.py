import json
from .base import Tool
from datetime import datetime
from .teacar import TEACar, IROS_WS_DIR

class EvalController(Tool):
    name = "eval_controller"
    description = (
        "Run a deployed controller on the physical 1/16 racing car on a track. "
        "The controller drives steering while a human controls throttle and "
        "decides when to capture an image and the corresponding user action, at a fixed frequency. "
        "The run ends once n_images images have been collected. "
        "The collected images are stored in an images folder, and a labels.csv file records the "
        "steering angle and throttle value for each image. "
        "The folder is compressed into a .tar.gz file, whose name is given in the result."
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

        data_dir = "collect_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = f"data/{data_dir}"
        data_tar = f"{data_path}.tar.gz"
        remote_tar_path = f"{IROS_WS_DIR}/{data_tar}"

        try:
            with TEACar() as client:
                stdin, stdout, stderr = client.exec_command(
                    f"cd {IROS_WS_DIR} && "
                    "source devel/setup.bash && "
                    "roslaunch iros_bringup self_evolve_paper.launch "
                    f"model_path:={controller_path} "
                    f"exit_threshold:={n_images} "
                    f"data_folder:={data_path} && "
                    f"tar -czf {data_tar} {data_path}"
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
            "result": f"Finished evaluating the controller, data saved as {remote_tar_path} on the car"
        }, ensure_ascii=False)