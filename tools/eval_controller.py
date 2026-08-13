import json
from .base import Tool
from datetime import datetime
from .teacar import TEACar, IROS_WS_DIR
from pathlib import Path

CTE_PROJECT_DIR = "/home/shared/projects/zed"
CTE_PYTHON_PATH = "/home/shared/envs/zed/bin/python"

class EvalController(Tool):
    name = "eval_controller"
    description = (
        "Run a deployed controller on the physical 1/16 racing car on a track marked "
        "with a guide line. The controller drives steering while a human controls "
        "throttle and decides when to capture an image and the corresponding user "
        "action, at a fixed frequency. The run ends once n_images images have been "
        "collected.\n"
        "The collected images are stored in an images folder, and a labels.csv file "
        "records the steering angle and throttle value for each image. The folder is "
        "compressed into a .tar.gz file, whose name is given in the result.\n"
        "While the car runs on the track, a ceiling-mounted camera facing down at the "
        "track photographs the car and sends the images to a server. The server computes "
        "the distance between the car and the track's guide line (the cross-track error, "
        "or cte) and writes the result to a csv file, whose path is given in the result."
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

        postfix = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path("collect_" + postfix)
        data_path = "data" / data_dir
        data_tar = data_dir.with_suffix(".tar.gz")

        cte_filename = Path('cte_'+postfix)
        cte_path = "data" / cte_filename

        remote_tar_path = f"{IROS_WS_DIR}/{data_tar}"

        try:
            teacar = TEACar()
            with teacar as car_client:
                car_in, car_out, _ = car_client.exec_command(
                    f"cd {IROS_WS_DIR} && "
                    "source devel/setup.bash && "
                    "roslaunch iros_bringup self_evolve_paper.launch "
                    f"model_path:={controller_path} "
                    f"exit_threshold:={n_images} "
                    f"data_folder:={data_path} && "
                    f"tar -czf {data_tar} {data_path}"
                )
                car_out.channel.set_combine_stderr(True)

                cte_client = teacar.jump
                cte_in, cte_out, _ = cte_client.exec_command(
                    f"cd {CTE_PROJECT_DIR} && "
                    f"{CTE_PYTHON_PATH} stream.py "
                    f"--output-dir {cte_path}",
                    get_pty=True
                )
                cte_out.channel.set_combine_stderr(True)

                car_out_text = car_out.read().decode()
                car_exit_status = car_out.channel.recv_exit_status()

                # Send Ctrl+C
                cte_in.channel.send("\x03")

                cte_out_text = cte_out.read().decode()

        except Exception as e:
            return json.dumps({
                "car_stdout": "",
                "cte_stdout": "",
                "result": f"Failed to evaluate {controller_path}: {e}"
            }, ensure_ascii=False)

        if car_exit_status != 0:
            return json.dumps({
                "car_stdout": car_out_text,
                "cte_stdout": cte_out_text,
                "result": f"Failed to evaluate {controller_path}: "
                          f"car command exited with status {car_exit_status}, no data saved"
            }, ensure_ascii=False)

        return json.dumps({
            "car_stdout": car_out_text,
            "cte_stdout": cte_out_text,
            "result": "Finished evaluating the controller, "
                      f"collected data saved as {remote_tar_path} on the car, "
                      f"cte saved as {cte_path} on the server"
        }, ensure_ascii=False)