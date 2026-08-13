import json
from .base import Tool
from pathlib import Path
from datetime import datetime
from .teacar import TEACar, IROS_WS_DIR
from .eval_controller import CTE_PROJECT_DIR

class TransferEvalResults(Tool):
    name = "transfer_eval_results"
    description = (
        "Transfer the results of an eval_controller run from the physical 1/16 racing "
        "car to the local host over SFTP/SSH. Downloads two files: the collected-data "
        ".tar.gz from the car (remote_data_path), and the cte csv file, recording the "
        "car's distance from the track's guide line, from the jump host "
        "(remote_cte_path). Both files are saved into local_folder, resolved relative "
        "to the workspace directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "remote_data_path": {
                "type": "string",
                "description": "Path to the collected-data .tar.gz on the car to download (the tar path returned by eval_controller)."
            },
            "remote_cte_path": {
                "type": "string",
                "description": "Path to the cte csv file on the jump host to download (the cte path returned by eval_controller)."
            },
            "local_folder": {
                "type": "string",
                "description": "Destination folder on the local host, resolved relative to the workspace directory."
            }
        },
        "required": ["remote_data_path", "remote_cte_path", "local_folder"]
    }

    def run(self, remote_data_path: str, remote_cte_path: str, local_folder: str, workspace_dir=None, **_):

        remote_data_path = Path(remote_data_path)
        remote_cte_path = Path(remote_cte_path)
        local_folder = (Path(workspace_dir) / local_folder).resolve()
        local_data_path = local_folder / remote_data_path.name
        local_cte_path = local_folder / remote_cte_path.name

        try:
            local_folder.mkdir(parents=True, exist_ok=True)
            teacar = TEACar()
            with teacar as car:

                with car.open_sftp() as sftp:
                    sftp.chdir(IROS_WS_DIR)
                    sftp.get(str(remote_data_path), str(local_data_path))

                with teacar.jump.open_sftp() as sftp:
                    sftp.chdir(CTE_PROJECT_DIR)
                    sftp.get(str(remote_cte_path), str(local_cte_path))

        except Exception as e:
            return f"Failed to transfer: {e}"


        return f"Transferred collected data from {remote_data_path} to {local_data_path}"\
               f"Transferred cte from {remote_cte_path} to {local_cte_path}"