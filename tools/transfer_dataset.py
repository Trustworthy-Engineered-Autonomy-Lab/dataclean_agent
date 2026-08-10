import json
from .base import Tool
from pathlib import Path
from datetime import datetime
from .teacar import TEACar

class TransferDataset(Tool):
    name = "transfer_dataset"
    description = (
        "Transfer a file (e.g. the dataset .tar.gz produced by eval_controller) from the "
        "physical 1/16 racing car to the local host over SFTP/SSH."
    )
    parameters = {
        "type": "object",
        "properties": {
            "remote_path": {
                "type": "string",
                "description": "Path to the file on the car to download (e.g. the .tar.gz path returned by eval_controller)."
            },
            "local_path": {
                "type": "string",
                "description": "Destination path on the local host, resolved relative to the workspace directory."
            }
        },
        "required": ["remote_path", "local_path"]
    }

    def run(self, remote_path: str, local_path: str, workspace_dir=None, **_):

        local_path = (Path(workspace_dir) / local_path).resolve()
        remote_path = Path(remote_path)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with TEACar() as client:

                with client.open_sftp() as sftp:
                    sftp.get(str(remote_path), str(local_path))

        except Exception as e:
            return f"Failed to transfer {remote_path}: {e}"


        return f"Transferred {remote_path} to {local_path}"