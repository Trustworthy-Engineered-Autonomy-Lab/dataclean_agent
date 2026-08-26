import json
import hashlib
import re
import shlex
from pathlib import PurePosixPath
from pathlib import Path
from .base import Tool
from .teacar import TEACar, CONTROLLER_TARGET_DIR, HOSTNAME
from .image_contract import DEPLOYMENT_INPUT_SHAPE, INPUT_CONTRACT_VERSION
from .utils import _load, _save, _ensure_constraints, record_observation, print_progress

class DeployController(Tool):
    name = "deploy_controller"
    description = (
        "Copy a trained ONNX controller file to the physical 1/16 racing car over Jump "
        "Host SFTP/SSH when the conversational task calls for physical deployment."
    )
    parameters = {
        "type": "object",
        "properties": {
            "controller_path": {
                "type": "string",
                "description": "Optional relative path to the ONNX controller file. Defaults to active_controller weights_onnx."
            }
        },
        "required": []
    }
    
    def run(self, controller_path=None, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if s.get("termination_required"):
            raise ValueError("Task is completed; create a new task before deployment")
        target_file = controller_path
        if not target_file:
            ctrl = s.get("active_controller") or {}
            target_file = ctrl.get("weights_onnx") or (ctrl.get("metrics") or {}).get("weights_onnx_artifact")
            if not target_file and ctrl.get("id"):
                target_file = f"{ctrl['id']}.onnx"

        if not target_file:
            raise ValueError("No controller_path specified and no active_controller ONNX model found in state.")
        file_name = Path(str(target_file)).name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", file_name):
            raise ValueError("controller_path contains unsafe characters")
        from .utils import _artifact
        source_path = _artifact(workspace_dir, file_name, branch=branch).resolve()

        if not source_path.exists():
            raise FileNotFoundError(f"Controller file not found: {source_path}")

        # Validate the artifact itself rather than trusting stale task metadata.
        # The car sends NHWC camera frames directly to this exported wrapper.
        try:
            import onnx
            graph = onnx.load(str(source_path), load_external_data=False).graph
            if len(graph.input) != 1:
                raise ValueError("controller ONNX must expose exactly one image input")
            dims = graph.input[0].type.tensor_type.shape.dim
            shape = [int(dim.dim_value) if int(dim.dim_value) > 0 else None for dim in dims]
        except Exception as exc:
            raise ValueError(f"Unable to validate controller ONNX input contract: {exc}") from exc
        if len(shape) != 4 or tuple(shape[-3:]) != DEPLOYMENT_INPUT_SHAPE:
            raise ValueError(
                f"Controller ONNX input shape {shape} is incompatible; expected "
                f"NHWC [N,{DEPLOYMENT_INPUT_SHAPE[0]},{DEPLOYMENT_INPUT_SHAPE[1]},"
                f"{DEPLOYMENT_INPUT_SHAPE[2]}]. Retrain the controller."
            )

        print_progress(f"[DeployController] Connecting through Jump Host to car {HOSTNAME} via SFTP...")
        try:
            with TEACar() as client:
                target_path = PurePosixPath(CONTROLLER_TARGET_DIR) / source_path.name
                quoted_dir = shlex.quote(CONTROLLER_TARGET_DIR)
                _, mkdir_out, _ = client.exec_command(
                    f"mkdir -p {quoted_dir} && test -d {quoted_dir}"
                )
                if mkdir_out.channel.recv_exit_status() != 0:
                    raise RuntimeError("Unable to create or access remote controller directory")

                with client.open_sftp() as sftp:
                    print_progress(f"[DeployController] Uploading {source_path.name} -> {target_path}...")
                    sftp.put(str(source_path), str(target_path))

                    data_candidates = [
                        source_path.with_name(source_path.name + ".data"),
                        source_path.with_name(source_path.name + ".tmp.data"),
                    ]

                    for data_file in data_candidates:
                        if data_file.exists():
                            target_data_path = PurePosixPath(CONTROLLER_TARGET_DIR) / data_file.name
                            print_progress(
                                f"[DeployController] Uploading companion data {data_file.name}..."
                            )
                            sftp.put(str(data_file), str(target_data_path))

        except Exception as e:
            res = {"status": "failed", "error": f"Failed to deploy {source_path.name}: {e}"}
            return json.dumps(res, ensure_ascii=False)

        res = {
            "status": "success",
            "controller_deployed": source_path.name,
            "controller_id": (s.get("active_controller") or {}).get("id"),
            "controller_fingerprint": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "target_path": str(target_path),
            "car_host": HOSTNAME,
            "onnx_input_shape": shape,
            "input_contract_version": INPUT_CONTRACT_VERSION,
        }
        s["last_deployed_controller"] = res
        record_observation(s, "deploy_controller", res, workspace_dir=workspace_dir, branch=branch)
        _save(workspace_dir, s, branch=branch)

        return json.dumps(res, ensure_ascii=False)
