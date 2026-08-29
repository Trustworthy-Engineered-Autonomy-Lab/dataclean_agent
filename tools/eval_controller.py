from datetime import datetime
from pathlib import Path
import json
import os
import re
import shlex
import time
import uuid
from .base import Tool
from .teacar import (
    TEACar, HOSTNAME, IROS_WS_DIR, CONTROLLER_TARGET_DIR, _remote_absolute_path,
)
from .utils import _load, _save, _ensure_constraints, record_observation, append_ledger, print_progress, _anonymize_source_name

CTE_PROJECT_DIR = _remote_absolute_path(
    os.environ.get("DATACLEAN_CTE_PROJECT_DIR", "/home/shared/projects/zed"),
    "DATACLEAN_CTE_PROJECT_DIR",
)
CTE_PYTHON_PATH = _remote_absolute_path(
    os.environ.get("DATACLEAN_CTE_PYTHON", "/home/shared/envs/zed/bin/python"),
    "DATACLEAN_CTE_PYTHON",
)
EVAL_TIMEOUT_SECONDS = int(os.environ.get("DATACLEAN_EVAL_TIMEOUT_SECONDS", "1800"))
CTE_STARTUP_SECONDS = float(os.environ.get("DATACLEAN_CTE_STARTUP_SECONDS", "2"))
if EVAL_TIMEOUT_SECONDS <= 0:
    raise ValueError("DATACLEAN_EVAL_TIMEOUT_SECONDS must be positive")
if not 0 <= CTE_STARTUP_SECONDS <= 60:
    raise ValueError("DATACLEAN_CTE_STARTUP_SECONDS must be in [0, 60]")

class EvalController(Tool):
    name = "eval_controller"
    description = (
        "Run a deployed controller on the physical 1/16 racing car on a track marked with a guide line: "
        "the controller drives steering while a human controls throttle. "
        "Concurrently, a ceiling-mounted ZED camera tracks the car and computes Cross-Track Error (CTE). "
        "The run ends once n_images images are collected, archiving images on car and CTE csv on server."
    )
    parameters = {
        "type": "object",
        "properties": {
            "controller_path": {
                "type": "string",
                "description": "Optional filename of controller already deployed on car. Defaults to last deployed controller."
            },
            "n_images": {
                "type": "integer",
                "description": "Number of images to collect before the evaluation run exits (default 500)."
            }
        },
        "required": []
    }
    
    def run(self, controller_path=None, n_images=500, branch="main", workspace_dir=None,
            cancel_event=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if s.get("termination_required"):
            raise ValueError("Task is completed; create a new task before physical evaluation")
        max_dep = (s.get("constraints") or {}).get("max_deployments")
        if max_dep is not None and int(s.get("deployments", 0)) >= int(max_dep):
            raise ValueError(f"Maximum deployment limit reached ({max_dep})")

        target_file = controller_path
        if not target_file:
            last_dep = s.get("last_deployed_controller") or {}
            target_file = last_dep.get("controller_deployed")
        if not target_file:
            ctrl = s.get("active_controller") or {}
            target_file = ctrl.get("weights_onnx") or (ctrl.get("metrics") or {}).get("weights_onnx_artifact")
            if target_file:
                target_file = Path(target_file).name

        if not target_file:
            raise ValueError("No controller_path provided and no deployed controller found in state. Run deploy_controller first.")
        target_file = Path(str(target_file)).name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", target_file):
            raise ValueError("controller_path contains unsafe characters")
        deployed_controller = s.get("last_deployed_controller") or {}
        if deployed_controller.get("controller_deployed") != target_file:
            raise ValueError(
                "eval_controller requires the exact controller recorded by deploy_controller"
            )
        if not deployed_controller.get("controller_fingerprint"):
            raise ValueError("Deployed controller has no immutable fingerprint; redeploy it first")

        if not isinstance(n_images, int) or isinstance(n_images, bool):
            raise ValueError("n_images must be an integer")
        num_images = n_images
        if not 1 <= num_images <= 100000:
            raise ValueError("n_images must be in [1, 100000]")
        image_budget_used = int(s.get("collection_images_budget_used", 0))
        image_budget_cap = (s.get("constraints") or {}).get("max_collection_images_total")
        if image_budget_cap is not None and image_budget_used + num_images > int(image_budget_cap):
            raise ValueError(
                f"Collection-image budget exceeded: {image_budget_used}+{num_images}>"
                f"{int(image_budget_cap)}"
            )
        
        deployment_run_id = "run_" + uuid.uuid4().hex
        collection_id = "collection_" + uuid.uuid4().hex
        anonymous_source = _anonymize_source_name(collection_id)
        postfix = deployment_run_id
        data_dir = f"collect_{postfix}"
        data_path = f"data/{data_dir}"
        data_tar = f"{data_path}.tar.gz"
        remote_tar_path = f"{IROS_WS_DIR}/{data_tar}"

        cte_filename = f"cte_{postfix}.csv"
        cte_rel_path = f"data/{cte_filename}"
        remote_cte_path = f"{CTE_PROJECT_DIR}/{cte_rel_path}"

        active_controller = s.get("active_controller") or {}
        deployment_run = {
            "deployment_run_id": deployment_run_id,
            "collection_id": collection_id,
            "anonymous_source": anonymous_source,
            "task_id": branch,
            "round": int(s.get("round", 0)),
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "controller_id": deployed_controller.get("controller_id") or active_controller.get("id"),
            "controller_fingerprint": deployed_controller.get("controller_fingerprint"),
            "controller_path": target_file,
            "n_images_target": num_images,
            "remote_data_path": remote_tar_path,
            "remote_cte_path": remote_cte_path,
        }
        # Reserve the physical-evaluation budget before any external side
        # effect. Failed/interrupted attempts are not free retries.
        d = int(s.get("deployments", 0)) + 1
        deployment_run["deployment"] = d
        s["deployments"] = d
        s["collection_images_budget_used"] = image_budget_used + num_images
        s.setdefault("deployment_runs", []).append(deployment_run)
        _save(workspace_dir, s, branch=branch)

        print_progress(f"[EvalController] Launching car evaluation & ZED tracking camera ({num_images} images)...")

        try:
            teacar = TEACar()
            with teacar as car_client:
                # Start the measurement system before the vehicle so the first
                # segment of the run is not silently missing from the CTE trace.
                cte_client = teacar.jump
                cte_cmd = (
                    f"cd {shlex.quote(CTE_PROJECT_DIR)} && mkdir -p data && "
                    f"{shlex.quote(CTE_PYTHON_PATH)} stream.py --output "
                    f"{shlex.quote(cte_rel_path)}"
                )
                print_progress(f"[EvalController] Executing ZED tracking stream on server: {cte_cmd}")
                cte_in, cte_out, cte_err = cte_client.exec_command(cte_cmd, get_pty=True)
                cte_out.channel.set_combine_stderr(True)

                # Do not start the car when the tracking process dies during
                # initialization. Without this gate, a run can consume data and
                # move the vehicle while producing no valid CTE trace.
                startup_deadline = time.monotonic() + CTE_STARTUP_SECONDS
                while time.monotonic() < startup_deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        cte_out.channel.close()
                        raise InterruptedError("Physical evaluation cancelled during CTE startup")
                    if cte_out.channel.exit_status_ready():
                        startup_output = cte_out.read().decode(errors="replace")
                        startup_status = cte_out.channel.recv_exit_status()
                        raise RuntimeError(
                            "CTE tracker exited before car launch "
                            f"(status={startup_status}): {startup_output[-300:]}"
                        )
                    time.sleep(0.1)

                remote_data_dir = f"{IROS_WS_DIR}/{data_path}"
                remote_image_dir = f"{remote_data_dir}/images"
                controller_remote = f"{CONTROLLER_TARGET_DIR}/{target_file}"
                cmd = (
                    f"mkdir -p {shlex.quote(remote_image_dir)} && "
                    f"cd {shlex.quote(IROS_WS_DIR)} && "
                    "source devel/setup.bash && "
                    "roslaunch iros_bringup self_evolve_paper.launch "
                    f"{shlex.quote('model_path:=' + controller_remote)} "
                    f"{shlex.quote('exit_threshold:=' + str(num_images))} "
                    f"{shlex.quote('data_folder:=' + remote_data_dir)} && "
                    f"test -d {shlex.quote(remote_data_dir)} && "
                    f"tar -czf {shlex.quote(remote_tar_path)} "
                    f"-C {shlex.quote(IROS_WS_DIR)} {shlex.quote(data_path)}"
                )
                print_progress(f"[EvalController] Executing ROS command on car: {cmd}")
                car_in, car_out, car_err = car_client.exec_command(cmd, get_pty=True)
                car_out.channel.set_combine_stderr(True)

                deadline = time.monotonic() + EVAL_TIMEOUT_SECONDS
                while not car_out.channel.exit_status_ready():
                    if cancel_event is not None and cancel_event.is_set():
                        car_out.channel.close()
                        raise InterruptedError("Physical evaluation cancelled")
                    if time.monotonic() >= deadline:
                        car_out.channel.close()
                        raise TimeoutError(
                            f"Physical evaluation exceeded {EVAL_TIMEOUT_SECONDS} seconds"
                        )
                    time.sleep(0.25)
                car_out_text = car_out.read().decode(errors="replace")
                car_exit_status = car_out.channel.recv_exit_status()
                if car_exit_status != 0:
                    raise RuntimeError(f"physical evaluation command exited with status {car_exit_status}")

                # Send Ctrl+C interrupt signal to ZED stream
                try:
                    cte_in.channel.send("\x03")
                except Exception:
                    pass
                cte_deadline = time.monotonic() + 15
                while not cte_out.channel.exit_status_ready() and time.monotonic() < cte_deadline:
                    time.sleep(0.1)
                if not cte_out.channel.exit_status_ready():
                    cte_out.channel.close()
                cte_out_text = cte_out.read().decode(errors="replace")

        except Exception as e:
            try:
                if "cte_in" in locals():
                    cte_in.channel.send("\x03")
            except Exception:
                pass
            deployment_run["status"] = "failed"
            deployment_run["error"] = str(e)
            deployment_run["failed_at"] = datetime.now().astimezone().isoformat()
            _save(workspace_dir, s, branch=branch)
            err_res = {
                "status": "failed",
                "deployment_run_id": deployment_run_id,
                "result": f"Failed to evaluate {target_file}: {e}",
                "stdout": "",
                "stderr": str(e)
            }
            return json.dumps(err_res, ensure_ascii=False)

        res = {
            "status": "success",
            "deployment_run_id": deployment_run_id,
            "collection_id": collection_id,
            "deployment": d,
            "controller_evaluated": target_file,
            "anonymous_source": anonymous_source,
            "n_images_target": num_images,
            "collection_images_budget_used": s["collection_images_budget_used"],
            "collection_images_budget_cap": image_budget_cap,
            "cte_mean": None,
            "cte_status": "pending_transfer_and_parse",
            "improvement_from_previous": None,
            "remote_data_path": remote_tar_path,
            "remote_tar_path": remote_tar_path,
            "remote_cte_path": remote_cte_path,
            "car_stdout_snippet": car_out_text[-300:] if car_out_text else "",
            "cte_stdout_snippet": cte_out_text[-300:] if cte_out_text else "",
            "result": f"Evaluation run finished. Transfer artifacts to parse real CTE and form optional N_t."
        }

        deployment_run["status"] = "completed"
        deployment_run["completed_at"] = datetime.now().astimezone().isoformat()

        s["last_eval_observation"] = res
        record_observation(s, "eval_controller", res, workspace_dir=workspace_dir, branch=branch)
        append_ledger(s, {
            "stage": "eval_controller",
            "round": s.get("round"),
            "deployment": d,
            "controller": target_file,
            "cte_mean": None,
            "remote_data_path": remote_tar_path,
            "remote_cte_path": remote_cte_path
        })

        _save(workspace_dir, s, branch=branch)

        return json.dumps(res, ensure_ascii=False)
