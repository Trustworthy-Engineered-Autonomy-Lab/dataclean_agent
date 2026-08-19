"""Standard-library-first deployment preflight for a lab server."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = (
    "fastapi", "uvicorn", "openai", "numpy", "scipy", "sklearn",
    "matplotlib", "PIL", "torch", "torchvision", "paramiko", "onnx",
)
REQUIRED_TOOLS = {
    "configure_dataset", "configure_task_dataset", "define_task",
    "get_pipeline_state", "evaluate", "train_detector",
    "score_and_fit", "partition", "resolve", "train_controller",
    "deploy_controller", "eval_controller", "transfer_eval_results",
    "commit_round", "assess_stopping",
}


def _module_status():
    status = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            status[name] = {"ok": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            status[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return status


def _workspace_status(path):
    if path is None:
        return None
    workspace = Path(path).expanduser().resolve()
    root = workspace / ".dataclean"
    tasks = root / "tasks"
    dataset = root / "dataset.json"
    return {
        "path": str(workspace),
        "exists": workspace.is_dir(),
        "writable": os.access(workspace, os.W_OK) if workspace.exists() else False,
        "dataset_registry": dataset.is_file(),
        "task_count": sum(1 for p in tasks.iterdir() if p.is_dir()) if tasks.is_dir() else 0,
    }


def _ssh_status():
    known_hosts = Path(os.environ.get(
        "DATACLEAN_KNOWN_HOSTS", "~/.ssh/known_hosts"
    )).expanduser()
    policy = os.environ.get(
        "DATACLEAN_SSH_HOST_KEY_POLICY", "accept-new"
    ).strip().lower()
    parent = known_hosts.parent
    jump_key = Path(os.environ.get(
        "DATACLEAN_JUMP_KEY", "~/.ssh/id_ed25519"
    )).expanduser()
    return {
        "host_key_policy": policy,
        "host_key_policy_valid": policy in {"accept-new", "strict"},
        "known_hosts_path": str(known_hosts),
        "known_hosts_exists": known_hosts.is_file(),
        "known_hosts_parent_writable": parent.is_dir() and os.access(parent, os.W_OK),
        "strict_ready": policy != "strict" or known_hosts.is_file(),
        "jump_host": os.environ.get("DATACLEAN_JUMP_HOST", "ece-d4100-w02.ad.ufl.edu"),
        "jump_user": os.environ.get("DATACLEAN_JUMP_USER", "jianangu"),
        "jump_key_path": str(jump_key),
        "jump_key_exists": jump_key.is_file(),
        "car_host": os.environ.get("DATACLEAN_CAR_HOST", "teacar2"),
        "car_user": os.environ.get("DATACLEAN_CAR_USER", "nvidia"),
        "car_password_environment_present": "DATACLEAN_CAR_PASSWORD" in os.environ,
    }


def diagnose(workspace=None):
    modules = _module_status()
    registered = set()
    registry_error = None
    if all(modules[name]["ok"] for name in ("numpy", "PIL")):
        try:
            from tools import Tool
            registered = set(Tool._registry)
        except Exception as exc:
            registry_error = f"{type(exc).__name__}: {exc}"
    missing_tools = sorted(REQUIRED_TOOLS - registered)
    missing_modules = sorted(name for name, row in modules.items() if not row["ok"])
    python_ok = sys.version_info >= (3, 9)
    report = {
        "ok": python_ok and not missing_modules and not missing_tools and registry_error is None,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_supported": python_ok,
        "platform": platform.platform(),
        "modules": modules,
        "tool_registry_error": registry_error,
        "missing_required_tools": missing_tools,
        "workspace": _workspace_status(workspace),
        "ssh": _ssh_status(),
        "configured_environment": sorted(
            key for key in os.environ
            if key.startswith("DATACLEAN_") and not key.endswith(("KEY", "PASSWORD"))
        ),
        "secret_environment_present": sorted(
            key for key in os.environ
            if key.startswith("DATACLEAN_") and key.endswith(("KEY", "PASSWORD"))
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="Optional lab workspace to inspect read-only")
    args = parser.parse_args()
    report = diagnose(args.workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
