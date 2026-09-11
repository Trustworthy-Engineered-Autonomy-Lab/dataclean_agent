"""
ChatGPT Agent Trajectory Recorder
Records complete execution trace of ChatGPT agent through pipeline
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import _task_dir


class TrajectoryRecorder:
    """Records ChatGPT agent tool calls, results, and execution metadata"""

    def __init__(self, workspace_dir: str, task_name: str = "default"):
        self.workspace_dir = Path(workspace_dir)
        self.task_name = task_name
        self.traj_dir = _task_dir(workspace_dir, branch=task_name, create=True) / "agent_trajectories"
        self.traj_dir.mkdir(exist_ok=True)

        # Create trajectory file with timestamp
        self.timestamp = datetime.now().isoformat()
        self.turn_id = f"traj_{int(time.time() * 1000)}"
        self.traj_file = self.traj_dir / f"{self.turn_id}.json"

        # Initialize trajectory structure
        self.trajectory = {
            "metadata": {
                "turn_id": self.turn_id,
                "task_name": task_name,
                "start_time": self.timestamp,
                "llm_model": None,
                "temperature": None,
                "seed": None,
                "prompt_version": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "pipeline_stage": "init",
            "tool_calls": [],
            "state_changes": [],
            "errors": [],
            "summary": {
                "total_steps": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_seconds": 0,
                "final_state": "init",
                "total_retries": 0,
            }
        }
        self.start_time = time.time()
        self._retry_count_per_tool = {}
        self.save()

    def record_agent_config(self, model: str, temperature: Optional[float],
                           seed: Optional[int], prompt_version: str):
        """Record ChatGPT agent configuration"""
        self.trajectory["metadata"].update({
            "llm_model": model,
            "temperature": temperature,
            "seed": seed,
            "prompt_version": prompt_version,
        })
        self.save()

    def record_pipeline_stage(self, stage: str):
        """Record which pipeline stage agent is in"""
        self.trajectory["pipeline_stage"] = stage
        self.save()

    def record_tool_call(self, tool_name: str, tool_args: Dict[str, Any],
                        step_index: int, timestamp: Optional[str] = None) -> int:
        """Record a tool call by ChatGPT"""
        call_index = len(self.trajectory["tool_calls"])

        if tool_name not in self._retry_count_per_tool:
            self._retry_count_per_tool[tool_name] = 0
        self._retry_count_per_tool[tool_name] += 1

        call_timestamp = timestamp or datetime.now().isoformat()
        call_record = {
            "call_index": call_index,
            "tool_name": tool_name,
            "args": tool_args,
            "pipeline_step": step_index,
            "call_timestamp": call_timestamp,
            "call_time_ms": int(time.time() * 1000),
            "status": "pending",
            "result": None,
            "error": None,
            "duration_seconds": None,
            "latency_ms": None,
            "state_changed": False,
            "output_artifacts": [],
            "retry_count": self._retry_count_per_tool[tool_name],
            "decision_summary": None,
            "observation": None,
        }
        self.trajectory["tool_calls"].append(call_record)
        self.trajectory["summary"]["total_steps"] += 1
        self.save()
        return call_index

    def record_tool_result(self, call_index: int, result: Any,
                          success: bool = True, error: Optional[str] = None,
                          artifacts: Optional[List[str]] = None,
                          latency_ms: Optional[float] = None,
                          observation: Optional[str] = None,
                          decision_summary: Optional[str] = None):
        """Record the result of a tool call"""
        if call_index >= len(self.trajectory["tool_calls"]):
            return

        call_record = self.trajectory["tool_calls"][call_index]
        call_record["status"] = "succeeded" if success else "failed"
        call_record["result"] = self._serialize_result(result)
        call_record["error"] = error
        call_record["duration_seconds"] = time.time() - self.start_time

        if latency_ms is None:
            current_time_ms = int(time.time() * 1000)
            call_time_ms = call_record.get("call_time_ms", current_time_ms)
            latency_ms = current_time_ms - call_time_ms
        call_record["latency_ms"] = max(0, latency_ms)

        call_record["output_artifacts"] = artifacts or []
        call_record["observation"] = observation or (error if not success else None)
        call_record["decision_summary"] = decision_summary

        if success:
            self.trajectory["summary"]["successful_calls"] += 1
        else:
            self.trajectory["summary"]["failed_calls"] += 1
            self.trajectory["summary"]["total_retries"] += 1
            if error:
                self.trajectory["errors"].append({
                    "tool": call_record["tool_name"],
                    "call_index": call_index,
                    "error_message": error,
                    "timestamp": datetime.now().isoformat(),
                    "retry_count": call_record["retry_count"],
                })

        self.save()

    def record_token_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage from ChatGPT API"""
        self.trajectory["metadata"]["input_tokens"] += input_tokens
        self.trajectory["metadata"]["output_tokens"] += output_tokens
        self.trajectory["metadata"]["total_tokens"] += input_tokens + output_tokens
        self.save()

    def set_final_state(self, state: str):
        """Set final state of the trajectory"""
        assert state in ["completed", "wait", "aborted", "paused", "in_progress"]
        self.trajectory["summary"]["final_state"] = state
        self.save()

    def record_state_change(self, change_type: str, details: Dict[str, Any],
                           call_index: Optional[int] = None):
        """Record state changes"""
        change_record = {
            "change_type": change_type,
            "timestamp": datetime.now().isoformat(),
            "call_index": call_index,
            "details": details,
        }
        self.trajectory["state_changes"].append(change_record)

        if call_index is not None and call_index < len(self.trajectory["tool_calls"]):
            self.trajectory["tool_calls"][call_index]["state_changed"] = True

        self.save()

    def _serialize_result(self, result: Any) -> Any:
        """Serialize result to JSON-compatible format"""
        if isinstance(result, (str, int, float, bool, type(None))):
            return result
        if isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}
        if isinstance(result, (list, tuple)):
            return [self._serialize_result(item) for item in result]
        return str(result)

    def save(self):
        """Save trajectory to disk"""
        try:
            self.traj_file.write_text(
                json.dumps(self.trajectory, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            print(f"[Warning] Failed to save trajectory: {e}")

    def finalize(self):
        """Mark trajectory as complete"""
        self.trajectory["metadata"]["end_time"] = datetime.now().isoformat()
        self.trajectory["summary"]["final_state"] = "completed"
        self.trajectory["summary"]["total_duration_seconds"] = (
            time.time() - self.start_time
        )
        self.save()

    def get_trajectory_path(self) -> str:
        """Get path to trajectory file"""
        return str(self.traj_file)
