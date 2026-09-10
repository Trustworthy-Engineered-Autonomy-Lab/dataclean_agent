"""
Trajectory Analysis and Query Utilities
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class TrajectoryAnalyzer:
    """Analyze and query recorded trajectories"""

    def __init__(self, traj_file: str):
        """Load trajectory from file"""
        self.traj_file = Path(traj_file)
        with open(self.traj_file) as f:
            self.trajectory = json.load(f)

    @staticmethod
    def load_from_workspace(workspace_dir: str, task_id: str) -> Optional['TrajectoryAnalyzer']:
        """Load the most recent trajectory for one task"""
        from .io import _task_dir
        traj_dir = _task_dir(workspace_dir, branch=task_id, create=False) / "agent_trajectories"
        if not traj_dir.exists():
            return None

        traj_files = sorted(traj_dir.glob("*.json"), reverse=True)
        if not traj_files:
            return None

        return TrajectoryAnalyzer(str(traj_files[0]))

    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary"""
        return {
            "turn_id": self.trajectory["metadata"]["turn_id"],
            "model": self.trajectory["metadata"]["llm_model"],
            "start_time": self.trajectory["metadata"]["start_time"],
            "end_time": self.trajectory["metadata"].get("end_time"),
            "final_state": self.trajectory["summary"].get("final_state", "unknown"),
            "total_steps": self.trajectory["summary"]["total_steps"],
            "successful_calls": self.trajectory["summary"]["successful_calls"],
            "failed_calls": self.trajectory["summary"]["failed_calls"],
            "duration_seconds": self.trajectory["summary"]["total_duration_seconds"],
            "input_tokens": self.trajectory["metadata"].get("input_tokens", 0),
            "output_tokens": self.trajectory["metadata"].get("output_tokens", 0),
            "total_tokens": self.trajectory["metadata"].get("total_tokens", 0),
            "total_retries": self.trajectory["summary"].get("total_retries", 0),
        }

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls in sequence"""
        return self.trajectory["tool_calls"]

    def get_tool_call_by_step(self, step_index: int) -> Optional[Dict[str, Any]]:
        """Get tool call for a specific pipeline step"""
        for call in self.trajectory["tool_calls"]:
            if call["pipeline_step"] == step_index:
                return call
        return None

    def get_state_changes(self) -> List[Dict[str, Any]]:
        """Get all state changes"""
        return self.trajectory["state_changes"]

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors"""
        return self.trajectory["errors"]

    def get_artifacts(self) -> List[str]:
        """Get all generated artifacts"""
        artifacts = []
        for call in self.trajectory["tool_calls"]:
            artifacts.extend(call["output_artifacts"])
        return artifacts

    def print_summary(self):
        """Print human-readable summary"""
        summary = self.get_summary()
        print("\n" + "="*70)
        print("TRAJECTORY SUMMARY")
        print("="*70)
        print(f"Turn ID:          {summary['turn_id']}")
        print(f"Model:            {summary['model']}")
        print(f"Start Time:       {summary['start_time']}")
        print(f"End Time:         {summary['end_time']}")
        print(f"Total Duration:   {summary['duration_seconds']:.2f}s")
        print(f"Final State:      {summary['final_state']}")
        print(f"\nExecutions:")
        print(f"  Total Steps:      {summary['total_steps']}")
        print(f"  Successful:       {summary['successful_calls']}")
        print(f"  Failed:           {summary['failed_calls']}")
        success_rate = (summary['successful_calls'] / summary['total_steps'] * 100
                       if summary['total_steps'] > 0 else 0)
        print(f"  Success Rate:     {success_rate:.1f}%")
        print(f"\nToken Usage:")
        print(f"  Input:            {summary['input_tokens']}")
        print(f"  Output:           {summary['output_tokens']}")
        print(f"  Total:            {summary['total_tokens']}")
        print(f"  Retries:          {summary['total_retries']}")

    def print_pipeline_trace(self):
        """Print pipeline execution trace"""
        print("\n" + "="*70)
        print("PIPELINE EXECUTION TRACE")
        print("="*70)

        for call in self.trajectory["tool_calls"]:
            status = "✓" if call["status"] == "succeeded" else "✗"
            latency = call.get("latency_ms", 0)
            print(f"\n[{status}] Step {call['pipeline_step']}: {call['tool_name']}")
            print(f"    Status: {call['status']}")
            print(f"    Latency: {latency}ms")
            print(f"    Retry: {call.get('retry_count', 1)}")
            if call["status"] == "failed" and call["error"]:
                print(f"    Error: {call['error']}")
            if call.get('output_artifacts'):
                print(f"    Artifacts: {', '.join(call['output_artifacts'])}")
