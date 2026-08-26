"""Tool registry.

Core dataset/state tools stay importable for configuration tests even when optional
ML, VLM, plotting, or physical-car dependencies are not installed. A production
installation from requirements.txt registers the complete tool set.
"""

from importlib import import_module

from .base import Tool
from .configure_dataset import ConfigureDataset
from .configure_task_dataset import ConfigureTaskDataset
from .pipeline_state import PipelineState
from .agent_protocol import (
    AssessExperimentEpisode,
    ProposeExperimentEpisode,
    ReconcileInterruptedAction,
    WithdrawExperimentEpisode,
)
from .define_task import DefineTask
from .list_tasks import ListTasks
from .partition import Partition
from .commit_round import CommitRound
from .assess_stopping import AssessStopping

_OPTIONAL_IMPORT_ERRORS = {}


def _optional(label, importer):
    try:
        return importer()
    except Exception as exc:
        # A binary/version mismatch (notably torchvision vs torch) often raises
        # RuntimeError rather than ImportError. Keep dataset/task management
        # available and surface the exact unavailable production action in
        # /api/health and doctor.py instead of crashing server startup.
        _OPTIONAL_IMPORT_ERRORS[label] = f"{type(exc).__name__}: {exc}"
        return None


TrainDetector = _optional("train_detector", lambda: import_module(".train_detector", __name__).TrainDetector)
ScoreAndFit = _optional("score_and_fit", lambda: import_module(".score_and_fit", __name__).ScoreAndFit)
Resolve = _optional("resolve", lambda: import_module(".resolve", __name__).Resolve)
Evaluate = _optional("evaluate", lambda: import_module(".evaluate", __name__).Evaluate)
TrainController = _optional("train_controller", lambda: import_module(".train_controller", __name__).TrainController)
DeployController = _optional("deploy_controller", lambda: import_module(".deploy_controller", __name__).DeployController)
EvalController = _optional("eval_controller", lambda: import_module(".eval_controller", __name__).EvalController)
TransferEvalResults = _optional("transfer_eval_results", lambda: import_module(".transfer_eval_results", __name__).TransferEvalResults)


def optional_dependency_errors():
    return dict(_OPTIONAL_IMPORT_ERRORS)


__all__ = [
    "Tool", "ConfigureDataset", "ConfigureTaskDataset", "PipelineState", "DefineTask", "ListTasks",
    "Partition", "CommitRound", "AssessStopping",
    "ProposeExperimentEpisode", "AssessExperimentEpisode", "ReconcileInterruptedAction",
    "WithdrawExperimentEpisode",
    "optional_dependency_errors",
]
__all__ += [name for name in (
    "TrainDetector", "ScoreAndFit", "Resolve", "Evaluate", "TrainController",
    "DeployController", "EvalController", "TransferEvalResults",
) if globals().get(name) is not None]
