"""Unlabeled round report: plot every D_t score, count retention using actual C_t."""

import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import Tool
from .dataset import _anonymize_source_name, _load_dataset_snapshot
from .detector_contract import normality_scores, score_contract
from .io import (
    _load, _save, _artifact, _task_artifact_reference, _write_json_atomic,
    record_observation,
)


REPORT_KIND = "pcc_retention_report"


def _record_identity(record):
    return (record["image"], float(record["steering"]), _anonymize_source_name(record["source"]))


class Evaluate(Tool):
    name = "evaluate"
    description = (
        "Generate an unlabeled data report after C_t is resolved: plot PCC for ALL "
        "pre-partition D_t samples and count each anonymous source's retained/removed "
        "samples by actual membership in controller input C_t (including VLM accepts). "
        "Does not rescore, split, train, advance rounds, or read ground-truth labels. "
        "Removed means not in C_t, not confirmed anomalous. Optional round_index selects a completed past round."
    )
    parameters = {
        "type": "object",
        "properties": {
            "round_index": {"type": "integer", "minimum": 0,
                            "description": "Round to report; omit for the current round."},
        },
        "required": [],
    }

    def _inputs(self, state, round_index, workspace_dir, branch):
        if round_index == int(state.get("round", 0)):
            input_ref = state.get("round_input_dataset")
            input_fingerprint = state.get("round_input_fingerprint")
            clean_ref = state.get("active_clean_dataset")
            scores_ref = state.get("latest_scores")
        else:
            matches = [entry for entry in state.get("round_history", []) if entry.get("round") == round_index]
            if len(matches) != 1:
                raise ValueError("Requested round has no unique completed-round record")
            history = matches[0]
            input_ref = history.get("input_dataset")
            input_fingerprint = history.get("input_fingerprint")
            clean_ref = history.get("clean_dataset")
            scores_ref = history.get("scores_artifact") or (
                (history.get("observations") or {}).get("score_and_fit") or {}
            ).get("score_artifact")
        if not input_ref or not clean_ref:
            raise ValueError("This round has no final C_t yet. Resolve clean_data before reporting retention.")

        input_path = _task_artifact_reference(workspace_dir, branch, input_ref)
        clean_path = _task_artifact_reference(workspace_dir, branch, clean_ref)
        input_data = _load_dataset_snapshot(input_path)
        clean_data = _load_dataset_snapshot(clean_path)
        for payload, role in ((input_data, "round_input"), (clean_data, "clean")):
            if payload["task_id"] != branch or payload["role"] != role or payload["round"] != round_index:
                raise ValueError("Report dataset task/role/round does not match the requested round")
        if input_data["fingerprint"] != input_fingerprint:
            raise ValueError("Report D_t fingerprint does not match round state")
        metadata = clean_data.get("metadata") or {}
        if metadata.get("round_input_fingerprint", input_fingerprint) != input_fingerprint:
            raise ValueError("C_t does not descend from the requested D_t")
        scores_ref = metadata.get("scores_artifact") or scores_ref
        if not scores_ref:
            raise ValueError("No exact PCC artifact is recorded for this clean dataset; cannot guess scores")
        scores_path = _task_artifact_reference(workspace_dir, branch, scores_ref)
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        normality_scores(scores)

        input_map = {r["id"]: r for r in input_data["records"]}
        score_map = {r["id"]: r for r in scores}
        clean_ids = set(clean_data["ids"])
        if len(score_map) != len(scores) or set(score_map) != set(input_map):
            raise ValueError("PCC scores must cover D_t exactly once; missing/extra/duplicate samples found")
        if not clean_ids <= set(input_map):
            raise ValueError("C_t contains samples outside the requested D_t")
        for record in scores + clean_data["records"]:
            if _record_identity(record) != _record_identity(input_map[record["id"]]):
                raise ValueError("Source/image/steering identity differs between D_t, PCC scores and C_t")
        ordered = [score_map[r["id"]] for r in input_data["records"]]
        return input_path, clean_path, scores_path, input_data, clean_data, ordered

    @staticmethod
    def _source_counts(records, clean_ids):
        input_counts = Counter(_anonymize_source_name(r["source"]) for r in records)
        kept_counts = Counter(_anonymize_source_name(r["source"]) for r in records if r["id"] in clean_ids)
        return [{
            "source": source, "input_count": count, "kept_count": kept_counts[source],
            "removed_count": count - kept_counts[source],
            "retention_rate": kept_counts[source] / count,
        } for source, count in sorted(input_counts.items())]

    @staticmethod
    def _plot(records, clean_ids, round_index, output):
        # Stable sorting retains input order within each anonymous source.
        ordered = sorted(records, key=lambda r: _anonymize_source_name(r["source"]))
        scores = np.array(normality_scores(ordered))
        retained = np.array([r["id"] in clean_ids for r in ordered], dtype=bool)
        x = np.arange(len(ordered))
        fig, ax = plt.subplots(figsize=(14, 6))
        try:
            ax.scatter(x[~retained], scores[~retained], s=5, alpha=.55, color="#a6adb8",
                       label="Not in final C_t (not an anomaly label)")
            ax.scatter(x[retained], scores[retained], s=5, alpha=.65, color="#2477b8",
                       label="Retained in final C_t (including VLM accepts)")
            counts = Counter(_anonymize_source_name(r["source"]) for r in ordered)
            start = 0
            centers = []
            for count in counts.values():
                if start:
                    ax.axvline(start - .5, color="#d4d8df", linewidth=.7)
                centers.append(start + (count - 1) / 2)
                start += count
            rotated = len(counts) > 5 or any(len(name) > 8 for name in counts)
            ax.set_xticks(centers, list(counts), rotation=60 if rotated else 0,
                          ha="right" if rotated else "center", fontsize=8)
            ax.set_xlim(-.5, len(ordered) - .5)
            ax.set_ylim(-1.02, 1.02)
            ax.set_xlabel("All pre-partition D_t samples, grouped by anonymous source")
            ax.set_ylabel("PCC (higher = more normal)")
            ax.set_title(f"Round {round_index}: full D_t PCC distribution (n={len(ordered)})")
            ax.grid(axis="y", alpha=.2)
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            fig.savefig(output, dpi=150)
        finally:
            plt.close(fig)

    def run(self, round_index=None, branch="main", workspace_dir=None, **_):
        state = _load(workspace_dir, branch=branch)
        if round_index is None:
            round_index = int(state.get("round", 0))
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        input_path, clean_path, scores_path, input_data, clean_data, records = self._inputs(
            state, round_index, workspace_dir, branch,
        )
        clean_ids = set(clean_data["ids"])
        plot_path = _artifact(workspace_dir, f"pcc_distribution_r{round_index}.png", branch)
        report_path = _artifact(workspace_dir, f"data_report_r{round_index}.json", branch)
        report = {
            "report_kind": REPORT_KIND, "schema_version": 1, "round": round_index,
            "score_contract": score_contract(),
            "input_count": len(records), "kept_count": len(clean_ids),
            "removed_count": len(records) - len(clean_ids),
            "sources": self._source_counts(records, clean_ids),
            "input_fingerprint": input_data["fingerprint"],
            "clean_fingerprint": clean_data["fingerprint"],
            "input_artifact": input_path.name, "clean_artifact": clean_path.name,
            "scores_artifact": scores_path.name,
            "plot_scope": "all pre-partition D_t samples",
            "retention_basis": "actual final C_t membership, before controller train/validation split",
            "removed_definition": "D_t minus C_t; includes quarantined/unreviewed samples, not confirmed anomalies",
            "artifacts": {"report_json": report_path.name, "pcc_distribution": plot_path.name},
        }
        self._plot(records, clean_ids, round_index, plot_path)
        _write_json_atomic(report_path, report)
        # Do not attach a historical report to the current round's observations.
        if round_index == int(state.get("round", 0)):
            record_observation(state, "evaluate", report, workspace_dir=workspace_dir, branch=branch)
        else:
            for entry in state.get("round_history", []):
                if entry.get("round") == round_index:
                    entry.setdefault("observations", {})["evaluate"] = report
        _save(workspace_dir, state, branch=branch)
        return json.dumps(report, ensure_ascii=False)
