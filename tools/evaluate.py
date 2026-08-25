import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import median_abs_deviation
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .base import Tool
from .utils import (_load, _save, _artifact, _combined_score, _quantiles,
                   _threshold, record_observation, append_ledger, ROOT,
                   _dataset_config, _write_json_atomic)
from .io import _task_artifact_reference

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
})


def _fmt(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return f"{float(v):{fmt}}"


class Evaluate(Tool):
    name = "evaluate"
    agent_exposed = False
    description = (
        "Open-loop evaluation logger: generates metrics report (.txt), per-source filter details (.txt), and anomaly score scatter plot (.png). "
        "Calculates labeled metrics (AUC, P/R/F1) for human review without exposing ground-truth labels to agent decision logic."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["full", "keep", "cleandata"],
                "description": "Evaluation target: 'full' (all scored samples), 'keep' (detector-retained region), or 'cleandata' (final clean dataset)."
            },
            "threshold_mode": {
                "type": "string",
                "enum": ["partition", "mean_std"],
                "description": "Threshold mode for metrics calculation: 'partition' (uses applied partition threshold) or 'mean_std' (recalculated mean+std)."
            },
            "detector_id": {
                "type": "string",
                "description": "Optional specific detector score artifact ID to evaluate. Defaults to state.latest_scores."
            },
        },
        "required": ["target"]
    }

    # ---- helpers ----------------------------------------------------------

    def _load_scores(self, workspace_dir, branch, detector_id, s):
        if detector_id:
            artifact_dir = _artifact(workspace_dir, "placeholder", branch=branch).parent
            requested = Path(str(detector_id)).name
            if requested != str(detector_id):
                raise ValueError("detector_id must be an artifact filename or detector ID")
            direct = artifact_dir / requested
            matches = [direct] if direct.is_file() else list(
                artifact_dir.glob(f"scores_*_{requested}.json")
            )
            matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise ValueError(
                    f"detector_id must identify exactly one score artifact; found {len(matches)}"
                )
            path = matches[0]
        else:
            ref = s.get("latest_scores")
            path = (
                _task_artifact_reference(workspace_dir, branch, ref)
                if ref else None
            )
            if path is None:
                raise ValueError(
                    "The active round has no exact score artifact; run score_and_fit or pass detector_id"
                )
        if not path.exists():
            raise FileNotFoundError(f"Scores file does not exist: {path}")
        try:
            scores = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to parse scores file {path.name}: {e}") from e
        return path, scores

    def _target_records(self, scored_map, s, target, branch, workspace_dir):
        if target == "full":
            return list(scored_map.values()), "full (all scored)"
        if target == "keep":
            part = s.get("latest_partition")
            if not part or part.get("keep_ids") is None:
                raise ValueError("target=keep requires partition result (run partition first).")
            keep_ids = part["keep_ids"]
            return [scored_map[i] for i in keep_ids if i in scored_map], "keep (detector-retained)"
        if target == "cleandata":
            cpath = s.get("active_clean_dataset")
            if not cpath:
                raise ValueError("target=cleandata requires active_clean_dataset (run resolve first).")
            cp = Path(cpath)
            cp = _task_artifact_reference(workspace_dir, branch, cp)
            try:
                data = json.loads(cp.read_text())
            except (json.JSONDecodeError, OSError) as e:
                raise ValueError(f"Failed to parse clean dataset {cp.name}: {e}") from e
            ids = data.get("ids") or [r["id"] for r in data.get("records", [])]
            return [scored_map[i] for i in ids if i in scored_map], "cleandata (final)"
        raise ValueError(f"Unknown target: {target}")

    def _threshold(self, scores, mode, part_threshold, part_method=None):
        if mode == "mean_std":
            thr = _threshold(scores, k=1.0)
            return thr, f"Mean + Std ({thr:.4f})"
        if part_threshold is not None:
            label = part_method or "partition"
            return float(part_threshold), f"{label} ({part_threshold:.4f})"
        thr = _threshold(scores, k=1.0)
        return thr, f"Mean + Std ({thr:.4f})"

    def _ground_truth(self, workspace_dir, recs):
        labels_path = Path(workspace_dir) / ROOT / "evaluation_labels.json"
        label_map = {}
        provenance = None
        if labels_path.exists():
            payload = json.loads(labels_path.read_text())
            raw = payload.get("labels", payload)
            if not isinstance(raw, dict):
                raise ValueError("evaluation_labels.json must contain a sample_id -> label mapping")
            for sample_id, value in raw.items():
                if value in (0, "0", False, "normal"):
                    label_map[str(sample_id)] = 0
                elif value in (1, "1", True, "anomalous", "anomaly"):
                    label_map[str(sample_id)] = 1
            provenance = "sample_level:evaluation_labels.json"
        else:
            reg = _dataset_config(workspace_dir)
            source_labels = {}
            for source in reg.get("sources", []):
                value = source.get("evaluation_label")
                if value in (0, "0", False, "normal"):
                    source_labels[source["name"]] = 0
                elif value in (1, "1", True, "anomalous", "anomaly"):
                    source_labels[source["name"]] = 1
            if source_labels:
                label_map = {r["id"]: source_labels[r["source"]] for r in recs if r["source"] in source_labels}
                provenance = "source_level:manifest (coarse)"
        labels = np.array([label_map.get(r["id"], -1) for r in recs], dtype=int)
        return labels, provenance or "unlabeled"

    def _compute(self, recs, label, threshold, threshold_label):
        if not recs:
            raise ValueError("Evaluation target is empty")
        ids = [r["id"] for r in recs]
        src = np.array([r["source"] for r in recs])
        score = np.array([float(r["anomaly_score"]) for r in recs], float)
        pcc = np.array([float(r.get("pcc", 0.0)) for r in recs], float)
        serr = np.array([float(r.get("steer_error", 0.0)) for r in recs], float)
        known = label >= 0
        known_label = label[known]
        known_score = score[known]
        has_both = len(known_label) > 0 and known_label.min() != known_label.max()
        res = {"n_samples": len(recs), "n_labeled": int(known.sum())}

        if has_both:
            res["auc"] = round(float(roc_auc_score(known_label, known_score)), 5)
        else:
            res["auc"] = None

        pred = (score >= threshold).astype(int)
        if has_both:
            res["metrics_at_threshold"] = {
                "threshold": round(float(threshold), 5),
                "threshold_label": threshold_label,
                "precision": round(float(precision_score(known_label, pred[known], zero_division=0)), 5),
                "recall": round(float(recall_score(known_label, pred[known], zero_division=0)), 5),
                "f1": round(float(f1_score(known_label, pred[known], zero_division=0)), 5),
            }
            tn, fp, fn, tp = confusion_matrix(known_label, pred[known], labels=[0, 1]).ravel()
            res["metrics_at_threshold"].update({"tp": int(tp), "fp": int(fp),
                                                  "tn": int(tn), "fn": int(fn)})
        else:
            res["metrics_at_threshold"] = {"note": "Target contains only a single class; P/R/F1 not applicable"}

        res["score_stats"] = {
            **_quantiles(score),
            "mad": round(float(median_abs_deviation(score)), 5),
        }

        res["retention_purity"] = round(float((known_label == 0).mean()), 5) if len(known_label) else None

        if has_both and np.any(serr > 0):
            sweep = {}
            for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                c = _combined_score(pcc, serr, a)
                sweep[round(a, 1)] = round(float(roc_auc_score(known_label, c[known])), 5)
            res["alpha_sweep_auc"] = sweep
            raw = _combined_score(pcc, serr, 0.5)
            wsweep = {}
            for w in [1, 3, 5, 10, 20, 50]:
                filtered = ndi.median_filter(raw, size=w) if w > 1 else raw
                wsweep[w] = round(float(roc_auc_score(known_label, filtered[known])), 5)
            res["window_sweep_auc"] = wsweep

        return res, label, score, pcc, serr, src

    def _plot_scatter(self, recs, score, label, threshold, threshold_label,
                      title, out, alpha):
        src_names = [r["source"] for r in recs]
        order = sorted(range(len(recs)), key=lambda i: src_names[i])

        ordered_score = score[order]
        ordered_label = label[order]
        ordered_src = [src_names[i] for i in order]
        idx = np.arange(len(ordered_score))

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.scatter(idx, ordered_score, s=2, alpha=0.5, color="tab:blue",
                   label=f"Raw Combined ($\\alpha$={alpha:g})")

        win = min(51, max(3, len(ordered_score) // 100))
        if len(ordered_score) > win:
            ax.plot(idx, ndi.median_filter(ordered_score, size=win),
                    color="tab:red", linewidth=1.0, label="Median Filtered")

        ax.axhline(y=threshold, color="green", linestyle=":",
                   label=f"Threshold: {threshold_label}")

        boundaries = []
        prev = None
        for i, sn in enumerate(ordered_src):
            if sn != prev and prev is not None:
                boundaries.append(i)
            prev = sn
        for b in boundaries:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Anomaly Score (higher = more anomalous)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # ---- text outputs (reference-aligned) ----------------------------------

    def _write_metrics_txt(self, res, target_label, threshold, threshold_label, out, alpha):
        """Write metrics_result_*.txt aligned to user's reference."""
        lines = []
        lines.append(f"Anomaly Detection Metrics \u2014 {target_label}")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Score type: Combined (\u03b1={alpha:g})")
        lines.append(f"AUC: {_fmt(res.get('auc'))}")
        lines.append(f"Threshold: {_fmt(threshold)} ({threshold_label})")
        lines.append("")
        mt = res.get("metrics_at_threshold", {})
        if "precision" in mt:
            lines.append(f"Precision: {_fmt(mt['precision'])}")
            lines.append(f"Recall:    {_fmt(mt['recall'])}")
            lines.append(f"F1 Score:  {_fmt(mt['f1'])}")
            lines.append("")
            lines.append(
                f"TP: {mt['tp']}  FP: {mt['fp']}  "
                f"TN: {mt['tn']}  FN: {mt['fn']}")
            cm = mt.get("confusion_matrix")
            if cm:
                lines.append("")
                lines.append(f"Confusion matrix (on full set, retain decision):")
                lines.append(f"  normal kept   : {cm.get('normal_kept')}")
                lines.append(f"  normal removed: {cm.get('normal_removed')}")
                lines.append(f"  anomaly kept  : {cm.get('anomaly_kept')}")
                lines.append(f"  anomaly removed: {cm.get('anomaly_removed')}")
                if mt.get("convention"):
                    lines.append(f"  ({mt['convention']})")
        else:
            lines.append(mt.get("note", "N/A"))
        lines.append("")
        cm = res.get("cleaning_metrics")
        if cm:
            lines.append("[Cleaning metrics (vs full scored set)]")
            lines.append(f"  retention_purity : {_fmt(cm.get('retention_purity'))}")
            lines.append(f"  normal_retention : {_fmt(cm.get('normal_retention'))}  (kept normal / total normal)")
            lines.append(f"  anomaly_leak     : {_fmt(cm.get('anomaly_leak'))}  (kept anomaly / total anomaly)")
            lines.append(f"  target  normal={cm.get('target_normal')}  anomaly={cm.get('target_anomaly')}")
            lines.append(f"  full    normal={cm.get('total_normal')}  anomaly={cm.get('total_anomaly')}")
            lines.append("")
        # Alpha sweep
        asw = res.get("alpha_sweep_auc", {})
        if asw:
            lines.append("[Combined AUC sweep over \u03b1]")
            for a in sorted(asw):
                lines.append(f"  \u03b1={a:.1f} : AUC = {_fmt(asw[a])}")
            lines.append("")
        wsw = res.get("window_sweep_auc", {})
        if wsw:
            lines.append("[Median filter window sweep (\u03b1=0.5)]")
            for w in sorted(wsw):
                lines.append(f"  window={w:<5} : AUC = {_fmt(wsw[w])}")
        lines.append("=" * 50)
        out.write_text("\n".join(lines), encoding="utf-8")

    def _write_filter_txt(self, recs, score, threshold, target_label,
                           threshold_label, ss, out, alpha):
        by_src = {}
        for r in recs:
            by_src.setdefault(r["source"], []).append(r)

        lines = []
        lines.append(f"Model: {target_label}")
        lines.append(f"Score type: Combined (\u03b1={alpha:g})")
        st = ss.get("score_stats", {})
        lines.append(
            f"Score stats -> Mean: {_fmt(st.get('mean'))}, "
            f"Std: {_fmt(st.get('std'))}, "
            f"Median: {_fmt(st.get('median'))}, "
            f"MAD: {_fmt(st.get('mad'))}")
        lines.append(f"Threshold: {_fmt(threshold)} ({threshold_label})")
        lines.append("=" * 50)
        lines.append("")

        for src_name in sorted(by_src.keys()):
            src_recs = by_src[src_name]
            total = len(src_recs)
            passed = [r for r in src_recs if float(r["anomaly_score"]) < threshold]
            rate = len(passed) / total * 100 if total > 0 else 0.0
            pidx = sorted(int(r["id"].split(":")[1]) for r in passed)

            lines.append(f"--- {src_name} ---")
            lines.append(f"Total: {total}, Passed: {len(passed)}, Rate: {rate:.2f}%")
            lines.append(f"Passed indices: {self._format_indices(pidx)}")
            lines.append("-" * 28)
            lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _format_indices(idxs):
        if not idxs:
            return "[]"
        if len(idxs) <= 30:
            return str(idxs)
        head = ", ".join(str(i) for i in idxs[:15])
        tail = ", ".join(str(i) for i in idxs[-15:])
        return f"[{head}, ..., {tail}]"

    def run(self, target="full", threshold_mode="partition",
            detector_id=None, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        alpha = float(s.get("score_alpha", 0.5))
        spath, scored = self._load_scores(workspace_dir, branch, detector_id, s)
        scored_map = {r["id"]: r for r in scored}
        recs, target_label = self._target_records(scored_map, s, target, branch, workspace_dir)
        target_label_en = {"full": "Full Scored Set",
                           "keep": "Detector Keep Region (keep)",
                           "cleandata": "Final Clean Set (cleandata)"}[target]

        part = s.get("latest_partition") or {}
        part_threshold = part.get("threshold")
        threshold, threshold_label = self._threshold(
            [r["anomaly_score"] for r in recs], threshold_mode, part_threshold,
            part.get("threshold_method"))

        label, gt_provenance = self._ground_truth(workspace_dir, recs)
        res, label, score, pcc, serr, src = self._compute(
            recs, label, threshold, threshold_label)

        if target in ("keep", "cleandata"):
            target_ids = {r["id"] for r in recs}
            full_src_arr = np.array([r["source"] for r in scored])
            full_score = np.array([float(r["anomaly_score"]) for r in scored])
            full_label, _ = self._ground_truth(workspace_dir, scored)
            known = full_label >= 0
            full_pred = np.array([0 if r["id"] in target_ids else 1 for r in scored])
            total_normal = int((full_label[known] == 0).sum())
            total_anomaly = int((full_label[known] == 1).sum())
            tgt_normal = int((label == 0).sum())
            tgt_anomaly = int((label == 1).sum())
            has_both = known.any() and full_label[known].min() != full_label[known].max()
            res["auc"] = (round(float(roc_auc_score(full_label[known], full_score[known])), 5)
                          if has_both else None)
            if has_both:
                cm = confusion_matrix(full_label[known], full_pred[known], labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                res["metrics_at_threshold"] = {
                    "threshold": round(float(threshold), 5),
                    "threshold_label": threshold_label,
                    "precision": round(float(precision_score(full_label[known], full_pred[known], zero_division=0)), 5),
                    "recall": round(float(recall_score(full_label[known], full_pred[known], zero_division=0)), 5),
                    "f1": round(float(f1_score(full_label[known], full_pred[known], zero_division=0)), 5),
                    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                    "confusion_matrix": {
                        "normal_kept": int(tn), "normal_removed": int(fp),
                        "anomaly_kept": int(fn), "anomaly_removed": int(tp),
                    },
                    "convention": "Calculated on full set per target retention decision; positive=anomaly (to be removed), pred_positive=removed",
                }
            else:
                res["metrics_at_threshold"] = {"note": "No usable held-out binary labels"}
            res["cleaning_metrics"] = {
                "retention_purity": res.get("retention_purity"),
                "normal_retention": round(tgt_normal / total_normal, 5) if total_normal else None,
                "anomaly_leak": round(tgt_anomaly / total_anomaly, 5) if total_anomaly else None,
                "target_normal": tgt_normal, "target_anomaly": tgt_anomaly,
                "total_normal": total_normal, "total_anomaly": total_anomaly,
            }

        if target == "cleandata" and part.get("keep_ids") is not None:
            keep_ids = part["keep_ids"]
            full_labels_for_map, _ = self._ground_truth(workspace_dir, scored)
            label_by_id = {r["id"]: int(full_labels_for_map[i]) for i, r in enumerate(scored)}
            keep_labels = [label_by_id[i] for i in keep_ids if label_by_id.get(i, -1) >= 0]
            if keep_labels and res.get("retention_purity") is not None:
                keep_purity = float(np.mean([1 if l == 0 else 0 for l in keep_labels]))
                res["purity_vs_detector_keep"] = round(
                    float(res["retention_purity"] - keep_purity), 5)
                res["detector_keep_purity"] = round(keep_purity, 5)

        prefix = f"eval_r{s.get('round', 0)}_{target}"

        metrics_json = _artifact(workspace_dir, f"{prefix}_metrics.json", branch=branch)
        _write_json_atomic(metrics_json, {
            "target": target, "target_label": target_label,
            "scores_artifact": spath.name, "ground_truth_provenance": gt_provenance,
            "score_alpha": alpha,
            "threshold_mode": threshold_mode, "n_samples": res["n_samples"],
            **res,
        })

        metrics_txt = _artifact(workspace_dir, f"{prefix}_metrics.txt", branch=branch)
        self._write_metrics_txt(res, target_label, threshold, threshold_label, metrics_txt, alpha)

        filter_txt = _artifact(workspace_dir, f"{prefix}_filter.txt", branch=branch)
        all_score = np.array([float(r["anomaly_score"]) for r in scored])
        full_stats = {"score_stats": {**_quantiles(all_score),
                                      "mad": round(float(median_abs_deviation(all_score)), 5)}}
        self._write_filter_txt(scored, all_score, threshold, "full scored set",
                                threshold_label, full_stats, filter_txt, alpha)

        scatter = _artifact(workspace_dir, f"{prefix}_score_scatter.png", branch=branch)
        full_src_arr2 = np.array([r["source"] for r in scored])
        full_score = np.array([float(r["anomaly_score"]) for r in scored])
        full_label, _ = self._ground_truth(workspace_dir, scored)
        known = full_label >= 0
        if known.any() and full_label[known].min() != full_label[known].max():
            full_auc = round(float(roc_auc_score(full_label[known], full_score[known])), 5)
        else:
            full_auc = None
        plot_title = (f"Anomaly Score — full distribution"
                      f" (target={target_label}, threshold={threshold_label}, full AUC={_fmt(full_auc)})")
        self._plot_scatter(scored, full_score, full_label, threshold, threshold_label,
                           plot_title, scatter, alpha)

        summary = {
            "target": target,
            "target_label": target_label_en,
            "n_samples": res["n_samples"],
            "threshold_label": threshold_label,
            "artifacts": {
                "metrics_json": metrics_json.name,
                "metrics_txt": metrics_txt.name,
                "filter_txt": filter_txt.name,
                "score_scatter": scatter.name,
            },
        }
        record_observation(s, "evaluate", summary, workspace_dir=workspace_dir, branch=branch)
        append_ledger(s, {
            "stage": "evaluate",
            "round": s.get("round"),
            "target": target,
            "threshold": round(threshold, 5),
            "n_samples": res["n_samples"],
        })
        _save(workspace_dir, s, branch=branch)

        return json.dumps(summary, ensure_ascii=False)
