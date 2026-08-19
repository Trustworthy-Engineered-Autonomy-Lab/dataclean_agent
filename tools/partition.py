import json
from pathlib import Path
import numpy as np
from .base import Tool
from .decision_policy import effective_action, record_decision
from .utils import (_load, _save, _ensure_constraints, record_observation,
                    append_ledger, _task_artifact_reference)


class Partition(Tool):
    name = "partition"
    description = (
        "Partition dataset using anomaly score threshold into retained (keep) and candidate anomaly (gray) regions. "
        "Omit threshold to calculate data-driven threshold candidates (Otsu, KDE valley, bimodal) and score statistics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "description": "Keep/gray lower threshold. Omit to run candidate analysis only.",
            },
            "gray_upper_threshold": {
                "type": "number",
                "description": "Optional gray/discard upper threshold. Scores at or above it are confident detector discards and are not sent to VLM.",
            },
            "rationale": {
                "type": "string",
                "description": "Required when applying a split: concise reason grounded in returned score statistics/candidates."
            },
        },
        "required": [],
    }

    def run(self, threshold=None, gray_upper_threshold=None, rationale=None,
            branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if s.get("round_status") not in ("scored", "partitioned"):
            raise ValueError("Partition requires current-round scores and cannot run after resolution")
        if not s.get("latest_scores"):
            raise ValueError("No scores file found. Run score_and_fit before partition.")
        if s.get("score_round") not in (None, s.get("round")):
            raise ValueError("Score artifact belongs to a different round")
        score_path = _task_artifact_reference(
            workspace_dir, branch, s["latest_scores"]
        )
        rec = json.loads(score_path.read_text())
        scores = np.array([r["anomaly_score"] for r in rec], dtype=float)
        if len(scores) == 0 or not np.all(np.isfinite(scores)):
            raise ValueError("Score artifact is empty or contains non-finite anomaly scores")

        candidates = self._compute_candidates(scores)
        stats = self._score_stats(scores)

        if threshold is None:
            summary = {
                "mode": "analyze",
                "n_samples": int(len(rec)),
                "candidates": candidates,
                "score_stats": stats,
            }
            record_observation(s, "partition", summary, workspace_dir=workspace_dir, branch=branch)
            _save(workspace_dir, s, branch=branch)
            return json.dumps(summary, ensure_ascii=False)

        proposed = {
            "threshold": float(threshold),
            "gray_upper_threshold": (
                None if gray_upper_threshold is None else float(gray_upper_threshold)
            ),
        }
        effective, source = effective_action(s, "partition", proposed)
        method = "agent_selected"
        if source == "fixed_policy":
            rule = effective.get("rule", "fixed")
            method = f"fixed:{rule}"
            if rule == "fixed":
                if effective.get("value") is None:
                    raise ValueError("fixed partition rule requires value")
                thr = float(effective["value"])
            elif rule == "mean_std":
                thr = float(candidates["mean_plus_std_threshold"])
            elif rule == "otsu":
                thr = float(candidates["otsu_threshold"])
            elif rule == "kde_valley":
                if candidates["kde_valley_threshold"] is None:
                    raise ValueError("Fixed KDE baseline is undefined for this score distribution")
                thr = float(candidates["kde_valley_threshold"])
            else:
                raise ValueError(f"Unknown fixed partition rule: {rule}")
            upper_rule = effective.get("gray_upper_rule", "none")
            if upper_rule == "none":
                upper = None
            elif upper_rule == "fixed":
                if effective.get("gray_upper_value") is None:
                    raise ValueError("fixed gray upper rule requires gray_upper_value")
                upper = float(effective["gray_upper_value"])
            elif upper_rule == "mean_plus_2std":
                upper = float(np.mean(scores) + 2.0 * np.std(scores))
            elif upper_rule == "quantile":
                q = float(effective.get("gray_upper_quantile", 0.95))
                if not 0 < q < 1:
                    raise ValueError("gray_upper_quantile must be in (0, 1)")
                upper = float(np.quantile(scores, q))
            else:
                raise ValueError(f"Unknown fixed gray upper rule: {upper_rule}")
            effective = {**effective, "threshold": thr, "gray_upper_threshold": upper}
            rationale = rationale or f"Preregistered fixed baseline rule: {rule}"
        else:
            thr = float(effective["threshold"])
            upper = effective.get("gray_upper_threshold")
            upper = None if upper is None else float(upper)
            if not rationale or not str(rationale).strip():
                raise ValueError("Adaptive partition decisions require an observation-based rationale")
        if not np.isfinite(thr):
            raise ValueError("threshold must be finite")
        if upper is not None and (not np.isfinite(upper) or upper <= thr):
            raise ValueError("gray_upper_threshold must be finite and greater than threshold")
        keep = [r for r in rec if r["anomaly_score"] < thr]
        gray = [
            r for r in rec
            if r["anomaly_score"] >= thr and (upper is None or r["anomaly_score"] < upper)
        ]
        discard = [] if upper is None else [r for r in rec if r["anomaly_score"] >= upper]
        s["latest_partition"] = {
            "threshold": round(thr, 5),
            "gray_upper_threshold": None if upper is None else round(upper, 5),
            "threshold_method": method,
            "keep_ids": [r["id"] for r in keep],
            "gray_ids": [r["id"] for r in gray],
            "discard_ids": [r["id"] for r in discard],
            "keep_count": len(keep),
            "gray_count": len(gray),
            "discard_count": len(discard),
            "scores_artifact": s.get("latest_scores"),
        }
        s["round_status"] = "partitioned"
        record_decision(
            s,
            "partition",
            proposed,
            effective,
            str(rationale),
            source,
            observation={"candidates": candidates, "score_stats": stats},
        )
        summary = {
            "mode": "split",
            "threshold_applied": round(thr, 5),
            "keep_count": int(len(keep)),
            "gray_count": int(len(gray)),
            "discard_count": int(len(discard)),
            "keep_ratio": round(len(keep) / max(1, len(rec)), 5),
            "gray_upper_threshold": None if upper is None else round(upper, 5),
            "threshold_method": method,
            "decision_source": source,
            "candidates": candidates,
            "score_stats": stats,
        }
        record_observation(s, "partition", summary, workspace_dir=workspace_dir, branch=branch)
        append_ledger(s, {
            "stage": "partition",
            "round": s.get("round"),
            "threshold": round(thr, 5),
            "keep": int(len(keep)),
            "gray": int(len(gray)),
            "discard": int(len(discard)),
        })
        _save(workspace_dir, s, branch=branch)
        return json.dumps(summary, ensure_ascii=False)

    def _compute_candidates(self, scores):
        bc = self._bimodality_coefficient(scores)
        is_bimodal = bool(bc is not None and bc >= 0.555)
        return {
            "otsu_threshold": self._otsu_threshold(scores),
            "kde_valley_threshold": self._kde_valley(scores),
            "bimodality_coefficient": bc,
            "is_bimodal": is_bimodal,
            "mean_plus_std_threshold": round(float(np.mean(scores) + np.std(scores)), 5),
        }

    @staticmethod
    def _otsu_threshold(scores):
        if len(scores) < 3:
            return None
        n_bins = min(128, max(10, len(scores) // 10))
        hist, edges = np.histogram(scores, bins=n_bins)
        p = hist / max(hist.sum(), 1)
        mid = (edges[:-1] + edges[1:]) / 2
        w0 = np.cumsum(p)
        w1 = 1.0 - w0
        w0s = np.clip(w0, 1e-12, None)
        w1s = np.clip(w1, 1e-12, None)
        cum = np.cumsum(p * mid)
        mu_total = cum[-1]
        mu0 = cum / w0s
        mu1 = (mu_total - cum) / w1s
        between = w0 * w1 * (mu0 - mu1) ** 2
        return round(float(mid[int(np.argmax(between))]), 5)

    @staticmethod
    def _kde_valley(scores):
        try:
            from scipy.stats import gaussian_kde
        except Exception:
            return None
        if len(scores) < 8:
            return None
        try:
            kde = gaussian_kde(scores)
            xs = np.linspace(float(scores.min()), float(scores.max()), 500)
            d = kde(xs)
        except Exception:
            return None
        order = 5
        mins = []
        for i in range(order, len(d) - order):
            window = d[i - order:i + order + 1]
            if d[i] == window.min() and d[i] < d[i - 1] and d[i] < d[i + 1]:
                mins.append(i)
        maxs = []
        for i in range(order, len(d) - order):
            window = d[i - order:i + order + 1]
            if d[i] == window.max() and d[i] > d[i - 1] and d[i] > d[i + 1]:
                maxs.append(i)
        valid = [m for m in mins if any(mx < m for mx in maxs) and any(mx > m for mx in maxs)]
        if not valid:
            return None
        return round(float(xs[valid[int(np.argmin(d[valid]))]]), 5)

    @staticmethod
    def _bimodality_coefficient(scores):
        try:
            from scipy.stats import skew, kurtosis
        except Exception:
            return None
        n = len(scores)
        if n < 8:
            return None
        g1 = float(skew(scores, bias=False))
        # The finite-sample BC formula expects excess kurtosis here. Using
        # Pearson kurtosis and then adding the correction double-counted 3.
        g2 = float(kurtosis(scores, fisher=True, bias=False))
        if n > 3:
            denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        else:
            denom = g2 + 3.0
        if denom <= 0:
            return None
        bc = (g1 ** 2 + 1) / denom
        return round(float(bc), 5)

    @staticmethod
    def _bimodal_threshold(scores):
        bc = Partition._bimodality_coefficient(scores)
        if bc is None or bc < 0.555:
            return None
        return Partition._kde_valley(scores)

    @staticmethod
    def _score_stats(scores):
        if len(scores) == 0:
            return {}
        q = np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9])
        return {
            "n": int(len(scores)),
            "mean": round(float(np.mean(scores)), 5),
            "std": round(float(np.std(scores)), 5),
            "min": round(float(np.min(scores)), 5),
            "max": round(float(np.max(scores)), 5),
            "q10": round(float(q[0]), 5),
            "q25": round(float(q[1]), 5),
            "median": round(float(q[2]), 5),
            "q75": round(float(q[3]), 5),
            "q90": round(float(q[4]), 5),
        }
