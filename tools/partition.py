import json
from pathlib import Path
import numpy as np
from .base import Tool
from .utils import _load, _save, _ensure_constraints, record_observation, append_ledger


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
                "description": "Partitioning threshold value. Omit to run candidate analysis only.",
            },
        },
        "required": [],
    }

    def run(self, threshold=None, branch="main", workspace_dir=None, **_):
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if not s.get("latest_scores"):
            raise ValueError("No scores file found. Run score_and_fit before partition.")
        rec = json.loads(Path(s["latest_scores"]).read_text())
        scores = np.array([r["anomaly_score"] for r in rec], dtype=float)

        candidates = self._compute_candidates(scores)
        stats = self._score_stats(scores)

        c = s.get("constraints") or {}
        locked_thr = c.get("locked_threshold")
        if locked_thr is not None and threshold is None:
            threshold = float(locked_thr)

        if threshold is None:
            summary = {
                "mode": "analyze",
                "n_samples": int(len(rec)),
                "candidates": candidates,
                "score_stats": stats,
            }
            record_observation(s, "partition", summary)
            _save(workspace_dir, s, branch=branch)
            return json.dumps(summary, ensure_ascii=False)

        thr = float(threshold)
        keep = [r for r in rec if r["anomaly_score"] < thr]
        gray = [r for r in rec if r["anomaly_score"] >= thr]
        s["latest_partition"] = {
            "threshold": round(thr, 5),
            "keep": keep,
            "gray": gray,
        }
        summary = {
            "mode": "split",
            "threshold_applied": round(thr, 5),
            "keep_count": int(len(keep)),
            "gray_count": int(len(gray)),
            "keep_ratio": round(len(keep) / max(1, len(rec)), 5),
            "candidates": candidates,
            "score_stats": stats,
        }
        record_observation(s, "partition", summary)
        append_ledger(s, {
            "stage": "partition",
            "round": s.get("round"),
            "threshold": round(thr, 5),
            "keep": int(len(keep)),
            "gray": int(len(gray)),
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
        kde = gaussian_kde(scores)
        xs = np.linspace(float(scores.min()), float(scores.max()), 500)
        d = kde(xs)
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
        g1 = float(skew(scores))
        g2 = float(kurtosis(scores, fisher=False))
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
