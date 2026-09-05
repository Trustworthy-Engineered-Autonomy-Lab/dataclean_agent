import json
import numpy as np

from .base import Tool
from .decision_policy import effective_action, record_decision
from .detector_contract import normality_scores, score_contract
from .pcc_plot import plot_strategy_distribution
from .utils import (
    _load, _save, _artifact, _ensure_constraints, record_observation, append_ledger,
    _task_artifact_reference,
)

STRATEGIES = ("mean_std", "kmeans", "kde")
MEAN_STD_K_VALUES = tuple(round(i / 10, 1) for i in range(21))
MEAN_STD_PLOT_K_VALUES = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5)
KDE_BANDWIDTH_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
KMEANS_RANDOM_STATE = 0
ANOMALY_UB_PERCENT = 20.0


# This is intentionally scoped to partition arbitration.  It is not part of
# the global Agent prompt, so its aggressive threshold policy cannot control
# unrelated training, deployment, or round-transition decisions.
PARTITION_PRIOR_PROMPT = r'''You are a threshold arbitration expert for unsupervised anomaly detection.

## Task
You are given the distribution of Pearson Correlation Coefficient (PCC) reconstruction-quality scores from an autoencoder, over an UNLABELED image-action driving dataset. Each sample has one PCC score. You must decide a single threshold tau: samples with PCC < tau are removed as suspected misbehavior; samples with PCC >= tau are kept.

Prior knowledge: HIGH PCC = normal driving (good reconstruction); LOW PCC = suspected anomalous driving.

## Cleaning policy (IMPORTANT)
Cut AGGRESSIVELY. It is acceptable to remove some genuinely normal samples, because a downstream VLM goodness-review recovers wrongly-removed normal samples back into the training set. Therefore, when in doubt, cut MORE, not less.
Do NOT be conservative to protect normal samples; the recovery stage handles that.

## Inputs
1. IMAGE: strategy-specific, label-free views of all PCC scores. The KDE view keeps the PCC-versus-sample-index scatter as its main panel and includes a PCC-versus-density KDE panel. A KDE threshold is drawn only when a stable valley exists:
   - tau_kmeans (K-means, K=2, midpoint of the two centroids)
   - tau_kde (deepest KDE valley between the two highest peaks)
   - tau_meanstd (mu - k*sigma)
2. STRUCTURED CONTEXT (authoritative numbers, use these over reading pixels):
   - bimodality_coefficient (BC, Sarle's): {BC_VALUE}
   - BC bimodality gate: bimodal if BC >= 0.555, else unimodal
   - tau_kmeans = {T_KMEANS} -> removes {DEL_KMEANS}% of data
   - tau_kde = {T_KDE} -> removes {DEL_KDE}% of data (or unavailable when no stable valley exists)
   - tau_meanstd = {T_MEANSTD} (k={K_VALUE}) -> removes {DEL_MEANSTD}% of data
   - PCC range: [{PCC_MIN}, {PCC_MAX}], mean={PCC_MEAN}, std={PCC_STD}
   - max plausible anomaly ratio (prior upper bound): 20%

## Decision rule (MUST follow, in order)
Step 1 - Read the BC gate:
   - If BC >= 0.555 (BIMODAL): PREFER tau_kmeans (K=2). The two centroids give a stable split of the two modes.
   - If BC < 0.555 (UNIMODAL), or the BC value is unavailable: do NOT choose a statistical strategy. Directly estimate the plausible anomaly ratio from the current PCC statistics, the PCC scatter plot, and previous-round VLM aggregate feedback when available. This estimate is a belief about the current round, not a ground-truth label or a fixed normal-rate assumption.
Step 2 - Sanity-adjust the preferred tau using the image and context:
   - For the BIMODAL branch, confirm the preferred tau lands at or just above the visible separation between the low-PCC tail and the main score band. Use the KDE density panel and structured candidates as authoritative; the scatter panel alone does not show density.
   - For the UNIMODAL/BC-unavailable branch, choose an estimated anomaly ratio from 0% to the 20% hard upper bound, then use the corresponding empirical lower-tail PCC quantile as the threshold. Do not select mean-std, K-Means, or KDE in this branch.
   - Given the aggressive policy, if two candidates are close, pick the one that removes MORE (larger tau), UNLESS it would exceed the 20% max plausible anomaly ratio by a large margin.
   - Never choose a tau whose deletion ratio grossly exceeds 20%.
Step 3 - tau_kde is a cross-check, not the default in the BIMODAL branch: only override the BC-preferred choice if a stable KDE valley is available, clearly deeper/cleaner, AND better placed on the gap. If KDE reports no stable valley, do not invent a KDE threshold. KDE is not a choice in the UNIMODAL/BC-unavailable branch.

## Reasoning steps (think in this order)
1. Report BC and whether the gate says bimodal or unimodal.
2. Describe the visible PCC separation and low-score tail from the plot, and report whether the KDE density agrees with the BC gate.
3. For the BIMODAL branch, identify which candidate tau sits best on the gap. For the UNIMODAL/BC-unavailable branch, estimate the plausible anomaly-ratio range using current evidence and previous-round VLM feedback when available.
4. Check whether the aggressive high-recall policy justifies a larger tau without exceeding 20% deletion.
5. Give the final tau. In the UNIMODAL/BC-unavailable branch, report that it came from direct anomaly-ratio estimation and an empirical lower-tail quantile, not a statistical candidate strategy.

## Output - JSON ONLY, no prose, no markdown fences
{
  "bimodal": true/false,
  "bc_value": 0.0,
  "preferred_method": "kmeans" | "meanstd" | "kde" | "direct_ratio",
  "estimated_anomaly_ratio_percent": 0.0,
  "estimated_anomaly_ratio_range_percent": [0.0, 0.0],
  "chosen_tau": 0.0,
  "expected_deletion_ratio": 0.0,
  "gap_location": "<short description>",
  "rationale": "<2-3 sentences: BC gate result, gap placement, why this tau given the aggressive-recall policy>"
}

Runtime adapter: when operating through the partition function interface, express the selected method, supported hyperparameters, and rationale through tool arguments. Do not expose private chain-of-thought.'''


def _prior_value(value):
    if value is None:
        return "unavailable"
    return str(value)


def _format_partition_prior(stats, candidates, plot_name, previous_vlm_feedback=None, *, mean_std_k=1.0):
    """Fill the senior threshold prior with current observable evidence."""
    statistical_branch = stats.get("bimodality_gate") == "bimodal"
    mean_items = candidates.get("mean_std") or []
    mean_item = next((item for item in mean_items if item.get("k") == round(float(mean_std_k), 1)), None)
    kmeans_item = candidates.get("kmeans_reference") or {}
    kde_item = candidates.get("kde_reference") or {}
    if not statistical_branch:
        mean_item = kmeans_item = kde_item = {}
    prior = PARTITION_PRIOR_PROMPT
    replacements = {
        "{BC_VALUE}": _prior_value(stats.get("bimodality_coefficient")),
        "{T_KMEANS}": _prior_value(kmeans_item.get("threshold")),
        "{DEL_KMEANS}": _prior_value(kmeans_item.get("remove_ratio")),
        "{T_KDE}": _prior_value(kde_item.get("threshold")),
        "{DEL_KDE}": _prior_value(kde_item.get("remove_ratio")),
        "{T_MEANSTD}": _prior_value((mean_item or {}).get("threshold")),
        "{K_VALUE}": f"{round(float(mean_std_k), 1):.1f}",
        "{DEL_MEANSTD}": _prior_value((mean_item or {}).get("remove_ratio")),
        "{PCC_MIN}": _prior_value(stats.get("min")),
        "{PCC_MAX}": _prior_value(stats.get("max")),
        "{PCC_MEAN}": _prior_value(stats.get("mean")),
        "{PCC_STD}": _prior_value(stats.get("std")),
    }
    for key, value in replacements.items():
        prior = prior.replace(key, value)
    prior += (
        "\n\nRuntime rendering note: the mean-std, K-Means, and KDE views are provided "
        "as three separate, label-free images rather than one combined image. The "
        "KDE image contains the PCC-versus-sample-index scatter plus a PCC-versus-density "
        "panel; use that density panel when judging peaks or valleys. Use each image "
        "for its corresponding candidate analysis. "
        "Artifact names: "
        + json.dumps(plot_name, ensure_ascii=False, sort_keys=True)
        + "."
    )
    feedback = previous_vlm_feedback or {"available": False, "reason": "No previous-round VLM review is available (first round or no VLM review)."}
    if feedback.get("available"):
        prior += (
            "\n\nPrevious-round VLM aggregate feedback (directional evidence only; not labels): "
            + json.dumps(feedback, ensure_ascii=False, sort_keys=True)
            + ". A high accepted/selected ratio supports a lower current anomaly-rate belief; "
            "a low ratio supports a higher belief. Many unresolved or technical failures reduce confidence."
        )
    else:
        prior += "\n\nPrevious-round VLM aggregate feedback: unavailable. " + str(feedback.get("reason", "No usable prior feedback."))
    if not statistical_branch:
        prior += (
            "\n\nBranch constraint: the BC gate is not bimodal. Statistical candidate strategies "
            "are diagnostic only and must not be selected. Return a direct anomaly-ratio "
            "estimate and let the runtime convert it to an empirical lower-tail quantile."
        )
    shape = candidates.get("shape_consistency") or {}
    if shape:
        prior += (
            "\n\nShape-consistency note: BC gate={bc}; KDE shape={kde}; "
            "conflict={conflict}. Treat this as observable uncertainty and "
            "mention it in the rationale; do not manufacture a KDE valley."
        ).format(
            bc=shape.get("bc_gate", "unavailable"),
            kde=shape.get("kde_shape", "unavailable"),
            conflict=shape.get("conflict", False),
        )
    return prior


def _previous_vlm_feedback(state):
    """Return only the previous round's aggregate VLM evidence for partitioning."""
    current_round = int(state.get("round", 0))
    if current_round <= 0:
        return {"available": False, "reason": "First round has no previous VLM review."}
    entries = [
        entry for entry in (state.get("round_history") or [])
        if int(entry.get("round", -1)) == current_round - 1
    ]
    if not entries:
        return {"available": False, "reason": "Previous round history is unavailable."}
    observation = (entries[-1].get("observations") or {}).get("resolve") or {}
    selected = int(observation.get("vlm_selected", 0) or 0)
    accepted = int(observation.get("vlm_accepted", 0) or 0)
    unresolved = int(observation.get("vlm_unresolved", 0) or 0)
    technical = int(observation.get("vlm_technical_failures", 0) or 0)
    successful = int(observation.get("vlm_successful_responses", 0) or 0)
    if selected <= 0:
        return {"available": False, "reason": "Previous round did not provide a usable VLM-selected sample count."}
    return {
        "available": True,
        "previous_round": current_round - 1,
        "vlm_selected": selected,
        "vlm_accepted": accepted,
        "vlm_unresolved": unresolved,
        "vlm_technical_failures": technical,
        "vlm_successful_responses": successful,
        "acceptance_rate_over_selected": round(accepted / selected, 6),
        "usable_feedback_fraction": round(successful / selected, 6),
    }


class Partition(Tool):
    name = "partition"
    description = (
        "Analyze or apply a PCC split. With no strategy or estimated anomaly ratio, return candidates for "
        "mean-k*std (k=0.0..2.0 step 0.1), K-means (K=2), and KDE. When BC is below 0.555 or unavailable, "
        "the Agent may apply a direct estimated anomaly ratio, converted to an empirical lower-tail PCC quantile. "
        "PCC is interpreted using the partition prior and the current score evidence. KDE is "
        "unavailable when no stable valley is detected; it never falls back to a median or quantile. "
        "All supported strategies use a single keep/gray boundary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "strategy": {"type": "string", "enum": list(STRATEGIES),
                         "description": "Provide for the bimodal candidate branch; omit for analysis or direct anomaly-ratio mode."},
            "estimated_anomaly_ratio_percent": {
                "type": "number", "minimum": 0, "maximum": 20,
                "description": "For BC<0.555/unavailable: estimated current-round anomaly percentage, converted to the empirical lower-tail PCC quantile; hard maximum is 20%.",
            },
            "mean_std_k": {"type": "number", "minimum": 0, "maximum": 2,
                           "description": "For mean_std: k in [0.0, 2.0] at increments of 0.1."},
            "kmeans_k": {"type": "integer", "enum": [2],
                         "description": "For kmeans: two one-dimensional PCC clusters."},
            "kmeans_boundary": {"type": "string", "enum": ["only"],
                                "description": "K=2 midpoint boundary for keep/gray."},
            "kde_bandwidth_scale": {"type": "number", "minimum": 0.5, "maximum": 2.0,
                                     "description": "For KDE: multiplier of Scott bandwidth; allowed 0.50..2.00 in 0.25 steps."},
            "kde_valley_index": {"type": "integer", "minimum": 0,
                                 "description": "For KDE: index of a returned valley at the selected bandwidth."},
            "evidence": {
                "type": "object",
                "description": "Required for adaptive application; concise auditable evidence, not private chain-of-thought.",
                "properties": {
                    "distribution_shape": {"type": "string"},
                    "prior_assumptions_and_uncertainty": {"type": "string"},
                    "candidate_comparison": {"type": "string", "description": "Required for statistical candidates; for direct anomaly-ratio mode, briefly state why no candidate strategy is being used."},
                    "main_risk": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
            "rationale": {"type": "string",
                          "description": "Required when applying: evidence-based reason using returned candidates and task context."},
        },
        "required": [],
    }

    def run(self, strategy=None, mean_std_k=None, kmeans_k=None,
            kmeans_boundary=None, kde_bandwidth_scale=None, kde_valley_index=None,
            estimated_anomaly_ratio_percent=None,
            evidence=None, rationale=None, branch="main", workspace_dir=None, **kwargs):
        if "threshold" in kwargs:
            raise ValueError("Arbitrary thresholds are unsupported; choose a supported strategy")
        s = _load(workspace_dir, branch=branch)
        _ensure_constraints(s, branch)
        if s.get("round_status") not in ("scored", "partitioned"):
            raise ValueError("Partition requires current-round scores and cannot run after resolution")
        if not s.get("latest_scores"):
            raise ValueError("No scores file found. Run score_and_fit before partition.")
        if s.get("score_round") not in (None, s.get("round")):
            raise ValueError("Score artifact belongs to a different round")
        score_path = _task_artifact_reference(workspace_dir, branch, s["latest_scores"])
        records = json.loads(score_path.read_text())
        scores = np.asarray(normality_scores(records), dtype=float)
        stats = self._score_stats(scores)
        candidates = self._compute_candidates(scores)
        candidates["shape_consistency"] = self._shape_consistency(
            stats.get("bimodality_gate"), candidates.get("kde", {})
        )
        plot_artifacts, plot_errors = self._write_partition_plots(
            scores, s.get("round", 0), workspace_dir, branch, candidates
        )
        previous_vlm_feedback = _previous_vlm_feedback(s)
        prior = _format_partition_prior(
            stats, candidates, plot_artifacts, previous_vlm_feedback
        )
        candidate_view = candidates
        if stats.get("bimodality_gate") != "bimodal":
            candidate_view = {
                "selection_disabled": True,
                "reason": "BC is below 0.555 or unavailable; statistical strategies are not a decision branch.",
                "shape_consistency": candidates.get("shape_consistency"),
            }
        agent_visible_artifacts = [
            {"name": name, "kind": "image", "purpose": f"Partition {strategy} candidate PCC scatter plot"}
            for strategy, name in plot_artifacts.items()
        ]
        if strategy is None and estimated_anomaly_ratio_percent is None:
            summary = {"mode": "analyze", "score_contract": score_contract(),
                       "n_samples": len(records), "score_stats": stats, "candidates": candidate_view,
                       "previous_round_vlm_feedback": previous_vlm_feedback,
                       "anomaly_ratio_upper_bound_percent": ANOMALY_UB_PERCENT,
                       "partition_plots": plot_artifacts, "plot_errors": plot_errors,
                       "partition_prior_prompt": prior,
                       "agent_visible_artifacts": agent_visible_artifacts}
            record_observation(s, "partition", summary, workspace_dir=workspace_dir, branch=branch)
            _save(workspace_dir, s, branch=branch)
            return json.dumps(summary, ensure_ascii=False)

        proposed = {
            "strategy": strategy,
            "estimated_anomaly_ratio_percent": estimated_anomaly_ratio_percent,
        }
        for key, value in (("mean_std_k", mean_std_k), ("kmeans_k", kmeans_k),
                           ("kmeans_boundary", kmeans_boundary),
                           ("kde_bandwidth_scale", kde_bandwidth_scale),
                           ("kde_valley_index", kde_valley_index)):
            if value is not None:
                proposed[key] = value
        effective, source = effective_action(s, "partition", proposed)
        strategy = effective.get("strategy")
        estimated_ratio = effective.get("estimated_anomaly_ratio_percent")
        threshold_mode = effective.get("threshold_mode")
        if threshold_mode is None:
            threshold_mode = (
                "estimated_anomaly_ratio"
                if strategy in (None, "") and estimated_ratio is not None
                else "candidate"
            )
        direct_ratio_mode = threshold_mode == "estimated_anomaly_ratio"
        if direct_ratio_mode:
            if stats.get("bimodality_gate") == "bimodal":
                raise ValueError("Direct anomaly-ratio partition is only valid when BC is below 0.555 or unavailable")
            if strategy not in (None, ""):
                raise ValueError("Direct anomaly-ratio partition cannot include a statistical strategy")
            if estimated_ratio is None:
                raise ValueError("estimated_anomaly_ratio_percent is required for direct anomaly-ratio partition")
        elif strategy not in STRATEGIES:
            raise ValueError("Partition strategy must be mean_std, kmeans, or kde")
        if source.startswith("agent") and not str(rationale or "").strip():
            raise ValueError("Adaptive partition decisions require an observation-based rationale")
        evidence = evidence if isinstance(evidence, dict) else {}
        if source.startswith("agent"):
            required_evidence = [
                "distribution_shape", "prior_assumptions_and_uncertainty",
                "main_risk", "confidence",
            ]
            if not direct_ratio_mode:
                required_evidence.append("candidate_comparison")
            missing = [key for key in required_evidence if not str(evidence.get(key) or "").strip()]
            if missing:
                raise ValueError("Adaptive partition evidence is incomplete: " + ", ".join(missing))
            if evidence.get("confidence") not in ("low", "medium", "high"):
                raise ValueError("Adaptive partition evidence.confidence must be low, medium, or high")
        if source == "fixed_policy" and not str(rationale or "").strip():
            rationale = "Preregistered fixed partition strategy"
        if direct_ratio_mode:
            threshold, params, candidate_id = self._select_estimated_ratio(
                scores, estimated_ratio
            )
            partition_method = "estimated_anomaly_ratio"
        else:
            threshold, params, candidate_id = self._select_candidate(strategy, effective, candidates)
            partition_method = strategy
        keep = [r for r in records if float(r["normality_score"]) >= threshold]
        gray = [r for r in records if float(r["normality_score"]) < threshold]
        removal_percent = len(gray) / max(1, len(records)) * 100.0
        if removal_percent > ANOMALY_UB_PERCENT + 1e-9:
            raise ValueError(
                f"Selected threshold removes {removal_percent:.2f}% of data, "
                f"above the fixed {ANOMALY_UB_PERCENT:.0f}% anomaly upper bound"
            )
        effective_params = {"strategy": strategy, "threshold_mode": threshold_mode, **params}
        if evidence:
            effective_params["evidence"] = evidence
        s["latest_partition"] = {
            "threshold": threshold,
            "score_contract": score_contract(), "threshold_method": partition_method,
            "strategy": strategy, "threshold_mode": threshold_mode,
            "strategy_params": params, "candidate_id": candidate_id,
            "evidence": evidence,
            "keep_ids": [r["id"] for r in keep], "gray_ids": [r["id"] for r in gray],
            "keep_count": len(keep), "gray_count": len(gray),
            "removal_percent": round(removal_percent, 5),
            "scores_artifact": s.get("latest_scores"),
        }
        decision_entry = record_decision(
            s, "partition", {**proposed, "evidence": evidence}, effective_params,
            str(rationale), source,
            observation={"score_stats": stats, "candidate_id": candidate_id,
                         "threshold_mode": threshold_mode,
                         "previous_round_vlm_feedback": previous_vlm_feedback,
                         "candidates": candidates, "evidence": evidence},
        )
        s["round_status"] = "partitioned"
        summary = {"mode": "split", "score_contract": score_contract(), "strategy": strategy,
                   "threshold_mode": threshold_mode,
                   "strategy_params": params, "candidate_id": candidate_id,
                   "estimated_anomaly_ratio_percent": params.get("estimated_anomaly_ratio_percent"),
                   "threshold_applied": threshold, "keep_count": len(keep),
                   "gray_count": len(gray),
                   "removal_percent": round(removal_percent, 5),
                   "keep_ratio": round(len(keep) / max(1, len(records)), 5),
                   "gray_ratio": round(len(gray) / max(1, len(records)), 5),
                   "score_stats": stats, "candidates": candidate_view,
                   "previous_round_vlm_feedback": previous_vlm_feedback,
                   "evidence": evidence,
                   "anomaly_ratio_upper_bound_percent": ANOMALY_UB_PERCENT,
                   "partition_plots": plot_artifacts, "plot_errors": plot_errors,
                   "partition_prior_prompt": prior,
                   "agent_visible_artifacts": agent_visible_artifacts}
        record_observation(s, "partition", summary, workspace_dir=workspace_dir,
                           branch=branch, decision=decision_entry)
        append_ledger(s, {"stage": "partition", "round": s.get("round"), "strategy": partition_method,
                          "threshold": threshold, "keep": len(keep), "gray": len(gray)})
        _save(workspace_dir, s, branch=branch)
        return json.dumps(summary, ensure_ascii=False)

    def _compute_candidates(self, scores):
        candidates = {"mean_std": self._mean_std_candidates(scores),
                      "kmeans": self._kmeans_candidates(scores), "kde": self._kde_candidates(scores),
                      "selection_note": "Apply the partition prior using the current candidates and the fixed 20% anomaly upper bound."}
        kmeans = candidates["kmeans"]
        if kmeans.get("available"):
            for model in kmeans.get("models", []):
                if model.get("k") == 2:
                    candidates["kmeans_reference"] = next(
                        (item for item in model.get("boundaries", []) if item.get("boundary") == "only"),
                        None,
                    )
                    break
        kde = candidates["kde"]
        if kde.get("available"):
            valleys = []
            for scale in kde.get("bandwidth_scales", []):
                for valley in scale.get("valleys", []):
                    valleys.append({**valley, "bandwidth_scale": scale.get("bandwidth_scale")})
            if valleys:
                top_pair_valleys = [
                    item for item in valleys if item.get("between_two_highest_peaks")
                ]
                if top_pair_valleys:
                    valleys = top_pair_valleys
                candidates["kde_reference"] = min(
                    valleys,
                    key=lambda item: (float(item.get("valley_density", float("inf"))),
                                      -int(item.get("stability_support", 0))),
                )
            kde["status"] = "stable_valley" if candidates.get("kde_reference") else "no_stable_valley"
            kde["reference"] = candidates.get("kde_reference")
        else:
            kde["status"] = "unavailable"
            kde["reference"] = None
        return candidates

    @staticmethod
    def _shape_consistency(bc_gate, kde):
        if not kde or kde.get("available") is False:
            kde_shape = "unavailable"
        elif kde.get("status") == "stable_valley":
            kde_shape = "multimodal_with_valley"
        else:
            kde_shape = "unimodal_or_no_stable_valley"
        conflict = bc_gate == "bimodal" and kde_shape != "multimodal_with_valley"
        return {
            "bc_gate": bc_gate or "unavailable",
            "kde_shape": kde_shape,
            "conflict": conflict,
        }

    @staticmethod
    def _partition_counts(scores, threshold):
        keep = int(np.sum(scores >= threshold))
        return {"threshold": round(float(threshold), 6), "keep_count": keep,
                "gray_count": int(len(scores) - keep),
                "keep_ratio": round(keep / max(1, len(scores)), 6),
                "remove_ratio": round((len(scores) - keep) / max(1, len(scores)) * 100, 6)}

    def _write_partition_plots(self, scores, round_index, workspace_dir, branch, candidates):
        """Render one label-free PCC scatter plot per supported strategy."""
        names = {
            "mean_std": f"pcc_partition_mean_std_r{int(round_index)}.png",
            "kmeans": f"pcc_partition_kmeans_r{int(round_index)}.png",
            "kde": f"pcc_partition_kde_r{int(round_index)}.png",
        }
        lines = {
            "mean_std": [
                {
                    "threshold": next(item["threshold"] for item in candidates["mean_std"] if item["k"] == k),
                    "label": f"k={k:.1f}",
                    "color": "green" if k == 1.0 else "#555555",
                    "linestyle": ":" if k == 1.0 else "--",
                }
                for k in MEAN_STD_PLOT_K_VALUES
            ],
            "kmeans": [],
            "kde": [],
        }
        if candidates.get("kmeans_reference"):
            item = candidates["kmeans_reference"]
            lines["kmeans"] = [{"threshold": item["threshold"], "label": "K-Means K=2"}]
        if candidates.get("kde_reference"):
            item = candidates["kde_reference"]
            lines["kde"] = [{
                "threshold": item["threshold"],
                "label": f"KDE (bandwidth={float(item['bandwidth_scale']):.2f})",
            }]

        artifacts, errors = {}, {}
        for strategy, filename in names.items():
            path = _artifact(workspace_dir, filename, branch=branch)
            try:
                plot_strategy_distribution(
                    scores,
                    path,
                    round_index,
                    strategy,
                    lines[strategy],
                    kde_data=candidates.get("kde") if strategy == "kde" else None,
                )
                artifacts[strategy] = path.name
            except Exception as exc:
                errors[strategy] = f"{type(exc).__name__}: {exc}"
        return artifacts, errors

    def _mean_std_candidates(self, scores):
        mean, std = float(np.mean(scores)), float(np.std(scores))
        return [{"candidate_id": f"mean_std:k={k:.1f}", "strategy": "mean_std", "k": k,
                 **self._partition_counts(scores, mean - k * std)} for k in MEAN_STD_K_VALUES]

    def _kmeans_candidates(self, scores):
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except Exception as exc:
            return {"available": False, "error": f"scikit-learn unavailable: {exc}"}
        if len(scores) < 4 or float(np.ptp(scores)) == 0:
            return {"available": False, "error": "Insufficient score variation for K-means"}
        result = {"available": True, "random_state": KMEANS_RANDOM_STATE, "models": []}
        x = scores.reshape(-1, 1)
        for k in (2,):
            try:
                model = KMeans(n_clusters=k, n_init=20, max_iter=300,
                               random_state=KMEANS_RANDOM_STATE).fit(x)
                centers = np.sort(model.cluster_centers_.ravel())
                labels = np.argmin(np.abs(x - centers.reshape(1, -1)), axis=1)
                counts = [int(np.sum(labels == i)) for i in range(k)]
                if min(counts) == 0:
                    continue
                sil = float(silhouette_score(x, labels, sample_size=min(2000, len(x)),
                                             random_state=KMEANS_RANDOM_STATE))
                info = {"k": k, "centers": [round(float(v), 6) for v in centers],
                        "cluster_counts": counts, "inertia": round(float(model.inertia_), 6),
                        "silhouette": round(sil, 6), "boundaries": []}
                for i in range(k - 1):
                    boundary = float((centers[i] + centers[i + 1]) / 2)
                    info["boundaries"].append({"candidate_id": "kmeans:k=2:boundary=only",
                                               "boundary": "only",
                                               **self._partition_counts(scores, boundary)})
                result["models"].append(info)
            except Exception as exc:
                result.setdefault("errors", []).append(f"k={k}: {exc}")
        return result

    def _kde_candidates(self, scores):
        try:
            from scipy.stats import gaussian_kde
            from scipy.signal import find_peaks
        except Exception as exc:
            return {"available": False, "error": f"scipy unavailable: {exc}"}
        if len(scores) < 8 or float(np.ptp(scores)) == 0:
            return {"available": False, "error": "Insufficient scores or variation for KDE"}
        xs = np.linspace(float(scores.min()), float(scores.max()), 1024)
        scales = []
        for scale in KDE_BANDWIDTH_SCALES:
            try:
                kde = gaussian_kde(scores, bw_method=lambda obj, q=scale: obj.scotts_factor() * q)
                density = kde(xs)
                peaks, _ = find_peaks(density, distance=max(5, len(xs) // 100),
                                      prominence=max(float(np.max(density)) * .01, 1e-12))
                top_two = tuple(sorted(
                    peaks[np.argsort(density[peaks])[-2:]]
                )) if len(peaks) >= 2 else tuple()
                valleys = []
                for index, (left, right) in enumerate(zip(peaks[:-1], peaks[1:])):
                    valley = int(left + np.argmin(density[left:right + 1]))
                    valleys.append({"index": index, "threshold": round(float(xs[valley]), 6),
                                    **self._partition_counts(scores, float(xs[valley])),
                                    "left_peak": round(float(xs[left]), 6),
                                    "right_peak": round(float(xs[right]), 6),
                                    "left_peak_density": round(float(density[left]), 8),
                                    "right_peak_density": round(float(density[right]), 8),
                                    "between_two_highest_peaks": (int(left), int(right)) == top_two,
                                    "valley_density": round(float(density[valley]), 8)})
                scales.append({"bandwidth_scale": scale, "bandwidth_factor": round(float(kde.factor), 8),
                               "peak_count": int(len(peaks)),
                               "peaks": [round(float(xs[i]), 6) for i in peaks], "valleys": valleys})
            except Exception as exc:
                scales.append({"bandwidth_scale": scale, "error": str(exc), "valleys": []})
        tolerance = max(.01, .03 * float(np.ptp(scores)))
        all_valleys = [v for item in scales for v in item.get("valleys", [])]
        for item in scales:
            for valley in item.get("valleys", []):
                support = sum(abs(v["threshold"] - valley["threshold"]) <= tolerance
                              for v in all_valleys if v is not valley)
                valley["stability_support"] = int(support)
                valley["candidate_id"] = (f"kde:bandwidth={item['bandwidth_scale']:.2f}:"
                                           f"valley={valley['index']}")
        return {"available": True, "bandwidth_scales": scales,
                "stability_tolerance": round(tolerance, 6)}

    def _select_estimated_ratio(self, scores, raw_ratio):
        """Convert the Agent's direct anomaly-rate belief into a lower-tail cutoff."""
        if isinstance(raw_ratio, bool):
            raise ValueError("estimated_anomaly_ratio_percent must be a number")
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("estimated_anomaly_ratio_percent must be a number") from exc
        if not np.isfinite(ratio) or ratio < 0 or ratio > ANOMALY_UB_PERCENT:
            raise ValueError(
                f"estimated_anomaly_ratio_percent must be in [0, {ANOMALY_UB_PERCENT:.0f}]"
            )
        ratio = round(ratio, 6)
        if ratio == 0:
            threshold = float(np.nextafter(np.min(scores), -np.inf))
        else:
            threshold = float(np.quantile(scores, ratio / 100.0))
        params = {
            "estimated_anomaly_ratio_percent": ratio,
            "threshold_source": "empirical_lower_tail_quantile",
        }
        candidate_id = f"direct_ratio:p={ratio:.6f}"
        return threshold, params, candidate_id

    def _select_candidate(self, strategy, params, candidates):
        if strategy == "mean_std":
            raw = params.get("mean_std_k")
            if raw is None:
                raise ValueError("mean_std requires mean_std_k in [0.0, 2.0] step 0.1")
            k = round(float(raw), 1)
            if abs(float(raw) - k) > 1e-7 or k < 0 or k > 2:
                raise ValueError("mean_std_k must be a multiple of 0.1 in [0.0, 2.0]")
            item = next((x for x in candidates["mean_std"] if x["k"] == k), None)
            if item is None:
                raise ValueError("mean_std_k is not in the supported grid")
            return float(item["threshold"]), {"k": k}, item["candidate_id"]
        if strategy == "kmeans":
            k = int(params.get("kmeans_k", 0))
            boundary = params.get("kmeans_boundary") or "only"
            if k != 2 or boundary != "only":
                raise ValueError("kmeans requires kmeans_k=2 and kmeans_boundary=only")
            for model in candidates.get("kmeans", {}).get("models", []):
                if model["k"] == k:
                    for item in model["boundaries"]:
                        if item["boundary"] == boundary:
                            params = {"k": k, "boundary": boundary}
                            return float(item["threshold"]), params, item["candidate_id"]
            raise ValueError("Requested K-means candidate is unavailable")
        if strategy == "kde":
            raw_scale, raw_index = params.get("kde_bandwidth_scale"), params.get("kde_valley_index")
            if raw_scale is None or raw_index is None:
                raise ValueError("kde requires bandwidth scale and valley index")
            scale = min(KDE_BANDWIDTH_SCALES, key=lambda x: abs(x - float(raw_scale)))
            if abs(float(raw_scale) - scale) > 1e-7:
                raise ValueError("kde_bandwidth_scale must be one of 0.50..2.00 in 0.25 steps")
            index = int(raw_index)
            for item in candidates.get("kde", {}).get("bandwidth_scales", []):
                if item.get("bandwidth_scale") == scale:
                    for valley in item.get("valleys", []):
                        if valley["index"] == index:
                            return float(valley["threshold"]), {"bandwidth_scale": scale, "valley_index": index}, valley["candidate_id"]
            raise ValueError("Requested KDE valley candidate is unavailable")
        raise ValueError("Unknown partition strategy")

    @staticmethod
    def _score_stats(scores):
        q = np.quantile(scores, [.01, .05, .10, .25, .50, .75, .90, .95, .99])
        mean, std = float(np.mean(scores)), float(np.std(scores))
        centered = scores - mean
        skew = float(np.mean(centered ** 3) / std ** 3) if std > 0 else 0.0
        kurt = float(np.mean(centered ** 4) / std ** 4 - 3.0) if std > 0 else 0.0
        n = len(scores)
        if n > 3 and std > 0:
            bc_denominator = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
            bimodality = float((skew ** 2 + 1.0) / bc_denominator) if bc_denominator else None
        else:
            bimodality = None
        return {"n": int(len(scores)), "mean": round(mean, 6), "std": round(std, 6),
                "min": round(float(np.min(scores)), 6), "max": round(float(np.max(scores)), 6),
                "q01": round(float(q[0]), 6), "q05": round(float(q[1]), 6),
                "q10": round(float(q[2]), 6), "q25": round(float(q[3]), 6),
                "median": round(float(q[4]), 6), "q75": round(float(q[5]), 6),
                "q90": round(float(q[6]), 6), "q95": round(float(q[7]), 6),
                "q99": round(float(q[8]), 6), "skewness": round(skew, 6),
                "excess_kurtosis": round(kurt, 6), "range": round(float(np.ptp(scores)), 6),
                "bimodality_coefficient": round(bimodality, 6) if bimodality is not None else None,
                "bimodality_gate": (
                    "bimodal" if bimodality is not None and bimodality >= 0.555 else "unimodal"
                    if bimodality is not None else "unavailable"
                )}
