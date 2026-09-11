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

STRATEGIES = ("mean_std", "kmeans", "kde", "mean_std_ratio")
MEAN_STD_K_VALUES = tuple(round(i / 10, 1) for i in range(21))
MEAN_STD_PLOT_K_VALUES = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5)
KDE_BANDWIDTH_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
KMEANS_RANDOM_STATE = 0

ANOMALY_RATIO_MIN_PERCENT = 0.1
ANOMALY_RATIO_MAX_PERCENT = 5.0
ANOMALY_RATIO_DISPLAY_VALUES = (0.5, 1.0, 2.0, 3.0, 5.0)


# This is intentionally scoped to partition arbitration.  It is not part of
# the global Agent prompt, so its threshold policy cannot control
# unrelated training, deployment, or round-transition decisions.
PARTITION_PRIOR_PROMPT = r'''You are a threshold arbitration expert for unsupervised anomaly detection.

## Task
You are given the distribution of Pearson Correlation Coefficient (PCC) reconstruction-quality scores from an autoencoder, over an UNLABELED image-action driving dataset. Each sample has one PCC score. You must decide a single threshold tau: samples with PCC < tau are removed as suspected misbehavior; samples with PCC >= tau are kept.

Prior knowledge: HIGH PCC = normal driving (good reconstruction); LOW PCC = suspected anomalous driving.

## Cleaning policy
Choose the threshold from the currently observed score distribution, plot evidence,
and available downstream feedback. Do not impose a preferred amount of removal:
neither retaining more nor removing more is correct without supporting evidence.

## Inputs
1. IMAGE: strategy-specific, label-free views of all PCC scores. The KDE view keeps the PCC-versus-sample-index scatter as its main panel and includes a PCC-versus-density KDE panel. A KDE threshold is drawn only when a stable valley exists:
   - tau_kmeans (K-means, K=2, midpoint of the two centroids)
   - tau_meanstd_ratio (mu - k*sigma, where k is derived from an estimated anomaly ratio -- NOT chosen from a fixed grid)
   - tau_kde (deepest KDE valley between the two highest peaks) -- fallback only, see Step 1 below
2. STRUCTURED CONTEXT (authoritative numbers, use these over reading pixels):
   - bimodality_coefficient (BC, Sarle's): {BC_VALUE}
   - BC bimodality gate (ENFORCED, not advisory): bimodal if BC >= 0.555, else unimodal
   - tau_kmeans = {T_KMEANS} -> removes {DEL_KMEANS}% of data
   - tau_kde = {T_KDE} -> removes {DEL_KDE}% of data (or unavailable when no stable valley exists)
   - estimated-ratio reference points (mu - k*sigma at several illustrative ratios; calibration context only, you must still supply your own estimated_anomaly_ratio_percent): {T_RATIO_LIST}
   - PCC range: [{PCC_MIN}, {PCC_MAX}], mean={PCC_MEAN}, std={PCC_STD}

## Decision rule (MUST follow -- this is an ENFORCED gate, not a preference)
Step 1 - Read the BC gate:
   - If BC >= 0.555 (BIMODAL): you MUST select kmeans (K=2 midpoint). It is the only primary strategy the runtime accepts in this case.
   - If BC < 0.555 (UNIMODAL), or BC is unavailable: anomalies are assumed rare. You MUST select mean_std_ratio and supply estimated_anomaly_ratio_percent -- your best estimate of the percent of samples that are anomalous. The runtime converts this to a z-score k = norm.ppf(1 - ratio/100) and computes threshold = mean - k*std. Do NOT pick k directly and do NOT pick a point on the mean-k*std grid; estimate the ratio itself from the visible evidence (tail shape, low-score density, prior-round VLM feedback).
   - kde is never a primary choice in either branch. It is a rare fallback/backstop: select it only when the primary strategy for your branch (kmeans for bimodal, mean_std_ratio for unimodal/unavailable) clearly does not fit the visible evidence. Selecting kde REQUIRES both (a) a stable KDE valley (kde status = stable_valley) and (b) an explicit kde_fallback_justification explaining why the primary strategy was rejected. The runtime rejects a kde selection made without a stable valley or without this justification.
Step 2 - Sanity-adjust the selected tau using the image and context:
   - BIMODAL case: confirm tau_kmeans lands at or just above the visible separation between the low-PCC tail and the main score band. Use the KDE density panel and structured candidates as authoritative; the scatter panel alone does not show density.
   - UNIMODAL/BC-unavailable case: use the visible low-score tail, the estimated-ratio reference points, and any prior-round VLM feedback to judge whether your estimated_anomaly_ratio_percent is plausible; a ratio near {ANOMALY_RATIO_MIN}% implies almost no removal, a ratio near {ANOMALY_RATIO_MAX}% implies more removal.
Step 3 - kde fallback check: only invoke the Step 1 escape hatch when the primary strategy's threshold clearly does not sit on the visible gap/tail AND a stable KDE valley exists. Never invent a KDE threshold when no stable valley is reported.

## Reasoning steps (think in this order)
1. Report BC and state which branch it selects: kmeans (bimodal) or mean_std_ratio (unimodal/unavailable).
2. Describe the visible PCC separation and low-score tail from the plot, and report whether the KDE density agrees with the BC signal.
3. For the bimodal branch, confirm tau_kmeans against the visible gap. For the unimodal/unavailable branch, state your estimated_anomaly_ratio_percent and why (tail shape, density, prior VLM feedback), and report the resulting tau.
4. Only if the primary strategy clearly does not fit the evidence, state the kde_fallback_justification and confirm a stable KDE valley before switching to kde.
5. Check whether the selected deletion ratio is supported by the available evidence and state the main uncertainty.
6. Give the final tau and which strategy and parameters produced it.

## Output - JSON ONLY, no prose, no markdown fences
{
  "bimodal": true/false,
  "bc_value": 0.0,
  "preferred_method": "kmeans" | "mean_std_ratio" | "kde",
  "estimated_anomaly_ratio_percent": 0.0,
  "chosen_tau": 0.0,
  "expected_deletion_ratio": 0.0,
  "gap_location": "<short description>",
  "kde_fallback_justification": "<required only if preferred_method is kde, else omit or leave empty>",
  "rationale": "<2-3 sentences: BC gate, gap placement, and evidence supporting this threshold>"
}

Runtime adapter: when operating through the partition function interface, express the selected method, supported hyperparameters, and rationale through tool arguments. Do not expose private chain-of-thought.'''


def _prior_value(value):
    if value is None:
        return "unavailable"
    return str(value)


def _format_ratio_list(items):
    if not items:
        return "unavailable"
    return "; ".join(
        f"ratio={item['estimated_anomaly_ratio_percent']:.2f}% -> tau={item['threshold']:.6f} "
        f"(removes {item['remove_ratio']:.2f}%)"
        for item in items
    )


def _format_partition_prior(stats, candidates, plot_name, previous_vlm_feedback=None, *, mean_std_k=1.0):
    """Fill the senior threshold prior with current observable evidence."""
    mean_items = candidates.get("mean_std") or []
    mean_item = next((item for item in mean_items if item.get("k") == round(float(mean_std_k), 1)), None)
    kmeans_item = candidates.get("kmeans_reference") or {}
    kde_item = candidates.get("kde_reference") or {}
    ratio_items = candidates.get("mean_std_ratio") or []
    prior = PARTITION_PRIOR_PROMPT
    replacements = {
        "{BC_VALUE}": _prior_value(stats.get("bimodality_coefficient")),
        "{T_KMEANS}": _prior_value(kmeans_item.get("threshold")),
        "{DEL_KMEANS}": _prior_value(kmeans_item.get("remove_ratio")),
        "{T_KDE}": _prior_value(kde_item.get("threshold")),
        "{DEL_KDE}": _prior_value(kde_item.get("remove_ratio")),
        "{T_RATIO_LIST}": _format_ratio_list(ratio_items),
        "{ANOMALY_RATIO_MIN}": f"{ANOMALY_RATIO_MIN_PERCENT:.2f}",
        "{ANOMALY_RATIO_MAX}": f"{ANOMALY_RATIO_MAX_PERCENT:.2f}",
        "{PCC_MIN}": _prior_value(stats.get("min")),
        "{PCC_MAX}": _prior_value(stats.get("max")),
        "{PCC_MEAN}": _prior_value(stats.get("mean")),
        "{PCC_STD}": _prior_value(stats.get("std")),
    }
    for key, value in replacements.items():
        prior = prior.replace(key, value)
    prior += (
        "\n\nRuntime rendering note: the mean-std, K-Means, estimated-ratio mean-std, and KDE "
        "views are provided as separate, label-free images rather than one combined image. The "
        "KDE image contains the PCC-versus-sample-index scatter plus a PCC-versus-density "
        "panel; use that density panel when judging peaks or valleys. The estimated-ratio image "
        "shows reference mean-k*std lines at illustrative anomaly-ratio percentages; your own "
        "estimated_anomaly_ratio_percent need not match one of these lines exactly. Use each image "
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
    bc_gate = stats.get("bimodality_gate")
    if bc_gate == "bimodal":
        prior += (
            "\n\nBC gate (ENFORCED): BC indicates a bimodal distribution; you must select "
            "kmeans (K=2 midpoint). kde is legal only as a justified fallback with a stable "
            "valley; mean_std / mean_std_ratio are not legal choices in this branch."
        )
    elif bc_gate == "unimodal":
        prior += (
            "\n\nBC gate (ENFORCED): BC indicates a unimodal distribution; you must select "
            "mean_std_ratio and supply estimated_anomaly_ratio_percent (anomalies are assumed "
            "rare). kde is legal only as a justified fallback with a stable valley; kmeans / "
            "plain mean_std are not legal choices in this branch."
        )
    else:
        prior += (
            "\n\nBC gate (ENFORCED): the bimodality coefficient is unavailable for this round, "
            "treated the same as unimodal; you must select mean_std_ratio and supply "
            "estimated_anomaly_ratio_percent. kde is legal only as a justified fallback with a "
            "stable valley."
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
        "Analyze or apply a PCC split. With no strategy, return candidates for mean-k*std "
        "(k=0.0..2.0 step 0.1, reference only), K-means (K=2), an estimated-ratio mean-k*std "
        "reference grid, KDE, and a bimodality-coefficient (BC) gate. To apply a split, the "
        "Agent selects kmeans, mean_std_ratio, or kde; plain mean_std remains legal only for "
        "preregistered fixed_policy/experimental_control decisions, not adaptive agent choices. "
        "For adaptive agent decisions the BC gate is enforced: BC>=0.555 requires kmeans; "
        "BC<0.555 or unavailable requires mean_std_ratio (estimated_anomaly_ratio_percent "
        "converted via mean - norm.ppf(1-ratio/100)*std); kde is legal only as a justified "
        "fallback with a stable valley. PCC is interpreted using the partition prior and the "
        "current score evidence. KDE is unavailable when no stable valley is detected; it "
        "never falls back to a median or quantile. "
        "All supported strategies use a single keep/gray boundary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "strategy": {"type": "string", "enum": list(STRATEGIES),
                         "description": "Provide to apply a split using this statistical candidate strategy; omit for analysis mode."},
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
            "estimated_anomaly_ratio_percent": {"type": "number", "minimum": ANOMALY_RATIO_MIN_PERCENT,
                                                 "maximum": ANOMALY_RATIO_MAX_PERCENT,
                                                 "description": (
                                                     f"For mean_std_ratio: estimated percent of anomalous samples (assumed rare), in "
                                                     f"[{ANOMALY_RATIO_MIN_PERCENT}, {ANOMALY_RATIO_MAX_PERCENT}]. The runtime converts this "
                                                     "to k = norm.ppf(1 - ratio/100) and computes threshold = mean - k*std; do not pick k directly."
                                                 )},
            "evidence": {
                "type": "object",
                "description": "Required for adaptive application; concise auditable evidence, not private chain-of-thought.",
                "properties": {
                    "distribution_shape": {"type": "string"},
                    "prior_assumptions_and_uncertainty": {"type": "string"},
                    "candidate_comparison": {"type": "string", "description": "Required when applying a split: compare the chosen candidate against the alternatives (mean-k*std, kmeans, kde) and the BC gate."},
                    "main_risk": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "kde_fallback_justification": {"type": "string", "description": "Required only when strategy=kde: explain why kmeans (bimodal) or mean_std_ratio (unimodal/unavailable) do not fit the visible evidence."},
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
        agent_visible_artifacts = [
            {"name": name, "kind": "image", "purpose": f"Partition {strategy} candidate PCC scatter plot"}
            for strategy, name in plot_artifacts.items()
        ]
        if strategy is None:
            summary = {"mode": "analyze", "score_contract": score_contract(),
                       "n_samples": len(records), "score_stats": stats, "candidates": candidates,
                       "previous_round_vlm_feedback": previous_vlm_feedback,
                       "partition_plots": plot_artifacts, "plot_errors": plot_errors,
                       "partition_prior_prompt": prior,
                       "agent_visible_artifacts": agent_visible_artifacts}
            record_observation(s, "partition", summary, workspace_dir=workspace_dir, branch=branch)
            _save(workspace_dir, s, branch=branch)
            return json.dumps(summary, ensure_ascii=False)

        proposed = {"strategy": strategy}
        for key, value in (("mean_std_k", mean_std_k), ("kmeans_k", kmeans_k),
                           ("kmeans_boundary", kmeans_boundary),
                           ("kde_bandwidth_scale", kde_bandwidth_scale),
                           ("kde_valley_index", kde_valley_index),
                           ("estimated_anomaly_ratio_percent", estimated_anomaly_ratio_percent)):
            if value is not None:
                proposed[key] = value
        effective, source = effective_action(s, "partition", proposed)
        strategy = effective.get("strategy")
        if strategy not in STRATEGIES:
            raise ValueError("Partition strategy must be mean_std, kmeans, kde, or mean_std_ratio")
        if source.startswith("agent") and not str(rationale or "").strip():
            raise ValueError("Adaptive partition decisions require an observation-based rationale")
        evidence = evidence if isinstance(evidence, dict) else {}
        if source.startswith("agent"):
            required_evidence = [
                "distribution_shape", "prior_assumptions_and_uncertainty",
                "candidate_comparison", "main_risk", "confidence",
            ]
            missing = [key for key in required_evidence if not str(evidence.get(key) or "").strip()]
            if missing:
                raise ValueError("Adaptive partition evidence is incomplete: " + ", ".join(missing))
            if evidence.get("confidence") not in ("low", "medium", "high"):
                raise ValueError("Adaptive partition evidence.confidence must be low, medium, or high")
        if source == "agent":
            self._enforce_bc_gate(strategy, stats.get("bimodality_gate"), evidence,
                                   (candidates.get("kde") or {}).get("status"))
        if source == "fixed_policy" and not str(rationale or "").strip():
            rationale = "Preregistered fixed partition strategy"
        threshold, params, candidate_id = self._select_candidate(strategy, effective, candidates, stats)
        keep = [r for r in records if float(r["normality_score"]) >= threshold]
        gray = [r for r in records if float(r["normality_score"]) < threshold]
        removal_percent = len(gray) / max(1, len(records)) * 100.0
        effective_params = {"strategy": strategy, **params}
        if evidence:
            effective_params["evidence"] = evidence
        s["latest_partition"] = {
            "threshold": threshold,
            "score_contract": score_contract(), "threshold_method": strategy,
            "strategy": strategy,
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
                         "previous_round_vlm_feedback": previous_vlm_feedback,
                         "candidates": candidates, "evidence": evidence},
        )
        s["round_status"] = "partitioned"
        summary = {"mode": "split", "score_contract": score_contract(), "strategy": strategy,
                   "strategy_params": params, "candidate_id": candidate_id,
                   "threshold_applied": threshold, "keep_count": len(keep),
                   "gray_count": len(gray),
                   "removal_percent": round(removal_percent, 5),
                   "keep_ratio": round(len(keep) / max(1, len(records)), 5),
                   "gray_ratio": round(len(gray) / max(1, len(records)), 5),
                   "score_stats": stats, "candidates": candidates,
                   "previous_round_vlm_feedback": previous_vlm_feedback,
                   "evidence": evidence,
                   "partition_plots": plot_artifacts, "plot_errors": plot_errors,
                   "partition_prior_prompt": prior,
                   "agent_visible_artifacts": agent_visible_artifacts}
        record_observation(s, "partition", summary, workspace_dir=workspace_dir,
                           branch=branch, decision=decision_entry)
        append_ledger(s, {"stage": "partition", "round": s.get("round"), "strategy": strategy,
                          "threshold": threshold, "keep": len(keep), "gray": len(gray)})
        _save(workspace_dir, s, branch=branch)
        return json.dumps(summary, ensure_ascii=False)

    def _compute_candidates(self, scores):
        candidates = {"mean_std": self._mean_std_candidates(scores),
                      "kmeans": self._kmeans_candidates(scores), "kde": self._kde_candidates(scores),
                      "mean_std_ratio": self._mean_std_ratio_candidates(scores),
                      "selection_note": "Apply the partition prior using the current candidates and current evidence."}
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
            "mean_std_ratio": f"pcc_partition_mean_std_ratio_r{int(round_index)}.png",
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
            "mean_std_ratio": [
                {
                    "threshold": item["threshold"],
                    "label": f"ratio={item['estimated_anomaly_ratio_percent']:.1f}% (k={item['derived_k']:.2f})",
                    "color": "green" if item["estimated_anomaly_ratio_percent"] == 1.0 else "#555555",
                    "linestyle": ":" if item["estimated_anomaly_ratio_percent"] == 1.0 else "--",
                }
                for item in candidates.get("mean_std_ratio", [])
            ],
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

    def _mean_std_ratio_candidates(self, scores):
        from scipy.stats import norm
        mean, std = float(np.mean(scores)), float(np.std(scores))
        out = []
        for ratio in ANOMALY_RATIO_DISPLAY_VALUES:
            k = float(norm.ppf(1.0 - ratio / 100.0))
            threshold = mean - k * std
            out.append({
                "candidate_id": f"mean_std_ratio:ratio={ratio:.2f}",
                "strategy": "mean_std_ratio",
                "estimated_anomaly_ratio_percent": ratio,
                "derived_k": round(k, 6),
                **self._partition_counts(scores, threshold),
            })
        return out

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

    @staticmethod
    def _enforce_bc_gate(strategy, bc_gate, evidence, kde_status):
        """Raise ValueError if strategy violates the enforced BC gate.

        Only called for source == "agent" (pure adaptive tactical choice); see run().
        """
        if strategy == "kde":
            justification = str((evidence or {}).get("kde_fallback_justification") or "").strip()
            if not justification:
                raise ValueError(
                    "kde is a fallback only: adaptive partition decisions choosing kde require "
                    "evidence.kde_fallback_justification explaining why the primary strategy for "
                    "this BC branch (kmeans for bimodal, mean_std_ratio for unimodal/unavailable) "
                    "does not fit the visible evidence"
                )
            if kde_status != "stable_valley":
                raise ValueError(
                    "kde fallback requires a stable KDE valley (candidates.kde.status == 'stable_valley')"
                )
            return
        if bc_gate == "bimodal":
            if strategy != "kmeans":
                raise ValueError(
                    "BC indicates a bimodal distribution (BC >= 0.555): adaptive partition "
                    "decisions must use kmeans (K=2 midpoint), or kde as a justified fallback"
                )
            return
        if strategy != "mean_std_ratio":
            raise ValueError(
                "BC indicates a unimodal distribution or BC is unavailable: adaptive partition "
                "decisions must use mean_std_ratio (estimated_anomaly_ratio_percent), or kde as "
                "a justified fallback"
            )

    def _select_candidate(self, strategy, params, candidates, stats):
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
        if strategy == "mean_std_ratio":
            try:
                from scipy.stats import norm
            except Exception as exc:
                raise ValueError(f"scipy unavailable: {exc}") from exc
            raw_ratio = params.get("estimated_anomaly_ratio_percent")
            if raw_ratio is None:
                raise ValueError(
                    "mean_std_ratio requires estimated_anomaly_ratio_percent in "
                    f"[{ANOMALY_RATIO_MIN_PERCENT}, {ANOMALY_RATIO_MAX_PERCENT}]"
                )
            ratio = float(raw_ratio)
            if not (ANOMALY_RATIO_MIN_PERCENT <= ratio <= ANOMALY_RATIO_MAX_PERCENT):
                raise ValueError(
                    "estimated_anomaly_ratio_percent must be in "
                    f"[{ANOMALY_RATIO_MIN_PERCENT}, {ANOMALY_RATIO_MAX_PERCENT}]"
                )
            k = float(norm.ppf(1.0 - ratio / 100.0))
            mean, std = float(stats["mean"]), float(stats["std"])
            threshold = mean - k * std
            out_params = {
                "estimated_anomaly_ratio_percent": round(ratio, 4),
                "derived_k": round(k, 6),
            }
            return float(threshold), out_params, f"mean_std_ratio:ratio={ratio:.2f}"
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
