"""Versioned IROS2026 detector and score semantics (no ML dependencies)."""

import math

DETECTOR_ARCHITECTURE = "iros2026-action-conditioned-cae-224-v1"
SCORE_CONTRACT_VERSION = "iros2026-pcc-normality-v1"


def score_contract():
    return {
        "version": SCORE_CONTRACT_VERSION,
        "field": "normality_score",
        "formula": "PCC(image, reconstruction)",
        "range": [-1.0, 1.0],
        "higher_is": "normal",
        "keep_rule": "normality_score >= threshold",
        "gray_rule": "normality_score < threshold; optionally >= gray_lower_threshold",
        "discard_rule": "normality_score < gray_lower_threshold (only if specified)",
        "calibrated_probability": False,
    }


def normality_scores(records):
    """Reject legacy composite artifacts instead of silently reversing their meaning."""
    if not records:
        raise ValueError("Score artifact is empty")
    values = []
    for record in records:
        if record.get("score_contract_version") != SCORE_CONTRACT_VERSION:
            raise ValueError(
                "Legacy/incompatible score artifact: rescore with the IROS2026 detector. "
                "Old composite anomaly thresholds cannot be reused as PCC thresholds."
            )
        score = float(record["normality_score"])
        pcc = float(record["pcc"])
        if not math.isfinite(score) or not -1.0 <= score <= 1.0 or score != pcc:
            raise ValueError("normality_score must be finite raw PCC in [-1, 1]")
        values.append(score)
    return values


def require_partition_contract(partition):
    if (partition.get("score_contract") or {}).get("version") != SCORE_CONTRACT_VERSION:
        raise ValueError(
            "This partition uses the legacy anomaly-score convention. "
            "Train/score and partition with the IROS2026 detector before resolution."
        )


def review_rank(record, strategy, threshold):
    """Lower PCC is riskier; distance-based sampling is orientation invariant."""
    score = float(record["normality_score"])
    distance = abs(score - threshold)
    if strategy == "pollution_defense":
        return score
    if strategy == "rare_behavior_recovery":
        return (0 if abs(record["steering"]) >= .35 else 1, distance)
    if strategy == "information_gain":
        return distance
    if strategy == "verification":
        return -distance
    raise ValueError(f"Unknown sampling_strategy: {strategy}")
