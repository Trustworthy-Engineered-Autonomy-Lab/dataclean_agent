POLICIES = {
    "score": {
        "composite_stats": "Composite anomaly score based on mean/std",
    },
    "partition": {
        "data_driven": "Unsupervised two-step partition: Step 1 (no threshold) returns Otsu/KDE valley/bimodal coefficient candidates + BC scalar + score stats; Step 2 specifies threshold to execute partition.",
    },
    "resolve": {
        "vlm": "Gray zone reviewed by local VLM, keep + accepted samples form clean dataset",
        "auto_keep": "No VLM, retain keep zone, drop gray zone",
        "inspect_only": "Gray zone reviewed by local VLM without outputting clean dataset (audit branch)",
    },
    "evaluate": {
        "openloop": "Open-loop metrics calculation/visualization (AUC/P/R/F1/purity on full/keep/cleandata)",
    },
    "train_detector": {
        "retrain": "Retrain detector every round",
        "reuse": "Reuse detector from previous round",
    },
    "train_controller": {
        "default": "Default controller training policy",
    },
    "deploy": {
        "cte_proxy": "Deterministic proxy CTE for evaluation",
    },
}

DEFAULT_PIPELINE = {
    "train_detector": "retrain",
    "score": "composite_stats",
    "partition": "data_driven",
    "resolve": "vlm",
    "evaluate": "openloop",
    "train_controller": "default",
    "deploy": "cte_proxy",
}

STAGE_ORDER = ["train_detector", "score", "partition", "resolve", "evaluate", "train_controller", "deploy"]

STAGE_LABEL = {
    "train_detector": "Detector Training",
    "score": "Scoring",
    "partition": "Partitioning",
    "resolve": "Resolve Clean Dataset",
    "evaluate": "Open-Loop Evaluation",
    "train_controller": "Controller Training",
    "deploy": "Deployment",
}

TASK_TYPES = ["threshold_ablation", "detector_ablation", "agent_vs_baseline", "custom"]
TASK_TYPE_LABEL = {
    "main": "Main Branch",
    "threshold_ablation": "Threshold Ablation",
    "detector_ablation": "Detector Ablation",
    "agent_vs_baseline": "Agent vs Baseline",
    "custom": "Custom Experiment",
}
TASK_TYPE_DESC = {
    "threshold_ablation": "Partition threshold policy ablation",
    "detector_ablation": "Detector retrain/reuse and lambda ablation",
    "agent_vs_baseline": "Agent adaptive vs fixed rules baseline comparison",
    "custom": "Custom experiment",
}

LEDGER_FIELDS = {
    "partition": [("threshold", "th"), ("keep", "keep"), ("gray", "gray")],
    "evaluate": [("target", "eval"), ("threshold", "th"), ("n_samples", "n")],
}


def stage_label(stage):
    return STAGE_LABEL.get(stage, stage)


def policy_description(stage, policy):
    return (POLICIES.get(stage) or {}).get(policy, policy)


def task_type_label(tt):
    return TASK_TYPE_LABEL.get(tt, tt)


def describe_task_types():
    return ", ".join(f"{t} ({TASK_TYPE_DESC.get(t, '')})" for t in TASK_TYPES)


def describe_default_pipeline():
    return " -> ".join(STAGE_ORDER)


def policies_payload():
    return {
        "stages": STAGE_ORDER,
        "stage_label": STAGE_LABEL,
        "policies": POLICIES,
        "default_pipeline": DEFAULT_PIPELINE,
        "task_types": TASK_TYPES,
        "task_type_label": TASK_TYPE_LABEL,
    }


def stage_policy(stage, pipeline=None):
    pipeline = pipeline or {}
    if stage in pipeline:
        return pipeline[stage]
    return DEFAULT_PIPELINE.get(stage)


def validate_pipeline(pipeline):
    if not isinstance(pipeline, dict):
        return False, ["pipeline must be a stage->policy dict mapping."]
    errors = []
    for stage, pol in pipeline.items():
        if stage not in POLICIES:
            errors.append(f"Unknown stage: {stage} (options: {list(POLICIES)})")
        elif pol not in POLICIES[stage]:
            errors.append(f"Unknown policy for stage {stage}: {pol} (options: {list(POLICIES[stage])})")
    return (len(errors) == 0), errors


def default_pipeline_for(task_type=None):
    return dict(DEFAULT_PIPELINE)
