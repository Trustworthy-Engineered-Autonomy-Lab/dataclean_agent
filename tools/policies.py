POLICIES = {
    "score": {
        "robust_composite": "Within-round robust composite of image reconstruction and image-to-steering prediction errors",
    },
    "partition": {
        "data_driven": "Unsupervised two-step partition: analyze candidates, then choose keep/gray lower threshold and optional gray/discard upper threshold.",
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
        "retrain_first_then_reuse": "Train on D_0, then reuse the same detector in later rounds",
    },
    "train_controller": {
        "default": "Default controller training policy",
    },
    "deploy_controller": {
        "physical_upload": "Upload the trained ONNX controller to the configured physical car host",
    },
    "eval_controller": {
        "physical_collect": "Run the deployed controller on the car and collect a task-local evaluation dataset",
    },
    "transfer_eval_results": {
        "collection_artifact": "Transfer the car evaluation dataset back as a verified CollectionArtifact",
    },
    "transition": {
        "clean_only": "Commit D_(t+1)=C_t",
        "deploy_collect_merge": "Commit D_(t+1)=C_t union newly collected N_t",
    },
}

DEFAULT_PIPELINE = {
    "train_detector": "retrain",
    "score": "robust_composite",
    "partition": "data_driven",
    "resolve": "vlm",
    "evaluate": "openloop",
    "train_controller": "default",
    "deploy_controller": "physical_upload",
    "eval_controller": "physical_collect",
    "transfer_eval_results": "collection_artifact",
    "transition": "clean_only",
}

STAGE_ORDER = [
    "train_detector", "score", "partition", "resolve", "evaluate", "train_controller",
    "deploy_controller", "eval_controller", "transfer_eval_results", "transition",
]

STAGE_LABEL = {
    "train_detector": "Detector Training",
    "score": "Scoring",
    "partition": "Partitioning",
    "resolve": "Resolve Clean Dataset",
    "evaluate": "Open-Loop Evaluation",
    "train_controller": "Controller Training",
    "deploy_controller": "Physical Controller Deployment",
    "eval_controller": "Physical Car Evaluation and Collection",
    "transfer_eval_results": "Evaluation Dataset Transfer",
    "transition": "Next-round Dataset Commit",
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
    "partition": [("threshold", "th"), ("keep", "keep"), ("gray", "gray"), ("discard", "discard")],
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
