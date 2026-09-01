POLICIES = {
    "score": {
        "pcc": "IROS2026 raw reconstruction PCC as weak reconstruction-agreement evidence; no class-label interpretation, composite weighting, or normalization",
    },
    "partition": {
        "data_driven": "Analyze PCC candidates from mean-k*std, K-means (K=2), and KDE using the partition threshold prior; use a single keep/gray boundary and a fixed 30% maximum plausible anomaly ratio.",
    },
    "resolve": {
        "vlm": "Gray zone reviewed by local VLM, keep + accepted samples form clean dataset",
        "auto_keep": "No VLM, retain keep zone, drop gray zone",
        "inspect_only": "Gray zone reviewed by local VLM without outputting clean dataset (audit branch)",
    },
    "evaluate": {
        "data_report": "Plot all D_t PCC scores and report anonymous source retention from final C_t; no labels or quality metrics",
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

# Adaptive tasks must not inherit tactical choices merely because the actions
# exist.  A populated stage->policy mapping made the model treat "retrain",
# "vlm", and "clean_only" as an implicit fixed script.  Explicit pipeline
# preferences remain supported, while a normal adaptive task starts empty and
# reasons over the capability graph below.
DEFAULT_PIPELINE = {}

# State files created before the adaptive capability-graph migration contain
# this automatically generated mapping even though the user never selected it.
# Keep it recognizable so Agent context can avoid treating legacy defaults as
# an explicit preregistration.
LEGACY_ADAPTIVE_DEFAULT_PIPELINE = {
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

CAPABILITY_GRAPH = {
    "kind": "adaptive_experiment_graph",
    "dependencies": {
        "train_detector": ["round_input"],
        "score_and_fit": ["detector_ready"],
        "partition": ["scored"],
        "resolve": ["partitioned"],
        "evaluate": ["resolved_clean_dataset", "matching_round_scores"],
        "train_controller": ["resolved_clean_dataset"],
        "deploy_controller": ["trained_controller"],
        "eval_controller": ["deployed_controller"],
        "transfer_eval_results": ["completed_deployment_run"],
        "commit_round": ["resolved_clean_dataset"],
    },
    "round_outputs": {
        "clean_only": "D_(t+1)=C_t",
        "deploy_collect_merge": "D_(t+1)=C_t union N_t",
    },
    "notes": [
        "Dependencies define legal action order, not a mandatory checklist.",
        "Controller training, deployment, collection, and round advancement are driven by the conversational goal.",
    ],
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
    "evaluate": "PCC and Source Retention Report",
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
LEDGER_FIELDS = {
    "partition": [("threshold", "th"), ("keep", "keep"), ("gray", "gray")],
}


def policies_payload():
    return {
        "stages": STAGE_ORDER,
        "stage_label": STAGE_LABEL,
        "policies": POLICIES,
        "default_pipeline": DEFAULT_PIPELINE,
        "capability_graph": CAPABILITY_GRAPH,
        "task_types": TASK_TYPES,
        "task_type_label": TASK_TYPE_LABEL,
    }


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


def capability_graph_payload():
    return {
        "kind": CAPABILITY_GRAPH["kind"],
        "dependencies": {
            key: list(value)
            for key, value in CAPABILITY_GRAPH["dependencies"].items()
        },
        "round_outputs": dict(CAPABILITY_GRAPH["round_outputs"]),
        "notes": list(CAPABILITY_GRAPH["notes"]),
    }


def agent_pipeline_projection(pipeline, execution_mode="adaptive_agent"):
    pipeline = dict(pipeline or {})
    if execution_mode == "adaptive_agent" and pipeline == LEGACY_ADAPTIVE_DEFAULT_PIPELINE:
        return {}
    return pipeline
