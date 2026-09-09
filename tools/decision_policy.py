import time
import uuid


DECISION_FIELDS = {
    "detector": "detector",
    "score": "score",
    "partition": "partition",
    "resolve": "resolve",
    "controller": "controller",
    "transition": "transition",
    "stopping": "stopping",
}


def validate_fixed_policy(policy):
    """Fail fast on an incomplete or internally invalid preregistration."""
    if not isinstance(policy, dict):
        return ["fixed_policy must be an object"]
    required = set(DECISION_FIELDS)
    errors = [f"missing action: {name}" for name in sorted(required - set(policy))]
    for name in required & set(policy):
        if not isinstance(policy[name], dict):
            errors.append(f"{name} must be an object")
    if errors:
        return errors

    detector = policy["detector"]
    for field in ("strategy", "learning_rate", "epochs", "lambda_value", "seed"):
        if field not in detector:
            errors.append(f"detector.{field} is required")
    if detector.get("strategy") not in ("retrain", "reuse", "retrain_first_then_reuse"):
        errors.append("detector.strategy must be retrain, reuse, or retrain_first_then_reuse")
    elif detector.get("strategy") == "reuse":
        errors.append("detector.strategy=reuse has no round-0 checkpoint; use retrain_first_then_reuse")
    try:
        if not 5e-6 <= float(detector["learning_rate"]) <= 5e-4:
            errors.append("detector.learning_rate is outside [5e-6, 5e-4]")
        if not 1 <= int(detector["epochs"]) <= 120:
            errors.append("detector.epochs is outside [1, 120]")
        if detector.get("n_reference_latents") is not None and int(detector["n_reference_latents"]) < 1:
            errors.append("detector.n_reference_latents must be >= 1")
        if not 1 <= int(detector.get("batch_size", 256)) <= 512:
            errors.append("detector.batch_size must be in [1, 512]")
    except (KeyError, TypeError, ValueError):
        pass
    allowed_lambdas = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
    if detector.get("lambda_value") not in allowed_lambdas:
        errors.append("detector.lambda_value is invalid")
    if "steer_lambda" in detector:
        errors.append("detector.steer_lambda was removed with the old steering-prediction head")
    if policy["score"].get("method") != "pcc" or "alpha" in policy["score"]:
        errors.append("score must preregister method=pcc without alpha (reconstruction agreement only)")
    partition = policy["partition"]
    strategy = partition.get("strategy")
    if strategy not in ("mean_std", "kmeans", "kde"):
        errors.append("partition.strategy must be mean_std, kmeans, or kde")
    if strategy == "mean_std":
        try:
            k = float(partition["mean_std_k"])
            if not 0 <= k <= 2 or abs(k * 10 - round(k * 10)) > 1e-7:
                errors.append("partition.mean_std_k must be a 0.1 grid value in [0, 2]")
        except (KeyError, TypeError, ValueError):
            errors.append("partition.mean_std_k is required for mean_std")
    if strategy == "kmeans":
        if partition.get("kmeans_k") != 2:
            errors.append("partition.kmeans_k must be 2")
        boundary = partition.get("kmeans_boundary")
        if boundary != "only":
            errors.append("K=2 requires kmeans_boundary=only")
    if strategy == "kde":
        try:
            scale = float(partition["kde_bandwidth_scale"])
            if scale not in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
                errors.append("partition.kde_bandwidth_scale must be 0.50..2.00 in 0.25 steps")
            int(partition["kde_valley_index"])
        except (KeyError, TypeError, ValueError):
            errors.append("partition.kde_bandwidth_scale and kde_valley_index are required for kde")
    if policy["resolve"].get("resolution_policy") not in ("vlm", "auto_keep"):
        errors.append("resolve.resolution_policy is invalid")
    for field in ("resolution_policy", "budget", "sampling_strategy", "accept_confidence"):
        if field not in policy["resolve"]:
            errors.append(f"resolve.{field} is required")
    if policy["resolve"].get("sampling_strategy") not in (
        "pollution_defense", "rare_behavior_recovery", "information_gain", "verification"
    ):
        errors.append("resolve.sampling_strategy is invalid")
    if policy["resolve"].get("accept_confidence") not in ("low", "medium", "high"):
        errors.append("resolve.accept_confidence is invalid")
    try:
        if int(policy["resolve"].get("budget")) < 1:
            errors.append("resolve.budget must be >= 1")
        elif int(policy["resolve"].get("budget")) > 200:
            errors.append("resolve.budget must be <= 200 (fixed per-round VLM cap)")
    except (TypeError, ValueError):
        errors.append("resolve.budget must be an integer")
    for field in ("epochs", "batch_size", "lr", "weight_decay", "validation_fraction", "seed"):
        if field not in policy["controller"]:
            errors.append(f"controller.{field} is required")
    try:
        if int(policy["controller"].get("epochs")) < 1:
            errors.append("controller.epochs must be >= 1")
        elif int(policy["controller"].get("epochs")) > 100:
            errors.append("controller.epochs must be <= 100")
        if int(policy["controller"].get("batch_size")) < 1:
            errors.append("controller.batch_size must be >= 1")
        elif int(policy["controller"].get("batch_size")) > 512:
            errors.append("controller.batch_size must be <= 512")
        if not 1e-6 <= float(policy["controller"].get("lr")) <= 0.01:
            errors.append("controller.lr must be in [1e-6, 0.01]")
        if not 0 <= float(policy["controller"].get("weight_decay")) <= 0.1:
            errors.append("controller.weight_decay must be in [0, 0.1]")
        if not 0.05 <= float(policy["controller"].get("validation_fraction")) <= 0.4:
            errors.append("controller.validation_fraction is outside [0.05, 0.4]")
    except (TypeError, ValueError):
        errors.append("controller numeric fields are invalid")
    if policy["transition"].get("transition_policy") not in ("clean_only", "deploy_collect_merge"):
        errors.append("transition.transition_policy is invalid")
    try:
        if int(policy["stopping"].get("max_rounds")) < 1:
            errors.append("stopping.max_rounds must be >= 1")
    except (TypeError, ValueError):
        errors.append("stopping.max_rounds must be an integer")
    return errors


def effective_action(state, decision, proposed):
    """Resolve an action through the same interface for agent and fixed arms.

    Adaptive tasks use the proposed action. Fixed baselines use the preregistered
    fixed_policy entry, preventing the LLM from accidentally changing the arm.
    """
    if decision != "stopping" and state.get("termination_required"):
        raise ValueError("Task is completed by a terminal stop decision; create a new task")
    max_rounds = (state.get("constraints") or {}).get("max_rounds")
    if (
        decision != "stopping"
        and max_rounds is not None
        and int(state.get("round", 0)) >= int(max_rounds)
    ):
        raise ValueError("Maximum cleaning rounds reached; record the stopping decision")
    mode = state.get("execution_mode", "adaptive_agent")
    if mode == "adaptive_agent":
        controls = (state.get("experimental_controls") or {}).get(decision) or {}
        if controls and not isinstance(controls, dict):
            raise ValueError(f"experimental_controls.{decision} must be an object")
        return {**proposed, **controls}, "agent+experimental_control" if controls else "agent"
    if mode != "fixed_baseline":
        raise ValueError(f"Unknown execution_mode: {mode}")
    policy = state.get("fixed_policy") or {}
    if decision not in policy:
        raise ValueError(
            f"Fixed baseline has no preregistered '{decision}' action; "
            "the baseline spec is incomplete"
        )
    expected = policy[decision]
    if not isinstance(expected, dict):
        raise ValueError(f"fixed_policy.{decision} must be an object")
    return dict(expected), "fixed_policy"


def record_decision(state, decision, proposed, effective, rationale, source, observation=None):
    entry = {
        "decision_id": "decision_" + uuid.uuid4().hex[:16],
        "round": int(state.get("round", 0)),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "source": source,
        "proposed": proposed,
        "effective": effective,
        "rationale": rationale,
        "observation": observation or {},
    }
    state.setdefault("decision_trace", []).append(entry)
    return entry
