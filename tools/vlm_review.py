import json
import base64
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .utils import (_artifact, _get_vlm_client, _build_vlm_prompt, _parse_vlm_response,
                   resolve_vlm_config, record_observation, _reset_vlm_budget_if_new_round,
                   print_progress)


def run_vlm_review(workspace_dir, s, branch, budget, sampling_strategy, accept_confidence,
                   vlm_model=None, vlm_base_url=None, vlm_api_key=None):
    p = s.get("latest_partition")
    if not p:
        raise ValueError("Must run partition before executing VLM review.")

    _reset_vlm_budget_if_new_round(s)

    per_round = s.get("vlm_budget_per_round", 500)
    remaining_budget = per_round - s.get("vlm_budget_used_this_round", 0)

    gray_candidates = p.get("gray", [])
    if not gray_candidates:
        return [], [], 0

    if remaining_budget <= 0:
        raise ValueError("VLM budget exhausted for the current round.")

    n = min(int(budget), remaining_budget, len(gray_candidates))

    def rank(r):
        score = r["anomaly_score"]
        if sampling_strategy == "pollution_defense":
            return -score
        if sampling_strategy == "rare_behavior_recovery":
            return -(score + (0.4 if abs(r["steering"]) >= .35 else 0))
        return -score

    selected = sorted(gray_candidates, key=rank)[:n]
    cfg = resolve_vlm_config(vlm_model, vlm_base_url, vlm_api_key, state_vlm=s.get("vlm"))
    client = _get_vlm_client(cfg)
    levels = {"low": 0, "medium": 1, "high": 2}
    accepted, reviewed = [], []

    total_steps = len(selected)
    step = 0
    last_pct = -10.0
    for idx, r in enumerate(selected):
        img_full_path = Path(workspace_dir) / r["image"]
        cv_img = cv2.imread(str(img_full_path))
        if cv_img is None:
            pil_img = Image.open(img_full_path).convert('RGB')
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode('.jpg', cv_img)
        b64_img = base64.b64encode(buf).decode('utf-8')

        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _build_vlm_prompt(r["steering"])},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}], max_tokens=512, temperature=0.1
            )
            vlm_label, conf, raw_dict = _parse_vlm_response(response.choices[0].message.content)
        except Exception as e:
            vlm_label, conf, raw_dict = "anomalous", "low", {"reasoning": f"vLLM endpoint call error: {str(e)}"}

        is_clean = (vlm_label == "normal")
        accept = is_clean and levels.get(conf, 0) >= levels.get(accept_confidence, 0)

        reviewed.append({
            "sample_id": r["id"], "source": r["source"],
            "label": "keep" if is_clean else "discard", "confidence": conf,
            "reason": raw_dict.get("reasoning", f"{cfg['model']} review assessment"), "accepted": accept
        })
        if accept:
            accepted.append(r)

        step += 1
        pct = step / total_steps * 100
        if pct - last_pct >= 10.0 or step == total_steps:
            print_progress(f"[VLM Review] Progress {step}/{total_steps} ({pct:3.0f}%)")
            last_pct = pct

    vlm_review_path = _artifact(workspace_dir, f"vlm_review_r{s['round'] + 1}.json", branch=branch)
    vlm_review_path.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2))

    s["vlm_budget_used_this_round"] += n
    record_observation(s, "vlm", {
        "reviewed": n, "accepted": len(accepted),
        "used_this_round": s["vlm_budget_used_this_round"],
        "remaining_this_round": per_round - s["vlm_budget_used_this_round"],
        "strategy": sampling_strategy, "review_artifact": vlm_review_path.name
    })
    return accepted, reviewed, n
