import json
import re
from pathlib import Path
from openai import OpenAI

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "./models/Qwen3-VL-8B-Instruct"
DEFAULT_VLM_API_KEY = "vllm-is-awesome"

__all__ = [
    "VLLM_BASE_URL", "VLLM_MODEL_NAME", "DEFAULT_VLM_API_KEY",
    "resolve_vlm_config", "_get_vlm_client",
    "_steering_to_text", "_build_vlm_prompt", "_parse_vlm_response",
]


def _vlm_settings():
    p = Path(__file__).resolve().parent.parent / "settings.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    return {k: d[k] for k in ("vlm_model", "vlm_base_url", "vlm_api_key") if d.get(k)}


def resolve_vlm_config(model=None, base_url=None, api_key=None, state_vlm=None):
    sv = state_vlm or {}
    st = _vlm_settings()
    return {
        "model":    model or sv.get("model") or st.get("vlm_model") or VLLM_MODEL_NAME,
        "base_url": base_url or sv.get("base_url") or st.get("vlm_base_url") or VLLM_BASE_URL,
        "api_key":  api_key or sv.get("api_key") or st.get("vlm_api_key") or DEFAULT_VLM_API_KEY,
    }


def _get_vlm_client(cfg=None):
    if cfg is None:
        cfg = resolve_vlm_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def _steering_to_text(steering: float) -> str:
    s = float(steering)
    def fmt(label): return f"{s:.4f} ({label})"
    if abs(s) < 0.05:  return fmt("straight - typical lane keeping")
    if abs(s) < 0.15:  return fmt("slight %s - lane adjustment or gentle curve" % ("right turn" if s > 0 else "left turn"))
    if abs(s) < 0.35:  return fmt("moderate %s - curve, lane change, or overtake" % ("right turn" if s > 0 else "left turn"))
    if abs(s) < 0.65:  return fmt("sharp %s - sharp turn, intersection, or hazard avoidance" % ("right turn" if s > 0 else "left turn"))
    return fmt("extreme %s - extreme turn or aggressive maneuver" % ("right turn" if s > 0 else "left turn"))


def _build_vlm_prompt(steering: float) -> str:
    steering_desc = _steering_to_text(steering)
    return f"""You are a strict quality inspector for autonomous driving training datasets. Your task is to evaluate whether the image and steering action pair is clean enough for high-quality imitation learning dataset curation.

Context: Data is sourced from CARLA simulator including urban driving, lane changes, intersections, parked vehicles, and diverse weather conditions. This sample is flagged as a candidate anomaly. Determine if it meets quality standards. When in doubt, reject it.

Steering Command: {steering_desc} (range [-1, 1], negative=left, positive=right)

Return JSON format strictly without extra text:
{{
  "road_geometry": "straight|curve_left|curve_right|intersection|other",
  "car_position": "centered|slightly_off|adjacent_lane|off_road",
  "steering_justified": true|false,
  "label": "normal|anomalous",
  "anomaly_type": "none|visibility_failure|erratic_action|environmental_violation",
  "confidence": "high|medium|low",
  "reasoning": "<concise reason for acceptance or rejection>"
}}"""


def _parse_vlm_response(result_text: str):
    cleaned = re.sub(r"```(?:json)?", "", result_text).strip()
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not json_match:
        return "anomalous", "low", {"reasoning": "Failed to parse JSON response"}
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return "anomalous", "low", {"reasoning": "JSON decode error"}

    label_str = data.get("label", "anomalous").lower()
    conf_str = data.get("confidence", "low").lower()
    return label_str, conf_str, data
