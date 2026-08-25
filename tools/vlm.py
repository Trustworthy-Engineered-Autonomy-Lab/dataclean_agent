import json
import os
import re
from pathlib import Path
try:
    from openai import OpenAI
except ImportError:  # Core dataset/state tools can run without VLM dependencies.
    OpenAI = None

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
        "model":    model or sv.get("model") or os.environ.get("DATACLEAN_VLM_MODEL") or st.get("vlm_model") or VLLM_MODEL_NAME,
        "base_url": base_url or sv.get("base_url") or os.environ.get("DATACLEAN_VLM_BASE_URL") or st.get("vlm_base_url") or VLLM_BASE_URL,
        "api_key":  api_key or os.environ.get("DATACLEAN_VLM_API_KEY") or st.get("vlm_api_key") or DEFAULT_VLM_API_KEY,
    }


def _get_vlm_client(cfg=None):
    if OpenAI is None:
        raise RuntimeError("The openai package is required for VLM review")
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
    return f"""You are a conservative auditor for autonomous-driving image/action data. Assess only evidence visible in this single frame. Do not mark rare turns or obstacle avoidance anomalous merely because the steering magnitude is large. If temporal context, speed, calibration, or road geometry is insufficient to justify either decision, return unresolved.

Context: Data may include CARLA or physical-car driving, urban roads, lane changes, intersections, obstacles, parked vehicles, and diverse weather. The sample was selected by an unsupervised detector and is not known to be anomalous.

DECISION LOGIC (in order):

Step 1 - Obstacle analysis:
  Is there an obstacle/hazard on the drivable path? Assess proximity:
    NONE   : no obstacle or off to the side / not blocking
    FAR    : visible but distant (upper third, small, not imminent)
    NEAR   : large and close, occupying lower/central area, imminent threat

Step 2 - Road/lane geometry:
  Analyze the road structure in the LOWER-MIDDLE of image (nearest the car):
    Position: far_left / left / center / right / far_right
    Heading : curves_left / straight / curves_right

Step 3 - Expected steering:
  Determine the plausible steering range given the scene:
    No/far obstacle + center lane     -> ~[-0.2, +0.2]  (follow line)
    No/far obstacle + left lane       -> ~[-0.6, -0.15] (follow line)
    No/far obstacle + right lane      -> ~[+0.15, +0.6] (follow line)
    NEAR obstacle (typical avoidance) -> ~[+0.6, +1.0] or ~[-1.0, -0.6] (open side)
    Steering with SAME sign as expected direction OR within ±0.3 = PLAUSIBLE

Step 4 - Consistency verdict (be LENIENT):
  CONSISTENT   : recorded steering falls in expected range or has correct sign
  MISBEHAVIOR  : opposite sign to expected direction with clear scene cue,
                 OR grossly wrong (e.g., straight into near obstacle)
  UNCERTAIN    : borderline, ambiguous geometry, or poor visibility

Recorded steering: {steering_desc} (range [-1, 1], negative=left, positive=right)

Return JSON format strictly without extra text:
{{
  "obstacle": {{"present": true/false, "proximity": "none|far|near", "position": "left|center|right"}},
  "road_geometry": {{"visible": true/false, "position": "far_left|left|center|right|far_right", "heading": "curves_left|straight|curves_right"}},
  "expected_steering_range": [min, max],
  "steering_justified": true|false,
  "label": "normal|anomalous|unresolved",
  "confidence": "high|medium|low",
  "reasoning": "<one sentence: obstacle proximity, geometry, and steering consistency>"
}}"""


def _parse_vlm_response(result_text: str):
    cleaned = re.sub(r"```(?:json)?", "", result_text).strip()
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not json_match:
        return "unresolved", "low", {"reasoning": "Failed to parse JSON response"}
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return "unresolved", "low", {"reasoning": "JSON decode error"}

    label_str = str(data.get("label", "unresolved")).lower()
    conf_str = str(data.get("confidence", "low")).lower()
    if label_str not in {"normal", "anomalous", "unresolved"}:
        label_str = "unresolved"
    if conf_str not in {"high", "medium", "low"}:
        conf_str = "low"

    reasoning = data.get("reasoning", "VLM assessment")
    return label_str, conf_str, {"reasoning": reasoning, **data}
