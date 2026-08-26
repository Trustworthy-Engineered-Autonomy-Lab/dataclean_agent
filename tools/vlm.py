import hashlib
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
VLM_PROMPT_VERSION = "driving-audit-compact-v2"

__all__ = [
    "VLLM_BASE_URL", "VLLM_MODEL_NAME", "DEFAULT_VLM_API_KEY", "VLM_PROMPT_VERSION",
    "resolve_vlm_config", "_get_vlm_client",
    "_steering_to_text", "_build_vlm_prompt", "_extract_vlm_response",
    "_parse_vlm_response", "vlm_prompt_hash",
]


def _vlm_settings():
    p = Path(__file__).resolve().parent.parent / "settings.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    return {
        k: d[k]
        for k in ("vlm_model", "vlm_base_url", "vlm_api_key", "vlm_max_tokens")
        if d.get(k) is not None and d.get(k) != ""
    }


def _optional_positive_int(value, name):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or null")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer or null") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer or null")
    return parsed


def resolve_vlm_config(model=None, base_url=None, api_key=None, state_vlm=None):
    sv = state_vlm if isinstance(state_vlm, dict) else {}
    st = _vlm_settings()
    max_tokens = _optional_positive_int(
        sv.get("max_tokens")
        if sv.get("max_tokens") is not None
        else os.environ.get("DATACLEAN_VLM_MAX_TOKENS", st.get("vlm_max_tokens")),
        "VLM max_tokens",
    )
    return {
        "model":    model or sv.get("model") or os.environ.get("DATACLEAN_VLM_MODEL") or st.get("vlm_model") or VLLM_MODEL_NAME,
        "base_url": base_url or sv.get("base_url") or os.environ.get("DATACLEAN_VLM_BASE_URL") or st.get("vlm_base_url") or VLLM_BASE_URL,
        "api_key":  api_key or os.environ.get("DATACLEAN_VLM_API_KEY") or st.get("vlm_api_key") or DEFAULT_VLM_API_KEY,
        # The repository settings pin a reproducible cap; task state or the
        # environment may override it for an explicitly designed experiment.
        "max_tokens": max_tokens,
    }


def _get_vlm_client(cfg=None):
    if OpenAI is None:
        raise RuntimeError("The openai package is required for VLM review")
    if cfg is None:
        cfg = resolve_vlm_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _content_part_text(part):
    if isinstance(part, str):
        return part
    text = _field(part, "text")
    if isinstance(text, str):
        return text
    nested = _field(text, "value")
    return nested if isinstance(nested, str) else None


def _extract_vlm_response(response):
    """Normalize a Chat Completions response without treating reasoning as an answer."""
    choices = _field(response, "choices", [])
    if not isinstance(choices, (list, tuple)) or not choices:
        return None, {
            "status": "missing_choice",
            "finish_reason": None,
            "content_type": None,
            "has_reasoning_content": False,
        }

    choice = choices[0]
    message = _field(choice, "message")
    finish_reason = _field(choice, "finish_reason")
    content = _field(message, "content")
    # vLLM renamed reasoning_content to reasoning. Accept both so audit data is
    # correct across the server versions used by the lab.
    reasoning = _field(message, "reasoning")
    if reasoning is None:
        reasoning = _field(message, "reasoning_content")
    metadata = {
        "status": "success",
        "finish_reason": None if finish_reason is None else str(finish_reason),
        "content_type": type(content).__name__,
        # Record only presence. Hidden model reasoning is neither parsed nor persisted.
        "has_reasoning_content": bool(reasoning),
    }

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, (list, tuple)):
        text = "\n".join(
            piece for piece in (_content_part_text(part) for part in content) if piece
        ).strip()
    elif isinstance(content, dict):
        text = (_content_part_text(content) or "").strip()
    elif content is None:
        text = ""
    else:
        metadata["status"] = "unsupported_content_type"
        return None, metadata

    if metadata["finish_reason"] in {"length", "max_tokens"}:
        metadata["status"] = "output_truncated"
        return (text or None), metadata
    if not text:
        metadata["status"] = "empty_content"
        return None, metadata
    return text, metadata


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
    return f"""Audit one autonomous-driving image/steering pair conservatively. Use only visible single-frame evidence; detector selection is not proof of anomaly. Rare turns and obstacle avoidance are not anomalous merely because steering is large. If visibility, geometry, or temporal context is insufficient, choose unresolved.

Inspect the lower-middle drivable area:
1. Obstacle: none/off-path, far/small, or near/blocking; note its side.
2. Road: nearest visible lane position and heading.
3. Plausible steering: center/straight about [-0.2,0.2]; left road about [-0.6,-0.15]; right road about [0.15,0.6]; a near obstacle may justify steering toward the open side up to ±1.0. Correct sign or within ±0.3 is generally plausible.
4. Label anomalous only for clear opposite/grossly unsafe steering; otherwise normal or unresolved.

Recorded steering: {steering_desc}; range [-1,1], negative=left, positive=right.

The total output budget is strict. Finish analysis efficiently and reserve space for the final answer. Return exactly one compact JSON object, no Markdown or extra text. Keep reasoning to one short sentence:
{{
  "obstacle": {{"present": true/false, "proximity": "none|far|near", "position": "left|center|right"}},
  "road_geometry": {{"visible": true/false, "position": "far_left|left|center|right|far_right", "heading": "curves_left|straight|curves_right"}},
  "expected_steering_range": [min, max],
  "steering_justified": true|false,
  "label": "normal|anomalous|unresolved",
  "confidence": "high|medium|low",
  "reasoning": "<one sentence: obstacle proximity, geometry, and steering consistency>"
}}"""


def vlm_prompt_hash():
    """Stable hash for the prompt template independent of sample steering."""
    marker = 0.123456
    template = _build_vlm_prompt(marker).replace(_steering_to_text(marker), "<STEERING>")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _parse_vlm_response(result_text: str):
    if not isinstance(result_text, str):
        return "unresolved", "low", {
            "reasoning": "VLM response content was not text",
            "parse_status": "unsupported_content_type",
        }
    cleaned = re.sub(r"```(?:json)?", "", result_text).strip()
    if not cleaned:
        return "unresolved", "low", {
            "reasoning": "VLM response content was empty",
            "parse_status": "empty_content",
        }
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not json_match:
        return "unresolved", "low", {
            "reasoning": "VLM response did not contain a JSON object",
            "parse_status": "malformed_json",
        }
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return "unresolved", "low", {
            "reasoning": "VLM response contained malformed JSON",
            "parse_status": "malformed_json",
        }
    if not isinstance(data, dict):
        return "unresolved", "low", {
            "reasoning": "VLM JSON output was not an object",
            "parse_status": "invalid_schema",
        }

    label_str = str(data.get("label", "unresolved")).lower()
    conf_str = str(data.get("confidence", "low")).lower()
    if label_str not in {"normal", "anomalous", "unresolved"}:
        return "unresolved", "low", {
            "reasoning": "VLM JSON contained an invalid label",
            "parse_status": "invalid_schema",
        }
    if conf_str not in {"high", "medium", "low"}:
        return "unresolved", "low", {
            "reasoning": "VLM JSON contained an invalid confidence",
            "parse_status": "invalid_schema",
        }

    reasoning = data.get("reasoning", "VLM assessment")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "VLM assessment"
    return label_str, conf_str, {**data, "reasoning": reasoning, "parse_status": "success"}
