import json
import hashlib
import threading
import queue as _queue
import time
import uuid
import re
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from tools.utils import set_progress_queue
from tools.agent_protocol import (
    action_requires_episode,
    authorize_action_call,
    current_execution_fingerprint,
    finish_action_call,
    record_turn_completed,
)

AGENT_PROMPT_VERSION = "conversation-driven-runtime-v6"
_NON_OPERATIONAL_ARGUMENTS = {"rationale", "request_basis"}
_PROTOCOL_MUTATION_TOOLS = {
    "propose_experiment_episode", "assess_experiment_episode",
    "withdraw_experiment_episode", "reconcile_interrupted_action",
}


def _unsupported_request_options(exc, request):
    """Only downgrade options explicitly rejected by a compatible endpoint.

    Invalid tool schemas, authentication, rate limits and server/network errors
    must retain their original error instead of causing an unrelated retry.
    """
    if getattr(exc, "status_code", None) not in (400, 422) and not isinstance(exc, TypeError):
        return []
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    code = str(error.get("code") or getattr(exc, "code", "") or "").lower()
    parameter = str(error.get("param") or getattr(exc, "param", "") or "")
    message = str(error.get("message") or exc).lower()
    if parameter.startswith(("tools", "messages")) or code == "invalid_function_parameters":
        return []
    unsupported = code in {"unsupported_parameter", "unknown_parameter"} or any(
        word in message for word in (
            "unsupported", "not supported", "unknown parameter", "unrecognized",
            "unexpected keyword", "extra inputs are not permitted",
        )
    )
    if not unsupported:
        return []
    return [key for key in ("stream_options", "seed") if key in request and (
        parameter == key or parameter.startswith(key + ".")
        or (not parameter and re.search(r"\b" + key + r"\b", message))
    )]


def _is_state_changing_agent_call(name, args):
    return (
        name == "configure_task_dataset"
        or name in _PROTOCOL_MUTATION_TOOLS
        or action_requires_episode(name, args)
    )


def _operational_call_signature(name, args):
    operational = {
        key: value for key, value in (args or {}).items()
        if key not in _NON_OPERATIONAL_ARGUMENTS
    }
    return name + ":" + json.dumps(
        operational, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _tool_result_envelope(name, result):
    """Project heterogeneous legacy tool payloads onto one runtime contract."""
    parse_error = None
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except Exception:
            payload = {"value": result}
            parse_error = "Tool returned a non-JSON result"
    else:
        payload = result
    if isinstance(payload, dict) and {
        "ok", "status", "data", "error", "retryable", "state_changed", "artifact_refs"
    }.issubset(payload):
        return json.dumps(payload, ensure_ascii=False)
    probe = payload if isinstance(payload, dict) else {}
    explicit = str(probe.get("status", "")).lower()
    failed = bool(
        parse_error or probe.get("error") or probe.get("cancelled") or probe.get("protocol_rejected")
        or probe.get("ok") is False
        or explicit in {"failed", "error", "cancelled", "rejected"}
    )
    error = parse_error or probe.get("error") or probe.get("protocol_error")
    envelope = {
        "ok": not failed,
        "status": "failed" if failed else "succeeded",
        "code": probe.get("code") or f"{name}.{'failed' if failed else 'succeeded'}",
        "data": payload,
        "error": str(error) if error is not None else None,
        "retryable": bool(probe.get("retryable", False)),
        "state_changed": probe.get("state_changed"),
        "artifact_refs": probe.get("artifact_refs") or [],
    }
    return json.dumps(envelope, ensure_ascii=False)


def _classify_tool_result(result):
    """Return one status interpretation shared by streaming UI and trajectory."""
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    try:
        payload = json.loads(text)
    except Exception:
        return False, "failed", "Tool returned a non-JSON result"
    if not isinstance(payload, dict):
        return False, "failed", "Tool result must be a JSON object"
    explicit = str(payload.get("status", "")).lower()
    failed = bool(
        payload.get("error")
        or payload.get("cancelled")
        or payload.get("protocol_rejected")
        or payload.get("ok") is False
        or explicit in {"failed", "error", "cancelled", "rejected"}
    )
    error = payload.get("error") or payload.get("protocol_error")
    return (not failed), (explicit or ("failed" if failed else "succeeded")), error


def _tool_result_retryable(result):
    """Read the retry hint from the normalized runtime envelope."""
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    try:
        payload = json.loads(text)
    except Exception:
        return False
    return bool(payload.get("retryable", False)) if isinstance(payload, dict) else False


def _tool_result_event(name, result):
    ok, status, error = _classify_tool_result(result)
    return {"type": "tool_result", "name": name, "result": result,
            "ok": ok, "status": status, "error": error}

def _clean_json_raw(raw):
    if not isinstance(raw, str):
        return "{}"
    s = raw.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    return s or "{}"


def _public_model_content(content):
    """Remove private reasoning blocks from model-visible output and persistence."""
    if not content:
        return ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
    # Some local backends truncate a reasoning block without its closing tag.
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
    return text.strip()

def _extract_text_tool_calls(content):
    if not content:
        return []
    import re
    calls = []
    blocks = re.findall(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>', content)
    if not blocks:
        blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    for idx, snippet in enumerate(blocks):
        try:
            start = snippet.find("{")
            if start < 0:
                continue
            data, _ = json.JSONDecoder().raw_decode(snippet[start:])
            name = data.get("name") or data.get("function", {}).get("name")
            raw_args = data.get("arguments") or data.get("parameters") or data.get("function", {}).get("arguments") or {}
            if name:
                args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                calls.append({"id": f"call_text_{idx}", "name": name, "arguments": args_str})
        except Exception:
            pass
    return calls

class Turn:
    def __init__(self, client, model, history, tools, context, temperature=None, seed=None,
                 max_model_steps=24, max_tool_calls=40):
        self.client, self.model, self.history, self.tools, self.context = client, model, history, tools, context or {}
        self.temperature, self.seed = temperature, seed
        self.stop = threading.Event(); self.messages=[]; self.stopped=False
        self.paused=False; self.pause_reason=None
        self.usage={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
        self.max_model_steps=max_model_steps; self.max_tool_calls=max_tool_calls
        self.started_monotonic = time.monotonic()
        system_prompt = next((m.get("content", "") for m in history if m.get("role") == "system"), "")
        self.agent_metadata = {
            "turn_id": "turn_" + uuid.uuid4().hex[:16],
            "model": model,
            "temperature": temperature,
            "requested_seed": seed,
            "effective_seed_parameter": seed,
            "prompt_version": AGENT_PROMPT_VERSION,
            "prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.model_steps = 0
        self.tool_calls_used = 0
        # A failure is meaningful only in the execution context in which it
        # occurred. Values keep bounded retry metadata for genuinely transient
        # errors; changed experiment state invalidates the record lazily.
        self.failed_operational_calls = {}

    def _execution_fingerprint(self):
        workspace_dir = self.context.get("workspace_dir")
        branch = self.context.get("branch")
        if not workspace_dir or not branch:
            return None
        try:
            return current_execution_fingerprint(workspace_dir, branch)
        except Exception:
            # If state cannot be observed, do not claim that runtime conditions
            # are unchanged and accidentally suppress a valid action.
            return None

    def _remember_failed_call(self, signature, result):
        fingerprint = self._execution_fingerprint()
        if fingerprint is None:
            return
        previous = self.failed_operational_calls.get(signature)
        # Keep a true bound even when a failed external attempt leaves behind a
        # partial artifact and therefore changes its own fingerprint. A separate
        # successful prerequisite is handled by _failed_retry_block, which sees
        # that the context changed before the next attempt and drops this record.
        attempts = int(previous.get("attempts", 0)) + 1 if previous else 1
        self.failed_operational_calls[signature] = {
            "execution_fingerprint": fingerprint,
            "attempts": attempts,
            "retryable": _tool_result_retryable(result),
            "error": _classify_tool_result(result)[2],
        }

    def _failed_retry_block(self, signature):
        """Return a stale failure only when relevant execution state is unchanged."""
        previous = self.failed_operational_calls.get(signature)
        if not previous:
            return None
        fingerprint = self._execution_fingerprint()
        if fingerprint is None:
            return None
        if previous.get("execution_fingerprint") != fingerprint:
            self.failed_operational_calls.pop(signature, None)
            return None
        # Permit one automatic retry for a tool that explicitly classified its
        # failure as transient. Persistent failures are then replanned normally.
        if previous.get("retryable") and int(previous.get("attempts", 0)) < 2:
            return None
        return previous
    def interrupt(self): self.stop.set()
    def __iter__(self):
        try:
            yield from self._iterate()
        finally:
            self.agent_metadata["duration_seconds"] = round(
                time.monotonic() - self.started_monotonic, 6
            )
            record_turn_completed(
                self.context.get("workspace_dir"),
                self.context.get("branch"),
                self.agent_metadata,
                self.usage,
                self.model_steps,
                self.tool_calls_used,
                self.stopped or self.stop.is_set(),
            )

    def _iterate(self):
        model_steps=0; tool_calls_used=0
        while not self.stop.is_set():
            model_steps += 1
            self.model_steps = model_steps
            if model_steps > self.max_model_steps:
                self.paused = True
                self.pause_reason = "model_step_limit"
                self.agent_metadata["pause_reason"] = self.pause_reason
                msg = (
                    f"本次执行已达到单回合推理步数上限（{self.max_model_steps}）。"
                    "当前实验状态已经保存，这不是任务失败；请发送“继续”以从当前状态恢复。"
                )
                self.messages.append({"role":"assistant", "content":msg})
                yield {"type":"content", "text":msg}
                return
            content=""; calls={}
            try:
                raw_messages = self.history + self.messages
                sanitized_messages = []
                for idx, m in enumerate(raw_messages):
                    if idx > 0 and m.get("role") == "system":
                        sanitized_messages.append({
                            "role": "user",
                            "content": f"[Notice] {m.get('content', '')}"
                        })
                    else:
                        sanitized_messages.append(m)

                request = {
                    "model": self.model,
                    "messages": sanitized_messages,
                    "tools": self.tools.all_schemas(),
                    "tool_choice": "auto",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                if self.seed is not None:
                    request["seed"] = self.seed
                while True:
                    self.agent_metadata["api_request_attempts"] = int(
                        self.agent_metadata.get("api_request_attempts", 0)
                    ) + 1
                    try:
                        stream_context = self.client.chat.completions.create(**request)
                        break
                    except Exception as exc:
                        unsupported = _unsupported_request_options(exc, request)
                        if not unsupported:
                            raise
                        # Each option can be removed at most once. Never strip
                        # tool definitions or change the model to hide an error.
                        fallback = self.agent_metadata.setdefault("request_capability_fallback", [])
                        for key in unsupported:
                            request.pop(key)
                            if key not in fallback:
                                fallback.append(key)
                        if "seed" in unsupported:
                            self.agent_metadata["effective_seed_parameter"] = None
                with stream_context as stream:
                    for chunk in stream:
                        usage = getattr(chunk, "usage", None)
                        if usage:
                            for k in self.usage:
                                self.usage[k] += getattr(usage, k, 0) or 0
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        d=choices[0].delta
                        # reasoning_content is deliberately neither streamed nor
                        # persisted. Only experiment evidence and concise public
                        # rationales belong in the audit trail.
                        delta_content = getattr(d, "content", None)
                        if delta_content:
                            content += delta_content
                        for tc in getattr(d, "tool_calls", None) or []:
                            call=calls.setdefault(tc.index,{"id":None,"name":"","arguments":""})
                            if tc.id: call["id"]=tc.id
                            if tc.function and tc.function.name: call["name"] += tc.function.name
                            if tc.function and tc.function.arguments: call["arguments"] += tc.function.arguments
                        if self.stop.is_set(): break
            except Exception as e:
                err_msg = f"\n[模型请求失败]: {type(e).__name__}: {e}"
                yield {"type": "content", "text": err_msg}
                self.messages.append({"role": "assistant", "content": err_msg})
                return
            if self.stop.is_set():
                self.stopped=True
                content = _public_model_content(content)
                if content:
                    self.messages.append({"role":"assistant","content":content})
                    yield {"type":"content", "text":content}
                return
            content = _public_model_content(content)
            if content:
                yield {"type":"content", "text":content}
            if not calls:
                extracted = _extract_text_tool_calls(content)
                if extracted:
                    ordered = extracted
                else:
                    self.messages.append({"role":"assistant","content":content}); return
            else:
                ordered=[calls[i] for i in sorted(calls)]
            for call in ordered:
                if not call.get("id"):
                    call["id"] = "call_" + uuid.uuid4().hex
            self.messages.append({"role":"assistant","content":content or None,"tool_calls":[{"id":c["id"],"type":"function","function":{"name":c["name"],"arguments":c["arguments"]}} for c in ordered]})
            state_changing_call_seen = False
            for call_index, call in enumerate(ordered):
                if self.stop.is_set():
                    self.stopped = True
                    yield from self._cancel_pending_calls(ordered[call_index:])
                    return
                tool_calls_used += 1
                self.tool_calls_used = tool_calls_used
                if tool_calls_used > self.max_tool_calls:
                    result = json.dumps({
                        "error": f"tool-call limit {self.max_tool_calls} reached",
                        "code": "turn_tool_call_limit",
                        "retryable": True,
                    })
                    result = _tool_result_envelope(call["name"], result)
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    # Complete every tool-call ID emitted in this assistant
                    # message so the persisted Chat Completions history remains
                    # valid when the user resumes in a later turn.
                    for pending in ordered[call_index + 1:]:
                        pending_result = _tool_result_envelope(
                            pending["name"],
                            json.dumps({
                                "error": "Not executed because the per-turn tool-call limit was reached",
                                "code": "turn_tool_call_limit",
                                "retryable": True,
                            }),
                        )
                        yield _tool_result_event(pending["name"], pending_result)
                        self.messages.append({
                            "role": "tool",
                            "content": pending_result,
                            "tool_call_id": pending["id"],
                        })
                    self.paused = True
                    self.pause_reason = "tool_call_limit"
                    self.agent_metadata["pause_reason"] = self.pause_reason
                    msg = (
                        f"本次执行已达到单回合工具调用上限（{self.max_tool_calls}）。"
                        "已完成的实验状态已经保存，这不是任务失败；请发送“继续”以恢复。"
                    )
                    yield {"type": "content", "text": msg}
                    self.messages.append({"role": "assistant", "content": msg})
                    return
                raw = _clean_json_raw(call["arguments"])
                try: args=json.loads(raw)
                except json.JSONDecodeError as e:
                    warn = ("[Warning] Tool {} arguments are not valid JSON (retry required): {}. Raw snippet: {}").format(call["name"], e, (raw or "")[:200])
                    result = json.dumps({"error": warn}, ensure_ascii=False)
                    result = _tool_result_envelope(call["name"], result)
                    yield {"type":"tool_call","name":call["name"],"args":{}}
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    continue
                if not isinstance(args, dict):
                    result = _tool_result_envelope(call["name"], json.dumps({
                        "error": "Tool arguments must decode to a JSON object",
                        "retryable": True,
                    }, ensure_ascii=False))
                    yield {"type":"tool_call","name":call["name"],"args":{}}
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    continue
                yield {"type":"tool_call","name":call["name"],"args":args}
                is_state_changing = _is_state_changing_agent_call(call["name"], args)
                if is_state_changing and state_changing_call_seen:
                    result = _tool_result_envelope(call["name"], json.dumps({
                        "error": (
                            "Only one state-changing action is allowed per model step. "
                            "Observe the earlier result before deciding another mutation."
                        ),
                        "code": "multiple_state_changes_in_model_step",
                        "retryable": True,
                    }, ensure_ascii=False))
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({
                        "role": "tool", "content": result,
                        "tool_call_id": call["id"],
                    })
                    continue
                if is_state_changing:
                    # Reserve the model step even if execution fails. The model
                    # has not observed that failure yet, so a second mutation
                    # emitted in the same response cannot be a valid replan.
                    state_changing_call_seen = True
                operational_signature = _operational_call_signature(call["name"], args)
                failed_retry = self._failed_retry_block(operational_signature)
                if failed_retry is not None:
                    result = _tool_result_envelope(call["name"], json.dumps({
                        "error": (
                            "This exact operation already failed and the relevant execution "
                            "state has not changed. Replan from the previous error or perform "
                            "the missing prerequisite before retrying."
                        ),
                        "code": "unchanged_failed_retry",
                        "retryable": False,
                        "previous_error": failed_retry.get("error"),
                    }, ensure_ascii=False))
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    # This is an observation for the Agent, not a canned answer to
                    # the user. Continue the loop so it can inspect state, satisfy
                    # a prerequisite, choose another action, or explain a genuine
                    # external blocker in the context of the user's request.
                    continue
                try:
                    tool=self.tools.get(call["name"])
                    if not getattr(tool, "agent_exposed", True):
                        raise KeyError(f"Tool '{call['name']}' is not exposed to the Agent")
                except KeyError as e:
                    result=json.dumps({"error":str(e)}, ensure_ascii=False)
                    result = _tool_result_envelope(call["name"], result)
                    self._remember_failed_call(operational_signature, result)
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    continue
                call_metadata = {**self.agent_metadata, "model_step": model_steps,
                                 "tool_call_index": tool_calls_used}
                try:
                    authorization = authorize_action_call(
                        self.context.get("workspace_dir"),
                        self.context.get("branch"),
                        call["name"],
                        args,
                        call_metadata,
                    )
                except Exception as exc:
                    result = json.dumps({"error": str(exc), "protocol_rejected": True}, ensure_ascii=False)
                    result = _tool_result_envelope(call["name"], result)
                    self._remember_failed_call(operational_signature, result)
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    continue
                pq=_queue.Queue()
                _box={"result":None,"exc":None}
                def _run(_tool=tool,_args=args,_box=_box,_pq=pq,_metadata=call_metadata):
                    set_progress_queue(_pq)
                    try:
                        _box["result"]=_tool.run(**{
                            **_args,
                            **self.context,
                            "agent_metadata": _metadata,
                            "cancel_event": self.stop,
                        }) if _tool else json.dumps({"error":"unknown tool"})
                    except Exception as e:
                        _box["exc"]=e
                    finally:
                        set_progress_queue(None)
                th=threading.Thread(target=_run,daemon=True); th.start()
                while th.is_alive():
                    try:
                        line=pq.get(timeout=0.2)
                    except _queue.Empty:
                        if self.stop.is_set(): break
                        continue
                    yield {"type":"tool_progress","name":call["name"],"text":line}
                if self.stop.is_set() and th.is_alive():
                    th.join(timeout=2.0)
                    self.stopped = True
                    still_running = th.is_alive()
                    result = json.dumps({
                        "cancelled": True,
                        "tool_still_stopping": still_running,
                        "requires_reconciliation": still_running,
                        "error": (
                            "Tool cancellation is still in progress; reconcile before another action"
                            if still_running else "Tool was cancelled"
                        ),
                    }, ensure_ascii=False)
                    # If the worker is still alive, retain executing_action. It may
                    # yet have external side effects, so declaring it finished would
                    # make a blind retry possible.
                    if not still_running:
                        result = _tool_result_envelope(call["name"], result)
                        try:
                            finish_action_call(
                                self.context.get("workspace_dir"), self.context.get("branch"),
                                authorization, call["name"], result,
                            )
                        except Exception as exc:
                            result = json.dumps({
                                "error": "Action was interrupted and protocol persistence failed",
                                "protocol_error": str(exc),
                                "requires_reconciliation": True,
                            }, ensure_ascii=False)
                    result = _tool_result_envelope(call["name"], result)
                    yield _tool_result_event(call["name"], result)
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    yield from self._cancel_pending_calls(ordered[call_index + 1:])
                    return
                th.join()
                result = _box["result"] if _box["exc"] is None else json.dumps({"error":str(_box["exc"])})
                result = _tool_result_envelope(call["name"], result)
                try:
                    finish_action_call(
                        self.context.get("workspace_dir"), self.context.get("branch"),
                        authorization, call["name"], result,
                    )
                except Exception as exc:
                    result = json.dumps({
                        "error": "Tool returned but the action outcome was not durably recorded",
                        "tool_result": result,
                        "protocol_error": str(exc),
                        "requires_reconciliation": True,
                    }, ensure_ascii=False)
                result = _tool_result_envelope(call["name"], result)
                if not _classify_tool_result(result)[0]:
                    # Record after protocol persistence: the retry comparison must
                    # use the complete post-attempt execution context.
                    self._remember_failed_call(operational_signature, result)
                yield _tool_result_event(call["name"], result)
                self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
            if self.stop.is_set():
                self.stopped = True
                return

    def _cancel_pending_calls(self, calls):
        """Complete the API message contract without executing queued actions."""
        for call in calls:
            result = _tool_result_envelope(call["name"], {
                "cancelled": True,
                "error": "Not executed because the user interrupted this turn",
                "code": "turn_cancelled",
                "state_changed": False,
            })
            self.messages.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
            yield _tool_result_event(call["name"], result)

class Agent:
    def __init__(self, api_key, base_url, model, temperature=None, seed=None):
        if OpenAI is None:
            raise RuntimeError("The openai package is required to run the language-model agent")
        client_args = {"api_key": api_key or "local-openai-compatible"}
        if base_url:
            client_args["base_url"] = base_url
        self.client=OpenAI(**client_args); self.model=model
        self.temperature, self.seed = temperature, seed
    def run_turn(self, history, tools, context=None):
        return Turn(
            self.client, self.model, history, tools, context,
            temperature=self.temperature, seed=self.seed,
        )
