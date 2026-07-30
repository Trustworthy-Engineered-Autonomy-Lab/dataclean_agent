import json
import threading
import queue as _queue
from openai import OpenAI
from tools.utils import set_progress_queue

def _clean_json_raw(raw):
    if not isinstance(raw, str):
        return "{}"
    s = raw.strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    return s or "{}"

def _extract_text_tool_calls(content):
    if not content:
        return []
    import re
    calls = []
    matches = re.findall(r'<tool_call>\s*({.*?})\s*</tool_call>', content, re.DOTALL)
    if not matches:
        matches = re.findall(r'```(?:json)?\s*(\{\s*"name"\s*:.*?\}?)\s*```', content, re.DOTALL)
    for idx, snippet in enumerate(matches):
        try:
            data = json.loads(snippet.strip())
            name = data.get("name") or data.get("function", {}).get("name")
            raw_args = data.get("arguments") or data.get("parameters") or data.get("function", {}).get("arguments") or {}
            if name:
                calls.append({"id": f"call_text_{idx}", "name": name, "arguments": args_str})
        except Exception:
            pass
    return calls

class Turn:
    def __init__(self, client, model, history, tools, context):
        self.client, self.model, self.history, self.tools, self.context = client, model, history, tools, context or {}
        self.stop = threading.Event(); self.messages=[]; self.stopped=False
        self.usage={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    def interrupt(self): self.stop.set()
    def __iter__(self):
        while not self.stop.is_set():
            content=""; calls={}
            with self.client.chat.completions.create(model=self.model, messages=self.history+self.messages, tools=self.tools.all_schemas(), tool_choice="auto", stream=True, stream_options={"include_usage":True}) as stream:
                for chunk in stream:
                    if chunk.usage:
                        for k in self.usage: self.usage[k] += getattr(chunk.usage, k) or 0
                    if not chunk.choices: continue
                    d=chunk.choices[0].delta
                    rc = getattr(d, 'reasoning_content', None)
                    if rc: yield {"type":"thinking","text":rc}
                    if d.content: content += d.content; yield {"type":"content","text":d.content}
                    for tc in d.tool_calls or []:
                        call=calls.setdefault(tc.index,{"id":None,"name":"","arguments":""})
                        if tc.id: call["id"]=tc.id
                        if tc.function and tc.function.name: call["name"] += tc.function.name
                        if tc.function and tc.function.arguments: call["arguments"] += tc.function.arguments
                    if self.stop.is_set(): break
            if self.stop.is_set():
                self.stopped=True
                if content: self.messages.append({"role":"assistant","content":content})
                return
            if not calls:
                extracted = _extract_text_tool_calls(content)
                if extracted:
                    ordered = extracted
                else:
                    self.messages.append({"role":"assistant","content":content}); return
            else:
                ordered=[calls[i] for i in sorted(calls)]
            self.messages.append({"role":"assistant","content":content or None,"tool_calls":[{"id":c["id"],"type":"function","function":{"name":c["name"],"arguments":c["arguments"]}} for c in ordered]})
            for call in ordered:
                raw = _clean_json_raw(call["arguments"])
                try: args=json.loads(raw)
                except json.JSONDecodeError as e:
                    warn = ("[Warning] Tool {} arguments are not valid JSON (retry required): {}. Raw snippet: {}").format(call["name"], e, (raw or "")[:200])
                    result = json.dumps({"error": warn}, ensure_ascii=False)
                    yield {"type":"tool_call","name":call["name"],"args":{}}
                    yield {"type":"tool_result","name":call["name"],"result":result}
                    self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})
                    continue
                yield {"type":"tool_call","name":call["name"],"args":args}
                tool=self.tools.get(call["name"])
                pq=_queue.Queue()
                set_progress_queue(pq)
                _box={"result":None,"exc":None}
                def _run(_tool=tool,_args=args,_box=_box):
                    try:
                        _box["result"]=_tool.run(**{**_args, **self.context}) if _tool else json.dumps({"error":"unknown tool"})
                    except Exception as e:
                        _box["exc"]=e
                th=threading.Thread(target=_run,daemon=True); th.start()
                while th.is_alive():
                    try:
                        line=pq.get(timeout=0.2)
                    except _queue.Empty:
                        if self.stop.is_set(): break
                        continue
                    yield {"type":"tool_progress","name":call["name"],"text":line}
                th.join()
                set_progress_queue(None)
                result = _box["result"] if _box["exc"] is None else json.dumps({"error":str(_box["exc"])})
                yield {"type":"tool_result","name":call["name"],"result":result}
                self.messages.append({"role":"tool","content":result,"tool_call_id":call["id"]})

class Agent:
    def __init__(self, api_key, base_url, model): self.client=OpenAI(api_key=api_key, base_url=base_url); self.model=model
    def run_turn(self, history, tools, context=None): return Turn(self.client, self.model, history, tools, context)
