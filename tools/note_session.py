import json
from .base import Tool
from .io import _load_session, _save_session

class NoteSession(Tool):
    name = "note_session"
    description = (
        "Read and update per-task session working memory (goal, hypothesis, open questions, learnings, status). "
        "Omit all parameters to perform a read-only query of session memory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Goal of current task/experiment."
            },
            "hypothesis": {
                "type": "string",
                "description": "Current experimental hypothesis."
            },
            "status": {
                "type": "string",
                "description": "Current workflow status (e.g. understanding, proposing, executing, reflecting, completed)."
            },
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Replace the entire open questions list."
            },
            "learnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Replace the entire learnings list."
            },
            "add_open_question": {
                "type": "string",
                "description": "Append a single open question."
            },
            "add_learning": {
                "type": "string",
                "description": "Append a single learning/finding."
            }
        },
        "required": []
    }

    def run(self, goal=None, hypothesis=None, status=None, open_questions=None,
            learnings=None, add_open_question=None, add_learning=None,
            branch="main", workspace_dir=None, **_):
        s = _load_session(workspace_dir, branch=branch)
        changed = {}
        if goal is not None:
            s["goal"] = goal; changed["goal"] = goal
        if hypothesis is not None:
            s["hypothesis"] = hypothesis; changed["hypothesis"] = hypothesis
        if status is not None:
            s["status"] = status; changed["status"] = status
        if open_questions is not None:
            s["open_questions"] = [str(q) for q in open_questions]
            changed["open_questions"] = s["open_questions"]
        if learnings is not None:
            s["learnings"] = [str(x) for x in learnings]
            changed["learnings"] = s["learnings"]
        if add_open_question:
            s["open_questions"].append(add_open_question)
            changed["open_questions"] = s["open_questions"]
        if add_learning:
            s["learnings"].append(add_learning)
            changed["learnings"] = s["learnings"]

        if changed:
            _save_session(workspace_dir, branch, s)
        return json.dumps({"session": s, "changed": changed}, ensure_ascii=False)
