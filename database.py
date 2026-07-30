import json
import sqlite3
import threading
from pathlib import Path

DB_FILENAME = ".agent.db"
_workspace: Path | None = None
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL,
  role TEXT NOT NULL, content TEXT, tool_calls TEXT, tool_call_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS task_conversations (
  task_id TEXT PRIMARY KEY, conversation_id INTEGER NOT NULL
);
"""

def open_workspace(path: Path) -> None:
    global _workspace
    _workspace = path.resolve()

def close_workspace() -> None:
    global _workspace
    _workspace = None

def current_workspace_dir() -> Path | None:
    return _workspace

def current_workspace() -> str | None:
    return str(_workspace) if _workspace else None

def connection() -> sqlite3.Connection:
    if not _workspace:
        raise RuntimeError("no workspace open")
    db = _workspace / DB_FILENAME
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    if str(db) not in conns:
        conns[str(db)] = sqlite3.connect(db)
        conns[str(db)].row_factory = sqlite3.Row
        conns[str(db)].executescript(SCHEMA)
        conns[str(db)].commit()
    return conns[str(db)]

def get_task_chat(task_id):
    con = connection()
    row = con.execute("SELECT conversation_id FROM task_conversations WHERE task_id=?", (task_id,)).fetchone()
    if row:
        return Chat.get(row["conversation_id"])
    chat = Chat.create(title="task:" + task_id)
    con.execute("INSERT INTO task_conversations(task_id, conversation_id) VALUES(?,?)", (task_id, chat.id))
    con.commit()
    return chat

class Chat:
    _locks: dict[int, threading.Lock] = {}
    _guard = threading.Lock()
    def __init__(self, **fields): self.__dict__.update(fields)
    @classmethod
    def create(cls, title="New chat"):
        c = connection(); cur = c.execute("INSERT INTO conversations(title) VALUES(?)", (title,)); c.commit()
        return cls.get(cur.lastrowid)
    @classmethod
    def all(cls):
        return [cls(**dict(r)) for r in connection().execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()]
    @classmethod
    def get(cls, chat_id):
        row = connection().execute("SELECT * FROM conversations WHERE id=?", (chat_id,)).fetchone()
        return cls(**dict(row)) if row else None
    def lock(self):
        with self._guard:
            return self._locks.setdefault(self.id, threading.Lock())
    def append(self, role, content=None, tool_calls=None, tool_call_id=None):
        c = connection(); c.execute("INSERT INTO messages(conversation_id,role,content,tool_calls,tool_call_id) VALUES(?,?,?,?,?)", (self.id, role, content, json.dumps(tool_calls) if tool_calls else None, tool_call_id)); c.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (self.id,)); c.commit()
    def messages(self):
        out=[]
        for r in connection().execute("SELECT role,content,tool_calls,tool_call_id FROM messages WHERE conversation_id=? ORDER BY id", (self.id,)):
            m={"role":r["role"],"content":r["content"]}
            if r["tool_calls"]: m["tool_calls"]=json.loads(r["tool_calls"])
            if r["tool_call_id"]: m["tool_call_id"]=r["tool_call_id"]
            out.append(m)
        return out
