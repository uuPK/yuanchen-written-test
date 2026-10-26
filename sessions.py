import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from tools import ToolContext


DB_PATH = Path(__file__).resolve().parent / "data" / "agent.sqlite3"


def _check_scope(*values: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("用户 ID 和 session ID 必须是非空字符串。")


@contextmanager
def _database(db_path: Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=10.0)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    context_json TEXT,
                    last_run_json TEXT,
                    version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                )
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
            if "context_json" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN context_json TEXT")
            yield connection


def _session_record(row: sqlite3.Row) -> dict:
    return {
        "user_id": row["user_id"], "session_id": row["session_id"],
        "messages": json.loads(row["messages_json"]),
        "context_state": json.loads(row["context_json"]) if row["context_json"] else None,
        "last_run": json.loads(row["last_run_json"]) if row["last_run_json"] else None,
        "version": row["version"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def create_session(context: ToolContext) -> dict:
    _check_scope(context.user_id, context.session_id)
    now = datetime.now(timezone.utc).isoformat()
    with _database(context.db_path) as connection:
        try:
            connection.execute(
                "INSERT INTO sessions (user_id, session_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (context.user_id, context.session_id, now, now),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("当前用户已存在该 session，请使用 /use 选择。") from error
        row = connection.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
            (context.user_id, context.session_id),
        ).fetchone()
        return _session_record(row)


def load_session(context: ToolContext) -> dict:
    _check_scope(context.user_id, context.session_id)
    with _database(context.db_path) as connection:
        row = connection.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
            (context.user_id, context.session_id),
        ).fetchone()
        if row is None:
            raise ValueError("当前用户下不存在该 session。")
        return _session_record(row)


def open_session(context: ToolContext) -> dict:
    _check_scope(context.user_id, context.session_id)
    now = datetime.now(timezone.utc).isoformat()
    with _database(context.db_path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO sessions (user_id, session_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (context.user_id, context.session_id, now, now),
        )
        row = connection.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
            (context.user_id, context.session_id),
        ).fetchone()
        return _session_record(row)


def list_sessions(user_id: str, db_path: Path = DB_PATH) -> list[dict]:
    _check_scope(user_id)
    with _database(db_path) as connection:
        rows = connection.execute(
            "SELECT session_id, version, created_at, updated_at, last_run_json FROM sessions "
            "WHERE user_id = ? ORDER BY updated_at DESC, session_id",
            (user_id,),
        ).fetchall()
        return [
            {
                "session_id": row["session_id"], "version": row["version"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "last_run": json.loads(row["last_run_json"]) if row["last_run_json"] else None,
            }
            for row in rows
        ]


def save_session(context: ToolContext, result: dict, expected_version: int) -> None:
    _check_scope(context.user_id, context.session_id)
    if type(expected_version) is not int or expected_version < 0:
        raise ValueError("session 版本必须是非负整数。")
    last_run = {key: result[key] for key in ("run_id", "status", "steps", "error", "trace_path")}
    last_run["compaction_calls"] = result.get("compaction_calls", 0)
    last_run["deduplicated_calls"] = result.get("deduplicated_calls", 0)
    messages_json = json.dumps(result["messages"], ensure_ascii=False, allow_nan=False)
    last_run_json = json.dumps(last_run, ensure_ascii=False, allow_nan=False)
    context_json = json.dumps(result["context_state"], ensure_ascii=False, allow_nan=False) if result.get("context_state") is not None else None
    with _database(context.db_path) as connection:
        cursor = connection.execute(
            "UPDATE sessions SET messages_json = ?, context_json = ?, last_run_json = ?, version = version + 1, updated_at = ? "
            "WHERE user_id = ? AND session_id = ? AND version = ?",
            (messages_json, context_json, last_run_json, datetime.now(timezone.utc).isoformat(),
             context.user_id, context.session_id, expected_version),
        )
        if cursor.rowcount != 1:
            raise ValueError("session 不存在或已被其他窗口更新；本次历史未覆盖写入，请重新加载。执行记录见 trace。")
