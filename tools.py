import ast
import json
import math
import operator
import sqlite3
from contextlib import closing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    session_id: str
    db_path: Path = Path(__file__).resolve().parent / "data" / "agent.sqlite3"
    run_id: str | None = None
    tool_call_id: str | None = None
    on_replay: Callable[[], None] | None = None


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
MAX_AST_NODES = 64
MAX_NUMBER = 1e100


def _checked_number(value: int | float) -> int | float:
    if abs(value) > MAX_NUMBER or not math.isfinite(value):
        raise ToolError("NUMERIC_RANGE", "数字或中间结果必须有限，且绝对值不超过 1e100。")
    return value


def _calculate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _checked_number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate_node(node.operand)
        return _checked_number(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ToolError("DIVISION_BY_ZERO", "除数不能为零。")
        value = BINARY_OPERATORS[type(node.op)](left, right)
        return _checked_number(value)
    raise ToolError("INVALID_EXPRESSION", "只支持数字、正负号、四则运算和括号。")


def calculator(expression: str, context: ToolContext) -> dict:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ToolError("INVALID_EXPRESSION", "表达式语法不正确。") from error
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ToolError("EXPRESSION_TOO_COMPLEX", "表达式过于复杂，请减少运算数量。")
    return {"result": _calculate_node(tree.body)}


MOCK_SEARCH_RESULTS = (
    {
        "title": "Python 入门",
        "summary": "Python 基础语法与函数示例。",
        "url": "https://example.com/python",
    },
    {
        "title": "SQLite 任务存储",
        "summary": "使用 SQLite 保存和查询任务。",
        "url": "https://example.com/sqlite",
    },
    {
        "title": "Agent 工具调用",
        "summary": "通过名称、描述和参数 Schema 注册工具。",
        "url": "https://example.com/agent-tools",
    },
)


def search(query: str, context: ToolContext) -> dict:
    keywords = query.casefold().split()
    results = []
    for item in MOCK_SEARCH_RESULTS:
        text = f"{item['title']} {item['summary']}".casefold()
        if any(keyword in text for keyword in keywords):
            results.append(dict(item))
    return {"results": results}


def _todo_record(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "title": row["title"], "completed": bool(row["completed"])}


def todo(
    action: str,
    context: ToolContext,
    id: int | None = None,
    title: str | None = None,
) -> dict:
    db_path = Path(context.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    scope = (context.user_id, context.session_id)
    invocation = None
    if action != "list" and (context.run_id is not None or context.tool_call_id is not None):
        if any(not isinstance(value, str) or not value.strip() for value in (context.run_id, context.tool_call_id)):
            raise ToolError("INVALID_CONTEXT", "待办去重必须同时提供非空 run_id 和 tool_call_id。")
        invocation = (*scope, context.run_id, context.tool_call_id)
    request_json = json.dumps({"action": action, "id": id, "title": title.strip() if title is not None else None},
                              ensure_ascii=False, sort_keys=True, allow_nan=False)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1))
                )
            """)
            if invocation is not None:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS todo_invocations (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                        run_id TEXT NOT NULL, tool_call_id TEXT NOT NULL,
                        request_json TEXT NOT NULL, result_json TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id, run_id, tool_call_id)
                    )
                """)
                previous = connection.execute(
                    "SELECT request_json, result_json FROM todo_invocations "
                    "WHERE user_id = ? AND session_id = ? AND run_id = ? AND tool_call_id = ?", invocation,
                ).fetchone()
                if previous is not None:
                    if previous["request_json"] != request_json:
                        raise ToolError("IDEMPOTENCY_CONFLICT", "同一次待办调用的参数发生变化，拒绝重复写入。")
                    if context.on_replay is not None:
                        context.on_replay()
                    return json.loads(previous["result_json"])
            if action == "list":
                rows = connection.execute(
                    "SELECT id, title, completed FROM todos "
                    "WHERE user_id = ? AND session_id = ? ORDER BY id",
                    scope,
                ).fetchall()
                return {"todos": [_todo_record(row) for row in rows]}
            if action == "add":
                cursor = connection.execute(
                    "INSERT INTO todos (user_id, session_id, title) VALUES (?, ?, ?)",
                    (*scope, title.strip()),
                )
                id = cursor.lastrowid
            elif action == "update":
                cursor = connection.execute(
                    "UPDATE todos SET title = ? WHERE id = ? AND user_id = ? AND session_id = ?",
                    (title.strip(), id, *scope),
                )
            elif action == "complete":
                cursor = connection.execute(
                    "UPDATE todos SET completed = 1 WHERE id = ? AND user_id = ? AND session_id = ?",
                    (id, *scope),
                )
            else:
                raise ToolError("INVALID_ARGUMENTS", "不支持的 todo 操作。")
            if action != "add" and cursor.rowcount == 0:
                raise ToolError("TODO_NOT_FOUND", "当前用户和 session 中不存在该任务。")
            row = connection.execute(
                "SELECT id, title, completed FROM todos "
                "WHERE id = ? AND user_id = ? AND session_id = ?",
                (id, *scope),
            ).fetchone()
            data = {"todo": _todo_record(row)}
            if invocation is not None:
                connection.execute(
                    "INSERT INTO todo_invocations VALUES (?, ?, ?, ?, ?, ?)",
                    (*invocation, request_json, json.dumps(data, ensure_ascii=False, allow_nan=False)),
                )
            return data
