import json
import sqlite3
from collections.abc import Callable
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError

from tools import ToolContext, ToolError, calculator, search, todo


TOOL_REGISTRY = {}


def register_tool(name: str, description: str, parameters: dict, execute: Callable) -> None:
    if name in TOOL_REGISTRY:
        raise ValueError(f"工具名称已注册：{name}")
    Draft202012Validator.check_schema(parameters)
    TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": deepcopy(parameters),
        "execute": execute,
    }


def get_tool_definitions() -> list[dict]:
    return [
        {key: deepcopy(tool[key]) for key in ("name", "description", "parameters")}
        for tool in TOOL_REGISTRY.values()
    ]


def _failure(code: str, message: str) -> dict:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def call_tool(name: str, arguments: dict, context: ToolContext) -> dict:
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return _failure("TOOL_NOT_FOUND", "工具未注册。")
    tool = TOOL_REGISTRY[name]
    try:
        Draft202012Validator(tool["parameters"]).validate(arguments)
        if not isinstance(context, ToolContext) or any(
            not isinstance(value, str) or not value.strip()
            for value in (context.user_id, context.session_id)
        ):
            raise ToolError("INVALID_CONTEXT", "执行上下文必须提供非空用户 ID 和 session ID。")
        data = tool["execute"](**arguments, context=context)
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        return {"ok": True, "data": data, "error": None}
    except ValidationError as error:
        field = ".".join(str(part) for part in error.absolute_path) or "arguments"
        return _failure("INVALID_ARGUMENTS", f"{field}: {error.message}")
    except ToolError as error:
        return _failure(error.code, str(error))
    except (sqlite3.Error, OSError):
        return _failure("STORAGE_ERROR", "无法读写任务数据库，请检查路径、权限或数据库状态。")
    except Exception as error:
        return _failure("TOOL_EXECUTION_ERROR", f"工具执行失败（{type(error).__name__}）。")


register_tool(
    "calculator",
    "计算包含数字、正负号、加减乘除和括号的表达式。",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "minLength": 1, "maxLength": 256, "pattern": r"\S"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    calculator,
)

register_tool(
    "search",
    "在固定模拟资料的标题和摘要中搜索；空格分词、忽略大小写，任意关键词匹配，按固定顺序返回。",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": r"\S"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    search,
)

register_tool(
    "todo",
    "管理当前用户和 session 的任务：add 新增、list 查询、update 修改标题、complete 标记完成。",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "update", "complete"]},
            "id": {"type": "integer", "minimum": 1, "maximum": 9223372036854775807},
            "title": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": r"\S"},
        },
        "required": ["action"],
        "additionalProperties": False,
        "oneOf": [
            {
                "properties": {"action": {"const": "add"}},
                "required": ["title"],
                "not": {"required": ["id"]},
            },
            {
                "properties": {"action": {"const": "list"}},
                "not": {"anyOf": [{"required": ["id"]}, {"required": ["title"]}]},
            },
            {"properties": {"action": {"const": "update"}}, "required": ["id", "title"]},
            {
                "properties": {"action": {"const": "complete"}},
                "required": ["id"],
                "not": {"required": ["title"]},
            },
        ],
    },
    todo,
)
