import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from jsonschema import SchemaError

from tool_registry import TOOL_REGISTRY, call_tool, get_tool_definitions, register_tool
from tools import ToolContext, calculator, search, todo
from helpers import temporary_directory


class ToolTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory("agent-test-tools-")
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "todos.sqlite3"
        self.context = ToolContext("user-a", "session-a", self.db_path)

    def invoke(self, name, arguments, context=None):
        result = call_tool(name, arguments, self.context if context is None else context)
        self.assertEqual(set(result), {"ok", "data", "error"})
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        if result["ok"]:
            self.assertIsInstance(result["data"], dict)
            self.assertIsNone(result["error"])
        else:
            self.assertIsNone(result["data"])
            self.assertEqual(set(result["error"]), {"code", "message"})
        return result

    def assert_failure(self, name, arguments, code, context=None):
        result = self.invoke(name, arguments, context)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], code)
        return result

    def add_todo(self, title="学习工具", context=None):
        result = self.invoke("todo", {"action": "add", "title": title}, context)
        self.assertTrue(result["ok"])
        return result["data"]["todo"]

    def test_registered_tools_have_metadata_and_normal_functions(self):
        definitions = get_tool_definitions()
        self.assertEqual([item["name"] for item in definitions], ["calculator", "search", "todo"])
        for definition in definitions:
            self.assertEqual(set(definition), {"name", "description", "parameters"})
            self.assertTrue(definition["description"])
        json.dumps(definitions)
        for name, function in (("calculator", calculator), ("search", search), ("todo", todo)):
            self.assertIs(TOOL_REGISTRY[name]["execute"], function)
        definitions[0]["parameters"]["properties"].clear()
        self.assertIn("expression", TOOL_REGISTRY["calculator"]["parameters"]["properties"])

    def test_new_function_can_be_registered_and_called(self):
        received = []

        def echo(text, context):
            received.append(context)
            return {"text": text}

        with patch.dict(TOOL_REGISTRY):
            register_tool("echo", "测试函数注册", {
                "type": "object", "properties": {"text": {"type": "string"}},
                "required": ["text"], "additionalProperties": False,
            }, echo)
            result = self.invoke("echo", {"text": "你好"})
            self.assertEqual(result["data"], {"text": "你好"})
            self.assertEqual(received, [self.context])

    def test_duplicate_tool_name_is_rejected(self):
        with self.assertRaises(ValueError):
            register_tool("calculator", "重复工具", {"type": "object"}, Mock())
        self.assertIs(TOOL_REGISTRY["calculator"]["execute"], calculator)

    def test_invalid_schema_is_rejected_during_registration(self):
        with self.assertRaises(SchemaError):
            register_tool("invalid-schema", "错误 Schema", {"type": "unknown-type"}, Mock())
        self.assertNotIn("invalid-schema", TOOL_REGISTRY)

    def test_unknown_tool_is_reported(self):
        for name in ("missing", None, []):
            with self.subTest(name=name):
                self.assert_failure(name, {}, "TOOL_NOT_FOUND")

    def test_invalid_arguments_never_execute_the_function(self):
        execute = Mock()
        with patch.dict(TOOL_REGISTRY["calculator"], {"execute": execute}):
            for arguments in (None, [], "{}", {}, {"expression": 123}, {"expression": "1", "extra": 2}):
                with self.subTest(arguments=arguments):
                    self.assert_failure("calculator", arguments, "INVALID_ARGUMENTS")
        execute.assert_not_called()

    def test_invalid_context_is_reported_before_execution(self):
        for context in (None, {}, ToolContext("", "session", self.db_path), ToolContext("user", " ", self.db_path)):
            with self.subTest(context=context):
                result = call_tool("todo", {"action": "add", "title": "学习工具"}, context)
                self.assertEqual(result["error"]["code"], "INVALID_CONTEXT")
        self.assertFalse(self.db_path.exists())

    def test_unexpected_execution_error_has_consistent_structure(self):
        with patch.dict(TOOL_REGISTRY["calculator"], {"execute": Mock(side_effect=RuntimeError("internal"))}):
            result = self.assert_failure("calculator", {"expression": "1"}, "TOOL_EXECUTION_ERROR")
        self.assertIn("RuntimeError", result["error"]["message"])

    def test_calculator_supports_numbers_operators_and_parentheses(self):
        examples = [
            ("42", 42), ("2 + 3 * 4", 14), ("(2 + 3) * 4 / 2", 10.0),
            ("-(2 + 3) * +2", -10), ("2 * -3", -6), ("1.5 + .5 - 1", 1.0),
            (" 4 / 2 ", 2.0), ("0.1 + 0.2", 0.3),
        ]
        for expression, expected in examples:
            with self.subTest(expression=expression):
                result = self.invoke("calculator", {"expression": expression})
                self.assertTrue(result["ok"])
                self.assertAlmostEqual(result["data"]["result"], expected)

    def test_calculator_reports_division_by_zero(self):
        for expression in ("1 / 0", "1 / (2 - 2)", "1 / -0.0"):
            with self.subTest(expression=expression):
                self.assert_failure("calculator", {"expression": expression}, "DIVISION_BY_ZERO")

    def test_calculator_rejects_unsupported_syntax(self):
        expressions = ["1 +", "(1 + 2", "1; 2", "2 ** 3", "4 // 2", "5 % 2", "True", "None",
                       "1j", "[1, 2]", "1 < 2", "abs(-1)", "x + 1", "(1).__class__"]
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assert_failure("calculator", {"expression": expression}, "INVALID_EXPRESSION")

    def test_calculator_does_not_execute_injected_python(self):
        target = self.db_path.parent / "should-not-exist"
        expression = f"__import__('pathlib').Path({str(target)!r}).write_text('executed')"
        self.assertLessEqual(len(expression), 256)
        self.assert_failure("calculator", {"expression": expression}, "INVALID_EXPRESSION")
        self.assertFalse(target.exists())

    def test_calculator_bounds_expression_size_and_numeric_results(self):
        self.assert_failure("calculator", {"expression": "1" * 257}, "INVALID_ARGUMENTS")
        self.assert_failure("calculator", {"expression": "+".join(["1"] * 40)}, "EXPRESSION_TOO_COMPLEX")
        for expression in ("1e309", "1e101", "1e100 * 10"):
            with self.subTest(expression=expression):
                self.assert_failure("calculator", {"expression": expression}, "NUMERIC_RANGE")

    def test_search_matches_keywords_in_fixed_order(self):
        results = self.invoke("search", {"query": "pYtHoN SQLITE"})["data"]["results"]
        self.assertEqual([item["title"] for item in results], ["Python 入门", "SQLite 任务存储"])
        for item in results:
            self.assertEqual(set(item), {"title", "summary", "url"})
        summary_match = self.invoke("search", {"query": "基础语法"})["data"]["results"]
        self.assertEqual(summary_match[0]["url"], "https://example.com/python")

    def test_search_returns_empty_results_for_no_match(self):
        result = self.invoke("search", {"query": "没有匹配的关键词"})
        self.assertEqual(result["data"], {"results": []})

    def test_search_result_mutation_does_not_change_mock_data(self):
        first = self.invoke("search", {"query": "Python"})
        first["data"]["results"][0]["title"] = "已修改"
        second = self.invoke("search", {"query": "Python"})
        self.assertEqual(second["data"]["results"][0]["title"], "Python 入门")

    def test_search_rejects_invalid_parameters(self):
        for arguments in ({}, {"query": ""}, {"query": " \t\n"}, {"query": 123}, {"query": "x" * 201}):
            with self.subTest(arguments=arguments):
                self.assert_failure("search", arguments, "INVALID_ARGUMENTS")

    def test_todo_lifecycle_keeps_the_same_id(self):
        task = self.add_todo("  学习工具  ")
        task_id = task["id"]
        self.assertIs(type(task_id), int)
        self.assertEqual(task, {"id": task_id, "title": "学习工具", "completed": False})
        updated = self.invoke("todo", {"action": "update", "id": task_id, "title": "练习工具"})
        self.assertEqual(updated["data"]["todo"]["id"], task_id)
        self.assertEqual(updated["data"]["todo"]["title"], "练习工具")
        for _ in range(2):
            completed = self.invoke("todo", {"action": "complete", "id": task_id})
            self.assertEqual(completed["data"]["todo"], {"id": task_id, "title": "练习工具", "completed": True})
        listed = self.invoke("todo", {"action": "list"})
        self.assertEqual(listed["data"]["todos"], [completed["data"]["todo"]])

    def test_todo_new_scope_has_no_tasks(self):
        self.assertEqual(self.invoke("todo", {"action": "list"})["data"], {"todos": []})

    def test_todo_ids_are_unique_and_list_order_is_stable(self):
        first = self.add_todo("同名任务")
        second = self.add_todo("同名任务")
        self.assertNotEqual(first["id"], second["id"])
        self.assertGreater(first["id"], 0)
        self.assertEqual(self.invoke("todo", {"action": "list"})["data"]["todos"], [first, second])

    def test_todo_persists_when_context_and_connection_are_recreated(self):
        task = self.add_todo()
        restored = ToolContext("user-a", "session-a", Path(str(self.db_path)))
        self.assertEqual(self.invoke("todo", {"action": "list"}, restored)["data"]["todos"], [task])

    def test_todo_isolates_users_and_sessions_for_reads_and_writes(self):
        contexts = [ToolContext(user, session, self.db_path)
                    for user in ("user-a", "user-b") for session in ("session-a", "session-b")]
        tasks = [self.add_todo(f"任务 {index}", context) for index, context in enumerate(contexts)]
        for index, context in enumerate(contexts):
            self.assertEqual(self.invoke("todo", {"action": "list"}, context)["data"]["todos"], [tasks[index]])
            for other, task in enumerate(tasks):
                if other != index:
                    self.assert_failure("todo", {"action": "update", "id": task["id"], "title": "越权修改"},
                                        "TODO_NOT_FOUND", context)
                    self.assert_failure("todo", {"action": "complete", "id": task["id"]},
                                        "TODO_NOT_FOUND", context)
        for index, context in enumerate(contexts):
            self.assertEqual(self.invoke("todo", {"action": "list"}, context)["data"]["todos"], [tasks[index]])

    def test_todo_unknown_id_is_reported(self):
        self.assert_failure("todo", {"action": "update", "id": 999, "title": "不存在"}, "TODO_NOT_FOUND")
        self.assert_failure("todo", {"action": "complete", "id": 999}, "TODO_NOT_FOUND")

    def test_todo_schema_rejects_invalid_action_fields_before_opening_database(self):
        cases = [
            {}, {"action": "delete"}, {"action": "add"}, {"action": "add", "title": " "},
            {"action": "add", "title": 1}, {"action": "add", "title": "x" * 201},
            {"action": "add", "title": "任务", "id": 1}, {"action": "list", "id": 1},
            {"action": "list", "title": "任务"}, {"action": "update", "id": 1},
            {"action": "update", "title": "任务"}, {"action": "complete"},
            {"action": "complete", "id": 1, "title": "任务"},
        ]
        cases.extend({"action": "complete", "id": value} for value in (0, -1, True, "1", 1.5, 2**63))
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assert_failure("todo", arguments, "INVALID_ARGUMENTS")
        self.assertFalse(self.db_path.exists())

    def test_todo_rejects_identity_and_database_path_in_tool_arguments(self):
        for key in ("user_id", "session_id", "db_path", "context"):
            with self.subTest(key=key):
                self.assert_failure("todo", {"action": "add", "title": "任务", key: "other"}, "INVALID_ARGUMENTS")
        self.assertFalse(self.db_path.exists())

    def test_todo_uses_sql_parameters_for_titles_and_scope(self):
        title = "任务'); DROP TABLE todos; --"
        task = self.add_todo(title)
        injected_scope = ToolContext("' OR 1=1 --", "session-a", self.db_path)
        self.assertEqual(self.invoke("todo", {"action": "list"}, injected_scope)["data"]["todos"], [])
        self.assertEqual(self.invoke("todo", {"action": "list"})["data"]["todos"], [task])

    def test_todo_storage_failures_have_consistent_errors(self):
        directory_context = ToolContext("user", "session", self.db_path.parent)
        self.assert_failure("todo", {"action": "list"}, "STORAGE_ERROR", directory_context)
        self.db_path.write_bytes(b"not a sqlite database")
        self.assert_failure("todo", {"action": "list"}, "STORAGE_ERROR")


if __name__ == "__main__":
    unittest.main()
