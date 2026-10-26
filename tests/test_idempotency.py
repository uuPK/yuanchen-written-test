import json
import sqlite3
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

from tool_registry import call_tool
from tools import ToolContext
from helpers import temporary_directory


class TodoIdempotencyTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory("agent-test-dedupe-")
        self.addCleanup(temporary.cleanup)
        self.db_path = Path(temporary.name) / "agent.sqlite3"
        self.replayed = Mock()
        self.context = ToolContext("user-a", "budget", self.db_path, "run-a", "call-a", self.replayed)
        self.arguments = {"action": "add", "title": "预算复核"}

    def invoke(self, arguments=None, context=None):
        return call_tool("todo", self.arguments if arguments is None else arguments,
                         self.context if context is None else context)

    def rows(self, table):
        self.assertIn(table, ("todos", "todo_invocations"))
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute(f"SELECT * FROM {table}").fetchall()

    def test_replay_returns_original_result_with_same_stable_id(self):
        first = self.invoke()
        restored = replace(self.context, db_path=Path(str(self.db_path)))
        second = self.invoke(context=restored)
        self.assertTrue(first["ok"])
        self.assertEqual(second, first)
        self.assertEqual(len(self.rows("todos")), 1)
        self.assertEqual(len(self.rows("todo_invocations")), 1)
        self.replayed.assert_called_once_with()

    def test_same_title_with_new_call_or_new_run_is_a_new_write(self):
        results = [self.invoke(context=context) for context in (
            self.context, replace(self.context, tool_call_id="call-b"), replace(self.context, run_id="run-b"),
        )]
        self.assertEqual(len({item["data"]["todo"]["id"] for item in results}), 3)
        self.assertEqual(len(self.rows("todos")), 3)
        self.replayed.assert_not_called()

    def test_same_ids_for_other_user_or_session_do_not_share_cached_result(self):
        contexts = (self.context, replace(self.context, session_id="weekly"), replace(self.context, user_id="user-b"))
        results = [self.invoke(context=context) for context in contexts]
        self.assertEqual(len({item["data"]["todo"]["id"] for item in results}), 3)
        for context, expected in zip(contexts, results):
            self.assertEqual(self.invoke(context=context), expected)
            listed = self.invoke({"action": "list"}, context)["data"]["todos"]
            self.assertEqual(listed, [expected["data"]["todo"]])

    def test_successful_id_cannot_be_reused_with_different_parameters(self):
        first = self.invoke()
        conflict = self.invoke({"action": "add", "title": "改了参数"})
        self.assertEqual(conflict["error"]["code"], "IDEMPOTENCY_CONFLICT")
        self.assertEqual(self.invoke({"action": "list"})["data"]["todos"], [first["data"]["todo"]])
        self.assertEqual(len(self.rows("todo_invocations")), 1)

    def test_replayed_update_does_not_overwrite_a_later_edit_and_complete_is_cached(self):
        task_id = self.invoke()["data"]["todo"]["id"]
        update_context = replace(self.context, tool_call_id="update-1")
        arguments = {"action": "update", "id": task_id, "title": "第一版"}
        original = self.invoke(arguments, update_context)
        self.invoke({**arguments, "title": "第二版"}, replace(self.context, tool_call_id="update-2"))
        self.assertEqual(self.invoke(arguments, update_context), original)
        self.assertEqual(self.invoke({"action": "list"})["data"]["todos"][0]["title"], "第二版")
        complete_context = replace(self.context, tool_call_id="complete-1")
        completed = self.invoke({"action": "complete", "id": task_id}, complete_context)
        self.assertEqual(self.invoke({"action": "complete", "id": task_id}, complete_context), completed)
        self.assertTrue(self.invoke({"action": "list"})["data"]["todos"][0]["completed"])
        self.assertEqual(len(self.rows("todo_invocations")), 4)

    def test_failure_saving_dedupe_record_rolls_back_todo_write_and_allows_retry(self):
        self.invoke()
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute("CREATE TRIGGER fail_cache BEFORE INSERT ON todo_invocations BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        retry_context = replace(self.context, tool_call_id="retry")
        failed = self.invoke(context=retry_context)
        self.assertEqual(failed["error"]["code"], "STORAGE_ERROR")
        self.assertEqual(len(self.rows("todos")), 1)
        self.assertEqual(len(self.rows("todo_invocations")), 1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute("DROP TRIGGER fail_cache")
        self.assertTrue(self.invoke(context=retry_context)["ok"])
        self.assertEqual(len(self.rows("todos")), 2)

    def test_failed_action_does_not_reserve_the_invocation_id(self):
        failed = self.invoke({"action": "complete", "id": 999})
        self.assertEqual(failed["error"]["code"], "TODO_NOT_FOUND")
        self.assertTrue(self.invoke()["ok"])
        self.assertEqual(len(self.rows("todos")), 1)

    def test_partial_execution_ids_are_rejected(self):
        for context in (replace(self.context, run_id=None), replace(self.context, tool_call_id=" ")):
            with self.subTest(context=context):
                self.assertEqual(self.invoke(context=context)["error"]["code"], "INVALID_CONTEXT")

    def test_concurrent_replay_commits_only_one_todo(self):
        barrier = Barrier(2)

        def invoke():
            barrier.wait(timeout=5)
            return self.invoke()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(invoke) for _ in range(2)]
            results = [future.result(timeout=10) for future in futures]
        self.assertTrue(results[0]["ok"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(self.rows("todos")), 1)
        self.replayed.assert_called_once_with()

    def test_fresh_process_replays_persisted_result_without_another_insert(self):
        first = self.invoke()
        script = """
import json, sys
from pathlib import Path
from tool_registry import call_tool
from tools import ToolContext
context = ToolContext('user-a', 'budget', Path(sys.argv[1]), 'run-a', 'call-a')
print(json.dumps(call_tool('todo', {'action':'add', 'title':'预算复核'}, context), ensure_ascii=True))
"""
        process = subprocess.run([sys.executable, "-c", script, str(self.db_path)],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(process.stdout), first)
        self.assertEqual(len(self.rows("todos")), 1)


if __name__ == "__main__":
    unittest.main()
