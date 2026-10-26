import json
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import openai

from runtime import run_agent
from context_manager import ContextConfig, SUMMARY_PREFIX
from sessions import create_session, list_sessions, load_session, open_session, save_session
from helpers import completion, temporary_directory, tool_call
from tool_registry import call_tool
from tools import ToolContext


class SessionTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory("agent-test-sessions-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db_path = self.directory / "agent.sqlite3"
        self.context = ToolContext("user-a", "budget", self.db_path)
        self.requests = []

    def turn(self, context, text, responses, max_steps=6):
        session = load_session(context)
        client = Mock()
        sequence = iter(responses)

        def create(**request):
            self.requests.append(deepcopy(request))
            response = next(sequence)
            if isinstance(response, Exception):
                raise response
            return response

        client.chat.completions.create.side_effect = create
        result = run_agent(client, "deepseek-flash", text, context, history=session["messages"],
                           max_steps=max_steps, trace_dir=self.directory / "traces", context_state=session["context_state"])
        save_session(context, result, session["version"])
        return result

    def test_create_open_and_list_preserve_existing_session(self):
        created = create_session(self.context)
        self.assertEqual(created["messages"], [])
        self.assertIsNone(created["last_run"])
        self.assertEqual(created["version"], 0)
        self.turn(self.context, "你好", [completion("你好")])
        saved = load_session(self.context)
        self.assertEqual(open_session(self.context), saved)
        with self.assertRaisesRegex(ValueError, "已存在"):
            create_session(self.context)
        listed = list_sessions("user-a", self.db_path)
        self.assertEqual([item["session_id"] for item in listed], ["budget"])
        self.assertNotIn("messages", listed[0])
        self.assertEqual(listed[0]["last_run"], saved["last_run"])

    def test_full_protocol_messages_round_trip_and_resume(self):
        create_session(self.context)
        first = self.turn(self.context, "计算 2+3", [
            completion(calls=[tool_call("saved-call", "calculator", {"expression": "2+3"})], reasoning="工具前字段"),
            completion("结果 5", reasoning="最终回答字段"),
        ])
        restored = load_session(ToolContext("user-a", "budget", Path(str(self.db_path))))
        self.assertEqual(restored["messages"], first["messages"])
        self.assertIsNone(restored["messages"][2]["content"])
        self.assertEqual(restored["messages"][2]["tool_calls"][0]["id"], "saved-call")
        self.assertEqual(restored["messages"][3]["tool_call_id"], "saved-call")
        self.assertEqual(restored["messages"][-1]["reasoning_content"], "最终回答字段")
        self.requests.clear()
        second = self.turn(self.context, "继续", [completion("继续回答")], max_steps=1)
        self.assertEqual(self.requests[0]["messages"][:-1], restored["messages"])
        self.assertEqual(second["steps"], 1)
        self.assertNotEqual(first["run_id"], second["run_id"])

    def test_two_windows_alternate_and_other_user_cannot_read_or_modify(self):
        budget = self.context
        weekly = ToolContext("user-a", "weekly", self.db_path)
        other = ToolContext("user-b", "budget", self.db_path)
        for context in (budget, weekly, other):
            create_session(context)
        first = self.turn(budget, "创建预算复核待办", [
            completion(calls=[tool_call("budget-add", "todo", {"action": "add", "title": "预算复核"})]),
            completion("已创建预算复核"),
        ])
        budget_id = json.loads(first["messages"][-2]["content"])["data"]["todo"]["id"]
        self.turn(weekly, "创建周报待办", [
            completion(calls=[tool_call("weekly-add", "todo", {"action": "add", "title": "撰写周报"})]),
            completion("已创建周报"),
        ])
        for context, original, title in ((budget, "创建预算复核待办", "预算复核"),
                                         (weekly, "创建周报待办", "撰写周报")):
            self.requests.clear()
            result = self.turn(context, "查一下刚才的待办", [
                completion(calls=[tool_call("list-own", "todo", {"action": "list"})]),
                completion("已查询当前会话"),
            ])
            inputs = [item["content"] for item in self.requests[0]["messages"] if item["role"] == "user"]
            self.assertEqual(inputs, [original, "查一下刚才的待办"])
            tasks = json.loads(result["messages"][-2]["content"])["data"]["todos"]
            self.assertEqual([item["title"] for item in tasks], [title])
        self.requests.clear()
        rejected = self.turn(other, "查询并尝试修改其他用户的待办", [
            completion(calls=[
                tool_call("other-list", "todo", {"action": "list"}),
                tool_call("other-update", "todo", {"action": "update", "id": budget_id, "title": "越权修改"}),
            ]),
            completion("无法访问其他用户待办"),
        ])
        self.assertEqual(len(self.requests[0]["messages"]), 2)
        feedback = [json.loads(item["content"]) for item in rejected["messages"] if item["role"] == "tool"]
        self.assertEqual(feedback[0]["data"]["todos"], [])
        self.assertEqual(feedback[1]["error"]["code"], "TODO_NOT_FOUND")
        self.assertEqual([item["session_id"] for item in list_sessions("user-b", self.db_path)], ["budget"])
        self.assertEqual(call_tool("todo", {"action": "list"}, budget)["data"]["todos"][0]["title"], "预算复核")

    def test_fresh_python_process_restores_and_modifies_original_todo(self):
        create_session(self.context)
        result = self.turn(self.context, "新增预算复核", [
            completion(calls=[tool_call("before-restart", "todo", {"action": "add", "title": "预算复核"})]),
            completion("预算复核已保存"),
        ])
        task_id = json.loads(result["messages"][-2]["content"])["data"]["todo"]["id"]
        script = """
import json, sys
from pathlib import Path
from sessions import load_session
from tools import ToolContext
from tool_registry import call_tool
context = ToolContext('user-a', 'budget', Path(sys.argv[1]))
session = load_session(context)
listed = call_tool('todo', {'action': 'list'}, context)
updated = call_tool('todo', {'action': 'update', 'id': int(sys.argv[2]), 'title': '预算复核已修改'}, context)
print(json.dumps({'session': session, 'listed': listed, 'updated': updated}, ensure_ascii=True))
"""
        process = subprocess.run([sys.executable, "-c", script, str(self.db_path), str(task_id)],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
        restored = json.loads(process.stdout)
        self.assertEqual(restored["session"]["messages"], result["messages"])
        self.assertEqual(restored["listed"]["data"]["todos"][0]["id"], task_id)
        self.assertEqual(restored["updated"]["data"]["todo"]["title"], "预算复核已修改")
        self.assertEqual(call_tool("todo", {"action": "list"}, self.context)["data"]["todos"][0]["title"], "预算复核已修改")

    def test_max_steps_and_request_error_states_are_saved_and_can_continue(self):
        create_session(self.context)
        limited = self.turn(self.context, "计算", [
            completion(calls=[tool_call("limit-call", "calculator", {"expression": "1+2"})]),
        ], max_steps=1)
        self.assertEqual(load_session(self.context)["last_run"]["status"], "max_steps")
        self.turn(self.context, "继续", [completion("结果是 3")])
        failed = self.turn(self.context, "网络失败测试", [openai.APIConnectionError(request=Mock())])
        saved = load_session(self.context)
        self.assertEqual(saved["last_run"]["error"], failed["error"])
        self.assertEqual(saved["last_run"]["status"], "error")
        self.assertEqual(saved["version"], 3)
        self.assertIn(limited["messages"][-1], saved["messages"])

    def test_stale_version_cannot_overwrite_newer_history(self):
        initial = create_session(self.context)
        result = self.turn(self.context, "新消息", [completion("新回答")])
        with self.assertRaisesRegex(ValueError, "已被其他窗口更新"):
            save_session(self.context, {**result, "messages": []}, initial["version"])
        self.assertEqual(load_session(self.context)["messages"], result["messages"])

    def test_missing_or_other_user_session_cannot_be_loaded_or_saved(self):
        create_session(self.context)
        result = self.turn(self.context, "私有消息", [completion("私有回答")])
        for context in (ToolContext("user-b", "budget", self.db_path),
                        ToolContext("user-a", "missing", self.db_path)):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, "不存在"):
                    load_session(context)
                with self.assertRaises(ValueError):
                    save_session(context, result, expected_version=0)
        self.assertEqual(list_sessions("user-b", self.db_path), [])

    def test_scope_uses_bound_sql_parameters(self):
        create_session(self.context)
        injected = ToolContext("' OR 1=1 --", "budget", self.db_path)
        self.assertEqual(list_sessions(injected.user_id, self.db_path), [])
        with self.assertRaises(ValueError):
            load_session(injected)
        create_session(injected)
        self.assertEqual(load_session(injected)["user_id"], injected.user_id)
        self.assertEqual(len(list_sessions("user-a", self.db_path)), 1)

    def test_invalid_scope_is_rejected_before_database_creation(self):
        for user_id, session_id in (("", "s"), ("u", " "), (None, "s"), ("u", None)):
            with self.subTest(user_id=user_id, session_id=session_id), self.assertRaises(ValueError):
                open_session(ToolContext(user_id, session_id, self.db_path))
        self.assertFalse(self.db_path.exists())

    def test_old_database_is_migrated_without_losing_messages_or_identity(self):
        messages = [{"role": "system", "content": "旧规则"}, {"role": "user", "content": "旧消息"}]
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute("""CREATE TABLE sessions (
                    user_id TEXT, session_id TEXT, messages_json TEXT, last_run_json TEXT,
                    version INTEGER, created_at TEXT, updated_at TEXT, PRIMARY KEY(user_id, session_id))""")
                connection.execute("INSERT INTO sessions VALUES (?, ?, ?, NULL, 2, 'old', 'old')",
                                   (self.context.user_id, self.context.session_id, json.dumps(messages)))
        restored = load_session(self.context)
        self.assertEqual(restored["messages"], messages)
        self.assertIsNone(restored["context_state"])
        self.assertEqual(restored["version"], 2)
        self.assertEqual(load_session(self.context), restored)

    def test_summary_and_recent_messages_are_saved_atomically_and_restored(self):
        session = create_session(self.context)
        history = [{"role": "system", "content": "旧固定规则"}]
        for index in range(4):
            history.extend([{"role": "user", "content": "旧资料" * 2000 if index == 0 else f"近期 {index}"},
                            {"role": "assistant", "content": "已处理", "reasoning_content": "协议字段"}])
        client = Mock()
        client.chat.completions.create.side_effect = lambda **request: completion(
            "目标：预算复核；金额 1800；ID=7。" if request["tool_choice"] == "none" else "本轮回答")
        result = run_agent(client, "deepseek-flash", "继续", self.context, history=history,
                           context_config=ContextConfig(window_tokens=9000, output_tokens=512, summary_tokens=512),
                           trace_dir=self.directory / "traces")
        self.assertEqual(result["status"], "completed")
        save_session(self.context, result, session["version"])
        restored = load_session(self.context)
        self.assertEqual(restored["context_state"], result["context_state"])
        self.assertTrue(restored["context_state"]["summary"])
        self.assertEqual(restored["messages"], result["messages"])
        self.requests.clear()
        self.turn(self.context, "再次追问", [completion("恢复成功")])
        messages = self.requests[0]["messages"]
        self.assertEqual(sum(item.get("content", "").startswith(SUMMARY_PREFIX) for item in messages), 1)
        self.assertNotIn("旧资料" * 10, json.dumps(messages, ensure_ascii=False))

    def test_request_failure_after_committed_write_is_saved_and_followup_does_not_repeat_add(self):
        create_session(self.context)
        failed = self.turn(self.context, "创建预算待办", [
            completion(calls=[tool_call("committed-add", "todo", {"action": "add", "title": "已写入预算"})]),
            openai.APIConnectionError(request=Mock()),
        ])
        self.assertEqual(failed["status"], "error")
        saved = load_session(self.context)
        self.assertEqual(saved["last_run"]["status"], "error")
        self.assertEqual(saved["context_state"]["recent_messages"][-1]["role"], "tool")
        tasks = call_tool("todo", {"action": "list"}, self.context)["data"]["todos"]
        self.assertEqual(len(tasks), 1)
        recovered = self.turn(self.context, "查询刚才的待办", [
            completion(calls=[tool_call("query-after-error", "todo", {"action": "list"})]), completion("查到了。"),
        ])
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(call_tool("todo", {"action": "list"}, self.context)["data"]["todos"], tasks)


if __name__ == "__main__":
    unittest.main()
