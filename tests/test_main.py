import io
import os
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import openai
from openai.types.chat import ChatCompletionMessage

import main
from sessions import list_sessions, load_session, open_session
from tools import ToolContext
from helpers import temporary_directory


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        temporary = temporary_directory("agent-test-cli-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        stack.enter_context(patch("main.TRACE_DIR", self.directory / "traces"))
        stack.enter_context(patch("main.DB_PATH", self.directory / "agent.sqlite3"))
        stack.enter_context(patch("sys.argv", ["main.py", "--once"]))
        stack.enter_context(patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "deepseek-flash",
        }, clear=True))
        self.load_env = stack.enter_context(patch("main.load_dotenv"))
        self.read_input = stack.enter_context(patch("builtins.input", return_value="  你好  "))
        self.stdout = stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
        self.stderr = stack.enter_context(patch("sys.stderr", new_callable=io.StringIO))
        self.factory = stack.enter_context(patch("main.openai.OpenAI"))
        self.client = self.factory.return_value.__enter__.return_value
        self.choice = SimpleNamespace(
            finish_reason="stop",
            message=ChatCompletionMessage(role="assistant", content="  你好，我是助手。  ", reasoning_content="公开返回字段"),
        )
        self.client.chat.completions.create.return_value = SimpleNamespace(choices=[self.choice])

    def test_success_sends_one_request_and_prints_answer(self):
        self.assertEqual(main.main(), 0)
        self.assertEqual(self.stdout.getvalue(), "助手：你好，我是助手。\n")
        self.assertIn("Trace：", self.stderr.getvalue())
        self.client.chat.completions.create.assert_called_once()
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["messages"][1], {"role": "user", "content": "你好"})
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual([tool["function"]["name"] for tool in request["tools"]], ["calculator", "search", "todo"])
        self.assertEqual(self.factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(self.load_env.call_args.args[0], main.Path(main.__file__).with_name(".env"))
        self.factory.return_value.__exit__.assert_called_once()

    def test_missing_key_stops_before_input_and_request(self):
        del os.environ["DEEPSEEK_API_KEY"]
        self.assertEqual(main.main(), 1)
        self.assertIn("缺少 DEEPSEEK_API_KEY", self.stderr.getvalue())
        self.read_input.assert_not_called()
        self.factory.assert_not_called()

    def test_blank_model_stops_before_request(self):
        os.environ["DEEPSEEK_MODEL"] = " "
        self.assertEqual(main.main(), 1)
        self.assertIn("DEEPSEEK_MODEL 不能为空", self.stderr.getvalue())
        self.factory.assert_not_called()

    def test_blank_input_stops_before_request(self):
        self.read_input.return_value = "  "
        self.assertEqual(main.main(), 1)
        self.assertIn("输入不能为空", self.stderr.getvalue())
        self.factory.assert_not_called()

    def test_cancelled_input_stops_before_request(self):
        for error in (EOFError(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.read_input.side_effect = error
                self.assertEqual(main.main(), 130)
                self.factory.assert_not_called()

    def test_request_errors_are_reported_without_provider_details(self):
        failures = [
            (openai.APITimeoutError(request=Mock()), "请求超时"),
            (openai.APIConnectionError(request=Mock()), "无法连接 DeepSeek"),
            (openai.APIError("private provider detail", request=Mock(), body=None), "兼容性"),
        ]
        for status in (400, 401, 402, 429, 500):
            response = Mock(status_code=status, headers={}, request=Mock())
            error = openai.APIStatusError("private provider detail", response=response, body=None)
            failures.append((error, f"HTTP {status}"))
        for error, expected in failures:
            with self.subTest(error=expected):
                self.client.chat.completions.create.side_effect = error
                self.assertEqual(main.main(), 1)
                self.assertIn(expected, self.stderr.getvalue())
                self.assertNotIn("private provider detail", self.stderr.getvalue())
                self.assertEqual(self.stdout.getvalue(), "")

    def test_empty_choices_are_reported(self):
        self.client.chat.completions.create.return_value = SimpleNamespace(choices=[])
        self.assertEqual(main.main(), 1)
        self.assertIn("没有返回候选回答", self.stderr.getvalue())

    def test_empty_answer_is_reported(self):
        for answer in (None, "  "):
            with self.subTest(answer=answer):
                self.choice.message.content = answer
                self.assertEqual(main.main(), 1)
                self.assertIn("没有返回可显示的回答", self.stderr.getvalue())

    def test_incomplete_answer_is_not_reported_as_success(self):
        self.choice.finish_reason = "length"
        self.assertEqual(main.main(), 1)
        self.assertIn("结束原因：length", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")

    def test_environment_file_read_failure_is_reported(self):
        self.load_env.side_effect = OSError("无法读取配置文件")
        self.assertEqual(main.main(), 1)
        self.assertIn("无法读取配置文件", self.stderr.getvalue())
        self.factory.assert_not_called()

    def test_step_limit_returns_distinct_exit_code(self):
        self.choice.finish_reason = "tool_calls"
        self.choice.message = ChatCompletionMessage.model_validate({
            "role": "assistant", "content": None, "reasoning_content": "先计算",
            "tool_calls": [{"id": "cli-call", "type": "function", "function": {
                "name": "calculator", "arguments": '{"expression":"1+2"}',
            }}],
        })
        with patch("sys.argv", ["main.py", "--once", "--max-steps", "1"]):
            self.assertEqual(main.main(), 2)
        self.assertIn("已达到 1 次模型请求上限", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")
        self.client.chat.completions.create.assert_called_once()
        session = load_session(ToolContext("demo-user", "demo-session", main.DB_PATH))
        self.assertEqual(session["last_run"]["status"], "max_steps")
        self.assertEqual(session["messages"][-1]["role"], "tool")

    def test_invalid_step_limit_stops_before_request(self):
        with patch("sys.argv", ["main.py", "--max-steps", "0"]):
            self.assertEqual(main.main(), 1)
        self.assertIn("max_steps 必须是正整数", self.stderr.getvalue())
        self.factory.assert_not_called()

    def test_unsupported_tool_call_is_reported_without_crashing_trace_output(self):
        self.choice.finish_reason = "tool_calls"
        self.choice.message = ChatCompletionMessage.model_validate({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "unsupported", "type": "custom", "custom": {"name": "unsupported", "input": "test"}}],
        })
        self.assertEqual(main.main(), 1)
        self.assertIn("工具调用类型不支持", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")

    def test_restarting_cli_restores_history_reasoning_and_last_run(self):
        self.assertEqual(main.main(), 0)
        context = ToolContext("demo-user", "demo-session", main.DB_PATH)
        first = load_session(context)
        self.read_input.return_value = "继续刚才的话题"
        self.assertEqual(main.main(), 0)
        second = load_session(context)
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][:3], first["messages"])
        self.assertEqual(request["messages"][2]["reasoning_content"], "公开返回字段")
        self.assertEqual(request["messages"][3], {"role": "user", "content": "继续刚才的话题"})
        self.assertEqual(second["version"], 2)
        self.assertNotEqual(first["last_run"]["run_id"], second["last_run"]["run_id"])

    def test_interactive_switching_uses_only_selected_history(self):
        self.read_input.side_effect = [
            "预算话题", "/new weekly", "周报话题", "/use budget", "继续预算", "/exit",
        ]
        requests = []

        def respond(**request):
            requests.append(deepcopy(request))
            return SimpleNamespace(choices=[self.choice])

        self.client.chat.completions.create.side_effect = respond
        with patch("sys.argv", ["main.py", "--session-id", "budget"]):
            self.assertEqual(main.main(), 0)
        user_inputs = [[item["content"] for item in request["messages"] if item["role"] == "user"]
                       for request in requests]
        self.assertEqual(user_inputs, [["预算话题"], ["周报话题"], ["预算话题", "继续预算"]])
        self.assertEqual(load_session(ToolContext("demo-user", "budget", main.DB_PATH))["version"], 2)
        self.assertEqual(load_session(ToolContext("demo-user", "weekly", main.DB_PATH))["version"], 1)

    def test_list_sessions_needs_no_api_key_and_only_lists_current_user(self):
        open_session(ToolContext("demo-user", "visible", main.DB_PATH))
        open_session(ToolContext("other-user", "hidden", main.DB_PATH))
        del os.environ["DEEPSEEK_API_KEY"]
        with patch("sys.argv", ["main.py", "--list-sessions"]):
            self.assertEqual(main.main(), 0)
        self.assertIn("visible", self.stdout.getvalue())
        self.assertNotIn("hidden", self.stdout.getvalue())
        self.read_input.assert_not_called()
        self.factory.assert_not_called()

    def test_new_session_gets_generated_id_and_empty_history(self):
        open_session(ToolContext("demo-user", "existing", main.DB_PATH))
        with patch("sys.argv", ["main.py", "--once", "--new-session"]):
            self.assertEqual(main.main(), 0)
        sessions = list_sessions("demo-user", main.DB_PATH)
        created = next(item for item in sessions if item["session_id"] != "existing")
        self.assertEqual(len(created["session_id"]), 32)
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual([item["role"] for item in request["messages"]], ["system", "user"])

    def test_bad_selection_keeps_current_session_and_commands_do_not_reach_model(self):
        self.read_input.side_effect = ["/use missing", "/new demo-session", "/unknown", "正常提问", "/exit"]
        with patch("sys.argv", ["main.py"]):
            self.assertEqual(main.main(), 0)
        self.client.chat.completions.create.assert_called_once()
        self.assertIn("不存在该 session", self.stderr.getvalue())
        self.assertEqual(load_session(ToolContext("demo-user", "demo-session", main.DB_PATH))["version"], 1)

    def test_database_failure_is_reported_without_request(self):
        with patch("main.DB_PATH", self.directory):
            self.assertEqual(main.main(), 1)
        self.assertIn("无法读写 session 数据库", self.stderr.getvalue())
        self.factory.assert_not_called()

    def test_interactive_eof_exits_without_an_extra_model_request(self):
        self.read_input.side_effect = ["你好", EOFError()]
        with patch("sys.argv", ["main.py"]):
            self.assertEqual(main.main(), 0)
        self.client.chat.completions.create.assert_called_once()

    def test_invalid_once_command_returns_failure_without_model_request(self):
        for command in ("/unknown", "/use", "/use missing", "/new demo-session"):
            with self.subTest(command=command):
                self.read_input.return_value = command
                self.assertEqual(main.main(), 1)
        self.factory.assert_not_called()

    def test_valid_once_command_returns_success_without_model_request(self):
        self.read_input.return_value = "/sessions"
        self.assertEqual(main.main(), 0)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
