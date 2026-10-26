import json
import os
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import openai
from dotenv import load_dotenv

from context_manager import ContextConfig, ContextManager, SUMMARY_PREFIX
from runtime import run_agent
from sessions import load_session, open_session, save_session
from tool_registry import call_tool
from tools import ToolContext
from helpers import assert_trace, temporary_directory


@unittest.skipUnless(os.getenv("RUN_LIVE_TESTS") == "1", "未启用真实模型测试；设置 RUN_LIVE_TESTS=1 后单独运行")
class LiveModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        cls.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        cls.model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
        if not cls.api_key or not cls.model:
            raise RuntimeError("真实模型测试已启用，但缺少 DeepSeek 密钥或模型配置。")

    def setUp(self):
        temporary = temporary_directory("agent-test-live-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db_path = self.directory / "agent.sqlite3"
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com", timeout=60, max_retries=0)
        self.addCleanup(self.client.close)

    def turn(self, session_id, text, user_id="live-user-a", max_steps=6, config=None, history=None):
        context = ToolContext(user_id, session_id, self.db_path)
        saved = open_session(context)
        result = run_agent(self.client, self.model, text, context,
                           history=saved["messages"] if history is None else history,
                           context_state=saved["context_state"] if history is None else None,
                           context_config=config, max_steps=max_steps, trace_dir=self.directory / "traces")
        save_session(context, result, saved["version"])
        assert_trace(self, result, context)
        self.assertNotIn(self.api_key, Path(result["trace_path"]).read_text(encoding="utf-8"))
        print(json.dumps({"live_session": session_id, "status": result["status"], "steps": result["steps"],
                          "compaction_calls": result["compaction_calls"], "error": result["error"],
                          "compact_failures": [item["message"] for item in result["trace"] if item["event"] == "compact_failed"],
                          "tools": [item["name"] for item in result["trace"] if item["event"] == "tool_result"]},
                         ensure_ascii=False), flush=True)
        return result, context

    def todos(self, context):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute(
                "SELECT id, title, completed FROM todos WHERE user_id = ? AND session_id = ? ORDER BY id",
                (context.user_id, context.session_id),
            ).fetchall()

    def results(self, result, name):
        return [event["result"] for event in result["trace"] if event["event"] == "tool_result" and event["name"] == name]

    def test_real_autonomous_tools_followups_isolation_and_process_restart(self):
        budget, budget_context = self.turn("budget", "计算 12000 的 15%，并创建预算复核待办，标题带上计算结果。")
        self.assertEqual(budget["status"], "completed", budget["error"])
        self.assertGreaterEqual(budget["steps"], 2)
        self.assertTrue(any(item["ok"] and item["data"]["result"] == 1800 for item in self.results(budget, "calculator")))
        self.assertTrue(any(item["ok"] and "todo" in item["data"] for item in self.results(budget, "todo")))
        original_budget = self.todos(budget_context)
        self.assertEqual(len(original_budget), 1)
        self.assertEqual(original_budget[0][2], 0)

        weekly, weekly_context = self.turn("weekly", "创建一条本周周报待办。")
        self.assertEqual(weekly["status"], "completed", weekly["error"])
        self.assertEqual(len(self.todos(weekly_context)), 1)
        explained, _ = self.turn("budget", "刚才的结果是什么意思？")
        self.assertEqual(explained["status"], "completed", explained["error"])
        self.assertIn("1800", explained["answer"])
        self.assertNotIn("本周周报", json.dumps(explained["context_state"], ensure_ascii=False))

        completed, _ = self.turn("weekly", "把刚才那条待办标为完成。")
        self.assertEqual(completed["status"], "completed", completed["error"])
        self.assertEqual(self.todos(weekly_context)[0][2], 1)
        self.assertTrue(any(item["ok"] and item["data"].get("todo", {}).get("completed") for item in self.results(completed, "todo")))
        isolated, other_context = self.turn("budget", "查询当前会话的待办列表。", user_id="live-user-b")
        self.assertEqual(isolated["status"], "completed", isolated["error"])
        self.assertEqual(self.todos(other_context), [])
        self.assertTrue(any(item["ok"] and item["data"].get("todos") == [] for item in self.results(isolated, "todo")))
        denied = call_tool("todo", {"action": "complete", "id": original_budget[0][0]}, other_context)
        self.assertEqual(denied["error"]["code"], "TODO_NOT_FOUND")
        self.assertEqual(self.todos(budget_context), original_budget)

        script = """
import sys
from pathlib import Path
import main
main.DB_PATH = Path(sys.argv[1])
main.TRACE_DIR = Path(sys.argv[2])
sys.argv = ['main.py', '--once', '--user-id', 'live-user-a', '--session-id', 'budget']
raise SystemExit(main.main())
"""
        process = subprocess.run([sys.executable, "-c", script, str(self.db_path), str(self.directory / "traces")],
                                 input="把刚才那条待办标为完成。\n", capture_output=True, encoding="utf-8", timeout=180,
                                 cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONUTF8": "1"})
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertNotIn(self.api_key, process.stdout + process.stderr)
        restored = load_session(budget_context)
        self.assertEqual(restored["version"], 3)
        self.assertEqual(self.todos(budget_context), [(original_budget[0][0], original_budget[0][1], 1)])
        self.assertIn("计算 12000", restored["context_state"]["recent_messages"][0]["content"])

        # 重放真实模型刚才产生的写入调用，直接核对原始工具结果及数据库。
        events = [json.loads(line) for line in Path(restored["last_run"]["trace_path"]).read_text(encoding="utf-8").splitlines()]
        writes = [item for item in events if item["event"] == "tool_start" and item["name"] == "todo"
                  and json.loads(item["arguments"]).get("action") == "complete"]
        self.assertTrue(writes)
        write = writes[-1]
        replayed = []
        execution = replace(budget_context, run_id=restored["last_run"]["run_id"], tool_call_id=write["tool_call_id"],
                            on_replay=lambda: replayed.append(True))
        replay = call_tool("todo", json.loads(write["arguments"]), execution)
        original = next(item["result"] for item in events if item["event"] == "tool_result" and item["tool_call_id"] == write["tool_call_id"])
        self.assertEqual(replay, original)
        self.assertEqual(replayed, [True])
        self.assertEqual(len(self.todos(budget_context)), 1)
        print("LIVE_DB_OK: budget=1 completed, weekly=1 completed, other_user=0; restart and replay verified", flush=True)

    def test_real_compaction_preserves_constraints_and_restores_single_summary(self):
        history = []
        fixtures = [
            ("预算金额 1800；币种必须为 CNY，禁止删除待办；标记 BUDGET-1800-XYZ；路径 data/agent.sqlite3。背景：" + "budget-note " * 500, "已记录。"),
            ("用户纠正：不能使用 USD，只允许查询或修改已有任务。补充资料：" + "review-note " * 500, "已确认纠正。"),
            ("进度一：没有修改数据，等待指示。背景：" + "progress " * 160, "已记录进度一。"),
            ("进度二：继续保留已有目标及约束。背景：" + "pending " * 160, "已记录进度二。"),
        ]
        for question, answer in fixtures:
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer, "reasoning_content": "预置验证历史字段"}]
        config = ContextConfig(window_tokens=11000, output_tokens=1024, summary_tokens=1024, compact_threshold=0.75)
        first, context = self.turn("compact", "复述历史中的币种、禁止事项、验证标记和路径。", config=config, history=history)
        self.assertEqual(first["status"], "completed", first["error"])
        self.assertTrue(any(item["event"] == "context_compacted" for item in first["trace"]))
        second, _ = self.turn("compact", "再次核对币种、禁止事项、验证标记和路径。", config=replace(config, compact_threshold=0.01))
        self.assertEqual(second["status"], "completed", second["error"])
        self.assertTrue(any(item["event"] == "context_compacted" for item in second["trace"]))
        restored = load_session(context)
        for exact in ("CNY", "BUDGET-1800-XYZ", "data/agent.sqlite3"):
            self.assertIn(exact, restored["context_state"]["summary"])
            self.assertIn(exact, second["answer"])
        self.assertIn("删除", restored["context_state"]["summary"])
        self.assertEqual(sum(item.get("content", "").startswith(SUMMARY_PREFIX) for item in second["messages"]), 1)
        for result in (first, second):
            for event in result["trace"]:
                if event["event"] in ("context_budget", "compact_request"):
                    self.assertLessEqual(event["estimated_tokens"], event["input_budget"])

    def test_real_model_tool_call_stops_at_step_limit_and_keeps_result(self):
        result, context = self.turn("limit", "请使用 calculator 计算 41+1，然后解释工具返回的结果。", max_steps=1)
        self.assertEqual(result["status"], "max_steps", result["error"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["steps"], 1)
        self.assertTrue(any(item["ok"] and item["data"]["result"] == 42 for item in self.results(result, "calculator")))
        restored = load_session(context)
        self.assertEqual(restored["last_run"]["status"], "max_steps")
        self.assertEqual(restored["context_state"]["recent_messages"][-1]["role"], "tool")

    def test_real_developer_rules_are_accepted_by_deepseek(self):
        manager = ContextManager(self.client, self.model,
                                 [{"role": "system", "content": "遵守固定规则，按要求简短作答。"},
                                  {"role": "developer", "content": "回答中必须包含精确标记 ACCEPTANCE_OK。"}],
                                 [], recent_messages=[{"role": "user", "content": "请确认。"}])
        request = manager.prepare_input()
        self.assertNotIn("developer", [item["role"] for item in request["messages"]])
        response = self.client.chat.completions.create(**request)
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertIn("ACCEPTANCE_OK", response.choices[0].message.content)


if __name__ == "__main__":
    unittest.main()
