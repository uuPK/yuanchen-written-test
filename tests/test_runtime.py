import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import openai

from runtime import SYSTEM_PROMPT, run_agent
from context_manager import ContextConfig, ContextManager, SUMMARY_PREFIX, estimate_tokens
from tool_registry import TOOL_REGISTRY, call_tool, get_tool_definitions
from tools import ToolContext
from helpers import assert_trace, completion, temporary_directory, tool_call


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory("agent-test-runtime-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.context = ToolContext("runtime-user", "runtime-session", self.directory / "agent.sqlite3")
        self.client = Mock()
        self.requests = []

    def scripted_model(self, responses):
        sequence = iter(responses)

        def create(**request):
            self.requests.append(deepcopy(request))
            response = next(sequence)
            if isinstance(response, Exception):
                raise response
            return response

        self.client.chat.completions.create.side_effect = create

    def run_turn(self, text="用户请求", **options):
        return run_agent(self.client, "deepseek-flash", text, self.context,
                         trace_dir=self.directory / "traces", **options)

    def test_direct_reply_is_decided_by_model_even_when_prompt_mentions_tools(self):
        self.scripted_model([completion("请先提供要处理的数据。")])
        with patch("runtime.call_tool") as execute:
            result = self.run_turn("计算、搜索、创建待办这些功能你都能做吗？")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["answer"], "请先提供要处理的数据。")
        execute.assert_not_called()
        self.assertEqual(self.requests[0]["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(self.requests[0]["tool_choice"], "auto")
        self.assertEqual([item["function"]["name"] for item in self.requests[0]["tools"]], ["calculator", "search", "todo"])

    def test_single_tool_call_keeps_id_message_and_reasoning(self):
        self.scripted_model([
            completion("我先计算。", [tool_call("calc-1", "calculator", {"expression": "2+3"})], "第一步推理"),
            completion("结果是 5。", reasoning="第二步推理"),
        ])
        result = self.run_turn()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], 2)
        second_input = self.requests[1]["messages"]
        self.assertEqual([item["role"] for item in second_input], ["system", "user", "assistant", "tool"])
        self.assertEqual(second_input[2]["reasoning_content"], "第一步推理")
        self.assertEqual(second_input[2]["content"], "我先计算。")
        self.assertEqual(second_input[2]["tool_calls"][0]["id"], "calc-1")
        self.assertEqual(second_input[3]["tool_call_id"], "calc-1")
        self.assertEqual(json.loads(second_input[3]["content"])["data"], {"result": 5})
        self.assertEqual(result["messages"][-1]["reasoning_content"], "第二步推理")

    def test_dependent_multistep_calculation_and_todo(self):
        self.scripted_model([
            completion(calls=[tool_call("budget-calc", "calculator", {"expression": "12000*15/100"})]),
            completion(calls=[tool_call("budget-todo", "todo", {"action": "add", "title": "预算复核：1800"})]),
            completion("12000 的 15% 是 1800，预算复核待办已创建。"),
        ])
        result = self.run_turn("计算 12000 的 15%，并创建预算复核待办")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["status"], "completed")
        tools = [item for item in result["messages"] if item["role"] == "tool"]
        self.assertEqual([item["tool_call_id"] for item in tools], ["budget-calc", "budget-todo"])
        self.assertEqual(json.loads(tools[0]["content"])["data"]["result"], 1800)
        tasks = call_tool("todo", {"action": "list"}, self.context)["data"]["todos"]
        self.assertEqual(tasks[0]["title"], "预算复核：1800")
        self.assertEqual(len(self.requests[1]["messages"]), 4)
        self.assertEqual(len(self.requests[2]["messages"]), 6)

    def test_multiple_tool_calls_in_one_response_execute_sequentially(self):
        self.scripted_model([
            completion(calls=[
                tool_call("add-first", "todo", {"action": "add", "title": "顺序测试"}),
                tool_call("list-second", "todo", {"action": "list"}),
            ]),
            completion("已完成。"),
        ])
        result = self.run_turn()
        results = [item for item in result["trace"] if item["event"] == "tool_result"]
        self.assertEqual([item["tool_call_id"] for item in results], ["add-first", "list-second"])
        self.assertEqual(len(results[1]["result"]["data"]["todos"]), 1)
        self.assertEqual([item["role"] for item in self.requests[1]["messages"]],
                         ["system", "user", "assistant", "tool", "tool"])
        starts_and_ends = [item["event"] for item in result["trace"] if item["event"].startswith("tool_")]
        self.assertEqual(starts_and_ends, ["tool_start", "tool_result", "tool_start", "tool_result"])

    def test_tool_error_is_returned_to_model_and_model_can_correct_arguments(self):
        self.scripted_model([
            completion(calls=[tool_call("bad", "calculator", {"expression": "1/0"})]),
            completion(calls=[tool_call("fixed", "calculator", {"expression": "1/2"})]),
            completion("修正后的结果是 0.5。"),
        ])
        result = self.run_turn()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(json.loads(self.requests[1]["messages"][-1]["content"])["error"]["code"], "DIVISION_BY_ZERO")
        self.assertEqual(json.loads(self.requests[2]["messages"][-1]["content"])["data"]["result"], 0.5)

    def test_malformed_json_wrong_types_and_unknown_tools_are_fed_back(self):
        cases = [
            (tool_call("invalid-json", "calculator", "{"), "INVALID_ARGUMENTS"),
            (tool_call("invalid-object", "calculator", "[]"), "INVALID_ARGUMENTS"),
            (tool_call("invalid-field", "calculator", {"expression": 123}), "INVALID_ARGUMENTS"),
            (tool_call("unknown", "missing", {}), "TOOL_NOT_FOUND"),
        ]
        for call, code in cases:
            with self.subTest(code=code, call_id=call["id"]):
                self.requests.clear()
                self.scripted_model([completion(calls=[call]), completion("收到工具错误。")])
                result = self.run_turn()
                self.assertEqual(result["status"], "completed")
                feedback = self.requests[1]["messages"][-1]
                self.assertEqual(feedback["tool_call_id"], call["id"])
                self.assertEqual(json.loads(feedback["content"])["error"]["code"], code)

    def test_last_allowed_step_executes_all_tools_and_stops_without_extra_request(self):
        self.scripted_model([completion(calls=[
            tool_call("last-add", "todo", {"action": "add", "title": "上限前执行"}),
            tool_call("last-list", "todo", {"action": "list"}),
        ])])
        result = self.run_turn(max_steps=1)
        self.assertEqual(result["status"], "max_steps")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["error"]["code"], "MAX_STEPS")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual([item["role"] for item in result["messages"]], ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(len(call_tool("todo", {"action": "list"}, self.context)["data"]["todos"]), 1)
        self.assertEqual(result["trace"][-1]["status"], "max_steps")

    def test_final_answer_on_last_step_counts_as_completed(self):
        self.scripted_model([completion("直接回答。")])
        self.assertEqual(self.run_turn(max_steps=1)["status"], "completed")

    def test_step_budget_resets_and_history_reasoning_is_preserved_without_mutation(self):
        self.scripted_model([completion("上一轮回答。", reasoning="上一轮推理")])
        first = self.run_turn(max_steps=1)
        history = deepcopy(first["messages"])
        self.requests.clear()
        self.scripted_model([completion("本轮回答。")])
        second = self.run_turn("继续追问", history=history, max_steps=1)
        self.assertEqual(second["steps"], 1)
        self.assertEqual(history, first["messages"])
        self.assertEqual([item["role"] for item in self.requests[0]["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(self.requests[0]["messages"][2]["reasoning_content"], "上一轮推理")

    def test_trace_is_valid_jsonl_and_correlates_step_call_and_result(self):
        self.scripted_model([
            completion(calls=[tool_call("trace-call", "calculator", {"expression": "3*4"})]),
            completion("结果是 12。"),
        ])
        result = self.run_turn()
        saved = [json.loads(line) for line in Path(result["trace_path"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(saved, result["trace"])
        self.assertTrue(all(item["run_id"] == result["run_id"] for item in saved))
        self.assertTrue(all(item["user_id"] == self.context.user_id and item["session_id"] == self.context.session_id for item in saved))
        calls = [item for item in saved if item["event"] in ("tool_start", "tool_result")]
        self.assertEqual([(item["step"], item["tool_call_id"]) for item in calls], [(1, "trace-call"), (1, "trace-call")])
        self.assertEqual(calls[1]["result"]["data"]["result"], 12)

    def test_truncated_or_inconsistent_responses_never_execute_tools(self):
        responses = [
            completion(calls=[tool_call("partial", "todo", {"action": "add", "title": "不得执行"})], finish_reason="length"),
            completion(calls=[tool_call("wrong-stop", "todo", {"action": "add", "title": "不得执行"})], finish_reason="stop"),
            completion("无调用", finish_reason="tool_calls"),
            completion(calls=[{"id": "unsupported", "type": "custom", "custom": {"name": "unsupported", "input": "test"}}]),
            completion("  "),
            SimpleNamespace(choices=[]),
        ]
        for response in responses:
            with self.subTest(response=response):
                self.scripted_model([response])
                with patch("runtime.call_tool") as execute:
                    result = self.run_turn()
                self.assertEqual(result["status"], "error")
                expected = "MODEL_TRUNCATED" if getattr(response, "choices", []) and response.choices[0].finish_reason == "length" else "MODEL_PROTOCOL_ERROR"
                self.assertEqual(result["error"]["code"], expected)
                execute.assert_not_called()

    def test_missing_or_duplicate_call_ids_never_execute_partial_batch(self):
        for ids in (("",), ("same", "same")):
            with self.subTest(ids=ids):
                self.scripted_model([completion(calls=[
                    tool_call(call_id, "todo", {"action": "add", "title": "不得执行"}) for call_id in ids
                ])])
                with patch("runtime.call_tool") as execute:
                    result = self.run_turn()
                self.assertEqual(result["status"], "error")
                execute.assert_not_called()

    def test_model_request_error_is_traced_without_raw_provider_details(self):
        errors = [
            openai.APIConnectionError(request=Mock()),
            openai.APITimeoutError(request=Mock()),
            openai.APIStatusError("private-provider-detail", response=Mock(status_code=401, headers={}, request=Mock()), body=None),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.scripted_model([error])
                result = self.run_turn()
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["steps"], 1)
                self.assertEqual(result["trace"][-1]["status"], "error")
                self.assertNotIn("private-provider-detail", Path(result["trace_path"]).read_text(encoding="utf-8"))

    def test_invalid_step_budgets_are_rejected_without_model_requests(self):
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.run_turn(max_steps=budget)
        self.client.chat.completions.create.assert_not_called()

    def test_context_is_checked_and_can_compact_inside_tool_loop(self):
        config = ContextConfig(window_tokens=10000, output_tokens=512, summary_tokens=512)
        text = "先计算，再解释"
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for index in range(3):
            history.extend([{"role": "user", "content": f"历史 {index}"},
                            {"role": "assistant", "content": "历史答复", "reasoning_content": "保留字段"}])
        definitions = [{"type": "function", "function": item} for item in get_tool_definitions()]
        probe = ContextManager(self.client, "deepseek-flash", history[:1], definitions,
                               recent_messages=history[1:] + [{"role": "user", "content": text}], config=config)
        base_count = estimate_tokens(probe.prepare_input())
        history[1]["content"] += "x" * (int(config.input_budget * config.compact_threshold) - base_count - 100)
        responses = iter([completion(calls=[tool_call("during-loop", "calculator", {"expression": "1+2"})]),
                          completion("结果是 3")])

        def respond(**request):
            self.requests.append(deepcopy(request))
            if request["tool_choice"] == "none":
                return completion("旧历史已处理；关键约束保持不变。")
            return next(responses)

        self.client.chat.completions.create.side_effect = respond
        result = self.run_turn(text, history=history, context_config=config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item["tool_choice"] for item in self.requests], ["auto", "none", "auto"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["compaction_calls"], 1)
        second = self.requests[-1]["messages"]
        self.assertEqual(second[-2]["tool_calls"][0]["id"], "during-loop")
        self.assertEqual(second[-1]["tool_call_id"], "during-loop")
        for request in self.requests:
            self.assertLessEqual(estimate_tokens(request) + request["max_tokens"], config.window_tokens)
        self.assertEqual(sum(event["event"] == "context_budget" for event in result["trace"]), 2)

    def test_over_budget_input_is_not_sent_and_error_preserves_original_messages(self):
        config = ContextConfig(window_tokens=8000, output_tokens=512, summary_tokens=512)
        result = self.run_turn("不可压缩的当前输入" * 1000, context_config=config)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "CONTEXT_LIMIT")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["context_state"]["recent_messages"][0]["content"], "不可压缩的当前输入" * 1000)
        self.client.chat.completions.create.assert_not_called()

    def test_restored_summary_is_used_once_and_fixed_rules_are_reintroduced(self):
        state = {"summary": "历史事实：金额 1800，待办 ID=7。", "recent_messages": [
            {"role": "user", "content": "此前要求"},
            {"role": "assistant", "content": "此前答复", "reasoning_content": "历史字段"},
        ]}
        self.scripted_model([completion("含义是预算的 15% 为 1800。")])
        result = self.run_turn("刚才的结果是什么意思？", history=[{"role": "system", "content": "旧规则"}], context_state=state)
        request = self.requests[0]
        self.assertEqual(request["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(request["messages"][1]["content"], SUMMARY_PREFIX + state["summary"])
        self.assertEqual(request["messages"][2:4], state["recent_messages"])
        self.assertEqual(result["context_state"]["summary"], state["summary"])

    def test_followup_completes_previous_todo_and_ambiguity_can_end_with_clarification(self):
        self.scripted_model([
            completion(calls=[tool_call("followup-add", "todo", {"action": "add", "title": "预算复核"})]),
            completion("已创建预算复核"),
        ])
        first = self.run_turn("创建预算复核")
        task_id = json.loads(first["messages"][-2]["content"])["data"]["todo"]["id"]
        self.scripted_model([
            completion(calls=[tool_call("followup-complete", "todo", {"action": "complete", "id": task_id})]),
            completion("刚才的待办已完成"),
        ])
        finished = self.run_turn("把刚才那条待办标为完成。", context_state=first["context_state"])
        self.assertEqual(finished["status"], "completed")
        self.assertTrue(call_tool("todo", {"action": "list"}, self.context)["data"]["todos"][0]["completed"])
        self.scripted_model([completion("你指的是预算复核还是周报？请提供标题或 ID。")])
        with patch("runtime.call_tool") as execute:
            clarified = self.run_turn("把那条待办标为完成", context_state=finished["context_state"])
        self.assertIn("请提供", clarified["answer"])
        execute.assert_not_called()

    def test_unparseable_output_is_reported_and_never_executes_tools(self):
        responses = ["not a response", SimpleNamespace(choices=[SimpleNamespace(message={}, finish_reason="stop")]),
                     SimpleNamespace(choices=[SimpleNamespace(message={"role": "assistant", "content": 123}, finish_reason="stop")]),
                     SimpleNamespace(choices=completion("回答").choices, usage=object())]
        for response in responses:
            with self.subTest(response=response):
                self.scripted_model([response])
                with patch("runtime.call_tool") as execute:
                    result = self.run_turn()
                self.assertEqual(result["error"]["code"], "MODEL_PARSE_ERROR")
                execute.assert_not_called()
                self.assertEqual([item["role"] for item in result["messages"]], ["system", "user"])
                assert_trace(self, result, self.context)

    def test_sdk_validation_and_json_decode_failures_do_not_leak_raw_details(self):
        errors = [
            openai.APIResponseValidationError(response=Mock(status_code=200, headers={}, request=Mock()), body={"detail": "private-body"}),
            json.JSONDecodeError("private-body", "invalid", 0),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.scripted_model([error])
                result = self.run_turn()
                self.assertEqual(result["error"]["code"], "MODEL_PARSE_ERROR")
                self.assertNotIn("private-body", Path(result["trace_path"]).read_text(encoding="utf-8"))
                assert_trace(self, result, self.context)

    def test_tool_execution_and_non_json_result_failures_are_fed_back_to_model(self):
        for execute in (Mock(side_effect=RuntimeError("private-internal")), Mock(return_value={"bad": object()}),
                        Mock(return_value={"bad": float("nan")})):
            with self.subTest(execute=execute):
                self.requests.clear()
                self.scripted_model([completion(calls=[tool_call("broken-tool", "calculator", {"expression": "1+2"})]),
                                     completion("工具失败，请稍后再试。")])
                with patch.dict(TOOL_REGISTRY["calculator"], {"execute": execute}):
                    result = self.run_turn()
                feedback = json.loads(self.requests[-1]["messages"][-1]["content"])
                self.assertEqual(feedback["error"]["code"], "TOOL_EXECUTION_ERROR")
                self.assertEqual(result["status"], "completed")
                self.assertNotIn("private-internal", json.dumps(result, ensure_ascii=False))
                assert_trace(self, result, self.context)

    def test_nonstandard_or_deep_json_arguments_are_returned_as_errors(self):
        for arguments in ('{"expression":NaN}', '[' * 1500 + '0' + ']' * 1500):
            with self.subTest(arguments=arguments[:20]):
                self.requests.clear()
                self.scripted_model([completion(calls=[tool_call("bad-json", "calculator", arguments)]), completion("请修正参数。")])
                result = self.run_turn()
                self.assertEqual(json.loads(self.requests[-1]["messages"][-1]["content"])["error"]["code"], "INVALID_ARGUMENTS")
                assert_trace(self, result, self.context)

    def test_repeated_todo_call_across_steps_is_replayed_and_traced(self):
        call = tool_call("same-write", "todo", {"action": "add", "title": "只创建一次"})
        self.scripted_model([completion(calls=[call]), completion(calls=[call]), completion("已创建一次。")])
        result = self.run_turn()
        tasks = call_tool("todo", {"action": "list"}, self.context)["data"]["todos"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(result["deduplicated_calls"], 1)
        feedback = [event["result"] for event in result["trace"] if event["event"] == "tool_result"]
        self.assertEqual(feedback[0], feedback[1])
        self.assertEqual(sum(event["event"] == "tool_replayed" for event in result["trace"]), 1)
        assert_trace(self, result, self.context)

    def test_reusing_call_id_with_changed_arguments_reports_conflict(self):
        self.scripted_model([
            completion(calls=[tool_call("same-write", "todo", {"action": "add", "title": "第一版"})]),
            completion(calls=[tool_call("same-write", "todo", {"action": "add", "title": "第二版"})]),
            completion("调用标识冲突，未重复创建。"),
        ])
        result = self.run_turn()
        tasks = call_tool("todo", {"action": "list"}, self.context)["data"]["todos"]
        self.assertEqual([item["title"] for item in tasks], ["第一版"])
        failures = [item for item in result["trace"] if item["event"] == "tool_result" and not item["result"]["ok"]]
        self.assertEqual(failures[0]["result"]["error"]["code"], "IDEMPOTENCY_CONFLICT")
        assert_trace(self, result, self.context)

    def test_new_run_with_same_model_call_id_can_create_another_task(self):
        results = []
        for _ in range(2):
            self.scripted_model([completion(calls=[tool_call("same-id", "todo", {"action": "add", "title": "同名任务"})]), completion("已创建。")])
            results.append(self.run_turn())
        self.assertNotEqual(results[0]["run_id"], results[1]["run_id"])
        self.assertEqual(len(call_tool("todo", {"action": "list"}, self.context)["data"]["todos"]), 2)

    def test_model_cancellation_is_recorded_before_reraising(self):
        self.client.chat.completions.create.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_turn()
        path = next((self.directory / "traces").glob("*.jsonl"))
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events[-1]["status"], "cancelled")

    def test_malformed_restored_context_state_fails_before_any_request(self):
        for state in ([], {}, {"summary": "缺少近期消息"}, {"recent_messages": []}):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "上下文状态"):
                self.run_turn(context_state=state)
        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
