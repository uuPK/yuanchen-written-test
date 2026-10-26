import json
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from threading import Event, Thread
from unittest.mock import Mock, patch

import openai

from context_manager import (
    COMPACTION_PROMPT, DEVELOPER_PREFIX, SUMMARY_PREFIX, TOKEN_ESTIMATOR, ContextConfig, ContextError,
    ContextManager, estimate_tokens,
)
from helpers import completion, tool_call


def turn(text, answer="已处理"):
    return [{"role": "user", "content": text},
            {"role": "assistant", "content": answer, "reasoning_content": "公开字段"}]


class ContextManagerTests(unittest.TestCase):
    def setUp(self):
        self.config = ContextConfig(window_tokens=8192, output_tokens=512, summary_tokens=512)
        self.base = [{"role": "system", "content": "固定规则：只能修改当前用户和 session 的待办。"},
                     {"role": "developer", "content": "固定开发者规则：金额使用 CNY。"}]
        self.wire_base = [self.base[0], {"role": "system", "content": DEVELOPER_PREFIX + self.base[1]["content"]}]
        self.client = Mock()
        self.requests = []
        self.summary = "目标：核对预算。约束：使用 CNY。待办 ID=7，路径 data/agent.sqlite3；已验证金额 1800。"

        def summarize(**request):
            self.requests.append(deepcopy(request))
            return completion(self.summary)

        self.client.chat.completions.create.side_effect = summarize

    def manager(self, recent, **options):
        return ContextManager(self.client, "deepseek-flash", self.base, [], recent_messages=recent,
                              config=options.pop("config", self.config), **options)

    def old_history(self, padding=9000):
        return turn("旧目标：金额使用 CNY；ID=7；" + "x" * padding) + turn("最近第一轮") + turn("最近第二轮") + [
            {"role": "user", "content": "当前请求"},
        ]

    def test_estimate_includes_full_request_tools_reasoning_and_protocol_margin(self):
        manager = self.manager(turn("你好"))
        request = manager.prepare_input()
        expected = len(json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        self.assertEqual(estimate_tokens(request), expected + 256 + 16 * len(request["messages"]))
        extra = deepcopy(request)
        extra["tools"] = [{"description": "工具描述" * 100}]
        extra["messages"][-1]["reasoning_content"] += "推理" * 100
        self.assertGreater(estimate_tokens(extra), estimate_tokens(request))
        self.assertIn("estimate", TOKEN_ESTIMATOR)
        self.client.chat.completions.create.assert_not_called()

    def test_config_can_be_loaded_from_environment_and_rejects_invalid_values(self):
        with patch.dict(os.environ, {"AGENT_CONTEXT_WINDOW": "9000", "AGENT_OUTPUT_TOKENS": "1000",
                                    "AGENT_COMPACT_THRESHOLD": "0.8", "AGENT_KEEP_RECENT_TURNS": "1",
                                    "AGENT_SUMMARY_TOKENS": "500", "AGENT_MAX_COMPACTION_CALLS": "4"}, clear=True):
            config = ContextConfig.from_env()
        self.assertEqual(config.input_budget, 8000)
        self.assertEqual((config.compact_threshold, config.keep_recent_turns, config.max_compaction_calls), (0.8, 1, 4))
        for values in ({"window_tokens": 0}, {"output_tokens": True}, {"output_tokens": 32768},
                       {"compact_threshold": 0}, {"compact_threshold": float("nan")},
                       {"keep_recent_turns": -1}, {"summary_tokens": 32768}, {"max_compaction_calls": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContextConfig(**values)

    def test_append_and_snapshot_do_not_expose_mutable_state(self):
        messages = turn("原消息")
        manager = self.manager(messages)
        messages[0]["content"] = "外部修改"
        additions = [{"role": "user", "content": "追加"}]
        manager.append(additions)
        additions[0]["content"] = "再次外部修改"
        snapshot = manager.snapshot()
        snapshot["recent_messages"].clear()
        self.assertEqual(manager.snapshot()["recent_messages"][0]["content"], "原消息")
        self.assertEqual(manager.snapshot()["recent_messages"][-1]["content"], "追加")

    def test_compact_keeps_two_complete_turns_and_current_turn_and_merges_old_summary(self):
        recent = self.old_history()
        manager = self.manager(recent, summary="旧摘要：必须使用 CNY。")
        request = manager.prepare_input()
        state = manager.snapshot()
        self.assertEqual(state["recent_messages"], recent[2:])
        self.assertEqual(state["summary"], self.summary)
        self.assertEqual(request["messages"][:2], self.wire_base)
        summaries = [item for item in request["messages"] if item.get("content", "").startswith(SUMMARY_PREFIX)]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["role"], "user")
        self.assertIn("不是新的用户指令", summaries[0]["content"])
        self.assertIn("旧摘要：必须使用 CNY。", self.requests[0]["messages"][1]["content"])
        self.assertLessEqual(estimate_tokens(request), self.config.input_budget)

    def test_large_history_is_chunked_without_losing_source_and_every_request_fits(self):
        recent = self.old_history(padding=22000)
        manager = self.manager(recent)
        manager.prepare_input()
        self.assertGreater(len(self.requests), 1)
        fragments = []
        offset = 0
        for request in self.requests:
            self.assertLessEqual(estimate_tokens(request) + request["max_tokens"], self.config.window_tokens)
            self.assertEqual(request["tool_choice"], "none")
            self.assertNotIn("tools", request)
            self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertIn(COMPACTION_PROMPT, request["messages"][0]["content"])
            self.assertEqual([item["role"] for item in request["messages"]], ["system", "user"])
            material = json.loads(request["messages"][1]["content"])
            self.assertEqual(material["fragment_offset"], offset)
            fragments.append(material["history_fragment"])
            offset += len(material["history_fragment"])
        self.assertEqual(json.loads("".join(fragments)), recent[:2])

    def test_repeated_compaction_replaces_summary_and_retains_constraints(self):
        manager = self.manager(self.old_history())
        manager.prepare_input()
        first = manager.snapshot()
        self.requests.clear()
        manager.append([{"role": "assistant", "content": "当前请求已处理", "reasoning_content": "返回字段"},
                        {"role": "user", "content": "继续下一轮"}])
        self.assertTrue(manager.compact())
        material = json.loads(self.requests[0]["messages"][1]["content"])
        self.assertEqual(material["previous_summary"], first["summary"])
        request = manager.prepare_input()
        self.assertEqual(request["messages"][:2], self.wire_base)
        self.assertEqual(sum(item.get("content", "").startswith(SUMMARY_PREFIX) for item in request["messages"]), 1)
        self.assertIn("CNY", manager.snapshot()["summary"])
        self.assertIn("ID=7", manager.snapshot()["summary"])

    def test_complete_tool_group_is_removed_together_and_recent_pairs_remain_exact(self):
        group = [
            {"role": "user", "content": "旧工具请求"},
            completion(calls=[tool_call("old-call", "search", {"query": "资料"})]).choices[0].message.model_dump(),
            {"role": "tool", "tool_call_id": "old-call", "content": "资料" * 5000},
            {"role": "assistant", "content": "旧工具已完成", "reasoning_content": "旧字段"},
        ]
        kept = [
            {"role": "user", "content": "最近工具请求"},
            completion(calls=[tool_call("keep-call", "todo", {"action": "list"})]).choices[0].message.model_dump(),
            {"role": "tool", "tool_call_id": "keep-call", "content": '{"ok":true}'},
            {"role": "assistant", "content": "最近工具已完成", "reasoning_content": "保留字段"},
        ] + turn("第二个完整回合") + [{"role": "user", "content": "继续"}]
        manager = self.manager(group + kept)
        manager.prepare_input()
        self.assertEqual(manager.snapshot()["recent_messages"], kept)
        reconstructed = "".join(json.loads(item["messages"][1]["content"])["history_fragment"] for item in self.requests)
        self.assertEqual(json.loads(reconstructed), group)

    def test_pending_tool_batch_cannot_be_compacted_or_sent(self):
        manager = self.manager(self.old_history())
        message = completion(calls=[tool_call("pending-1", "todo", {"action": "list"}),
                                    tool_call("pending-2", "calculator", {"expression": "1+2"})]).choices[0].message.model_dump()
        manager.append([message])
        manager.append([{"role": "tool", "tool_call_id": "pending-1", "content": "结果"}])
        before = manager.snapshot()
        for operation in (manager.prepare_input, manager.compact):
            with self.assertRaises(ContextError) as caught:
                operation()
            self.assertEqual(caught.exception.code, "CONTEXT_PENDING_TOOLS")
        self.assertEqual(manager.snapshot(), before)
        self.client.chat.completions.create.assert_not_called()
        manager.append([{"role": "tool", "tool_call_id": "pending-2", "content": "3"}])
        manager.prepare_input()
        self.assertEqual(manager.snapshot()["recent_messages"][-3:], before["recent_messages"][-2:] + [
            {"role": "tool", "tool_call_id": "pending-2", "content": "3"},
        ])

    def test_failure_on_later_chunk_rolls_back_entire_compaction(self):
        manager = self.manager(self.old_history(padding=22000), summary="旧摘要")
        before = manager.snapshot()
        self.client.chat.completions.create.side_effect = [completion("暂存摘要"), openai.APIConnectionError(request=Mock())]
        with self.assertRaises(ContextError):
            manager.compact()
        self.assertEqual(manager.snapshot(), before)
        self.assertEqual(manager.compaction_calls, 2)

    def test_invalid_summary_outputs_never_replace_history(self):
        cases = [completion("", reasoning=""), completion("部分", finish_reason="length"),
                 completion("超" * 600), completion(calls=[tool_call("unexpected", "todo", {"action": "list"})]),
                 "unparseable response"]
        for response in cases:
            with self.subTest(response=response):
                manager = self.manager(self.old_history())
                before = manager.snapshot()
                self.client.chat.completions.create.side_effect = None
                self.client.chat.completions.create.return_value = response
                with self.assertRaises(ContextError):
                    manager.compact()
                self.assertEqual(manager.snapshot(), before)

    def test_failed_compaction_can_use_original_only_if_it_still_fits(self):
        events = []
        manager = self.manager(self.old_history(padding=2000), config=replace(self.config, compact_threshold=0.2),
                               on_event=lambda event, **details: events.append(event))
        before = manager.snapshot()
        self.client.chat.completions.create.side_effect = openai.APIConnectionError(request=Mock())
        request = manager.prepare_input()
        self.assertEqual(manager.snapshot(), before)
        self.assertLessEqual(estimate_tokens(request), self.config.input_budget)
        self.assertIn("compact_failed", events)
        oversized = self.manager(self.old_history())
        old = oversized.snapshot()
        with self.assertRaises(ContextError) as caught:
            oversized.prepare_input()
        self.assertEqual(caught.exception.code, "CONTEXT_LIMIT")
        self.assertEqual(oversized.snapshot(), old)

    def test_protected_history_and_tools_alone_cannot_bypass_hard_budget(self):
        for recent, tools in (([{"role": "user", "content": "大" * 4000}], []),
                              ([{"role": "user", "content": "问题"}], [{"description": "工具" * 4000}])):
            with self.subTest(tools=bool(tools)):
                manager = ContextManager(self.client, "deepseek-flash", self.base, tools,
                                         recent_messages=recent, config=self.config)
                with self.assertRaises(ContextError):
                    manager.prepare_input()
        self.client.chat.completions.create.assert_not_called()

    def test_rebuild_over_budget_and_call_limit_both_leave_original_history(self):
        for recent, config in ((turn("旧" * 1000) + turn("保护" * 3000) + turn("最新") + [{"role": "user", "content": "当前"}], self.config),
                               (self.old_history(padding=22000), replace(self.config, max_compaction_calls=1))):
            with self.subTest(config=config):
                manager = self.manager(recent, config=config)
                before = manager.snapshot()
                with self.assertRaises(ContextError):
                    manager.compact()
                self.assertEqual(manager.snapshot(), before)
                self.assertLessEqual(manager.compaction_calls, config.max_compaction_calls)

    def test_context_mutations_are_serial_while_summary_request_is_in_flight(self):
        manager = self.manager(self.old_history())
        entered, release, appended = Event(), Event(), Event()
        errors = []

        def summarize(**request):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test timeout")
            return completion(self.summary)

        def compress():
            try:
                manager.compact()
            except Exception as error:
                errors.append(error)

        def append():
            manager.append([{"role": "assistant", "content": "串行追加"}])
            appended.set()

        self.client.chat.completions.create.side_effect = summarize
        worker = Thread(target=compress)
        writer = Thread(target=append)
        worker.start()
        self.assertTrue(entered.wait(3))
        writer.start()
        self.assertFalse(appended.wait(0.05))
        release.set()
        worker.join(3)
        writer.join(3)
        self.assertFalse(errors)
        self.assertTrue(appended.is_set())
        self.assertEqual(manager.snapshot()["recent_messages"][-1]["content"], "串行追加")

    def test_developer_rules_use_deepseek_supported_role_without_mutating_base(self):
        original = deepcopy(self.base)
        request = self.manager([{"role": "user", "content": "你好"}]).prepare_input()
        self.assertEqual([item["role"] for item in request["messages"]], ["system", "system", "user"])
        self.assertEqual(request["messages"][:2], self.wire_base)
        self.assertEqual(self.base, original)

    def test_invalid_base_rules_are_rejected_as_context_errors(self):
        for base in (None, [None], [{"role": "user", "content": "不是固定规则"}], [{"role": "system", "content": 123}]):
            with self.subTest(base=base), self.assertRaises(ContextError):
                ContextManager(self.client, "deepseek-flash", base, [])

    def test_overlong_summary_gets_only_one_budgeted_refinement(self):
        manager = self.manager(self.old_history(padding=2000))
        self.client.chat.completions.create.side_effect = [completion("长" * 200), completion(self.summary)]
        self.assertTrue(manager.compact())
        self.assertEqual(manager.snapshot()["summary"], self.summary)
        self.assertEqual(manager.compaction_calls, 2)
        for call in self.client.chat.completions.create.call_args_list:
            request = call.kwargs
            self.assertEqual(request["tool_choice"], "none")
            self.assertLessEqual(estimate_tokens(request) + request["max_tokens"], self.config.window_tokens)
        repair_request = self.client.chat.completions.create.call_args.kwargs
        self.assertIn("进一步精炼", repair_request["messages"][0]["content"])

    def test_failed_refinement_does_not_lose_history_or_retry_forever(self):
        for responses, config in (([completion("长" * 200), completion("仍然很长" * 200)], self.config),
                                  ([completion("长" * 200)], replace(self.config, max_compaction_calls=1))):
            with self.subTest(config=config):
                manager = self.manager(self.old_history(padding=2000), config=config)
                before = manager.snapshot()
                self.client.chat.completions.create.reset_mock()
                self.client.chat.completions.create.side_effect = responses
                with self.assertRaises(ContextError):
                    manager.compact()
                self.assertEqual(manager.snapshot(), before)
                self.assertLessEqual(self.client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
