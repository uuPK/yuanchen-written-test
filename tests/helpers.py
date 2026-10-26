import json
from pathlib import Path
from tempfile import TemporaryDirectory

from openai.types.chat import ChatCompletion


def temporary_directory(prefix):
    root = Path(__file__).resolve().parents[1] / "data" / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(prefix=prefix, dir=root)


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
    }}


def completion(content=None, calls=None, reasoning="公开返回的推理字段", finish_reason=None):
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls is not None:
        message["tool_calls"] = calls
    return ChatCompletion.model_validate({
        "id": "test-response", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": finish_reason or ("tool_calls" if calls else "stop"), "message": message}],
    })


def assert_trace(test, result, context):
    events = [json.loads(line) for line in Path(result["trace_path"]).read_text(encoding="utf-8").splitlines()]
    test.assertEqual(events, result["trace"])
    test.assertEqual([item["sequence"] for item in events], list(range(1, len(events) + 1)))
    pending = {}
    for event in events:
        test.assertEqual((event["run_id"], event["user_id"], event["session_id"]),
                         (result["run_id"], context.user_id, context.session_id))
        test.assertIn("+00:00", event["time"])
        key = (event["step"], event.get("tool_call_id"))
        if event["event"] == "tool_start":
            test.assertNotIn(key, pending)
            pending[key] = event["name"]
        elif event["event"] == "tool_result":
            test.assertEqual(pending.pop(key), event["name"])
            test.assertEqual(set(event["result"]), {"ok", "data", "error"})
            test.assertGreaterEqual(event["duration_ms"], 0)
        elif event["event"] == "model_request":
            test.assertFalse(pending)
        elif event["event"] == "compact_request":
            test.assertLessEqual(event["estimated_tokens"], event["input_budget"])
    test.assertFalse(pending)
    test.assertEqual(events[-1]["event"], "run_end")
    test.assertEqual(events[-1]["status"], result["status"])
    test.assertEqual(sum(item["event"] == "model_request" for item in events), result["steps"])
    test.assertGreaterEqual(result["duration_ms"], 0)
