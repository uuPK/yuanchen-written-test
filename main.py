import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

import openai
from dotenv import load_dotenv

from context_manager import ContextConfig
from runtime import TRACE_DIR, run_agent
from sessions import DB_PATH, create_session, list_sessions, load_session, open_session, save_session
from tools import ToolContext


def _print_trace(result: dict) -> None:
    for event in result["trace"]:
        prefix = f"[step {event['step']}]"
        if event["event"] == "model_request":
            print(f"{prefix} 请求模型，消息数：{event['message_count']}", file=sys.stderr)
        elif event["event"] == "model_response":
            calls = event["message"].get("tool_calls", [])
            if calls:
                names = [call.get("function", {}).get("name", "未知工具") for call in calls]
                decision = "调用工具：" + ", ".join(names)
            else:
                decision = "返回文本"
            print(f"{prefix} 模型选择{decision}", file=sys.stderr)
        elif event["event"] == "tool_start":
            print(f"{prefix} {event['name']} ({event['tool_call_id']}) 参数：{event['arguments']}", file=sys.stderr)
        elif event["event"] == "tool_result":
            print(f"{prefix} {event['name']} ({event['tool_call_id']}) 结果：{json.dumps(event['result'], ensure_ascii=False)}", file=sys.stderr)
        elif event["event"] == "tool_replayed":
            print(f"{prefix} {event['name']} ({event['tool_call_id']}) 已去重，返回原调用结果。", file=sys.stderr)
        elif event["event"] == "context_compacted":
            print(f"{prefix} 上下文压缩：估算 {event['estimated_tokens_before']} → {event['estimated_tokens_after']} token", file=sys.stderr)
        elif event["event"] == "compact_failed":
            print(f"{prefix} 压缩未完成，保留原历史：{event['message']}", file=sys.stderr)
    print(f"Trace：{result['trace_path']}", file=sys.stderr)


def _print_sessions(user_id: str) -> None:
    sessions = list_sessions(user_id, DB_PATH)
    if not sessions:
        print("当前用户没有 session。")
    for session in sessions:
        status = session["last_run"]["status"] if session["last_run"] else "new"
        print(f"{session['session_id']} | 已保存请求数：{session['version']} | 状态：{status}")


def _print_session(session: dict) -> None:
    print(f"当前用户：{session['user_id']}，session：{session['session_id']}，"
          f"已恢复 {len(session['messages'])} 条消息。", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="手写 ReAct Agent：由模型自主选择回答或调用工具。")
    parser.add_argument("--max-steps", type=int, default=6, help="本次用户请求最多调用模型的次数")
    parser.add_argument("--user-id", default="demo-user")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--session-id", help="选择并恢复 session，不存在时创建；默认 demo-session")
    selection.add_argument("--new-session", action="store_true", help="使用自动生成的 ID 创建新 session")
    parser.add_argument("--list-sessions", action="store_true", help="列出当前用户的 session 后退出")
    parser.add_argument("--once", action="store_true", help="仅处理一次输入后退出")
    args = parser.parse_args()
    try:
        user_id = args.user_id.strip()
        session_id = uuid4().hex if args.new_session else (
            args.session_id if args.session_id is not None else "demo-session"
        ).strip()
        if not user_id or not session_id:
            raise ValueError("用户 ID 和 session ID 不能为空。")
        if args.max_steps < 1:
            raise ValueError("max_steps 必须是正整数。")
        if args.list_sessions:
            _print_sessions(user_id)
            return 0
        load_dotenv(Path(__file__).with_name(".env"))
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
        if not api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，请在环境变量或 .env 中配置。")
        if not model:
            raise ValueError("DEEPSEEK_MODEL 不能为空。")
        context_config = ContextConfig.from_env()
        context = ToolContext(user_id, session_id, DB_PATH)
        session = create_session(context) if args.new_session else open_session(context)
        _print_session(session)
        if not args.once:
            print("命令：/new [名称]、/use 名称、/sessions、/exit", file=sys.stderr)
        exit_code = 0
        while True:
            try:
                user_input = input("你：").strip()
            except EOFError:
                if args.once:
                    raise
                return exit_code
            if user_input.startswith("/"):
                command, _, name = user_input.partition(" ")
                name = name.strip()
                if command == "/exit":
                    return exit_code
                command_exit_code = 0
                try:
                    if command == "/sessions":
                        _print_sessions(user_id)
                    elif command in ("/new", "/use"):
                        if command == "/use" and not name:
                            raise ValueError("用法：/use session名称")
                        selected = ToolContext(user_id, name or uuid4().hex, DB_PATH)
                        session = create_session(selected) if command == "/new" else load_session(selected)
                        context = selected
                        _print_session(session)
                    else:
                        raise ValueError("未知命令；支持 /new [名称]、/use 名称、/sessions、/exit。")
                except ValueError as error:
                    print(f"错误：{error}", file=sys.stderr)
                    command_exit_code = 1
                if args.once:
                    return command_exit_code
                continue
            if not user_input:
                if args.once:
                    raise ValueError("输入不能为空。")
                print("输入不能为空。", file=sys.stderr)
                continue

            session = load_session(context)
            with openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                timeout=60.0,
                max_retries=0,
            ) as client:
                result = run_agent(
                    client=client, model=model, user_input=user_input, context=context,
                    history=session["messages"], max_steps=args.max_steps, trace_dir=TRACE_DIR,
                    context_state=session["context_state"], context_config=context_config,
                )
            _print_trace(result)
            save_session(context, result, expected_version=session["version"])
            if result["status"] == "completed":
                print(f"助手：{result['answer']}")
                exit_code = 0
            else:
                print(result["error"]["message"], file=sys.stderr)
                exit_code = 2 if result["status"] == "max_steps" else 1
            if args.once:
                return exit_code
    except (EOFError, KeyboardInterrupt):
        print("\n已取消输入或执行。", file=sys.stderr)
        return 130
    except sqlite3.Error:
        print("错误：无法读写 session 数据库，请检查路径、权限或数据库状态。", file=sys.stderr)
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
