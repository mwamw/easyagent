"""Default terminal interaction, using only public Agent APIs."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
from typing import Awaitable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.filters import to_filter
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console
from rich.text import Text

from agent import BasicAgent
from core.Exception import AgentStopRequested, ToolInterruption
from runtime import RuntimeEvent, RuntimeEventType
from runtime.interactions import InteractionReply
from ui.base import BaseAgentUI
from .rendering import TerminalRenderer, safe_text


class TerminalUI(BaseAgentUI):
    def __init__(
        self, *, show_tool_results: bool = False, streaming: bool = True,
        max_iter: int = 20, console: Console | None = None,
        input: Input | None = None, output: Output | None = None,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.console = console or Console()
        self.renderer = TerminalRenderer(self.console, show_tool_results=show_tool_results)
        self.streaming = streaming
        self.max_iter = max_iter
        self.input, self.output = input, output
        self._session: PromptSession | None = None
        self._running = False
        self._exit = False
        self._task: asyncio.Task | None = None
        self._agent: BasicAgent | None = None
        self._subscription: str | None = None

    @property
    def session(self) -> PromptSession:
        if self._session is None:
            bindings = KeyBindings()

            @bindings.add("enter")
            def submit(event):
                event.current_buffer.validate_and_handle()

            @bindings.add("escape", "enter")
            def newline(event):
                event.current_buffer.insert_text("\n")

            self._session = PromptSession(
                input=self.input, output=self.output, history=InMemoryHistory(),
                multiline=True, key_bindings=bindings, enable_history_search=True,
                reserve_space_for_menu=0,
            )
            window = self._session.app.layout.current_window
            window.dont_extend_height = to_filter(True)
            window.height = Dimension(min=1, max=8)
        return self._session

    async def _input(self, message: str) -> str:
        return (await self.session.prompt_async(message)).strip()

    async def run(self, agent: BasicAgent) -> None:
        if self._running:
            raise RuntimeError("This UI is already running")
        self._running, self._exit, self._agent = True, False, agent
        loop = asyncio.get_running_loop()

        def event_handler(event: RuntimeEvent) -> None:
            if self._running:
                loop.call_soon_threadsafe(self._runtime_event, event)

        self._subscription = agent.event_bus.subscribe(event_handler, event_types=(
            RuntimeEventType.TOOL_INVOKE_STARTED,
            RuntimeEventType.TOOL_INVOKE_COMPLETED,
            RuntimeEventType.TOOL_INVOKE_FAILED,
        ))
        self.console.print(Text(f"EasyAgent · {safe_text(agent.name)}", style="bold cyan"))
        self.console.print(Text(f"Workspace: {safe_text(agent.runtime.workspace_root)}", style="dim"))
        self.console.print(Text(f"Model: {safe_text(agent.llm.model)}", style="dim"))
        self.console.print("Enter 发送 · Alt+Enter 换行 · Ctrl+C 取消输入/请求停止 · /help", markup=False)
        try:
            while not self._exit:
                if agent.get_pending_interruption() is not None:
                    await self._interaction(agent)
                    continue
                try:
                    query = await self._input("› ")
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                if not query:
                    continue
                if query.startswith("/"):
                    self._command(query)
                    continue
                self.console.print(Text("你", style="bold green"))
                self.console.print(safe_text(query), markup=False)
                agent.clear_stop_request()
                await self._operation(agent, self._invoke(agent, query), new_turn=True)
        finally:
            await self.aclose()

    def _command(self, command: str) -> None:
        if command in ("/exit", "/quit"):
            self._exit = True
        elif command == "/help":
            self.console.print(
                "/help  帮助\n/tools 查看最近一轮完整工具结果\n"
                "/clear 仅清理屏幕，不修改会话\n/exit 退出\n"
                "运行中不接受新请求。Ctrl+C 协作停止，等待已开始的工具结束。",
                markup=False,
            )
        elif command == "/clear":
            self.console.clear()
        elif command == "/tools":
            self.renderer.details()
        else:
            self.console.print("未知命令，使用 /help。", markup=False)

    async def _invoke(self, agent: BasicAgent, query: str) -> None:
        if self.streaming:
            async with aclosing(agent.astream(query, max_iter=self.max_iter)) as events:
                async for event in events:
                    self.renderer.event(event)
        else:
            result = await agent.ainvoke(query, max_iter=self.max_iter)
            self.renderer.text = result

    async def _continue(self, agent: BasicAgent, reply: InteractionReply) -> None:
        pending = agent.get_pending_interruption()
        try:
            await agent.arespond_to_interruption(reply)
        finally:
            current = agent.get_pending_interruption()
            if pending and (not current or current.get("request_id") != pending.get("request_id")):
                item = self.renderer.tools.get(str(pending.get("tool_id") or ""))
                if item is not None and item.status == "needs_confirmation":
                    status = {"deny": "denied", "answer": "answered", "approve": "resolved"}[reply.action]
                    self.renderer.tool_result(
                        item.call_id, item.name, status,
                        reply.answer or f"User response: {reply.action}",
                    )
        if agent.is_stop_requested():
            raise AgentStopRequested()
        async with aclosing(agent.astream_continue(max_iter=self.max_iter)) as events:
            async for event in events:
                self.renderer.event(event)

    async def _operation(self, agent: BasicAgent, operation: Awaitable[None], *, new_turn: bool) -> None:
        async def execute():
            await operation

        self.renderer.start(new_turn=new_turn)
        self._task = asyncio.create_task(execute())
        terminal_input = self.session.app.input

        def keys_ready():
            for key in terminal_input.read_keys():
                if key.key == Keys.ControlC:
                    agent.request_stop("User requested cancellation")
                    self.renderer.set_status("正在停止：等待当前操作返回，不再启动后续操作")
            if terminal_input.closed:
                agent.request_stop("Terminal input closed")
                self._exit = True

        try:
            with terminal_input.raw_mode(), terminal_input.attach(keys_ready):
                while not self._task.done():
                    await asyncio.wait({self._task}, timeout=1 / 12)
                    self.renderer.refresh()
                await self._task
        except ToolInterruption:
            self.renderer.set_status("等待用户回复")
        except AgentStopRequested:
            for item in self.renderer.tools.values():
                if item.status == "pending":
                    item.status = "interrupted"
            self.console.print("已停止后续执行。已启动的外部/后台进程不会因此自动终止。", markup=False)
        except asyncio.CancelledError:
            agent.request_stop("UI is closing")
            # Do not abandon a thread-backed Tool and falsely release the Agent.
            try:
                await asyncio.shield(self._task)
            except Exception:
                pass
            raise
        except NotImplementedError:
            self.console.print("当前 Executor 或权限引擎不支持这项交互；可拒绝待处理操作或退出。", markup=False)
        except Exception as exc:
            self.console.print(f"请求失败：{type(exc).__name__}。会话保留，可重试或退出。", markup=False)
        finally:
            if self._task is not None and not self._task.done():
                agent.request_stop("UI operation ended; waiting for Agent cleanup")
                try:
                    await asyncio.shield(self._task)
                except Exception:
                    pass
            self._task = None
            self.renderer.finish()

    def _runtime_event(self, event: RuntimeEvent) -> None:
        if not self._running or self._task is None:
            return
        data = event.data
        name = str(data.get("tool_name") or "tool")
        if event.type == RuntimeEventType.TOOL_INVOKE_STARTED:
            self.renderer.set_status("处理工具请求：" + name)
        else:
            result = data.get("result")
            status = getattr(result, "status", "error" if event.type == RuntimeEventType.TOOL_INVOKE_FAILED else "success")
            if getattr(result, "error_type", None) == "permission_denied":
                status = "denied"
            call_id = str(data.get("tool_call_id") or name)
            content = str(data.get("output") or "")
            item = self.renderer.tools.get(call_id)
            if item is None or item.status != status or item.content != content:
                self.renderer.tool_result(call_id, name, status, content)
                self.renderer.tools[call_id].arguments = dict(data.get("arguments") or {})

    async def _interaction(self, agent: BasicAgent) -> None:
        pending = agent.get_pending_interruption()
        metadata = dict(pending.get("metadata") or {})
        kind = metadata.get("interaction_type")
        self.console.print(Text("需要你的回复", style="bold yellow"))
        self.console.print(safe_text(pending.get("tool_name", "")), markup=False)
        self.console.print(Text("Workspace: " + safe_text(agent.runtime.workspace_root)))
        self.console.print(safe_text(json.dumps(pending.get("tool_args", {}), ensure_ascii=False, indent=2, default=str)), markup=False)
        self.console.print(safe_text(pending.get("message", "")), markup=False)
        request_id = str(pending.get("request_id") or "")
        try:
            if kind == "ask_user_question":
                answer = await self._questions(metadata.get("questions") or [])
                if answer is None:
                    self._exit = True
                    return
                reply = InteractionReply(request_id, "deny" if answer == "/deny" else "answer", answer if answer != "/deny" else "")
            else:
                supported = metadata.get("permission_behavior") == "ask" or kind in ("enter_plan_mode", "exit_plan_mode")
                while True:
                    prompt = "1 允许本次 / 2 拒绝 / /exit： " if supported else "此交互暂不支持批准。2 拒绝 / /exit： "
                    choice = await self._input(prompt)
                    if choice == "/exit":
                        self._exit = True
                        return
                    if choice in ("2", "n", "no", "拒绝"):
                        reply = InteractionReply(request_id, "deny")
                        break
                    if supported and choice in ("1", "y", "yes", "允许"):
                        reply = InteractionReply(request_id, "approve")
                        break
            agent.clear_stop_request()
            await self._operation(agent, self._continue(agent, reply), new_turn=False)
        except KeyboardInterrupt:
            self.console.print("回复未提交，中断仍待处理。", markup=False)
        except EOFError:
            self._exit = True

    async def _questions(self, questions: list[dict]) -> str | None:
        answers = []
        for question in questions or [{"question": "请输入回复"}]:
            self.console.print(safe_text(question.get("question", "")), markup=False)
            options = list(question.get("options") or [])
            for index, option in enumerate(options, 1):
                self.console.print(
                    f"{index}. {safe_text(option.get('label', ''))} — {safe_text(option.get('description', ''))}",
                    markup=False,
                )
            while True:
                answer = await self._input("选择编号或输入回答（多选用逗号分隔；/deny 拒绝；/exit 退出）： ")
                if answer == "/exit":
                    return None
                if answer == "/deny":
                    return answer
                if not answer:
                    continue
                parts = [value.strip() for value in answer.split(",")]
                if options and all(value.isdigit() and 1 <= int(value) <= len(options) for value in parts):
                    if len(parts) > 1 and not question.get("multiSelect"):
                        self.console.print("该问题只能单选，请重新回答。", markup=False)
                        continue
                    answer = ", ".join(str(options[int(value) - 1].get("label", "")) for value in parts)
                break
            answers.append({"question": question.get("question", ""), "answer": answer})
        return json.dumps({"answers": answers}, ensure_ascii=False)

    async def aclose(self) -> None:
        self._exit = True
        if self._session is not None and self._session.app.is_running:
            self._session.app.exit(exception=EOFError())
        if self._task is not None and not self._task.done() and self._agent is not None:
            self._agent.request_stop("UI is closing")
            try:
                await asyncio.shield(self._task)
            except Exception:
                pass
        if self._subscription is not None and self._agent is not None:
            self._agent.event_bus.unsubscribe(self._subscription)
            self._subscription = None
        self.renderer.finish()
        self._running = False
        self._agent = None
