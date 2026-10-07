"""Incremental terminal presentation; no permission or Agent execution logic."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from runtime import AgentStreamEvent, AgentStreamEventType

_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b.")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def safe_text(value: object) -> str:
    return _CONTROL.sub("", _ESCAPES.sub("", str(value)))


def summary(value: object, limit: int = 100) -> str:
    text = " ".join(safe_text(value).split())
    return text if len(text) <= limit else text[:max(0, limit - 1)] + "…"


@dataclass
class ToolDisplay:
    name: str
    call_id: str
    arguments: dict
    status: str = "pending"
    content: str = ""


@dataclass
class SafeMarkdown:
    content: str

    def __rich_console__(self, console, options):
        # Incomplete/unknown lexer names can activate broken third-party plugins.
        # Materialize before emitting so fallback cannot duplicate half a message.
        try:
            segments = list(console.render(Markdown(self.content), options))
        except Exception:
            segments = list(console.render(Text(self.content), options))
        yield from segments


class TerminalRenderer:
    def __init__(self, console: Console, *, show_tool_results: bool = False):
        self.console = console
        self.show_tool_results = show_tool_results
        self.text = ""
        self.status = "等待模型"
        self.started = time.monotonic()
        self.tools: dict[str, ToolDisplay] = {}
        self.live: Live | None = None
        self._dirty = True
        self._last_refresh = 0.0

    def start(self, *, new_turn: bool = True) -> None:
        if new_turn:
            self.tools.clear()
        self.text = ""
        self.status = "等待模型"
        self.started = time.monotonic()
        self._dirty = True
        if self.console.is_terminal:
            self.live = Live("", console=self.console, auto_refresh=False, transient=True,
                             vertical_overflow="ellipsis", redirect_stdout=True, redirect_stderr=True)
            self.live.start()

    def set_status(self, status: str) -> None:
        self.status = safe_text(status)
        self._dirty = True

    def refresh(self) -> None:
        # Streaming updates are coalesced; idle tool timers refresh once a second.
        now = time.monotonic()
        if self.live is None or (not self._dirty and now - self._last_refresh < 1):
            return
        elapsed = now - self.started
        # Never parse incomplete code fences on each token. Bound preview work,
        # not the retained answer; final output below uses the complete text.
        content = Text(safe_text(self.text[-12000:]))
        self.live.update(Group(content, Text(f"{self.status} · {elapsed:.1f}s · Ctrl+C 请求停止", style="dim")), refresh=True)
        self._dirty = False
        self._last_refresh = now

    def _commit_text(self) -> None:
        if self.text:
            self.console.print(Text("Agent", style="bold cyan"))
            self.console.print(SafeMarkdown(safe_text(self.text)))
            self.text = ""
        if self.live is not None:
            self.live.update("", refresh=True)

    def event(self, event: AgentStreamEvent) -> None:
        kind = event.type
        if kind == AgentStreamEventType.REASONING_DELTA:
            return
        if kind == AgentStreamEventType.TEXT_DELTA:
            self.text += event.content or ""
            self.status = "正在回答"
        elif kind == AgentStreamEventType.TOOL_CALL:
            self._commit_text()
            name = str(event.data.get("tool_name") or "tool")
            call_id = str(event.data.get("tool_call_id") or name)
            arguments = dict(event.data.get("arguments") or {})
            self.tools[call_id] = ToolDisplay(name, call_id, arguments)
            self.console.print(Text(f"  → {safe_text(name)}  {summary(json.dumps(arguments, ensure_ascii=False, default=str))}", style="dim"))
            # A model-selected call is not yet proof of authorization/execution.
            self.status = f"处理工具请求：{name}"
        elif kind == AgentStreamEventType.TOOL_RESULT:
            status = str(event.data.get("status") or "success")
            if getattr(event.data.get("result"), "error_type", None) == "permission_denied":
                status = "denied"
            self.tool_result(
                str(event.data.get("tool_call_id") or ""), str(event.data.get("tool_name") or "tool"),
                status, event.content or "",
            )
        elif kind == AgentStreamEventType.FINAL:
            self.text = event.content if event.content is not None else self.text
            self._commit_text()
            self.status = "完成"
        # ERROR events are followed by the same exception; the controller displays it once.
        self._dirty = True

    def tool_result(self, call_id: str, name: str, status: str, content: str) -> None:
        item = self.tools.setdefault(call_id, ToolDisplay(name, call_id, {}))
        if item.status == status and item.content == content:
            return
        item.status, item.content = status, content
        marker = {"success": "✓", "answered": "✓", "resolved": "✓", "needs_confirmation": "?"}.get(status, "✗")
        line = f"  {marker} {safe_text(name)} · {safe_text(status)}"
        if self.show_tool_results:
            line += "  " + summary(content, 160)
        self.console.print(Text(line))
        self.status = "等待模型"
        self._dirty = True

    def finish(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None
        self._commit_text()

    def details(self) -> None:
        if not self.tools:
            self.console.print("最近一轮没有工具记录。", markup=False)
        for item in self.tools.values():
            self.console.print(Text(safe_text(f"{item.name} · {item.status}"), style="bold"))
            self.console.print(safe_text(json.dumps(item.arguments, ensure_ascii=False, indent=2, default=str)), markup=False)
            self.console.print(safe_text(item.content), markup=False)
