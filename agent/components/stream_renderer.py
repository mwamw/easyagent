"""Stream display renderer interfaces and default console implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStreamDisplayRenderer(ABC):
    """Abstract stream renderer used by BasicAgent for CLI output."""

    @abstractmethod
    def create_state(self) -> Any:
        pass

    @abstractmethod
    def render_event(self, state: Any, event: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def render_final(self, state: Any, final_text: str) -> None:
        pass


class ConsoleStreamDisplayRenderer(BaseStreamDisplayRenderer):
    """Default console renderer preserving BasicAgent's current stdout format."""

    def create_state(self) -> dict[str, Any]:
        return {
            "current_round": 0,
            "current_section": None,
            "thinking_text": "",
            "content_text": "",
            "tool_calls": "",
        }

    @staticmethod
    def _start_round(state: dict[str, Any], round_number: int) -> None:
        if state["current_round"] > 0:
            print()
        print(f"round {round_number}")
        state["current_round"] = round_number
        state["current_section"] = "round"
        state["thinking_text"] = ""
        state["content_text"] = ""
        state["tool_calls"] = ""

    @staticmethod
    def _print_header(state: dict[str, Any], header: str) -> None:
        if state["current_section"] is None:
            print(f"{header}:")
        elif state["current_section"] != header:
            print()
            print(f"{header}:")
        state["current_section"] = header

    @classmethod
    def _append_text(
        cls,
        state: dict[str, Any],
        header: str,
        state_key: str,
        text: str,
    ) -> None:
        if not text:
            return
        cls._print_header(state, header)
        print(text, end="", flush=True)
        state[state_key] += text

    @classmethod
    def _append_snapshot(
        cls,
        state: dict[str, Any],
        header: str,
        state_key: str,
        full_text: str,
    ) -> None:
        if not full_text:
            return
        delta = cls._snapshot_suffix(state[state_key], full_text)
        if not delta:
            return
        cls._append_text(state, header, state_key, delta)

    @staticmethod
    def _snapshot_suffix(displayed: str, full_text: str) -> str:
        if not full_text:
            return ""
        if full_text.startswith(displayed):
            return full_text[len(displayed):]
        if displayed:
            return ""
        return full_text

    def render_event(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "round_start":
            self._start_round(state, int(event.get("round", 1) or 1))
            return
        if event_type == "thinking_delta":
            self._append_text(
                state,
                "thinking content",
                "thinking_text",
                event.get("delta", "") or "",
            )
            return
        if event_type == "text_delta":
            self._append_text(
                state,
                "content",
                "content_text",
                event.get("delta", "") or "",
            )
            return
        if event_type in {"tool_calls", "final_response", "final"}:
            self._append_snapshot(
                state,
                "thinking content",
                "thinking_text",
                event.get("thinking", "") or "",
            )
            self._append_snapshot(
                state,
                "content",
                "content_text",
                event.get("content", "") or "",
            )
        if event_type == "tool_call":
            self._append_text(
                state,
                "tool_calls",
                "tool_calls",
                f"{event.get('tool_name', '')} : {event.get('tool_args', '')}\n",
            )
            return
        if event_type == "interruption":
            self._append_text(
                state,
                "interrupt",
                "content_text",
                f"{event.get('content', '')}\n",
            )

    def render_final(self, state: dict[str, Any], final_text: str) -> None:
        if state["current_section"] is None:
            print("final res:")
        else:
            print()
            print("final res:")
        print(final_text)
        state["current_section"] = "final res"


__all__ = ["BaseStreamDisplayRenderer", "ConsoleStreamDisplayRenderer"]
