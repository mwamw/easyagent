"""Plan execution state and model reminders; authorization belongs to the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from core.Exception import ExecutionModeError
from runtime.metamessage import MetaMessageManager, MetaMessage, MetaMessageLifecycle
from runtime.agent import AgentRuntime
from Tool.ToolRegistry import ToolRegistry

from .models import ExecutionMode, PlanModeConfig, PlanModeState


class BasePlanMode(ABC):
    def get_prompt_blocks(self, context):
        return []

    @property
    @abstractmethod
    def mode(self) -> ExecutionMode:
        raise NotImplementedError

    @abstractmethod
    def bind(
        self,
        *,
        runtime: AgentRuntime,
        tool_registry: ToolRegistry | None,
    ) -> "BasePlanMode":
        raise NotImplementedError

    @abstractmethod
    def enter(self, *, allowed_actions: list[str] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def exit(self) -> None:
        raise NotImplementedError


class PlanModeManager(BasePlanMode):
    """Owns plan state and coordinates explicit module dependencies."""

    def __init__(self, config: PlanModeConfig | None = None):
        self.config = config or PlanModeConfig()
        self._state = PlanModeState()
        self._runtime: AgentRuntime | None = None
        self._tool_registry: ToolRegistry | None = None

    @property
    def state(self):
        return self._state

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode(self._runtime.execution_mode) if self._runtime else ExecutionMode.EXECUTE

    @property
    def is_active(self) -> bool:
        return self.mode == ExecutionMode.PLAN

    def bind(
        self,
        *,
        runtime: AgentRuntime,
        tool_registry: ToolRegistry | None,
    ) -> "PlanModeManager":
        if not isinstance(runtime, AgentRuntime):
            raise TypeError("runtime must be AgentRuntime")
        if self._runtime is not None and self._runtime is not runtime:
            raise RuntimeError("A Plan module cannot be shared between Agents")
        if tool_registry is not None and not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be ToolRegistry or None")
        if self._runtime is None:
            self._state.allowed_actions = list(runtime.plan_allowed_actions)
        self._runtime = runtime
        self._tool_registry = tool_registry
        return self

    def _require_bound(self) -> MetaMessageManager:
        if self._runtime is None:
            raise ExecutionModeError("PlanModeManager 尚未绑定运行时依赖。")
        return self._runtime.metamessages

    def _update_runtime(self) -> None:
        if self._runtime is not None:
            self._runtime.set_execution_mode(
                self._runtime.execution_mode,
                allowed_actions=tuple(self.state.allowed_actions),
            )

    def install_tools(self) -> None:
        if not self.config.register_tools:
            return
        self._require_bound()
        if self._tool_registry is None:
            raise ExecutionModeError("Plan 工具需要先安装 ToolRegistry。")
        from .tools import (
            EnterPlanModeTool,
            ExitPlanModeTool,
            register_enter_plan_mode_tool,
            register_exit_plan_mode_tool,
        )

        enter_tool = self._tool_registry.get_tool("EnterPlanMode")
        if enter_tool is None:
            register_enter_plan_mode_tool(self._tool_registry, plan_manager=self)
        elif isinstance(enter_tool, EnterPlanModeTool):
            enter_tool.plan_manager = self
        else:
            raise ExecutionModeError("ToolRegistry 中的 EnterPlanMode 名称已被占用。")

        exit_tool = self._tool_registry.get_tool("ExitPlanMode")
        if exit_tool is None:
            register_exit_plan_mode_tool(self._tool_registry, plan_manager=self)
        elif isinstance(exit_tool, ExitPlanModeTool):
            exit_tool.plan_manager = self
        else:
            raise ExecutionModeError("ToolRegistry 中的 ExitPlanMode 名称已被占用。")

    def enter(self, *, allowed_actions: list[str] | None = None) -> None:
        metamessages = self._require_bound()
        if self.is_active:
            if allowed_actions is not None:
                self.state.allowed_actions = list(allowed_actions)
                self._update_runtime()
            return
        actions = list(allowed_actions) if allowed_actions is not None else list(self.config.allowed_actions)
        self.state.entered_at = datetime.now()
        self.state.allowed_actions = actions
        self.state.exit_requested = False
        self._runtime.set_execution_mode("plan", allowed_actions=tuple(actions))
        metamessages.emit(
            MetaMessage(
                name="plan_mode_enter",
                content=self._build_enter_message(),
                lifecycle=MetaMessageLifecycle.PERMANENT,
                metadata={"source": "plan", "mode": "plan"},
            )
        )
        self._update_runtime()

    def request_exit(self, *, allowed_actions: list[str] | None = None) -> None:
        if not self.is_active:
            raise ExecutionModeError("当前 Agent 不在 plan 模式。")
        self.state.exit_requested = True
        if allowed_actions is not None:
            self.state.allowed_actions = list(allowed_actions)

    def exit(self) -> None:
        metamessages = self._require_bound()
        self.state.exit_requested = False
        self._runtime.set_execution_mode("execute")
        metamessages.emit(
            MetaMessage(
                name="plan_mode_exit",
                content=self.config.exit_message,
                lifecycle=MetaMessageLifecycle.PERMANENT,
                metadata={"source": "plan", "mode": "execute"},
            )
        )
        self._update_runtime()

    def _build_enter_message(self) -> str:
        actions = list(self.state.allowed_actions)
        if not actions:
            return self.config.enter_message
        detail = f"Explicitly allowed actions: {', '.join(actions)}"
        message = self.config.enter_message
        if message.rstrip().endswith("</system-reminder>"):
            return message.rstrip().removesuffix("</system-reminder>") + detail + "\n</system-reminder>"
        return f"{message}\n{detail}"

    def get_prompt_blocks(self, context):
        from prompt import PromptBlock
        if not self.is_active:
            return []
        return [PromptBlock(
            "plan_mode",
            "Plan mode still active (see full instructions earlier in conversation). "
            "Read-only inspection and planning only; do not modify files or run non-readonly tools. "
            "Use ExitPlanMode to request approval when the plan is ready.",
            placement="system_reminder", order=190,
            metadata={"cache_partition": "dynamic", "cacheable": False},
        )]

    def export_state(self) -> dict:
        return {
            "config": self.config.model_dump(mode="python"),
            "state": {**self.state.model_dump(mode="python"), "mode": self.mode.value},
        }

    def restore_state(self, payload: dict | None) -> None:
        data = dict(payload or {})
        self.config = PlanModeConfig.model_validate(data.get("config") or {})
        state = dict(data.get("state") or {})
        mode = ExecutionMode(state.pop("mode", "execute"))
        self._state = PlanModeState.model_validate(state)
        if self._runtime is not None:
            self._runtime.set_execution_mode(mode.value, allowed_actions=tuple(self.state.allowed_actions))

    def reset_session_state(self) -> None:
        self._state = PlanModeState()
        if self._runtime is not None:
            self._runtime.set_execution_mode("execute")


__all__ = ["BasePlanMode", "PlanModeManager"]
