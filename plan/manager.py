"""Standalone plan-mode module built on permission and meta-message modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from core.Exception import ExecutionModeError
from core.permissions import PermissionContext, PermissionMode
from metamessage import BaseMetaMessageManager, MetaMessage, MetaMessageLifecycle
from Tool.ToolRegistry import ToolRegistry

from .models import ExecutionMode, ModeController, PlanModeConfig


class BasePlanMode(ABC):
    @property
    @abstractmethod
    def mode(self) -> ExecutionMode:
        raise NotImplementedError

    @abstractmethod
    def bind(
        self,
        *,
        permission_context: PermissionContext,
        metamessage_manager: BaseMetaMessageManager,
        tool_registry: ToolRegistry | None,
        runtime_refresher: Callable[[], None],
    ) -> "BasePlanMode":
        raise NotImplementedError

    @abstractmethod
    def enter(self, *, allowed_actions: list[str] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def exit(self, *, permission_mode: PermissionMode | str = PermissionMode.DEFAULT) -> None:
        raise NotImplementedError


class PlanModeManager(BasePlanMode):
    """Owns plan state and coordinates explicit module dependencies."""

    def __init__(self, config: PlanModeConfig | None = None):
        self.config = config or PlanModeConfig()
        self._controller = ModeController()
        self._permission_context: PermissionContext | None = None
        self._metamessage_manager: BaseMetaMessageManager | None = None
        self._tool_registry: ToolRegistry | None = None
        self._runtime_refresher: Callable[[], None] = lambda: None

    @property
    def state(self):
        return self._controller.state

    @property
    def mode(self) -> ExecutionMode:
        return self._controller.mode

    @property
    def is_active(self) -> bool:
        return self.mode == ExecutionMode.PLAN

    def bind(
        self,
        *,
        permission_context: PermissionContext,
        metamessage_manager: BaseMetaMessageManager,
        tool_registry: ToolRegistry | None,
        runtime_refresher: Callable[[], None],
    ) -> "PlanModeManager":
        if not isinstance(permission_context, PermissionContext):
            raise TypeError("permission_context must be PermissionContext")
        if not isinstance(metamessage_manager, BaseMetaMessageManager):
            raise TypeError("metamessage_manager must be BaseMetaMessageManager")
        if tool_registry is not None and not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be ToolRegistry or None")
        self._permission_context = permission_context
        self._metamessage_manager = metamessage_manager
        self._tool_registry = tool_registry
        self._runtime_refresher = runtime_refresher
        return self

    def _require_bound(self) -> tuple[PermissionContext, BaseMetaMessageManager]:
        if self._permission_context is None or self._metamessage_manager is None:
            raise ExecutionModeError("PlanModeManager 尚未绑定运行时依赖。")
        return self._permission_context, self._metamessage_manager

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
        permission_context, metamessages = self._require_bound()
        if self.is_active:
            if allowed_actions is not None:
                self.state.allowed_actions = list(allowed_actions)
                self._runtime_refresher()
            return
        actions = list(allowed_actions) if allowed_actions is not None else list(self.config.allowed_actions)
        self._controller.enter(allowed_actions=actions)
        permission_context.set_mode(PermissionMode.PLAN)
        metamessages.emit(
            MetaMessage(
                name="plan_mode_enter",
                content=self._build_enter_message(),
                lifecycle=MetaMessageLifecycle.PERMANENT,
                metadata={"source": "plan", "mode": "plan"},
            )
        )
        self._runtime_refresher()

    def request_exit(self, *, allowed_actions: list[str] | None = None) -> None:
        if not self.is_active:
            raise ExecutionModeError("当前 Agent 不在 plan 模式。")
        self._controller.request_exit(allowed_actions=allowed_actions)

    def exit(self, *, permission_mode: PermissionMode | str = PermissionMode.DEFAULT) -> None:
        permission_context, metamessages = self._require_bound()
        self._controller.exit()
        permission_context.set_mode(permission_mode)
        metamessages.emit(
            MetaMessage(
                name="plan_mode_exit",
                content=self.config.exit_message,
                lifecycle=MetaMessageLifecycle.PERMANENT,
                metadata={"source": "plan", "mode": "execute"},
            )
        )
        self._runtime_refresher()

    def _build_enter_message(self) -> str:
        actions = list(self.state.allowed_actions)
        if not actions:
            return self.config.enter_message
        return f"{self.config.enter_message}\nExplicitly allowed actions: {', '.join(actions)}"

    def export_state(self) -> dict:
        return {
            "config": self.config.model_dump(mode="python"),
            "state": self._controller.export_state(),
        }

    def restore_state(self, payload: dict | None) -> None:
        data = dict(payload or {})
        self.config = PlanModeConfig.model_validate(data.get("config") or {})
        self._controller.restore_state(data.get("state"))


__all__ = ["BasePlanMode", "PlanModeManager"]
