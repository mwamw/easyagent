"""
多 Agent 协作 — 共享上下文 (Blackboard Pattern)

SharedContext 作为多 Agent 协作的共享黑板，
记录完整的消息链条和全局元数据，
供编排器和各 Agent 查阅协作进展。
"""
from __future__ import annotations

from typing import Any, Optional
from datetime import datetime
import logging

from .message import AgentMessage, MessageType

logger = logging.getLogger(__name__)


class SharedContext:
    """
    多 Agent 共享上下文
    
    核心职责：
    1. 记录完整的 Agent 间消息链（不可变审计链)
    2. 提供按 Agent / 类型过滤消息的查询能力
    3. 管理全局元数据（如任务目标、轮次计数等）
    
    设计原则：
    - 只追加、不修改（append-only）
    - 线程安全（通过 list 的 append 原子性保证基本安全）
    """

    def __init__(self, original_query: str = ""):
        self._messages: list[AgentMessage] = []
        self._metadata: dict[str, Any] = {}
        self._original_query = original_query
        self._created_at = datetime.now()

        logger.debug("SharedContext 初始化完成 (query=%s)", original_query[:50])

    # ==================== 消息管理 ====================

    def add_message(self, message: AgentMessage) -> None:
        """追加一条消息到上下文"""
        self._messages.append(message)
        logger.debug("SharedContext 新增消息: %s", message)

    def add(
        self,
        sender: str,
        receiver: str,
        content: str,
        msg_type: MessageType = "task",
        **metadata: Any,
    ) -> AgentMessage:
        """便捷方法：直接创建并追加消息"""
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            metadata=metadata,
        )
        self.add_message(msg)
        return msg

    # ==================== 查询 ====================

    @property
    def messages(self) -> list[AgentMessage]:
        """返回完整消息列表（只读副本）"""
        return list(self._messages)

    @property
    def original_query(self) -> str:
        return self._original_query

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def get_messages_for(self, agent_name: str) -> list[AgentMessage]:
        """获取与某 Agent 相关的消息（发给它的或它发的）"""
        return [
            m for m in self._messages
            if m.receiver == agent_name
            or m.receiver == "all"
            or m.sender == agent_name
        ]

    def get_messages_by_type(self, msg_type: MessageType) -> list[AgentMessage]:
        """按消息类型过滤"""
        return [m for m in self._messages if m.msg_type == msg_type]

    def get_last_result(self) -> Optional[AgentMessage]:
        """获取最后一条 result 类型消息"""
        results = self.get_messages_by_type("result")
        return results[-1] if results else None

    # ==================== 格式化 ====================

    def get_full_transcript(self) -> str:
        """
        生成纯文本格式的完整对话记录
        
        用于注入 Agent 的 Prompt 中，让 Agent 了解协作全貌。
        """
        if not self._messages:
            return "(暂无协作记录)"

        lines = [f"=== 协作记录 (共 {len(self._messages)} 条消息) ==="]
        for msg in self._messages:
            direction = f"{msg.sender} → {msg.receiver}"
            lines.append(f"[{msg.msg_type.upper()}] {direction}:")
            lines.append(f"  {msg.content}")
            lines.append("")
        return "\n".join(lines)

    def get_summary(self) -> str:
        """生成协作概要（适合日志输出）"""
        senders = set(m.sender for m in self._messages)
        type_counts = {}
        for m in self._messages:
            type_counts[m.msg_type] = type_counts.get(m.msg_type, 0) + 1

        return (
            f"SharedContext: {len(self._messages)} 条消息, "
            f"参与者={senders}, "
            f"类型分布={type_counts}"
        )

    # ==================== 元数据 ====================

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def __repr__(self) -> str:
        return f"SharedContext(messages={len(self._messages)}, query='{self._original_query[:30]}')"
