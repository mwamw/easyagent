"""
记忆系统上下文来源

对接 memory/V2/MemoryManage，从多层记忆中检索相关上下文。
"""
from typing import List, Optional, Any
from context.window import ContextItem
from context.source.base import BaseContextSource


# 记忆类型到优先级的默认映射
_MEMORY_PRIORITY = {
    "working": 0.85,     # 工作记忆优先级最高
    "episodic": 0.65,    # 情景记忆次之
    "semantic": 0.60,    # 语义记忆
    "perceptual": 0.50,  # 感知记忆
}


class MemoryContextSource(BaseContextSource):
    """从 V2 记忆系统获取上下文"""

    def __init__(
        self,
        memory_manage: Any = None,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
    ):
        """
        Args:
            memory_manage: MemoryManage 实例
            memory_types: 要检索的记忆类型列表（默认全部）
            limit: 每种类型最多返回条数
            user_id: 用户 ID 过滤
        """
        self.memory_manage = memory_manage
        self.memory_types = memory_types
        self.limit = limit
        self.user_id = user_id

    def fetch(self, query: str, max_tokens: int = 0, **kwargs) -> List[ContextItem]:
        if self.memory_manage is None:
            return []

        items = []

        # 确定要检索的记忆类型
        available = getattr(self.memory_manage, "memory_types", {})
        types_to_search = self.memory_types or list(available.keys())

        for mem_type in types_to_search:
            memory_obj = available.get(mem_type)
            if memory_obj is None:
                continue

            # 工作记忆：直接获取全部（不需要搜索）
            if mem_type == "working":
                if hasattr(memory_obj, "get_all_memories"):
                    memories = memory_obj.get_all_memories()
                    if self.limit > 0:
                        memories = memories[-self.limit:]
                else:
                    continue
            else:
                # 其他类型：语义搜索
                try:
                    search_kwargs = {"limit": self.limit}
                    if self.user_id:
                        search_kwargs["user_id"] = self.user_id
                    memories = memory_obj.search_memory(query, **search_kwargs)
                except Exception:
                    continue

            base_priority = _MEMORY_PRIORITY.get(mem_type, 0.5)

            for i, mem in enumerate(memories):
                content = mem.content if hasattr(mem, "content") else str(mem)
                content=f"Working Memory [id:{mem.id}]: {content}" if mem_type=="working" else f"{mem_type.capitalize()} Memory: {content}"
                metadata = {
                    "memory_type": mem_type,
                    "memory_id": getattr(mem, "id", ""),
                    "importance": getattr(mem, "importance", 0.5),
                }

                # 重要性越高优先级越高
                importance = getattr(mem, "importance", 0.5)
                priority = base_priority * (0.7 + 0.3 * importance)

                items.append(ContextItem(
                    content=content,
                    source="memory",
                    priority=min(priority, 1.0),
                    metadata=metadata,
                ))

        return items

    @property
    def source_name(self) -> str:
        return "memory"
