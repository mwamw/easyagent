from __future__ import annotations

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Callable, Literal, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage

logger = logging.getLogger(__name__)

MEMORY_TYPE_LABELS = {
    "working": "工作记忆",
    "episodic": "情景记忆",
    "semantic": "语义记忆",
    "perceptual": "感知记忆",
}


def _supported_types_text(memory_manage: MemoryManage) -> str:
    supported = memory_manage.get_supported_type()
    if isinstance(supported, dict):
        supported = supported.keys()
    if not supported:
        return "none"
    return ", ".join(str(item) for item in supported)


def _serialize_memory(memory: Any) -> dict[str, Any]:
    return {
        "memory_id": getattr(memory, "id", ""),
        "memory_type": getattr(memory, "type", ""),
        "content": getattr(memory, "content", ""),
        "importance": getattr(memory, "importance", 0.0),
        "metadata": getattr(memory, "metadata", {}) or {},
    }


def _build_memory_tool_prompt(tool_name: str, memory_manage: MemoryManage) -> str:
    supported = _supported_types_text(memory_manage)
    prompts = {
        "add_memory_tool": (
            f"Currently enabled memory types: {supported}.\n"
            "Save information only when it will remain useful in later turns.\n"
            "- `working` is for current task constraints, plans, and intermediate conclusions.\n"
            "- `semantic` is for stable facts, preferences, and conventions.\n"
            "- `episodic` is for experiences and events that actually occurred.\n"
            "- `perceptual` is for multimodal content needed for later reference."
        ),
        "search_memory_tool": (
            f"Currently enabled memory types: {supported}.\n"
            "Search memory when the current conversation, context, and latest tool results are insufficient.\n"
            "Prefer the most relevant memory types instead of searching all types indiscriminately.\n"
            "Results are historical context, not necessarily current truth. Verify them against current evidence."
        ),
        "get_memory_tool": (
            "Use only when you have known memory IDs and need their complete contents.\n"
            "If you only have a clue, use `search_memory_tool` first."
        ),
        "update_memory_tool": (
            "Use when the user corrects a memory, plans change, or facts become invalid.\n"
            "Preserve information that remains valid; do not replace long-term memory with temporary noise."
        ),
        "remove_memory_tool": (
            "Use when the user asks to forget something, or a memory is wrong, stale, duplicated, or meaningless.\n"
            "Before deleting, verify that the memory ID matches the intended content."
        ),
        "memory_maintenance_tool": (
            f"Currently enabled memory types: {supported}.\n"
            "Use `stats` to inspect state, `consolidate` to integrate valuable memories, and `forget` to clean up low-value memories. `clear` is destructive.\n"
            "Perform maintenance when a task ends, the topic changes, memory becomes disorganized, or the user explicitly requests it."
        ),
    }
    return prompts.get(tool_name, "")

# ==========================================
# 辅助函数
# ==========================================
def _infer_modality(file_path: Optional[str]) -> str:
    if not file_path:
        return "text"
    file_extension = file_path.split(".")[-1].lower()
    if file_extension in ["jpg", "jpeg", "png", "gif", "bmp", "webp"]:
        return "image"
    elif file_extension in ["mp3", "wav", "aac", "flac", "ogg", "m4a"]:
        return "audio"
    elif file_extension in ["mp4", "avi", "mov", "mkv", "flv", "wmv"]:
        return "video"
    else:
        return "text"

# ==========================================
# 1. AddMemoryTool
# ==========================================
class _MemoryTool(Tool):
    async def arun(self, parameters: dict):
        return await asyncio.to_thread(self.run, parameters)


class AddMemoryParam(BaseModel):
    content: str = Field(description="memory content")
    memory_type: Literal["working", "episodic", "semantic", "perceptual"] = Field(description="Memory type: working for task context, episodic for past events, semantic for stable facts, perceptual for multimodal information.")
    importance: float = Field(description="memory importance:0-1", default=0.5)
    metadata: Optional[Dict[str, Any]] = Field(description="memory metadata", default_factory=dict)
    modality: Optional[Literal["text", "image", "audio", "video"]] = Field(description="memory modality,need to be set when memory_type is perceptual", default="text")
    file_path: Optional[str] = Field(description="memory file path,need to be set when memory_type is perceptual and modality is not text", default=None)

class AddMemoryTool(_MemoryTool):
    def __init__(self, memory_manage: MemoryManage,current_session_id:Optional[str]=None, *, session_id_provider=None):
        name = "add_memory_tool"
        self.memory_manage = memory_manage
        self.current_session_id = current_session_id
        self.session_id_provider = session_id_provider
        self.conversation_count = 0
        description = f"Add memory (enabled types: {_supported_types_text(self.memory_manage)}) to preserve task context, experiences, facts, or multimodal data."
        super().__init__(
            name,
            description,
            AddMemoryParam,
            guidance="Write memory only when it will be useful in later turns; do not store temporary noise as long-term memory.",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            source="builtin",
            tags=["memory", "write"],
        )
    def get_current_session_id(self):
        return self.session_id_provider() if self.session_id_provider else self.current_session_id
    def run(self, parameters: dict) -> ToolResult:
        try:
            session_id = self.get_current_session_id()
            content = parameters.get("content","")
            memory_type = parameters.get("memory_type", "working")
            importance = parameters.get("importance", 0.5)
            metadata = parameters.get("metadata") or {}
            modality = parameters.get("modality")
            file_path = parameters.get("file_path")

            if memory_type == "perceptual":
                modality = modality or _infer_modality(file_path)
                metadata["raw_data"] = file_path
                metadata["modality"] = modality
            
            metadata.update({
                "session_id": session_id,
                "conversation_count": self.conversation_count,
                "timestamp": datetime.now().isoformat()
            })
            
            memory_id = self.memory_manage.add_memory(content, memory_type, importance, metadata)
            self.conversation_count += 1

            return ToolResult.success(
                f"✅ 记忆已添加 (ID: {memory_id[:8]}...)",
                structured_data={
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "importance": importance,
                    "session_id": self.current_session_id,
                },
                metadata={"memory_tool": self.name, "memory_type": memory_type},
            )
        except Exception as e:
            return ToolResult.error(f"❌ 添加记忆失败: {str(e)}", error_type="memory_add_failed")

# ==========================================
# 2. SearchMemoryTool
# ==========================================
class SearchMemoryParam(BaseModel):
    query: str = Field(description="Memory search query.")
    memory_types: Optional[list[str]] = Field(description="Types to search, e.g. ['working', 'episodic']. Defaults to all enabled types; narrow the types when known.", default=None)
    limit: int = Field(description="Maximum number of relevant memories to return.", default=10)
    importance_threshold: float = Field(description="Minimum importance threshold; defaults to 0.0.", default=0.0)
    use_session_id:bool=Field(description="Restrict results to the current session, primarily for episodic memory. Defaults to no session filter.", default=False)
class SearchMemoryTool(_MemoryTool):
    def __init__(self, memory_manage: MemoryManage,current_session_id:Optional[str]=None, *, session_id_provider=None):
        name = "search_memory_tool"
        self.memory_manage = memory_manage
        self.current_session_id=current_session_id
        self.session_id_provider = session_id_provider
        description = "Search memory semantically with a text query and return relevant excerpts of past knowledge and experiences."

        super().__init__(
            name,
            description,
            SearchMemoryParam,
            guidance="Search when current context is insufficient. Narrow to relevant memory types and treat results as historical clues, not unquestionable facts.",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            read_only=True,
            source="builtin",
            tags=["memory", "search"],
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            query = parameters.get("query","")
            memory_types = parameters.get("memory_types")
            limit = parameters.get("limit", 10)
            importance_threshold = parameters.get("importance_threshold", 0.0)
            use_session_id=parameters.get("use_session_id",False)
            session_id=None
            if use_session_id:
                session_id = self.session_id_provider() if self.session_id_provider else self.current_session_id
                if not session_id:
                    raise ValueError("Current session scope is unavailable")
            results = self.memory_manage.search_memory(query, memory_types, limit, importance_threshold,session_id=session_id)

            if not results:
                return ToolResult.success(
                    f"未找到和 '{query}' 相关的信息",
                    structured_data=[],
                    metadata={"query": query, "memory_types": memory_types or []},
                )

            formatted_results = [f"🔍 找到 {len(results)} 条相关记忆:"]
            serialized_results: list[dict[str, Any]] = []
            for i, memory in enumerate(results, 1):
                memory_type_label = MEMORY_TYPE_LABELS.get(memory.type, memory.type)
                if use_session_id and memory.metadata.get("session_id") != session_id:
                    continue
                
                content_preview = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
                formatted_results.append(
                    f"{i}. [{memory_type_label}] (memory_id:{memory.id}) content_preview:{content_preview} (重要性: {memory.importance:.2f})"
                )
                serialized_results.append(_serialize_memory(memory))
            return ToolResult.success(
                "\n".join(formatted_results),
                structured_data=serialized_results,
                metadata={
                    "query": query,
                    "memory_types": memory_types or [],
                    "use_session_id": use_session_id,
                },
            )
        except Exception as e:
            return ToolResult.error(f"❌ 搜索记忆失败: {str(e)}", error_type="memory_search_failed")

# ==========================================
# 3. GetMemoryTool
# ==========================================
class GetMemoryParam(BaseModel):
    memory_ids: list[str] = Field(description="Memory IDs whose complete details should be retrieved.")

class GetMemoryTool(_MemoryTool):
    def __init__(self, memory_manage: MemoryManage):
        name = "get_memory_tool"
        self.memory_manage = memory_manage
        description = "Retrieve complete details for known memory IDs. Use only when the IDs are already known."
        super().__init__(
            name,
            description,
            GetMemoryParam,
            guidance="Fetch full content only for known memory IDs. Use search_memory_tool first when you only have a clue.",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            read_only=True,
            source="builtin",
            tags=["memory", "read"],
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            memory_ids = parameters.get("memory_ids", [])
            results = self.memory_manage.get_memories(memory_ids)

            if not results:
                return ToolResult.success("没有找到指定的记忆", structured_data=[])

            formatted_results = [f"🔍 获取到 {len(results)} 条记录的完整内容:"]
            for i, memory in enumerate(results, 1):
                memory_type_label = MEMORY_TYPE_LABELS.get(memory.type, memory.type)

                formatted_results.append(
                    f"{i}. [{memory_type_label}] (memory_id:{memory.id})\n完整内容: {memory.content}\n(重要性: {memory.importance:.2f})\n---"
                )
            return ToolResult.success(
                "\n".join(formatted_results),
                structured_data=[_serialize_memory(memory) for memory in results],
                metadata={"memory_ids": memory_ids},
            )
        except Exception as e:
            return ToolResult.error(f"❌ 获取记忆失败: {str(e)}", error_type="memory_get_failed")

# ==========================================
# 4. RemoveMemoryTool
# ==========================================
class RemoveMemoryParam(BaseModel):
    memory_id: str = Field(description="ID of the memory to remove.")

class RemoveMemoryTool(_MemoryTool):
    def __init__(self, memory_manage: MemoryManage):
        name = "remove_memory_tool"
        self.memory_manage = memory_manage
        description = "Remove a memory by its ID."
        super().__init__(
            name,
            description,
            RemoveMemoryParam,
            guidance="Remove memories only when the user asks to forget them or they are incorrect, stale, or duplicated. Verify the memory ID first.",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            destructive=True,
            source="builtin",
            tags=["memory", "delete"],
        )

    def run(self, parameters: dict) -> ToolResult:
        memory_id = parameters.get("memory_id","")
        try:
            success = self.memory_manage.remove_memory(memory_id)
            if success:
                return ToolResult.success(
                    f"✅ 记忆已删除 (ID: {memory_id[:8]}...)",
                    structured_data={"memory_id": memory_id, "removed": True},
                )
            return ToolResult.success("⚠️ 未找到要删除的记忆", structured_data={"memory_id": memory_id, "removed": False})
        except Exception as e:
            return ToolResult.error(f"❌ 删除记忆失败: {str(e)}", error_type="memory_remove_failed")

# ==========================================
# 5. UpdateMemoryTool
# ==========================================
class UpdateMemoryParam(BaseModel):
    memory_id: str = Field(description="ID of the memory to update.")
    content: str = Field(description="Replacement memory content.")
    importance: Optional[float] = Field(description="New importance score.", default=None)
    metadata: Optional[Dict[str, Any]] = Field(description="New metadata.", default=None)

class UpdateMemoryTool(_MemoryTool):
    def __init__(self, memory_manage: MemoryManage):
        name = "update_memory_tool"
        self.memory_manage = memory_manage
        description = "Update memory content, importance, or metadata by ID."
        super().__init__(
            name,
            description,
            UpdateMemoryParam,
            guidance="Update when the user corrects information or facts become invalid; do not overwrite memories with unverified guesses.",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            source="builtin",
            tags=["memory", "update"],
        )

    def run(self, parameters: dict) -> ToolResult:
        memory_id = parameters.get("memory_id","")
        content = parameters.get("content","")
        importance = parameters.get("importance",0.5)
        metadata = parameters.get("metadata") or {}
        try:
            success = self.memory_manage.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                metadata=metadata
            )
            if success:
                return ToolResult.success(
                    "✅ 记忆已更新",
                    structured_data={"memory_id": memory_id, "updated": True},
                )
            return ToolResult.success("⚠️ 未找到要更新的记忆", structured_data={"memory_id": memory_id, "updated": False})
        except Exception as e:
            return ToolResult.error(f"❌ 更新记忆失败: {str(e)}", error_type="memory_update_failed")

# ==========================================
# 6. MemoryMaintenanceTool (Stats, Consolidate, Forget, Clear)
# ==========================================
class MemoryMaintenanceParam(BaseModel):
    action: Literal["stats", "consolidate", "forget", "clear"] = Field(
        description="Maintenance action: stats inspects state, consolidate transfers valuable memories, forget removes low-value memories, and clear destructively erases memories."
    )
    # 对于 consolidate
    source_type: Optional[Literal["working", "episodic", "semantic", "perceptual"]] = Field(description="Source memory type; required for consolidate.", default=None)
    target_type: Optional[Literal["working", "episodic", "semantic", "perceptual"]] = Field(description="Target memory type; required for consolidate.", default=None)
    # 对于 forget 和 consolidate 的重要性阈值
    threshold: Optional[float] = Field(description="Threshold for forgetting or minimum importance for consolidation.", default=0.5)
    # 对于 forget 
    strategy: Optional[Literal["time", "importance", "capacity"]] = Field(description="Forgetting strategy; used with forget.", default="importance")
    max_age_days: Optional[int] = Field(description="Maximum age in days for time-based forgetting.", default=30)
    memory_type: Optional[Literal["working", "episodic", "semantic", "perceptual"]] = Field(description="Memory type to clear; omitting this with clear erases all types.", default=None)
class MemoryMaintenanceTool(_MemoryTool):
    def __init__(self, memory_manage: MemoryManage):
        name = "memory_maintenance_tool"
        self.memory_manage = memory_manage
        description = "Maintain memory: stats inspects state, consolidate transfers valuable memories, forget removes low-value memories, and clear erases one or all memory types."
        super().__init__(
            name,
            description,
            MemoryMaintenanceParam,
            guidance="Maintain memory when a task ends, the topic changes, memory becomes disorganized, or the user requests it. The clear action is destructive.",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            destructive=True,
            source="builtin",
            tags=["memory", "maintenance"],
        )

    def run(self, parameters: dict) -> ToolResult:
        action = parameters.get("action")
        try:
            if action == "stats":
                stats = self.memory_manage.get_memory_stats()
                stats_info = [
                    f"📈 记忆系统统计",
                    f"总记忆数: {stats['total_memories']}",
                    f"启用的记忆类型: {', '.join(stats['enabled_types'])}"
                ]
                return ToolResult.success("\n".join(stats_info), structured_data=stats)
            elif action == "consolidate":
                source = parameters.get("source_type")
                target = parameters.get("target_type")
                threshold = parameters.get("threshold", 0.5)
                if not source or not target:
                    return ToolResult.error("❌ consolidate 操作需要明确 source_type 和 target_type", error_type="invalid_parameters")
                merged_count = self.memory_manage.merge_memories(source, target, threshold)
                return ToolResult.success(
                    f"✅ 整合完成，共从 {source} 转移 {merged_count} 条高价值记忆至 {target}",
                    structured_data={"action": action, "source_type": source, "target_type": target, "count": merged_count},
                )
            elif action == "forget":
                strategy = parameters.get("strategy", "importance")
                threshold = parameters.get("threshold", 0.1)
                max_age_days = parameters.get("max_age_days", 30)
                num = self.memory_manage.forget_memory(strategy, threshold, max_age_days)
                return ToolResult.success(
                    f"✅ 根据 {strategy} 策略触发了遗忘，共清理了 {num} 条记忆",
                    structured_data={"action": action, "strategy": strategy, "count": num},
                )
            elif action == "clear":
                memory_type = parameters.get("memory_type")
                self.memory_manage.clear_memories(memory_type)
                return ToolResult.success(
                    "✅ 危险操作执行完毕：所有记忆都已被清空",
                    structured_data={"action": action, "memory_type": memory_type or "all"},
                )
            else:
                return ToolResult.error(f"❌ 未知的维护操作: {action}", error_type="invalid_parameters")
        except Exception as e:
            return ToolResult.error(f"❌ 维护操作执行失败: {str(e)}", error_type="memory_maintenance_failed")

# ==========================================
# 便利注册接口
# ==========================================
def build_memory_tools(memory_manage, *, session_id_provider: Callable[[], str] | None = None):
    return [
        AddMemoryTool(memory_manage, session_id_provider=session_id_provider),
        SearchMemoryTool(memory_manage, session_id_provider=session_id_provider),
        GetMemoryTool(memory_manage),
        UpdateMemoryTool(memory_manage),
        RemoveMemoryTool(memory_manage),
        MemoryMaintenanceTool(memory_manage),
    ]


def register_memory_tools(
    memory_manage: MemoryManage,
    registry: ToolRegistry,
    *,
    expose_in_deferred: bool | None = True,
    session_id_provider: Callable[[], str] | None = None,
):
    """
    一键向 ToolRegistry 里注册所有的 Memory 细分工具。
    """
    tools = build_memory_tools(memory_manage, session_id_provider=session_id_provider)
    for tool in tools:
        registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    logger.info(f"成功将 {len(tools)} 个 Memory 工具注册到系统。")
