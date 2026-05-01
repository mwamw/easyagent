from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from memory.V2.MemoryManage import MemoryManage
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime
import logging

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
        return "无"
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
            f"当前可用记忆类型: {supported}。\n"
            "只在信息对后续轮次仍有价值时写入记忆。\n"
            "- `working` 用于当前任务约束、计划和中间结论。\n"
            "- `semantic` 用于长期稳定的事实、偏好和约定。\n"
            "- `episodic` 用于真实发生过的经历或历史事件。\n"
            "- `perceptual` 用于需要后续引用的多模态内容。"
        ),
        "search_memory_tool": (
            f"当前可用记忆类型: {supported}。\n"
            "只有在当前对话、上下文和最新工具结果不足时，再搜索记忆。\n"
            "优先指定最可能相关的记忆类型，避免无差别搜索所有类型。\n"
            "搜索结果是历史上下文，不保证仍然为真；回答前应与当前事实交叉验证。"
        ),
        "get_memory_tool": (
            "仅当你已经拿到明确的 memory_id，并且确实需要查看完整内容时使用。\n"
            "如果只是在回忆大致线索，优先使用 `search_memory_tool`。"
        ),
        "update_memory_tool": (
            "当旧记忆被用户纠正、计划发生变化或事实已失效时使用。\n"
            "更新时应保留真正仍然有效的信息，不要把临时噪音覆盖进长期记忆。"
        ),
        "remove_memory_tool": (
            "当用户明确要求忘记，或某条记忆已经错误、过时、重复、无意义时使用。\n"
            "删除前确认 memory_id 与目标内容匹配，避免误删。"
        ),
        "memory_maintenance_tool": (
            f"当前可用记忆类型: {supported}。\n"
            "`stats` 用于查看状态；`consolidate` 用于整合高价值记忆；`forget` 用于清理低价值记忆；`clear` 是危险操作。\n"
            "只有在任务完成、话题切换、记忆明显混乱或用户明确要求时，才做维护操作。"
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
class AddMemoryParam(BaseModel):
    content: str = Field(description="memory content")
    memory_type: Literal["working", "episodic", "semantic", "perceptual"] = Field(description="memory type,working(工作记忆用来保存当前任务的关键上下文信息),episodic(事件记忆用来保存用户过去的经历和事件),semantic(语义记忆用来保存事实知识和概念),perceptual(感知记忆用来保存多模态信息)")
    importance: float = Field(description="memory importance:0-1", default=0.5)
    metadata: Optional[Dict[str, Any]] = Field(description="memory metadata", default_factory=dict)
    modality: Optional[Literal["text", "image", "audio", "video"]] = Field(description="memory modality,need to be set when memory_type is perceptual", default="text")
    file_path: Optional[str] = Field(description="memory file path,need to be set when memory_type is perceptual and modality is not text", default=None)

class AddMemoryTool(Tool):
    def __init__(self, memory_manage: MemoryManage,current_session_id:Optional[str]=None):
        name = "add_memory_tool"
        self.memory_manage = memory_manage
        self.current_session_id = current_session_id
        self.conversation_count = 0
        description = f"添加新的记忆（当前支持: {_supported_types_text(self.memory_manage)}）。用于保存当前关键上下文、经历、事实知识或多模态数据。"
        super().__init__(
            name,
            description,
            AddMemoryParam,
            guidance="仅在信息对后续轮次仍有价值时写入记忆；不要把临时噪音或当前一步的显然内容写入长期记忆。",
            prompt=_build_memory_tool_prompt(name, self.memory_manage),
            source="builtin",
            tags=["memory", "write"],
        )
    def get_current_session_id(self):
        return self.current_session_id
    def run(self, parameters: dict) -> ToolResult:
        try:
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                "session_id": self.current_session_id,
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
    query: str = Field(description="search query 记忆搜索词")
    memory_types: Optional[list[str]] = Field(description="要搜索的记忆类型，例如 ['working', 'episodic']。如果不填写默认搜所有类型。如果确定需要搜索的类型，请指定", default=None)
    limit: int = Field(description="返回最多几条相关的记忆", default=10)
    importance_threshold: float = Field(description="只返回重要性大于该值的记忆，默认为 0.0", default=0.0)
    use_session_id:bool=Field(description="是否仅仅搜索当前会话的记忆,如果为空则不进行过滤,主要用来过滤episodic类型的记忆", default=False)
class SearchMemoryTool(Tool):
    def __init__(self, memory_manage: MemoryManage,current_session_id:Optional[str]=None):
        name = "search_memory_tool"
        self.memory_manage = memory_manage
        self.current_session_id=current_session_id
        description = "根据你提供的文本搜索词汇，在记忆库中语义搜索并返回相关的记忆片段。这是回忆过去知识和经历的最常用方法。"

        super().__init__(
            name,
            description,
            SearchMemoryParam,
            guidance="只有在当前上下文不足时才搜索记忆；优先限定最可能相关的记忆类型，并把结果当作历史线索而非绝对真相。",
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
                session_id=self.current_session_id
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
    memory_ids: list[str] = Field(description="要获取详情的记忆ID列表")

class GetMemoryTool(Tool):
    def __init__(self, memory_manage: MemoryManage):
        name = "get_memory_tool"
        self.memory_manage = memory_manage
        description = "根据一组指定的 memory_id 批量获取这些记忆的完整详细内容。只有当你知道具体的 ID 时才使用此工具。"
        super().__init__(
            name,
            description,
            GetMemoryParam,
            guidance="仅在你已经确定 memory_id 时才获取完整内容；如果只是回忆线索，先用 search_memory_tool。",
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
    memory_id: str = Field(description="要删除的记忆 ID")

class RemoveMemoryTool(Tool):
    def __init__(self, memory_manage: MemoryManage):
        name = "remove_memory_tool"
        self.memory_manage = memory_manage
        description = "根据具体的 memory_id 删除某一条记忆。"
        super().__init__(
            name,
            description,
            RemoveMemoryParam,
            guidance="只在用户明确要求忘记，或记忆已经错误、失效、重复时删除；删除前确认 memory_id 是否匹配。",
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
    memory_id: str = Field(description="要更新的记忆 ID")
    content: str = Field(description="新覆盖的记忆内容")
    importance: Optional[float] = Field(description="新的重要性评分", default=None)
    metadata: Optional[Dict[str, Any]] = Field(description="新的元数据", default=None)

class UpdateMemoryTool(Tool):
    def __init__(self, memory_manage: MemoryManage):
        name = "update_memory_tool"
        self.memory_manage = memory_manage
        description = "根据记忆 ID，更新修改已有记忆的内容、重要性或元数据。"
        super().__init__(
            name,
            description,
            UpdateMemoryParam,
            guidance="当用户纠正旧信息、偏好变化或原记忆失效时更新；避免把未经确认的新猜测写回旧记忆。",
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
        description="系统维护操作的类型。stats(获取系统状态), consolidate(跨类型转移整合高价值记忆), forget(模拟人类遗忘机制自动清理不重要特征的记忆), clear(格式化危险操作)"
    )
    # 对于 consolidate
    source_type: Optional[Literal["working", "episodic", "semantic", "perceptual"]] = Field(description="整合来源的记忆类型 (仅 action=consolidate 需要)", default=None)
    target_type: Optional[Literal["working", "episodic", "semantic", "perceptual"]] = Field(description="整合去往的记忆类型 (仅 action=consolidate 需要)", default=None)
    # 对于 forget 和 consolidate 的重要性阈值
    threshold: Optional[float] = Field(description="遗忘阈值 或 整合的重要性底线", default=0.5)
    # 对于 forget 
    strategy: Optional[Literal["time", "importance", "capacity"]] = Field(description="遗忘的策略类型 (仅 action=forget 需要)", default="importance")
    max_age_days: Optional[int] = Field(description="时间遗忘策略的天数 (仅 action=forget 需要)", default=30)
    memory_type: Optional[Literal["working", "episodic", "semantic", "perceptual"]] = Field(description="清空指定类型的记忆 (仅 action=clear 需要) 默认清除所有记忆", default=None)
class MemoryMaintenanceTool(Tool):
    def __init__(self, memory_manage: MemoryManage):
        name = "memory_maintenance_tool"
        self.memory_manage = memory_manage
        description = "提供对记忆系统的宏观维护。可以使用 stats 查询当前记忆的容量；使用 consolidate 将短期工作记忆合并到长期记忆；使用 forget 让系统自动清理无用记忆；使用 clear 清空指定类型的记忆或全部记忆。"
        super().__init__(
            name,
            description,
            MemoryMaintenanceParam,
            guidance="仅在任务结束、话题切换、记忆混乱或用户明确要求时做维护；`clear` 属于危险操作，不要轻易使用。",
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
def register_memory_tools(
    memory_manage: MemoryManage,
    registry: ToolRegistry,
    *,
    expose_in_deferred: bool | None = True,
):
    """
    一键向 ToolRegistry 里注册所有的 Memory 细分工具。
    """
    session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tools = [
        AddMemoryTool(memory_manage,current_session_id=session_id),
        SearchMemoryTool(memory_manage,current_session_id=session_id),
        GetMemoryTool(memory_manage),
        UpdateMemoryTool(memory_manage),
        RemoveMemoryTool(memory_manage),
        MemoryMaintenanceTool(memory_manage)
    ]
    for tool in tools:
        registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    logger.info(f"成功将 {len(tools)} 个 Memory 工具注册到系统。")
