from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from memory.V2.MemoryManage import MemoryManage
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

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
        description = f"添加新的记忆（当前支持 {self.memory_manage.get_supported_type()} 四种类型）。用于保存当前关键上下文、经历、事实知识或多模态数据。"
        super().__init__(name, description, AddMemoryParam)
    def get_current_session_id(self):
        return self.current_session_id
    def run(self, parameters: dict) -> str:
        try:
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            content = parameters.get("content")
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

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"

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
        
        super().__init__(name, description, SearchMemoryParam)

    def run(self, parameters: dict) -> str:
        try:
            query = parameters.get("query")
            memory_types = parameters.get("memory_types")
            limit = parameters.get("limit", 10)
            importance_threshold = parameters.get("importance_threshold", 0.0)
            use_session_id=parameters.get("use_session_id",False)
            session_id=None
            if use_session_id:
                session_id=self.current_session_id
            results = self.memory_manage.search_memory(query, memory_types, limit, importance_threshold, session_id, None)

            if not results:
                return f"未找到和 '{query}' 相关的信息"

            formatted_results = [f"🔍 找到 {len(results)} 条相关记忆:"]
            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.type, memory.type)
                if use_session_id and memory.metadata.get("session_id") != session_id:
                    continue
                
                content_preview = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
                formatted_results.append(
                    f"{i}. [{memory_type_label}] (memory_id:{memory.id}) content_preview:{content_preview} (重要性: {memory.importance:.2f})"
                )
            return "\n".join(formatted_results)
        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"

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
        super().__init__(name, description, GetMemoryParam)

    def run(self, parameters: dict) -> str:
        try:
            memory_ids = parameters.get("memory_ids", [])
            results = self.memory_manage.get_memories(memory_ids)

            if not results:
                return "没有找到指定的记忆"

            formatted_results = [f"🔍 获取到 {len(results)} 条记录的完整内容:"]
            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.type, memory.type)

                formatted_results.append(
                    f"{i}. [{memory_type_label}] (memory_id:{memory.id})\n完整内容: {memory.content}\n(重要性: {memory.importance:.2f})\n---"
                )
            return "\n".join(formatted_results)
        except Exception as e:
            return f"❌ 获取记忆失败: {str(e)}"

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
        super().__init__(name, description, RemoveMemoryParam)

    def run(self, parameters: dict) -> str:
        memory_id = parameters.get("memory_id")
        try:
            success = self.memory_manage.remove_memory(memory_id)
            return f"✅ 记忆已删除 (ID: {memory_id[:8]}...)" if success else "⚠️ 未找到要删除的记忆"
        except Exception as e:
            return f"❌ 删除记忆失败: {str(e)}"

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
        super().__init__(name, description, UpdateMemoryParam)

    def run(self, parameters: dict) -> str:
        memory_id = parameters.get("memory_id")
        content = parameters.get("content")
        importance = parameters.get("importance")
        metadata = parameters.get("metadata")
        try:
            success = self.memory_manage.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                metadata=metadata
            )
            return "✅ 记忆已更新" if success else "⚠️ 未找到要更新的记忆"
        except Exception as e:
            return f"❌ 更新记忆失败: {str(e)}"

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
        super().__init__(name, description, MemoryMaintenanceParam)

    def run(self, parameters: dict) -> str:
        action = parameters.get("action")
        try:
            if action == "stats":
                stats = self.memory_manage.get_memory_stats()
                stats_info = [
                    f"📈 记忆系统统计",
                    f"总记忆数: {stats['total_memories']}",
                    f"启用的记忆类型: {', '.join(stats['enabled_types'])}"
                ]
                return "\n".join(stats_info)
            elif action == "consolidate":
                source = parameters.get("source_type")
                target = parameters.get("target_type")
                threshold = parameters.get("threshold", 0.5)
                if not source or not target:
                    return "❌ consolidate 操作需要明确 source_type 和 target_type"
                merged_count = self.memory_manage.merge_memories(source, target, threshold)
                return f"✅ 整合完成，共从 {source} 转移 {merged_count} 条高价值记忆至 {target}"
            elif action == "forget":
                strategy = parameters.get("strategy", "importance")
                threshold = parameters.get("threshold", 0.1)
                max_age_days = parameters.get("max_age_days", 30)
                num = self.memory_manage.forget_memory(strategy, threshold, max_age_days)
                return f"✅ 根据 {strategy} 策略触发了遗忘，共清理了 {num} 条记忆"
            elif action == "clear":
                memory_type = parameters.get("memory_type")
                self.memory_manage.clear_memories(memory_type)
                return "✅ 危险操作执行完毕：所有记忆都已被清空"
            else:
                return f"❌ 未知的维护操作: {action}"
        except Exception as e:
            return f"❌ 维护操作执行失败: {str(e)}"

# ==========================================
# 便利注册接口
# ==========================================
def register_memory_tools(memory_manage: MemoryManage, registry: ToolRegistry):
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
        registry.registerTool(tool)
    logger.info(f"成功将 {len(tools)} 个 Memory 工具注册到系统。")
