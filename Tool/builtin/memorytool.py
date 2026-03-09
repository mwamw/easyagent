from memory.V2.MemoryManage import MemoryManage
from Tool.BaseTool import Tool
from pydantic import BaseModel, Field
from typing import Type, Literal,Optional,Dict,Any
from datetime import datetime
class AddMemoryParam(BaseModel):
    content:str=Field(description="memory content")
    memory_type:Literal["working","episodic","semantic","perceptual"]=Field(description="memory type,working(工作记忆用来保存当前任务的关键上下文信息),episodic(事件记忆用来保存用户过去的经历和事件),semantic(语义记忆用来保存事实知识和概念),perceptual(感知记忆用来保存多模态信息)")
    importance:float=Field(description="memory importance:0-1")
    metadata:Optional[Dict[str,Any]]=Field(description="memory metadata",default={})
    modality:Optional[Literal["text","image","audio","video"]]=Field(description="memory modality,need to be set when memory_type is perceptual",default="text")
    file_path:Optional[str]=Field(description="memory file path,need to be set when memory_type is perceptual and modality is not text",default=None)
class RemoveMemoryParam(BaseModel):
    memory_id:str=Field(description="memory id")

class GetMemoryParam(BaseModel):
    memory_ids:list[str]=Field(description="memory ids")

class SearchMemoryParam(BaseModel):
    query:str=Field(description="search query")
    memory_types:Optional[list[str]]=Field(description="memory types to search (e.g., ['working', 'episodic','semantic','perceptual'])", default=None)
    limit:int=Field(description="limit", default=10)
    importance_threshold:Optional[float]=Field(description="importance threshold", default=None)


class UpdateMemoryParam(BaseModel):
    memory_id:str=Field(description="memory id")
    content:str=Field(description="new memory content")
    importance:Optional[float]=Field(description="new memory importance", default=None)
    metadata:Optional[Dict[str,Any]]=Field(description="new memory metadata", default=None)

class ForgetMemoryParam(BaseModel):
    strategy:Literal["time","importance","capacity"]=Field(description="forget strategy")
    threshold:float=Field(description="forget threshold", default=0.1)
    max_age_days:int=Field(description="max age days", default=30)

class ConsolidateMemoryParam(BaseModel):
    source_type:Literal["working","episodic","semantic","perceptual"]=Field(description="source memory type")
    target_type:Literal["working","episodic","semantic","perceptual"]=Field(description="target memory type")
    importance_threshold:float=Field(description="importance threshold", default=0.5)

class MemoryToolParam(BaseModel):
    action: Literal["add", "remove","get","search", "forget", "stats", "update", "consolidate", "clear"] = Field(
        description="action to take,add(添加记忆),remove(删除记忆),get(根据ids获取记忆完整内容),search(根据内容搜索记忆),forget(遗忘记忆),stats(获取统计信息),update(更新记忆),consolidate(整合记忆),clear(清空所有记忆)"
    )
    add_param:Optional[AddMemoryParam]=Field(description="add memory param", default=None)
    remove_param:Optional[RemoveMemoryParam]=Field(description="remove memory param", default=None)
    get_param:Optional[GetMemoryParam]=Field(description="get memory param", default=None)
    search_param:Optional[SearchMemoryParam]=Field(description="search memory param", default=None)
    update_param:Optional[UpdateMemoryParam]=Field(description="update memory param", default=None)
    forget_param:Optional[ForgetMemoryParam]=Field(description="forget memory param", default=None)
    consolidate_param:Optional[ConsolidateMemoryParam]=Field(description="consolidate memory param", default=None)

class MemoryTool(Tool):
    def __init__(self, memory_manage: MemoryManage):
        name = "memory_tool"
        self.memory_manage = memory_manage
        self.current_session_id = None
        self.conversation_count = 0
        description = f"""
            一个强大的多模态记忆管理工具，允许智能体（Agent）交互式地存储和检索各种上下文或长短期信息。支持的记忆类型有:{self.memory_manage.get_supported_type()}\n"
            "工具支持以下操作(action)：\n"
            "- add: 添加新的记忆（支持 working, episodic, semantic, perceptual 四种类型）。\n"
            "- remove: 根据特定的 memory_id 删除记忆。\n"
            "- get: 根据一组 memory_id 批量获取记忆的完整详细内容。\n"
            "- search: 根据文本内容语义搜索相关的记忆片段。\n"
            "- forget: 模拟人类遗忘机制，可基于容量限制、重要性或时间衰减来清理记忆。\n"
            "- update: 更新现有记忆的内容或重要性等属性。\n"
            "- consolidate: 将记忆从一种类型整合或沉淀到另一种类型（例如将高价值的从工作记忆整合到语义记忆）。\n"
            "- stats: 获取当前系统的记忆统计数据。\n"
            "- clear: 危险操作，清空系统的所有记忆。"
        """
        super().__init__(name, description, MemoryToolParam)

    def run(self, parameters:dict) -> str | dict | list:
        action = parameters.get("action")
        
        if action == "add":
            p = parameters.get("add_param") or {}
            return self._add_memory(
                content=p.get("content"),
                memory_type=p.get("memory_type", "working"),
                importance=p.get("importance", 0.5),
                metadata=p.get("metadata"),
                modality=p.get("modality"),
                file_path=p.get("file_path")
            )
        elif action == "remove":
            p = parameters.get("remove_param") or {}
            return self._remove_memory(memory_id=p.get("memory_id"))
        elif action == "get":
            p = parameters.get("get_param") or {}
            return self._get_memory(memory_ids=p.get("memory_ids", []))
        elif action == "search":
            p = parameters.get("search_param") or {}
            return self._search_memory(
                query=p.get("query"),
                memory_types=p.get("memory_types"),
                limit=p.get("limit", 10),
                importance_threshold=p.get("importance_threshold", 0.0)
            )
        elif action == "update":
            p = parameters.get("update_param") or {}
            return self._update_memory(
                memory_id=p.get("memory_id"),
                content=p.get("content"),
                importance=p.get("importance"),
                metadata=p.get("metadata")
            )
        elif action == "forget":
            p = parameters.get("forget_param") or {}
            return self.forget(
                strategy=p.get("strategy"),
                threshold=p.get("threshold", 0.1),
                max_age_days=p.get("max_age_days", 30)
            )
        elif action == "consolidate":
            p = parameters.get("consolidate_param") or {}
            return self._consolidate_memory(
                source_type=p.get("source_type"),
                target_type=p.get("target_type"),
                importance_threshold=p.get("importance_threshold", 0.5)
            )
        elif action == "stats":
            return self._get_stats()
        elif action == "clear":
            return self._clear()
        else:
            return f"Unknown action: {action}"
    
    def _add_memory(self,content:str="",memory_type:str="working",importance:float=0.5,metadata:Optional[dict]=None,modality:Optional[str]=None,file_path:Optional[str]=None)->str:
        try:
            
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata=metadata or {}
            if memory_type=="perceptual":
                modality=modality or self._infer_modality(file_path)
                metadata["raw_data"]=file_path
                metadata["modality"]=modality
            
            metadata.update({
                "session_id":self.current_session_id,
                "conversation_count":self.conversation_count,
                "timestamp":datetime.now().isoformat()
            })
            
            memory_id=self.memory_manage.add_memory(content,memory_type,importance,metadata)
            self.conversation_count+=1

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"
                
    def _infer_modality(self,file_path:Optional[str])->str:
        if not file_path:
            return "text"
        file_extension=file_path.split(".")[-1].lower()
        if file_extension in ["jpg","jpeg","png","gif","bmp","webp"]:
            return "image"
        elif file_extension in ["mp3","wav","aac","flac","ogg","m4a"]:
            return "audio"
        elif file_extension in ["mp4","avi","mov","mkv","flv","wmv"]:
            return "video"
        else:
            return "text"

    def _remove_memory(self,memory_id:str)->str:
        try:
            success=self.memory_manage.remove_memory(memory_id)
            return f"✅ 记忆已删除 (ID: {memory_id[:8]}...)" if success else "⚠️ 未找到要删除的记忆"
        except Exception as e:
            return f"❌ 删除记忆失败: {str(e)}"
    
    def _search_memory(self,query:str,memory_types:Optional[list[str]]=["working"],limit:int=10,
        importance_threshold:float=0.0):

        try:
            session_id=self.current_session_id

            results=self.memory_manage.search_memory(query,memory_types,limit,importance_threshold,None,session_id)


            if not results:
                return f"未找到和'{query}'相关的信息"

            formatted_results=[]
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")
            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.type, memory.type)

                content_preview = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
                formatted_results.append(
                    f"{i}. [{memory_type_label}] (memory_id:{memory.id}) content_preview:{content_preview} (重要性: {memory.importance:.2f})"
                )

            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"

    def _get_memory(self,memory_ids:list[str]):
        try:
            results=self.memory_manage.get_memories(memory_ids)

            formatted_results=[]
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")
            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆"
                }.get(memory.type, memory.type)

                content_preview = memory.content
                formatted_results.append(
                    f"{i}. [{memory_type_label}] (memory_id:{memory.id}) complete_content:{content_preview} (重要性: {memory.importance:.2f})"
                )
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"❌ 获取记忆失败: {str(e)}"
        
    def forget(self,strategy:str,threshold:float=0.1,max_age_days:int=30):
        try:
            num=self.memory_manage.forget_memory(strategy,threshold,max_age_days)

            return f"根据{strategy}遗忘了{num}条记忆"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {str(e)}"
    


    def _get_stats(self) -> str:
        """获取统计信息"""
        try:
            stats = self.memory_manage.get_memory_stats()

            stats_info = [
                f"📈 记忆系统统计",
                f"总记忆数: {stats['total_memories']}",
                f"启用的记忆类型: {', '.join(stats['enabled_types'])}",
                f"会话ID: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}"
            ]

            return "\n".join(stats_info)

        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"

    def _update_memory(self,memory_id:str,content:str,importance:Optional[float]=None,metadata:Optional[Dict[str,Any]]=None):
        try:
            success = self.memory_manage.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance or None,
                metadata=metadata or None
            )
            return "✅ 记忆已更新" if success else "⚠️ 未找到要更新的记忆"
        except Exception as e:
            return f"❌ 更新记忆失败: {str(e)}"

    
    def _consolidate_memory(self,source_type:str,target_type:str,importance_threshold:float=0.5):
        try:
            merged_count=self.memory_manage.merge_memories(source_type,target_type,importance_threshold)

            return f"Merged {merged_count} memories from {source_type} to {target_type}"
        except Exception as e:
            return f"❌ 整合记忆失败: {str(e)}"
            


    def _clear(self):
        try:
            self.memory_manage.clear_memories()

            return "✅ 记忆已清空"

        except Exception as e:
            return f"❌ 清空记忆失败: {str(e)}"
