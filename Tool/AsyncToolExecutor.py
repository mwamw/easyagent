import asyncio
import concurrent.futures
from typing import Any,List,Callable

from .ToolRegistry import ToolRegistry

class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self,tool_name:str,parameters:dict):
        loop=asyncio.get_event_loop()
        def _execute():
            return self.registry.executeTool(tool_name,parameters)
        result=await loop.run_in_executor(self.executor,_execute)
        return result
    
    async def execute_tools_parallel(self,tasks:List[dict[str,Any]])->List[str]:
        async_tasks=[]
        for task in tasks:
            async_tasks.append(self.execute_tool_async(task["tool_name"],task["parameters"]))
        results=await asyncio.gather(*async_tasks)
        return results
    
    def __del__(self):
        if hasattr(self,'executor'):
            self.executor.shutdown(wait=True)
