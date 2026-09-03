import asyncio
import concurrent.futures
from typing import Any,List

from .BaseTool import ToolResult
from .ToolRegistry import ToolRegistry

class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self,tool_name:str,parameters:dict):
        loop = asyncio.get_running_loop()
        def _execute():
            return self.registry.execute_tool(tool_name,parameters)
        result=await loop.run_in_executor(self.executor,_execute)
        return result

    async def execute_tool_result_async(self, tool_name: str, parameters: dict) -> ToolResult:
        loop = asyncio.get_running_loop()

        def _execute():
            return self.registry.execute_tool_result(tool_name, parameters)

        return await loop.run_in_executor(self.executor, _execute)
    
    async def execute_tools_parallel(self,tasks:List[dict[str,Any]])->List[str]:
        async_tasks=[]
        for task in tasks:
            async_tasks.append(self.execute_tool_async(task["tool_name"],task["parameters"]))
        results=await asyncio.gather(*async_tasks)
        return results

    async def execute_tool_results_parallel(self, tasks: List[dict[str, Any]]) -> List[ToolResult]:
        async_tasks = []
        for task in tasks:
            async_tasks.append(
                self.execute_tool_result_async(task["tool_name"], task["parameters"])
            )
        return await asyncio.gather(*async_tasks)
    
    def __del__(self):
        if hasattr(self,'executor'):
            self.executor.shutdown(wait=True)
