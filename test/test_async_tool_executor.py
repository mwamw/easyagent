import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import time
from pydantic import BaseModel, Field

from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from Tool.AsyncToolExecutor import AsyncToolExecutor


# ==================== 测试用的参数模型 ====================

class AddParams(BaseModel):
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")


class SlowTaskParams(BaseModel):
    delay: float = Field(description="延迟秒数")
    value: str = Field(description="返回值")


class FailParams(BaseModel):
    message: str = Field(description="错误信息")


# ==================== 测试用的工具 ====================

class AddTool(Tool):
    """加法工具"""
    def run(self, params: dict):
        return params["a"] + params["b"]


class SlowTool(Tool):
    """模拟耗时操作的工具"""
    def run(self, params: dict):
        time.sleep(params["delay"])
        return params["value"]


class FailTool(Tool):
    """会抛出异常的工具"""
    def run(self, params: dict):
        raise ValueError(params["message"])


# ==================== Fixtures ====================

@pytest.fixture
def registry():
    """创建并注册工具的 ToolRegistry"""
    reg = ToolRegistry()
    reg.registerTool(AddTool("add", "两数相加", AddParams))
    reg.registerTool(SlowTool("slow_task", "模拟耗时任务", SlowTaskParams))
    reg.registerTool(FailTool("fail", "会失败的工具", FailParams))
    return reg


@pytest.fixture
def executor(registry):
    """创建 AsyncToolExecutor"""
    return AsyncToolExecutor(registry, max_workers=4)


# ==================== 测试用例 ====================

class TestAsyncToolExecutor:
    """AsyncToolExecutor 测试类"""

    @pytest.mark.asyncio
    async def test_execute_tool_async_success(self, executor):
        """测试单个工具异步执行成功"""
        result = await executor.execute_tool_async("add", {"a": 10, "b": 20})
        assert result == "30"

    @pytest.mark.asyncio
    async def test_execute_tool_async_not_found(self, executor):
        """测试执行不存在的工具"""
        with pytest.raises(ValueError, match="Tool .* not found"):
            await executor.execute_tool_async("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_params(self, executor):
        """测试传递无效参数"""
        with pytest.raises(ValueError, match="Invalid parameters"):
            await executor.execute_tool_async("add", {"a": "not_a_number", "b": 20})

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_basic(self, executor):
        """测试多个工具并行执行"""
        tasks = [
            {"tool_name": "add", "parameters": {"a": 1, "b": 2}},
            {"tool_name": "add", "parameters": {"a": 10, "b": 20}},
            {"tool_name": "add", "parameters": {"a": 100, "b": 200}},
        ]
        results = await executor.execute_tools_parallel(tasks)
        
        assert len(results) == 3
        assert results[0] == "3"
        assert results[1] == "30"
        assert results[2] == "300"

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_is_concurrent(self, executor):
        """测试并行执行确实是并发的，而非串行"""
        delay = 0.3
        tasks = [
            {"tool_name": "slow_task", "parameters": {"delay": delay, "value": "task1"}},
            {"tool_name": "slow_task", "parameters": {"delay": delay, "value": "task2"}},
            {"tool_name": "slow_task", "parameters": {"delay": delay, "value": "task3"}},
        ]
        
        start_time = time.time()
        results = await executor.execute_tools_parallel(tasks)
        elapsed_time = time.time() - start_time
        
        # 如果是并行执行，总时间应该接近 delay，而不是 3 * delay
        assert elapsed_time < delay * 2  # 留一些余量
        assert len(results) == 3
        assert set(results) == {"task1", "task2", "task3"}

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_empty_list(self, executor):
        """测试空任务列表"""
        results = await executor.execute_tools_parallel([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_single_task(self, executor):
        """测试只有一个任务的情况"""
        tasks = [{"tool_name": "add", "parameters": {"a": 5, "b": 5}}]
        results = await executor.execute_tools_parallel(tasks)
        
        assert len(results) == 1
        assert results[0] == "10"

    @pytest.mark.asyncio
    async def test_execute_tools_parallel_mixed_tools(self, executor):
        """测试混合执行不同类型的工具"""
        tasks = [
            {"tool_name": "add", "parameters": {"a": 1, "b": 1}},
            {"tool_name": "slow_task", "parameters": {"delay": 0.1, "value": "done"}},
        ]
        results = await executor.execute_tools_parallel(tasks)
        
        assert len(results) == 2
        assert results[0] == "2"
        assert results[1] == "done"


class TestAsyncToolExecutorLifecycle:
    """AsyncToolExecutor 生命周期测试"""

    def test_executor_initialization(self, registry):
        """测试执行器初始化"""
        executor = AsyncToolExecutor(registry, max_workers=2)
        
        assert executor.registry is registry
        assert executor.executor._max_workers == 2

    def test_executor_cleanup(self, registry):
        """测试执行器清理（__del__）"""
        executor = AsyncToolExecutor(registry, max_workers=2)
        thread_pool = executor.executor
        
        # 手动调用 __del__
        del executor
        
        # 线程池应该已经关闭
        assert thread_pool._shutdown


# ==================== 使用装饰器注册工具的测试 ====================

class TestAsyncToolExecutorWithDecorator:
    """测试与装饰器注册的工具配合使用"""

    @pytest.mark.asyncio
    async def test_decorator_registered_tool(self):
        """测试通过装饰器注册的工具"""
        registry = ToolRegistry()
        
        class MultiplyParams(BaseModel):
            x: int
            y: int
        
        @registry.tool("multiply", "两数相乘", MultiplyParams)
        def multiply(x: int, y: int):
            return x * y
        
        executor = AsyncToolExecutor(registry)
        result = await executor.execute_tool_async("multiply", {"x": 6, "y": 7})
        
        assert result == "42"


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
