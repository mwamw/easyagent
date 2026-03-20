"""
MCP工具集成模块，用于EasyAgent。

该模块将MCP远程工具桥接到本地EasyAgent工具实例中。
"""

from __future__ import annotations
import asyncio
import threading
from typing import Any, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel, Field, create_model

from ..BaseTool import Tool
from ..ToolRegistry import ToolRegistry
from  mcp import MCPClient


class MCPClientProtocol(Protocol):
    """MCP客户端协议定义，定义了MCP客户端应该实现的方法"""
    def is_connected(self) -> bool:
        """检查是否已连接"""
        ...

    async def connect(self) -> None:
        """建立连接"""
        ...

    async def disconnect(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """断开连接"""
        ...

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有远程工具"""
        ...

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """调用远程工具"""
        ...


def _json_type_to_python_type(type_name: Optional[str]) -> Any:
    """
    将JSON Schema类型转换为Python类型。
    
    Args:
        type_name: JSON类型名称字符串
        
    Returns:
        对应的Python类型
    """
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(type_name or "", Any)


def _sanitize_model_name(tool_name: str) -> str:
    """
    清理工具名称，使其适合作为Pydantic模型名。
    将非字母数字字符替换为下划线。
    
    Args:
        tool_name: 原始工具名称
        
    Returns:
        清理后的安全名称
    """
    normalized = []
    for ch in tool_name:
        if ch.isalnum() or ch == "_":
            normalized.append(ch)
        else:
            normalized.append("_")
    safe = "".join(normalized).strip("_")
    return safe or "mcp_tool"


def _build_pydantic_model_from_schema(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    根据JSON Schema构建Pydantic模型。
    
    Args:
        tool_name: MCP工具名称
        schema: JSON Schema字典
        
    Returns:
        动态生成的Pydantic模型类
    """
    # 从schema中提取properties和required字段
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

    # 如果没有属性，返回空参数模型
    if not properties:
        class EmptyParams(BaseModel):
            pass
        return EmptyParams

    # 构建模型字段字典
    fields: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            field_schema = {}

        # 将JSON类型转换为Python类型
        py_type = _json_type_to_python_type(field_schema.get("type"))
        description = field_schema.get("description", "")

        # 根据是否必需字段，构建不同的字段定义
        if field_name in required:
            # 必需字段
            fields[field_name] = (py_type, Field(description=description))
        else:
            # 可选字段，使用default值
            default = field_schema.get("default", None)
            fields[field_name] = (
                Optional[py_type],
                Field(default=default, description=description),
            )

    # 动态创建模型
    model_name = f"MCP_{_sanitize_model_name(tool_name)}_Params"
    return create_model(model_name, **fields)


def _run_coroutine_sync(coro):
    """
    在同步环境中运行异步协程。
    
    如果已在事件循环中，则创建新线程运行；否则直接运行。
    
    Args:
        coro: 要运行的协程对象
        
    Returns:
        协程执行结果
        
    Raises:
        协程执行过程中的异常
    """
    try:
        # 检查是否已有运行的事件循环
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有事件循环，直接运行协程
        return asyncio.run(coro)

    # 已有事件循环，创建新线程避免阻塞
    result_holder: Dict[str, Any] = {"value": None, "error": None}

    def _runner() -> None:
        """在新线程中创建新的事件循环并运行协程"""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            loop.close()

    # 启动守护线程执行
    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    # 检查是否有异常并抛出
    if result_holder["error"] is not None:
        raise result_holder["error"]

    return result_holder["value"]


class MCPWrappedTool(Tool):
    """
    将单个MCP远程工具包装为EasyAgent工具。
    充当适配器，在EasyAgent中统一调用MCP远程工具。
    """

    def __init__(
        self,
        manager: "MCPToolManager",
        tool_info: Dict[str, Any],
        prefix: str = "",
    ):
        """
        初始化被包装的MCP工具。
        
        Args:
            manager: MCPToolManager实例，用于执行远程工具调用
            tool_info: MCP工具描述字典（包含name、description、input_schema等）
            prefix: 工具名前缀
        """
        self.manager = manager
        self.tool_info = tool_info
        self.mcp_tool_name = tool_info.get("name", "unknown")

        # 构建最终的工具名称
        tool_name = f"{prefix}{self.mcp_tool_name}" if prefix else self.mcp_tool_name
        description = tool_info.get("description") or f"MCP tool: {self.mcp_tool_name}"
        input_schema = tool_info.get("input_schema") or {}
        
        # 从schema构建参数模型
        parameters = _build_pydantic_model_from_schema(self.mcp_tool_name, input_schema)

        # 调用父类初始化
        super().__init__(name=tool_name, description=description, parameters=parameters)

    def run(self, parameters: dict):
        """
        执行远程MCP工具。
        
        Args:
            parameters: 工具参数字典
            
        Returns:
            工具执行结果，转换为字符串
        """
        result = self.manager.execute_tool(self.mcp_tool_name, parameters)
        # 统一转换为字符串格式返回
        if isinstance(result, (dict, list)):
            return str(result)
        if result is None:
            return ""
        return str(result)

class MCPToolManager:
    """
    管理MCP连接和远程工具注册。
    负责连接MCP服务器、列出远程工具、执行工具调用，并将其注册到ToolRegistry。
    """

    def __init__(
        self,
        server_source: Any,
        server_args: Optional[List[str]] = None,
        transport_type: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        tool_prefix: str = "",
        auto_connect: bool = True,
        client: Optional[MCPClientProtocol] = None,
        **transport_kwargs: Any,
    ):
        """
        初始化MCPToolManager。
        
        Args:
            server_source: MCP服务器源标识（如命令或文件路径）
            server_args: 启动服务器的命令行参数
            transport_type: 传输类型（如'stdio'、'sse'等）
            env: 启动服务器的环境变量
            tool_prefix: 注册工具时的名称前缀
            auto_connect: 是否在需要时自动连接
            client: 自定义MCPClient实例，如果为None则创建默认实例
            **transport_kwargs: 传输层的额外参数
        """
        self.tool_prefix = tool_prefix
        self.auto_connect = auto_connect
        self._wrapped_tools: List[MCPWrappedTool] = []

        # 使用提供的客户端或创建新客户端
        self.client: MCPClientProtocol = client or MCPClient(
            server_source=server_source,
            server_args=server_args,
            transport_type=transport_type,
            env=env,
            **transport_kwargs,
        )

    def connect(self) -> None:
        """建立与MCP服务器的连接"""
        if self.client.is_connected():
            return
        _run_coroutine_sync(self.client.connect())

    def close(self) -> None:
        """关闭与MCP服务器的连接"""
        if not self.client.is_connected():
            return
        _run_coroutine_sync(self.client.disconnect())

    def ensure_connected(self) -> None:
        """
        确保已连接到MCP服务器。
        如果未连接且auto_connect为True，则自动连接。
        """
        if self.client.is_connected():
            return
        if not self.auto_connect:
            raise RuntimeError("MCP client is not connected.")
        self.connect()

    def list_remote_tools(self) -> List[Dict[str, Any]]:
        """
        获取MCP服务器上所有可用的远程工具列表。
        
        Returns:
            工具描述字典列表
        """
        return _run_coroutine_sync(self._list_remote_tools_async())

    async def _list_remote_tools_async(self) -> List[Dict[str, Any]]:
        """在单一协程上下文中完成连接与 list_tools，避免跨事件循环问题。"""
        if not self.auto_connect and not self.client.is_connected():
            raise RuntimeError("MCP client is not connected.")

        # auto_connect 模式下，每次调用使用独立连接生命周期，避免会话跨 loop 失效。
        if self.auto_connect:
            await self.client.connect()
            try:
                return await self.client.list_tools()
            finally:
                await self.client.disconnect()

        return await self.client.list_tools()

    async def alist_remote_tools(self) -> List[Dict[str, Any]]:
        """异步版本：供 async 上下文直接调用。"""
        return await self._list_remote_tools_async()

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行指定的远程MCP工具。
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        return _run_coroutine_sync(self._execute_tool_async(tool_name, arguments))

    async def _execute_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """在单一协程上下文中完成连接与 call_tool，避免跨事件循环问题。"""
        if not self.auto_connect and not self.client.is_connected():
            raise RuntimeError("MCP client is not connected.")

        if self.auto_connect:
            await self.client.connect()
            try:
                return await self.client.call_tool(tool_name, arguments)
            finally:
                await self.client.disconnect()

        return await self.client.call_tool(tool_name, arguments)

    async def aexecute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """异步版本：供 async 上下文直接调用。"""
        return await self._execute_tool_async(tool_name, arguments)

    def register_to_registry(self, registry: ToolRegistry) -> List[MCPWrappedTool]:
        """
        将所有远程MCP工具注册到ToolRegistry。
        
        Args:
            registry: ToolRegistry实例
            
        Returns:
            已注册的被包装工具列表
        """
        # 获取远程工具列表
        remote_tools = self.list_remote_tools()
        wrapped: List[MCPWrappedTool] = []

        # 为每个远程工具创建包装器并注册
        for tool_info in remote_tools:
            tool = MCPWrappedTool(
                manager=self,
                tool_info=tool_info,
                prefix=self.tool_prefix,
            )
            registry.registerTool(tool)
            wrapped.append(tool)

        # 保存已注册的包装工具
        self._wrapped_tools = wrapped
        return wrapped

    def get_wrapped_tools(self) -> List[MCPWrappedTool]:
        """获取已注册的被包装工具列表副本"""
        return list(self._wrapped_tools)


def register_mcp_tools(
    registry: ToolRegistry,
    server_source: Any,
    server_args: Optional[List[str]] = None,
    transport_type: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tool_prefix: str = "",
    auto_connect: bool = True,
    **transport_kwargs: Any,
) -> MCPToolManager:
    """
    便捷函数：创建MCPToolManager并将所有远程MCP工具注册到ToolRegistry。
    
    Args:
        registry: ToolRegistry实例
        server_source: MCP服务器源标识
        server_args: 启动服务器的命令行参数
        transport_type: 传输类型
        env: 环境变量
        tool_prefix: 工具名前缀
        auto_connect: 是否自动连接
        **transport_kwargs: 额外的传输参数
        
    Returns:
        创建的MCPToolManager实例
    """
    # 创建管理器
    manager = MCPToolManager(
        server_source=server_source,
        server_args=server_args,
        transport_type=transport_type,
        env=env,
        tool_prefix=tool_prefix,
        auto_connect=auto_connect,
        **transport_kwargs,
    )
    # 注册所有远程工具
    manager.register_to_registry(registry)
    return manager


def mcptool(
    server_source: Any,
    server_args: Optional[List[str]] = None,
    transport_type: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    tool_prefix: str = "",
    auto_connect: bool = True,
    **transport_kwargs: Any,
) -> MCPToolManager:
    """
    语法糖函数：返回MCPToolManager实例，可配合ToolRegistry.registry(...)使用。
    
    通常用法：
        registry = ToolRegistry.registry(
            mcptool(server_source="my_server", ...)
        )
    
    Args:
        server_source: MCP服务器源标识
        server_args: 启动服务器的命令行参数
        transport_type: 传输类型
        env: 环境变量
        tool_prefix: 工具名前缀
        auto_connect: 是否自动连接
        **transport_kwargs: 额外的传输参数
        
    Returns:
        MCPToolManager实例
    """
    return MCPToolManager(
        server_source=server_source,
        server_args=server_args,
        transport_type=transport_type,
        env=env,
        tool_prefix=tool_prefix,
        auto_connect=auto_connect,
        **transport_kwargs,
    )
