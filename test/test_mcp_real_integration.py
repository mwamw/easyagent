import asyncio
import os
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv()

from mcp import MCPClient
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_mcp_tools, mcptool
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent


class MCPRealIntegrationRunner:
    def __init__(self):
        print("========== 初始化真实 MCP + LLM 集成测试环境 ==========")
        self.workspace = _project_root
        self.examples_dir = os.path.join(self.workspace, "mcp", "examples")
        self.python_server_script = os.path.join(self.examples_dir, "real_python_mcp_server.py")
        self.python_text_server_script = os.path.join(self.examples_dir, "real_python_mcp_server_text.py")
        self.python_structured_server_script = os.path.join(self.examples_dir, "real_python_mcp_server_structured.py")

        required_scripts = [
            self.python_server_script,
            self.python_text_server_script,
            self.python_structured_server_script,
        ]
        for script in required_scripts:
            if not os.path.exists(script):
                raise FileNotFoundError(f"未找到真实 Python MCP Server 脚本: {script}")

        self.npx_service_specs = [
            {
                "name": "filesystem",
                "server_source": [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    self.workspace,
                ],
                "expected_any_tools": ["list_directory", "list_dir", "read_file"],
                "sample_call_candidates": [
                    ("list_directory", {"path": "."}),
                    ("list_dir", {"path": "."}),
                    ("read_file", {"path": "README.md"}),
                ],
            },
            {
                "name": "memory",
                "server_source": [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-memory",
                ],
                "expected_any_tools": ["create_memory", "add_memory", "search_memory", "list_memories"],
                "sample_call_candidates": [],
            },
            {
                "name": "time",
                "server_source": [
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-time",
                ],
                "expected_any_tools": ["get_current_time", "get_time", "convert_time"],
                "sample_call_candidates": [],
            },
        ]

        self.llm = EasyLLM()
        print("✅ EasyLLM 初始化完成（真实 LLM）")

    def _node_major_version(self) -> int:
        try:
            out = subprocess.check_output(["node", "-v"], text=True).strip()
            if out.startswith("v"):
                out = out[1:]
            return int(out.split(".")[0])
        except Exception:
            return 0

    async def test_mcp_client_python_stdio(self):
        print("\n========== 测试1: MCPClient 直连 Python MCP Server（真实） ==========")
        client = MCPClient(server_source=["python", self.python_server_script])

        await client.connect()
        try:
            tools = await client.list_tools()
            tool_names = [t["name"] for t in tools]
            print("发现工具:", tool_names)

            assert "echo" in tool_names, "echo 工具缺失"
            assert "add" in tool_names, "add 工具缺失"
            assert "repeat" in tool_names, "repeat 工具缺失"

            r1 = await client.call_tool("echo", {"text": "hello"})
            r2 = await client.call_tool("add", {"a": 2, "b": 9})
            r3 = await client.call_tool("repeat", {"text": "mcp", "times": 3})

            print("echo =>", r1)
            print("add =>", r2)
            print("repeat =>", r3)

            assert str(r1).startswith("echo:"), "echo 返回异常"
            assert str(r2) == "11", "add 返回异常"
            assert "mcp mcp mcp" in str(r3), "repeat 返回异常"

            ok = await client.ping()
            assert ok is True, "MCP ping 失败"
            print("✅ MCPClient + Python MCP Server 真实链路通过")
        finally:
            await client.disconnect()

    def test_mcp_tool_manager_python_stdio(self):
        print("\n========== 测试2: MCPToolManager 注册/调用 Python MCP（真实） ==========")
        registry = ToolRegistry()
        manager = register_mcp_tools(
            registry=registry,
            server_source=["python", self.python_server_script],
            tool_prefix="py_",
        )

        try:
            tool_names = sorted(registry.tools.keys())
            print("注册工具:", tool_names)

            assert "py_echo" in registry.tools, "py_echo 未注册"
            assert "py_add" in registry.tools, "py_add 未注册"

            out1 = registry.executeTool("py_echo", {"text": "tool manager"})
            out2 = registry.executeTool("py_add", {"a": 10, "b": 5})

            print("py_echo =>", out1)
            print("py_add =>", out2)

            assert "echo:" in out1, "py_echo 调用异常"
            assert out2 == "15", "py_add 调用异常"
            print("✅ MCPToolManager + ToolRegistry 真实链路通过")
        finally:
            manager.close()

    async def test_mcp_client_npx_filesystem(self):
        print("\n========== 测试3: MCPClient 直连 npx filesystem MCP（真实） ==========")
        major = self._node_major_version()
        if major < 18:
            print(f"⚠️ Node.js 版本为 {major}，低于 18，跳过 npx 真实测试")
            return

        client = MCPClient(
            server_source=[
                "npx",
                "-y",
                "@modelcontextprotocol/server-filesystem",
                self.workspace,
            ]
        )

        await client.connect()
        try:
            tools = await client.list_tools()
            tool_names = [t["name"] for t in tools]
            print("发现工具:", tool_names)

            assert "list_directory" in tool_names or "list_dir" in tool_names, "目录工具缺失"

            if "list_directory" in tool_names:
                out = await client.call_tool("list_directory", {"path": "."})
            else:
                out = await client.call_tool("list_dir", {"path": "."})

            print("list_directory/list_dir =>", out)
            assert out is not None and len(str(out)) > 0, "npx 文件系统工具返回为空"
            print("✅ MCPClient + npx filesystem 真实链路通过")
        finally:
            await client.disconnect()

    async def test_mcp_client_python_text_stdio(self):
        print("\n========== 测试4: MCPClient 直连 Python 文本 MCP 服务（真实） ==========")
        client = MCPClient(server_source=["python", self.python_text_server_script])

        await client.connect()
        try:
            tools = await client.list_tools()
            tool_names = [t["name"] for t in tools]
            print("发现工具:", tool_names)

            assert "word_count" in tool_names, "word_count 工具缺失"
            assert "reverse_text" in tool_names, "reverse_text 工具缺失"
            assert "upper_text" in tool_names, "upper_text 工具缺失"

            r1 = await client.call_tool("word_count", {"text": "easy agent mcp"})
            r2 = await client.call_tool("reverse_text", {"text": "abc"})
            r3 = await client.call_tool("upper_text", {"text": "mcp"})

            print("word_count =>", r1)
            print("reverse_text =>", r2)
            print("upper_text =>", r3)

            assert str(r1) == "3", "word_count 返回异常"
            assert str(r2) == "cba", "reverse_text 返回异常"
            assert str(r3) == "MCP", "upper_text 返回异常"
            print("✅ MCPClient + Python 文本 MCP 服务真实链路通过")
        finally:
            await client.disconnect()

    async def test_mcp_client_python_structured_stdio(self):
        print("\n========== 测试5: MCPClient 直连 Python 结构化 MCP 服务（真实） ==========")
        client = MCPClient(server_source=["python", self.python_structured_server_script])

        await client.connect()
        try:
            tools = await client.list_tools()
            tool_names = [t["name"] for t in tools]
            print("发现工具:", tool_names)

            assert "make_profile" in tool_names, "make_profile 工具缺失"
            assert "sum_list" in tool_names, "sum_list 工具缺失"
            assert "make_tags" in tool_names, "make_tags 工具缺失"

            r1 = await client.call_tool("make_profile", {"name": "wxd", "age": 18})
            r2 = await client.call_tool("sum_list", {"numbers": [1, 2, 3, 4]})
            r3 = await client.call_tool("make_tags", {"topic": "mcp"})

            print("make_profile =>", r1)
            print("sum_list =>", r2)
            print("make_tags =>", r3)

            assert "\"name\": \"wxd\"" in str(r1), "make_profile 返回异常"
            assert str(r2) == "10", "sum_list 返回异常"
            assert "mcp" in str(r3), "make_tags 返回异常"
            print("✅ MCPClient + Python 结构化 MCP 服务真实链路通过")
        finally:
            await client.disconnect()

    async def test_mcp_client_multiple_npx_services(self):
        print("\n========== 测试6: MCPClient 直连多个 npx MCP 服务（真实） ==========")
        major = self._node_major_version()
        if major < 18:
            print(f"⚠️ Node.js 版本为 {major}，低于 18，跳过多 npx 服务测试")
            return

        for spec in self.npx_service_specs:
            print(f"\n---- npx 服务: {spec['name']} ----")
            client = MCPClient(server_source=spec["server_source"])
            try:
                await client.connect()
                tools = await client.list_tools()
                tool_names = [t["name"] for t in tools]
                print("发现工具:", tool_names)

                expected = spec["expected_any_tools"]
                assert any(name in tool_names for name in expected), (
                    f"{spec['name']} 未发现预期工具，期望之一: {expected}"
                )

                for tool_name, params in spec["sample_call_candidates"]:
                    if tool_name in tool_names:
                        out = await client.call_tool(tool_name, params)
                        print(f"{tool_name} =>", out)
                        assert out is not None and len(str(out)) > 0, f"{tool_name} 返回为空"
                        break

                print(f"✅ npx 服务 {spec['name']} 真实链路通过")
            except Exception as e:
                # 某些 npx server 可能受运行环境、版本和网络影响。不中断整套测试。
                print(f"⚠️ npx 服务 {spec['name']} 测试失败（环境相关可跳过）: {e}")
            finally:
                await client.disconnect()

    def test_agent_integration_python_mcp_real_llm(self):
        print("\n========== 测试7: BasicAgent + 真实 LLM + Python MCP 工具 ==========")
        registry = ToolRegistry()
        manager = register_mcp_tools(
            registry=registry,
            server_source=["python", self.python_server_script],
            tool_prefix="py_",
        )

        agent = BasicAgent(
            name="real-mcp-agent-py",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            system_prompt=(
                "你是一个会调用工具的助手。"
                "当用户需要计算或回显时，优先调用对应的 MCP 工具并给出结果。"
            ),
            verbose_thinking=False,
        )

        try:
            query = "请调用 py_add 计算 18 + 24，并仅返回结果。"
            print("User:", query)
            resp = agent.invoke(query)
            print("Agent:", resp)

            # 真实 LLM 输出可能带自然语言，这里只做弱断言
            assert "42" in str(resp), "Agent 未正确使用工具或结果不含 42"
            print("✅ BasicAgent + 真实 LLM + Python MCP 工具链路通过")
        finally:
            manager.close()

    def test_agent_integration_npx_mcp_real_llm(self):
        print("\n========== 测试8: BasicAgent + 真实 LLM + npx filesystem MCP ==========")
        major = self._node_major_version()
        if major < 18:
            print(f"⚠️ Node.js 版本为 {major}，低于 18，跳过 npx + Agent 真实测试")
            return

        registry = ToolRegistry()
        manager = register_mcp_tools(
            registry=registry,
            server_source=[
                "npx",
                "-y",
                "@modelcontextprotocol/server-filesystem",
                self.examples_dir,
            ],
            tool_prefix="fs_",
        )

        agent = BasicAgent(
            name="real-mcp-agent-npx",
            llm=self.llm,
            enable_tool=True,
            tool_registry=registry,
            system_prompt=(
                "你是一个会调用文件系统 MCP 工具的助手。"
                "当用户要求查看目录时，请调用对应工具。"
            ),
            verbose_thinking=False,
        )

        try:
            query = "请列出当前可访问目录下的文件名。"
            print("User:", query)
            resp = agent.invoke(query)
            print("Agent:", resp)

            # 弱断言：至少包含 examples 目录中的脚本文件名之一
            expected_any = [
                "real_python_mcp_server.py",
                "example_real_python_stdio_client.py",
                "example_real_npx_filesystem_client.py",
            ]
            assert any(name in str(resp) for name in expected_any), "Agent 输出中未包含预期文件名"
            print("✅ BasicAgent + 真实 LLM + npx filesystem MCP 链路通过")
        finally:
            manager.close()

    def run_all(self):
        print("\n========== 开始执行真实 MCP 全链路测试 ==========")
        asyncio.run(self.test_mcp_client_python_stdio())
        self.test_mcp_tool_manager_python_stdio()
        asyncio.run(self.test_mcp_client_npx_filesystem())
        asyncio.run(self.test_mcp_client_python_text_stdio())
        asyncio.run(self.test_mcp_client_python_structured_stdio())
        asyncio.run(self.test_mcp_client_multiple_npx_services())
        self.test_agent_integration_python_mcp_real_llm()
        self.test_agent_integration_npx_mcp_real_llm()
        print("\n🏁 全部真实 MCP 集成测试执行完成")


if __name__ == "__main__":
    runner = MCPRealIntegrationRunner()
    runner.run_all()
