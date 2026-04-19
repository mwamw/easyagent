import json
import os
import sys

from dotenv import load_dotenv
load_dotenv("/home/wxd/LLM/EasyAgent/example/.env")

project_root = "/home/wxd/LLM/EasyAgent"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.llm import EasyLLM
from core.Config import Config
from agent import BasicAgent
from runtime import TeamManager
from task import SQLiteTaskStore, TaskService
from Tool import ToolRegistry
from Tool.builtin import (
    register_filesystem_tools,
    register_file_write_tool,
    register_file_edit_tool,
    register_shell_tools,
    register_agent_tool,
    register_send_message_tool,
    register_team_create_tool,
    register_team_delete_tool,
)

workspace = project_root
example_dir = os.path.join(project_root, "example")

llm = EasyLLM(provider="openai")
config = Config(workspace_root=workspace, allowed_roots=[workspace], enable_worktree=True)
task_service = TaskService(SQLiteTaskStore(os.path.join(example_dir, "db", "phase2_check.db")))

registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root=workspace)
register_file_write_tool(registry, workspace_root=workspace)
register_file_edit_tool(registry, workspace_root=workspace)
register_shell_tools(registry, workspace_root=workspace)

agent = BasicAgent(
    name="SchemaChecker",
    llm=llm,
    tool_registry=registry,
    enable_tool=True,
    config=config,
    task_service=task_service,
)

agent_tool = register_agent_tool(registry, parent_agent=agent, workspace_root=workspace)
team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
agent_tool.agent_runtime.bind_team_manager(team_manager)
register_send_message_tool(registry, agent_runtime=agent_tool.agent_runtime, parent_agent=agent)
register_team_create_tool(registry, team_manager=team_manager)
register_team_delete_tool(registry, team_manager=team_manager)
agent.bind_runtime(agent_runtime=agent_tool.agent_runtime, team_manager=team_manager)

# Now dump ALL tools
tools = registry.get_openai_tools()
for i, tool in enumerate(tools):
    name = tool["function"]["name"]
    params = tool["function"]["parameters"]
    required = params.get("required", [])
    properties = list(params.get("properties", {}).keys())

    missing = [r for r in required if r not in properties]
    status = "❌ BROKEN" if missing else "✅ OK"
    print(f"[{i}] {name}: {status}")
    if missing:
        print(f"    required={required}")
        print(f"    properties={properties}")
        print(f"    MISSING: {missing}")
        print(f"    Full schema:")
        print(json.dumps(params, indent=4, ensure_ascii=False))
    
    # Also check nested properties recursively
    def check_nested(schema, path=""):
        if not isinstance(schema, dict):
            return
        req = schema.get("required", [])
        props = list(schema.get("properties", {}).keys())
        if req and props:
            nested_missing = [r for r in req if r not in props]
            if nested_missing:
                print(f"    ❌ NESTED at {path}: required={req}, properties={props}, MISSING={nested_missing}")
                print(f"    Full nested schema:")
                print(json.dumps(schema, indent=4, ensure_ascii=False))
        for k, v in schema.get("properties", {}).items():
            check_nested(v, f"{path}.{k}")
        if "items" in schema and isinstance(schema["items"], dict):
            check_nested(schema["items"], f"{path}[items]")
    
    check_nested(params, name)
