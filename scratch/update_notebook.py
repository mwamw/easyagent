import json

# Define the helper code to add to the first code cell
helper_code = """
from agent import BasicAgent
from core.llm import EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)

async def test_agent_with_tool(registry, prompt: str):
    agent = BasicAgent(
        name="TestAgent",
        llm=llm,
        tool_registry=registry,
        enable_tool=True,
    )
    print("\\n=== [Agent 调用测试] ===")
    print(f"提问: {prompt}")
    try:
        res = await agent.astream_invoke(prompt, max_iter=3)
        print(f"Agent最终回复: {res}")
    except Exception as e:
        print(f"Agent调用异常: {e}")
"""

def update_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Initialization block
            if "==== 环境初始化 ====" in source:
                if "# [Agent 调用测试]" not in source:
                    cell["source"].append("\n# [Agent 调用测试]\n")
                    for line in helper_code.strip().split('\n'):
                        cell["source"].append(line + "\n")
            
            # Tool blocks
            if "==== 1. Calculator Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请帮我计算一下 (3+2)*10 等于多少？')\n")
            elif "==== 2. FileRead Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, f'请读取当前目录下的临时文件 {test_file} 的前三行并告诉我内容。')\n")
            elif "==== 3. Glob Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, f'请搜索 {SCRATCH_DIR} 目录下所有的 .py 文件有哪些？')\n")
            elif "==== 4. Grep Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, f'请帮我在 {SCRATCH_DIR} 下搜索包含 \"Hello\" 的内容。')\n")
            elif "==== 5. FileWrite Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, f'请把一段测试文本 \"Hello from agent!\" 写入到 {write_path} 文件中，完全覆盖写入。')\n")
            elif "==== 6. FileEdit Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, f'请把文件 {edit_path} 中的 \"def add(a: int, b: int) -> int:\" 替换成没有类型注解的 \"def add(a, b):\"。')\n")
            elif "==== 7. Bash Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请通过终端帮我执行一个 echo 命令输出 Hello bash agent!。')\n")
            elif "==== 8. TodoWrite Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '当前有一个重构数据库连池的TODO，请帮我把状态改为已完成。')\n")
            elif "==== 9. TaskCreate" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请帮我查一下现在有哪些任务？如果有任务，请帮我把状态改为 completed。')\n")
            elif "==== 10. Config Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请帮我查一下当前的 workspace_root 配置是什么？然后尝试将其改为新的路径。')\n")
            elif "==== 11. WebFetch Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请帮我抓取 https://example.com 并提取一下主要内容。')\n")
            elif "==== 12. WebSearch Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请帮我搜索一下 python asyncio 的最新教程。')\n")
            elif "==== 13. NotebookEdit Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, f'请在 notebook {nb_path} 中插入一个包含 print(Hello) 的代码块。')\n")
            elif "==== 14. TaskOutput" in source and "test_agent_with_tool" not in source:
                # Need to give the agent task_id explicitly
                cell["source"].append("\nif task_id:\n    await test_agent_with_tool(registry, f'请帮我查询后台任务 {task_id} 的输出。')\n")
            elif "==== 15. Agent Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请帮我启动一个名为 helper 的子 agent，让它后台执行去查一下项目根目录下有哪些文。')\n")
            elif "==== 16. SendMessage Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请将一句问候语偷偷发给名为 somebody 的 agent。')\n")
            elif "==== 17. TeamCreate" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请创建一个叫 new-team 的团队，描述随意。然后查询一下是不是创建成功了？')\n")
            elif "==== 18. AskUserQuestion Tool ====" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '请向我提问，问我喜欢 python 还是 java，并提供相应的选项让我选。')\n")
            elif "==== 19. EnterPlanMode" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '这是一个复杂任务，请向我申请进入计划模式。')\n")
            elif "==== 20. EnterWorktree" in source and "test_agent_with_tool" not in source:
                cell["source"].append("\nawait test_agent_with_tool(registry, '我要修改一个危险文件，请帮我开启一个隔离的 worktree。')\n")
                
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        
update_notebook("/home/wxd/LLM/EasyAgent/example/test_builtin_tools.ipynb")
