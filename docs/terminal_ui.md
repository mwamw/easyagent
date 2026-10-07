# 默认终端 UI

UI 是 Agent 的外部交互层，不是安装到 Executor 中的能力模块。
所有 UI 实现继承 `BaseAgentUI`，使用 Agent 的公开接口；没有 `with_ui()`，
也没有新的 Agent Loop。终端依赖为可选安装项。

## 启动

```bash
conda activate S4Code
python -m pip install -e ".[ui]"
# 编辑模型名称、服务地址并设置配置引用的环境变量后：
easyagent-chat example/configuration/coding.yaml
# 等价入口：
python -m easyagent.ui example/configuration/coding.yaml
```

原来的 `easyagent-run --query` 仍是单请求、非交互入口。
easyagent-chat 要求交互式终端；重定向/管道请使用 easyagent-run。

可以在原有 Agent 配置中添加平级 UI 设置：

```yaml
ui:
  implementation: terminal
  config:
    streaming: true
    show_tool_results: false
    max_iter: 20
```

省略 ui 时，chat 启动器选用 terminal。设 enabled: false 会禁止 chat 启动；
`load_agent()` 仍只创建 Agent，不实例化 UI，也不验证 UI 实现的专属选项。
聊天启动器先创建/校验 UI，再加载 Agent。

## 使用方式

- Enter 发送；Alt+Enter（或 Esc、Enter）换行；支持多行粘贴。
- 输入历史仅保存在本次 UI 实例内存中，不自动写入磁盘。
- 空闲时 Ctrl+C 清理当前输入；Ctrl+D 或 /exit 退出。
- 运行中不接受第二个请求，Ctrl+C 发出协作停止请求。
- /help 显示帮助；/tools 显示最近一轮完整工具参数与结果。
- /clear 只清理屏幕，绝不清空 Agent 历史或重置会话。

模型文本通过 astream 增量显示；工具默认显示摘要和状态，不刷出所有输出。
streaming: false 时使用 ainvoke 显示完整结果；工具状态仍来自公共 RuntimeEvent。
审批恢复后的继续执行使用 astream_continue。

活动区域最多约每秒刷新 12 次，静止时只更新计时；不使用全屏 alternate screen。
最终回答以 Markdown/代码块展示。最终事件不会把已展示的答案重复打印。
生成中使用纯文本预览，避免不完整代码块触发高亮插件错误；预览最多取最近
12000 个字符，不截断保存的回答。最终 Markdown 渲染失败时降级为纯文本。
reasoning_delta 不默认展示；不会主动打印 system prompt、system reminder、
完整模型请求或底层异常正文。工具返回内容中的终端控制序列会过滤。
工具参数和完整结果可能包含敏感信息，请不要把终端录屏当作已脱敏日志。

## 审批、提问与取消

UI 显示工具名、参数、工作区和请求理由。需要审批时必须明确选择
“允许本次”或“拒绝”；空输入不会默认批准，不提供永久允许快捷选项。
AskUserQuestion 显示题目、选项，也支持自由文本和多选。
Plan 的进入/退出请求在用户确认后走 Plan 模块的公开操作。

取消不是进程强杀：发出停止请求后，UI 仍等待已经启动的工具结束，
不会提前释放 Agent 的调用锁或接受新请求。外部服务、后台 Bash 进程及工具
已经产生的副作用不会因此自动撤销。等待模型无输出时，停止可能要等到下一
个流事件或请求超时；界面会明确显示等待状态。

为保持异步界面响应，Tool 的默认 arun 现在通过 asyncio.to_thread 调用 run；
已有原生 arun 实现不变。依赖线程亲和性的自定义工具应实现自己的 arun，
并确保不阻塞 UI 事件循环。UI 不取消线程等待来假装工具已停止。

## Python 与自定义 UI

```python
from easyagent.configuration import load_agent
from easyagent.ui import TerminalUI

async def main():
    agent = load_agent("./agent.yaml")
    ui = TerminalUI()
    try:
        await ui.run(agent)
    finally:
        await ui.aclose()
        await agent.aclose()
```

UI 借用 Agent，不隐式关闭它。chat 启动器拥有其创建的 Agent，并负责关闭；
外部 Python 调用者自行管理生命周期。

新的界面只需实现稳定的小契约：

```python
from easyagent.ui import BaseAgentUI

class MyUI(BaseAgentUI):
    async def run(self, agent):
        # 接收输入，消费 agent.astream(...)，处理交互。
        ...

    async def aclose(self):
        # 释放界面自己的连接、订阅和任务，不关闭借用的 Agent。
        ...
```

可通过 UIRegistry 显式注册名字、Pydantic 配置模型和 UI 工厂，再调用
`load_ui(path, registry=registry)`。工厂必须返回 BaseAgentUI。
不支持从 YAML 执行任意 Python 导入字符串。

## 公共交互接口

新增的交互入口不仅供终端使用，其他 UI 也可调用：

```python
from easyagent.ui import InteractionReply

pending = agent.get_pending_interruption()
await agent.arespond_to_interruption(
    InteractionReply(pending["request_id"], "approve")
)
async for event in agent.astream_continue():
    ...
```

reply.action 可为 approve、deny、answer，answer 用于工具向用户提问。
请求 ID 必须与当前中断匹配，防止旧界面的确认误用于新动作。
回答/拒绝作为对应 tool result 写入，继续执行不伪造新的用户消息。
一次模型响应包含多个工具调用时，恢复只处理尚未完成的调用。

授权回复重新经过 Executor 的 Hook、Permission 和 Tool 流程，不直接调用
Tool.run，也不越过显式 DENY。BasePermissionEngine 提供可选的
`approval(grant)` 上下文管理协议；内置引擎实现临时 Grant 并保证退出时清理。
自定义引擎未实现此协议时，批准会明确失败，仍可拒绝；不会默认绕过它。
自定义 Executor 可实现 arespond_to_interruption 与 astream_continue；
未实现时明确报不支持，不调用默认执行器的私有方法兜底。

目前不包含会话浏览器、checkpoint 回溯、多 Agent 面板或配置编辑器。
未知类型的工具交互默认只允许拒绝或退出，不猜测批准后的业务行为。

测试：test/test_terminal_ui.py 使用真实 prompt_toolkit 管道输入；
test/test_ui_interactions.py 验证审批、提问、多工具恢复、取消和历史配对。
均不调用收费模型。未安装 UI 可选依赖时，终端专属测试会跳过。
