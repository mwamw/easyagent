# EasyAgent Language v1

EasyAgent Language（`.ea`）是一门面向 **Agent Harness 装配**的声明式领域语言。
它提供命名组件、类型化引用、局部导入、参数化模板和启动声明，不是工作流图 DSL，
也不是运行在 Agent 内部的另一套执行引擎。

Python 创造能力，`.ea` 组合能力，UI 使用能力。原有 Python 和 YAML 接口继续保留。

## 1. 从一个可运行的文件开始

在 EasyAgent 仓库目录、已有执行环境中安装：

```bash
conda activate S4Code
python -m pip install -e ".[ui]"
easyagent-lang check example/language/minimal.ea
easyagent-lang explain example/language/minimal.ea
```

`check` 不读取环境变量、不创建客户端、不连接 MCP、不执行工具，也不会创建数据库。
它只读取源文件并检查语法、引用、类型和已注册的配置 Schema。

设置示例引用的 `EASYAGENT_MODEL`、`EASYAGENT_API_KEY`，按需设置
`EASYAGENT_BASE_URL` 后运行：

```bash
easyagent-lang run example/language/minimal.ea
easyagent-lang run example/language/minimal.ea --query "解释什么是 Agent Harness"
```

交互模式需要终端和 UI 可选依赖。单次 `--query` 通过 `ainvoke` 执行，遇到 ASK
会报告需要交互，不会自动批准。所有命令也可以用 `python -m easyagent.language` 调用。

不调用任何模型服务的体验入口：

```bash
python -m example.language.offline_demo --check
python -m example.language.offline_demo --query "你好"
python -m example.language.offline_demo
```

离线启动器显式注册了 `offline` 模型；通用 CLI 不会根据 `.ea` 自动导入这个 Python 类。

## 2. 最小语法

```ea
model Main {
    provider = "openai"
    model = env("EASYAGENT_MODEL")
    api_key = env("EASYAGENT_API_KEY")
}

agent Assistant {
    model = Main
    workspace = path(".")
    context { max_tokens = 32000 }
    permission using rule_based { approval_mode = "ask_for_approval" }
    tools { filesystem() ask_user_question() }
}

run Assistant with terminal { streaming = true }
```

- 声明形式为 `类别 名称 [using 实现] { 配置 }`；省略 using 表示 default 实现。
- Agent 内既可以引用命名组件，也可以内联声明组件。
- 组件块中的字段直接对应注册实现的配置，无需再嵌套 config。
- Agent 的 `config = {...}` 对应现有 `Config`；它不会取代模块自己的配置。
- 名称区分大小写，支持 ASCII 字母、数字、下划线，不能以数字开头。
- 普通字符串使用双引号和 JSON 转义；三双引号是保留换行的原始字符串。
- 支持 `#` 和 `//` 行注释。字段间逗号、分号可省略；列表和调用参数使用逗号。
- 对象可以用 `key = value` 或 `"key": value`；重复字段和参数是错误，不会覆盖。
- `true` / `false` 是布尔值，`none` / `null` 表示空值。
- 不隐式把字符串转成数值，也不把布尔值当整数使用。

例如源码写 `max_tokens = "32000"`，检查阶段就会给出文件、行列和类型错误。
诊断不打印原始源码行或输入值，避免把密钥复制到报错中。

## 3. Agent 的能力与可选功能

| 语言字段 | 配置层对应 | 行为 |
| --- | --- | --- |
| model | model | 必填，选择模型实现 |
| context / executor / prompt | 同名字段 | 可省略，使用普通 Agent 的默认组件；不可设 none |
| permission | permissions | 可替换权限引擎；省略或 none 表示不安装 |
| memory / plan / tasks | 同名字段 | 选择注册实现，或 none |
| session / observability | 同名字段 | 例如 default 内存实现、sqlite 实现 |
| hooks / callbacks / codeintel / worktree / multi_agent | 同名字段 | 选择注册实现，或 none |
| tools | tools | 工具加载条目，不替换 ToolRegistry |
| skills | skills | Skill 目录，不替换 Skill 机制 |
| mcp | mcp | Server 配置，不替换 MCP 机制 |

工具配置示例：

```ea
tools {
    filesystem()
    file_edit()
    disabled bash(sandbox = {
        enabled = true
        fail_if_unavailable = true
        allow_unsandboxed_commands = false
        network = "deny"
    })
}
skills { directory(path("./skills")) }
memory = none
```

`disabled` 工具不会加载实现或解析其环境变量。启用前必须再次通过配置校验。
框架仍负责需要工具能力时创建 ToolRegistry；语言不重写各模块的接入机制。

MCP 采用有类型校验的对象配置：

```ea
mcp = {
    enabled = true
    servers = [{
        name = "research"
        source = "python"
        source_type = "command"
        args = ["-m", "your_mcp_server"]
        env = { API_KEY = env("RESEARCH_API_KEY") }
    }]
}
```

编译不会连接 Server。MCP 的安装和实际连接仍由现有框架机制负责。

## 4. 环境变量和路径

只有两个内置值构造函数：

```ea
api_key = env("SERVICE_API_KEY")
base_url = env("SERVICE_URL", default = "http://localhost:8000/v1")
workspace = path("./workspace")
```

`env` 返回 String，在构建所选 Agent 时读取。静态检查只检查类型，具体值的枚举、
范围等约束在绑定后重新检查。没有 `eval`、shell substitution 或隐式 `${...}` 展开。
环境变量名使用可移植的字母、数字、下划线格式；default 必须是普通文本。

`path` 返回 Path。相对路径相对于**包含该 path 表达式的源文件**，不相对于进程 cwd。
`path(env("WORKSPACE"))` 也支持，环境变量绑定后再解析相对路径。

模板内写的 `path("./state")` 相对于模板文件；调用方传入的 `path("./project")`
相对于调用方文件。workspace 和 prompt.file 要求 Path；其他组件的路径配置也应显式
使用 path。传入普通字符串的旧式组件路径，仍遵循底层 Builder 的入口文件相对语义。

静态检查不要求目录已经存在。`check --environment` 会绑定环境变量并通过 Builder
检查 workspace、Skill、Prompt 等本地资源，但不会创建 Agent；它仍不证明服务可用、
凭据有效或 OS 沙箱可用。

## 5. 导入和模板

```ea
import { Main as MainModel } from "./models.ea"
import { ResearchHarness } from "./shared/research.ea"

agent PaperAssistant = ResearchHarness(
    workspace = path("./papers"),
    model = MainModel,
    budget = 64000
)
```

只支持本地 `.ea` 文件的显式名称导入，可以重导出已导入名称。不支持 Python 导入、
网络下载或自动安装扩展。导入不会执行被导入文件的 run 声明；入口只使用当前文件的 run。
同名声明、缺失名称、循环导入和循环组件引用都会失败，支持普通前向引用。

模板定义：

```ea
define ResearchHarness(workspace: Path, model: Model, budget: Integer = 32000) -> Agent {
    agent {
        workspace = workspace
        model = model
        context { max_tokens = budget }
        tools { filesystem() web_fetch() }
    }
}
```

模板参数必须带类型，调用采用命名参数。默认参数可引用此前的参数；默认表达式和模板体
使用定义文件的词法作用域，调用者局部名称不会意外改变它们。
模板体是单个返回表达式，可以选择写 `return`，不支持语句循环、赋值或递归。

值类型：String、Path、Integer、Number、Boolean。Integer 可以用于 Number 参数。
组件类型：Model、Context、Permission、Memory、Prompt、Executor、Plan、Tasks、Session、
Observability、Hooks、Callbacks、CodeIntel、Worktree、MultiAgent、Agent。

所有声明、未调用的模板也会检查；模板参数相关的取值约束在具体调用时再次检查。
复用的是**定义**，不是运行实例。每次 build 都创建独立 Agent、Runtime 和组件；
共享数据库路径等外部资源属于显式配置，不代表共享 Agent 历史和锁。

## 6. 启动、检查和选择 Agent

```bash
easyagent-lang check research.ea
easyagent-lang check research.ea --environment --agent PaperAssistant
easyagent-lang explain research.ea
easyagent-lang run research.ea --agent PaperAssistant --query "阅读当前项目"
easyagent-lang schema
```

一个文件最多一个 run 声明。未指定 Agent 时，优先使用 run 的目标；没有 run 且只有一个
Agent 时使用它，否则要求显式选择。只包含模板的库文件也可以单独 check。

`run A with terminal {...}` 配置外部 UI，不是 agent.with_ui，也不参与模型工具循环。
`run A` 或没有 UI 声明的交互运行使用默认终端 UI。UI 配置与 Agent 配置独立。
`--query` 不创建 UI。explain 只展示组件和工具选择，不展开配置值或密钥。

CLI 退出码：0 成功，2 源码/配置/启动参数错误，3 执行失败或需要交互，130 用户中断。

## 7. Python 使用和扩展

```python
from easyagent.language import compile_file

program = compile_file("./research.ea")
agent = program.build("PaperAssistant")
try:
    print(agent.invoke("阅读当前项目"))
finally:
    agent.close()
```

`build()` 返回普通 BasicAgent，可以继续使用公开 `with_*`、四种调用 API、会话和交互接口。
`program.configuration()` 返回绑定后的 AgentConfiguration，**可能包含明文密钥，不要记录**。
可用 `environ={...}` 显式传入环境变量映射；未传时使用 os.environ。

新实现通过现有 ConfigurationRegistry 注册，无需修改语法或执行器：

```python
from easyagent.configuration import Options, default_registry
from easyagent.context import ContextManager
from easyagent.language import compile_file
from pydantic import Field

class DomainContextOptions(Options):
    budget: int = Field(gt=0)

registry = default_registry()
registry.register_context(
    "domain",
    config_model=DomainContextOptions,
    factory=lambda options, build: ContextManager(max_tokens=options.budget),
)
program = compile_file("./domain.ea", registry=registry)
```

此后可以写 `context using domain { budget = 16000 }`。自定义 Memory、Permission、
Executor 等使用各自注册方法；工具通过 register_tools 加入现有 ToolRegistry。
配置模型提供类型和约束，工厂仍通过公共接口装配。Source 不能自行加载任意 Python 工厂。
第三方注册代码是受信任 Python；其配置校验器应保持纯校验，不执行 IO 或有副作用的操作。

自定义 UI 通过 UIRegistry 传给 compile_file 的 ui_registry 参数，同样先检查配置，再按需创建。

## 8. 实现边界

```text
.ea → Lexer / Parser → 名称和类型检查 → 配置 Schema 检查 → 编译后的配方
                                                            ↓
YAML → 配置解析 ───────────────────────────────────→ AgentBuilder
                                                            ↓
                                                  公开构造接口 / with_*
                                                            ↓
                                                       BasicAgent
```

编译器不生成 Python 源码、不调用 eval/exec、不持有运行中的 Agent。
Builder 新增 build_document / validate_document，使两个前端共用装配、路径检查和失败清理。
Runtime、Permission、Context、Tool 和 Executor 不需要知道源文件是 YAML 还是 `.ea`。

实现目录：language/syntax.py、lexer.py、parser.py、evaluation.py、checking.py、compiler.py、
program.py；公共入口在 easyagent/language.py。

v1 不提供通用控制流、运算符、任意函数执行、远程包管理、LSP 或热更新。
改变推理循环仍通过 Python 实现 Executor 并在语言里选择它；不把模型的每一步行动写成固定流程。
附带的 VS Code 本地扩展仅提供语法高亮、括号和注释支持，不冒充完整语义补全。

为限制误写导致的资源消耗，单文件最多 200 万字符、10 万 token，最多导入 128 个文件、
导入深度 32、模板调用深度 64，表达式展开最多 10 万步。更深的语法嵌套会明确失败。
这些限制不构成 OS 沙箱。已注册工具仍可能操作文件、网络和进程，安全边界由权限和工具执行后端保障。

## 示例与验证

- `example/language/minimal.ea`：最小助手。
- `example/language/coding.ea`：工具、权限、Skill、Plan、Session、子 Agent 装配。
- `example/language/research.ea`：跨文件导入和参数化 Harness。
- `example/language/offline_demo.py`：无 API 的真实编译和运行链路。
- `test/test_language.py`：类型、导入、模板、配置扩展、四种调用路径、隔离和错误处理。

```bash
python -m pytest test/test_language.py test/test_configuration.py -q
```
