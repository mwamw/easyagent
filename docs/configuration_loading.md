# 配置文件装配 Agent

这是现有 Python API 上的可选装配入口，不是新的 Agent 类型、执行流程或
Workflow DSL。配置加载后返回普通 `BasicAgent`；原来的构造函数、`with_*`、
`invoke/ainvoke/stream/astream` 和模块扩展方式均保持不变。

## 快速使用

如果希望先生成一个有独立启动入口的应用目录，使用
`buildagent --file agent.yaml --output ./my-agent`，再运行 `python my-agent/app.py`。
目录结构、资源打包和密钥处理见[应用生成指南](application_build.md)。

```bash
python -m pip install -e .
export OPENAI_API_KEY='your-key'
# 先将示例中的 your-model 改成服务实际提供的模型。
easyagent-run example/configuration/minimal.yaml --check
easyagent-run example/configuration/minimal.yaml --query "解释 Agent Harness"
```

未更新 console-script 安装时也可以运行：

```bash
python -m easyagent.configuration example/configuration/minimal.yaml --check
```

脚本仅执行一个请求，不自动批准权限 ASK，也不提供新的终端交互系统。
遇到审批或提问中断会返回非零退出码。需要审批恢复、流式 UI 或长期会话时，
使用下面的 Python 入口处理现有框架协议。

```python
from easyagent.configuration import load_agent, validate_agent

steps = validate_agent("./agent.yaml")  # 静态校验，不创建模型客户端或连接 MCP
agent = load_agent("./agent.yaml")
try:
    result = agent.invoke("分析当前项目")
    print(result)
    # 已配置 session 时，可按原有 API 显式保存，不会自动保存每轮：
    # agent.save_session()
finally:
    agent.close()
```

加载后仍可调用 `agent.with_memory(my_memory)` 等代码装配接口，重复安装遵循原有规则。
加载不生成或执行 Python 源文件。

## 配置约定

```yaml
version: 1
agent:
  name: project-assistant
  workspace: ./workspace
  config:
    tool_schema_mode: deferred
model:
  provider: openai
  model: your-model
  api_key: {env: OPENAI_API_KEY}
prompt:
  file: ./system.md
context:
  implementation: default
  config: {max_tokens: 32000}
tools:
  - name: filesystem
  - name: bash
    enabled: false
skills:
  directories: [./skills]
permissions:
  implementation: rule_based
  config: {approval_mode: ask_for_approval}
memory:
  enabled: false
```

- 未出现的可选能力不安装；出现的能力默认开启，`enabled: false` 关闭。
- Model 必须声明。Prompt、Context、Executor 缺省时沿用 BasicAgent 的默认实现，
  但不能显式关闭。空映射 `plan: {}` 表示开启；`plan: null` 等同省略，
  不能用 `plan: true` 代替配置对象。
- 可替换模块使用 `implementation` 与实现专属的 `config`；缺省实现名为 `default`。
- Model 支持上面的便捷写法，也支持
  `model: {implementation: default, config: {provider: openai, model: your-model}}`。
- `prompt.text` 与 `prompt.file` 二选一。默认 composer 配置
  `include_defaults: false` 时仍保留显式提供的身份提示词。
- `agent.config` 接受现有 Config 字段；workspace 和 allowed_roots 必须在
  agent 顶层设置，避免两处配置互相覆盖。显式 `agent.config.temperature`
  优先于 model.temperature。

所有显式配置路径以 **配置文件目录** 为基准，不随进程启动目录变化：
workspace、allowed_roots、prompt 文件、Skill 目录、SQLite 文件、权限目录、
multi-agent/worktree 存储目录和 Bash 沙箱路径均如此。省略工具 workspace 时
继承 Agent workspace；改变工具工作区并不自动放宽 allowed_roots。
workspace 和输入目录必须已存在；输出数据库/存储目录在实际装配时按实现创建。
命令名、域名、URL、提示词和任意扩展字符串不会被当作路径自动转换。

## 内置支持清单

| 配置项 | 实现与主要选项 |
| --- | --- |
| model | default：EasyLLM 的 provider/model/api_key/base_url/temperature/max_tokens/timeout/default_headers；provider 专属值放 provider_options |
| prompt | default：include_defaults、include_module_blocks |
| context | default：max_tokens；其他上下文算法可注册自定义工厂 |
| executor | default：现有 DefaultAgentExecutor |
| permissions | default / rule_based：approval_mode、policy_directory、policy_paths |
| memory | default：MemoryManage 的工作记忆实现；user_id、max_capacity、max_working_token、importance_threshold、decay_factor、batch_size |
| plan | default：allowed_actions、register_tools、enter_message、exit_message |
| tasks | default：TaskService + InMemoryTaskStore |
| session | default：InMemorySessionStore；sqlite：path |
| observability | default：内存记录；sqlite：path |
| hooks / callbacks | default：现有默认实现 |
| codeintel | default：现有 LSP 默认实现，可用性仍取决于安装的语言服务器 |
| worktree | default：git_binary、storage_dir，需要工作区属于 Git 仓库 |
| multi_agent | default：storage_dir、max_background_tasks；最后装配以获取父 Agent 已安装的能力 |

默认 Memory 不是自动配置了所有数据库的万能记忆系统。不自动创建向量服务、
Neo4j 或下载嵌入模型；复杂记忆组合通过自定义 Memory 工厂提供。依赖按实际
使用的实现安装，默认工作记忆需要 NumPy 等该实现的依赖。

Plan 的开启表示安装能力，不是立即进入 plan 模式。需要硬权限拦截时必须同时
安装权限引擎；框架允许只安装 Plan 提示能力，配置层不偷偷开启 Permission。

内置工具加载项：
`filesystem`（FileRead/List/Glob/Grep）、`file_write`、`file_edit`、
`notebook_edit`、`bash`（Bash/TaskOutput/TaskStop）、
`calculator`、`web_search`、`web_fetch`、`ask_user_question`。
有工作区的工具支持 workspace_root、allowed_roots、cwd、expose_in_deferred。
Bash 还支持 shell、command_timeout_ms、max_background_tasks、max_output_chars、
sandbox。沙箱配置复用已有语义；`--check` 不运行 Bubblewrap 或承诺 OS 隔离可用。

## Skill 与 MCP

Skill 通过 `skills.directories` 加载，使用已有 SKILL.md 解析与装配机制。
Skill、MCP 和 ToolRegistry 不提供替换整个管理器的配置选项。

```yaml
mcp:
  servers:
    - name: documentation
      source_type: url
      source: https://your-server.example/mcp
      tool_prefix: docs_
    - name: local-service
      source_type: command
      source: python
      args: [/absolute/path/to/server.py]
      env:
        SERVICE_TOKEN: {env: SERVICE_TOKEN}
```

source_type 可为 command、file、url。只有 file 类型的 source 按配置目录解析；
命令参数保持原样，文件参数请使用绝对路径。此初版不提供每个 MCP Server 的 cwd
配置或全套 transport/auth 选项。加载 MCP 会启动进程或建立连接；只加载可信配置。

## 注册自己的实现

不让模块实现 from_yaml，也不让配置文件指定任意 Python 导入路径。
扩展者显式注册名称、配置类型和构造函数：

```python
from easyagent.configuration import Options, default_registry, load_agent

class CompanyPolicyOptions(Options):
    policy_directory: str

registry = default_registry()
registry.register_permission(
    "company",
    config_model=CompanyPolicyOptions,
    factory=lambda options, context: CompanyPermissionEngine(
        policy_directory=context.paths.resolve(options.policy_directory),
    ),
)
agent = load_agent("./agent.yaml", registry=registry)
```

其中 CompanyPermissionEngine 是应用已实现的 BasePermissionEngine 子类。
配置选择 `permissions: {implementation: company, config: {...}}`。

同样支持 register_model、register_memory、register_context、register_executor、
register_prompt、register_multi_agent 和 register_tools。模块工厂得到
类型化选项以及 BuildContext（路径信息、已创建的 LLM），**没有整个 Agent**。
模型工厂只得到选项和路径信息。工厂返回正常模块，加载器通过现有 with_* 安装。
可以用闭包注入已有依赖；返回的对象应为本次 Agent 新建且可由它拥有。

需要额外静态校验/路径转换的工厂可提供 prepare(options, paths)，必须是只读的。
其他能力可以使用 register_installer(kind, name, config_model=..., install=...)：
这是高级装配回调，可以操作 Agent 的公开接口，但不应把 Agent 传入模块内部。
不支持任意新顶层 YAML 字段；新增可替换实现使用已有能力类别。

register_tools 接收注册函数 (registry, options, context)，可贡献多个工具；
可声明 tool_names 以提前检查冲突。未声明或远端动态提供的工具名，在实际装配时
仍会检查重名，不会静默覆盖。注册表默认拒绝同名工厂覆盖，各次 default_registry()
互相独立。

## 校验、安全与资源

- 使用 YAML 安全加载器，拒绝重复键、对象执行标签和循环引用。
- 未知字段、未知实现、错误参数、显式缺失的环境变量及输入路径会报错。
- `{env: NAME}` 是显式引用；不做任意模板表达式求值。关闭的能力不要求其凭据。
- 错误不输出配置值、密钥或外部异常正文；可使用自定义工厂内部的安全日志排查问题。
- `validate_agent`/`--check` 只做静态校验，返回选中的装配项；不连接模型/MCP、
  创建数据库、运行工具或保证外部服务可用。没有显式 env 引用时，模型原有凭据
  默认逻辑仍可能在实际创建客户端时才发现错误。
- 装配失败关闭已创建的 Agent；未安装成功的新组件执行可用的 close()。
  自定义构造函数在返回对象之前失败时，必须自行释放它已经创建的资源。
  高级 installer 也负责自己未成功交给 Agent 的资源。
  工厂可通过 BuildContext.on_failure(cleanup) 登记额外的失败回滚；
  成功返回后不会执行这些回滚，资源仍须交给具备正常关闭机制的组件管理。
- 成功后按既有 Agent 生命周期管理资源，调用者负责 close()/aclose()。
  配置层不承诺替现有工具新增进程回收、自动会话保存或审批 UI。

示例见 [minimal.yaml](../example/configuration/minimal.yaml) 和
[coding.yaml](../example/configuration/coding.yaml)。相关回归测试位于
`test/test_configuration.py`，使用离线模型和 MCP 替身，不调用收费模型。
