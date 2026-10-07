# 从 YAML 生成可运行 Agent 应用

`buildagent` 把配置和明确引用的资源生成到一个新目录，提供可直接执行的 `app.py`。
它不调用模型，也不生成第二套 Agent Loop；运行时仍通过 AgentBuilder 创建普通 BasicAgent。

## 生成和启动

在 EasyAgent 仓库中更新安装（已有依赖的环境可加 `--no-deps --no-build-isolation`）：

```bash
conda activate S4Code
python -m pip install -e ".[ui]"

buildagent --file example/configuration/coding.yaml --output ./coding-app
```

未更新命令入口时，等价命令是 `python -m easyagent.application --file ...`。
省略 `--output` 时，在当前目录生成 `<配置文件名>-app`。

生成结果：

```text
coding-app/
├── app.py             # 可执行入口
├── agent.yaml         # 可继续编辑的配置
├── assets/            # Prompt、Skill、权限文件（按配置添加）
├── state/             # 新的应用本地运行数据
├── .env.example       # 只有变量名，不含原密钥值
├── requirements.txt
├── build.json         # 版本、环境变量及外部依赖说明
├── .gitignore
└── README.md
```

修改生成配置中的模型名称；参照 `.env.example` 填写应用目录下自己的 `.env`，然后：

```bash
python coding-app/app.py --check
python coding-app/app.py
python coding-app/app.py --query "介绍项目结构"
python coding-app/app.py --workspace /path/to/project --query "阅读代码"
```

不带参数时启动终端 UI；`--query` 执行一次请求；`--check` 验证配置和本地资源，不调用模型。
单次请求遇到 ASK 会退出并提示使用交互模式，绝不会自动批准。UI 显式关闭时使用 `--query`。
从任何 cwd 都可以用应用的绝对路径启动。

## 批量和预检查

```bash
buildagent --file "configs/*.yaml" --output ./apps
buildagent --file agent.yaml --check
```

支持 shell 展开后的多个文件，也支持引号中的 glob。批量生成到
`apps/<文件名>-app/`，先检查所有输入再写文件；不同目录的同名输入需分别构建。
现有输出目录一律拒绝覆盖，不提供隐式强制覆盖。

## 路径、密钥与部署边界

| 内容 | 生成时如何处理 |
| --- | --- |
| Prompt、Skill | 复制显式引用的文件/目录；Skill 脚本保留可执行权限 |
| 权限目录 | 复制 settings.json、settings.local.json 和显式 policy_paths |
| 工作区、allowed_roots、工具与沙箱路径 | 保留原路径的绝对指向，不复制整个项目 |
| 相对会话/观测/协作/Worktree 存储 | 改为应用内部 state 路径，不复制历史数据库 |
| 显式绝对存储路径 | 保留为外部依赖，迁移时自行调整 |
| 密钥 | 保留 env 引用；常见明文密钥字段转换成新的 env 引用 |
| MCP Server、命令参数、外部程序 | 不自动打包或安装，需单独部署 |

运行时只读取应用目录的 `.env`，进程环境变量优先；不搜索上级目录的 `.env`。
其中的值用于配置中的显式 `{env: NAME}` 引用，不修改进程全局环境，不在应用之间串用。
构建无需读取密钥值。Prompt/Skill/权限文件按可信资产复制，**这不是完整的秘密扫描器**，
分发前仍需检查文件正文。常见私密文件、数据库、缓存不打包；资产符号链接会被拒绝。
打包上限为 10000 个资产文件、合计 50 MiB。

`--workspace` 只覆盖 Agent 默认工作区，不改写显式的工具路径和 allowed_roots。
环境变量提供的路径应使用绝对路径；需要复制的资产路径必须在源 YAML 中写成字面路径。

生成的是 **Python 应用目录，不是包含解释器和所有依赖的独立二进制**。当前环境可以直接运行；
其他机器需要安装包含 `easyagent.application` 的相同 EasyAgent 源码及所用模块的可选依赖。
requirements.txt 记录依赖，不保证这一分支的功能已经发布到包索引。

当前生成器接受 `.yaml` / `.yml`，使用内置注册表；不会序列化自定义 Python 工厂。
自定义模块需要开发者提供注册代码和启动器；`.ea` 仍通过 `easyagent-lang run` 运行。

## Python 入口

```python
from easyagent.application import build_application, create_application_agent

folder = build_application("agent.yaml", output="./my-agent")
agent = create_application_agent(folder)
try:
    print(agent.invoke("你好"))
finally:
    agent.close()
```

开发者可向 `create_application_agent` / `run_application` 显式传入配置注册表，
模块本身及原有 Python、YAML、DSL 的调用方式不变。
