# Phase G: SDK 收口与通用 Agent 能力整理

## 本阶段完成了什么

Phase G 现在已经完成了第一版 SDK 收口。重点不是重写内部目录，而是给外部项目提供稳定入口和安装边界。

这次的核心产物有四块：

1. 新增 `pyproject.toml`
   - 现在仓库已经有正式安装入口
   - 提供了 `mcp / rag / memory / dev` 四组 extras

2. 新增公共 SDK 门面包 `easyagent/`
   - `easyagent.__init__`
   - `easyagent.agents`
   - `easyagent.llms`
   - `easyagent.tools`
   - `easyagent.tasks`
   - `easyagent.permissions`
   - `easyagent.session`
   - `easyagent.runtime`
   - `easyagent.mcp`
   - `easyagent.hooks`
   - `easyagent.guardrails`
   - `easyagent.skills`
   - `easyagent.context`
   - `easyagent.codeintel`
   - `easyagent.rag`
   - `easyagent.memory`

3. 新增公共 API 文档与 example 索引
   - `docs/framework_api.md`
   - `example/README.md`

4. README 默认用法切到 SDK 入口
   - 文档里的推荐导入方式现在是 `from easyagent import ...`

## 现阶段框架的变换

在 Phase G 之前，EasyAgent 虽然功能已经很多，但对外仍然像“源码仓库”：

- 文档默认教你直接导入内部路径
- 没有统一的安装元数据
- 没有明确告诉外部项目哪些路径是稳定 API
- examples 也没有清晰区分框架示例和产品式示例

现在它已经开始具备 SDK 形态：

- 安装边界由 `pyproject.toml` 定义
- 公共入口统一收口到 `easyagent`
- 外部项目不需要再依赖 `agent/`、`core/`、`Tool/` 的内部布局
- 文档和示例开始按 SDK 视角组织

一句话说，框架从“内部结构可用”变成了“外部接入有明确边界”。

## 一个具体过程例子

现在一个外部项目想接 EasyAgent，不需要再这样写：

```python
from agent import BasicAgent
from core.llm import EasyLLM
from Tool.ToolRegistry import ToolRegistry
```

而是直接走稳定入口：

```python
from easyagent import BasicAgent, EasyLLM, ToolRegistry
from easyagent.mcp import register_mcp_tools
from easyagent.permissions import PermissionContext, PermissionRule, PermissionBehavior
```

这个区别很重要：

- 前者把外部项目绑死在内部目录结构上
- 后者把内部实现和公共 SDK 边界分开了

以后内部即使继续演进，只要 `easyagent.*` 保持兼容，上层产品就不需要跟着一起改目录导入。

## 本阶段新增的关键接口

- `easyagent`
- `easyagent.agents`
- `easyagent.llms`
- `easyagent.tools`
- `easyagent.tasks`
- `easyagent.permissions`
- `easyagent.session`
- `easyagent.runtime`
- `easyagent.mcp`
- `easyagent.hooks`
- `easyagent.guardrails`
- `easyagent.skills`
- `easyagent.context`
- `easyagent.codeintel`
- `easyagent.rag`
- `easyagent.memory`

## 一个真实 example

真实 example 已放在：

- `example/example_phaseg_sdk_release.py`

这个 example 用的就是你指定的真实 `EasyLLM(...)` 配置。它完整演示：

- 只使用 `easyagent` 公共 SDK 导入
- 创建 `ToolRegistry`
- 配置权限规则
- 构造 `BasicAgent`
- 保存并恢复 session
- 查看 restore report

这个 example 我没有执行，保留给后续手动调试。

## 本阶段验证

我跑过：

```bash
python -m pytest test/test_sdk_public_api.py -q
python -m pytest test/test_session_persistence.py -k 'basic_agent_restores_mode_permissions_and_current_task or close_returns_cleanup_report' -q
```

另外也做了语法检查：

```bash
python -m py_compile easyagent/__init__.py example/example_phaseg_sdk_release.py
```

## 下一步

按当前计划，Phase G 完成后，主线不再是“框架收口”，而是增强支线：

- codeintel 的离线索引与 workspace 缓存
- observability / metrics / benchmark
- 更细粒度的 provider 与跨语言优化
