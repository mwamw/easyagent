# EasyAgent Skill 技能系统使用指南

> **版本**: 1.0.0 | **最后更新**: 2026-03-27

## 目录

- [概述](#概述)
- [核心架构](#核心架构)
- [快速开始](#快速开始)
- [BaseSkill 基类详解](#baseskill-基类详解)
- [SkillManager 管理器](#skillmanager-管理器)
- [SkillRegistry 全局注册中心](#skillregistry-全局注册中心)
- [零代码 Skill 定义](#零代码-skill-定义)
- [内置 Skill](#内置-skill)
- [Agent 集成](#agent-集成)
- [高级用法](#高级用法)
- [API 参考](#api-参考)

---

## 概述

Skill（技能）系统是 EasyAgent 框架的**模块化能力注入架构**。它将 Agent 的能力抽象为可插拔的「技能包」，每个 Skill 封装了：

| 组成部分 | 说明 |
|----------|------|
| **Tools** | 一组相关工具（注入到 Agent 的 ToolRegistry） |
| **Prompt** | 系统提示词片段（指导 LLM 如何使用该能力） |
| **ContextSource** | 上下文来源（可选，注入到 ContextManager） |
| **生命周期钩子** | activate / deactivate / before_invoke / after_invoke |

### 设计理念

- **按需加载**：Agent 只挂载当前需要的 Skill，避免上下文窗口膨胀
- **运行时动态切换**：可以在运行期间随时激活/停用 Skill
- **零代码定义**：支持通过 YAML 或 Markdown 文件声明 Skill，无需编写 Python 代码
- **依赖管理**：Skill 之间可以声明依赖关系，自动级联激活

---

## 核心架构

```
┌─────────────────────────────────────────────────┐
│                    BaseAgent                     │
│  ┌─────────────────────────────────────────────┐ │
│  │              SkillManager                    │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │ │
│  │  │ Skill A │ │ Skill B │ │ Skill C │ ...   │ │
│  │  │ (active)│ │ (active)│ │(inactive│       │ │
│  │  └──┬──┬───┘ └──┬──┬───┘ └─────────┘       │ │
│  │     │  │        │  │                         │ │
│  │  Tools Prompt Tools Prompt                   │ │
│  └─────┼──┼────────┼──┼────────────────────────┘ │
│        │  │        │  │                           │
│  ┌─────▼──┼────────▼──┼──┐  ┌──────────────────┐ │
│  │   ToolRegistry        │  │  System Prompt    │ │
│  └───────────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              SkillRegistry (全局单例)              │
│  class 注册 / factory 注册 / 目录自动发现           │
│  .py / .yaml / .yml / .md                        │
└─────────────────────────────────────────────────┘
```

**关键组件**：

| 组件 | 职责 |
|------|------|
| `BaseSkill` | Skill 抽象基类，定义标准接口 |
| `SkillConfig` | Skill 配置数据类（Pydantic Model） |
| `SkillManager` | 管理 Skill 生命周期，注入工具和 prompt |
| `SkillRegistry` | 全局 Skill 类/工厂注册中心，支持自动发现 |
| `YAMLSkillLoader` | 从 YAML 文件加载 Skill |
| `MarkdownSkillLoader` | 从 Markdown 文件加载 Skill |

---

## 快速开始

### 1. 使用内置 Skill

```python
from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM
from skill.builtin.calculator_skill import CalculatorSkill
from skill.builtin.web_search_skill import WebSearchSkill

llm = EasyLLM(provider="openai", model="gpt-4")

agent = BasicAgent(name="my_agent", llm=llm)
agent.with_skill(CalculatorSkill())     # 添加计算能力
agent.with_skill(WebSearchSkill())      # 添加搜索能力

result = agent.invoke("计算 2^10 + 3^5 的结果")
```

### 2. 自定义 Skill

```python
from skill.base import BaseSkill, SkillConfig
from Tool.BaseTool import Tool
from pydantic import BaseModel, Field

# 1. 定义工具
class TranslateParams(BaseModel):
    text: str = Field(description="要翻译的文本")
    target_lang: str = Field(default="en", description="目标语言")

class TranslateTool(Tool):
    def __init__(self):
        super().__init__("translate_tool", "将文本翻译为目标语言", TranslateParams)

    def run(self, parameters: dict) -> str:
        # 实际翻译逻辑
        return f"Translated: {parameters['text']}"

# 2. 定义 Skill
class TranslateSkill(BaseSkill):
    def __init__(self):
        config = SkillConfig(
            name="translate",
            description="多语言翻译技能",
            version="1.0.0",
            tags=["translate", "language", "i18n"],
            priority=5,
        )
        super().__init__(config)

    def get_tools(self) -> list:
        return [TranslateTool()]

    def get_prompt(self) -> str:
        return """## 翻译能力
你具备多语言翻译能力。当用户要求翻译时，请使用 translate_tool 工具。
- 支持中英日韩等多种语言
- 可以自动识别源语言
"""

# 3. 使用
agent.with_skill(TranslateSkill())
```

### 3. 零代码 YAML Skill

创建 `skills/my_skill.yaml`：

```yaml
name: code_reviewer
description: "代码审查技能"
version: "1.0"
tags: [code, review]
priority: 5
tools:
  - builtin: calculator  # 引用内置工具
prompt: |
  ## 代码审查能力
  你具备代码审查能力，能够分析代码质量、发现潜在 bug。
  - 检查代码风格
  - 识别常见错误模式
  - 给出改进建议
```

加载并使用：

```python
from skill.yaml_loader import YAMLSkillLoader

skill = YAMLSkillLoader.load("skills/my_skill.yaml")
agent.with_skill(skill)
```

---

## BaseSkill 基类详解

### SkillConfig 配置

`SkillConfig` 是 Pydantic BaseModel，定义 Skill 的元信息：

```python
from skill.base import SkillConfig

config = SkillConfig(
    name="my_skill",              # 唯一标识名称（必填）
    description="技能描述",        # 功能描述
    version="1.0.0",              # 版本号
    tags=["tag1", "tag2"],        # 标签（用于搜索/分类）
    priority=5,                   # 优先级（数值越大，prompt 越靠前）
    auto_activate=True,           # 注册到 SkillManager 时是否自动激活
    dependencies=["other_skill"], # 依赖的其他 Skill 名称
    extra={"key": "value"},       # 自定义扩展配置
)
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | **必填** | Skill 唯一标识 |
| `description` | `str` | `""` | 功能描述 |
| `version` | `str` | `"1.0.0"` | 版本号 |
| `tags` | `List[str]` | `[]` | 分类标签 |
| `priority` | `int` | `0` | 优先级，越大越靠前 |
| `auto_activate` | `bool` | `True` | 注册时自动激活 |
| `dependencies` | `List[str]` | `[]` | 依赖列表 |
| `extra` | `Dict` | `{}` | 自定义配置 |

### 核心抽象方法

每个自定义 Skill **必须实现**以下两个方法：

```python
class MySkill(BaseSkill):
    def get_tools(self) -> List[Tool]:
        """返回此 Skill 提供的所有 Tool 实例"""
        return [MyTool()]

    def get_prompt(self) -> str:
        """返回注入到 system prompt 的指导文本"""
        return "## 使用指南\n..."
```

### 可选方法

```python
class MySkill(BaseSkill):
    def get_context_sources(self) -> List[BaseContextSource]:
        """返回 ContextSource 列表，注册到 ContextManager（可选）"""
        return []

    def on_activate(self, agent) -> None:
        """激活时调用（初始化资源、建立连接等）"""
        pass

    def on_deactivate(self, agent) -> None:
        """停用时调用（释放资源、关闭连接等）"""
        pass

    def on_before_invoke(self, query: str) -> str:
        """Agent invoke 前调用，可修改 query（预处理）"""
        return query

    def on_after_invoke(self, query: str, response: str) -> str:
        """Agent invoke 后调用，可修改 response（后处理）"""
        return response
```

### 工具属性

```python
skill = MySkill()
skill.name              # Skill 名称
skill.is_active         # 是否激活
skill.tags              # 标签列表
skill.priority          # 优先级
skill.get_tool_names()  # 获取所有工具名称列表
skill.to_dict()         # 序列化为字典
```

---

## SkillManager 管理器

`SkillManager` 是 Skill 的生命周期管理中心，通常由 `BaseAgent` 自动创建和使用。

### 基本用法

```python
from skill.manager import SkillManager

manager = SkillManager()
manager.bind_agent(agent)  # 绑定 Agent（BaseAgent 构造时自动完成）
```

### 注册与注销

```python
# 注册（auto_activate=True 时自动激活）
manager.register(my_skill)

# 链式注册
manager.register(skill_a).register(skill_b)

# 注销（先停用再移除）
manager.unregister("my_skill")
```

> **注意**：注册同名 Skill 会抛出 `ValueError`，需要先注销再重新注册。

### 激活与停用

```python
# 手动激活（auto_activate=False 的 Skill）
manager.activate("my_skill")

# 停用
manager.deactivate("my_skill")
```

**激活操作流程**：
1. 检查依赖 → 自动激活未激活的依赖
2. 将 Skill 的 Tools 注册到 Agent 的 `ToolRegistry`
3. 将 Skill 的 ContextSources 注册到 Agent 的 `ContextManager`
4. 调用 `skill.on_activate(agent)` 生命周期钩子
5. 标记为激活状态

**停用操作流程**：
1. 检查依赖关系 → 先停用依赖此 Skill 的其他 Skill
2. 从 `ToolRegistry` 移除 Skill 的所有 Tools
3. 调用 `skill.on_deactivate(agent)` 生命周期钩子
4. 标记为停用状态

### 查询方法

```python
manager.has_skill("name")       # 是否已注册
manager.is_active("name")       # 是否已激活
manager.get_skill("name")       # 获取 Skill 实例
manager.get_active_skills()     # 获取所有激活的 Skill（按 priority 降序）
manager.get_all_skills()        # 获取所有已注册的 Skill
manager.list_skills()           # 返回所有 Skill 的信息列表

manager.skill_count             # 已注册 Skill 数量
manager.active_count            # 已激活 Skill 数量
manager.active_skill_names      # 已激活 Skill 名称列表
```

### Prompt 聚合

```python
prompt = manager.build_skills_prompt()
```

将所有激活 Skill 的 prompt 按 **priority 降序** 拼接为单个文本。`BasicAgent.get_enhanced_prompt()` 内部会自动调用此方法。

### 拦截链

```python
# 在 Agent invoke 之前/之后由 SkillManager 代理调用
query = manager.on_before_invoke(query)         # 按 priority 降序链式修改
response = manager.on_after_invoke(query, response)  # 按 priority 降序链式修改
```

在 `BasicAgent.invoke()` 中已自动集成调用。

---

## SkillRegistry 全局注册中心

`SkillRegistry` 是一个**全局单例**，用于 Skill 类的注册、工厂管理和自动发现。

### 获取实例

```python
from skill.registry import SkillRegistry

registry = SkillRegistry.instance()   # 获取全局单例
SkillRegistry.reset()                 # 重置单例（仅测试用）
```

### 注册 Skill 类

```python
from skill.builtin.calculator_skill import CalculatorSkill

# 方式 1：手动注册
registry.register_class(CalculatorSkill)           # 自动推断名称 "calculator"
registry.register_class(CalculatorSkill, "calc")   # 指定名称

# 方式 2：装饰器注册
@registry.skill("my_skill")
class MySkill(BaseSkill):
    ...

# 方式 3：工厂函数注册
def create_skill(**kwargs):
    return MySkill(**kwargs)

registry.register_factory("my_skill", create_skill)
```

### 创建 Skill 实例

```python
skill = registry.create("calculator")              # 按名称创建
skill = registry.create("my_skill", api_key="xxx")  # 带参数创建
```

### 自动发现

从目录自动扫描 `.py`、`.yaml`、`.yml`、`.md` 文件并注册：

```python
registered_names = registry.discover_from_directory("./skills/")
print(registered_names)  # ["skill_a", "skill_b", ...]
```

**发现规则**：

| 文件类型 | 处理方式 |
|----------|---------|
| `.py` | 导入模块，扫描所有 `BaseSkill` 子类，注册为 class |
| `.yaml` / `.yml` | 使用 `YAMLSkillLoader.load()` 解析，注册为 factory |
| `.md` | 使用 `MarkdownSkillLoader.load()` 解析，注册为 factory |

> 以 `_` 开头的 Python 文件会被跳过。

### 查询

```python
registry.has("calculator")            # 是否已注册
registry.list_available_names()       # 所有注册名称列表
registry.list_available()             # 带元信息的详细列表
```

### 类名自动转换规则

注册时不显式指定名称的话，会按以下规则从类名推断：

| 类名 | 推断名称 |
|------|---------|
| `CalculatorSkill` | `calculator` |
| `MyWebSearchSkill` | `my_web_search` |
| `MCPSkill` | `m_c_p` |

规则：移除 `Skill` 后缀 → CamelCase 转 snake_case。

---

## 零代码 Skill 定义

### YAML 格式

```yaml
# skills/research.yaml
name: research                  # 必填
description: "研究技能"          # 可选
version: "1.0"                  # 可选，默认 "1.0.0"
tags: [research, analysis]      # 可选
priority: 5                     # 可选，默认 0
auto_activate: true             # 可选，默认 true
dependencies: []                # 可选

# 工具引用（可选）
tools:
  - builtin: web_search         # 字典格式，引用内置工具
  - builtin: calculator
  - calculator                  # 简写格式（等价于 {builtin: calculator}）

# prompt 支持多行文本
prompt: |
  ## 研究能力
  你具备深度研究和分析能力。
  - 使用搜索工具获取最新信息
  - 使用计算器进行数据分析

# 自定义配置（可选）
config:
  max_results: 10
  language: zh
```

**加载方式**：

```python
from skill.yaml_loader import YAMLSkillLoader

# 加载单个文件
skill = YAMLSkillLoader.load("skills/research.yaml")

# 批量加载目录
skills = YAMLSkillLoader.load_directory("skills/")
```

### Markdown 格式

Markdown Skill 使用 **YAML frontmatter** 定义配置，**Markdown 正文** 作为 prompt：

```markdown
---
name: code_assistant
description: "代码辅助技能"
version: "2.0"
tags: [code, programming]
priority: 8
tools:
  - builtin: calculator
---

## 代码辅助能力

你具备代码分析和辅助编程的能力。

### 使用场景
- 当用户提供代码段时，进行分析和优化建议
- 使用 calculator 进行数学计算
- 帮助调试代码问题

### 注意事项
- 总是解释你的推理过程
- 提供可运行的代码示例
```

**加载方式**：

```python
from skill.yaml_loader import MarkdownSkillLoader

# 加载单个文件
skill = MarkdownSkillLoader.load("skills/code_assistant.md")

# 批量加载目录
skills = MarkdownSkillLoader.load_directory("skills/")
```

> **Tip**：如果 frontmatter 中没有 `name` 字段，会自动从文件名推断（去掉扩展名）。

### 当前支持的内置工具名称

在 YAML/Markdown 的 `tools` 中可引用的内置工具：

| 名称 | 对应工具类 |
|------|-----------|
| `calculator` | `Tool.builtin.calculator.CalculatorTool` |
| `web_search` | `Tool.builtin.search.WebSearchTool` |

---

## 内置 Skill

### CalculatorSkill

数学计算技能，封装 `CalculatorTool`。

```python
from skill.builtin.calculator_skill import CalculatorSkill

skill = CalculatorSkill()
# name="calculator", priority=3, tags=["math", "calculation", "compute"]
agent.with_skill(skill)
```

**提供工具**: `CalculatorTool`  
**适用场景**: 复杂数学运算、统计计算、单位换算

---

### WebSearchSkill

联网搜索技能，封装 `WebSearchTool`。

```python
from skill.builtin.web_search_skill import WebSearchSkill

skill = WebSearchSkill(api_key="your-api-key")
# name="web_search", priority=5, tags=["search", "web", "real-time", "information"]
agent.with_skill(skill)
```

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `api_key` | `Optional[str]` | 搜索 API 密钥 |
| `**kwargs` | | 传递给 `WebSearchTool` 的额外参数 |

**提供工具**: `WebSearchTool`  
**适用场景**: 实时信息查询、事实验证、新闻搜索

---

### MemorySkill

V2 记忆系统技能，封装完整的记忆能力。

```python
from memory.V2.MemoryManage import MemoryManage
from skill.builtin.memory_skill import MemorySkill

mm = MemoryManage(config, user_id="user1", ...)
skill = MemorySkill(
    memory_manage=mm,
    session_id="session_001",           # 可选，默认自动生成
    include_context_source=True,        # 可选，是否提供 MemoryContextSource
)
# name="memory", priority=10, tags=["memory", "knowledge", "recall", "working_memory"]
agent.with_skill(skill)
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `memory_manage` | `MemoryManage` | **必填** | V2 记忆管理器实例 |
| `session_id` | `Optional[str]` | 自动生成 | 会话 ID |
| `include_context_source` | `bool` | `True` | 是否包含 MemoryContextSource |

**提供的 6 个工具**:
| 工具 | 功能 |
|------|------|
| `AddMemoryTool` | 添加记忆 |
| `SearchMemoryTool` | 搜索记忆 |
| `GetMemoryTool` | 获取指定记忆 |
| `UpdateMemoryTool` | 更新记忆 |
| `RemoveMemoryTool` | 删除记忆 |
| `MemoryMaintenanceTool` | 记忆维护（清理过期） |

**Prompt 特性**:
- 自动注入记忆系统使用指南
- 自动注入 Working Memory 便签本内容（实时）
- 当 `include_context_source=True` 时，还会提供 `MemoryContextSource`

---

### MCPSkill

MCP（Model Context Protocol）远程工具技能，动态发现并封装 MCP 服务器的远程工具。

```python
from skill.builtin.mcp_skill import MCPSkill

skill = MCPSkill(
    server_source="path/to/mcp_server.py",
    transport_type="stdio",
    tool_prefix="mcp_",           # 工具名前缀（可选）
    auto_connect=True,            # 自动连接（可选）
    skill_name="my_mcp_server",   # 自定义名称（可选）
)
agent.with_skill(skill)
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `server_source` | `Any` | **必填** | MCP 服务器源标识 |
| `server_args` | `Optional[List[str]]` | `None` | 启动参数 |
| `transport_type` | `Optional[str]` | `None` | 传输类型 |
| `env` | `Optional[Dict]` | `None` | 环境变量 |
| `tool_prefix` | `str` | `""` | 工具名前缀 |
| `auto_connect` | `bool` | `True` | 是否自动连接 |
| `skill_name` | `Optional[str]` | 自动生成 | Skill 名称 |

**特性**:
- 自动连接 MCP 服务器并发现远程工具
- 停用时自动关闭 MCP 连接（`on_deactivate`）
- 工具名自动添加前缀避免冲突

---

## Agent 集成

### with_skill — 添加 Skill

```python
# 链式调用
agent = BasicAgent(name="demo", llm=llm)
agent.with_skill(CalculatorSkill()).with_skill(WebSearchSkill())
```

`with_skill()` 内部会自动：
1. 如果 Agent 没有 `ToolRegistry`，自动创建一个
2. 调用 `skill_manager.register(skill)` 注册 Skill
3. 如果 `auto_activate=True`（默认），自动激活

### with_memory — 记忆系统便捷方法

```python
agent.with_memory(memory_manage)
```

内部会自动创建 `MemorySkill` 并注册，无需手动创建。如果已经通过 `with_skill` 注册了 `MemorySkill`，则跳过自动注册。

### 运行时管理

```python
# 停用 Skill（不移除注册）
agent.deactivate_skill("calculator")

# 重新激活
agent.activate_skill("calculator")

# 完全移除（注销）
agent.remove_skill("calculator")
```

### Prompt 自动注入

在 `BasicAgent.get_enhanced_prompt()` 中，Skill 的 prompt 会自动拼接到系统提示词末尾：

```python
# BasicAgent.get_enhanced_prompt() 内部逻辑：
enhanced_prompt = f"...基础 prompt..."
enhanced_prompt += self._build_memory_prompt()   # 记忆系统 prompt
enhanced_prompt += self._build_skills_prompt()   # 所有激活 Skill 的 prompt
```

### invoke 拦截链

在 `BasicAgent.invoke()` 中自动集成：

```python
# invoke() 内部逻辑：
query = self.skill_manager.on_before_invoke(query)    # 前置拦截
# ... LLM 调用 ...
response = self.skill_manager.on_after_invoke(query, response)  # 后置拦截
```

---

## 高级用法

### 1. 构建前/后处理拦截器

利用 `on_before_invoke` 和 `on_after_invoke` 实现输入预处理和输出后处理：

```python
class AutoTranslateSkill(BaseSkill):
    """自动将非中文输入翻译为中文，再将中文输出翻译为原语言"""

    def __init__(self):
        config = SkillConfig(name="auto_translate", priority=100)  # 高优先级
        super().__init__(config)

    def get_tools(self) -> list:
        return []

    def get_prompt(self) -> str:
        return ""

    def on_before_invoke(self, query: str) -> str:
        # 检测语言，如果非中文则翻译
        if not self._is_chinese(query):
            return self._translate_to_chinese(query)
        return query

    def on_after_invoke(self, query: str, response: str) -> str:
        # 将响应翻译回原语言
        return self._translate_back(response)
```

### 2. 依赖管理

```python
class AdvancedAnalysisSkill(BaseSkill):
    def __init__(self):
        config = SkillConfig(
            name="advanced_analysis",
            dependencies=["calculator", "web_search"],  # 声明依赖
        )
        super().__init__(config)
    ...

# 注册顺序不影响依赖解析
manager.register(AdvancedAnalysisSkill())   # 自动检查依赖
manager.register(CalculatorSkill())
manager.register(WebSearchSkill())

# 激活时自动级联激活依赖
manager.activate("advanced_analysis")
# → 自动激活 "calculator" 和 "web_search"
```

### 3. 延迟激活

```python
skill = CalculatorSkill()
skill.config.auto_activate = False  # 注册时不自动激活

agent.with_skill(skill)             # 只注册，不激活
# ... 后续需要时 ...
agent.activate_skill("calculator")  # 手动激活
```

### 4. 动态切换 Skill

```python
# 场景：根据用户意图动态切换
if user_wants_coding:
    agent.activate_skill("code_helper")
    agent.deactivate_skill("web_search")
elif user_wants_research:
    agent.deactivate_skill("code_helper")
    agent.activate_skill("web_search")
```

### 5. 自动发现并注册

```python
# 项目目录结构:
# skills/
#   ├── custom_skill.py      → 自动发现其中的 BaseSkill 子类
#   ├── research.yaml         → 以 YAML 格式定义的 Skill
#   └── code_helper.md        → 以 Markdown 格式定义的 Skill

registry = SkillRegistry.instance()
names = registry.discover_from_directory("skills/")
print(names)  # ["custom", "research", "code_helper"]

# 按需创建并添加到 Agent
for name in names:
    skill = registry.create(name)
    agent.with_skill(skill)
```

### 6. 提供 ContextSource

如果 Skill 需要向 `ContextManager` 注入额外上下文，重写 `get_context_sources()`：

```python
class MySkill(BaseSkill):
    def get_context_sources(self) -> list:
        from context.source.base import BaseContextSource
        return [MyCustomContextSource()]
```

当 Skill 激活时，`SkillManager` 会自动将这些 `ContextSource` 注册到 Agent 的 `ContextManager`。

---

## API 参考

### skill.base

| 类 | 说明 |
|------|------|
| `SkillConfig` | Skill 配置数据类（Pydantic BaseModel） |
| `BaseSkill` | Skill 抽象基类 |

**BaseSkill 抽象方法**:
| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `get_tools()` | `List[Tool]` | **必须实现** — 返回工具列表 |
| `get_prompt()` | `str` | **必须实现** — 返回 prompt 片段 |

**BaseSkill 可选方法**:
| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `get_context_sources()` | `List[BaseContextSource]` | 返回上下文来源列表 |
| `on_activate(agent)` | `None` | 激活回调 |
| `on_deactivate(agent)` | `None` | 停用回调 |
| `on_before_invoke(query)` | `str` | invoke 前拦截 |
| `on_after_invoke(query, response)` | `str` | invoke 后拦截 |
| `get_tool_names()` | `List[str]` | 获取工具名称列表 |
| `to_dict()` | `Dict` | 序列化为字典 |

**BaseSkill 属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | Skill 名称 |
| `is_active` | `bool` | 是否处于激活状态 |
| `tags` | `List[str]` | 标签列表 |
| `priority` | `int` | 优先级 |
| `config` | `SkillConfig` | 配置对象 |

---

### skill.manager

| 类 | 说明 |
|------|------|
| `SkillManager` | Skill 管理器 |

**SkillManager 方法**:
| 方法 | 说明 |
|------|------|
| `bind_agent(agent)` | 绑定 Agent 实例 |
| `register(skill) → self` | 注册 Skill（支持链式调用） |
| `unregister(name)` | 注销 Skill |
| `activate(name)` | 激活 Skill |
| `deactivate(name)` | 停用 Skill |
| `get_skill(name) → BaseSkill` | 获取 Skill 实例 |
| `get_active_skills() → List` | 获取激活的 Skill（按 priority 降序） |
| `get_all_skills() → List` | 获取所有已注册的 Skill |
| `has_skill(name) → bool` | 检查是否已注册 |
| `is_active(name) → bool` | 检查是否已激活 |
| `list_skills() → List[Dict]` | 所有 Skill 信息列表 |
| `build_skills_prompt() → str` | 聚合 prompt |
| `on_before_invoke(query) → str` | 代理前置拦截 |
| `on_after_invoke(query, response) → str` | 代理后置拦截 |

**SkillManager 属性**:
| 属性 | 说明 |
|------|------|
| `skill_count` | 已注册数量 |
| `active_count` | 已激活数量 |
| `active_skill_names` | 已激活名称列表 |

---

### skill.registry

| 类 | 说明 |
|------|------|
| `SkillRegistry` | 全局 Skill 注册中心（单例） |

**SkillRegistry 方法**:
| 方法 | 说明 |
|------|------|
| `instance() → SkillRegistry` | 获取全局单例 |
| `reset()` | 重置单例 |
| `register_class(cls, name?)` | 注册 Skill 类 |
| `register_factory(name, factory)` | 注册工厂函数 |
| `skill(name?, **kwargs)` | 装饰器注册 |
| `create(name, **kwargs) → BaseSkill` | 创建实例 |
| `discover_from_directory(path) → List[str]` | 目录自动发现 |
| `has(name) → bool` | 检查是否注册 |
| `list_available_names() → List[str]` | 注册名称列表 |
| `list_available() → List[Dict]` | 详细信息列表 |

---

### skill.yaml_loader

| 类 | 说明 |
|------|------|
| `YAMLSkill` | YAML 定义的 Skill 实例 |
| `YAMLSkillLoader` | YAML 加载器 |
| `MarkdownSkill` | Markdown 定义的 Skill 实例 |
| `MarkdownSkillLoader` | Markdown 加载器 |

**Loader 方法**:
| 方法 | 说明 |
|------|------|
| `load(path) → Skill` | 加载单个文件 |
| `load_directory(path) → List[Skill]` | 加载目录下所有文件 |

---

### Agent API（skill 相关）

| 方法 | 说明 |
|------|------|
| `agent.with_skill(skill) → self` | 添加并激活 Skill（链式调用） |
| `agent.remove_skill(name)` | 移除 Skill |
| `agent.activate_skill(name)` | 激活 Skill |
| `agent.deactivate_skill(name)` | 停用 Skill |
| `agent.with_memory(mm) → self` | 便捷绑定记忆系统（内部自动创建 MemorySkill） |

---

## 常见问题

### Q: Skill 的工具名冲突怎么办？

如果两个 Skill 提供了同名工具，后注册的 Skill 中冲突的工具会被跳过（不会覆盖），并输出警告日志。

### Q: 停用 Skill 后工具会被移除吗？

是的，`SkillManager` 会精确记录每个 Skill 注入的工具名，停用时从 `ToolRegistry` 中移除。

### Q: 如何保证 Prompt 的顺序？

通过 `priority` 字段控制。`build_skills_prompt()` 按 priority **降序** 排列，数值越大的 Skill 的 prompt 越靠前。

### Q: 可以在 YAML/Markdown Skill 中引用自定义工具吗？

当前版本仅支持引用 `builtin` 内置工具（`calculator`、`web_search`）。如需使用自定义工具，请通过 Python 代码继承 `BaseSkill` 来实现。

### Q: MemorySkill 和 `_build_memory_prompt` 会冲突吗？

`MemorySkill` 提供了独立的记忆 prompt。`BaseAgent._build_memory_prompt()` 是旧的记忆 prompt 构建路径，两者可能存在 prompt 重复。推荐统一使用 `with_skill(MemorySkill(...))` 或 `with_memory()` 方式。
