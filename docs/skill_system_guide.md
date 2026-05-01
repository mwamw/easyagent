# Skill System Guide

Skill 是 EasyAgent 的“能力包”机制。它的目标不是简单地往系统提示词里追加一段文本，而是把一组会一起出现的能力封装成统一单元：

- prompt / instruction
- tool
- context source
- 生命周期钩子
- 暴露方式与缓存生命周期

如果你在做一个面向产品的 Agent，Skill 往往是“产品能力模块”的最佳承载层。比如：

- `code_review`
- `web_research`
- `frontend_design`
- `memory`
- `mcp`

相关文档：

- [Tool System Guide](./tool_system_guide.md)
- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Prompt System Guide](./prompt_system_guide.md)
- [Runtime Reminders Guide](./runtime_reminders_guide.md)

## 1. 核心对象

Skill 系统最重要的对象有：

- `BaseSkill`
  - Skill 的抽象基类。你自定义 Skill 时直接继承它。
- `SkillConfig`
  - Skill 的结构化配置，决定名字、描述、暴露模式、缓存生命周期等。
- `SkillManifest`
  - Skill 暴露给模型和注册表的统一元数据视图。
- `SkillRegistry`
  - 用来保存“有哪些 Skill 可用”的目录。
- `SkillManager`
  - 运行时控制器，负责注册、激活、停用、挂载工具、注入上下文。
- `SkillDiscoveryTool`
  - 给模型列出当前有哪些 Skill。
- `SkillTool`
  - 按需调用 Skill 正文和临时能力。
- `LoadSkillTool`
  - 把 Skill 长期挂载到当前 Agent。
- `UnloadSkillTool`
  - 从当前 Agent 卸载长期挂载的 Skill。

## 2. Skill 解决什么问题

Skill 适合解决下面这些场景：

1. 你希望把一组相关能力打包，而不是零散地注册很多 Tool。
2. 你需要“什么时候该使用这组能力”的提示，而不仅仅是工具 schema。
3. 你希望某些能力是默认常驻的，另一些能力按需展开。
4. 你需要把 prompt、tool、context source、生命周期钩子绑定在一起。

几个对比例子：

- 只想给模型一条规则：
  - 用 `PromptBlock` 或 [Prompt System Guide](./prompt_system_guide.md) 的 system block。
- 只想补一段当前请求的运行时信息：
  - 用 [Runtime Reminders Guide](./runtime_reminders_guide.md)。
- 需要模型主动调用、带结构化参数、可授权：
  - 用 Tool。
- 需要“一整组能力包”：
  - 用 Skill。

## 3. `SkillConfig` 字段逐项解释

`SkillConfig` 是你定义 Skill 时最重要的配置对象。当前主要字段如下：

- `name`
  - Skill 的唯一名字。模型看到的 listing、registry 查找、激活/停用都依赖它。
- `description`
  - Skill 的完整描述，面向开发者和 listing 使用。
- `version`
  - Skill 版本号。适合做产品内能力包版本管理。
- `tags`
  - 搜索、分类、筛选时用的标签。
- `priority`
  - 多个 Skill 同时注入时的排序参考。值越大越靠前。
- `auto_activate`
  - 注册到 `SkillManager` 时是否自动激活。对 resident skill 很常用。
- `dependencies`
  - 依赖的其他 Skill 名称列表。激活时会检查依赖。
- `listing_description`
  - 暴露给模型的短描述。比 `description` 更短，更适合 reminder / skill listing。
- `when_to_use`
  - 告诉模型“什么时候应该调用这个 Skill”。
- `exposure_mode`
  - `resident` 或 `on_demand`。
  - `resident` 表示它属于长期可见能力。
  - `on_demand` 表示它更像延迟加载能力。
- `execution_mode`
  - `mount` 或 `inline`。
  - `mount` 表示激活后把工具、上下文等挂到 Agent 上。
  - `inline` 表示以正文/上下文形式按需注入。
- `source_type`
  - skill 来源，例如 `python`、`yaml`、`markdown`、`folder`。
- `source_path`
  - skill 源文件或目录路径。
- `cache_lifecycle`
  - `resident` / `session` / `turn`。
  - 决定 Skill 正文默认进入哪个缓存分区。
- `extra`
  - 预留给产品方的扩展配置。

推荐理解：

- `exposure_mode` 决定“模型平时是否看得到它”
- `execution_mode` 决定“真正使用时怎么接到 Agent 上”
- `cache_lifecycle` 决定“进入请求时更像稳定前缀还是动态尾部”

## 4. `BaseSkill` 需要实现哪些方法

大多数 Skill 只需要认真实现下面几个方法：

- `get_tools()`
  - 返回这个 Skill 提供的所有 Tool。
- `get_prompt()`
  - 返回 Skill 的正文提示词。旧接口和默认正文入口。
- `get_body_prompt()`
  - 返回真正用于按需展开的正文。默认等于 `get_prompt()`。
- `get_context_sources()`
  - 若 Skill 还会向 ContextManager 注入上下文来源，就在这里返回。

另外还有几类很重要的辅助方法：

- `get_listing_description()`
  - 模型在 skill listing 里看到的短描述。
- `get_when_to_use()`
  - 告诉模型何时使用这个 Skill。
- `get_exposure_mode()`
  - 当前 Skill 是 resident 还是 on-demand。
- `get_execution_mode()`
  - 当前 Skill 是 mount 还是 inline。
- `get_cache_lifecycle()`
  - 当前 Skill 正文应该属于哪个 cache lifecycle。
- `build_manifest()`
  - 构建统一元数据视图。

以及生命周期钩子：

- `on_activate(agent)`
- `on_deactivate(agent)`
- `on_before_invoke(query)`
- `on_after_invoke(query, response)`

这些钩子适合做：

- 挂载资源
- 清理状态
- 预处理用户输入
- 后处理模型输出

## 5. 一个最小可运行 Skill

```python
from easyagent.skills import BaseSkill, SkillConfig
from easyagent.tools import Tool


class MySkill(BaseSkill):
    def __init__(self):
        super().__init__(
            SkillConfig(
                name="architecture_review",
                description="帮助模型做架构审查",
                listing_description="审查架构边界、依赖和抽象层次",
                when_to_use="当任务涉及架构设计、边界划分、模块职责时使用",
                exposure_mode="resident",
                execution_mode="mount",
                cache_lifecycle="session",
            )
        )

    def get_tools(self) -> list[Tool]:
        return []

    def get_prompt(self) -> str:
        return (
            "你现在具备 architecture_review 能力。"
            "优先分析模块边界、依赖方向、抽象泄漏和扩展点。"
        )
```

如果这个 Skill 不提供工具，只提供一套专业化提示词，它依然有价值。

## 6. `SkillManager` 的职责

`SkillManager` 是运行时总控，不是简单目录。

它主要负责：

1. 注册 / 注销 Skill
2. 激活 / 停用 Skill
3. 把 Skill 的 Tool 注入 `ToolRegistry`
4. 把 Skill 的 `ContextSource` 注入 `ContextManager`
5. 聚合 skill listing、skill policy、runtime skill context
6. 跟踪按需 Skill 的临时正文和临时挂载工具
7. 代理生命周期钩子

可以把它理解成：

- `SkillRegistry` 负责“有什么”
- `SkillManager` 负责“现在谁在生效”

## 7. Skill 的四种常见形态

实际使用时，Skill 通常有四种形态。

### 7.1 Resident skill

特点：

- 注册后可自动激活
- 通常出现在 skill listing 中
- 可以长期挂载工具
- 正文往往属于 `resident` 或 `session` cache lifecycle

适合：

- 默认产品能力
- 长期可用的领域能力

### 7.2 Session skill

特点：

- 当前会话内长期生效
- 适合某类工作流中途加载一段时间

适合：

- 当前会话阶段性开启的能力包

### 7.3 Turn skill

特点：

- 只在当前 invoke 有效
- invoke 结束后清理
- 不应污染长期 canonical history

适合：

- 临时分析模板
- 一次性 task-specific 技能正文

### 7.4 On-demand skill

特点：

- 默认只在 listing 中出现
- 被 `skill_tool` 命中后才真正展开正文和临时能力

适合：

- 很长的技能正文
- 低频但专业度高的能力
- 不想平时一直占据 prompt 和 tools budget 的能力

## 8. `skill_tool`、`load_skill_tool`、`unload_skill_tool` 的区别

这三个内置工具的定位不同：

### `SkillTool`

作用：

- 当前轮按需展开 Skill 正文
- 可能临时挂载 runtime tools
- 一般不应该把长期状态写回 Agent

适合：

- “这一轮需要某种额外专业能力”

### `LoadSkillTool`

作用：

- 把 Skill 长期加载到当前 Agent
- 可能影响后续稳定前缀和 cache signature

适合：

- “从现在开始，这个 Agent 都要具备某项能力”

### `UnloadSkillTool`

作用：

- 卸载长期挂载的 Skill

适合：

- 结束某段工作流或减少后续上下文负担

## 9. 一次典型执行流程

下面用一个 on-demand skill 举例：

1. 你把 `frontend_design` 注册到 `SkillRegistry` 和 `SkillManager`
2. README / runtime reminder / skill listing 告诉模型：有这个 skill，但正文未展开
3. 用户提出“帮我设计一个更有审美的首页”
4. 模型先调用 `skill_tool(frontend_design)`
5. `SkillManager` 记录它的 `SkillManifest`
6. runtime skill body 被加入 `on_demand_expansion` 或 `dynamic_tail`
7. 如果 Skill 附带临时 Tool，这些 Tool 以 runtime/turn visibility 注册
8. 后续这一轮模型在更完整的上下文下继续工作
9. invoke 结束后，runtime skill body 和临时 tools 被清理

这就是“Skill 不是永久把 prompt 变重，而是按需展开”的关键。

## 10. 如何把 Skill 集成到 `BasicAgent`

最常见接法：

```python
from easyagent import BasicAgent, EasyLLM
from easyagent.skills import SkillManager

llm = EasyLLM()
skill_manager = SkillManager()
skill_manager.register(MySkill(), auto_activate=True)

agent = BasicAgent(
    name="assistant",
    llm=llm,
    skill_manager=skill_manager,
)
```

更完整的产品接法通常还会同时接：

- `ToolRegistry`
- `ContextManager`
- `PermissionContext`
- `runtime reminders`

也就是说，Skill 几乎总是和 Tool、Prompt、Context 一起使用。

## 11. Skill 与其他模块的关系

### 和 Tool 的关系

Skill 可以带 Tool，但 Skill 不等于 Tool。

- Tool 解决“结构化调用”
- Skill 解决“能力包和使用时机”

### 和 Prompt 的关系

Skill 正文本质上是 prompt block 的一种来源，但它不应该总是直接进入 system core。

### 和 Context 的关系

某些 Skill 会额外提供 `ContextSource`，在构建请求时参与上下文拼装。

### 和 Cache 的关系

Skill 是最容易把 cache 打爆的模块之一，因为它经常既影响 prompt，又影响 tools。

因此现在推荐：

- resident skill：只稳定暴露 listing
- on-demand skill：正文按需展开
- turn skill：不要进入稳定前缀

## 12. 推荐的产品设计模式

### 模式一：默认 resident + 少量 on-demand

适合大多数产品。

- resident：
  - file_manager
  - task_planning
  - memory
- on-demand：
  - code_review
  - frontend_design
  - web_research

### 模式二：产品能力目录化

把大量能力都注册进 `SkillRegistry`，默认只给 listing，不给正文。

适合：

- Code Agent
- Research Agent
- 企业内部多场景产品

### 模式三：Skill 只提供 prompt，不提供 Tool

适合：

- 审稿、评审、教学、写作、风格约束类能力

## 13. 常见坑

### 坑一：把所有 Skill 正文都常驻进 system

后果：

- prompt 太重
- cache 命中率差
- 用户每次请求都带大量不用的说明

### 坑二：把 Skill 当作 Tool 的替代品

Skill 不适合承载所有结构化调用。真正需要参数校验、权限、确认、并行控制的能力还是应该做成 Tool。

### 坑三：只注册 Skill，不绑定 `SkillManager`

没有 `SkillManager`，Skill 无法参与激活、挂载工具、构建 runtime context。

### 坑四：临时 skill 不清理

turn/runtime skill 的正文和临时工具必须在 invoke 结束后清理，否则会污染后续请求。

### 坑五：忽略依赖关系

如果一个 Skill 依赖另一个 Skill，应该通过 `dependencies` 显式声明，而不是靠文档约定。

## 14. 何时自定义 Skill，何时直接写代码

优先写 Skill 的场景：

- 这是一个会复用的产品能力包
- 它同时涉及 prompt、tool、context
- 你希望模型知道“什么时候该调用它”

优先直接写 Tool 或 PromptBlock 的场景：

- 只是一个独立工具
- 只是某条固定系统规则
- 只是某次请求的临时上下文

如果你不确定，通常可以这样判断：

- “模型是否需要先知道这是一项独立能力？”  
  是：更适合 Skill
- “模型只需要一个可调用接口？”  
  是：更适合 Tool
