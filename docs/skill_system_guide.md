# Skill System Guide

EasyAgent 的 Skill 是目录式、声明式、按需展开的 Agent 功能模块。一个 Skill 只需要一个目录和其中的 `SKILL.md`，用户通过 `agent.with_skill(dir)` 开启能力，不需要继承 Python 基类、注册全局单例或实现生命周期接口。

Skill 系统采用三层渐进披露：

1. `skill_tool` 的 schema 描述提供通用调用规则。
2. `<available_skills>` 以 `system_reminder` 告诉模型当前可用 Skill 的名称和用途，不加载正文。
3. 模型调用 `skill_tool` 后，框架才读取完整 `SKILL.md`，并通过 `INVOCATION` MetaMessage 放到下一次 LLM request 的 history 末尾。

这保证大量低频 Skill 不会持续占用上下文，同时保持工具调用协议顺序正确。

## 1. 快速接入

```python
from pathlib import Path

from easyagent import BasicAgent, EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)

agent = BasicAgent(name="assistant", llm=llm).with_skill(Path("./skills"))
answer = agent.invoke("使用 repository-review 检查这次代码修改")
```

`with_skill` 接受三种用法：

```python
agent.with_skill("./skills")
agent.with_skill("./team-skills", "./project-skills")
agent.with_skill("./team-skills").with_skill("./project-skills")
```

传入路径可以是单个 Skill 目录，也可以是包含多个直接子 Skill 目录的集合目录。`with_skill` 自动完成以下装配：

- 创建 Agent 私有的 `SkillManager`
- 索引目录中的 `SKILL.md`
- 在需要时创建 `ToolRegistry`
- 安装唯一的模型入口 `skill_tool`
- 把 Skill listing 加入系统提示词的 `system_reminder`
- 订阅 RuntimeEvent，自动管理条件发现和正文回收；不修改权限策略

没有调用 `with_skill` 时，Agent 不创建 SkillManager，也不暴露 `skill_tool`。

## 2. 目录格式

```text
skills/
  repository-review/
    SKILL.md
  release-check/
    SKILL.md
```

最小 `SKILL.md`：

```markdown
---
name: repository-review
description: Inspect a repository change and produce an evidence-based review.
when_to_use: Use when the user asks for a code or patch review.
argument-hint: A file path, module, commit, or review objective.
recommended-tools: FileRead, Glob, Grep, List
---

# Repository Review

Review `$ARGUMENTS` as a senior maintainer.
The Skill directory is `${SKILL_DIR}`.
```

`name` 必须满足 `^[a-z0-9][a-z0-9-]{0,63}$`，并且必须与目录名一致。正文不会在索引阶段读入 `SkillManifest`，只有真正调用 Skill 时才读取。

## 3. Frontmatter

| 字段 | 必填 | 语义 |
| --- | --- | --- |
| `name` | 否 | Skill 名称，省略时使用目录名 |
| `description` | 是 | listing 中的简短能力描述 |
| `when_to_use` | 否 | 模型应主动调用 Skill 的条件和示例 |
| `argument-hint` | 否 | `args` 的格式提示 |
| `recommended-tools` | 否 | 工具使用建议，不授权工具，也不展开工具 Schema |
| `context` | 否 | `inline` 或 `fork`，默认 `inline` |
| `agent` | 否 | fork Skill 使用的 subagent 类型，只允许配合 `context: fork` |
| `model` | 否 | fork Skill 使用的模型，只允许配合 `context: fork` |
| `paths` | 否 | 条件发现的 gitignore 风格工作区路径模式 |
| `disable-model-invocation` | 否 | 为 `true` 时不进入 listing，模型也不能调用 |

未识别字段保存在 `SkillManifest.metadata`，供应用层扩展使用，但框架不会隐式执行这些字段。

正文支持两个替换变量：

- `$ARGUMENTS` 替换为 `skill_tool.args` 的原始字符串。
- `${SKILL_DIR}` 替换为当前 Skill 目录的绝对路径。

## 4. 一次真实执行的顺序

假设模型调用：

```json
{"skill": "repository-review", "args": "skill/manager.py"}
```

框架执行顺序如下：

1. Agent invoke 发布 `agent.invoke.started`。
2. PromptComposer 只生成 `<available_skills>`，此时 request 中没有 Skill 正文。
3. LLM 返回 `skill_tool` 调用。
4. SkillManager 校验名称、可见性和模型调用权限，然后延迟读取 `SKILL.md` 正文。
5. SkillManager 替换参数，返回工具建议，并把完整正文排入 `INVOCATION` MetaMessage 队列。
6. Executor 先把 Assistant tool call 和 `skill_tool` ToolResult 写入 history。
7. 构建下一次 LLM request 时，MetaMessageManager 执行 `flush()`，把 Skill 正文作为 user-role 消息追加在 ToolResult 后面。
8. 后续 LLM 调用可以使用正文；每次工具调用仍须经过正常授权。
9. Agent 完成、失败或中断时发布终态事件，框架删除 Skill 正文并清理 invocation 去重状态，不改变权限策略。

模型看到的关键顺序始终是：

```text
assistant(tool_call: skill_tool)
tool(skill_tool result)
user(<skill name="repository-review">...</skill>)
assistant(next reasoning/tool calls)
```

Skill 正文不会进入 provider 原生 system 字段，也不会被直接拼接到 ToolResult。这样不会破坏 Assistant tool call 与 Tool result 必须相邻的协议约束。

## 5. MetaMessage 生命周期

Skill 正文由 Skill 模块调用 `emit()` 生成，生命周期固定为 `MetaMessageLifecycle.INVOCATION`。用户不需要也不应该手工调用 `begin_invocation` 或 `end_invocation`，这两个接口不存在。

MetaMessageManager 只消费统一 RuntimeEvent：

- `agent.invoke.started` 开始一次 Agent 调用
- `llm.invoke.completed/failed` 回收 `REQUEST` 消息
- `agent.invoke.completed/failed/interrupted` 回收 `INVOCATION` 和 `REQUEST` 消息

`invocation_id` 只存在于 RuntimeEvent 和 trace，不进入 MetaMessageContext，也不会进入模型 history。临时 pending 消息与临时 injection 不写入 SessionSnapshot。

## 6. `paths` 条件发现

`paths` 不是 Skill 资源目录。它决定 Skill 何时出现在 listing 中，语义与 Claude Code 的条件 Skill 一致。

```yaml
paths: "src/**/*.{py,pyi}, tests/**"
```

规则支持 YAML 列表、逗号分隔、brace 展开和 gitignore 风格匹配。带 `paths` 的 Skill 初始隐藏。当成功完成的 filesystem/notebook 工具事件包含匹配路径时，SkillManager 自动激活它：

```text
FileRead(src/package/service.py)
  -> tool.invoke.completed
  -> paths 匹配 src/**/*.py
  -> python-review 加入后续 <available_skills>
```

激活状态属于会话级索引状态，跨 invoke 保留，并随 SessionSnapshotV3 恢复。`paths: "**"` 等价于无条件 Skill。工作区之外的路径和 URI 不参与匹配。

应用也可以显式通知自定义文件模块：

```python
activated = agent.skill_manager.activate_for_paths(["src/domain/model.py"])
```

更推荐自定义文件工具带上 `filesystem` 或 `notebook` tag，并通过标准 RuntimeEvent 执行链路自动触发。

## 7. `recommended-tools`

示例：

```yaml
recommended-tools: FileRead, Grep, Bash(git status:*)
```

SkillManager 返回已安装工具的建议条目；不可用的条目进入 `unavailableRecommendedTools`。
括号内的文本只是建议，不被解释成授权规则。Skill 不注册工具、不展开 Deferred Schema，
不调用权限引擎的规则管理接口。旧 `allowed-tools` 不具有任何授权效果，应迁移为 `recommended-tools`。

inline 和 fork Skill 均不能绕过 ASK 或 DENY；fork 委派调用仍通过 Agent 的正常工具授权入口。
激活、结束、失败、中断、恢复和关闭 Skill 都不修改既有权限规则。

## 8. Inline 与 Fork

### Inline

`context: inline` 在当前 Agent 中注入完整正文，适合需要继续和用户交互、依赖当前上下文或需要多轮工具调用的工作流。正文只在当前 Agent invoke 中保留。

### Fork

```yaml
context: fork
agent: reviewer
model: qwen3.5-9b
```

fork Skill 必须先通过 `agent.with_multi_agent()` 提供 `Agent` 工具。SkillManager 同步创建子 Agent，并将已展开的 Skill 正文作为完整任务 prompt。ToolResult 返回：

```json
{
  "success": true,
  "skill": "deep-review",
  "status": "forked",
  "agentId": "agent_...",
  "outputFile": "/workspace/.easyagent-agents/agent_....md",
  "result": "...",
  "recommendedTools": ["FileRead", "Grep"],
  "unavailableRecommendedTools": []
}
```

默认子 Agent 会获得自己的 SkillManager 和自己的 `skill_tool`，不会复用指向父 Agent 的工具实例，因此子 Agent 加载 Skill 不会污染父 Agent history。

## 9. ToolResult

inline 成功结果会明确告诉模型正文将在下一次 request 出现，并携带：

- `status=inline`
- `scope=invocation`
- `alreadyActive`
- `instructionSource`
- `skillDirectory`
- `recommendedTools`
- `unavailableRecommendedTools`

同一 invoke 内相同 Skill 与相同 `args` 会返回 `alreadyActive=true`，不会重复注入正文。错误类型包括：

- `skill_not_found`
- `skill_model_invocation_disabled`
- `skill_not_active`
- `skill_invoke_failed`

`skill_not_active` 结果携带 `conditionalPaths`，说明该 Skill 尚未被匹配文件激活。

## 10. 自定义 SkillManager

需要改变目录来源、listing 或激活策略时，可以继承 `SkillManager`，然后显式装配：

```python
class ProductSkillManager(SkillManager):
    def build_skill_listing_prompt(self) -> str:
        return super().build_skill_listing_prompt()


manager = ProductSkillManager().add_directories(["./skills"])
agent.with_skill(manager=manager)
```

自定义 manager 仍应遵守三个边界：正文延迟加载、通过 MetaMessage 注入、通过 RuntimeEvent 清理。Agent 不依赖全局注册中心。

## 11. Session 与关闭

SessionSnapshotV3 保存 Skill 根目录、全部 manifest 快照和已经激活的条件 Skill 名称。恢复时重新读取当前磁盘目录，以磁盘内容为事实源，并通过 `SessionRestoreReport` 报告缺失目录或 Skill。

不会持久化的内容包括当前 invoke 的正文、请求级消息和 invoke 去重键。`agent.close()` 会取消事件订阅并清理 Skill 本地状态；Skill 不创建或清理权限来源。

完整真实 LLM 示例见 `example/example_skill_runtime.py`，对应 Skill 见 `example/skills/repository-review/SKILL.md`。示例使用本地 OpenAI-compatible 服务，需由使用者自行启动服务后手动执行。
