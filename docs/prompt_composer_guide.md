# Prompt Composer Guide

`BasePromptComposer` 负责组织 prompt blocks。它的职责不是直接构造 provider request，也不是负责 replay history，而是回答：

> 这个 Agent 的系统层内容应该由哪些 block 组成、顺序是什么、每个 block 属于哪一层？

如果你要做的是“基于 EasyAgent 做不同 Agent 产品”，这个对象是产品化 system prompt 的重要扩展点。

相关文档：

- [Prompt System Guide](./prompt_system_guide.md)
- [Runtime Reminders Guide](./runtime_reminders_guide.md)
- [Skill System Guide](./skill_system_guide.md)

## 1. Prompt Composer 的职责边界

Prompt Composer 负责：

- 产出 `PromptBlock`
- 组织核心规则块
- 组织共享块
- 决定 tool inventory / skill listing 是否进入 prompt block
- 接受外部追加的 prompt block

Prompt Composer 不负责：

- provider-specific request 格式
- 最终 cache marker 布局
- replay history 构造
- tool result 回放
- runtime reminder 的最终 prepend 位置

推荐理解：

- Composer 负责“系统层有哪些块”
- RequestCompiler / RequestInput 负责“这些块最后怎么进入请求”

## 2. `BasePromptComposer` 的核心接口

这是产品定制最需要理解的部分。

### `get_enhanced_prompt(agent)`

返回最终渲染后的纯文本 prompt。

作用：

- 调试最终 prompt
- 兼容仍然依赖字符串 system prompt 的路径

你通常不会只重写这个方法，而是重写下面更底层的方法。

### `get_system_prompt_template(agent)`

返回 `SystemPromptTemplate`。

作用：

- 保留 block 级结构
- 让上游仍能按 block 排序和渲染

### `get_system_prompt_blocks(agent)`

最核心的方法。

它决定：

- identity block 是什么
- core blocks 有哪些
- tool inventory 要不要出现
- shared blocks 怎么拼

如果你想整体重排系统提示词结构，最常改的就是这个方法。

### `build_core_prompt_blocks(agent, *, start_order, include_tool_policy)`

构建核心行为规则块。

默认会组织这类内容：

- identity
- visibility
- task execution
- safety
- tool policy
- tone/style
- output efficiency

如果你想改的是“产品的核心行为规范”，通常改这里。

### `get_tool_catalog_prompt(agent)`

生成完整工具目录文本。

适合：

- 调试工具说明
- 做较重的 tool catalog 展示

### `get_tool_inventory_prompt(agent, *, include_parameters=False)`

生成工具概览文本。

常用于：

- 简洁 inventory
- 带参数的 full inventory

### `build_tool_inventory_block(agent, order)`

决定是否把工具概览作为单独 block 插入系统层。

如果你的产品完全依赖 deferred tools，只想保留很轻的 tool listing，这个方法就很重要。

### `build_shared_prompt_blocks(...)`

组织共享块。

典型内容包括：

- custom prompt
- skill policy
- skill listing
- memory
- mailbox
- extension blocks

这是“产品特定功能如何接到系统层”的主要入口。

### `get_extension_prompt_blocks(start_order)`

收集通过 `with_prompt_block(s)` 注册的扩展块。

### `with_prompt_block(block)`

注册单个扩展 block。

### `with_prompt_blocks(blocks)`

批量注册扩展 block。

## 3. 最小自定义示例

```python
from easyagent.prompting import BasePromptComposer, PromptBlock, SystemPromptTemplate


class ProductPromptComposer(BasePromptComposer):
    def __init__(self):
        self._extra = []

    def get_enhanced_prompt(self, agent):
        return self.get_system_prompt_template(agent).render()

    def get_system_prompt_template(self, agent):
        return SystemPromptTemplate(self.get_system_prompt_blocks(agent))

    def get_system_prompt_blocks(self, agent):
        blocks = [
            PromptBlock("identity", "你是 Acme Code Agent。", order=0),
            PromptBlock("policy", "先读代码，再给结论。", order=10),
        ]
        blocks.extend(self._extra)
        return blocks

    def build_core_prompt_blocks(self, agent, *, start_order, include_tool_policy):
        return []

    def get_tool_catalog_prompt(self, agent):
        return ""

    def get_tool_inventory_prompt(self, agent, *, include_parameters=False):
        return ""

    def build_tool_inventory_block(self, agent, order):
        return None

    def build_shared_prompt_blocks(self, agent, *, start_order, include_custom_prompt=True, include_memory=True, include_skills=True):
        return []

    def get_extension_prompt_blocks(self, start_order):
        return self._extra

    def with_prompt_block(self, block):
        self._extra.append(block)

    def with_prompt_blocks(self, blocks):
        self._extra.extend(blocks)
```

## 4. 每个主要方法在一次请求里的位置

一个典型请求的系统层构造过程是：

1. `BasicAgent` 准备本轮调用
2. 调用 `prompt_composer.get_system_prompt_blocks(agent)`
3. composer 内部可能进一步调用：
   - `build_core_prompt_blocks`
   - `build_tool_inventory_block`
   - `build_shared_prompt_blocks`
   - `get_extension_prompt_blocks`
4. 得到一组 `PromptBlock`
5. 这些 block 再进入 RequestCompiler / ReplayRequestInput
6. 最终与 runtime reminders、history、dynamic tail 一起组成完整请求

所以 Composer 发生在“真正请求编译”的前半段。

## 5. 什么时候应该改哪个方法

这是产品定制里最实用的一部分。

### 只想加一条固定产品规则

优先：

- `with_prompt_block(...)`

不要为了加一句话就重写整个 composer。

### 想整体重排系统提示词结构

重写：

- `get_system_prompt_blocks`

### 想只改核心行为规范

重写：

- `build_core_prompt_blocks`

### 想控制 tool inventory 怎么出现在系统层

重写：

- `get_tool_inventory_prompt`
- `build_tool_inventory_block`

### 想改变 memory / skill / custom prompt 的位置

重写：

- `build_shared_prompt_blocks`

### 想追加动态运行时信息

优先不要改 composer，而是使用：

- [Runtime Reminders Guide](./runtime_reminders_guide.md)

因为这类信息通常不应该直接进稳定 system block。

## 6. `PromptBlock` 的关键字段怎么理解

Composer 产出的核心单位是 `PromptBlock`。

你在设计自定义 block 时，最重要的是这些概念：

- `name`
  - block 名称
- `content`
  - block 正文
- `order`
  - 排序位置
- `metadata`
  - 额外信息，例如：
    - request layer
    - cache partition
    - cacheable

这意味着 Prompt Composer 不只是“拼字符串”，而是在声明：

- 这是什么 block
- 排在什么位置
- 进入请求时属于哪一层

## 7. Composer 与 Runtime Reminder 的边界

这是最容易写乱的地方。

Composer 适合承载：

- 稳定的系统级规则
- 长期存在的能力说明
- tool / skill 的稳定 listing

Reminder 适合承载：

- 当前日期
- 当前环境
- 当前 mailbox 状态
- 当前临时约束
- 动态产品上下文

简单判断规则：

- 这条信息如果每轮可能变，优先 reminder
- 这条信息如果是长期稳定规则，优先 composer

## 8. Composer 与 Skill / Memory 的关系

### Skill

Composer 只应负责：

- skill policy
- skill listing
- resident skill 的稳定说明

不应默认把所有 skill body 全塞进 system。

### Memory

Composer 只应负责 memory 的稳定呈现策略，或 very-high-level memory policy。  
真正动态的 memory 内容更适合走 context / reminder / dynamic tail。

## 9. 如何接到 `BasicAgent`

```python
from easyagent import BasicAgent, EasyLLM

agent = BasicAgent(
    name="assistant",
    llm=EasyLLM(),
    prompt_composer=ProductPromptComposer(),
)
```

如果你只是想在默认 composer 基础上追加一些块，通常不需要完全替换类，也可以在已有 composer 上用 `with_prompt_block`。

## 10. 推荐实践

### 保持 composer 的职责单一

只负责 block 组织，不要把 provider-specific、history、runtime 逻辑都塞进来。

### 系统层尽量稳定

让易变化内容去 reminder / dynamic tail，减少 system 抖动。

### 用 block 和 metadata 建模，而不是硬拼字符串

这样后续 request compiler 和 cache 才能正确理解结构。

### 少量增量需求优先用扩展 block

避免一开始就 fork 整个默认 composer。

## 11. 常见坑

### 坑一：把所有运行时信息都放进 composer

这样会导致系统层过于动态，破坏 cache 稳定性。

### 坑二：为追加一条规则而重写整个 composer

这会让维护成本快速上升。

### 坑三：在 composer 里直接做 provider request 拼装

这会破坏模块边界。

### 坑四：把 tool inventory 和 deferred tool 策略混为一谈

inventory 只是提示；真正的 schema 暴露仍然由 tool registry / request compiler 决定。

### 坑五：忽略 `order` 和 metadata

如果只关心文本，不关心层次信息，后面 cache 和 request frame 很难做对。
