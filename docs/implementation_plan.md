# EasyAgent 框架完善实现计划

完善 EasyAgent 框架，使其成为一个功能完整、可扩展的 Agent 开发框架。

## User Review Required

> [!IMPORTANT]
> 以下设计决策需要确认：
> 1. **Agent 类型优先级**：建议先完成 PlanningAgent 和 ConversationalAgent，其他类型后续迭代
> 2. **记忆系统**：是否需要支持向量数据库（如 ChromaDB/FAISS），还是先只做简单的对话记忆？
> 3. **多 Agent 协作**：这是高级功能，是否放到第二阶段？

---

## 目标项目结构

```
EasyAgent/
├── __init__.py
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── agent.py                   # [增强] BaseAgent 抽象基类
│   ├── llm.py                     # [保持] LLM 封装
│   ├── Message.py                 # [保持] 消息类型
│   ├── Config.py                  # [增强] 配置管理
│   └── Exception.py               # [增强] 异常体系
├── agent/                         # Agent 实现
│   ├── __init__.py
│   ├── BasicAgent.py              # [保持] 基础 Tool-Use Agent
│   ├── ReactAgent.py              # [完善] 显式思考链 Agent
│   ├── PlanningAgent.py           # [新建] 规划执行 Agent
│   ├── ConversationalAgent.py     # [新建] 对话记忆 Agent
│   └── StructuredOutputAgent.py   # [新建] 结构化输出 Agent
├── memory/                        # 记忆系统 [新模块]
│   ├── __init__.py
│   ├── base.py                    # BaseMemory 抽象基类
│   ├── buffer.py                  # ConversationBufferMemory
│   └── summary.py                 # ConversationSummaryMemory
├── prompt/                        # 提示词模板 [新模块]
│   ├── __init__.py
│   ├── template.py                # PromptTemplate / ChatPromptTemplate
│   └── defaults.py                # 预置提示词模板
├── output/                        # 输出解析器 [新模块]
│   ├── __init__.py
│   ├── base.py                    # BaseOutputParser
│   ├── json_parser.py             # JsonOutputParser
│   └── pydantic_parser.py         # PydanticOutputParser
├── Tool/                          # 工具模块
│   ├── __init__.py
│   ├── BaseTool.py                # [保持] 工具基类
│   ├── ToolRegistry.py            # [增强] 工具注册表
│   └── builtin/                   # [新建] 预置工具
│       ├── __init__.py
│       ├── search.py              # 搜索工具
│       └── calculator.py          # 计算器工具
└── test/                          # 测试
    ├── test_basicagent.py
    ├── test_memory.py
    └── test_output.py
```

---

## Proposed Changes

### 1. 核心模块增强

---

#### [MODIFY] [agent.py](file:///home/wxd/LLM/EasyAgent/core/agent.py)

增强 `BaseAgent` 基类：
- 添加 `memory` 属性支持记忆系统
- 添加 `callbacks` 支持回调钩子
- 添加 `stream_invoke()` 抽象方法

```python
class BaseAgent(ABC):
    def __init__(self, name, llm, system_prompt=None, description=None, 
                 config=None, memory=None, callbacks=None):
        self.memory = memory  # 新增：记忆系统
        self.callbacks = callbacks or []  # 新增：回调钩子
        # ... 其他保持不变
    
    @abstractmethod
    def stream_invoke(self, query: str, **kwargs) -> Generator[str, None, None]:
        """流式调用"""
        pass
    
    def on_agent_start(self, query: str):
        """回调：Agent 开始"""
        for callback in self.callbacks:
            callback.on_agent_start(query)
```

---

#### [MODIFY] [Exception.py](file:///home/wxd/LLM/EasyAgent/core/Exception.py)

扩展异常类型：

```python
class MemoryError(AgentError):
    """记忆系统异常"""
    pass

class OutputParseError(AgentError):
    """输出解析异常"""
    pass

class PromptTemplateError(AgentError):
    """提示词模板异常"""
    pass
```

---

### 2. 记忆系统

---

#### [NEW] [base.py](file:///home/wxd/LLM/EasyAgent/memory/base.py)

```python
from abc import ABC, abstractmethod
from typing import List
from core.Message import Message

class BaseMemory(ABC):
    """记忆系统基类"""
    
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """添加消息到记忆"""
        pass
    
    @abstractmethod
    def get_messages(self) -> List[Message]:
        """获取记忆中的消息"""
        pass
    
    @abstractmethod
    def get_context(self) -> str:
        """获取记忆上下文（用于注入到提示词）"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass
```

---

#### [NEW] [buffer.py](file:///home/wxd/LLM/EasyAgent/memory/buffer.py)

对话缓冲记忆：保存最近 N 轮对话

```python
class ConversationBufferMemory(BaseMemory):
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Message] = []
    
    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_context(self) -> str:
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])
```

---

#### [NEW] [summary.py](file:///home/wxd/LLM/EasyAgent/memory/summary.py)

摘要记忆：使用 LLM 对历史对话进行摘要

```python
class ConversationSummaryMemory(BaseMemory):
    def __init__(self, llm: EasyLLM, max_tokens: int = 500):
        self.llm = llm
        self.summary = ""
        self.buffer = []  # 待摘要的新消息
    
    def _summarize(self) -> None:
        """使用 LLM 生成摘要"""
        if len(self.buffer) >= 5:
            prompt = f"请将以下对话摘要为简洁的要点：\n{self.buffer}"
            self.summary = self.llm.invoke([{"role": "user", "content": prompt}])
            self.buffer.clear()
```

---

### 3. Agent 类型实现

---

#### [MODIFY] [ReactAgent.py](file:///home/wxd/LLM/EasyAgent/agent/ReactAgent.py)

完善显式思考链 Agent：
- 强制两步调用：先获取思考，再执行工具
- 维护 scratchpad 记录推理过程
- 输出 Thought → Action → Observation 链

---

#### [NEW] [PlanningAgent.py](file:///home/wxd/LLM/EasyAgent/agent/PlanningAgent.py)

规划执行 Agent：
- 第一步：分析任务，生成执行计划
- 第二步：逐步执行计划中的每个步骤
- 支持计划修正和重规划

```python
class PlanningAgent(BasicAgent):
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs):
        # 1. 生成计划
        plan = self._generate_plan(query)
        
        # 2. 逐步执行
        for step in plan:
            result = self._execute_step(step)
            if result.needs_replan:
                plan = self._replan(plan, result)
        
        return self._summarize_results()
```

---

#### [NEW] [ConversationalAgent.py](file:///home/wxd/LLM/EasyAgent/agent/ConversationalAgent.py)

对话记忆 Agent：
- 集成 Memory 系统
- 支持上下文注入
- 适合多轮对话场景

---

#### [NEW] [StructuredOutputAgent.py](file:///home/wxd/LLM/EasyAgent/agent/StructuredOutputAgent.py)

结构化输出 Agent：
- 强制输出符合指定 Schema
- 集成 PydanticOutputParser
- 自动重试解析失败的情况

---

### 4. 提示词模板系统

---

#### [NEW] [template.py](file:///home/wxd/LLM/EasyAgent/prompt/template.py)

```python
class PromptTemplate:
    """简单文本模板"""
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class ChatPromptTemplate:
    """对话模板"""
    def __init__(self, messages: List[dict]):
        self.messages = messages
    
    def format_messages(self, **kwargs) -> List[Message]:
        # 格式化每个消息模板
        pass
```

---

#### [NEW] [defaults.py](file:///home/wxd/LLM/EasyAgent/prompt/defaults.py)

预置提示词模板：
- `REACT_PROMPT` - ReAct Agent 模板
- `PLANNING_PROMPT` - 规划 Agent 模板
- `STRUCTURED_OUTPUT_PROMPT` - 结构化输出模板

---

### 5. 输出解析器

---

#### [NEW] [base.py](file:///home/wxd/LLM/EasyAgent/output/base.py)

```python
class BaseOutputParser(ABC):
    @abstractmethod
    def parse(self, output: str) -> Any:
        """解析 LLM 输出"""
        pass
    
    def get_format_instructions(self) -> str:
        """获取格式说明（注入到提示词）"""
        return ""
```

---

#### [NEW] [json_parser.py](file:///home/wxd/LLM/EasyAgent/output/json_parser.py)

```python
class JsonOutputParser(BaseOutputParser):
    def parse(self, output: str) -> dict:
        # 提取 JSON 块并解析
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(output)
```

---

#### [NEW] [pydantic_parser.py](file:///home/wxd/LLM/EasyAgent/output/pydantic_parser.py)

```python
class PydanticOutputParser(BaseOutputParser):
    def __init__(self, pydantic_model: Type[BaseModel]):
        self.model = pydantic_model
    
    def parse(self, output: str) -> BaseModel:
        data = json.loads(output)
        return self.model.model_validate(data)
    
    def get_format_instructions(self) -> str:
        return f"输出必须符合以下 JSON Schema:\n{self.model.model_json_schema()}"
```

---

### 6. 工具增强

---

#### [NEW] [search.py](file:///home/wxd/LLM/EasyAgent/Tool/builtin/search.py)

预置搜索工具（封装 SerpApi）

---

#### [NEW] [calculator.py](file:///home/wxd/LLM/EasyAgent/Tool/builtin/calculator.py)

预置计算器工具（安全执行数学表达式）

---

## Verification Plan

### 自动化测试

1. **记忆系统测试**
   ```bash
   cd /home/wxd/LLM/EasyAgent && python -m pytest test/test_memory.py -v
   ```

2. **输出解析器测试**
   ```bash
   cd /home/wxd/LLM/EasyAgent && python -m pytest test/test_output.py -v
   ```

3. **现有 Agent 测试**
   ```bash
   cd /home/wxd/LLM/EasyAgent/test && python test_basicagent.py
   ```

### 手动验证

1. **PlanningAgent 验证**：提问复杂任务如"帮我写一篇关于 GraphRAG 的调研报告"，验证是否能正确分解任务并逐步执行

2. **ConversationalAgent 验证**：进行多轮对话，验证是否能记住之前的上下文

3. **StructuredOutputAgent 验证**：要求输出特定格式（如 JSON），验证是否能正确解析

---

## 实现优先级

| 优先级 | 模块 | 工作量 | 说明 |
|--------|------|--------|------|
| P0 | memory/ 模块 | 2h | 基础设施，其他模块依赖 |
| P0 | output/ 模块 | 1.5h | 结构化输出必需 |
| P1 | prompt/ 模块 | 1h | 提升代码复用 |
| P1 | PlanningAgent | 2h | 核心 Agent 类型 |
| P1 | ConversationalAgent | 1.5h | 常用场景 |
| P2 | StructuredOutputAgent | 1h | 依赖 output 模块 |
| P2 | ReactAgent 完善 | 1h | 已有基础 |
| P3 | 预置工具 | 1h | 增强易用性 |
