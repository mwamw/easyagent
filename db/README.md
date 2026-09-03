# EasyAgent 会话管理与对话持久化

本文档介绍了 EasyAgent 中对话持久化与会话管理的设计和使用方法。系统的持久层代码主要集中在 `db` 目录下，并已和 Agent 架构深度集成，允许开发者将所有 Agent 的内部运行状态和完整对话历史保存到磁盘，从而实现系统崩溃恢复、多端同步以及长线任务记录。

---

## 🏗 核心组件

目前的持久化方案默认基于 **SQLite**，核心模块由以下两部分构成：

### 1. `session_store.py: SessionStore` 
负责 Agent 会话生命周期与核心状态的管理。
- **快照存储 (Snapshot)**: 使用 `SessionSnapshotV3` 持久化 BasicAgent 的 canonical history、runtime events 和已安装模块状态。
- **生命周期**: 自动记录创建 (`created_at`)、更新 (`updated_at`) 与最后访问时间 (`last_accessed_at`)。
- **过期管理**: 支持为特定会话设置 `expires_at`，并提供了 `cleanup_expired_sessions()` 定期清理失效数据。

### 2. `conversation_store.py: ConversationStore`
专门负责对话历史 (Message List) 的持久化落地。
- **Canonical 消息**: 统一存储 `CanonicalMessage` 与其 text/reasoning/tool_call/tool_result 内容块，不依赖任一 provider 的消息类。
- **增量式管理**: 通过 `replace_messages` 与 `load_messages` 控制对话窗口的重建。

---

## 🚀 快速入门

持久化能力由 `BaseAgent` 提供，当前唯一维护的可实例化实现是 `BasicAgent`。

### 保存会话
只要调用实例的 `save_session` 方法，Agent 就会自动将内部状态快照和消息列表存入本地 `SQLite` 环境（默认路径为 `db/easyagent_sessions.db`）。

```python
from agent import BasicAgent
from core.llm import EasyLLM

llm = EasyLLM(...)
agent = BasicAgent(name="assistant", llm=llm, system_prompt="You are a helper")

# 在经历几轮交互后...
agent.invoke("帮我写一个 python 脚本")

# 保存当前所有状态和对话历史上下文
agent.save_session("session_1001")
```

### 恢复会话
通过类方法 `load_session`，我们可以基于 `session_id` 于任意时刻重启该进程。注意需要补齐外部依赖（例如 llm 客户端实例或者 tool_registry）。

```python
from agent import BasicAgent

# 内置轻量模块可自动恢复；Tool、Context、Memory、MCP 等外部依赖需显式传入
restored_agent = BasicAgent.load_session(
    session_id="session_1001",
    llm=llm
)

# 直接从上次中断的地方继续对话
response = restored_agent.invoke("继续我刚才的问题")
```

---

## 🔧 高级会话管理 API

如果需要编写系统级的管理后台（类似对话历史侧边栏），可以直接实例化 `SessionStore` 进行元信息调度。

```python
from db.session_store import SessionStore

store = SessionStore()

# 1. 批量获取活跃的会话列表，用于前端 UI 渲染
active_sessions = store.list_sessions(limit=50)
for session in active_sessions:
    print(session["session_id"], session["agent_type"], session["updated_at"])

# 2. 从数据库删除此会话记录
store.delete_session("session_1001")

# 3. 释放资源，清理一切超期失效的会话
removed_count = store.cleanup_expired_sessions()
print(f"清除了 {removed_count} 条过期会话记录。")
```

## ⏳ 状态（Roadmap）

目前的实现基于 SQLite 本地化，能够满足单节点应用 99% 的核心需求：
- [x] 基于 SQLite 的会话持久化与恢复。
- [x] `SessionStore` 与 `ConversationStore` 生命周期管理。
- [x] `BasicAgent` 与模块化 `with_*` 能力的显式快照/恢复。
- [ ] 考虑到分布式部署，未来将可能扩展 **Redis 存储后端**，提供 `RedisSessionStore` 与 `RedisConversationStore` 抽象类。
