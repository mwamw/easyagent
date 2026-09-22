# 会话持久化与迁移

## 装配后恢复

SessionManager 保存数据，不根据实现类名创建 Agent、模型或工具。
旧 BasicAgent.load_session 工厂已移除；使用已装配实例的 restore_session。

```python
from easyagent import BasicAgent
from easyagent.session import SQLiteSessionStore

store = SQLiteSessionStore("./sessions-v4.db")
agent = BasicAgent("assistant", llm).with_session(store)
agent.add_user_message("Remember this conversation")
session_id = agent.save_session(metadata={"project": "demo"})

restored = build_agent()  # 应用自己的装配函数
restored.with_session(store)
restored.restore_session(session_id)
branch_id = restored.fork_session()
```

异步入口是 asave_session、arestore_session、afork_session。
原生异步 Store 会被等待；SQLite 使用操作级连接进行安全线程桥接。
取消异步写入不保证底层写入尚未提交；结果不确定时先重新读取，不盲目覆盖。

## 自定义存储和模块状态

BaseSessionStore 是领域契约；提供 SQLiteSessionStore 与 InMemorySessionStore。
create_or_update_session 使用 expected_revision：None 表示新建，已有记录必须匹配版本。
get_session 默认不更新访问时间；list_sessions 返回不含正文的摘要，
支持 offset、limit 和 metadata_filter。未知或不支持的操作明确报错。

自定义模块实现 export_state/restore_state，必要时提供 validate_state、fork_state，
然后调用 agent.register_session_state("name", component, version=1, required=False)。
需要清除未出现在目标快照中的会话级状态时，实现 reset_session_state；
LayeredMemory 用它清除旧会话的 working memory，不触碰共享长期后端。
版本必须明确；数据必须可序列化，不支持把任意 Python 对象转成字符串。
缺失必需组件会阻止恢复，可选组件通过 SessionRestoreReport 报告。

模型、工具、权限引擎和产品配置由调用方装配。恢复保留活对象身份。
fork 不改变源 Agent，不复制进程、后台任务或共享长期记忆。
未解决工具交互时不允许 fork。界面 checkpoint 和文件回滚不属于持久化层。

## 数据范围

V4 保存当前 canonical history、可恢复模块状态和诊断追踪。
它不是完整原始 transcript archive，也不承诺恢复执行中的外部任务。
原 db/ 代码只保留用于旧数据格式与迁移测试，不再用于 Agent 主路径。
上下文 token 的 API 校准基线不随快照恢复；恢复后先使用本地估算，直到新的
主模型响应返回有效 usage，避免复用与当前模型、提示词或工具不一致的旧计数。

## 从 V3 迁移

先停止使用待迁移存储的程序，然后预览：

```bash
python -m easyagent.migrate session --source ./sessions.db --target ./sessions-v4.db
python -m easyagent.migrate session --source ./sessions.db --target ./sessions-v4.db --apply
```

默认 dry-run。源数据库只读；目标必须是新数据库或空的 V4 数据库。
迁移保留 ID、元数据和时间；目标有会话时拒绝覆盖。
旧快照以前已经压缩丢失的原始记录无法恢复，迁移会标记 legacy_snapshot。
旧独立 messages 表仍保留在源数据库，不猜测它与快照的对应关系。

S4Code 默认改用 sessions-v4.db；旧 sessions.db 不删除或自动迁移。
