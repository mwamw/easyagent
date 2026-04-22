# Phase H: CodeIntel Workspace Cache + Offline Snapshot

## 本阶段完成了什么

这一阶段补的是 `Phase D` 之后最关键的一层增强：让 `codeintel` 不再只依赖“当前活着的 LSP 进程”，而是具备真正的 `workspace cache + offline symbol snapshot`。

现在已经落地的能力有：

- `codeintel/cache.py`
  - 新增 `WorkspaceCodeIntelCache`
  - 缓存 `document symbols`、`diagnostics`
  - 缓存精确查询结果：`definition / references`
  - 形成离线 `workspace symbols` 快照
- `CodeIntelManager`
  - 增加 `prewarm_workspace()`
  - 增加 `get_cache_status()`
  - provider 失败时，优先从 cache / offline index 回退
  - 支持 `export_state()` / `restore_state()` / `from_state()`
- codeintel tool
  - 新增 `CodeIntelCacheStatus`
  - 新增 `CodeIntelPrewarmWorkspace`
  - 原有 `FindDefinition / FindReferences / GetDocumentSymbols / GetWorkspaceSymbols / GetDiagnostics`
    在 provider 不可用时，会优先尝试 cache fallback
- session restore
  - `codeintel_runtime` 会进入 session snapshot
  - `load_session()` 会自动恢复 codeintel manager 和缓存快照

## 现阶段框架的变换

这一阶段之前，EasyAgent 的 codeintel 语义是：

- LSP 活着，就能查
- LSP 不可用，就只能退回 `FileRead / Grep / Glob`

这一阶段之后，框架语义变成了：

- LSP 活着时，优先走实时 provider
- 查询结果会被写进 workspace cache
- 对工作区做过 `prewarm` 之后，会形成离线 symbol snapshot
- 如果后续 provider 不稳定、重启、暂时缺失，部分查询可以直接从 cache / offline index 回退
- 这些 cache 还能随 session 一起保存和恢复

也就是说，EasyAgent 的 codeintel 现在从“实时查询层”升级成了“实时查询 + 离线快照”双层结构。

## 一个具体例子

假设 manager 要分析一个 Python 仓库里的 `TaskService`：

1. 先运行 `CodeIntelPrewarmWorkspace(path_prefix="task", max_files=50)`
2. 这一步会把 `task/` 目录下的 document symbols 和 diagnostics 写进 cache
3. 随后调用 `GetWorkspaceSymbols(query="TaskService")`
4. 如果此时 LSP 正常，优先返回实时结果
5. 如果随后 LSP server 崩掉了，或者恢复后临时不可用，再次调用 `GetWorkspaceSymbols(query="TaskService")`
6. manager 会直接从离线 symbol snapshot 里回退，给出 cached result

这跟之前最大的差异在于：现在 code agent 的“仓库理解”不再完全绑定单次 LSP 存活状态。

## 适合的使用方式

推荐流程：

1. 先用 `CodeIntelStatus` 看当前 provider 是否可用
2. 如果接下来要连续做很多 symbol / diagnostics 查询，先执行 `CodeIntelPrewarmWorkspace`
3. 用 `CodeIntelCacheStatus` 检查离线索引是否已经形成
4. 再进入 `GetWorkspaceSymbols / FindDefinition / FindReferences / GetDiagnostics`

对大仓库不要默认全量预热，优先传：

- `path_prefix`
- `max_files`

## 对外暴露的新接口

新增 public API：

- `codeintel.WorkspaceCodeIntelCache`
- `codeintel.CachedFileEntry`
- `codeintel.CachedQueryEntry`

新增 tool：

- `CodeIntelCacheStatus`
- `CodeIntelPrewarmWorkspace`

## 验收结果

本阶段已通过的契约验证包括：

- `prewarm -> cache status -> offline index` 全链路
- `workspace symbols` 在 provider 不可用时的离线 fallback
- `document symbols / diagnostics` 的 cache fallback
- `definition / references` 的 query cache fallback
- `session save/load` 后 codeintel cache 仍然可恢复

对应 example：

- `example/example_phaseh_codeintel_workspace_cache.py`

下一步主线：

- `observability`
  - token / cost / error type 聚合
  - trace summary
  - 更成体系的 metrics / benchmark
