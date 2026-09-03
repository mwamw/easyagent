# Skill Package Reference

完整使用说明见 [`docs/skill_system_guide.md`](../docs/skill_system_guide.md)。本文件只说明 `skill/` 包的代码边界。

## Public Objects

- `SkillManifest`：只保存 `SKILL.md` frontmatter 和源路径，不保存正文。
- `SkillManager`：Agent 私有的索引、条件发现、按需展开、临时权限和恢复控制器。
- `SkillTool`：唯一的模型调用入口，参数是 `skill` 和 `args`。
- `discover_skill_files()`：发现单个 Skill 目录或集合目录的直接子 Skill。
- `load_skill_manifest()`：只解析 frontmatter。
- `load_skill_body()`：调用时读取正文并替换参数。

## Package Layout

```text
skill/
  __init__.py
  base.py           # SkillManifest
  folder_loader.py  # SKILL.md discovery and lazy loading
  manager.py        # Agent-local runtime state
  tool.py           # model-facing skill_tool
```

该包不提供 Python Skill 基类、全局 registry、包装型 meta skill、Python bundle 自动扫描或动态 `tools.py`。Skill 的扩展面是目录协议和可替换 `SkillManager`，不是全局注册中心。

## Minimal Contract

```python
agent.with_skill("./skills")
agent.with_skill("./team-skills", "./project-skills")
agent.with_skill("./team-skills").with_skill("./project-skills")
```

Skill body 的标准生命周期是：`skill_tool` 调用时延迟读取，下一次 request 前作为 user-role `INVOCATION` MetaMessage 注入，Agent invoke 完成、失败或中断时自动回收。
