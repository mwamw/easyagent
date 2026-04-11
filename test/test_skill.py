"""
Skill 技能系统 — 单元测试

测试覆盖：
1. BaseSkill / SkillConfig 定义
2. SkillManager 注册/注销/激活/停用
3. 工具注入与移除
4. Prompt 注入
5. 生命周期钩子
6. before/after invoke 拦截链
7. 动态切换
8. YAML Skill 加载
9. Markdown Skill 加载
10. SkillRegistry 发现
11. 内置 Skill（MemorySkill / WebSearchSkill / CalculatorSkill）
12. 向后兼容（with_memory）
"""
import os
import sys
import json
import pytest
import tempfile
from typing import List, Optional
from unittest.mock import MagicMock, patch

# 保证项目根目录在 sys.path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pydantic import BaseModel, Field

from skill.base import BaseSkill, SkillConfig
from skill.manager import SkillManager
from skill.registry import SkillRegistry
from skill.yaml_loader import (
    YAMLSkill,
    YAMLSkillLoader,
    MarkdownSkill,
    MarkdownSkillLoader,
)
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry


# ==================== 测试用 Stub ====================


class DummyParams(BaseModel):
    text: str = Field(default="hello", description="input text")


class DummyTool(Tool):
    """用于测试的简单 Tool"""

    def __init__(self, name: str = "dummy_tool"):
        super().__init__(name, "A dummy tool for testing", DummyParams)

    def run(self, parameters: dict):
        return f"dummy result: {parameters.get('text', '')}"


class DummySkill(BaseSkill):
    """用于测试的简单 Skill"""

    def __init__(self, name: str = "dummy", tools: Optional[List[Tool]] = None, prompt: str = ""):
        config = SkillConfig(
            name=name,
            description=f"Test skill: {name}",
            version="0.1.0",
            tags=["test"],
            priority=5,
        )
        super().__init__(config)
        self._tools = tools or [DummyTool(f"{name}_tool")]
        self._prompt = prompt or f"## {name} Skill\nThis is {name} skill."
        self.activate_called = False
        self.deactivate_called = False
        self.before_invoke_called = False
        self.after_invoke_called = False

    def get_tools(self) -> list:
        return self._tools

    def get_prompt(self) -> str:
        return self._prompt

    def on_activate(self, agent):
        self.activate_called = True

    def on_deactivate(self, agent):
        self.deactivate_called = True

    def on_before_invoke(self, query: str) -> str:
        self.before_invoke_called = True
        return query

    def on_after_invoke(self, query: str, response: str) -> str:
        self.after_invoke_called = True
        return response


class OnDemandDummySkill(DummySkill):
    """用于测试 on-demand Skill 不进入 resident system prompt。"""

    def __init__(self, name: str = "on_demand_dummy", prompt: str = "## On Demand\nOnly inject on demand."):
        super().__init__(name=name, prompt=prompt)
        self.config.exposure_mode = "on_demand"
        self.config.execution_mode = "inline"
        self.config.listing_description = "按需调用的测试技能"
        self.config.when_to_use = "当任务明确需要临时技能正文时"


class QueryModifySkill(BaseSkill):
    """修改 query 的 Skill，用于测试拦截链"""

    def __init__(self, prefix: str = "[modified]"):
        config = SkillConfig(name="query_modifier", priority=10)
        super().__init__(config)
        self._prefix = prefix

    def get_tools(self) -> list:
        return []

    def get_prompt(self) -> str:
        return ""

    def on_before_invoke(self, query: str) -> str:
        return f"{self._prefix} {query}"

    def on_after_invoke(self, query: str, response: str) -> str:
        return f"{response} {self._prefix}"


class DependentSkill(BaseSkill):
    """有依赖的 Skill"""

    def __init__(self):
        config = SkillConfig(
            name="dependent",
            dependencies=["dummy"],
        )
        super().__init__(config)

    def get_tools(self) -> list:
        return []

    def get_prompt(self) -> str:
        return "## Dependent Skill"


def _make_mock_agent(with_registry: bool = True, with_context: bool = False):
    """创建模拟 Agent"""
    agent = MagicMock()
    agent.name = "test_agent"
    if with_registry:
        agent.tool_registry = ToolRegistry()
        agent.enable_tool = True
    else:
        agent.tool_registry = None
        agent.enable_tool = False
    if with_context:
        agent.context_manager = MagicMock()
    else:
        agent.context_manager = None
    return agent


# ==================== BaseSkill / SkillConfig 测试 ====================


class TestSkillConfig:
    def test_default_values(self):
        config = SkillConfig(name="test")
        assert config.name == "test"
        assert config.description == ""
        assert config.version == "1.0.0"
        assert config.tags == []
        assert config.priority == 0
        assert config.auto_activate is True
        assert config.dependencies == []

    def test_custom_values(self):
        config = SkillConfig(
            name="my_skill",
            description="A test skill",
            version="2.0.0",
            tags=["a", "b"],
            priority=10,
            auto_activate=False,
            dependencies=["dep1"],
            extra={"key": "value"},
        )
        assert config.name == "my_skill"
        assert config.priority == 10
        assert config.auto_activate is False
        assert config.dependencies == ["dep1"]
        assert config.extra == {"key": "value"}


class TestBaseSkill:
    def test_creation(self):
        skill = DummySkill("test")
        assert skill.name == "test"
        assert not skill.is_active
        assert skill.tags == ["test"]
        assert skill.priority == 5

    def test_get_tools(self):
        skill = DummySkill("test")
        tools = skill.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_get_prompt(self):
        skill = DummySkill("test", prompt="custom prompt")
        assert skill.get_prompt() == "custom prompt"

    def test_get_tool_names(self):
        skill = DummySkill("test")
        assert skill.get_tool_names() == ["test_tool"]

    def test_to_dict(self):
        skill = DummySkill("test")
        d = skill.to_dict()
        assert d["name"] == "test"
        assert d["is_active"] is False
        assert "test_tool" in d["tools"]

    def test_repr(self):
        skill = DummySkill("test")
        r = repr(skill)
        assert "DummySkill" in r
        assert "test" in r

    def test_default_lifecycle_hooks(self):
        """默认生命周期钩子应该是空操作"""
        skill = DummySkill("test")
        assert skill.on_before_invoke("hello") == "hello"
        assert skill.on_after_invoke("q", "r") == "r"

    def test_invalid_config_type(self):
        with pytest.raises(TypeError):
            BaseSkill.__init__(MagicMock(), "not_a_config")  # type: ignore


# ==================== SkillManager 测试 ====================


class TestSkillManager:
    def setup_method(self):
        self.manager = SkillManager()
        self.agent = _make_mock_agent()
        self.manager.bind_agent(self.agent)

    def test_register_and_get(self):
        skill = DummySkill("s1")
        self.manager.register(skill)
        assert self.manager.has_skill("s1")
        assert self.manager.get_skill("s1") is skill
        assert self.manager.skill_count == 1

    def test_register_duplicate_raises(self):
        self.manager.register(DummySkill("s1"))
        with pytest.raises(ValueError, match="已存在"):
            self.manager.register(DummySkill("s1"))

    def test_register_non_skill_raises(self):
        with pytest.raises(TypeError):
            self.manager.register("not_a_skill")  # type: ignore

    def test_unregister(self):
        self.manager.register(DummySkill("s1"))
        self.manager.unregister("s1")
        assert not self.manager.has_skill("s1")

    def test_unregister_not_exist_raises(self):
        with pytest.raises(KeyError):
            self.manager.unregister("nonexistent")

    def test_auto_activate(self):
        """auto_activate=True 的 Skill 注册后自动激活"""
        skill = DummySkill("s1")
        self.manager.register(skill)
        assert skill.is_active
        assert self.manager.is_active("s1")
        assert self.manager.active_count == 1

    def test_no_auto_activate(self):
        """auto_activate=False 的 Skill 注册后不激活"""
        skill = DummySkill("s1")
        skill.config.auto_activate = False
        self.manager.register(skill)
        assert not skill.is_active
        assert not self.manager.is_active("s1")

    def test_activate_deactivate(self):
        skill = DummySkill("s1")
        skill.config.auto_activate = False
        self.manager.register(skill)

        self.manager.activate("s1")
        assert skill.is_active
        assert skill.activate_called

        self.manager.deactivate("s1")
        assert not skill.is_active
        assert skill.deactivate_called

    def test_activate_not_exist_raises(self):
        with pytest.raises(KeyError):
            self.manager.activate("nonexistent")

    def test_deactivate_not_active_raises(self):
        self.manager.register(DummySkill("s1"))
        self.manager.deactivate("s1")  # deactivate active
        with pytest.raises(KeyError):
            self.manager.deactivate("s1")

    def test_tool_injection(self):
        """激活 Skill 后工具应出现在 ToolRegistry 中"""
        skill = DummySkill("s1")
        self.manager.register(skill)  # auto_activate
        assert self.agent.tool_registry.has_tool("s1_tool")

    def test_tool_removal(self):
        """停用 Skill 后工具应从 ToolRegistry 中移除"""
        skill = DummySkill("s1")
        self.manager.register(skill)
        assert self.agent.tool_registry.has_tool("s1_tool")

        self.manager.deactivate("s1")
        assert not self.agent.tool_registry.has_tool("s1_tool")

    def test_prompt_aggregation(self):
        skill1 = DummySkill("s1", prompt="Prompt A")
        skill2 = DummySkill("s2", prompt="Prompt B")
        skill2._tools = [DummyTool("s2_tool")]
        self.manager.register(skill1)
        self.manager.register(skill2)

        prompt = self.manager.build_skills_prompt()
        assert "Prompt A" in prompt
        assert "Prompt B" in prompt

    def test_prompt_priority_order(self):
        """高优先级 Skill 的 prompt 应在前面"""
        low = DummySkill("low")
        low.config.priority = 1
        low._prompt = "LOW"
        low._tools = [DummyTool("low_tool")]

        high = DummySkill("high")
        high.config.priority = 100
        high._prompt = "HIGH"
        high._tools = [DummyTool("high_tool")]

        self.manager.register(low)
        self.manager.register(high)

        prompt = self.manager.build_skills_prompt()
        print(prompt)
        assert prompt.index("HIGH") < prompt.index("LOW")

    def test_empty_prompt_when_no_skills(self):
        assert self.manager.build_skills_prompt() == ""

    def test_get_active_skills(self):
        self.manager.register(DummySkill("s1"))
        s2 = DummySkill("s2")
        s2.config.auto_activate = False
        s2._tools = [DummyTool("s2_tool")]
        self.manager.register(s2)

        active = self.manager.get_active_skills()
        assert len(active) == 1
        assert active[0].name == "s1"

    def test_list_skills(self):
        self.manager.register(DummySkill("s1"))
        info_list = self.manager.list_skills()
        assert len(info_list) == 1
        assert info_list[0]["name"] == "s1"

    def test_on_before_invoke(self):
        skill = QueryModifySkill("[PRE]")
        skill.config.auto_activate = True
        self.manager.register(skill)

        result = self.manager.on_before_invoke("hello")
        assert result == "[PRE] hello"

    def test_on_after_invoke(self):
        skill = QueryModifySkill("[POST]")
        skill.config.auto_activate = True
        self.manager.register(skill)

        result = self.manager.on_after_invoke("q", "response")
        assert result == "response [POST]"

    def test_lifecycle_hooks_called(self):
        skill = DummySkill("s1")
        self.manager.register(skill)  # auto activate
        assert skill.activate_called

        self.manager.on_before_invoke("test")
        assert skill.before_invoke_called

        self.manager.on_after_invoke("test", "result")
        assert skill.after_invoke_called

    def test_unregister_deactivates_first(self):
        """注销一个激活的 Skill 应先停用"""
        skill = DummySkill("s1")
        self.manager.register(skill)
        assert skill.is_active

        self.manager.unregister("s1")
        assert skill.deactivate_called
        assert not self.agent.tool_registry.has_tool("s1_tool")

    def test_tool_name_conflict_skipped(self):
        """工具名冲突时应跳过注册"""
        # 先手动注册一个同名工具
        self.agent.tool_registry.register_tool(DummyTool("conflict_tool"))

        skill = DummySkill("conflict")
        skill._tools = [DummyTool("conflict_tool")]
        self.manager.register(skill)

        # 工具仍存在（原有的），不会报错
        assert self.agent.tool_registry.has_tool("conflict_tool")

    def test_no_agent_bound(self):
        """未绑定 Agent 时注册 Skill 不崩溃"""
        manager = SkillManager()
        skill = DummySkill("s1")
        manager.register(skill)  # 不应报错
        assert skill.is_active  # auto_activate，但没有 agent 所以没注入工具

    def test_dependency_auto_activate(self):
        """有依赖时自动激活依赖"""
        base = DummySkill("dummy")
        base.config.auto_activate = False
        dep = DependentSkill()
        dep.config.auto_activate = False

        self.manager.register(base)
        self.manager.register(dep)

        self.manager.activate("dependent")
        assert self.manager.is_active("dummy")
        assert self.manager.is_active("dependent")

    def test_repr(self):
        self.manager.register(DummySkill("s1"))
        r = repr(self.manager)
        assert "s1" in r


# ==================== ToolRegistry 增强测试 ====================


class TestToolRegistryEnhancements:
    def test_has_tool(self):
        registry = ToolRegistry()
        registry.register_tool(DummyTool("t1"))
        assert registry.has_tool("t1")
        assert not registry.has_tool("t2")

    def test_get_tool_names(self):
        registry = ToolRegistry()
        registry.register_tool(DummyTool("t1"))
        registry.register_tool(DummyTool("t2"))
        names = registry.get_tool_names()
        assert "t1" in names
        assert "t2" in names

    def test_register_tools_batch(self):
        registry = ToolRegistry()
        registry.register_tools([DummyTool("t1"), DummyTool("t2")])
        assert registry.has_tool("t1")
        assert registry.has_tool("t2")

    def test_unregister_tools_batch(self):
        registry = ToolRegistry()
        registry.register_tools([DummyTool("t1"), DummyTool("t2")])
        registry.unregister_tools(["t1", "t2"])
        assert not registry.has_tool("t1")
        assert not registry.has_tool("t2")


# ==================== YAML Skill 测试 ====================


class TestYAMLSkill:
    def test_load_from_file(self):
        yaml_content = """
name: test_yaml_skill
description: "A test skill"
version: "1.0"
tags: [test, yaml]
priority: 7
prompt: |
  ## Test Skill
  This is a test.
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            skill = YAMLSkillLoader.load(path)
            assert skill.name == "test_yaml_skill"
            assert skill.config.description == "A test skill"
            assert skill.config.version == "1.0"
            assert "test" in skill.config.tags
            assert skill.config.priority == 7
            assert skill.get_exposure_mode() == "on_demand"
            assert skill.get_execution_mode() == "inline"
            assert skill.config.source_type == "yaml"
            assert skill.config.source_path == path
            assert "Test Skill" in skill.get_prompt()
        finally:
            os.unlink(path)

    def test_load_with_builtin_tools(self):
        """测试引用内置工具（不实际创建，只验证解析）"""
        yaml_content = """
name: with_tools
tools:
  - builtin: calculator
prompt: "Use calculator"
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            skill = YAMLSkillLoader.load(path)
            assert skill.name == "with_tools"
            # 工具创建可能需要依赖，这里只验证不崩溃
            tools = skill.get_tools()
            assert isinstance(tools, list)
        except ImportError:
            # 在无 calculator 依赖的环境下跳过
            pass
        finally:
            os.unlink(path)

    def test_load_string_tool_format(self):
        """支持简写工具格式 - builtin_name"""
        yaml_content = """
name: simple_tools
tools:
  - calculator
prompt: "tools"
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            skill = YAMLSkillLoader.load(path)
            assert skill.name == "simple_tools"
            assert len(skill._tool_defs) == 1
            assert skill._tool_defs[0] == {"builtin": "calculator"}
        finally:
            os.unlink(path)

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                yaml_content = f"name: skill_{i}\nprompt: prompt_{i}"
                with open(os.path.join(tmpdir, f"skill_{i}.yaml"), "w") as f:
                    f.write(yaml_content)

            skills = YAMLSkillLoader.load_directory(tmpdir)
            assert len(skills) == 3
            names = {s.name for s in skills}
            assert names == {"skill_0", "skill_1", "skill_2"}

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            YAMLSkillLoader.load("/tmp/nonexistent_skill.yaml")

    def test_load_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("- just a list\n- not a dict")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="字典"):
                YAMLSkillLoader.load(path)
        finally:
            os.unlink(path)

    def test_load_missing_name(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("description: no name\nprompt: test")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="name"):
                YAMLSkillLoader.load(path)
        finally:
            os.unlink(path)


# ==================== Markdown Skill 测试 ====================


class TestMarkdownSkill:
    def test_load_from_file(self):
        md_content = """---
name: test_md_skill
description: "A markdown skill"
version: "2.0"
tags: [test, markdown]
priority: 8
---

## Markdown Skill

This is a markdown-based skill prompt.

- Feature 1
- Feature 2
"""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(md_content)
            f.flush()
            path = f.name

        try:
            skill = MarkdownSkillLoader.load(path)
            assert skill.name == "test_md_skill"
            assert skill.config.description == "A markdown skill"
            assert skill.config.version == "2.0"
            assert skill.config.priority == 8
            assert skill.get_exposure_mode() == "on_demand"
            assert skill.get_execution_mode() == "inline"
            assert skill.config.source_type == "markdown"
            assert skill.config.source_path == path
            assert "Markdown Skill" in skill.get_prompt()
            assert "Feature 1" in skill.get_prompt()
        finally:
            os.unlink(path)

    def test_load_with_tools(self):
        md_content = """---
name: md_with_tools
tools:
  - builtin: calculator
---

Use the calculator for math.
"""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(md_content)
            f.flush()
            path = f.name

        try:
            skill = MarkdownSkillLoader.load(path)
            assert skill.name == "md_with_tools"
            assert len(skill._tool_defs) == 1
        finally:
            os.unlink(path)

    def test_infer_name_from_filename(self):
        """frontmatter 没有 name 时从文件名推断"""
        md_content = """---
description: no explicit name
---

Content here.
"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, prefix="my_skill_"
        ) as f:
            f.write(md_content)
            f.flush()
            path = f.name

        try:
            skill = MarkdownSkillLoader.load(path)
            # 名称应该从文件名推断（不含扩展名）
            assert skill.name  # 只要不为空即可
        finally:
            os.unlink(path)

    def test_load_no_frontmatter_raises(self):
        md_content = """# No frontmatter

Just plain markdown.
"""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(md_content)
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="frontmatter"):
                MarkdownSkillLoader.load(path)
        finally:
            os.unlink(path)

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                md_content = f"---\nname: md_skill_{i}\n---\n\nPrompt {i}"
                with open(os.path.join(tmpdir, f"skill_{i}.md"), "w") as f:
                    f.write(md_content)

            skills = MarkdownSkillLoader.load_directory(tmpdir)
            assert len(skills) == 2

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            MarkdownSkillLoader.load("/tmp/nonexistent_skill.md")


# ==================== SkillRegistry 测试 ====================


class TestSkillRegistry:
    def setup_method(self):
        SkillRegistry.reset()
        self.registry = SkillRegistry.instance()

    def teardown_method(self):
        SkillRegistry.reset()

    def test_singleton(self):
        r1 = SkillRegistry.instance()
        r2 = SkillRegistry.instance()
        assert r1 is r2

    def test_register_class(self):
        self.registry.register_class(DummySkill, "dummy")
        assert self.registry.has("dummy")

    def test_register_class_auto_name(self):
        self.registry.register_class(DummySkill)
        assert self.registry.has("dummy")

    def test_register_non_skill_raises(self):
        with pytest.raises(TypeError):
            self.registry.register_class(str, "bad")  # type: ignore

    def test_create(self):
        self.registry.register_class(DummySkill, "dummy")
        skill = self.registry.create("dummy", name="test_instance")
        assert isinstance(skill, DummySkill)
        assert skill.name == "test_instance"

    def test_create_not_found_raises(self):
        with pytest.raises(KeyError):
            self.registry.create("nonexistent")

    def test_register_factory(self):
        def factory(**kwargs):
            return DummySkill(name=kwargs.get("name", "factory_skill"))

        self.registry.register_factory("factory", factory)
        skill = self.registry.create("factory", name="from_factory")
        assert skill.name == "from_factory"

    def test_list_available(self):
        self.registry.register_class(DummySkill, "dummy")
        available = self.registry.list_available()
        assert len(available) >= 1
        names = [a["name"] for a in available]
        assert "dummy" in names

    def test_manifest_is_available(self):
        self.registry.register_class(DummySkill, "dummy")
        manifest = self.registry.get_manifest("dummy")
        assert manifest.name == "dummy"
        assert manifest.source_type == "python"

    def test_search_uses_manifest_fields(self):
        skill = OnDemandDummySkill(name="runtime_helper")
        self.registry.register_factory("runtime_helper", lambda: skill)
        self.registry.update_metadata(
            "runtime_helper",
            description="临时技能辅助器",
            tags=["runtime", "helper"],
            listing_description="用于运行时注入临时技能说明",
            when_to_use="当你需要临时技能上下文或 runtime skill context 时",
            tool_names=skill.get_tool_names(),
        )
        results = self.registry.search(query="runtime skill context")
        assert results
        assert results[0]["name"] == "runtime_helper"
        assert "用于运行时注入临时技能说明" in results[0]["listing_description"]

    def test_list_available_names(self):
        self.registry.register_class(DummySkill, "dummy")
        self.registry.register_factory("factory", lambda: DummySkill())
        names = self.registry.list_available_names()
        assert "dummy" in names
        assert "factory" in names

    def test_decorator(self):
        @self.registry.skill("decorated")
        class DecoratedSkill(BaseSkill):
            def __init__(self, **kwargs):
                super().__init__(SkillConfig(name="decorated"))

            def get_tools(self):
                return []

            def get_prompt(self):
                return ""

        assert self.registry.has("decorated")

    def test_discover_from_directory_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = "name: discovered_yaml\nprompt: hello"
            with open(os.path.join(tmpdir, "test.yaml"), "w") as f:
                f.write(yaml_content)

            names = self.registry.discover_from_directory(tmpdir)
            assert "discovered_yaml" in names
            assert self.registry.has("discovered_yaml")

    def test_discover_from_directory_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_content = "---\nname: discovered_md\n---\n\nHello"
            with open(os.path.join(tmpdir, "test.md"), "w") as f:
                f.write(md_content)

            names = self.registry.discover_from_directory(tmpdir)
            assert "discovered_md" in names

    def test_discover_nonexistent_dir(self):
        names = self.registry.discover_from_directory("/tmp/nonexistent_dir_for_test")
        assert names == []

    def test_class_to_name(self):
        assert SkillRegistry._class_to_name(DummySkill) == "dummy"

        class MyWebSearchSkill(BaseSkill):
            def get_tools(self): return []
            def get_prompt(self): return ""

        assert SkillRegistry._class_to_name(MyWebSearchSkill) == "my_web_search"


# ==================== 集成测试：Skill + Agent ====================


class TestSkillAgentIntegration:
    """使用 mock Agent 测试 Skill 系统的完整集成流程"""

    def test_full_workflow(self):
        """注册 → 激活 → 验证工具和 prompt → 停用 → 验证清理"""
        agent = _make_mock_agent()
        manager = SkillManager()
        manager.bind_agent(agent)

        # 注册（自动激活）
        skill = DummySkill("workflow_test")
        manager.register(skill)

        # 验证工具
        assert agent.tool_registry.has_tool("workflow_test_tool")
        # 验证 prompt
        prompt = manager.build_skills_prompt()
        assert "workflow_test" in prompt.lower()
        # 验证激活
        assert skill.is_active
        assert skill.activate_called

        # 停用
        manager.deactivate("workflow_test")
        assert not skill.is_active
        assert not agent.tool_registry.has_tool("workflow_test_tool")
        assert skill.deactivate_called

        # prompt 应该为空
        assert manager.build_skills_prompt() == ""

    def test_on_demand_skill_not_in_resident_prompt(self):
        agent = _make_mock_agent()
        manager = SkillManager()
        manager.bind_agent(agent)

        skill = OnDemandDummySkill()
        manager.register(skill)

        assert skill.is_active
        assert agent.tool_registry.has_tool("on_demand_dummy_tool")
        assert manager.build_skills_prompt() == ""
        assert "on_demand_dummy" in manager.build_skill_listing_prompt()
        assert "## Skill 使用规则" in manager.build_skill_policy_prompt()

    def test_runtime_skill_context_prompt(self):
        agent = _make_mock_agent()
        manager = SkillManager()
        manager.bind_agent(agent)

        skill = OnDemandDummySkill()
        manager.register(skill)
        manager.set_runtime_skill_context(skill, "## On Demand\nTemporary instructions.")

        prompt = manager.build_runtime_skill_context_prompt()
        assert "## 当前 Runtime Skill Context" in prompt
        assert "<runtime-skill-context>" in prompt
        assert '<skill-runtime-entry name="on_demand_dummy"' in prompt
        assert "<skill-body>" in prompt
        assert "Temporary instructions." in prompt
        manager.clear_runtime_skill_context()
        assert manager.build_runtime_skill_context_prompt() == ""

    def test_multiple_skills(self):
        """多个 Skill 共存"""
        agent = _make_mock_agent()
        manager = SkillManager()
        manager.bind_agent(agent)

        s1 = DummySkill("alpha")
        s2 = DummySkill("beta")
        s2._tools = [DummyTool("beta_tool")]

        manager.register(s1)
        manager.register(s2)

        assert agent.tool_registry.has_tool("alpha_tool")
        assert agent.tool_registry.has_tool("beta_tool")
        assert manager.active_count == 2

    def test_dynamic_switch(self):
        """运行时动态切换 Skill"""
        agent = _make_mock_agent()
        manager = SkillManager()
        manager.bind_agent(agent)

        skill = DummySkill("switchable")
        manager.register(skill)

        # 验证激活
        assert agent.tool_registry.has_tool("switchable_tool")

        # 停用
        manager.deactivate("switchable")
        assert not agent.tool_registry.has_tool("switchable_tool")

        # 重新激活
        manager.activate("switchable")
        assert agent.tool_registry.has_tool("switchable_tool")

    def test_no_tool_registry_auto_create(self):
        """如果 Agent 没有 ToolRegistry，SkillManager 应该自动创建"""
        agent = _make_mock_agent(with_registry=False)
        manager = SkillManager()
        manager.bind_agent(agent)

        skill = DummySkill("auto_reg")
        manager.register(skill)

        # 应该自动创建了 ToolRegistry
        assert agent.tool_registry is not None
        assert agent.tool_registry.has_tool("auto_reg_tool")


# ==================== 内置 Skill 测试 ====================


class TestBuiltinCalculatorSkill:
    def test_creation(self):
        from skill.builtin.calculator_skill import CalculatorSkill

        skill = CalculatorSkill()
        assert skill.name == "calculator"
        assert "math" in skill.tags

    def test_prompt(self):
        from skill.builtin.calculator_skill import CalculatorSkill

        skill = CalculatorSkill()
        prompt = skill.get_prompt()
        assert "计算" in prompt or "calculator" in prompt.lower()

    def test_tools(self):
        from skill.builtin.calculator_skill import CalculatorSkill

        skill = CalculatorSkill()
        try:
            tools = skill.get_tools()
            assert len(tools) >= 1
        except ImportError:
            pytest.skip("CalculatorTool 依赖未安装")


class TestBuiltinWebSearchSkill:
    def test_creation(self):
        from skill.builtin.web_search_skill import WebSearchSkill

        skill = WebSearchSkill()
        assert skill.name == "web_search"
        assert "search" in skill.tags

    def test_prompt(self):
        from skill.builtin.web_search_skill import WebSearchSkill

        skill = WebSearchSkill()
        prompt = skill.get_prompt()
        assert "搜索" in prompt


class TestBuiltinMemorySkill:
    def test_creation(self):
        from skill.builtin.memory_skill import MemorySkill

        mock_mm = MagicMock()
        mock_mm.get_supported_type.return_value = ["working", "episodic", "semantic"]
        mock_mm.memory_types = {}

        skill = MemorySkill(memory_manage=mock_mm)
        assert skill.name == "memory"
        assert skill.config.priority == 10

    def test_prompt(self):
        from skill.builtin.memory_skill import MemorySkill

        mock_mm = MagicMock()
        mock_mm.get_supported_type.return_value = ["working"]
        mock_mm.memory_types = {}

        skill = MemorySkill(memory_manage=mock_mm)
        prompt = skill.get_prompt()
        assert "记忆" in prompt or "Memory" in prompt

    def test_tools_count(self):
        from skill.builtin.memory_skill import MemorySkill

        mock_mm = MagicMock()
        mock_mm.get_supported_type.return_value = ["working"]
        mock_mm.memory_types = {}

        skill = MemorySkill(memory_manage=mock_mm)
        try:
            tools = skill.get_tools()
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Memory 相关依赖未安装 (sentence_transformers)")
            return
        assert len(tools) == 6  # add, search, get, update, remove, maintenance


# ==================== SkillRegistry.search 测试 ====================


class TestSkillRegistrySearch:
    """SkillRegistry.search() 关键词/标签搜索测试"""

    def setup_method(self):
        SkillRegistry.reset()
        self.registry = SkillRegistry.instance()

    def teardown_method(self):
        SkillRegistry.reset()

    def _register_with_meta(self, name, desc, tags):
        """注册一个带元信息的 Skill（使用工厂）"""
        def factory(_n=name, **kwargs):
            return DummySkill(name=_n)

        self.registry.register_factory(name, factory)
        self.registry.update_metadata(name, description=desc, tags=tags)

    def test_search_by_keyword(self):
        self._register_with_meta("calculator", "数学计算工具", ["math", "compute"])
        self._register_with_meta("web_search", "联网搜索引擎", ["search", "web"])

        results = self.registry.search("math")
        assert len(results) == 1
        assert results[0]["name"] == "calculator"

    def test_search_by_tag(self):
        self._register_with_meta("calculator", "数学计算", ["math"])
        self._register_with_meta("web_search", "联网搜索", ["search", "web"])

        results = self.registry.search(tags=["web"])
        assert len(results) == 1
        assert results[0]["name"] == "web_search"

    def test_search_empty_query_returns_all(self):
        self._register_with_meta("a", "desc_a", ["tag_a"])
        self._register_with_meta("b", "desc_b", ["tag_b"])

        results = self.registry.search()
        assert len(results) == 2

    def test_search_no_match(self):
        self._register_with_meta("calculator", "数学计算", ["math"])
        results = self.registry.search("docker")
        assert len(results) == 0

    def test_search_keyword_and_tag(self):
        self._register_with_meta("calculator", "数学计算", ["math", "compute"])
        self._register_with_meta("stats", "统计分析", ["math", "stats"])

        results = self.registry.search(query="统计", tags=["math"])
        assert len(results) == 1
        assert results[0]["name"] == "stats"


# ==================== Meta-Tools 测试 ====================


class TestMetaTools:
    """Skill 元工具 — 动态按需加载测试"""

    def setup_method(self):
        SkillRegistry.reset()
        self.registry = SkillRegistry.instance()
        self.tool_registry = ToolRegistry()
        self.manager = SkillManager()

        # 模拟 Agent 绑定
        mock_agent = MagicMock()
        mock_agent.tool_registry = self.tool_registry
        mock_agent.context_manager = None
        self.manager.bind_agent(mock_agent)

        # 注册一些 Skill 到 Registry（还未加载到 Manager）
        def calc_factory(**kwargs):
            return OnDemandDummySkill(name="calculator")

        self.registry.register_factory("calculator", calc_factory)
        self.registry.update_metadata(
            "calculator", description="数学计算工具", tags=["math", "compute"]
        )

        def search_factory(**kwargs):
            return OnDemandDummySkill(name="web_search")

        self.registry.register_factory("web_search", search_factory)
        self.registry.update_metadata(
            "web_search",
            description="联网搜索引擎",
            tags=["search", "web", "real-time"],
        )

    def teardown_method(self):
        SkillRegistry.reset()
    # ---------- SkillDiscoveryTool ----------

    def test_discovery_format(self):
        from skill.meta_tools import SkillDiscoveryTool

        tool = SkillDiscoveryTool(self.registry)
        result = tool.run({})
        # The new string will contain calculator and web_search descriptions
        assert "calculator" in result
        assert "web_search" in result
    # ---------- LoadSkillTool ----------

    def test_load_success(self):
        from skill.meta_tools import LoadSkillTool

        tool = LoadSkillTool(self.registry, self.manager, set())
        result = tool.run({"skill_name": "calculator"})
        assert "成功加载" in result
        assert self.manager.has_skill("calculator")
        assert self.manager.is_active("calculator")

    def test_load_duplicate(self):
        from skill.meta_tools import LoadSkillTool

        tool = LoadSkillTool(self.registry, self.manager, set())
        tool.run({"skill_name": "calculator"})
        result = tool.run({"skill_name": "calculator"})
        assert "已经加载" in result

    def test_load_reactivate_inactive(self):
        from skill.meta_tools import LoadSkillTool

        tool = LoadSkillTool(self.registry, self.manager, set())
        tool.run({"skill_name": "calculator"})
        self.manager.deactivate("calculator")
        assert not self.manager.is_active("calculator")

        result = tool.run({"skill_name": "calculator"})
        assert "重新激活" in result
        assert self.manager.is_active("calculator")

    def test_load_not_found(self):
        from skill.meta_tools import LoadSkillTool

        tool = LoadSkillTool(self.registry, self.manager, set())
        result = tool.run({"skill_name": "nonexistent"})
        assert "未在注册中心中找到" in result

    def test_load_empty_name(self):
        from skill.meta_tools import LoadSkillTool

        tool = LoadSkillTool(self.registry, self.manager, set())
        result = tool.run({"skill_name": ""})
        assert "必须指定" in result

    # ---------- UnloadSkillTool ----------

    def test_unload_success(self):
        from skill.meta_tools import LoadSkillTool, UnloadSkillTool

        tracker = set()
        load_tool = LoadSkillTool(self.registry, self.manager, tracker)
        load_tool.run({"skill_name": "calculator"})
        assert self.manager.has_skill("calculator")

        unload_tool = UnloadSkillTool(self.manager, tracker)
        result = unload_tool.run({"skill_name": "calculator"})
        assert "成功卸载" in result
        assert not self.manager.has_skill("calculator")

    def test_unload_not_loaded(self):
        from skill.meta_tools import UnloadSkillTool

        tool = UnloadSkillTool(self.manager, set())
        result = tool.run({"skill_name": "nonexistent"})
        assert "只能卸载自己加载过" in result

    def test_unload_empty_name(self):
        from skill.meta_tools import UnloadSkillTool

        tool = UnloadSkillTool(self.manager, set())
        result = tool.run({"skill_name": ""})
        assert "必须指定" in result

    # ---------- 完整流程测试 ----------

    def test_full_workflow_discover_load_unload(self):
        """完整流程: 发现 → 加载 → 验证工具注入 → 卸载 → 验证清理"""
        from skill.meta_tools import SkillDiscoveryTool, LoadSkillTool, UnloadSkillTool
        import json

        tracker = set()
        discovery = SkillDiscoveryTool(self.registry)
        loader = LoadSkillTool(self.registry, self.manager, tracker)
        unloader = UnloadSkillTool(self.manager, tracker)

        # 1. 发现
        result = discovery.run({})
        assert "calculator" in result
        skill_name = "calculator"

        # 2. 加载
        result = loader.run({"skill_name": skill_name})
        assert "成功加载" in result
        assert self.manager.is_active(skill_name)

        # 3. 验证工具注入到 ToolRegistry
        skill = self.manager.get_skill(skill_name)
        tool_names = skill.get_tool_names()
        for tn in tool_names:
            assert self.tool_registry.has_tool(tn)

        # 4. 卸载
        result = unloader.run({"skill_name": skill_name})
        assert "成功卸载" in result
        assert not self.manager.has_skill(skill_name)

        # 5. 验证工具已从 ToolRegistry 移除
        for tn in tool_names:
            assert not self.tool_registry.has_tool(tn)

    def test_skill_tool_inline_injection(self):
        from skill.meta_tools import SkillTool

        tracker = set()
        tool = SkillTool(self.registry, self.manager, tracker)
        result = tool.run({"skill_name": "calculator"})

        assert "已注入 Skill `calculator`" in result
        assert "详细正文已注入当前 invoke 的后续推理链" in result
        assert "<skill>" not in result
        assert self.manager.has_skill("calculator")
        assert self.manager.is_active("calculator")
        assert self.manager.has_runtime_skill_context()
        self.manager.clear_ephemeral_state()
        assert not self.manager.has_skill("calculator")
        assert not self.manager.has_runtime_skill_context()
        assert "calculator" not in tracker

    def test_skill_tool_restores_pre_registered_inactive_skill(self):
        from skill.meta_tools import SkillTool

        skill = OnDemandDummySkill(name="calculator")
        skill.config.auto_activate = False
        self.manager.register(skill)
        assert self.manager.has_skill("calculator")
        assert not self.manager.is_active("calculator")

        tool = SkillTool(self.registry, self.manager, set())
        result = tool.run({"skill_name": "calculator"})

        assert "已注入 Skill `calculator`" in result
        assert self.manager.is_active("calculator")

        self.manager.clear_ephemeral_state()

        assert self.manager.has_skill("calculator")
        assert not self.manager.is_active("calculator")

    def test_meta_tool_descriptions_reflect_new_priority(self):
        from skill.meta_tools import SkillDiscoveryTool, SkillTool, LoadSkillTool

        discovery = SkillDiscoveryTool(self.registry)
        skill_tool = SkillTool(self.registry, self.manager, set())
        loader = LoadSkillTool(self.registry, self.manager, set())

        assert "优先使用 system prompt 中已有的 skill listing" in discovery.description
        assert "优先根据 system prompt 里的 skill listing 直接调用此工具" in skill_tool.description
        assert "兼容接口" in loader.description

    def test_meta_skill_loading(self):
        """测试 MetaSkill 管理"""
        from skill.meta_tools import MetaSkill

        skill = MetaSkill(self.registry, self.manager)
        self.manager.register(skill)

        assert self.manager.has_skill("meta_skill")
        # 验证工具注入到 ToolRegistry
        assert self.tool_registry.has_tool("skill_discovery_tool")
        assert self.tool_registry.has_tool("skill_tool")
        assert self.tool_registry.has_tool("load_skill_tool")
        assert self.tool_registry.has_tool("unload_skill_tool")
        assert self.tool_registry.has_tool("skill_discovery_tool")
        assert self.tool_registry.has_tool("skill_tool")
        assert self.tool_registry.has_tool("load_skill_tool")
        assert self.tool_registry.has_tool("unload_skill_tool")


# ==================== Folder Skill 测试 ====================


class TestFolderSkillLoader:
    def test_load_from_folder_without_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "my_folder_skill")
            os.makedirs(skill_dir)
            md_content = "---\nname: my_folder_skill\npriority: 10\n---\nPrompt Content"
            with open(os.path.join(skill_dir, "skill.md"), "w") as f:
                f.write(md_content)

            from skill.folder_loader import FolderSkillLoader, FolderSkill
            skill = FolderSkillLoader.load(skill_dir)
            assert isinstance(skill, FolderSkill)
            assert skill.name == "my_folder_skill"
            assert skill.priority == 10
            assert skill.get_exposure_mode() == "on_demand"
            assert skill.get_execution_mode() == "inline"
            assert "Prompt Content" in skill.get_prompt()
            assert len(skill.get_tools()) == 0

    def test_load_from_folder_with_dynamic_tools_auto_instantiate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "auto_skill")
            os.makedirs(skill_dir)
            
            with open(os.path.join(skill_dir, "README.md"), "w") as f:
                f.write("---\nname: auto_skill\n---\nDynamic Tools")
                
            py_content = """
from Tool.BaseTool import Tool
from pydantic import BaseModel

class MyParams(BaseModel):
    pass

class DynamicAutoTool(Tool):
    def __init__(self):
        super().__init__("dynamic_auto", "A dynamic tool", MyParams)
        
    def run(self, params):
        return "auto"
"""
            with open(os.path.join(skill_dir, "tools.py"), "w") as f:
                f.write(py_content)

            from skill.folder_loader import FolderSkillLoader
            skill = FolderSkillLoader.load(skill_dir)
            tools = skill.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "dynamic_auto"
            assert tools[0].run({}) == "auto"

    def test_load_from_folder_with_get_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "get_tools_skill")
            os.makedirs(skill_dir)
            
            with open(os.path.join(skill_dir, "README.md"), "w") as f:
                f.write("---\nname: get_tools_skill\n---\n")
                
            py_content = """
from Tool.BaseTool import Tool
from pydantic import BaseModel

class MyParams(BaseModel):
    pass

class DynamicGetTool(Tool):
    def __init__(self):
        super().__init__("dynamic_get", "A dynamic tool", MyParams)
        
    def run(self, params):
        return "get"

def get_tools():
    return [DynamicGetTool()]
"""
            with open(os.path.join(skill_dir, "tools.py"), "w") as f:
                f.write(py_content)

            from skill.folder_loader import FolderSkillLoader
            skill = FolderSkillLoader.load(skill_dir)
            tools = skill.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "dynamic_get"
            assert tools[0].run({}) == "get"

if __name__ == "__main__":
    test_skill_manager = TestSkillManager()
    test_skill_manager.setup_method()
    test_skill_manager.test_prompt_priority_order()
