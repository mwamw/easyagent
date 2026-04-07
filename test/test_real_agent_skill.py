import sys
import os
import tempfile
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 将项目根目录加入模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM
from skill.registry import SkillRegistry
from skill.builtin.calculator_skill import CalculatorSkill
from skill.yaml_loader import YAMLSkillLoader, MarkdownSkillLoader
from skill.folder_loader import FolderSkillLoader
from skill.meta_tools import MetaSkill

class TestSkillBase:
    """提供通用的测试辅助方法"""
    def __init__(self):
        print("\n" + "="*80)
        print(f"🚀 开始测试套件: {self.__class__.__name__}")
        print("="*80)
        # 初始化真实的 LLM
        try:
            self.llm = EasyLLM()
            print(f"✅ LLM 加载成功 -> {self.llm.provide} | {self.llm.model}\n")
        except Exception as e:
            print(f"❌ LLM 加载失败: {e}\n")
            self.llm = None
            
        self.registry = SkillRegistry.instance()

    def run_agent_interaction(self, agent: BasicAgent, query: str):
        print(f"🙋 真实场景输入 >> {query}")
        print("-" * 80)
        print("⏳ Agent 正在思考和调用工具中...\n")
        try:
            response = agent.invoke(query)
            print("-" * 80)
            print(f"✨ Agent 返回 >> {response}")
        except Exception as e:
            print(f"❌ 运行过程中出现错误: {e}")
        print("=" * 80 + "\n")


class TestBuiltinSkills(TestSkillBase):
    """测试原生的内建 Skill"""
    
    def run(self):
        print(">>> 正在进行内建 Skill (CalculatorSkill) 的真实调用测试")
        agent = BasicAgent(
            name="Math Assistant",
            llm=self.llm,
            system_prompt="你是一个严谨的计算助手。必须使用提供的工具进行计算，不要自己凭空算。",
            verbose_thinking=False
        )
        
        # 挂载计算技能
        agent.with_skill(CalculatorSkill())
        
        # 互动测试
        self.run_agent_interaction(agent, "你能帮我计算 173 乘以 294 的结果吗？")


class TestYAMLSkill(TestSkillBase):
    """测试通过 YAML 加载的零代码 Skill"""
    
    def run(self):
        print(">>> 正在进行 YAML Skill 定义与加载的真实测试")
        yaml_content = """
name: test_yaml_math
description: 基础 YAML 数学技能
priority: 10
tools:
  - builtin: calculator
 prompt: |
  ## YAML 注入提示词
  系统已赋予你通过 YAML 定义的运算技能，随时使用工具回答任何数学问题。
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
            
        try:
            skill = YAMLSkillLoader.load(yaml_path)
            agent = BasicAgent(
                name="YAML Math Assistant",
                llm=self.llm,
                system_prompt="你是一个简单的助手。",
                verbose_thinking=False
            )
            agent.with_skill(skill)
            
            self.run_agent_interaction(agent, "请使用工具计算 2 的 16 次方是多少？")
        finally:
            os.unlink(yaml_path)


class TestMarkdownSkill(TestSkillBase):
    """测试通过 Markdown Frontmatter 加载的 Skill"""
    
    def run(self):
        print(">>> 正在进行 Markdown Skill 定义与加载的真实测试")
        md_content = """---
name: test_md_assistant
description: "Markdown助理"
priority: 8
tools:
  - calculator
---

## 你的工作指南
作为 Markdown-based Assistant，务必在回答数学问题前使用算术工具。
"""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(md_content)
            md_path = f.name
            
        try:
            skill = MarkdownSkillLoader.load(md_path)
            agent = BasicAgent(
                name="MD Assistant",
                llm=self.llm,
                system_prompt="你是一个智能小助手。",
                verbose_thinking=False
            )
            agent.with_skill(skill)
            
            self.run_agent_interaction(agent, "请计算 314 的平方")
        finally:
            os.unlink(md_path)


class TestFolderSkill(TestSkillBase):
    """测试 Claude Code 风格的 Folder-based Skill (包含动态 Python 工具加载)"""
    
    def run(self):
        print(">>> 正在进行 Folder-based Skill (测试目录 real_skills) 的真实调用测试")
        
        # 扫描并获取先前实现的 crypto_skill
        skills_dir = os.path.join(os.path.dirname(__file__), "real_skills")
        discovered = self.registry.discover_from_directory(skills_dir)
        
        if "crypto_skill" not in discovered:
            print("⚠️ 未找到 crypto_skill，请确保运行了生成该技能步骤的代码。跳过此测试。")
            return
            
        skill = self.registry.create("crypto_skill")
        agent = BasicAgent(
            name="Crypto Expert",
            llm=self.llm,
            system_prompt="你是一个加密技术极客。遇到哈希任务务必使用你的自定义工具。",
            verbose_thinking=False
        )
        agent.with_skill(skill)
        
        self.run_agent_interaction(agent, "你能用内置的加密工具帮我把 'OpenAI GPT-4' 计算一下 SHA-256 哈希值吗？")


class TestMetaToolsSkill(TestSkillBase):
    """测试 Meta Tools (Skill Discovery, Load, Unload) 能力 - 第七高级用法 / 模式 B"""
    
    def run(self):
        print(">>> 正在进行 Meta Tools (动态发现与加载) 模式的真实代理测试")
        
        # 预先向 Registry 注入一些知识和工具 (只注册，不加载到 Agent)
        self.registry.register_class(CalculatorSkill)
        self.registry.update_metadata("calculator", description="可以进行数学算数和公式计算")
        
        skills_dir = os.path.join(os.path.dirname(__file__), "real_skills")
        self.registry.discover_from_directory(skills_dir)
        
        agent = BasicAgent(
            name="Autonomous Agent",
            llm=self.llm,
            system_prompt=(
                "你是一个具有自主意识的 Agent。如果用户交给你的任务你目前没有合适的工具去完成，"
                "请首先使用 skill_discovery_tool 寻找对应的技能。找到后使用 load_skill_tool 加载它，然后再使用新装载的工具完成任务。"
            ),
            verbose_thinking=False
        )
        
        # 为 Agent 加载 MetaSkill，让其获得动态加载的能力
        meta_skill = MetaSkill(self.registry, agent.skill_manager)
        agent.with_skill(meta_skill)
        
        print("💡 当前初始阶段，Agent 仅拥有这几个工具:", agent.skill_manager.get_active_skills())
        
        # 询问一个需要它自己装载新技能的数学或者哈希问题
        self.run_agent_interaction(
            agent, 
            "我想计算 'Autonomous Mode' 这段文本的 SHA-256 值。请你自主寻找到具有哈希能力的技能并装载它，然后再计算。"
        )


def main():
    print("=" * 80)
    print("🎯 EasyAgent Skill 模块真实场景集成测试程序")
    print("  (仅以交互可视化的形式输出，包含多项特性全面测试)")
    print("=" * 80)
    
    tests = [
        TestBuiltinSkills(),
        TestYAMLSkill(),
        TestMarkdownSkill(),
        TestFolderSkill(),
        TestMetaToolsSkill()
    ]
    
    for test in tests:
        test.run()
        
    print("✅ 所有真实场景测试展示流程结束！\n")


if __name__ == "__main__":
    from agent.BasicAgent import BasicAgent
    from core.llm import EasyLLM
    from skill.registry import SkillRegistry
    from skill.builtin.calculator_skill import CalculatorSkill
    from skill.yaml_loader import YAMLSkillLoader, MarkdownSkillLoader
    from skill.folder_loader import FolderSkillLoader
    from skill.meta_tools import MetaSkill

    llm= EasyLLM()
    agent=BasicAgent(name="test_skill", llm=llm)
    agent.with_skill(CalculatorSkill())
    print(agent.get_enhanced_prompt())