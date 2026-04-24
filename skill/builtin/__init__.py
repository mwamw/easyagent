# 内置 Skill 包
from .agent_teams_skill import AgentTeamsSkill
from .code_review_skill import CodeReviewSkill
from .frontend_design_skill import FrontendDesignSkill
from .memory_skill import MemorySkill
from .debug_skill import DebugSkill
from .file_manager_skill import FileManagerSkill
from .linux_ops_skill import LinuxOpsSkill
from .product_strategy_skill import ProductStrategySkill
from .research_synthesis_skill import ResearchSynthesisSkill
from .teaching_skill import TeachingSkill
from .web_search_skill import WebSearchSkill
from .task_planning_skill import TaskPlanningSkill
from .web_research_skill import WebResearchSkill
from .calculator_skill import CalculatorSkill
from .mcp_skill import MCPSkill, MCPPromptSkill
from .writing_skill import WritingSkill


def register_builtin_skills(registry) -> list[str]:
    """将默认内置 Skill 类注册到 SkillRegistry。"""
    builtin_skill_classes = [
        ("agent_teams", AgentTeamsSkill),
        ("calculator", CalculatorSkill),
        ("code_review", CodeReviewSkill),
        ("debug", DebugSkill),
        ("file_manager", FileManagerSkill),
        ("frontend_design", FrontendDesignSkill),
        ("linux_ops", LinuxOpsSkill),
        ("memory", MemorySkill),
        ("mcp", MCPSkill),
        ("product_strategy", ProductStrategySkill),
        ("research_synthesis", ResearchSynthesisSkill),
        ("task_planning", TaskPlanningSkill),
        ("teaching", TeachingSkill),
        ("web_research", WebResearchSkill),
        ("web_search", WebSearchSkill),
        ("writing", WritingSkill),
    ]
    registered: list[str] = []
    for name, skill_class in builtin_skill_classes:
        if registry.has(name):
            continue
        registry.register_class(skill_class, name)
        registered.append(name)
    return registered

__all__ = [
    "AgentTeamsSkill",
    "CodeReviewSkill",
    "DebugSkill",
    "FileManagerSkill",
    "FrontendDesignSkill",
    "LinuxOpsSkill",
    "MemorySkill",
    "ProductStrategySkill",
    "ResearchSynthesisSkill",
    "TeachingSkill",
    "WebSearchSkill",
    "TaskPlanningSkill",
    "WebResearchSkill",
    "WritingSkill",
    "CalculatorSkill",
    "MCPSkill",
    "MCPPromptSkill",
    "register_builtin_skills",
]
