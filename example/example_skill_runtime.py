from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM
from skill.meta_tools import MetaSkill
from skill.registry import SkillRegistry


def main() -> None:
    # 1. 准备全局 SkillRegistry，并从目录发现 on-demand skills
    registry = SkillRegistry.instance()
    registry.discover_from_directory("./skills")

    # 2. 创建 Agent。这里不预加载业务 Skill，只加 MetaSkill
    llm = EasyLLM()
    agent = BasicAgent(name="skill_runtime_demo", llm=llm, enable_tool=True)
    agent.with_skill(MetaSkill(registry, agent.skill_manager))

    # 3. 此时主 system prompt 已包含：
    #    - skill policy
    #    - skill listing
    #    具体 skill 正文不会常驻。模型需要时会调用 skill_tool，
    #    skill 正文会进入当前回合的 runtime skill context。
    query = (
        "如果当前工具不足，请先查看可用技能并选择合适的 skill。"
        "然后按需调用该 skill 来完成任务。"
    )
    result = agent.invoke(query)
    print(result)


if __name__ == "__main__":
    main()
