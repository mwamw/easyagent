"""Inspect Skill progressive disclosure without making an LLM request."""

from __future__ import annotations

import json
from pathlib import Path

from easyagent import BasicAgent, EasyLLM
from runtime import RuntimeEventType


SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def main() -> None:
    llm = EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )
    agent = BasicAgent(name="skill-message-dump", llm=llm).with_skill(SKILLS_DIR)
    agent.event_bus.publish(
        RuntimeEventType.AGENT_INVOKE_STARTED,
        agent_id=agent.name,
        invocation_id="manual-inspection",
        data={"query": "Review skill/manager.py"},
    )
    result = agent.execute_tool_result(
        "skill_tool",
        {"skill": "repository-review", "args": "skill/manager.py"},
    )
    agent.history_store.append_tool_result(
        result.to_display_string(),
        "manual-skill-call",
        "skill_tool",
    )
    agent.metamessage_manager.flush()
    print(json.dumps(agent.get_history(), ensure_ascii=False, indent=2))

    agent.event_bus.publish(
        RuntimeEventType.AGENT_INVOKE_COMPLETED,
        agent_id=agent.name,
        invocation_id="manual-inspection",
    )
    print(json.dumps(agent.get_history(), ensure_ascii=False, indent=2))
    agent.close()


if __name__ == "__main__":
    main()
