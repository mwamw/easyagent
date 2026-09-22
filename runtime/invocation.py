"""State belonging to one invocation, independent of Agent lifetime."""

from dataclasses import dataclass
from enum import Enum

class AgentInvocationPhase(str, Enum):
    CREATED = "created"
    STARTED = "started"
    COMPILING = "compiling"
    LLM = "llm"
    TOOL = "tool"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class AgentInvocationState:
    invocation_id: str
    query: str
    stream: bool
    phase: AgentInvocationPhase = AgentInvocationPhase.CREATED
    round_number: int = 0
    output: str = ""
    error: BaseException | None = None


