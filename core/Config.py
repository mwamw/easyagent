from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import os


def _split_path_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item for item in (part.strip() for part in value.split(os.pathsep)) if item]


class Config(BaseModel):
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    debug: bool = False
    log_level: str = "INFO"

    max_history_length: int = 200
    trigger_threshold: int = 20

    workspace_root: Optional[str] = None
    allowed_roots: list[str] = Field(default_factory=list)
    shell: str = "bash"
    command_timeout_ms: int = 600000
    max_background_tasks: int = 8
    git_binary: str = "git"
    enable_worktree: bool = False
    interrupt_on_confirmation: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        max_tokens = os.getenv("MAX_TOKENS")
        if max_tokens:
            max_tokens = int(max_tokens)
        else:
            max_tokens = None

        workspace_root = os.getenv("WORKSPACE_ROOT")
        allowed_roots = _split_path_list(os.getenv("ALLOWED_ROOTS"))
        shell = os.getenv("SHELL") or "bash"

        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=max_tokens,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            shell=shell,
            command_timeout_ms=int(os.getenv("COMMAND_TIMEOUT_MS", "600000")),
            max_background_tasks=int(os.getenv("MAX_BACKGROUND_TASKS", "8")),
            git_binary=os.getenv("GIT_BINARY", "git"),
            enable_worktree=os.getenv("ENABLE_WORKTREE", "false").lower() == "true",
            interrupt_on_confirmation=os.getenv("INTERRUPT_ON_CONFIRMATION", "true").lower() == "true",
        )

    def get_allowed_roots(self) -> list[str]:
        roots = list(self.allowed_roots)
        if self.workspace_root and self.workspace_root not in roots:
            roots.insert(0, self.workspace_root)
        return roots

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()
