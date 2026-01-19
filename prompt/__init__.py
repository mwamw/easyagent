# Prompt module for EasyAgent
from .template import PromptTemplate, ChatPromptTemplate
from .defaults import (
    REACT_PROMPT,
    PLANNING_PROMPT,
    RAG_PROMPT,
    STRUCTURED_OUTPUT_PROMPT,
)

__all__ = [
    "PromptTemplate",
    "ChatPromptTemplate",
    "REACT_PROMPT",
    "PLANNING_PROMPT",
    "RAG_PROMPT",
    "STRUCTURED_OUTPUT_PROMPT",
]
