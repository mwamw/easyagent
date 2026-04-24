"""
History 专用压缩器接口与实现。

history 压缩直接返回“可继续喂给模型”的历史消息列表：
- 保持最终消息时间顺序
- 优先保留最近几轮原文
- 将较老历史压缩为一条或多条摘要消息
- 工具调用链按轮次整体保留
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import logging

from core.history import coerce_canonical_message, is_canonical_message
from context.token.counter import TokenCounter
from core.llm import EasyLLM

logger = logging.getLogger(__name__)


class BaseHistoryCompactor(ABC):
    """History 压缩器抽象接口。"""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self._counter = token_counter or TokenCounter()
        self._last_run_info: dict[str, Any] = {}

    def set_token_counter(self, token_counter: TokenCounter) -> None:
        self._counter = token_counter

    def get_last_run_info(self) -> dict[str, Any]:
        return dict(self._last_run_info)

    def _set_last_run_info(self, info: Optional[dict[str, Any]]) -> None:
        self._last_run_info = dict(info or {})

    @abstractmethod
    def compact(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """压缩 history，返回新的 history 消息列表。"""
        ...

    async def acompact(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.compact, history)

    def _message_token_count(self, message: Any) -> int:
        import json
        canonical = coerce_canonical_message(message)
        if canonical is not None:
            return 4 + self._counter.count(json.dumps(canonical.to_dict(), ensure_ascii=False))
        if isinstance(message, dict):
            return 4 + self._counter.count(json.dumps(message, ensure_ascii=False))
        return 4 + self._counter.count(str(message))

    def _messages_token_count(self, messages: List[Any]) -> int:
        if not messages:
            return 0
        total = 2
        for message in messages:
            total += self._message_token_count(message)
        return total


class RuleBasedHistoryCompactor(BaseHistoryCompactor):
    """默认的规则式 history 压缩器。"""

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        recent_turns: int = 4,
        min_recent_turns: int = 1,
    ):
        super().__init__(token_counter=token_counter)
        self.recent_turns = max(1, recent_turns)
        self.min_recent_turns = max(1, min_recent_turns)

    def compact(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        normalized = [self._clone_message(message) for message in history or []]

        logger.info("Compact History by truncating tool messages")

        result = []
        for msg_dict in normalized:
            msg = coerce_canonical_message(msg_dict)
            if not msg:
                result.append(msg_dict)
                continue
                
            for block in msg.content:
                if block.type == "function_response":
                    text = self._extract_text(block.output)
                    block.output = self._compact_text(text, limit=500)
                    
            result.append(msg.to_dict())

        self._set_last_run_info(
            {
                "compactor": self.__class__.__name__,
                "status": "success",
                "fallback_used": False,
                "recent_turns": self.recent_turns,
                "input_messages": len(normalized),
                "output_messages": len(result),
            }
        )
        return result

    async def acompact(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        return self.compact(history)

    def _group_turns(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        turns: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []

        for message in history:
            role = message.get("role")
            if role == "user" and current:
                turns.append(self._make_turn(current))
                current = [message]
            else:
                current.append(message)

        if current:
            turns.append(self._make_turn(current))
        return turns

    def _make_turn(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "messages": [self._clone_message(message) for message in messages],
            "token_count": self._messages_token_count(messages),
        }


    def _summarize_turns(
        self,
        turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not turns:
            return []

        lines = ["历史摘要："]
        for index, turn in enumerate(turns, start=1):
            parts = []
            for message in turn["messages"]:
                summary = self._summarize_message(message)
                if summary:
                    parts.append(summary)
            if parts:
                lines.append(f"- 第{index}轮: {' | '.join(parts)}")
        return self._summary_lines_to_messages(lines)

    def _summarize_message(self, message: Dict[str, Any]) -> str:
        if is_canonical_message(message):
            role = message.get("role", "assistant")
            parts = []
            for block in message.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = self._compact_text(self._extract_text(block), limit=100)
                    if text:
                        parts.append(text)
                elif block_type == "reasoning":
                    text = self._compact_text(self._extract_text(block), limit=100)
                    if text:
                        parts.append(f"thinking:{text}")
                elif block_type == "function_call":
                    name = block.get("name", "unknown_tool")
                    arguments = self._compact_text(self._stringify(block.get("arguments", "")), limit=100)
                    parts.append(f"调用工具 {name}({arguments})")
                elif block_type == "function_response":
                    name = block.get("name") or block.get("call_id") or "tool"
                    output = self._compact_text(self._extract_text(block.get("output")), limit=100)
                    parts.append(f"工具结果 {name}: {output}")
            if parts:
                return f"{role}: {' | '.join(parts)}"
            return f"{role}: {self._compact_text(self._extract_text(message.get('content')), limit=100)}"

        msg_type = message.get("type")
        role = message.get("role")

        if msg_type == "function_call":
            name = message.get("name", "unknown_tool")
            arguments = self._compact_text(self._stringify(message.get("arguments", "")), limit=100)
            return f"调用工具 {name}({arguments})"

        if msg_type == "function_call_output":
            output = self._compact_text(self._stringify(message.get("output", "")), limit=100)
            call_id = message.get("call_id", "")
            return f"工具结果[{call_id}]: {output}"

        if role == "assistant" and message.get("tool_calls"):
            text = self._compact_text(self._extract_text(message.get("content")), limit=100)
            tool_names = []
            for tool_call in message.get("tool_calls", []):
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                tool_names.append(function.get("name") or tool_call.get("name", "unknown_tool"))
            prefix = f"助手: {text}" if text else "助手调用工具"
            return f"{prefix} -> {', '.join(tool_names)}"

        if role in {"tool", "function"}:
            tool_name = message.get("name") or message.get("tool_call_id") or "tool"
            content = self._compact_text(self._extract_text(message.get("content")), limit=100)
            return f"工具 {tool_name}: {content}"

        if msg_type == "message":
            phase = message.get("phase")
            content = self._compact_text(self._extract_text(message.get("content")), limit=100)
            message_role = message.get("role", "assistant")
            if phase:
                return f"{message_role}[{phase}]: {content}"
            return f"{message_role}: {content}"

        if msg_type == "reasoning":
            summary = self._compact_text(self._extract_text(message.get("summary")), limit=100)
            if summary:
                return f"assistant[thinking]: {summary}"
            content = self._compact_text(self._extract_text(message.get("content")), limit=100)
            if content:
                return f"assistant[thinking]: {content}"
            return ""

        content = self._compact_text(self._extract_text(message.get("content")), limit=100)
        if role:
            return f"{role}: {content}"
        return self._stringify(message)

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            item_type = content.get("type")
            if item_type in {"text", "output_text", "summary_text"}:
                return str(content.get("text", "")).strip()
            if item_type == "reasoning":
                if content.get("text"):
                    return str(content.get("text", "")).strip()
                if content.get("summary") is not None:
                    return self._extract_text(content.get("summary"))
            if item_type == "function_call":
                name = content.get("name", "tool")
                return f"[function_call:{name}]"
            if item_type == "function_response":
                name = content.get("name") or content.get("call_id") or "tool"
                return f"[function_response:{name}]"
            if item_type == "tool_use":
                name = content.get("name", "tool")
                return f"[tool_use:{name}]"
            if item_type == "tool_result":
                return f"[tool_result:{self._extract_text(content.get('content'))}]"
            return self._stringify(content)
        if isinstance(content, list):
            fragments: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"text", "output_text", "summary_text"}:
                        fragments.append(str(item.get("text", "")).strip())
                    elif item.get("type") == "reasoning":
                        if item.get("text"):
                            fragments.append(str(item.get("text", "")).strip())
                        elif item.get("summary"):
                            fragments.append(self._extract_text(item.get("summary")))
                    elif item.get("type") == "tool_use":
                        name = item.get("name", "tool")
                        fragments.append(f"[tool_use:{name}]")
                    elif item.get("type") == "tool_result":
                        fragments.append(f"[tool_result:{self._extract_text(item.get('content'))}]")
                    elif item.get("type") == "function_call":
                        name = item.get("name", "tool")
                        fragments.append(f"[function_call:{name}]")
                    elif item.get("type") == "function_response":
                        name = item.get("name") or item.get("call_id") or "tool"
                        fragments.append(f"[function_response:{name}]")
                    else:
                        fragments.append(self._stringify(item))
                else:
                    fragments.append(str(item).strip())
            return " ".join(fragment for fragment in fragments if fragment).strip()
        return self._stringify(content)

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False, default=str)

    def _compact_text(self, text: str, limit: int = 500) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit].rstrip()}..."

    def _summary_lines_to_messages(
        self,
        lines: List[str],
        max_message_tokens: int = 180,
    ) -> List[Dict[str, Any]]:
        if not lines:
            return []

        messages: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        for line in lines:
            candidate_lines = [*current_lines, line]
            candidate_text = "\n".join(candidate_lines)
            if current_lines and self._counter.count(candidate_text) > max_message_tokens:
                messages.append(self._build_summary_message("\n".join(current_lines)))
                current_lines = [line]
                continue
            current_lines = candidate_lines

        if current_lines:
            messages.append(self._build_summary_message("\n".join(current_lines)))
        return messages

    @staticmethod
    def _build_summary_message(text: str) -> Dict[str, Any]:
        return {
            "record_type": "canonical_message",
            "role": "assistant",
            "provider": "history_compactor",
            "provider_message_type": "summary",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
            "metadata": {"compacted_summary": True},
        }


    def _clone_message(self, message: Any) -> Dict[str, Any]:
        canonical = coerce_canonical_message(message)
        if canonical is not None:
            return canonical.to_dict()
        raise ValueError("RuleBasedHistoryCompactor 只接受 canonical history。")


class LLMHistoryCompactor(BaseHistoryCompactor):
    """基于 LLM 的 history 压缩器，直接返回压缩后的几条历史消息。"""

    def __init__(
        self,
        llm: EasyLLM,
        token_counter: Optional[TokenCounter] = None,
        language: str = "zh",
        recent_turns: int = 4,
    ):
        super().__init__(token_counter=token_counter)
        self.llm = llm
        self.language = language
        self.recent_turns = max(1, recent_turns)
        self._formatter = RuleBasedHistoryCompactor(token_counter=token_counter)

    def set_token_counter(self, token_counter: TokenCounter) -> None:
        super().set_token_counter(token_counter)
        self._formatter.set_token_counter(token_counter)

    def compact(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        normalized = [self._clone_message(message) for message in history or []]
        if not normalized:
            self._set_last_run_info(
                {
                    "compactor": self.__class__.__name__,
                    "status": "skipped",
                    "fallback_used": False,
                    "mode": "sync",
                }
            )
            return []

        logger.info("Compact History by LLM")

        prompt = self._build_prompt(normalized)

        if self.language == "zh":
            system_prompt = """你是一个历史压缩器。你只输出 JSON 数组，不要输出解释、Markdown 或代码块。\n
                            要求：\n
                            1. 输出必须是 JSON 数组。\n
                            2. 数组元素必须是字符串，每个字符串是一段按时间顺序组织的摘要。\n
                            3. 保留重要的用户约束、已确认事实、关键工具结果和未完成任务。\n
                            4. 保持时间顺序，从旧到新。\n
                            5. 不要输出解释。\n\n"""
                        
        else:
            system_prompt = """you are a history compressor. You only output JSON arrays, do not output explanations, Markdown or code blocks.\n
                            Requirements:\n
                            1. Output must be a JSON array.\n
                            2. Each item must be a string summary.\n
                            3. Preserve key constraints, facts, important tool results, and unfinished tasks.\n
                            4. Keep chronological order from old to new.\n
                            5. Do not add any explanation.\n\n"""
        try:
            response = self.llm.invoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            messages = self._parse_response(response)
            self._set_last_run_info(
                {
                    "compactor": self.__class__.__name__,
                    "status": "success",
                    "fallback_used": False,
                    "mode": "sync",
                    "input_messages": len(normalized),
                    "output_messages": len(messages),
                }
            )
            return messages
        except Exception as exc:
            logger.warning("LLMHistoryCompactor 压缩失败: %s", exc)
            raise exc

    async def acompact(
        self,
        history: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        normalized = [self._clone_message(message) for message in history or []]
        if not normalized :
            self._set_last_run_info(
                {
                    "compactor": self.__class__.__name__,
                    "status": "skipped",
                    "fallback_used": False,
                    "mode": "async",
                }
            )
            return []
            
        logger.info("Compact History")

        prompt = self._build_prompt(normalized)
        if self.language == "zh":
            system_prompt = """你是一个历史压缩器。你只输出 JSON 数组，不要输出解释、Markdown 或代码块。\n
                            要求：\n
                            1. 输出必须是 JSON 数组。\n
                            2. 数组元素必须是字符串，每个字符串是一段按时间顺序组织的摘要。\n
                            3. 保留重要的用户约束、已确认事实、关键工具结果和未完成任务。\n
                            4. 保持时间顺序，从旧到新。\n
                            5. 不要输出解释。\n\n"""
                        
        else:
            system_prompt = """you are a history compressor. You only output JSON arrays, do not output explanations, Markdown or code blocks.\n
                            Requirements:\n
                            1. Output must be a JSON array.\n
                            2. Each item must be a string summary.\n
                            3. Preserve key constraints, facts, important tool results, and unfinished tasks.\n
                            4. Keep chronological order from old to new.\n
                            5. Do not add any explanation.\n\n"""
        try:
            response = await self.llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            messages = self._parse_response(response)
            if messages:
                self._set_last_run_info(
                    {
                        "compactor": self.__class__.__name__,
                        "status": "success",
                        "fallback_used": False,
                        "mode": "async",
                        "input_messages": len(normalized),
                        "output_messages": len(messages),
                    }
                )
                return messages
            else:
                raise Exception("LLMHistoryCompactor 异步压缩失败: %s")
        except Exception as exc:
            logger.warning("LLMHistoryCompactor 异步压缩失败: %s", exc)
            raise exc



    def _build_prompt(self, history: List[Dict[str, Any]]) -> str:
        transcript_lines = []
        for index, message in enumerate(history, start=1):
            summary = self._formatter._summarize_message(message)
            if summary:
                transcript_lines.append(f"{index}. {summary}")
        transcript = "\n".join(transcript_lines)
        if self.language == "zh":
            return (
                f"请把下面 canonical history 摘要压缩成数条摘要，"
                f"history:\n{transcript}"
            )
        return (
            f"Compress the canonical history summary into summaries, "

            f"history:\n{transcript}"
        )

    def _parse_response(self, response: Any) -> List[Dict[str, Any]]:
        text = str(response or "").strip()
        if not text:
            return []

        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end >= start:
            text = text[start:end + 1]

        payload = json.loads(text)
        if not isinstance(payload, list):
            return []

        messages: List[Dict[str, Any]] = []
        for item in payload:
            content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
            content = str(content or "").strip()
            if not content:
                continue
            messages.append(self._formatter._build_summary_message(content))
        return messages

    def _clone_message(self, message: Any) -> Dict[str, Any]:
        canonical = coerce_canonical_message(message)
        if canonical is not None:
            return canonical.to_dict()
        raise ValueError("LLMHistoryCompactor 只接受 canonical history。")





HistoryCompactor = LLMHistoryCompactor
