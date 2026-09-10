"""Smoke-check the actual Markdown snippets with fake responses and in-memory data.

No API requests, external databases, or model-performance experiments are run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from easyagent import EasyLLM
from easyagent.history import canonical_text_content
from easyagent.session import InMemorySessionStore, SessionConflictError
from easyagent.tools import Tool, ToolResult
from pydantic import BaseModel


def main():
    source = (Path(__file__).parent / "easyagent-whitepaper.zh-CN.md").read_text()
    examples = re.findall(r"```python\n(.*?)\n```", source, flags=re.S)
    assert len(examples) == 7
    module = ModuleType("easyagent_whitepaper_examples")
    sys.modules[module.__name__] = module
    for index, snippet in enumerate(examples, 1):
        exec(compile(snippet, f"whitepaper-example-{index}", "exec"), module.__dict__)

    class EvidenceRepository:
        def lookup(self, evidence_id):
            return {"text": "Measured result", "source": evidence_id} if evidence_id == "E-1" else None

    class SessionRepository:
        def __init__(self):
            self.backend = InMemorySessionStore()

        def write_session(self, **record):
            return self.backend.create_or_update_session(**record)

        def read_session(self, session_id, touch=False):
            return self.backend.get_session(session_id, touch=touch)

    class EmptyParameters(BaseModel):
        pass

    class MemoryNoteTool(Tool):
        def __init__(self):
            super().__init__("memory_note", "Read an example memory.", EmptyParameters, read_only=True)

        def run(self, parameters):
            return ToolResult.success("Remember evidence E-1")

    # Supplying a dummy SDK client prevents construction of network clients.
    llm = EasyLLM(provider="openai", model="whitepaper-test-only", api_key="not-a-real-key",
                  base_url="http://127.0.0.1:9/v1", client=SimpleNamespace(close=lambda: None))
    repository = EvidenceRepository()
    store = module.DomainSessionStore(SessionRepository())
    summarize = lambda history: "Evidence E-1 remains relevant."
    build = lambda: module.build_research_agent(llm, repository, [MemoryNoteTool()], summarize, store)
    agent = build()
    resumed = None
    try:
        assert agent.tool_registry.has_tool("lookup_evidence")
        assert agent.tool_registry.has_tool("memory_note")
        blocks = agent.get_system_prompt_blocks("Inspect evidence")
        assert {"identity", "workspace", "memory.evidence"}.issubset({block.name for block in blocks})
        result = agent.execute_tool_result("lookup_evidence", {"evidence_id": "E-1"})
        assert result.status == "success" and result.structured_data["source"] == "E-1"
        assert agent.execute_tool_result("lookup_evidence", {"evidence_id": "missing"}).status == "error"
        assert module.LookupEvidence(repository).validate_parameters({"evidence_id": "E-1"}) == {"evidence_id": "E-1"}
        try:
            module.LookupEvidence(repository).validate_parameters({"evidence_id": ""})
        except ValueError:
            pass
        else:
            raise AssertionError("Tool parameter validation did not reject an empty ID")
        compressed = module.EvidenceCompactor(summarize).compact([])
        assert "E-1" in canonical_text_content(compressed[0])

        agent.add_user_message("Persist this evidence context")
        session_id = agent.save_session()
        record = store.get_session(session_id)
        assert record["snapshot"]["schemaVersion"] == 4
        assert store.get_session("unknown") is None
        try:
            store.create_or_update_session(session_id, "BasicAgent", "test", record["snapshot"], expected_revision=None)
        except SessionConflictError:
            pass
        else:
            raise AssertionError("Repository must enforce the Store revision contract")
        resumed = build()
        resumed.restore_session(session_id)
        assert any("Persist this evidence" in canonical_text_content(message)
                   for message in resumed.history_store.canonical)

        class Client:
            async def recall(self, query):
                self.query = query
                return "Temporary evidence context"

        client = Client()

        class InjectedMemory(module.EvidenceMemory):
            def get_hooks(self):
                return [module.RecallBeforeContext(self, client)]

        resumed.with_memory(InjectedMemory([MemoryNoteTool()]), replace=True)
        fake_response = SimpleNamespace(content="example-response", reasoning_content=None, tool_calls=[], usage=None)
        with patch.object(llm._provider, "async_invoke_raw", new=AsyncMock(return_value=fake_response)) as invoke:
            assert asyncio.run(resumed.ainvoke("Read prior evidence")) == "example-response"
            assert invoke.await_count == 1
            assert "Temporary evidence context" in repr(invoke.call_args)
        assert client.query == "Read prior evidence"
        assert not any("Temporary evidence context" in canonical_text_content(message)
                       for message in resumed.history_store.canonical)
        print("PASS 7 Markdown examples: imports, assembly, prompt/module merge, tool validation/results,")
        print("memory tool/Hook registration, async request injection, temporary-message cleanup,")
        print("compactor output, Store revision conflict, save and restore. Model response was mocked.")
    finally:
        if resumed is not None:
            resumed.close()
        agent.close()
        llm.close()
        sys.modules.pop(module.__name__, None)


if __name__ == "__main__":
    main()
