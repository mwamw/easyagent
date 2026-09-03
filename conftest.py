from __future__ import annotations

from pathlib import Path

import pytest


INTEGRATION_FILES = {
    "test_Extractor.py",
    "test_MemoryManage.py",
    "test_Neo4jStore.py",
    "test_PerceptualMemory.py",
    "test_Qdramt.py",
    "test_agent_memory_integration.py",
    "test_async_tools.py",
    "test_episodememory.py",
    "test_mcp_module_integration.py",
    "test_memory_tool.py",
    "test_semanticmemory.py",
}

EXTERNAL_FILES = {
    "test_basicagent.py",
    "test_google.py",
    "test_mcp_real_integration.py",
    "test_real_agent_skill.py",
    "test_tool.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked as integration",
    )
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests marked as external",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_integration = config.getoption("--run-integration")
    run_external = config.getoption("--run-external")

    skip_integration = pytest.mark.skip(
        reason="integration tests are disabled by default; pass --run-integration to enable them",
    )
    skip_external = pytest.mark.skip(
        reason="external tests are disabled by default; pass --run-external to enable them",
    )

    for item in items:
        filename = Path(str(item.fspath)).name

        if filename in EXTERNAL_FILES:
            item.add_marker(pytest.mark.external)
            item.add_marker(pytest.mark.integration)
            if not run_external:
                item.add_marker(skip_external)
            continue

        if filename in INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
            if not run_integration:
                item.add_marker(skip_integration)
            continue

        item.add_marker(pytest.mark.unit)


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool:
    filename = Path(str(collection_path)).name
    if filename in EXTERNAL_FILES:
        return not config.getoption("--run-external")
    if filename in INTEGRATION_FILES:
        return not config.getoption("--run-integration")
    return False
