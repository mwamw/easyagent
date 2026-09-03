from __future__ import annotations

from pathlib import Path

from common import (
    CaseResult,
    append_jsonl,
    build_mock_registry,
    build_permission_context,
    create_llm,
    default_arg_parser,
    ensure_results_dir,
    result_to_dict,
    summarize_results,
    write_json,
    now_ms,
)
from Tool.runtime import SubagentRequest
from easyagent import BasicAgent, Config
from easyagent.runtime import AgentRuntimeManager, TeamManager


def make_runtime() -> AgentRuntimeManager:
    team_manager = TeamManager()

    def factory(request: SubagentRequest) -> BasicAgent:
        return BasicAgent(
            name=request.name or request.description or "worker",
            llm=create_llm(),
            system_prompt=(
                "You are a real EasyAgent subagent in a multi-agent runtime benchmark. "
                "Use benchmark tools when the task requests them and report concise results."
            ),
            config=Config(tool_schema_mode="deferred"),
        ).with_tool(build_mock_registry()).with_permissions(
            context=build_permission_context()
        )

    runtime = AgentRuntimeManager(agent_factory=factory, team_manager=team_manager)
    return runtime


def run_case(background: bool) -> CaseResult:
    start = now_ms()
    runtime = make_runtime()
    team = runtime.team_manager.create_team(name="review_team", description="Synthetic benchmark team")
    request = SubagentRequest(
        description="Analyze one module",
        prompt=(
            "Use `MockFileRead` with JSON arguments {'path': 'module_a.py'} to inspect module A. "
            "Then report one concise finding."
        ),
        name="worker_a",
        team_name=team.name,
        mode="default",
    )
    handle = runtime.run(request, run_in_background=background)
    if background:
        handle = runtime.wait(handle.agent_id, timeout_ms=180000)
    deliveries = runtime.send_message(
        recipient_type="team",
        recipient_id=team.team_id,
        sender_id="manager",
        content="Use concise reporting format.",
    )
    mailbox = runtime.read_mailbox(handle.agent_id)
    acked = runtime.ack_mailbox(handle.agent_id, ack_all=True, actor_id=handle.agent_id)
    listed = runtime.list_handles(team_id=team.team_id)
    success = (
        handle.status == "completed"
        and len(deliveries) == 1
        and len(mailbox) == 1
        and len(acked) == 1
        and len(listed) == 1
    )
    close_report = runtime.close()
    return CaseResult(
        case_id=f"multi_agent_{'background' if background else 'foreground'}",
        category="multi_agent_runtime",
        expected="completed",
        observed=handle.status,
        success=success,
        duration_ms=now_ms() - start,
        metrics={
            "background": background,
            "agent_id": handle.agent_id,
            "team_id": team.team_id,
            "mailbox_count": len(mailbox),
            "acked_count": len(acked),
            "team_handle_count": len(listed),
            "total_tool_use_count": handle.total_tool_use_count,
            "usage": handle.usage,
            "content": handle.content,
            "error": handle.error,
            "close_report": close_report,
        },
    )


def main() -> None:
    parser = default_arg_parser("E8 multi-agent runtime case study")
    args = parser.parse_args()
    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "e8_multi_agent_case.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    for background in [False, True]:
        for _ in range(max(args.repeat, 1)):
            result = run_case(background)
            results.append(result)
            append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary["experiment"] = "e8_multi_agent_case"
    out_path = Path(args.out) if args.out else out_dir / "e8_multi_agent_case_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
