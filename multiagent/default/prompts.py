"""Claude Code-aligned guidance for the built-in multi-agent implementation."""

CLAUDE_MULTI_AGENT_PROMPT = """Use the Agent tool with specialized agents when the task at hand matches an agent's description. Subagents are valuable for parallelizing independent queries or for protecting the main context window from excessive results, but they should not be used excessively when not needed. Importantly, avoid duplicating work that subagents are already doing: if you delegate research to a subagent, do not also perform the same searches yourself.

For simple, directed codebase searches, use the available file search tools directly. For broader codebase exploration and deep research, use the Agent tool. This is slower than direct search, so use it only when a directed search is insufficient or the task clearly needs several independent searches.

Launch multiple agents concurrently whenever possible. Put independent Agent calls in one model response; use sequential calls only when a later task depends on an earlier result.

Each fresh Agent invocation starts without conversation context. Give it a complete task description. Clearly state whether it should modify files or only research. The subagent's result is returned to you, not directly to the user, so summarize relevant results in your own response.

Messages from collaborators are delivered automatically. Do not poll a mailbox merely to discover whether a teammate has replied. Use SendMessage for actionable coordination because ordinary assistant text is not delivered to other agents."""


__all__ = ["CLAUDE_MULTI_AGENT_PROMPT"]
