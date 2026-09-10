# Module refactor execution notes

## Agreed boundaries

- Memory provides instructions, tools, and optional hooks; model-directed retrieval is the default.
- MetaMessage stays source-agnostic. Context assembles supplied content and controls one total budget.
- ConversationHistory is the existing in-memory history component, not a new memory mechanism.
- Session persistence restores data into an explicitly assembled Agent.
- Memory and RAG share domain-neutral storage and embedding infrastructure.
- Preserve existing data and use explicit migrations. S4Code uses public APIs, without adapters.
- No UI rewrite or product file-memory implementation.

Mandatory automatic recall, a MetaMessage prepare/commit redesign, and a full
transcript journal were removed from the prerequisites in the final discussion.

## Execution checklist

- [x] Replaceable memory contract, model-directed built-in memory, lightweight imports.
- [x] Context source/quota removal, async preparation, final request validation.
- [x] Targeted temporary-message/compaction safeguards without API redesign.
- [x] Public storage and embedding; migrate Memory and RAG consumers.
- [x] Session persistence and store contracts; explicit state restoration.
- [x] S4Code migration, examples, non-destructive data migration.
- [x] Framework and product regression tests.

## Baseline

Execution began from clean worktrees. Planning checks passed 23 focused tests.
Memory collection failed through eager Hugging Face imports and user-site Pandas;
isolate optional dependencies instead of changing the user's Conda environment.

## Verification and compatibility

- EasyAgent default suite: 544 passed, 2 skipped (2026-09-06).
- S4Code Python suite: 189 passed.
- Ink/Bun: 55 passed; TypeScript typecheck and SDK build passed.
- Public Qdrant backend exercised locally with supplied vectors; Chroma skipped
  because its optional dependency is unavailable. No real Neo4j/model API calls.
- Real-model RAG suites are now explicitly integration-gated; deterministic
  storage/RAG, migration, cancellation, and state-isolation tests run by default.
- EasyAgent is 0.8.0; S4Code requires >=0.8.0. Editable installs use this workspace.
- Source databases were not migrated. The CLI defaults to dry-run and requires
  --apply to populate a new/empty target. S4Code uses sessions-v4.db.
- The existing V2 implementation filenames remain; BaseMemoryLayer distinguishes
  its internal layer interface from the new public BaseMemory mechanism contract.
- MetaMessage emit/register/flush remains; ConversationHistory still stores the
  current canonical/replay state, not an invented complete transcript archive.

## Follow-up: restore API-first context sizing (2026-09-06)

The earlier Agent implementation used a provider-usage anchor plus locally
estimated deltas. That priority was lost during the request-pipeline refactor;
it is restored as an independent, per-Agent ContextUsageTracker.

- Sync, async, and both streaming executors capture successful main-model usage,
  without requiring Observability or reading a shared LLM's last call.
- Compaction prechecks, final input validation, and get_context_usage use the same
  preferred estimate. Local-only counts remain available for diagnostics.
- History replacement/restore, model, fixed prompt, tool, and counter changes
  invalidate incompatible anchors. New messages and expiring temporary context
  use signed local deltas; hidden or rewritten output uses an input-only base.
- Anthropic streaming merges start/end usage and includes cached input in context
  occupancy. OpenAI cached input is not counted twice. Billing fields are retained.
- S4Code refreshes its context cache when history or usage changes; Ink preserves
  the estimate source. No transcript rendering or terminal-layout changes.
- Final validation in the configured S4Code Conda environment: EasyAgent 575
  passed, 2 skipped; S4Code Python 190 passed; Bun 56 passed. TypeScript typecheck,
  SDK build, and both repositories' diff checks passed. No paid model calls.
