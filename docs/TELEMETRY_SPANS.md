# Telemetry — Span Inventory

Per the [SAP Cloud SDK telemetry guide](https://github.com/SAP/cloud-sdk-python/blob/main/src/sap_cloud_sdk/core/telemetry/user-guide.md),
the model is a **two-layer** approach:

1. **Auto-instrumentation** (`auto_instrument()` → Traceloop) captures LLM `chat`
   spans and LangChain/LangGraph tool calls automatically.
2. **Custom span** (`invoke_agent_span()`) wraps the run to add business identity
   (agent name, conversation id, user, tokens).

Intended hierarchy for one agent turn:

```
invoke_agent  (aif-analysis-agent)          ← we create via invoke_agent_span()
  ├─ chat            (sap/gpt-4.1)            ← auto-instrumented LLM calls
  ├─ execute_tool    (calculate_date_range_tool)
  ├─ chat
  ├─ execute_tool    (list_error_messages_tool)
  └─ chat …                                   ← tool-loop + finalizer LLM calls
```

## Spans this agent SHOULD produce (one query)

| Span | span_type | Source | Keep? | Description |
|---|---|---|---|---|
| **invoke_agent** `aif-analysis-agent` | invoke_agent | `invoke_agent_span()` in `stream()` | ✅ **required** | The agent turn. Carries `gen_ai.agent.name`, `gen_ai.conversation.id` (= A2A context_id), token totals, `user.id`. The one business-level span. |
| **chat** `sap/gpt-4.1` (×N) | chat | Traceloop auto | ✅ required | Each LLM call in the tool-loop + the structured-output finalizer call. N≈3–4 per turn. |
| **execute_tool** `<tool>` | execute_tool | Traceloop auto | ✅ required | One per `@tool` the LLM invoked (e.g. `calculate_date_range_tool`, `list_error_messages_tool`, `get_message_log_tool`…). |

## Spans that are NOISE (dropped at ingestion)

| Span | Why it appears | Action |
|---|---|---|
| **invoke_agent** `LangGraph` (`traceloop.span.kind = workflow`) | Traceloop emits its **own** framework workflow wrapper for the LangGraph run, named after the framework, not the agent. It duplicates the real `invoke_agent` span. | **Dropped** by the dashboard receiver (`ingest.ts`): any `invoke_agent` with agent_name `LangGraph`, `traceloop.span.kind = workflow`, or zero duration. |
| **invoke_agent** `LangGraph` (second copy) | Traceloop nests two graph wrappers (graph + workflow). | Same — dropped. |
| Manual `tracer.start_as_current_span("<tool>_tool")` | The agent USED to create explicit tool spans, which **duplicated** the Traceloop `execute_tool` spans (one was the parent of the other). | **Removed at source**: `tracer` is now a no-op shim (`_NoopTracer`) in `app/agent.py`. We rely solely on SAP auto-instrumentation for tool spans (the standard pattern). |

### Result
Before: **9 spans/query** (3 invoke_agent incl. 2 LangGraph dups, 2 execute_tool incl. duplicate, 4 chat).
After: **7 spans/query** — 1 invoke_agent (real) + 2 execute_tool + 4 chat. No duplicates, no framework wrappers.

## Why we don't create manual chat/tool spans

The SAP guide states manual `chat_span()` / `execute_tool_span()` are for use
**only when auto-instrumentation is not available** — they are mutually exclusive
with auto-instrumentation per call context. Since `auto_instrument()` (Traceloop)
already traces LangChain/LangGraph LLM and tool calls, creating our own would
double-count. So:

- ✅ We create exactly ONE custom span: `invoke_agent_span()` (business identity).
- ✅ chat + execute_tool come from auto-instrumentation.
- ❌ We do NOT manually wrap tools/LLM calls (no-op tracer enforces this).

## Attributes recorded on the invoke_agent span

Set in `app/agent.py stream()` via `invoke_agent_span(...)` + `span.set_attribute`:

| Attribute | Meaning |
|---|---|
| `gen_ai.agent.name` | `aif-analysis-agent` |
| `gen_ai.conversation.id` | the A2A `context_id` (→ dashboard `context_id` column) |
| `gen_ai.request.model` | `sap/gpt-4.1` |
| `gen_ai.usage.input_tokens` / `output_tokens` / `total_tokens` | from `usage_metadata` of the final LLM message |
| `user.id` | when XSUAA auth is on |
| `a2a.context_id` | duplicate of conversation id for convenience |
| `aif.query` / `gen_ai.prompt` | the user's query text for THIS turn — makes each turn (incl. follow-ups like "How do I fix that error?") distinguishable |
| `aif.memory.persistent` | true when the SAP Agent Memory client is bound |
| `aif.memory.history_messages` | number of prior messages loaded into the prompt this turn |
| `aif.memory.short_term_turns` | in-process short-term turns stored for this context_id |
| `aif.memory.has_prior_findings` | true when the previous turn's GUID-addressable findings were injected |

## Dashboard de-noise rule (single source of truth)

`server/ingest.ts` → `toRow()` drops, for `span_type === 'invoke_agent'`:
- `durMs === 0` (empty wrapper)
- `traceloop.span.kind === 'workflow'` (framework wrapper)
- `agent_name === 'LangGraph'` (framework graph span)

Everything else maps cleanly to invoke_agent / execute_tool / chat. Non-GenAI
infra spans (httpx, etc.) are skipped (no `gen_ai.operation.name`, no `_tool` name).
