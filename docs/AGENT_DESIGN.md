# AIF Monitoring A2A Agent — How It Works

This document explains the **migrated** `aif-analysis-agent`: an A2A-protocol agent
that monitors SAP AIF (Application Interface Framework) interfaces and returns
**structured card data** that SAP Joule renders as UI5 cards — while remaining a
normal A2A agent any client can call.

It is the A2A re-implementation of the original pro-code **Joule** AIF agent. The
orchestration + LLM reasoning that used to live inside Joule now run inside this
Python agent; Joule keeps only the card-rendering layer.

---

## 1. High-level architecture

```
                         ┌──────────────────────────────────────────────┐
 SAP Joule (BYOA) ──A2A──▶  aif-analysis-agent  (LangGraph + LiteLLM)     │
 or any A2A client        │                                              │
                          │  stream():                                   │
                          │   1. tool-loop  (model ⇄ tools)              │
                          │   2. finalizer  (JSON-mode → card contract)  │
                          │                                              │
                          │  tools ──▶ AIF_SRV OData V4 (mock/S4)         │
                          │  tools ──▶ AI Core document grounding (RAG)   │
                          │  LLM   ──▶ SAP AI Core via LiteLLM (sap/gpt-4.1)│
                          └──────────────────────────────────────────────┘
```

- **Transport:** A2A v1.x with `enable_v0_3_compat=True` (Joule speaks v0.3
  `message/send`; both work on the same server).
- **LLM:** `sap/gpt-4.1` through LiteLLM → SAP AI Core.
- **Data:** AIF_SRV OData V4 service (locally the public mockup; on CF a BTP
  Destination). RAG resolution content comes from AI Core document grounding.
- **Output:** every completed turn returns a human-readable **text** part *and* a
  structured **data** part — the `{ message, intent, data }` card contract.

---

## 2. The request lifecycle

A single user turn flows through `app/agent.py → CodemineAgent.stream()`:

1. **Status update** — yields `"Processing your request..."`.
2. **History assembly** — prior turns from A2A task history / Agent Memory are
   prepended so context-carried parameters work (e.g. "now resolve it").
3. **Tool-loop** — the LangGraph graph runs `model ⇄ tools` until the LLM stops
   calling tools. **This is where intent + parameters are read** (see §4).
4. **Finalizer** — every completed turn is packaged into the `{ message, intent,
   data }` card contract (§3) and returned as the A2A `data` part.

The bridge to A2A is `app/agent_executor.py`: each yielded item becomes A2A
message parts — a **text part** (`content`) and, when present, a **data part**
(`data`). Joule reads the data part.

---

## 3. How the agent stays "structured" for Joule

The original Joule agents returned a `{ message, intent, data }` JSON object that
Joule's `invoke_agent.yaml` routed on (`intent`) and fed to `render_*_card`
functions. To keep that **exact contract** after moving the brain into Python, the
agent runs a **finalizer step** after the tool-loop.

### Why a separate finalizer (and why JSON mode, not a typed schema)

SAP AI Core's structured-output endpoint (`response_format = json_schema`) runs in
**OpenAI strict mode**:

- every property must appear in `required`,
- `additionalProperties: false` is enforced,
- every schema node must have a `type`, and **discriminated unions are rejected**.

So binding a Pydantic union/`Optional` model via `.with_structured_output(...)`
fails with `400 "Missing 'count'"` / `"schema must have a 'type' key"`. Instead the
finalizer uses **JSON mode**:

```python
self.finalizer = ChatLiteLLM(
    model="sap/gpt-4.1",
    model_kwargs={"response_format": {"type": "json_object"}},
)
```

The **shape** is driven by `FINALIZER_PROMPT` (which documents each intent's `data`
fields) and then **parsed + normalized** through a flat Pydantic envelope
(`app/schemas.py → CardEnvelope`). The finalizer is told to copy values **only**
from the tool results (strict grounding — no invented interfaces/counts/steps).

### The contract

```json
{
  "message": "<Markdown fallback of the answer>",
  "intent":  "<one of the intents below>",
  "data":    { /* per-intent payload */ }
}
```

`CardEnvelope` keeps `data` as a flat superset of all intents' fields (all
optional), so the JSON validates regardless of intent and the renderer for that
intent reads only the fields it needs.

---

## 4. How intent + parameters are read

There is **no separate classifier**. The LLM's **tool choice in the loop already
encodes the intent**, and the **tool arguments encode the parameters**:

- "list interfaces" → the LLM calls `list_interfaces_tool` → intent `interface_list`.
- "errors for ORDERS in 2025" → `calculate_date_range_tool("2025")` then
  `list_error_messages_tool(from, to, interface_name="ORDERS")` → intent `worklist`.
- "fix message <GUID>" → `get_message_log_tool(guid)` + per-error
  `run_doc_error_catalog_tool(...)` → intent `resolution`.

The **finalizer then labels** the result (`intent`) and shapes `data`. Parameters
that arrive across turns (e.g. a GUID mentioned earlier) are available because the
conversation history is replayed into the prompt.

For any **time-bounded** request the LLM first calls `calculate_date_range_tool`,
which turns free text into OData V4 bounds:

| Input | from / to |
|---|---|
| `2025` | `2025-01-01T00:00:00Z` … `2025-12-31T23:59:59Z` |
| `2025-01-01 to 2025-06-30` | those two, normalized to `…Z` |
| `last week` (relative) | returns `resolved:false` — the LLM resolves it |

---

## 5. The intents

| Intent | User asks | Tools used | `data` payload | Joule render |
|---|---|---|---|---|
| **interface_list** | "which/list interfaces available" | `list_interfaces_tool` (+ key fields) | `count`, `interfaces[]` `{namespace, interfaceName, interfaceVersion, sapModule, about, searchBy}` | interface card |
| **worklist** | "show errors for a period / interface" | `calculate_date_range_tool`, `list_error_messages_tool` | `count`, `messages[]` `{messageGuid, namespace, interfaceName, interfaceVersion, status, processDate, logMessage}` | worklist card |
| **statistics / health** | "health check", "statistics", "error counts" | `calculate_date_range_tool`, `get_interface_statistics_tool` | `interfaceCount`, `interfaces[]` `{…, total, errors, warnings, success, health}` | statistics card |
| **message_detail** | a GUID read ("show message <GUID>") | `get_message_log_tool` | `messageGuid`, `interface`, `logCount`, `logEntries[]` `{msgType, msgId, msgNo, text}` | message card |
| **business_key** | a business object (customer, vendor, PO, cost center…) | `get_key_fields_tool` (+ `find_messages_by_key_value_tool` if a value is given) | `mode` (`interfaces`/`messages`), `count`, `interfaces[]` or `messages[]` | interface or worklist card |
| **resolution** | "fix/resolve message <GUID>" | `get_message_log_tool` + per-error `run_doc_error_catalog_tool` | `messageGuid`, `interface`, `errorCount`, `summary`, `errors[]` `{msgId, msgNo, messageText, rootCause, resolutionSteps[], restartSafe, grounded, sourceText}` | resolution card |
| **analysis** | "full analysis / monitoring report / overview" | date range + statistics + worklist + (top-error) RAG | *(none — Markdown report in `message`)* | Markdown |

`health` is the same card as `statistics`. `analysis` is the one intent that
renders as **Markdown** (not a card).

### Health thresholds

`Critical` = 6+ errors · `Warning` = 1–5 · `Healthy` = 0 (per interface, per period).

---

## 6. Tools (LangChain `@tool`)

AIF_SRV (OData V4) tools — see `app/agent.py`:

| Tool | Entity set | Purpose |
|---|---|---|
| `calculate_date_range_tool` | — | free-text period → V4 from/to bounds |
| `list_interfaces_tool` | `IfKeySet` + `KeyFieldsSet` | interfaces + searchable key-field labels |
| `list_error_messages_tool` | `IndexTableGenericSet` | worklist for a date window / interface / status |
| `get_interface_statistics_tool` | `InterfaceStatistics(p_datetime_from,p_datetime_to)` | per-interface counts + health |
| `get_key_fields_tool` | `KeyFieldsSet` | map a business object → key field (FieldName/Label/SemObj) |
| `find_messages_by_key_value_tool` | `KeyFieldValueSet` | messages matching FieldName **and** FieldValue |
| `get_message_log_tool` | `MessageLogSet` (`MsgType='E'`) | a message's error log entries by GUID |
| `run_doc_error_catalog_tool` | AI Core grounding | root cause + resolution steps for an error code |

> **Deprecated & removed from the tool list:** `run_analysis_tool` and
> `get_message_details_tool` (they used V2 `aifmonitoring/*` endpoints that 400
> against AIF_SRV). Replaced by the statistics/worklist/message-log tools above.

### OData V4 notes

- Envelope is `{ "value": [...] }` (not V2 `{ d: { results } }`).
- Dates are `Edm.DateTimeOffset` with trailing `Z`, filtered bare:
  `ProcessDate ge 2025-01-01T00:00:00Z and ProcessDate le 2025-12-31T23:59:59Z`.
- **Spaces in `$filter` must be `%20`, not `+`** — SAP Gateway returns 400 on `+`.
  `_aif_get()` builds the query string with `urllib.parse.quote` to enforce this.

---

## 7. Configuration

`.env` (local) / env vars (CF):

```dotenv
# LLM (SAP AI Core via LiteLLM)
AICORE_AUTH_URL=...            # bare host, NO /oauth/token
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=...
AICORE_RESOURCE_GROUP=default

# AIF backend — local mock (public, no auth):
AIF_ODATA_URL=https://aif-standard-mockup-service.cfapps.eu10-005.hana.ondemand.com/sap/opu/odata/sap/AIF_SRV
# AIF_OAUTH_* — leave blank for the mock; set for a protected backend.

DISABLE_AUTH=true              # local dev: skip XSUAA JWT validation
```

On Cloud Foundry the AIF URL + auth come from a **BTP Destination** (`dest-aif-service`);
`_get_destination_url()` prefers `AIF_ODATA_URL` when set, else resolves the bound
destination from `VCAP_SERVICES`.

---

## 8. Running & testing locally

```bash
# from the repo root
./run-local.ps1 -Agents "aif"        # venv + python -m app.main --port 9000
# agent card:  http://localhost:9000/.well-known/agent-card.json
```

Send an A2A `message/send` (Bruno collection: `bruno/aif-agent/`, env `local`):

```bash
curl -s -X POST http://localhost:9000/ -H "Content-Type: application/json" -d '{
  "jsonrpc":"2.0","id":"1","method":"message/send",
  "params":{"message":{"role":"user","messageId":"m1","contextId":"c1",
    "parts":[{"kind":"text","text":"Show me AIF interface analysis for 2025"}]}}}'
```

The response carries the **text** part (answer / report) and, on card intents, a
**data** part with `intent` + `data`. Verified intents against the live mock:

| Query | intent | result |
|---|---|---|
| "list interfaces available for monitoring" | `interface_list` | 8 interfaces |
| "Show me errors for ORDERS in 2025" | `worklist` | 3 errors |
| "Run a health check for all interfaces in 2025" | `health` | 8 interfaces |
| "Show me message <GUID>" | `message_detail` | log entries |
| "How do I fix message <GUID>" | `resolution` | per-error fixes + summary |
| "Which interfaces are connected to Customer" | `business_key` | mode=interfaces |
| "Show me AIF interface analysis for 2025" | `analysis` | full Markdown report |

Example Bruno test: `bruno/aif-agent/06 analysis-2025.bru` asserts the response
has both a Markdown report and a structured `data` part.

---

## 8a. Multi-turn conversational memory (per context_id)

A multi-turn agent passes an **evolving history of messages (user + assistant
roles) plus tool usage** back to the LLM each turn so it remembers prior actions.
This agent does that per A2A `context_id`:

- **Message history** — prior user/assistant turns come from the A2A task history
  (always available) or SAP Agent Memory (when bound on CF). Prepended to the
  prompt each turn.
- **Tool-usage carry-forward** — the actionable result of the previous turn's
  tools (the **MessageGuids** of errors shown, and the **interfaces** listed) is
  distilled by `_findings_from_card()` into a compact reference and injected as an
  **assistant message** (`_context_findings[context_id]`). This is needed because
  the GUID lives in the structured `data` part, not the visible text — so without
  it, a follow-up like "fix that error" would have no GUID to act on.

This lets references resolve without the user repeating identifiers:

| Turn 1 | Follow-up (no identifier) | Resolves to |
|---|---|---|
| "show errors for PAYMENTS in 2025" (worklist) | "details of **that message**" | message_detail on the remembered GUID |
| …then | "**fix it**" | resolution on the same GUID |
| "which interfaces are available" | "errors for **the PAYMENTS one**" | worklist for that interface |
| "health check 2025" | "fix the **critical** one's errors" | the critical interface |

The system prompt (the MULTI-TURN CONTEXT rule) instructs the LLM to resolve such
references against the injected blocks and never to invent a GUID/interface.

## 9. Joule integration (BYOA)

Joule reaches this agent through a **BTP Destination** and an `agent-request`
(`agent_type: remote`). The Joule side keeps the **existing** card layer:

- `invoke_agent.yaml` reads the remote **data part** (`{message, intent, data}`),
  routes on `intent`, and calls the matching `render_*_card` function — unchanged
  from the original Joule project.
- `render_interface_card`, `render_worklist_card`, `render_statistics_card`,
  `render_message_card`, `render_resolution_card` render the cards; `analysis`
  renders as Markdown text.

The agent supports A2A v0.3 (`enable_v0_3_compat=True`), so Joule's `message/send`
and v1.x `SendMessage` both work without code changes.

---

## 10. Source map

| File | Responsibility |
|---|---|
| `app/main.py` | A2A server, agent card, Starlette routes, telemetry-first init |
| `app/agent.py` | LangGraph tool-loop, **tools**, system prompt, **finalizer** |
| `app/schemas.py` | `CardEnvelope` / `CardData` — the structured contract |
| `app/agent_executor.py` | A2A bridge — emits text + **data** parts |
| `app/bootstrap.py` | `VCAP_SERVICES` → `AICORE_*`, OTel init |
| `app/auth.py` | XSUAA JWT middleware (skipped when `DISABLE_AUTH=true`) |
| `docs/AGENT_DESIGN.md` | this document |
