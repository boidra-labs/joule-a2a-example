"""LangGraph AIF interface-monitoring agent.

A2A agent that monitors SAP AIF interfaces over AIF_SRV (OData V4) and returns the
Joule { message, intent, data } card contract. Flow per turn:
  1. Load conversation history (A2A task history / Agent Memory)
  2. Tool-loop (model ⇄ tools) — the LLM picks tools = the intent, args = params
  3. Finalizer (JSON mode) packages the tool results into { message, intent, data }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Literal, Optional

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from opentelemetry import context as otelcontext, trace
from sap_cloud_sdk.core.telemetry import invoke_agent_span

from app.schemas import CardResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool tracing — we rely ENTIRELY on the SAP Cloud SDK auto-instrumentation
# (auto_instrument -> Traceloop), which already emits a span per @tool call.
# Adding our own tracer.start_as_current_span() per tool produced DUPLICATE
# (nested) tool spans in the dashboard. To stay standard with SAP and avoid the
# duplicates, `tracer` is a no-op: the existing `with tracer.start_as_current_span(
# ...) as span:` / `span.set_attribute(...)` lines in the tools keep working but
# emit nothing. The SAP auto-instrumented span is the single source of truth.
# ---------------------------------------------------------------------------
class _NoopSpan:
    def set_attribute(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _NoopTracer:
    def start_as_current_span(self, *_args, **_kwargs) -> "_NoopSpan":
        return _NoopSpan()


tracer = _NoopTracer()  # tool-level spans come from SAP auto-instrumentation

LLM_MODEL = "sap/gpt-4.1"
AGENT_ID = "aif-analysis-agent"

DEST_AIF_ALIAS = "dest-aif-service"

AICORE_GROUNDING_REPOSITORY = os.environ.get(
    "AICORE_GROUNDING_REPOSITORY", "a7554f34-629a-4433-8548-8c736ffbef55"
)

# Minimum vector-similarity score (chunk aggregatedScore, 0..1) for a grounding
# hit to count as a real match. Below this the retriever only returned weak
# nearest-neighbours (no catalog entry for the error) and we must NOT present
# them as the answer. Calibrated from live scores against this catalog:
#   good  "resolution steps for KI/260" -> best chunk 0.52  (doc exists)
#   bogus "partner function SP missing" -> best chunk 0.39  (no doc)
# 0.45 sits cleanly between them. Tunable without redeploy via this env var
# (raise toward 0.50 if false positives persist, lower if real lookups are
# wrongly rejected).
AICORE_GROUNDING_MIN_SCORE = float(os.environ.get("AICORE_GROUNDING_MIN_SCORE", "0.45"))

# Grounding backend for run_doc_error_catalog_tool:
#   "retrieval"     -> direct document-grounding retrieval/search + a relevance
#                      score gate (precise; rejects weak nearest-neighbour hits).
#   "orchestration" -> AI Core Orchestration /completion (retrieve -> template ->
#                      LLM) returning a grounded {root_cause, resolution_step[]}
#                      JSON. No score gate; relies on the prompt's
#                      resolvable_from_references self-assessment. Tunable via env.
GROUNDING_MODE = os.environ.get("GROUNDING_MODE", "retrieval").strip().lower()
# Full URL of the orchestration deployment's /completion endpoint (required only
# when GROUNDING_MODE=orchestration). e.g.
# https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/<id>/v2/completion
GROUNDING_ORCH_URL = os.environ.get("AICORE_ORCH_COMPLETION_URL", "")
# Orchestration model (sent inline in config.modules.prompt_templating.model).
GROUNDING_ORCH_MODEL = os.environ.get("AICORE_ORCH_MODEL", "gpt-4.1")
GROUNDING_ORCH_MODEL_VERSION = os.environ.get("AICORE_ORCH_MODEL_VERSION", "latest")

# Context var set by stream() so tool functions can read the current user/session key
# without needing an extra parameter threaded through LangGraph.
_current_session_key: ContextVar[str] = ContextVar("_current_session_key", default="default")

# Per-conversation findings memory, keyed by A2A context_id. Stores a compact,
# machine-readable reference of the errors surfaced in the LAST worklist /
# analysis / message_detail turn — each with its MessageGuid. On a follow-up like
# "how do I fix that error" (no GUID given), this is injected as an assistant
# message so the LLM can recover the GUID and call get_message_log_tool on it.
# In-process (works locally and on CF without the Agent Memory binding); the SAP
# Agent Memory path still persists the full turn text separately when bound.
_context_findings: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Short-term conversation memory (in-process, per A2A context_id).
#
# Used when the SAP Agent Memory service is not bound: keeps the last
# _SHORT_TERM_MAX_TURNS (user, assistant) turns per context_id in process so
# multi-turn conversations retain history. Survives across requests while the
# app runs; reset on restart, and NOT shared across instances. The SAP Agent
# Memory path (when a client is bound) takes precedence and persists durably.
# ---------------------------------------------------------------------------
from collections import deque  # noqa: E402

_SHORT_TERM_MAX_TURNS = 10
_short_term_memory: dict[str, deque] = {}


def _short_term_remember(context_id: str, query: str, response: str) -> None:
    """Append a (user, assistant) turn to this context's short-term memory."""
    if not context_id:
        return
    dq = _short_term_memory.get(context_id)
    if dq is None:
        dq = deque(maxlen=_SHORT_TERM_MAX_TURNS)
        _short_term_memory[context_id] = dq
    dq.append({"user": query, "assistant": response})


def _short_term_history(context_id: str) -> list:
    """Return this context's short-term turns as LangChain messages (oldest first)."""
    out: list = []
    for turn in _short_term_memory.get(context_id, ()):  # type: ignore[arg-type]
        out.append(HumanMessage(content=turn["user"]))
        out.append(AIMessage(content=turn["assistant"]))
    return out


def _short_term_clear(context_id: str | None = None) -> None:
    """Clear one context's short-term memory, or all of it (for tests)."""
    if context_id is None:
        _short_term_memory.clear()
    else:
        _short_term_memory.pop(context_id, None)


def _findings_from_card(data: dict | None) -> str:
    """Build a short reference of the PREVIOUS answer (errors with GUIDs, and
    interfaces) so a later turn can resolve references like 'that error', 'fix it',
    'the PAYMENTS one', 'the first interface', 'show errors for that interface'
    without the user repeating identifiers. Returns '' when nothing is referable.
    """
    if not isinstance(data, dict):
        return ""
    err_lines: list[str] = []
    iface_lines: list[str] = []

    def add_err(guid, interface, text):
        if guid:
            err_lines.append(f"- {interface or ''} | {text or ''} | MessageGuid={guid}")

    def add_iface(ns, name, ver, extra=""):
        if name:
            tag = f"- {ns or ''}/{name} v{ver or ''}"
            iface_lines.append(tag + (f" | {extra}" if extra else ""))

    # worklist / business_key messages (each has a GUID)
    for m in (data.get("messages") or []):
        if isinstance(m, dict):
            iface = f"{m.get('namespace','')}/{m.get('interfaceName','')} v{m.get('interfaceVersion','')}"
            add_err(m.get("messageGuid"), iface, m.get("logMessage"))
    # message_detail / resolution (single message)
    if data.get("messageGuid"):
        errs = data.get("errors") or data.get("logEntries") or []
        text = ""
        if errs and isinstance(errs[0], dict):
            text = errs[0].get("messageText") or errs[0].get("text") or ""
        add_err(data.get("messageGuid"), data.get("interface", ""), text)
    # interface_list / business_key(interfaces): interfaces the user can act on
    for it in (data.get("interfaces") or []):
        if isinstance(it, dict):
            extra = it.get("about") or it.get("searchBy") or ""
            # statistics rows carry health/errors
            if "errors" in it and isinstance(it.get("errors"), int):
                extra = f"errors={it.get('errors')} health={it.get('health','')}".strip()
            add_iface(it.get("namespace"), it.get("interfaceName"), it.get("interfaceVersion"), extra)

    blocks: list[str] = []
    if err_lines:
        blocks.append(
            "KNOWN ERRORS FROM THE PREVIOUS ANSWER (use the MessageGuid when the user "
            "refers to one of these without giving a GUID — e.g. 'fix that error'):\n"
            + "\n".join(err_lines[:20])
        )
    if iface_lines:
        blocks.append(
            "INTERFACES FROM THE PREVIOUS ANSWER (resolve references like 'the first "
            "one' / 'the PAYMENTS one' / 'that interface' to these namespace/name/"
            "version, e.g. to show their errors or statistics):\n"
            + "\n".join(iface_lines[:20])
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Destination resolver
# ---------------------------------------------------------------------------

def _get_destination_url(alias: str) -> tuple[str, dict]:
    env_key = "AIF_ODATA_URL"
    url = os.environ.get(env_key, "")
    if url:
        return url, {}

    vcap_raw = os.environ.get("VCAP_SERVICES", "{}")
    try:
        vcap = json.loads(vcap_raw)
        dest_bindings = vcap.get("destination", [])
        if dest_bindings:
            creds = dest_bindings[0].get("credentials", {})
            dest_service_url = creds.get("uri", "")
            client_id = creds.get("clientid", "")
            client_secret = creds.get("clientsecret", "")
            token_url = creds.get("url", "") + "/oauth/token"
            token_resp = httpx.post(
                token_url,
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
                timeout=10,
            )
            token_resp.raise_for_status()
            access_token = token_resp.json()["access_token"]
            dest_resp = httpx.get(
                f"{dest_service_url}/destination-configuration/v1/destinations/{alias}",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            dest_resp.raise_for_status()
            dest_data = dest_resp.json()
            dest_url = dest_data.get("destinationConfiguration", {}).get("URL", "")
            dest_token = dest_data.get("authTokens", [{}])[0].get("value", "")
            auth_headers = {"Authorization": f"Bearer {dest_token}"} if dest_token else {}
            return dest_url, auth_headers
    except Exception:
        logger.exception("Could not resolve destination '%s' from VCAP_SERVICES", alias)

    return "", {}


# ---------------------------------------------------------------------------
# AIF_SRV (OData V4) helpers — shared by the monitoring tools below.
# The mock service is OData V4: response envelope is { value: [...] } (NOT the V2
# { d: { results } }), dates are Edm.DateTimeOffset with a trailing Z and are
# filtered as bare literals: ProcessDate ge 2025-01-01T00:00:00Z.
# ---------------------------------------------------------------------------

# AIF status code -> meaning (IndexTableGenericSet.Status)
AIF_STATUS = {"E": "Error", "W": "Warning", "S": "Success", "P": "In process", "X": "Aborted"}


def _aif_get(path: str, params: dict | None = None) -> tuple[list[dict], Optional[str]]:
    """GET an AIF_SRV entity set and return (rows, error).

    rows is the OData V4 `value` array (empty on error). error is None on success
    or a short message string. Resolves the base URL from AIF_ODATA_URL (local
    mock) or the bound destination, and attaches optional OAuth/destination auth.
    """
    dest_url, auth_headers = _get_destination_url(DEST_AIF_ALIAS)
    if not dest_url:
        return [], f"Destination '{DEST_AIF_ALIAS}' / AIF_ODATA_URL not resolved"

    oauth_url = os.environ.get("AIF_OAUTH_URL", "")
    oauth_client_id = os.environ.get("AIF_OAUTH_CLIENT_ID", "")
    oauth_client_secret = os.environ.get("AIF_OAUTH_CLIENT_SECRET", "")
    if oauth_url and oauth_client_id and oauth_client_secret:
        token_resp = httpx.post(
            oauth_url,
            data={"grant_type": "client_credentials"},
            auth=(oauth_client_id, oauth_client_secret),
            timeout=10,
        )
        token_resp.raise_for_status()
        auth_headers["Authorization"] = f"Bearer {token_resp.json()['access_token']}"

    # Build the query string manually with %20-encoding. httpx (and urlencode's
    # default) encode spaces as '+', which SAP Gateway rejects in $filter with a
    # 400. quote(safe="...") keeps OData operators readable and uses %20 for spaces.
    from urllib.parse import quote

    qp = {"$format": "json"}
    if params:
        qp.update(params)
    # Keep OData punctuation readable; spaces are NOT in `safe`, so quote() emits
    # %20 (the form SAP Gateway accepts) rather than '+'.
    safe = "()/',:=$"
    query = "&".join(
        f"{quote(k, safe='$')}={quote(str(v), safe=safe)}" for k, v in qp.items()
    )
    url = f"{dest_url.rstrip('/')}/{path.lstrip('/')}?{query}"
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, headers={"Accept": "application/json", **auth_headers})
            resp.raise_for_status()
            body = resp.json()
        # V4 collection -> body["value"]; V4 single entity -> the body itself.
        if isinstance(body, dict) and "value" in body:
            return body["value"], None
        return [body] if isinstance(body, dict) else [], None
    except Exception as exc:  # noqa: BLE001
        logger.exception("AIF_SRV GET failed: %s", path)
        return [], str(exc)


def _v4_dt(value: str, is_end: bool = False) -> str:
    """Normalise a date/datetime string to a V4 DateTimeOffset literal (…Z).

    Robust to partial inputs the LLM may pass directly (e.g. a bare year '2025'
    or a date '2025-03-01') so the OData $filter is always a valid literal like
    2025-01-01T00:00:00Z — never '2025T00:00:00Z'.

    When is_end=True the value is treated as an INCLUSIVE upper bound and padded
    to the END of the period at 23:59:59:
      '2026'        -> 2026-12-31T23:59:59Z   (end of year)
      '2025-03'     -> 2025-03-31T23:59:59Z   (end of month)
      '2025-03-01'  -> 2025-03-01T23:59:59Z   (end of day)
    A date-only 'from' still starts at 00:00:00.
    """
    import calendar

    v = value.strip()
    if not v:
        return v
    if "T" not in v:
        parts = v.split("-")
        if is_end:
            if len(parts) == 1:            # 'YYYY' -> Dec 31
                v = f"{parts[0]}-12-31"
            elif len(parts) == 2:          # 'YYYY-MM' -> last day of month
                last = calendar.monthrange(int(parts[0]), int(parts[1]))[1]
                v = f"{parts[0]}-{parts[1]}-{last:02d}"
            # len 3 -> the given day
            v = v + "T23:59:59Z"
        else:
            if len(parts) == 1:            # 'YYYY' -> Jan 1
                v = f"{parts[0]}-01-01"
            elif len(parts) == 2:          # 'YYYY-MM' -> day 1
                v = f"{parts[0]}-{parts[1]}-01"
            v = v + "T00:00:00Z"
    if not v.endswith("Z"):
        v = v + "Z"
    return v


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SAP AIF (Application Interface Framework) interface monitoring analyst for Central Finance.

Your tools:
- calculate_date_range_tool: resolve a free-text period ('2025', 'last week', a range) into from/to DateTimeOffset bounds. Call FIRST for any time-bounded request.
- list_interfaces_tool: list interfaces available for monitoring (+ searchable key-field labels).
- list_error_messages_tool: the worklist — AIF messages for a date window, optionally filtered by interface/status.
- get_interface_statistics_tool: per-interface counts + health for a date window.
- get_key_fields_tool: key-field definitions (FieldName/Label/SemObj) per interface — maps a business object to its key field.
- find_messages_by_key_value_tool: find messages by a business-key FieldName + FieldValue.
- get_message_log_tool: read the ERROR log entries (MsgType='E') of one message by GUID.
- run_doc_error_catalog_tool: retrieves root cause and resolution steps for a specific error code from the SharePoint document catalog.
- get_interface_details_tool: returns a structured explanation of a SAP AIF interface given its namespace, name, and version.

Rules for AIF analysis (rules 1-10) — the full monitoring report (intent=analysis):
1. First call calculate_date_range_tool(period) to resolve the user's period to
   from/to. Then call, in any order: get_interface_statistics_tool(date_from, date_to)
   for the per-interface counts (ALL message types: success/warning/error/in-process),
   list_error_messages_tool(date_from, date_to, status='E') for the error rows, AND
   list_interfaces_tool() for the interface descriptions (used to label the traffic
   table for business users). (Do NOT use run_analysis_tool — it is deprecated.)
2. If the statistics and error tools both fail or return empty, return exactly: "No AIF interface data available for [period]." — nothing else.
3. If there are no errors at all, skip steps 4-8 (do NOT call run_doc_error_catalog_tool); go straight to step 9 with resolutions=[] and grounded_total=0.
4. If errors exist, group the error rows by distinct error (MSGID/MSGNO, or by log-message text when MSGID/MSGNO are absent) and rank by occurrence count (most common first). Take the TOP 20 distinct errors (break ties lexicographically). Let grounded_total = the number of distinct errors that exist (may be >20).
5. For EACH of those top-20 distinct errors, build the grounding query as: MSGID + "/" + MSGNO + " " + MSGTX (omit trailing space if MSGTX is absent).
6. Call run_doc_error_catalog_tool ONCE per distinct error (so up to 20 calls total — never more). Do not ground errors outside the top 20.
7. For each error build a resolution entry: { msgId, msgNo, messageText, affectedInterfaces:[...], occurrences:N, restartSafe:true|false, rootCause, resolutionSteps:[...], sourceText }. ALWAYS set msgId and msgNo to the real values (never blank). If grounding returns no match/fails, set rootCause = "No resolution guidance available in the document catalog for this error.", resolutionSteps = [], sourceText = "SAP standard documentation for <MSGID>/<MSGNO>". The builder shows at most the first 3 resolutionSteps, so order the most important steps first.
8. rootCause and resolutionSteps MUST come VERBATIM from the grounding response `groundingText`/grounded fields. Never synthesize them from your own knowledge.
9. Call build_analysis_report_tool(period, date_from, date_to, statistics=<interfaces[] from get_interface_statistics_tool>, error_rows=<messages[] from list_error_messages_tool>, resolutions=<the entries from step 7, or []>, grounded_total=<count from step 4, or 0>, interfaces_catalog=<interfaces[] from list_interfaces_tool>). Return its `report` value VERBATIM as your answer — do NOT rewrite, re-summarise, or recompute any of its tables, counts, or grade.
10. Criticality (used by the builder): 6+ occurrences = Critical, 1-5 = Warning, 0 = Healthy. Do NOT append any action menu or follow-up prompt after the report.

Rules for interface detail lookup (rule 19):
19. When the user asks about an interface by providing a namespace (NS), interface name (IFNAME), and/or version (IFVERSION) — or asks what an interface does, what it is for, or for insights about it — call get_interface_details_tool with those values.
    After the tool returns, use your SAP knowledge to produce a rich Markdown response in this exact format:

## Interface: [IFNAME] ([IFVERSION])
**Namespace:** [namespace]

### Overview
[2-3 sentence plain-language description of what this interface does in the context of SAP Central Finance / AIF]

### Purpose
[What business process or data flow this interface handles — e.g. accounting document replication, cost object mapping, etc.]

### Technical Details
| Field | Value |
|-------|-------|
| Namespace | [namespace] |
| Interface Name | [ifname] |
| Version | [ifversion] |
| Protocol | IDoc / BAPI / Proxy (based on your knowledge) |
| Direction | Inbound / Outbound |
| Source System | [likely source, e.g. ECC, S/4HANA] |
| Target System | Central Finance |

### Common Error Scenarios
- [Most frequent error type for this interface]
- [Second common issue]

### Tips for Monitoring
- [Practical tip 1]
- [Practical tip 2]

Rules for agent capabilities lookup (rule 20):
20. When the user asks what THIS agent can do, what capabilities it has, or what queries it supports — call get_agent_capabilities_tool('self').
    Then present the result in this Markdown format:

## [Agent Name] Capabilities

**Description:** [description from card]

### Skills
| Skill | Description |
|-------|-------------|
| [skill name] | [skill description] |

### Example Queries
[bullet list of examples from the skill]

Rules for message detail lookup (rule 18):
18. When the user provides a MSGGUID (a 32-character hex string) or asks to look up a specific AIF message, call get_message_log_tool with that GUID (NOT get_message_details_tool, which is deprecated for this service). Present the returned log entries (msgType, msgId, msgNo, text) in a readable Markdown table. This is the message_detail intent — a direct single-message lookup; do not run an analysis.

Rules for no-action reply (rule 17):
17. If the user replies "no thanks" after a report, respond: "Understood. Let me know if you need anything else."

Rules for AIF monitoring intents (rules 21-27) — use the AIF_SRV tools. For any
time-bounded request, FIRST call calculate_date_range_tool(period) to get from/to
(e.g. "2025" -> 2025-01-01T00:00:00Z .. 2025-12-31T23:59:59Z), then pass those to
the data tool. Do NOT invent dates.
DERIVING the `period` argument — pass the user's phrasing through; do NOT
pre-compute or substitute dates yourself:
  - Closed range ("2023 to today", "between Jan and June"): pass it verbatim
    ("2023 to today"). Keywords like today/now/yesterday resolve to the real
    current date — never end a range at "end of this year".
  - Open-ended start ("from 2023", "since March", "2023 onwards"): pass it as
    "from <start>" (e.g. period="from 2023"). The end auto-fills to today. Do
    NOT pass the bare year "2023" for an open-ended request — that would wrongly
    stop at 2023-12-31.
  - NO time bound at all (e.g. "Run a health check across all interfaces" with
    no date words): pass period="all" to get an unrestricted window (far past ->
    today). Do NOT invent "last week"/"last 7 days" or any window the user did
    not ask for.

MULTI-TURN CONTEXT: the conversation may contain "KNOWN ERRORS FROM THE PREVIOUS
ANSWER" and "INTERFACES FROM THE PREVIOUS ANSWER" reference blocks. When the user
refers to something from the previous turn WITHOUT repeating its identifier —
"fix that error", "resolve it", "the PAYMENTS one", "show its details", "the
first interface", "errors for that interface", "the critical one" — resolve the
reference against those blocks and reuse the MessageGuid / namespace+name+version
from there. Carry forward the previously used time period too unless the user
gives a new one. If exactly one candidate exists, use it; only ask to clarify
when several genuinely match. Never invent a GUID or interface.
21. "which/list interfaces available / can be monitored" (no business object) ->
    call list_interfaces_tool. Present a short list; the UI renders an interface card.
22. "list/show errors" for a period (and optional interface) -> resolve dates, then
    call list_error_messages_tool(date_from, date_to, namespace?, interface_name?,
    interface_version?, status='E'). This is the worklist.
23. "health check", "which interfaces are healthy/failing", "statistics", "error
    counts per interface" for a period -> resolve dates, then
    get_interface_statistics_tool(date_from, date_to). This is the statistics/health
    intent: respond with a SHORT per-interface health summary as a Markdown table.
    The table MUST include a column for EVERY message type returned by the tool, not
    just errors: Interface | Total | Success | Warnings | Errors | In-Process | Health.
    Always show the Success, Warnings and In-Process columns even when their value is
    0 — never omit message types. Copy the counts verbatim from the tool's
    interfaces[] (total, success, warnings, errors, inProcess, health). Do NOT produce
    the big "Interface Monitoring Analysis Report" template and do NOT append the
    action menu — that template belongs to the analysis intent (rule 27) only.
24. A business object (customer, vendor, business partner, cost center, PO, sales
    order, material, company code…): call get_key_fields_tool to map the object term
    to its key field (match on Label/SemObj). If the user gave a VALUE, then call
    find_messages_by_key_value_tool(field_name, field_value). If only the object
    (no value), report which interfaces carry it (from get_key_fields_tool).
25. A plain GUID READ ("show/look up message <GUID>") -> get_message_log_tool(guid).
    This is message_detail, NOT a resolution.
26. "fix/resolve/triage message <GUID>" -> get_message_log_tool(guid) to get every
    error (MsgType='E'); for EACH error call run_doc_error_catalog_tool with
    'MSGID/MSGNO Text'; ground rootCause + resolutionSteps verbatim from
    groundingText (never invent). Then synthesise a SHORT message-level summary
    across all errors. This is the resolution intent.
    FOLLOW-UP WITHOUT A GUID: when the user says "fix that error", "how do I
    resolve it", "resolve the PAYMENTS one", etc. and gives NO GUID, look in the
    "KNOWN ERRORS FROM THE PREVIOUS ANSWER" list in the conversation for the
    matching error (by interface and/or message text) and use its MessageGuid as
    the GUID for get_message_log_tool. If exactly one error was previously shown,
    use that one. Only ask the user to clarify if several match and the reference
    is ambiguous. Never invent a GUID.
27. ONLY when the user explicitly asks for a "full analysis", "monitoring report",
    "overview", or "analyze interfaces" for a period -> this is the analysis intent:
    follow analysis rules 1-10, which END by calling build_analysis_report_tool and
    returning its `report` verbatim. A plain "health check" / "statistics" is NOT
    analysis — use rule 23 instead.

The full report Markdown is assembled deterministically by build_analysis_report_tool
(rule 9): a risk callout; a health scorecard with a letter grade, a success rate, and a
breakdown of ALL message types (success / warning / error / in-process, with counts and
percentages from the statistics); an Active Interfaces table showing each interface's
business DESCRIPTION (not namespace/version) with all-type totals; a "Top 20 Most Common
Errors" list (ranked by occurrences, with criticality); and an "Error Resolutions" section
(top 20, each a short root cause + max 3 steps, ERROR-only, grounded verbatim). The report
is plain ASCII Markdown tables/text only — no charts, no emoji/icons — so SAP Joule renders
it cleanly. You never hand-build these sections — you only feed the tool the statistics,
error rows, grounded resolutions, and interface catalog, then return its output unchanged.
"""


# ---------------------------------------------------------------------------
# Finalizer — package the tool results into the Joule { message, intent, data }
# card contract. Runs AFTER the tool-loop, as a structured-output LLM call.
# The tool-loop already inferred the intent (which tools it chose) and the
# parameters (the args it passed); the finalizer just labels + shapes them.
# ---------------------------------------------------------------------------

FINALIZER_PROMPT = """You convert an AIF assistant's working transcript into ONE structured response object for the SAP Joule UI.

You are given: the user's request, the tools that were called and their JSON results, and the assistant's draft answer. Produce a single object with `intent`, a Markdown `message` fallback, and the matching `data`.

Pick `intent` from what the user asked for and which tools ran:
- interface_list — "which/list interfaces available". data = { count, interfaces:[{namespace, interfaceName, interfaceVersion, sapModule, about, searchBy}] }.
- worklist — "list/show errors for a period/interface". data = { count, messages:[{messageGuid, namespace, interfaceName, interfaceVersion, status, processDate, logMessage}] }.
- statistics / health — "health check", "statistics", "error counts per interface". data = { interfaceCount, interfaces:[{namespace, interfaceName, interfaceVersion, total, errors, warnings, success, inProcess, aborted, canceled, health}] }. total = all message types. health = Critical (6+ errors) / Warning (1-5) / Healthy (0). The `message` MUST be a Markdown table with a column for EVERY message type — Interface | Total | Success | Warnings | Errors | In-Process | Health — showing Success/Warnings/In-Process even when 0. Never reduce it to errors only.
- message_detail — a GUID read. data = { messageGuid, interface, logCount, logEntries:[{msgType, msgId, msgNo, text}] }.
- business_key — a business object (customer, vendor, cost center, PO…). data = { mode:'interfaces'|'messages', count, interfaces:[...] or messages:[...] }.
- resolution — "fix/resolve message <GUID>". data = { messageGuid, interface, errorCount, summary, errors:[{msgId, msgNo, messageText, rootCause, resolutionSteps[], restartSafe, grounded, sourceText}] }. sourceText is ALWAYS set: '[title](webUrl)' when a doc matched, else 'SAP standard documentation for <msgId>/<msgNo>'.
- analysis — a full monitoring report / overview. Set intent='analysis' and leave `data` null. (The full Markdown report from build_analysis_report_tool is the assistant draft; it is copied into `message` verbatim by the agent — do NOT rewrite, shorten, or relocate it.)

STRICT GROUNDING: copy values ONLY from the tool results and the assistant draft. Never invent interfaces, counts, error codes, root causes, or resolution steps. If a field is unknown use "—" (text) or 0 (number). Except for analysis (handled above), `message` is a concise Markdown fallback of the same content."""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def run_analysis_tool(period: str) -> dict[str, Any]:
    """Fetch AIF interface monitoring data from the AIF OData API.

    Args:
        period: Free-text time period, e.g. 'last 24 hours', 'last week'.
    """
    with tracer.start_as_current_span("run_analysis_tool", attributes={"aif.period": period}) as span:
        try:
            dest_url, auth_headers = _get_destination_url(DEST_AIF_ALIAS)
            if not dest_url:
                span.set_attribute("outcome", "misconfigured")
                return {"error": f"Destination '{DEST_AIF_ALIAS}' not resolved", "interfaceSummary": None}

            oauth_url = os.environ.get("AIF_OAUTH_URL", "")
            oauth_client_id = os.environ.get("AIF_OAUTH_CLIENT_ID", "")
            oauth_client_secret = os.environ.get("AIF_OAUTH_CLIENT_SECRET", "")
            if oauth_url and oauth_client_id and oauth_client_secret:
                token_resp = httpx.post(
                    oauth_url,
                    data={"grant_type": "client_credentials"},
                    auth=(oauth_client_id, oauth_client_secret),
                    timeout=10,
                )
                token_resp.raise_for_status()
                auth_headers["Authorization"] = f"Bearer {token_resp.json()['access_token']}"

            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{dest_url}/odata/v2/aifmonitoring/AIFErrorSummary",
                    headers={"Accept": "application/json", **auth_headers},
                )
                resp.raise_for_status()
                data = resp.json()
                span.set_attribute("outcome", "success")
                span.set_attribute("aif.record_count", len(data.get("d", {}).get("results", [])))
                return {"interfaceSummary": data}
        except Exception as exc:
            logger.exception("run_analysis_tool failed")
            span.set_attribute("outcome", "error")
            span.set_attribute("error.message", str(exc))
            return {"error": str(exc), "interfaceSummary": None}


_ORCH_SYSTEM_PROMPT = (
    "You are an SAP Application Interface Framework (AIF) error-resolution "
    "assistant supporting interface monitoring. Given an interface error "
    "identified by an error ID and an error number, diagnose the most likely "
    "root cause and produce concrete, ordered resolution steps. Ground every "
    "recommendation strictly in the supplied reference material (SAP Help and the "
    "customer's internal knowledge repository). Do not invent transaction codes, "
    "tables, SAP Notes, or configuration paths that are not supported by the "
    "references. If the references are insufficient to resolve the error, set "
    "resolvable_from_references to false, lower the confidence, and say what "
    "additional information is needed in root_cause. Keep steps short with max 3 "
    "steps and simple to understand. Respond with a single valid JSON object that "
    "conforms exactly to the required schema with 'root_cause', 'id' and "
    "'resolution_step' - no prose, no markdown, nothing outside the JSON."
)

_ORCH_USER_TEMPLATE = (
    "Resolve the following AIF interface monitoring error.\n\n"
    "Error ID: {{?errorId}}\nError Number: {{?errorNumber}}\n"
    "Error Message: {{?errorMessage}}\n\n"
    "Reference material:\n{{?groundingOutput}}"
)


def _orchestration_payload(error_id: str, error_number: str, error_message: str) -> dict[str, Any]:
    """Build the inline-config AI Core Orchestration /completion request body.

    Sends the full module config (document grounding over help.sap.com + a prompt
    template) plus placeholder_values for this error. The grounding module reads
    errorId/errorNumber/errorMessage, retrieves reference material, and injects it
    as groundingOutput into the prompt; the LLM returns the grounded resolution
    JSON. Pure (no I/O) so it is unit-testable.
    """
    return {
        "config": {
            "modules": {
                "grounding": {
                    "type": "document_grounding_service",
                    "config": {
                        "filters": [{
                            "id": "filter1",
                            "data_repositories": ["*"],
                            "search_config": {},
                            "data_repository_type": "help.sap.com",
                            "document_metadata": [],
                        }],
                        "placeholders": {
                            "input": ["errorId", "errorNumber", "errorMessage"],
                            "output": "groundingOutput",
                        },
                    },
                },
                "prompt_templating": {
                    "prompt": {
                        "template": [
                            {"role": "system", "content": _ORCH_SYSTEM_PROMPT},
                            {"role": "user", "content": _ORCH_USER_TEMPLATE},
                        ],
                    },
                    "model": {
                        "name": GROUNDING_ORCH_MODEL,
                        "version": GROUNDING_ORCH_MODEL_VERSION,
                        "params": {
                            "max_completion_tokens": 150,
                            "temperature": 0.1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                        },
                    },
                },
            },
        },
        "placeholder_values": {
            "errorId": error_id,
            "errorNumber": error_number,
            "errorMessage": error_message,
        },
    }


def _split_error_query(query: str) -> tuple[str, str, str]:
    """Split a 'MSGID/MSGNO MSGTX' grounding query into (id, number, text).

    'AIF/099 Processing terminated' -> ('AIF', '099', 'Processing terminated').
    Tolerant of a missing slash or missing text.
    """
    q = (query or "").strip()
    head, _, text = q.partition(" ")
    mid, slash, mno = head.partition("/")
    if not slash:
        return head, "", text.strip()
    return mid, mno, text.strip()


def _parse_orchestration_grounding(body: dict) -> dict[str, Any]:
    """Parse an AI Core Orchestration /completion response into a grounding result.

    The grounded answer is a JSON object (sometimes a JSON string) at
    final_result.choices[0].message.content with keys root_cause,
    resolution_step[], and optionally resolvable_from_references. The retrieved
    reference text sits at intermediate_results.grounding.data.grounding_result.

    Returns the same shape the analysis flow consumes:
    {match, rootCause, resolutionSteps[], groundingText, groundingSource}.
    No relevance score is available here, so we honour the model's
    resolvable_from_references self-assessment instead.
    """
    try:
        content = body["final_result"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {"match": False}

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {"match": False}
    if not isinstance(content, dict):
        return {"match": False}

    root_cause = (content.get("root_cause") or "").strip()
    steps = [s for s in (content.get("resolution_step") or []) if s]
    resolvable = content.get("resolvable_from_references", True)
    if not root_cause or resolvable is False:
        return {"match": False}

    grounding_text = ""
    try:
        grounding_text = body["intermediate_results"]["grounding"]["data"].get(
            "grounding_result", ""
        ) or ""
    except (KeyError, TypeError):
        grounding_text = ""

    return {
        "match": True,
        "rootCause": root_cause,
        "resolutionSteps": steps,
        "groundingText": grounding_text,
        "groundingSource": {"title": "AI Core Orchestration grounding",
                            "filePath": "", "webUrl": ""},
    }


def _select_grounding(body: dict, min_score: float) -> dict[str, Any]:
    """Pick the best-matching catalog document from a grounding response. Pure.

    Vector retrieval ALWAYS returns nearest neighbours, even for a query with no
    real catalog entry — so "a document came back" is NOT a match. We score each
    document by its highest chunk similarity (searchScores.aggregatedScore.value),
    take the top document, and only treat it as a match when that best score
    clears `min_score`. groundingText is built from that ONE document's chunks,
    ordered best-first — never a concatenation across unrelated documents.

    Returns {match: bool, bestScore: float, groundingText?, groundingSource?}.
    """
    def _chunk_score(ch: dict) -> float:
        try:
            v = ch["searchScores"]["aggregatedScore"]["value"]
            return float(v)
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _meta(items: list) -> dict:
        return {m["key"]: m["value"][0] for m in items if m.get("value")}

    try:
        docs = body["results"][0]["results"][0]["dataRepository"]["documents"]
    except (KeyError, IndexError, TypeError):
        docs = []
    if not docs:
        return {"match": False, "bestScore": 0.0}

    # Best document = the one whose strongest chunk scores highest.
    def _doc_best(doc: dict) -> float:
        return max((_chunk_score(c) for c in doc.get("chunks", [])), default=0.0)

    top_doc = max(docs, key=_doc_best)
    best = _doc_best(top_doc)
    if best < min_score:
        return {"match": False, "bestScore": best}

    # Emit only the matched document's chunks, ordered best-first.
    scored = sorted(top_doc.get("chunks", []), key=_chunk_score, reverse=True)
    grounding_text = "\n\n".join(c.get("content", "") for c in scored if c.get("content"))
    meta = _meta(top_doc.get("metadata", []))
    return {
        "match": True,
        "bestScore": best,
        "groundingText": grounding_text,
        "groundingSource": {
            "title": meta.get("title", ""),
            "filePath": meta.get("filePath", ""),
            "webUrl": meta.get("webUrl", ""),
        },
    }


@tool
def run_doc_error_catalog_tool(query: str) -> dict[str, Any]:
    """Retrieve root cause and resolution steps for an AIF error from the catalog.

    Two backends, selected by the GROUNDING_MODE env var:
      - 'retrieval' (default): document-grounding retrieval/search with a
        relevance score gate; returns groundingText from the best matching doc.
      - 'orchestration': AI Core Orchestration /completion (retrieve -> LLM)
        returning a grounded {rootCause, resolutionSteps[]} answer.
    Either way the result is {match, rootCause?/groundingText?, ...} for the
    analysis flow to build a resolution entry from.

    Args:
        query: Error query string in the form 'MSGID/MSGNO MSGTX'.
    """
    with tracer.start_as_current_span("run_doc_error_catalog_tool", attributes={"grounding.query": query}) as span:
        try:
            aicore_url = os.environ.get("AICORE_BASE_URL", "")
            if not aicore_url:
                span.set_attribute("outcome", "misconfigured")
                return {"error": "AICORE_BASE_URL not configured", "match": False}

            auth_url = os.environ.get("AICORE_AUTH_URL", "").rstrip("/") + "/oauth/token"
            token_resp = httpx.post(
                auth_url,
                data={"grant_type": "client_credentials"},
                auth=(os.environ.get("AICORE_CLIENT_ID", ""), os.environ.get("AICORE_CLIENT_SECRET", "")),
                timeout=10,
            )
            token_resp.raise_for_status()
            access_token = token_resp.json()["access_token"]
            span.set_attribute("grounding.mode", GROUNDING_MODE)

            # --- Orchestration mode: retrieve -> template -> LLM in one call ---
            # Returns a grounded {root_cause, resolution_step[]} JSON. No chunk
            # score is available, so relevance relies on the prompt's
            # resolvable_from_references self-assessment (parsed in
            # _parse_orchestration_grounding).
            if GROUNDING_MODE == "orchestration":
                if not GROUNDING_ORCH_URL:
                    span.set_attribute("outcome", "misconfigured")
                    return {"error": "AICORE_ORCH_COMPLETION_URL not configured", "match": False}
                err_id, err_no, err_msg = _split_error_query(query)
                orch_payload = _orchestration_payload(err_id, err_no, err_msg)
                with httpx.Client(timeout=60) as client:
                    resp = client.post(
                        GROUNDING_ORCH_URL,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                            "AI-Resource-Group": "resource",
                            "Authorization": f"Bearer {access_token}",
                        },
                        json=orch_payload,
                    )
                    resp.raise_for_status()
                    parsed = _parse_orchestration_grounding(resp.json())
                span.set_attribute("outcome", "success" if parsed.get("match") else "no_match")
                return parsed

            payload = {
                "query": query,
                "filters": [{
                    "id": "flt1",
                    "searchConfiguration": {"maxChunkCount": 20},
                    "dataRepositories": [AICORE_GROUNDING_REPOSITORY],
                    "dataRepositoryType": "vector",
                    "dataRepositoryMetadata": [],
                    "documentMetadata": [],
                    "chunkMetadata": [],
                }],
            }
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{aicore_url.rstrip('/')}/lm/document-grounding/retrieval/search",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "AI-Resource-Group": "resource",
                        "Authorization": f"Bearer {access_token}",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()

            selected = _select_grounding(body, AICORE_GROUNDING_MIN_SCORE)
            span.set_attribute("grounding.best_score", selected.get("bestScore", 0.0))
            span.set_attribute("grounding.min_score", AICORE_GROUNDING_MIN_SCORE)
            if not selected.get("match"):
                # A weak nearest-neighbour is NOT a match — returning it would
                # present an unrelated error's resolution as the answer.
                span.set_attribute("outcome", "no_match")
                return {"match": False, "bestScore": selected.get("bestScore", 0.0)}
            span.set_attribute("outcome", "success")
            span.set_attribute("grounding.document_title", selected["groundingSource"]["title"])
            return selected
        except Exception as exc:
            logger.exception("run_doc_error_catalog_tool failed")
            span.set_attribute("outcome", "error")
            span.set_attribute("error.message", str(exc))
            return {"error": str(exc), "match": False}


@tool
def get_message_details_tool(msgguid: str) -> dict[str, Any]:
    """Fetch full details of a specific AIF message by its MSGGUID.

    Args:
        msgguid: The AIF message GUID, e.g. '000000016071DFD7AD912FEE8284FD2D'.
    """
    with tracer.start_as_current_span("get_message_details_tool", attributes={"aif.msgguid": msgguid}) as span:
        try:
            dest_url, auth_headers = _get_destination_url(DEST_AIF_ALIAS)
            if not dest_url:
                span.set_attribute("outcome", "misconfigured")
                return {"error": f"Destination '{DEST_AIF_ALIAS}' not resolved", "message": None}

            oauth_url = os.environ.get("AIF_OAUTH_URL", "")
            oauth_client_id = os.environ.get("AIF_OAUTH_CLIENT_ID", "")
            oauth_client_secret = os.environ.get("AIF_OAUTH_CLIENT_SECRET", "")
            if oauth_url and oauth_client_id and oauth_client_secret:
                token_resp = httpx.post(
                    oauth_url,
                    data={"grant_type": "client_credentials"},
                    auth=(oauth_client_id, oauth_client_secret),
                    timeout=10,
                )
                token_resp.raise_for_status()
                auth_headers["Authorization"] = f"Bearer {token_resp.json()['access_token']}"

            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{dest_url}/odata/v2/aifmonitoring/CentralFinanceMessages",
                    params={"$filter": f"MSGGUID eq '{msgguid}'", "$format": "json"},
                    headers={"Accept": "application/json", **auth_headers},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("d", {}).get("results", [])
                if not results:
                    span.set_attribute("outcome", "not_found")
                    return {"message": None, "error": f"No message found for MSGGUID '{msgguid}'"}
                span.set_attribute("outcome", "success")
                return {"message": results[0]}
        except Exception as exc:
            logger.exception("get_message_details_tool failed")
            span.set_attribute("outcome", "error")
            span.set_attribute("error.message", str(exc))
            return {"error": str(exc), "message": None}


@tool
def get_interface_details_tool(namespace: str, ifname: str, ifversion: str) -> dict[str, Any]:
    """Return a structured description of a SAP AIF interface by its namespace, name, and version.

    Args:
        namespace: Interface namespace, e.g. '/FINCF'.
        ifname: Interface name, e.g. 'AC_DOC'.
        ifversion: Interface version, e.g. '0001'.
    """
    with tracer.start_as_current_span(
        "get_interface_details_tool",
        attributes={"aif.namespace": namespace, "aif.ifname": ifname, "aif.ifversion": ifversion},
    ) as span:
        span.set_attribute("outcome", "success")
        return {
            "namespace": namespace,
            "ifname": ifname,
            "ifversion": ifversion,
            "source": "llm_knowledge",
        }


def _iso_z(dt) -> str:
    """Format a datetime as a V4 DateTimeOffset literal (second precision, Z)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_date_range(period: str) -> dict[str, Any]:
    """Resolve a free-text period to {from, to, resolved}. Pure function (testable).

    Supported cases (case-insensitive):
      - last N hours/days/weeks/months/years  ('last 24 hours', 'last 7 days')
      - last hour/day/week/month/year         ('last week', 'last month', 'last year')
      - today / yesterday
      - this week / this month / this year
      - between X and Y  |  X to Y  |  X..Y  |  X - Y      (explicit range)
      - from X [only]    -> end auto-populated to now
      - absolute year '2025' | month '2025-03' | date '2025-03-15'
    'to' bounds always end at 23:59:59 of their period; relative windows end at now.
    """
    import re
    from datetime import datetime, timedelta, timezone

    raw = period.strip()
    p = raw.lower()
    now = datetime.now(timezone.utc).replace(microsecond=0)

    def day_start(d):
        return d.replace(hour=0, minute=0, second=0)

    def day_end(d):
        return d.replace(hour=23, minute=59, second=59)

    def endpoint(text, is_end):
        """Resolve a single range endpoint to a V4 literal, honouring the
        relative keywords today/now/yesterday. Anything else falls back to
        _v4_dt (years/months/dates). This stops a literal 'today' from being
        mangled into 'today-12-31T23:59:59Z' by _v4_dt's year-padding."""
        t = text.strip().lower()
        if t in ("today",):
            return _iso_z(day_end(now) if is_end else day_start(now))
        if t in ("now",):
            return _iso_z(now)
        if t in ("yesterday",):
            y = now - timedelta(days=1)
            return _iso_z(day_end(y) if is_end else day_start(y))
        return _v4_dt(text.strip(), is_end=is_end)

    # ---- no date restriction (explicit) ----
    if p in ("all", "all time", "any", "anytime", "no date restriction",
             "no restriction", "no date", "everything", "unbounded"):
        # The AIF data tools require a bounded window, so "no restriction" is a
        # very wide window from a far-past sentinel up to now.
        return {"from": _iso_z(day_start(now.replace(year=2000, month=1, day=1))),
                "to": _iso_z(day_end(now)), "resolved": True}

    # ---- today / yesterday ----
    if p == "today":
        return {"from": _iso_z(day_start(now)), "to": _iso_z(day_end(now)), "resolved": True}
    if p == "yesterday":
        y = now - timedelta(days=1)
        return {"from": _iso_z(day_start(y)), "to": _iso_z(day_end(y)), "resolved": True}

    # ---- this week / month / year (period-to-date) ----
    if p in ("this week", "current week"):
        start = day_start(now - timedelta(days=now.weekday()))  # Monday
        return {"from": _iso_z(start), "to": _iso_z(now), "resolved": True}
    if p in ("this month", "current month"):
        return {"from": _iso_z(day_start(now.replace(day=1))), "to": _iso_z(now), "resolved": True}
    if p in ("this year", "current year"):
        return {"from": _iso_z(day_start(now.replace(month=1, day=1))), "to": _iso_z(now), "resolved": True}

    # ---- last N <unit> / last <unit> ----
    m = re.match(r"^(?:last|past|previous)\s+(\d+)?\s*(hour|day|week|month|year)s?$", p)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2)
        if unit == "hour":
            start = now - timedelta(hours=n)
        elif unit == "day":
            start = now - timedelta(days=n)
        elif unit == "week":
            start = now - timedelta(weeks=n)
        elif unit == "month":
            start = now - timedelta(days=30 * n)   # calendar-month approximation
        else:  # year
            start = now - timedelta(days=365 * n)
        return {"from": _iso_z(start), "to": _iso_z(now), "resolved": True}

    # ---- explicit range: 'between X and Y' | 'X to Y' | 'X..Y' | 'X - Y' ----
    range_text = raw
    bm = re.match(r"^between\s+(.+?)\s+and\s+(.+)$", raw, re.IGNORECASE)
    if bm:
        return {"from": endpoint(bm.group(1), is_end=False), "to": endpoint(bm.group(2), is_end=True), "resolved": True}
    for sep in (" to ", "..", " - "):
        if sep in range_text:
            a, b = range_text.split(sep, 1)
            return {"from": endpoint(a, is_end=False), "to": endpoint(b, is_end=True), "resolved": True}

    # ---- from X (only) -> end auto-populated to now ----
    fm = re.match(r"^(?:from|since|after)\s+(.+)$", raw, re.IGNORECASE)
    if fm:
        return {"from": endpoint(fm.group(1), is_end=False), "to": _iso_z(now), "resolved": True}

    # ---- absolute year / month / date (bare) ----
    token = raw
    if re.fullmatch(r"\d{4}", token):                       # 2025
        return {"from": _v4_dt(token), "to": _v4_dt(token, is_end=True), "resolved": True}
    if re.fullmatch(r"\d{4}-\d{2}", token):                 # 2025-03
        return {"from": _v4_dt(token), "to": _v4_dt(token, is_end=True), "resolved": True}
    if re.match(r"^\d{4}-\d{2}-\d{2}", token):              # 2025-03-15[...]
        return {"from": _v4_dt(token[:10]), "to": _v4_dt(token[:10], is_end=True), "resolved": True}

    return {"from": None, "to": None, "resolved": False, "note": f"Unrecognised period '{period}'."}


# ---------------------------------------------------------------------------
# Full analysis report builder (deterministic, testable)
# ---------------------------------------------------------------------------

# Common Unicode punctuation -> ASCII, applied before the NFKD fallback so we
# get sensible replacements (en/em dash -> '-', smart quotes -> ', curly etc.).
_ASCII_MAP = {
    "—": "-", "–": "-", "‒": "-", "−": "-",   # dashes
    "·": "-", "•": "*", "…": "...",                 # middot, bullet, ellipsis
    "‘": "'", "’": "'", "“": '"', "”": '"',    # smart quotes
    "→": "->", "←": "<-", "↑": "^", "↓": "v",  # arrows
    " ": " ", "️": "",                                   # nbsp, variation selector
}


def _ascii(text: str) -> str:
    """Force a string to plain ASCII for SAP Joule's markdown renderer.

    Joule errors on non-ASCII (em dash, middle dot, arrows, emoji). We map the
    common punctuation explicitly, transliterate accented letters via NFKD
    (e.g. 'Période' -> 'Periode'), then drop anything still non-ASCII (e.g.
    emoji). Applied to the whole report so verbatim grounding/interface text
    can't reintroduce non-ASCII.
    """
    import unicodedata

    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    # NFKD splits accented chars into base + combining mark; encode/ignore drops
    # the marks and any remaining non-ASCII (emoji, CJK, etc.).
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


_TOP_N = 20          # top errors listed / resolved in the report
_MAX_BULLETS = 3     # max resolution bullets per error (short, for readability)


def _build_analysis_report(
    period: str,
    date_from: str,
    date_to: str,
    statistics: list[dict],
    error_rows: list[dict],
    resolutions: list[dict],
    grounded_total: int | None = None,
    interfaces_catalog: list[dict] | None = None,
) -> dict[str, Any]:
    """Assemble the full AIF monitoring report as Markdown. Pure function.

    All metrics are computed here so they are exact and consistent. Output is
    plain ASCII Markdown tables/text only — no charts, no emoji — so SAP Joule
    renders it cleanly. Grounding text from `resolutions` is placed VERBATIM.

    Args:
        period:        the user's free-text period (echoed in the header).
        date_from/to:  resolved V4 bounds (echoed in the header).
        statistics:    interfaces[] from get_interface_statistics_tool — drives
                       all counts (all message types) and per-interface health.
        error_rows:    messages[] from list_error_messages_tool. Accepted for the
                       analysis contract; not rendered directly.
        resolutions:   grounded error entries (top distinct, up to 20), each:
                       {msgId, msgNo, messageText, affectedInterfaces[],
                        occurrences, restartSafe, rootCause, resolutionSteps[],
                        sourceText}.
        grounded_total: distinct error count that existed (for the truncation
                        note). Defaults to len(resolutions).
        interfaces_catalog: interfaces[] from list_interfaces_tool — supplies the
                       business-friendly description (about/searchBy) joined on
                       namespace+name+version for the Active Interfaces table.
    """
    stats = statistics or []
    res = resolutions or []
    catalog = interfaces_catalog or []

    # description lookup keyed on (namespace, name, version)
    desc_by_key: dict[tuple, str] = {}
    for c in catalog:
        key = (c.get("namespace", ""), c.get("interfaceName", ""), c.get("interfaceVersion", ""))
        desc_by_key[key] = c.get("about") or c.get("searchBy") or ""

    def _iface_desc(s: dict) -> str:
        key = (s.get("namespace", ""), s.get("interfaceName", ""), s.get("interfaceVersion", ""))
        return desc_by_key.get(key) or s.get("interfaceName", "") or "-"

    def _id(r: dict) -> str:
        """msgId/msgNo, defensively avoiding the bare '-/-' for empty values."""
        mid = (r.get("msgId") or "").strip()
        mno = (r.get("msgNo") or "").strip()
        if mid and mno:
            return f"{mid}/{mno}"
        return mid or mno or "-"
    def _sum(field: str) -> int:
        return sum(int(s.get(field, 0) or 0) for s in stats)

    total = _sum("total")
    total_errors = _sum("errors")
    total_warnings = _sum("warnings")
    total_success = _sum("success")
    total_inprocess = _sum("inProcess")
    total_aborted = _sum("aborted")
    total_canceled = _sum("canceled")
    # Any status the endpoint didn't break out (keeps the breakdown summing to total).
    total_other = max(0, total - total_errors - total_warnings - total_success
                      - total_inprocess - total_aborted - total_canceled)

    def pct(n: int) -> float:
        return (100.0 * n / total) if total else 0.0

    err_pct = pct(total_errors)
    success_rate = pct(total_success)

    def grade(pct: float) -> str:
        if pct < 1:
            return "A"
        if pct < 3:
            return "B"
        if pct < 7:
            return "C"
        if pct < 15:
            return "D"
        return "F"

    active = [s for s in stats if int(s.get("total", 0) or 0) > 0]
    has_errors = total_errors > 0 and bool(res)

    DASH = "-"

    def criticality(occ: int) -> str:
        # Same thresholds as interface health: 6+ Critical, 1-5 Warning.
        return "Critical" if occ >= 6 else ("Warning" if occ >= 1 else "Low")

    # Most-common errors first (by occurrence count), then by criticality, capped
    # to the top N. This single ranking drives both the list and the resolutions.
    ranked = sorted(
        res,
        key=lambda r: (int(r.get("occurrences", 0) or 0),
                       criticality(int(r.get("occurrences", 0) or 0)) == "Critical"),
        reverse=True,
    )[:_TOP_N]

    # --- assemble Markdown ----------------------------------------------------
    # PURE ASCII ONLY. SAP Joule's renderer errors on non-ASCII (em dash, middle
    # dot, arrows, emoji), so use '-' placeholders, ASCII separators, word labels.
    out: list[str] = []
    out.append("# AIF Interface Monitoring Report")
    out.append(f"**Period:** {period} | {date_from[:10]} to {date_to[:10]}\n")

    if has_errors and ranked:
        top = ranked[0]
        out.append(
            f"> **Biggest risk:** {_id(top)} "
            f"\"{top.get('messageText') or DASH}\" - {top.get('occurrences', 0)} occurrences "
            f"across {len(top.get('affectedInterfaces', []) or [])} interface(s).\n"
            f"> **Do this first:** "
            f"{(top.get('resolutionSteps') or ['Review the resolution below.'])[0]}\n"
        )

    # Scorecard — calculation covers ALL message types (success / warning /
    # error / other), not just errors, from the per-interface statistics.
    out.append("## Health Scorecard")
    out.append(
        f"**Overall grade: {grade(err_pct)}** | "
        f"**Success rate: {success_rate:.1f}%** | {total} messages\n"
    )
    out.append("**Message breakdown (all types):**")
    # Itemise every status; skip the ones that are zero to keep it readable.
    for label, count in (
        ("success", total_success),
        ("warning", total_warnings),
        ("error", total_errors),
        ("in-process", total_inprocess),
        ("aborted", total_aborted),
        ("canceled", total_canceled),
        ("other", total_other),
    ):
        if count:
            out.append(f"- {count} {label} ({pct(count):.1f}%)")
    out.append("")

    # Active interfaces — business-friendly: show the interface DESCRIPTION (not
    # namespace/version) and totals across ALL message types.
    out.append("## Active Interfaces (traffic in period)")
    out.append("| Interface | Description | Total | Errors | Warnings | Success | In-Process | Health |")
    out.append("|-----------|-------------|-------|--------|----------|---------|------------|--------|")
    for s in sorted(active, key=lambda s: int(s.get("errors", 0) or 0), reverse=True):
        e = int(s.get("errors", 0) or 0)
        tot = int(s.get("total", 0) or 0)
        out.append(
            f"| {s.get('interfaceName', DASH)} | {_iface_desc(s)} | "
            f"{tot} | {e} | {s.get('warnings', 0)} | {s.get('success', 0)} | "
            f"{s.get('inProcess', 0)} | {s.get('health', DASH)} |"
        )
    if not active:
        out.append("| - | - | 0 | 0 | 0 | 0 | 0 | Healthy |")
    out.append("")

    if has_errors and ranked:
        # Top N most common errors, highest occurrence first, with criticality.
        out.append(f"## Top {len(ranked)} Most Common Errors")
        out.append("| # | Message | Occurrences | Cumulative |")
        out.append("|---|---------|-------------|------------|")
        cumulative = 0
        for i, r in enumerate(ranked, 1):
            occ = int(r.get("occurrences", 0) or 0)
            cumulative += occ
            out.append(f"| {i} | {r.get('messageText') or DASH} | {occ} | {cumulative} |")
        out.append("")

        # Top N resolutions — SHORT: root cause + max 3 bullets, verbatim grounding.
        out.append(f"## Error Resolutions (top {len(ranked)})")
        for r in ranked:
            out.append(f"### {_id(r)} - \"{r.get('messageText') or DASH}\"")
            out.append(
                f"**Root Cause:** {r.get('rootCause') or 'No resolution guidance available in the document catalog for this error.'}"
            )
            steps = (r.get("resolutionSteps") or [])[:_MAX_BULLETS]
            for j, step in enumerate(steps, 1):
                out.append(f"{j}. {step}")
            if r.get("sourceText"):
                out.append(f"**Source:** {r['sourceText']}")
            out.append("")

        gt = grounded_total if grounded_total is not None else len(res)
        if gt > len(ranked):
            out.append(f"_+{gt - len(ranked)} more distinct error(s) not shown (top {len(ranked)} by frequency)._")
    else:
        out.append("> **Healthy** - no errors in this period. Nothing to resolve.")

    return {"report": _ascii("\n".join(out).strip() + "\n")}


@tool
def build_analysis_report_tool(
    period: str,
    date_from: str,
    date_to: str,
    statistics: list[dict],
    error_rows: list[dict],
    resolutions: list[dict],
    grounded_total: int = -1,
    interfaces_catalog: list[dict] | None = None,
) -> dict[str, Any]:
    """Assemble the FULL AIF monitoring report (Markdown) for the analysis intent.

    Call this LAST in the analysis flow, after gathering statistics + error rows
    and grounding the top distinct errors. Return the `report` value VERBATIM as
    the answer. All metrics are computed here — do not recompute or alter them.

    Args:
        period: the user's period text, e.g. '2023 to today'.
        date_from: resolved V4 start, e.g. '2023-01-01T00:00:00Z'.
        date_to: resolved V4 end, e.g. '2026-06-16T23:59:59Z'.
        statistics: the interfaces[] list from get_interface_statistics_tool
            (drives all-message-type counts and per-interface health).
        error_rows: the messages[] list from list_error_messages_tool (status='E').
        resolutions: one entry per distinct top error (up to 20, ranked by
            occurrence), each with keys msgId, msgNo, messageText,
            affectedInterfaces (list), occurrences (int), restartSafe (bool),
            rootCause (verbatim from grounding), resolutionSteps (list, verbatim;
            only the first 3 are shown), sourceText. Use '-'/[] when unknown and
            'No resolution guidance available...' for rootCause if grounding missed.
        grounded_total: total number of DISTINCT errors that existed in the period
            (so the report can note how many were truncated). Pass -1 to default to
            len(resolutions).
        interfaces_catalog: the interfaces[] list from list_interfaces_tool, used to
            show a business-friendly description per interface (joined on
            namespace+name+version). Pass [] if not fetched.
    """
    with tracer.start_as_current_span("build_analysis_report_tool") as span:
        gt = None if grounded_total is None or grounded_total < 0 else grounded_total
        span.set_attribute("aif.interface_count", len(statistics or []))
        span.set_attribute("aif.resolution_count", len(resolutions or []))
        return _build_analysis_report(
            period, date_from, date_to, statistics, error_rows, resolutions, gt,
            interfaces_catalog=interfaces_catalog or [],
        )


@tool
def calculate_date_range_tool(period: str) -> dict[str, Any]:
    """Resolve a free-text period into OData V4 from/to DateTimeOffset bounds.

    Handles relative periods (last week/month/year, last N hours/days, today,
    yesterday, this week/month/year), explicit ranges ('between X and Y',
    'X to Y', 'X..Y') where today/now/yesterday on either end resolve to the
    real current date, open-ended 'from X' (end auto-set to today), absolute
    year/month/date, and 'all'/'any' (unrestricted: far past -> today). 'to'
    always ends at 23:59:59 of its period (or now for relative windows).

    Pass the user's phrasing through; do not substitute dates yourself. For an
    open-ended start use 'from <start>' (not the bare year). When the request
    has NO time bound, pass 'all' rather than inventing a window.

    Args:
        period: e.g. 'last week', 'last 24 hours', 'between 2025-01-01 and 2025-06-30',
                '2023 to today', 'from 2025-03-01', 'all', '2025', '2025-03', '2025-03-15'.
    """
    return _resolve_date_range(period)


@tool
def list_interfaces_tool() -> dict[str, Any]:
    """List the AIF interfaces available for monitoring (namespace, name, version).

    Also returns the searchable key-field LABELS per interface so the UI can show
    what business objects each interface can be searched by.
    """
    with tracer.start_as_current_span("list_interfaces_tool") as span:
        ifaces, err = _aif_get("IfKeySet")
        if err:
            span.set_attribute("outcome", "error")
            return {"error": err, "interfaces": [], "count": 0}
        # Key-field labels per interface, for searchBy.
        keyfields, _ = _aif_get("KeyFieldsSet")
        labels: dict[tuple, list[str]] = {}
        for kf in keyfields:
            key = (kf.get("Namespace"), kf.get("InterfaceName"), kf.get("InterfaceVersion"))
            label = kf.get("Label") or kf.get("FieldName")
            if label:
                labels.setdefault(key, []).append(label)
        out = []
        for it in ifaces:
            key = (it.get("Namespace"), it.get("InterfaceName"), it.get("InterfaceVersion"))
            out.append({
                "namespace": it.get("Namespace", ""),
                "interfaceName": it.get("InterfaceName", ""),
                "interfaceVersion": it.get("InterfaceVersion", ""),
                "searchBy": ", ".join(labels.get(key, [])) or "—",
            })
        span.set_attribute("outcome", "success")
        span.set_attribute("aif.interface_count", len(out))
        return {"interfaces": out, "count": len(out)}


@tool
def list_error_messages_tool(
    date_from: str,
    date_to: str,
    namespace: str = "",
    interface_name: str = "",
    interface_version: str = "",
    status: str = "E",
) -> dict[str, Any]:
    """List AIF messages (worklist) for a date window, optionally filtered by interface.

    Args:
        date_from: V4 DateTimeOffset start, e.g. '2025-01-01T00:00:00Z'.
        date_to: V4 DateTimeOffset end, e.g. '2025-12-31T23:59:59Z'.
        namespace: Optional interface namespace filter, e.g. 'NS1'.
        interface_name: Optional interface name filter, e.g. 'ORDERS'.
        interface_version: Optional interface version filter, e.g. '1'.
        status: AIF status to filter on (default 'E' = errors). Pass '' for all.
    """
    with tracer.start_as_current_span("list_error_messages_tool") as span:
        clauses = [
            f"ProcessDate ge {_v4_dt(date_from)}",
            f"ProcessDate le {_v4_dt(date_to, is_end=True)}",
        ]
        if status:
            clauses.append(f"Status eq '{status}'")
        if namespace:
            clauses.append(f"Namespace eq '{namespace}'")
        if interface_name:
            clauses.append(f"InterfaceName eq '{interface_name}'")
        if interface_version:
            clauses.append(f"InterfaceVersion eq '{interface_version}'")
        params = {"$filter": " and ".join(clauses), "$top": "100", "$orderby": "ProcessDate desc"}
        rows, err = _aif_get("IndexTableGenericSet", params)
        if err:
            span.set_attribute("outcome", "error")
            return {"error": err, "messages": [], "count": 0}
        msgs = [{
            "messageGuid": r.get("MessageGuid", ""),
            "namespace": r.get("Namespace", ""),
            "interfaceName": r.get("InterfaceName", ""),
            "interfaceVersion": r.get("InterfaceVersion", ""),
            "status": r.get("Status", ""),
            "processDate": r.get("ProcessDate", ""),
            "logMessage": r.get("LogMessage", "") or "",
        } for r in rows]
        span.set_attribute("outcome", "success")
        span.set_attribute("aif.record_count", len(msgs))
        return {"messages": msgs, "count": len(msgs)}


@tool
def get_interface_statistics_tool(date_from: str, date_to: str) -> dict[str, Any]:
    """Per-interface statistics and health for a date window (InterfaceStatistics).

    Args:
        date_from: V4 DateTimeOffset start, e.g. '2025-01-01T00:00:00Z'.
        date_to: V4 DateTimeOffset end, e.g. '2025-12-31T23:59:59Z'.
    """
    with tracer.start_as_current_span("get_interface_statistics_tool") as span:
        path = f"InterfaceStatistics(p_datetime_from={_v4_dt(date_from)},p_datetime_to={_v4_dt(date_to, is_end=True)})"
        rows, err = _aif_get(path)
        if err:
            span.set_attribute("outcome", "error")
            return {"error": err, "interfaces": [], "interfaceCount": 0}
        # The parameterized entity returns a single object with a `Set` array.
        body = rows[0] if rows else {}
        stat_rows = body.get("Set", []) if isinstance(body, dict) else []
        out = []
        for s in stat_rows:
            errs = int(s.get("NumberErrorMessages", 0) or 0)
            health = "Critical" if errs >= 6 else ("Warning" if errs >= 1 else "Healthy")
            out.append({
                "namespace": s.get("Namespace", ""),
                "interfaceName": s.get("InterfaceName", ""),
                "interfaceVersion": s.get("InterfaceVersion", ""),
                "total": int(s.get("NumberAllMessages", 0) or 0),
                "errors": errs,
                "warnings": int(s.get("NumberWarningMessages", 0) or 0),
                "success": int(s.get("NumberSuccessMessages", 0) or 0),
                # All remaining AIF statuses, so the report can show every type.
                "inProcess": int(s.get("NumberInProcessMessages", 0) or 0),
                "aborted": int(s.get("NumberAbortMessages", 0) or 0),
                "canceled": int(s.get("NumberCanceledMessages", 0) or 0),
                "health": health,
            })
        span.set_attribute("outcome", "success")
        span.set_attribute("aif.interface_count", len(out))
        return {"interfaces": out, "interfaceCount": len(out)}


@tool
def get_key_fields_tool(namespace: str = "", interface_name: str = "", interface_version: str = "") -> dict[str, Any]:
    """Return AIF key-field definitions (FieldName, Label, SemObj) per interface.

    Use to map a business object term (e.g. 'Customer') to its key field(s), and to
    show what each interface can be searched by. Optionally filter to one interface.

    Args:
        namespace: Optional namespace filter.
        interface_name: Optional interface name filter.
        interface_version: Optional interface version filter.
    """
    with tracer.start_as_current_span("get_key_fields_tool") as span:
        clauses = []
        if namespace:
            clauses.append(f"Namespace eq '{namespace}'")
        if interface_name:
            clauses.append(f"InterfaceName eq '{interface_name}'")
        if interface_version:
            clauses.append(f"InterfaceVersion eq '{interface_version}'")
        params = {"$filter": " and ".join(clauses)} if clauses else None
        rows, err = _aif_get("KeyFieldsSet", params)
        if err:
            span.set_attribute("outcome", "error")
            return {"error": err, "keyFields": [], "count": 0}
        kf = [{
            "namespace": r.get("Namespace", ""),
            "interfaceName": r.get("InterfaceName", ""),
            "interfaceVersion": r.get("InterfaceVersion", ""),
            "fieldName": r.get("FieldName", ""),
            "label": r.get("Label", "") or r.get("FieldName", ""),
            "semObj": r.get("SemObj", ""),
        } for r in rows]
        span.set_attribute("outcome", "success")
        return {"keyFields": kf, "count": len(kf)}


@tool
def find_messages_by_key_value_tool(field_name: str, field_value: str) -> dict[str, Any]:
    """Find AIF messages whose business-key field matches a value (KeyFieldValueSet).

    ALWAYS filters on BOTH FieldName AND FieldValue. Use after get_key_fields_tool
    has mapped a business object (e.g. 'Customer' -> 'KUNNR') to its field name.

    Args:
        field_name: The key field name, e.g. 'KUNNR'.
        field_value: The value to match, e.g. 'C-0000001'.
    """
    with tracer.start_as_current_span("find_messages_by_key_value_tool") as span:
        params = {"$filter": f"FieldName eq '{field_name}' and FieldValue eq '{field_value}'", "$top": "100"}
        rows, err = _aif_get("KeyFieldValueSet", params)
        if err:
            span.set_attribute("outcome", "error")
            return {"error": err, "messages": [], "count": 0}
        msgs = [{
            "messageGuid": r.get("MessageGuid", ""),
            "namespace": r.get("Namespace", ""),
            "interfaceName": r.get("InterfaceName", ""),
            "interfaceVersion": r.get("InterfaceVersion", ""),
            "fieldName": r.get("FieldName", ""),
            "fieldValue": r.get("FieldValue", ""),
        } for r in rows]
        span.set_attribute("outcome", "success")
        span.set_attribute("aif.record_count", len(msgs))
        return {"messages": msgs, "count": len(msgs)}


@tool
def get_message_log_tool(msgguid: str) -> dict[str, Any]:
    """Read the ERROR log entries (MsgType='E') of one AIF message by GUID (MessageLogSet).

    A message can have several errors. Use for message_detail and as the basis for
    resolution. Returns the interface coordinates and each error's MsgId/MsgNo/Text.

    Args:
        msgguid: The 32-char AIF message GUID.
    """
    with tracer.start_as_current_span("get_message_log_tool", attributes={"aif.msgguid": msgguid}) as span:
        params = {"$filter": f"MessageGuid eq '{msgguid}' and MsgType eq 'E'"}
        rows, err = _aif_get("MessageLogSet", params)
        if err:
            span.set_attribute("outcome", "error")
            return {"error": err, "logEntries": [], "logCount": 0, "interface": ""}
        interface = ""
        if rows:
            r0 = rows[0]
            interface = f"{r0.get('Namespace','')} / {r0.get('InterfaceName','')} v{r0.get('InterfaceVersion','')}"
        entries = [{
            "msgType": r.get("MsgType", ""),
            "msgId": r.get("MsgId", ""),
            "msgNo": r.get("MsgNo", ""),
            "text": r.get("Text", "") or "",
        } for r in rows]
        span.set_attribute("outcome", "success")
        span.set_attribute("aif.error_count", len(entries))
        return {"messageGuid": msgguid, "interface": interface, "logEntries": entries, "logCount": len(entries)}


@tool
def get_agent_capabilities_tool(agent: str = "self") -> dict[str, Any]:
    """Fetch THIS agent's card (capabilities, skills, example queries).

    Args:
        agent: Only 'self' is supported.
    """
    with tracer.start_as_current_span("get_agent_capabilities_tool", attributes={"agent": agent}) as span:
        base_url = os.environ.get("AGENT_PUBLIC_URL", "http://localhost:9000")
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(base_url.rstrip("/") + "/.well-known/agent-card.json")
                resp.raise_for_status()
                card = resp.json()
            span.set_attribute("outcome", "success")
            return {"agent": "self", "card": card}
        except Exception as exc:
            logger.exception("get_agent_capabilities_tool failed")
            span.set_attribute("outcome", "error")
            span.set_attribute("error.message", str(exc))
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _load_history(memory_client, context_id: str) -> list:
    """Load conversation history from Agent Memory and convert to LangChain messages."""
    if not memory_client:
        return []
    try:
        from sap_cloud_sdk.agent_memory import MessageRole
        messages = memory_client.list_messages(
            agent_id=AGENT_ID,
            invoker_id=context_id,
            message_group=context_id,
            limit=20,
        )
        history = []
        for m in messages:
            if m.role == MessageRole.USER:
                history.append(HumanMessage(content=m.content))
            elif m.role == MessageRole.ASSISTANT:
                history.append(AIMessage(content=m.content))
        return history
    except Exception:
        logger.warning("Failed to load conversation history from memory")
        return []


def _persist_turn(memory_client, context_id: str, query: str, response: str) -> None:
    """Persist user query + agent response as messages and a searchable memory."""
    if not memory_client:
        return
    try:
        from sap_cloud_sdk.agent_memory import MessageRole
        memory_client.add_message(
            agent_id=AGENT_ID,
            invoker_id=context_id,
            message_group=context_id,
            role=MessageRole.USER,
            content=query,
        )
        memory_client.add_message(
            agent_id=AGENT_ID,
            invoker_id=context_id,
            message_group=context_id,
            role=MessageRole.ASSISTANT,
            content=response,
        )
        memory_client.add_memory(
            agent_id=AGENT_ID,
            invoker_id=context_id,
            content=response,
            metadata={"type": "aif_report", "query": query},
        )
    except Exception:
        logger.warning("Failed to persist conversation turn to memory")


def _search_relevant_memories(memory_client, context_id: str, query: str) -> str:
    """Retrieve semantically relevant past memories for this context."""
    if not memory_client:
        return ""
    try:
        results = memory_client.search_memories(
            agent_id=AGENT_ID,
            invoker_id=context_id,
            query=query,
            threshold=0.65,
            limit=3,
        )
        if not results:
            return ""
        lines = [f"- {r.content[:800]}" for r in results]
        return "Relevant context from past conversations:\n" + "\n".join(lines)
    except Exception:
        logger.warning("Memory search failed — continuing without context")
        return ""


def _assemble_history(memory_client, context_id: str, a2a_history, query: str) -> list:
    """Build the prior-turns context for a request, scoped to context_id.

    ALWAYS does two lookups (the previous code only vector-searched as a
    fallback, so semantic recall never fired in multi-turn sessions):
      1. message history for this context_id (persistent memory, else the
         transient A2A task history),
      2. a vector/semantic search of this context's past memories for content
         relevant to the current query.
    Both are returned together (vector hits as a leading SystemMessage), so the
    model can always check historical data per context.
    """
    history = _load_history(memory_client, context_id)

    # No persistent memory? Use this context's in-process short-term history.
    if not history:
        history = _short_term_history(context_id)

    # Last resort: the transient A2A task history (single task).
    if not history and a2a_history:
        try:
            from a2a.types.a2a_pb2 import Role
            for entry in a2a_history:
                if entry["role"] == Role.ROLE_USER:
                    history.append(HumanMessage(content=entry["text"]))
                elif entry["role"] == Role.ROLE_AGENT:
                    history.append(AIMessage(content=entry["text"]))
        except Exception:
            logger.warning("Failed to map A2A history")

    # ALWAYS vector-search this context's memories for the current query.
    memory_context = _search_relevant_memories(memory_client, context_id, query)

    assembled: list = []
    if memory_context:
        assembled.append(SystemMessage(content=memory_context))
    assembled.extend(history)
    return assembled


# ---------------------------------------------------------------------------
# Agent graph
# ---------------------------------------------------------------------------

TOOLS = [
    # run_analysis_tool and get_message_details_tool are DEPRECATED (V2
    # aifmonitoring/* endpoints that 400 against the AIF_SRV service) — replaced by
    # get_interface_statistics_tool / list_error_messages_tool and get_message_log_tool.
    run_doc_error_catalog_tool,
    get_interface_details_tool,
    get_agent_capabilities_tool,
    # AIF_SRV monitoring tools (interface_list, worklist, statistics, business_key,
    # message_detail, resolution support, date-range resolution).
    calculate_date_range_tool,
    list_interfaces_tool,
    list_error_messages_tool,
    get_interface_statistics_tool,
    get_key_fields_tool,
    find_messages_by_key_value_tool,
    get_message_log_tool,
    build_analysis_report_tool,
]


@dataclass
class AgentResponse:
    status: Literal["input_required", "completed", "error"]
    message: str


class CodemineAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, memory_client=None) -> None:
        self.llm = ChatLiteLLM(model=LLM_MODEL).bind_tools(TOOLS)
        # Finalizer: a separate LLM in JSON mode. AI Core's structured-output
        # (response_format=json_schema) runs in OpenAI STRICT mode, which demands
        # every property be in `required` + additionalProperties:false — Pydantic
        # Optionals violate that (400 "Missing 'count'"). JSON mode just guarantees
        # a JSON object; the shape is driven by FINALIZER_PROMPT and validated by
        # CardEnvelope after parsing.
        self.finalizer = ChatLiteLLM(model=LLM_MODEL, model_kwargs={"response_format": {"type": "json_object"}})
        self.graph = self._build_graph()
        self.memory = memory_client

    async def _finalize_structured(self, messages: list, draft: str) -> Optional[dict]:
        """Run the structured-output finalizer over the completed transcript.

        `messages` is the full graph message list (system + history + tool calls +
        tool results + final AI draft). Returns the validated { message, intent,
        data } dict, or None if the LLM could not produce a valid object.
        """
        # Re-use the transcript so the finalizer sees the tool results verbatim,
        # then ask it to emit the structured object.
        transcript = [SystemMessage(content=FINALIZER_PROMPT)] + messages[1:] + [
            HumanMessage(
                content=(
                    "Produce the single structured response object now, grounded only "
                    "in the tool results above. Assistant draft answer:\n\n" + draft
                )
            )
        ]
        try:
            result = await self.finalizer.ainvoke(transcript)
            raw = result.content if hasattr(result, "content") else str(result)
            parsed = json.loads(raw)
            # The analysis report is assembled VERBATIM by build_analysis_report_tool
            # and is already the assistant draft. Joule renders it from
            # data.message (parts[1].data.message), so put the draft there
            # unchanged — never let the finalizer LLM re-summarise or relocate the
            # long Markdown report (it would land in the wrong field / be truncated).
            if isinstance(parsed, dict) and parsed.get("intent") == "analysis":
                parsed["message"] = draft
                parsed["data"] = None
            # SAP Joule's markdown renderer errors on non-ASCII (em dash, middle
            # dot, arrows, emoji). The finalizer LLM tends to emit those in the
            # message fallback, so force the rendered field to plain ASCII for
            # EVERY intent. (Analysis is already ASCII from the builder.)
            if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
                parsed["message"] = _ascii(parsed["message"])
            # Validate/normalise through CardEnvelope so the shape is consistent
            # (fills defaults, coerces types). Falls back to the raw dict if the
            # envelope can't validate but the JSON is still usable.
            try:
                return CardResponse(**parsed).model_dump()
            except Exception:
                logger.warning("Finalizer JSON didn't match CardEnvelope; passing through raw")
                return parsed if isinstance(parsed, dict) else None
        except Exception:
            logger.exception("Structured finalizer failed — falling back to markdown only")
        return None

    def _build_graph(self):
        tool_node = ToolNode(TOOLS)

        async def call_model(state: MessagesState):
            response = await self.llm.ainvoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "__end__"

        builder = StateGraph(MessagesState)
        builder.add_node("model", call_model)
        builder.add_node("tools", tool_node)
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", should_continue)
        builder.add_edge("tools", "model")
        return builder.compile()

    async def stream(
        self, query: str, context_id: str, a2a_history: list | None = None, parent_context=None, user_id: str = ""
    ) -> AsyncGenerator[dict, None]:
        span_attrs: dict = {"a2a.context_id": context_id, "gen_ai.request.model": LLM_MODEL}
        if user_id:
            span_attrs["user.id"] = user_id

        # Attach the incoming A2A trace context so our spans are children of the caller's trace.
        token = otelcontext.attach(parent_context) if parent_context is not None else None

        # Use explicit start/end instead of `with` — async generators receiving
        # GeneratorExit cause the `with` block's __exit__ to run in a different
        # async context, which raises "Token was created in a different Context".
        # invoke_agent_span propagates user.id to all child LLM/tool spans.
        span_cm = invoke_agent_span(
            provider="sap-aicore",
            agent_name=AGENT_ID,
            kind=trace.SpanKind.INTERNAL,
            conversation_id=context_id,
            attributes=span_attrs,
            propagate=bool(user_id),
        )
        span = span_cm.__enter__()

        try:
            yield {"is_task_complete": False, "require_user_input": False, "content": "Processing your request..."}

            try:
                # Stable session key: user_id from JWT when auth is on, else "default".
                session_key = user_id if user_id else "default"
                _current_session_key.set(session_key)

                # Always check per-context_id historical data: message history
                # AND a semantic/vector lookup of this context's past memories.
                history = _assemble_history(self.memory, context_id, a2a_history, query)

                # Inject the previous turn's GUID-addressable findings (per context_id)
                # as an assistant message, so "fix that error" can resolve the GUID
                # even though it lives in the data part, not the visible text.
                prior_findings = _context_findings.get(context_id, "")
                if prior_findings:
                    history = history + [AIMessage(content=prior_findings)]

                messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=query)]
                result = await self.graph.ainvoke({"messages": messages})
                last = result["messages"][-1]
                response = last.content

                if hasattr(last, "usage_metadata") and last.usage_metadata:
                    span.set_attribute("gen_ai.usage.input_tokens", last.usage_metadata.get("input_tokens", 0))
                    span.set_attribute("gen_ai.usage.output_tokens", last.usage_metadata.get("output_tokens", 0))
                    span.set_attribute("gen_ai.usage.total_tokens", last.usage_metadata.get("total_tokens", 0))

                # Completed turn -> package into the Joule card contract
                # so Joule's invoke_agent can route on `intent` and render a card.
                # The text part stays the human-readable answer; the structured
                # { message, intent, data } object rides along as the `data` part.
                _persist_turn(self.memory, context_id, query, response)
                # Always record the turn in in-process short-term memory too, so
                # follow-ups have history even when no Agent Memory service is bound.
                _short_term_remember(context_id, query, response)
                card = await self._finalize_structured(result["messages"], response)
                # Remember this turn's GUID-addressable findings for the next turn
                # in the same conversation (so "fix that error" finds the GUID).
                findings = _findings_from_card(card.get("data") if isinstance(card, dict) else None)
                if findings:
                    _context_findings[context_id] = findings
                completed: dict = {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": response,
                }
                if card is not None:
                    completed["data"] = card
                yield completed

            except Exception as e:
                logger.exception("Agent stream error")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                yield {"is_task_complete": True, "require_user_input": False, "content": f"Error: {e}"}
        finally:
            try:
                span_cm.__exit__(None, None, None)
            except Exception:
                pass
            if token is not None:
                try:
                    otelcontext.detach(token)
                except ValueError:
                    # Token was created in a different async context (GeneratorExit path) — safe to ignore
                    pass

    def invoke(self, query: str, context_id: str) -> AgentResponse:
        try:
            history = _load_history(self.memory, context_id)
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=query)]
            result = asyncio.run(self.graph.ainvoke({"messages": messages}))
            response = result["messages"][-1].content
            return AgentResponse(status="completed", message=response)
        except Exception as e:
            logger.exception("Agent invoke error")
            return AgentResponse(status="error", message=f"Error: {e}")
