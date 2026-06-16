"""Structured card-contract schemas for Joule rendering.

Every agent turn that should render a Joule card returns one of these models,
serialized as the A2A `data` part. The Joule side (invoke_agent.yaml) routes on
`intent` and feeds `data` to the matching render_*_card function — exactly the
{ message, intent, data } contract the original Joule agents used.

`message` is always a Markdown fallback (shown when a client can't render the
card). `intent` selects the card. `data` is the per-intent payload.

Field names mirror the AIF_SRV OData V4 entity sets (see the aif-srv-mock-service
memory): IndexTableGenericSet, MessageLogSet, KeyFieldsSet, KeyFieldValueSet,
IfKeySet, InterfaceStatistics.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# interface_list  (IfKeySet + KeyFieldsSet)
# ---------------------------------------------------------------------------


class InterfaceItem(BaseModel):
    namespace: str = Field(description="Interface namespace, verbatim, e.g. 'NS1'.")
    interfaceName: str = Field(description="Interface name, e.g. 'ORDERS'.")
    interfaceVersion: str = Field(description="Interface version, e.g. '1'.")
    sapModule: str = Field(
        default="—",
        description="SAP module inferred from name/namespace (SD/FI/MM/CFIN…). '—' if unsure.",
    )
    about: str = Field(
        default="—",
        description="ONE concise sentence on what the interface does. '—' if unknown. Never invent specifics.",
    )
    searchBy: str = Field(
        default="—",
        description="Comma-separated key-field LABELS this interface can be searched by (from KeyFieldsSet). '—' when none.",
    )


class InterfaceListData(BaseModel):
    count: int = Field(description="Number of interfaces.")
    interfaces: list[InterfaceItem]


class InterfaceListResponse(BaseModel):
    intent: Literal["interface_list"] = "interface_list"
    message: str = Field(description="Markdown fallback, e.g. 'Found N interfaces available for monitoring.'")
    data: InterfaceListData


# ---------------------------------------------------------------------------
# worklist  (IndexTableGenericSet, Status eq 'E')   — also used by business_key mode=messages
# ---------------------------------------------------------------------------


class WorklistMessage(BaseModel):
    messageGuid: str
    namespace: str
    interfaceName: str
    interfaceVersion: str
    status: str = Field(description="AIF status: E/W/S/P/X.")
    processDate: str = Field(description="ISO date-time of processing, e.g. '2025-03-14T09:00:00Z'.")
    logMessage: str = Field(default="", description="Short log message / error summary.")


class WorklistData(BaseModel):
    count: int
    messages: list[WorklistMessage]


class WorklistResponse(BaseModel):
    intent: Literal["worklist"] = "worklist"
    message: str = Field(description="Markdown fallback: a short headline + table of the error messages.")
    data: WorklistData


# ---------------------------------------------------------------------------
# statistics / health  (InterfaceStatistics)
# ---------------------------------------------------------------------------


class InterfaceStat(BaseModel):
    namespace: str
    interfaceName: str
    interfaceVersion: str
    total: int = Field(description="NumberAllMessages (every status).")
    errors: int = Field(description="NumberErrorMessages.")
    warnings: int = Field(description="NumberWarningMessages.")
    success: int = Field(description="NumberSuccessMessages.")
    inProcess: int = Field(default=0, description="NumberInProcessMessages.")
    aborted: int = Field(default=0, description="NumberAbortMessages.")
    canceled: int = Field(default=0, description="NumberCanceledMessages.")
    health: str = Field(description="Critical (6+ errors) / Warning (1-5) / Healthy (0).")


class StatisticsData(BaseModel):
    interfaceCount: int
    interfaces: list[InterfaceStat]


class StatisticsResponse(BaseModel):
    # intent is "health" or "statistics"; both render the statistics card.
    intent: Literal["statistics", "health"] = "statistics"
    message: str = Field(description="Markdown fallback summarising per-interface health.")
    data: StatisticsData


# ---------------------------------------------------------------------------
# message_detail  (MessageLogSet by GUID)
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    msgType: str = Field(description="E/W/S/I.")
    msgId: str
    msgNo: str
    text: str


class MessageDetailData(BaseModel):
    messageGuid: str
    interface: str = Field(description="e.g. 'NS1 / ORDERS v1'.")
    logCount: int
    logEntries: list[LogEntry]


class MessageDetailResponse(BaseModel):
    intent: Literal["message_detail"] = "message_detail"
    message: str = Field(description="Markdown fallback: header + a log-entries table.")
    data: MessageDetailData


# ---------------------------------------------------------------------------
# business_key  (KeyFieldsSet / KeyFieldValueSet) — two modes
# ---------------------------------------------------------------------------


class BusinessKeyData(BaseModel):
    mode: Literal["interfaces", "messages"] = Field(
        description="'interfaces' = which interfaces carry the business object; 'messages' = messages matching a key value."
    )
    count: int
    # mode=interfaces -> interfaces[]; mode=messages -> messages[]
    interfaces: list[InterfaceItem] = Field(default_factory=list)
    messages: list[WorklistMessage] = Field(default_factory=list)


class BusinessKeyResponse(BaseModel):
    intent: Literal["business_key"] = "business_key"
    message: str = Field(description="Markdown fallback for the business-object search result.")
    data: BusinessKeyData


# ---------------------------------------------------------------------------
# resolution  (MessageLogSet MsgType='E' + grounding) — single-section card per the Joule renderer
# ---------------------------------------------------------------------------


class ResolutionError(BaseModel):
    msgId: str
    msgNo: str
    messageText: str = Field(description="The error text, verbatim from the log.")
    rootCause: str = Field(default="", description="From groundingText / SAP docs. Empty when ungrounded.")
    resolutionSteps: list[str] = Field(default_factory=list, description="From groundingText / SAP docs.")
    restartSafe: Literal["yes", "no", "unknown"] = "unknown"
    grounded: bool = Field(description="true when real steps were produced for this error.")
    sourceText: str = Field(
        description="ALWAYS set. Markdown link '[title](webUrl)' when a doc matched, else 'SAP standard documentation for <msgId>/<msgNo>'."
    )


class ResolutionData(BaseModel):
    messageGuid: str
    interface: str
    errorCount: int
    summary: str = Field(
        description="SHORT consolidated resolution for the WHOLE message: shared root cause + merged ordered steps. Grounded only."
    )
    errors: list[ResolutionError]


class ResolutionResponse(BaseModel):
    intent: Literal["resolution"] = "resolution"
    message: str = Field(description="Markdown fallback: Overall resolution + per-error sections.")
    data: ResolutionData


# ---------------------------------------------------------------------------
# analysis  (markdown report — no card; message carries the full report)
# ---------------------------------------------------------------------------


class AnalysisResponse(BaseModel):
    intent: Literal["analysis"] = "analysis"
    message: str = Field(
        description="The full Markdown monitoring analysis report (Summary Table, Key Findings, Error Summary, Error Resolution)."
    )
    # No structured card data for analysis — it renders as markdown text in Joule.
    data: Optional[dict] = None


# ---------------------------------------------------------------------------
# Flat envelope used for structured output.
#
# IMPORTANT: SAP AI Core's structured-output (response_format) endpoint requires a
# STRICT JSON Schema where every node has a `type`. A discriminated Pydantic Union
# compiles to `oneOf`/`$ref` with no top-level `type` and is REJECTED with
# "schema must have a 'type' key". So the finalizer is bound to this single flat
# object instead: `intent` (string) + `data` (free-form object). The per-intent
# shape is enforced by the FINALIZER_PROMPT and tolerated by the Joule renderers.
# The typed *Response models above remain the documentation of each intent's data.
# ---------------------------------------------------------------------------

INTENTS = [
    "interface_list",
    "worklist",
    "statistics",
    "health",
    "message_detail",
    "business_key",
    "resolution",
    "analysis",
]


class CardData(BaseModel):
    """Superset of every intent's `data` fields, all optional.

    A flat, fully-typed object (every node has a `type`) so AI Core's structured
    output accepts it, while still giving the LLM concrete fields to populate per
    intent. Only the fields relevant to the chosen `intent` are filled; the rest
    stay null/empty and the Joule renderer for that intent ignores them.
    """

    # Common counters
    count: Optional[int] = Field(default=None, description="Item count (interface_list, worklist, business_key).")
    interfaceCount: Optional[int] = Field(default=None, description="Interface count (statistics/health).")
    # interface_list / business_key(interfaces) / statistics
    interfaces: list[dict] = Field(
        default_factory=list,
        description="interface_list: {namespace, interfaceName, interfaceVersion, sapModule, about, searchBy}. statistics: {namespace, interfaceName, interfaceVersion, total, errors, warnings, success, inProcess, aborted, canceled, health}.",
    )
    # worklist / business_key(messages)
    messages: list[dict] = Field(
        default_factory=list,
        description="worklist/business_key messages: {messageGuid, namespace, interfaceName, interfaceVersion, status, processDate, logMessage}.",
    )
    # business_key
    mode: Optional[str] = Field(default=None, description="business_key only: 'interfaces' or 'messages'.")
    # message_detail
    messageGuid: Optional[str] = Field(default=None, description="message_detail / resolution: the GUID.")
    interface: Optional[str] = Field(default=None, description="message_detail / resolution: e.g. 'NS1 / ORDERS v1'.")
    logCount: Optional[int] = Field(default=None, description="message_detail: number of log entries.")
    logEntries: list[dict] = Field(
        default_factory=list, description="message_detail: {msgType, msgId, msgNo, text}."
    )
    # resolution
    errorCount: Optional[int] = Field(default=None, description="resolution: number of errors in the message.")
    summary: Optional[str] = Field(default=None, description="resolution: SHORT merged whole-message resolution.")
    errors: list[dict] = Field(
        default_factory=list,
        description="resolution: {msgId, msgNo, messageText, rootCause, resolutionSteps[], restartSafe, grounded, sourceText}.",
    )


class CardEnvelope(BaseModel):
    """Flat { message, intent, data } contract — the A2A `data` part for Joule."""

    intent: str = Field(description="One of: " + ", ".join(INTENTS) + ".")
    message: str = Field(description="Markdown fallback of the answer.")
    data: Optional[CardData] = Field(
        default_factory=CardData,
        description="Per-intent payload; fill only the chosen intent's fields. Null for analysis (the report lives in `message`).",
    )


# Bound by the finalizer in agent.py.
CardResponse = CardEnvelope
