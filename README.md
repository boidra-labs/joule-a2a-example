# aif-analysis-agent

An A2A-protocol AI agent that monitors SAP AIF (Application Interface Framework) interfaces over **AIF_SRV** (OData V4) and returns the SAP Joule **`{ message, intent, data }`** card contract.

Runs on **SAP BTP Cloud Foundry** as a Docker image. Speaks A2A v0.3 and v1.0 (Joule compatible).

> See [`docs/AGENT_DESIGN.md`](docs/AGENT_DESIGN.md) for the full design: intents, the structured-output finalizer, tools, and the Joule integration.

---

## What it does

The LLM tool-loop reads the **intent** (which tools it calls) and **parameters** (the args), then a JSON-mode **finalizer** packages the result into `{ message, intent, data }`. Supported intents:

1. **interface_list** — list interfaces available for monitoring (+ searchable key-field labels).
2. **worklist** — list AIF error messages for a period / interface.
3. **statistics / health** — per-interface counts + health (Critical 6+ / Warning 1–5 / Healthy 0) for a period.
4. **analysis** — full Markdown monitoring analysis report for a period.
5. **message_detail** — full error log of a message by MSGGUID.
6. **resolution** — grounded root cause + steps per error (AI Core document grounding).
7. **business_key** — search by a business object (customer, vendor, PO, cost center…): which interfaces carry it, or matching messages.
8. **interface explanation** — what an interface does given namespace / name / version.

---

## Architecture

```
SAP Joule (BYOA)
    │
    │  A2A v0.3 / v1.0
    ▼
aif-analysis-agent
    │  LiteLLM (sap/gpt-4.1)
    ├──▶ SAP AI Core
    │
    ├──▶ BTP Destination ──▶ AIF_SRV OData V4 (mock / S/4HANA / CF)
    │
    ├──▶ AI Core Document Grounding (error catalog)
    │
    └──▶ OpenTelemetry ──▶ SAP Cloud Logging / Honeycomb
```

| Layer | Choice |
|---|---|
| Agent framework | LangGraph + LangChain |
| LLM gateway | LiteLLM + `sap-ai-sdk-gen` → SAP AI Core (`sap/gpt-4.1`) |
| A2A protocol | `a2a-sdk==1.0.2` (v0.3 compat enabled) |
| HTTP server | uvicorn + Starlette |
| Auth | XSUAA JWT middleware (RS256) |
| Memory | SAP Agent Memory (optional, multi-turn) |
| Observability | `sap-cloud-sdk` `auto_instrument()` → OTLP |
| Container | `python:3.13-slim` |
| Registry | `ghcr.io/codeminepl/aif-analysis-agent` |
| Runtime | SAP BTP Cloud Foundry |

---

## Tools

| Tool | Purpose | Entity set |
|---|---|---|
| `calculate_date_range_tool` | Resolve a free-text period ('2025', a range) into OData V4 from/to bounds | — |
| `list_interfaces_tool` | List interfaces + searchable key-field labels | `IfKeySet` + `KeyFieldsSet` |
| `list_error_messages_tool` | Worklist of AIF messages for a date window / interface / status | `IndexTableGenericSet` |
| `get_interface_statistics_tool` | Per-interface counts + health for a date window | `InterfaceStatistics` |
| `get_key_fields_tool` | Map a business object → key field (FieldName/Label/SemObj) | `KeyFieldsSet` |
| `find_messages_by_key_value_tool` | Messages matching FieldName **and** FieldValue | `KeyFieldValueSet` |
| `get_message_log_tool` | A message's ERROR log entries by GUID | `MessageLogSet` (`MsgType='E'`) |
| `run_doc_error_catalog_tool` | AI Core document grounding for root cause / resolution | grounding repo |
| `get_interface_details_tool` | Structured explanation of an interface (namespace / name / version) | LLM knowledge |
| `get_agent_capabilities_tool` | Fetches this agent's own card (skills + example queries) | self |

---

## Project structure

```
aif-analysis-agent/
├── Dockerfile                         # python:3.13-slim, port 9000
├── manifest.yml                       # CF deployment descriptor
├── requirements.txt
├── xs-security.json                   # XSUAA app descriptor
├── app/
│   ├── main.py                        # A2A server, agent card, Starlette routes
│   ├── agent.py                       # LangGraph agent, tools, system prompt
│   ├── agent_executor.py              # A2A SDK ↔ LangGraph bridge
│   ├── bootstrap.py                   # VCAP_SERVICES → AICORE_* + OTel init
│   └── auth.py                        # XSUAA JWT middleware
└── joule/a2a/
    ├── JOULE.EXT_aif_analysis_agent_1.0.0.daar
    ├── capability_context.yaml
    ├── functions/
    │   └── aif_analysis_agent.yaml    # Joule dialog function
    └── scenarios/
        └── aif_analysis/
            └── aif_analysis.yaml      # Joule scenario
```

---

## Analysis report format

When errors are found the agent produces a Markdown report with:

- **Summary Table** — interface name, description, type, version, total errors, warnings, criticality, top MSGID/MSGNO/MSGTX
- **Key Findings** — critical / warning / healthy interface counts, most frequent error, affected source systems
- **Error Summary** — ranked error list with occurrence counts and affected interfaces
- **Error Resolution** — verbatim content from the AI Core grounding repository for the top error

Criticality thresholds: 6+ errors = Critical, 1–5 = Warning, 0 = Healthy.

---

## Joule integration

The `joule/a2a/` folder contains the Design Time Artifacts (DTA) for SAP Joule BYOA:

| File | Purpose |
|---|---|
| `capability_context.yaml` | Declares context variables: `agent_context_id`, `agent_task_id`, `agent_state` |
| `functions/aif_analysis_agent.yaml` | Dialog function that calls the agent via `agent-request`, extracts report text and action buttons from the A2A response |
| `scenarios/aif_analysis/aif_analysis.yaml` | Joule scenario that invokes the function, sets response context, and renders the output message |
| `JOULE.EXT_aif_analysis_agent_1.0.0.daar` | Packaged DTA archive for upload to SAP BTP Joule |

The agent card URL registered in Joule points at the CF route via a BTP Destination. A2A v0.3 compatibility is permanently enabled (`enable_v0_3_compat=True`) so Joule's `message/send` calls work alongside v1.0 `SendMessage` clients.

---

## Deployment

### Prerequisites

- Docker + `gh` CLI installed, `gh auth login` done, `docker login ghcr.io` done
- CF CLI targeting `codemine-sa` org / `codemine_demo` space
- BTP services created (see below)

### 1. Build and push the Docker image

```bash
docker build -t ghcr.io/codeminepl/aif-analysis-agent:0.8.0 .
docker push ghcr.io/codeminepl/aif-analysis-agent:0.8.0
```

### 2. Create BTP services

```bash
cf create-service aicore          extended    aif-analysis-agent-aicore
cf create-service destination     lite        aif-analysis-agent-destination
cf create-service xsuaa           application aif-analysis-agent-xsuaa -c xs-security.json

# Optional
cf create-service hana-agent-memory  default  aif-analysis-agent-agent-memory
cf create-service application-logs   standard aif-analysis-agent-cloud-logging
```

### 3. Set environment variables

```bash
# GHCR pull secret
cf set-env aif-analysis-agent CF_DOCKER_PASSWORD <ghcr-pat>

# Observability (if not using cloud-logging binding)
cf set-env aif-analysis-agent OTEL_EXPORTER_OTLP_ENDPOINT "https://api.honeycomb.io"
cf set-env aif-analysis-agent OTEL_EXPORTER_OTLP_HEADERS  "x-honeycomb-team=<KEY>"
cf set-env aif-analysis-agent OTEL_EXPORTER_OTLP_PROTOCOL "http/protobuf"
cf set-env aif-analysis-agent OTEL_METRICS_EXPORTER       "none"
```

### 4. Deploy

```bash
cf login -a https://api.cf.eu10-005.hana.ondemand.com --sso
cf target -o codemine-sa -s codemine_demo
cf push -f manifest.yml --docker-username codemine-kwasniewskim
```

The app is live at: `https://aif-analysis-agent.cfapps.eu10-005.hana.ondemand.com`

Health check: `GET /.well-known/agent-card.json`

### Deployed services (manifest.yml)

| Service instance | Plan | Required |
|---|---|---|
| `aif-analysis-agent-aicore` | extended | Yes — LLM + document grounding |
| `aif-analysis-agent-destination` | lite | Yes — resolves `dest-aif-service` |
| `aif-analysis-agent-xsuaa` | application | Yes — JWT auth |
| `aif-analysis-agent-agent-memory` | default | Optional — multi-turn memory |
| `aif-analysis-agent-auditlog` | — | Optional |
| `aif-analysis-agent-cloud-logging` | — | Optional — logs + traces |

---

## Local development

Create `.env` in the project root:

```dotenv
AICORE_AUTH_URL=https://<subaccount>.authentication.eu10.hana.ondemand.com
AICORE_CLIENT_ID=sb-...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=https://api.ai.eu10.ml.hana.ondemand.com
AICORE_RESOURCE_GROUP=default

# AIF backend. Local testing uses the public AIF_SRV mockup (OData V4, no auth):
AIF_ODATA_URL=https://aif-standard-mockup-service.cfapps.eu10-005.hana.ondemand.com/sap/opu/odata/sap/AIF_SRV
DISABLE_AUTH=true
```

```bash
pip install -r requirements.txt
python -m app.main            # or: ../run-local.ps1 -Agents "aif"
```

Agent card available at: `http://localhost:9000/.well-known/agent-card.json`

---

## Example queries

```
Which interfaces are available for monitoring?
Show me AIF errors for ORDERS in 2025
Run a health check for all interfaces in 2025
Give me a full AIF interface analysis for 2025
Show details for message GUID 000000016071DFD7AD912FEE8284FD2D
How do I fix message 000000016071DFD7AD912FEE8284FD2D
Which interfaces are connected to Customer
What does interface /FINCF / AC_DOC version 0001 do?
```
