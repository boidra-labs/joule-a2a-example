# Load .env before anything else so AICORE_* vars are available to bootstrap
from dotenv import load_dotenv
load_dotenv()

# CRITICAL: Initialize telemetry and AI Core BEFORE importing AI frameworks.
from app.bootstrap import configure_aicore, configure_telemetry, configure_memory  # noqa: E402
configure_aicore()
configure_telemetry()

import logging  # noqa: E402
import os  # noqa: E402

import click  # noqa: E402
import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.routes.agent_card_routes import create_agent_card_routes  # noqa: E402
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore  # noqa: E402
from a2a.types.a2a_pb2 import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)

from app import __version__ as AGENT_VERSION  # noqa: E402
from app.agent_executor import CodemineAgentExecutor  # noqa: E402
from app.auth import XSUAAAuthMiddleware  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "9000"))


def build_app() -> Starlette:
    public_url = os.environ.get("AGENT_PUBLIC_URL", f"http://{HOST}:{PORT}/")

    skill = AgentSkill(
        id="aif-analysis-agent",
        name="aif-analysis-agent",
        description=(
            "SAP AIF interface monitoring agent for Central Finance. "
            "Returns structured { message, intent, data } responses that SAP Joule renders as cards. "
            "Capabilities: "
            "(1) List interfaces available for monitoring. "
            "(2) Worklist — list AIF error messages for a period / interface. "
            "(3) Statistics & health check per interface for a period. "
            "(4) Full monitoring analysis report for a period. "
            "(5) Look up a specific AIF message by MSGGUID. "
            "(6) Grounded error resolution for a message (root cause + steps). "
            "(7) Search by business object (customer, vendor, PO, cost center…) — which interfaces carry it, or matching messages. "
            "(8) Explain what a SAP AIF interface does given its namespace, name, and version."
        ),
        tags=["aif", "interface-monitoring", "central-finance", "s4hana", "analysis", "resolution"],
        examples=[
            "Which interfaces are available for monitoring?",
            "Show me AIF errors for ORDERS in 2025",
            "Run a health check for all interfaces in 2025",
            "Give me a full AIF interface analysis for 2025",
            "Show details for message GUID 000000016071DFD7AD912FEE8284FD2D",
            "How do I fix message 000000016071DFD7AD912FEE8284FD2D",
            "Which interfaces are connected to Customer",
            "What does interface /FINCF / AC_DOC version 0001 do?",
        ],
    )
    agent_card = AgentCard(
        name="aif-analysis-agent",
        description=(
            "An AI agent for SAP Central Finance analysts that monitors AIF interfaces over AIF_SRV: "
            "lists interfaces, builds error worklists and per-interface statistics/health, produces full "
            "monitoring analysis reports, looks up messages by GUID, grounds error resolutions, and searches "
            "by business object. Returns the Joule { message, intent, data } card contract."
        ),
        # Declare the JSON-RPC transport explicitly so A2A clients (e.g. the A2A
        # Inspector) know this interface supports message/send AND message/stream
        # (capabilities.streaming=True). An empty protocol_binding can make clients
        # fall back to non-streaming.
        supported_interfaces=[AgentInterface(url=public_url, protocol_binding="JSONRPC")],
        version=AGENT_VERSION,
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    task_store = InMemoryTaskStore()
    # Persistent per-context_id memory (history + vector). None if unbound — the
    # agent then falls back to transient A2A task history only.
    memory_client = configure_memory()
    executor = CodemineAgentExecutor(memory_client=memory_client)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    routes = [
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(handler, rpc_url="/", enable_v0_3_compat=True),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(XSUAAAuthMiddleware)
    return app


@click.command()
@click.option("--host", default=HOST)
@click.option("--port", default=PORT, type=int)
def main(host: str, port: int) -> None:
    app = build_app()
    logger.info("Starting A2A server at http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
