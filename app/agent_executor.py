"""A2A executor bridge — wires CodemineAgent into the a2a-sdk 1.x server runtime."""

from __future__ import annotations

import logging

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types.a2a_pb2 import Part, Role, Task, TaskState
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from opentelemetry.propagate import extract

from app.agent import CodemineAgent
from app.auth import current_user_id

logger = logging.getLogger(__name__)


def _text_part(text: str) -> Part:
    return Part(text=text)


def _data_part(data: dict) -> Part:
    return Part(data=json_format.ParseDict(data, Value()))


class CodemineAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agent = CodemineAgent(memory_client=None)

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        query = context.get_user_input()
        task_id = context.task_id
        context_id = context.context_id

        # Extract W3C traceparent from caller to link traces
        carrier = dict(getattr(context, "headers", {}) or {})
        parent_ctx = extract(carrier)

        # User identity set by XSUAAAuthMiddleware into the context var
        user_id = current_user_id.get()

        # Enqueue the Task object first — required before any status events
        task = Task(id=task_id, context_id=context_id)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        # Extract conversation history from the task so the agent sees prior turns
        # without needing the memory client (works locally and on CF).
        a2a_history = []
        if context.current_task and context.current_task.history:
            for msg in context.current_task.history:
                text = " ".join(p.text for p in msg.parts if p.HasField("text"))
                if text:
                    a2a_history.append({"role": msg.role, "text": text})

        try:
            async for item in self.agent.stream(
                query, context_id, a2a_history=a2a_history, parent_context=parent_ctx, user_id=user_id
            ):
                if not item["is_task_complete"] and not item["require_user_input"]:
                    msg = updater.new_agent_message([_text_part(item["content"])])
                    await updater.update_status(TaskState.TASK_STATE_WORKING, message=msg)
                elif item["require_user_input"]:
                    parts = [_text_part(item["content"])]
                    if "data" in item:
                        parts.append(_data_part(item["data"]))
                    msg = updater.new_agent_message(parts)
                    await updater.requires_input(message=msg)
                    return
                else:
                    parts = [_text_part(item["content"])]
                    if "data" in item:
                        parts.append(_data_part(item["data"]))
                    await updater.add_artifact(parts, name="agent_result")
                    await updater.complete()
                    return
        except Exception:
            logger.exception("Agent execution error")
            await updater.failed()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task_id = context.task_id
        context_id = context.context_id
        task = Task(id=task_id, context_id=context_id)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()
