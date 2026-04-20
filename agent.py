"""
LangGraph agent with OpenAI tools, MongoDB checkpointing, and Phoenix tracing.
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

import config
import db
from tools import all_tools

BOOKLY_SYSTEM_PROMPT = """You are an empathetic, concise customer support specialist for Bookly, an online bookstore.

Rules:
- Use tools for facts about orders, customers, and policies. Do not invent order status or policy details.
- Order IDs look like ORD-5001 or ORD-M001. If the user says "ORD M001" or similar, treat it as ORD-M001 when calling tools.
- If the customer asks about a specific order but does not give an order_id, ask for it before calling order lookup tools.
- Keep replies short and clear. Summarize tool output instead of pasting huge JSON.
- For returns or exchanges: confirm the replacement title and format (paperback, ebook, audiobook, hardcover), whether they want to keep their usual format preference, and whether to ship to the address on file or a new address. Use get_member_details to read preferences and shipping when helpful.
- Never refuse a password reset policy question: use search_policies with type password-reset when needed.

Tool output may be JSON; interpret it and respond in plain language."""

_phoenix_initialized = False


def _ensure_phoenix() -> None:
    """
    Instrument LangChain for Arize Phoenix once per process.

    Note: Legacy `phoenix.trace.langchain.LangChainInstrumentor` was removed in
    Phoenix 4.x; OpenInference is the supported path (see Phoenix docs).
    """
    global _phoenix_initialized
    if _phoenix_initialized:
        return
    tracer_provider = register()
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _phoenix_initialized = True


class BooklyState(TypedDict):
    """Conversation state; messages are accumulated via add_messages."""

    messages: Annotated[list[AnyMessage], add_messages]


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def build_checkpointer() -> MongoDBSaver:
    _require_env("MONGODB_URI")
    client = db.get_client()
    return MongoDBSaver(
        client,
        db_name=config.MONGODB_DB_NAME,
        checkpoint_collection_name=config.CHECKPOINT_COLLECTION,
    )


def build_graph():
    """Compile the LangGraph with tools and MongoDB-backed memory."""
    _ensure_phoenix()

    tools = all_tools()
    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=0.2,
    ).bind_tools(tools)

    def call_model(state: BooklyState):
        messages = [SystemMessage(content=BOOKLY_SYSTEM_PROMPT), *state["messages"]]
        reply = llm.invoke(messages)
        return {"messages": [reply]}

    def route_after_agent(state: BooklyState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(BooklyState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    checkpointer = build_checkpointer()
    return builder.compile(checkpointer=checkpointer)


_graph = None


def get_bookly_graph():
    """Singleton compiled graph (Streamlit-friendly)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
