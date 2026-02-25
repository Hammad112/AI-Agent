"""
agent/agent.py
--------------
LangGraph state machine agent with a custom BATCHED LLM router.
Import paths changed from flat modules to package paths.
"""

import os
import uuid
import json
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from core.llm_client import llm_call
from core.database import get_global_stats, get_business_meta
from core.logger import (
    log_message,
    log_tool_call,
    log_chunk_retrieval,
    log_agent_event,
    get_conversation_history,
)
from agent.rag_engine import (
    chunk_query_into_topics,
    retrieve_relevant_chunks,
    get_customer_context,
    build_composite_prompt,
)
from agent.tools import TOOLS


# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    user_id: int
    conversation_id: str
    query: str
    history: list[dict]
    all_chunks: list[dict]
    retrieved_chunks: list[dict]
    retrieved_chunk_ids: list[int]
    query_topics: list[str]
    customer_context: dict
    global_stats: dict
    tool_results: dict
    activated_tools: list[str]
    response: str
    business_name: str
    business_type: str


# ─────────────────────────────────────────────
# Node: receive_input
# ─────────────────────────────────────────────

def node_receive_input(state: AgentState) -> AgentState:
    history = get_conversation_history(state["conversation_id"], limit=10)
    log_message(state["conversation_id"], "user", state["query"])
    return {**state, "history": history}


# ─────────────────────────────────────────────
# Node: rag_retrieval
# ─────────────────────────────────────────────

def node_rag_retrieval(state: AgentState) -> AgentState:
    topics = chunk_query_into_topics(state["query"], state["history"])
    log_agent_event(state["conversation_id"], "query_topics", {"topics": topics})

    retrieved_chunks, chunk_ids = retrieve_relevant_chunks(
        topics, state["all_chunks"], top_n=5
    )
    if chunk_ids:
        log_chunk_retrieval(state["conversation_id"], chunk_ids)

    customer_ctx = get_customer_context(state["user_id"])
    global_stats = get_global_stats()

    return {
        **state,
        "query_topics": topics,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_chunk_ids": chunk_ids,
        "customer_context": customer_ctx,
        "global_stats": global_stats,
    }


# ─────────────────────────────────────────────
# Node: route_tools  (custom BATCHED LLM router)
# ─────────────────────────────────────────────

def node_route_tools(state: AgentState) -> AgentState:
    """
    Custom LLM-based router — decides ALL tools in a SINGLE LLM call.
    This is NOT LangChain's built-in router. It's a custom per-tool decision
    made by the LLM, but batched into one prompt to conserve API quota.
    """
    query = state["query"]
    topics = state["query_topics"]
    history_snippet = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["history"][-3:]
    )
    customer_tier = state.get("customer_context", {}).get("loyalty_tier", "bronze")

    tools_list = "\n".join(
        f"{i+1}. {name}: {desc}"
        for i, (name, (_, desc)) in enumerate(TOOLS.items())
    )

    prompt = f"""You are a routing engine for a customer service AI agent.
You must decide which tools to activate for the customer's query.

Customer query: "{query}"
Query topics: {topics}
Customer loyalty tier: {customer_tier}
Recent conversation:
{history_snippet or 'None'}

Available tools (name: purpose):
{tools_list}

Instructions:
- Review each tool carefully.
- Select ONLY the tools that are directly needed to handle this specific query.
- Do NOT activate tools unless the query clearly requires their action.
- Return ONLY a JSON array of tool names to activate, e.g.: ["tool_a", "tool_b"]
- If no tools are needed (e.g. general question), return: []

JSON array of tools to activate:"""

    activated = []
    try:
        raw = llm_call(prompt, temperature=0.0, max_tokens=200).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            tool_list = json.loads(raw[start:end])
            activated = [t for t in tool_list if t in TOOLS]
    except Exception as e:
        log_agent_event(state["conversation_id"], "router_error", {"error": str(e)})
        activated = []

    for tool_name in TOOLS:
        activated_flag = tool_name in activated
        log_tool_call(
            state["conversation_id"],
            tool_name,
            {"query": query, "topics": topics},
            {"activated": activated_flag},
            activated=activated_flag,
        )

    log_agent_event(state["conversation_id"], "tool_routing", {"activated_tools": activated})
    return {**state, "activated_tools": activated}


# ─────────────────────────────────────────────
# Node: execute_tools
# ─────────────────────────────────────────────

def node_execute_tools(state: AgentState) -> AgentState:
    tool_results: dict = {}
    for tool_name in state["activated_tools"]:
        if tool_name not in TOOLS:
            continue
        tool_fn, _ = TOOLS[tool_name]
        try:
            result = tool_fn(state)
            tool_results[tool_name] = result
            log_tool_call(
                state["conversation_id"],
                tool_name,
                {"query": state["query"]},
                result,
                activated=True,
            )
        except Exception as e:
            tool_results[tool_name] = {"success": False, "error": str(e)}
            log_agent_event(state["conversation_id"], "tool_error", {"tool": tool_name, "error": str(e)})

    return {**state, "tool_results": tool_results}


# ─────────────────────────────────────────────
# Node: build_response
# ─────────────────────────────────────────────

def node_build_response(state: AgentState) -> AgentState:
    composite_prompt = build_composite_prompt(
        query=state["query"],
        retrieved_chunks=state["retrieved_chunks"],
        customer_context=state["customer_context"],
        global_stats=state["global_stats"],
        tool_results=state["tool_results"],
        conversation_history=state["history"],
        business_name=state["business_name"],
        business_type=state["business_type"],
    )

    response = llm_call(composite_prompt, temperature=0.5, max_tokens=800)
    log_message(state["conversation_id"], "assistant", response)
    log_agent_event(
        state["conversation_id"],
        "response_generated",
        {
            "retrieved_chunks": len(state["retrieved_chunks"]),
            "activated_tools": state["activated_tools"],
            "response_length": len(response),
        },
    )
    return {**state, "response": response}


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("receive_input", node_receive_input)
    graph.add_node("rag_retrieval", node_rag_retrieval)
    graph.add_node("route_tools", node_route_tools)
    graph.add_node("execute_tools", node_execute_tools)
    graph.add_node("build_response", node_build_response)

    graph.set_entry_point("receive_input")
    graph.add_edge("receive_input", "rag_retrieval")
    graph.add_edge("rag_retrieval", "route_tools")
    graph.add_edge("route_tools", "execute_tools")
    graph.add_edge("execute_tools", "build_response")
    graph.add_edge("build_response", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

_compiled_graph = None


def get_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent_turn(
    user_id: int,
    conversation_id: str,
    query: str,
    all_chunks: list[dict],
    business_name: str,
    business_type: str,
) -> str:
    agent = get_agent()
    initial_state: AgentState = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query": query,
        "history": [],
        "all_chunks": all_chunks,
        "retrieved_chunks": [],
        "retrieved_chunk_ids": [],
        "query_topics": [],
        "customer_context": {},
        "global_stats": {},
        "tool_results": {},
        "activated_tools": [],
        "response": "",
        "business_name": business_name,
        "business_type": business_type,
    }
    final_state = agent.invoke(initial_state)
    return final_state["response"]
