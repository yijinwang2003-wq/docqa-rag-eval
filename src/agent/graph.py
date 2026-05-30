"""LangGraph workflow for Phase 1 agentic RAG."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    analyze_query,
    generate_response,
    maybe_rewrite_query,
    retrieve_context,
    score_confidence,
)
from src.agent.state import AgentState


def build_agent_graph():
    """Compile the single-agent RAG workflow."""

    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("rewrite_query", maybe_rewrite_query)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("score_confidence", score_confidence)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "score_confidence")
    graph.add_edge("score_confidence", END)

    return graph.compile()


def run_agent(question: str, retriever_name: str = "dense") -> AgentState:
    """Run the compiled agent graph for one question."""

    graph = build_agent_graph()
    return graph.invoke(
        {
            "question": question,
            "retriever_name": retriever_name,
            "trajectory": [],
        }
    )
