"""State objects for the agentic RAG workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document


class TrajectoryStep(TypedDict, total=False):
    """A compact log entry for one graph step."""

    step: str
    input: str
    output: str
    metadata: dict[str, Any]


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    retriever_name: str
    analysis: dict[str, Any]
    should_rewrite: bool
    rewritten_question: str
    effective_question: str
    documents: list[Document]
    sources: list[str]
    answer: str
    confidence: float
    fallback: bool
    fallback_reason: str
    trajectory: list[TrajectoryStep]


def add_trajectory_step(
    state: AgentState,
    *,
    step: str,
    input_text: str,
    output_text: str,
    metadata: dict[str, Any] | None = None,
) -> list[TrajectoryStep]:
    """Append a trajectory step without mutating the incoming state."""

    return [
        *state.get("trajectory", []),
        {
            "step": step,
            "input": input_text,
            "output": output_text,
            "metadata": metadata or {},
        },
    ]
