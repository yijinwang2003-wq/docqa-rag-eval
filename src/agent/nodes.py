"""LangGraph node functions for the single-agent RAG workflow."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from src.agent.state import AgentState, add_trajectory_step
from src.agent.tools import (
    document_sources,
    generate_answer,
    retrieve_documents,
    rewrite_query,
)


MIN_QUERY_WORDS_FOR_DIRECT_RETRIEVAL = 5
LOW_CONFIDENCE_THRESHOLD = 0.45


def analyze_query(state: AgentState) -> AgentState:
    """Analyze whether the query should be rewritten before retrieval."""

    question = state["question"].strip()
    tokens = re.findall(r"\w+", question.lower())
    vague_references = {"it", "this", "that", "they", "those", "these"}
    has_vague_reference = any(token in vague_references for token in tokens)
    should_rewrite = len(tokens) < MIN_QUERY_WORDS_FOR_DIRECT_RETRIEVAL or has_vague_reference
    analysis = {
        "word_count": len(tokens),
        "has_vague_reference": has_vague_reference,
        "reason": (
            "short_or_contextual_query"
            if should_rewrite
            else "query_is_specific_enough"
        ),
    }

    return {
        "analysis": analysis,
        "should_rewrite": should_rewrite,
        "effective_question": question,
        "trajectory": add_trajectory_step(
            state,
            step="query_analysis",
            input_text=question,
            output_text=analysis["reason"],
            metadata=analysis,
        ),
    }


def maybe_rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the query only when analysis marks it as underspecified."""

    question = state["question"].strip()
    if not state.get("should_rewrite"):
        return {
            "effective_question": question,
            "trajectory": add_trajectory_step(
                state,
                step="query_rewrite",
                input_text=question,
                output_text=question,
                metadata={"rewritten": False},
            ),
        }

    rewritten_question = rewrite_query(question)
    if not rewritten_question:
        rewritten_question = question

    return {
        "rewritten_question": rewritten_question,
        "effective_question": rewritten_question,
        "trajectory": add_trajectory_step(
            state,
            step="query_rewrite",
            input_text=question,
            output_text=rewritten_question,
            metadata={"rewritten": rewritten_question != question},
        ),
    }


def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve documents with the selected existing retriever."""

    query = state.get("effective_question") or state["question"]
    retriever_name = state.get("retriever_name", "dense")
    start = perf_counter()
    documents = retrieve_documents(query, retriever_name)
    latency_s = perf_counter() - start
    sources = document_sources(documents)

    return {
        "documents": documents,
        "sources": sources,
        "trajectory": add_trajectory_step(
            state,
            step="document_retrieval",
            input_text=query,
            output_text=f"retrieved {len(documents)} document(s)",
            metadata={
                "retriever": retriever_name,
                "sources": sources,
                "latency_s": latency_s,
            },
        ),
    }


def generate_response(state: AgentState) -> AgentState:
    """Generate a grounded answer from retrieved context."""

    question = state["question"].strip()
    documents = state.get("documents", [])
    start = perf_counter()
    answer = generate_answer(question, documents)
    latency_s = perf_counter() - start

    return {
        "answer": answer,
        "trajectory": add_trajectory_step(
            state,
            step="answer_generation",
            input_text=question,
            output_text=answer,
            metadata={
                "context_documents": len(documents),
                "latency_s": latency_s,
            },
        ),
    }


def score_confidence(state: AgentState) -> AgentState:
    """Score answer confidence with a small deterministic fallback heuristic."""

    answer = state.get("answer", "")
    documents = state.get("documents", [])
    confidence, reason = _confidence_score(answer, documents)
    fallback = confidence < LOW_CONFIDENCE_THRESHOLD

    return {
        "confidence": confidence,
        "fallback": fallback,
        "fallback_reason": reason if fallback else "",
        "trajectory": add_trajectory_step(
            state,
            step="confidence_scoring",
            input_text=answer,
            output_text=f"{confidence:.2f}",
            metadata={
                "fallback": fallback,
                "reason": reason,
            },
        ),
    }


def _confidence_score(answer: str, documents: list[Any]) -> tuple[float, str]:
    if not documents:
        return 0.0, "no_documents_retrieved"

    normalized_answer = answer.lower()
    uncertainty_markers = (
        "i do not know",
        "i don't know",
        "not contain the answer",
        "not provided",
        "cannot answer",
    )
    if any(marker in normalized_answer for marker in uncertainty_markers):
        return 0.25, "answer_declined_or_missing_context"

    score = 0.45
    score += min(len(documents), 4) * 0.1
    if len(answer.split()) >= 20:
        score += 0.1
    if any(document.metadata.get("source") for document in documents):
        score += 0.05

    return min(score, 0.95), "sufficient_context_heuristic"
