"""Compare baseline RAG with the single-agent RAG workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from src.agent.graph import run_agent
from src.config import PROJECT_ROOT
from src.evaluation import (
    EvalQuestion,
    RagRun,
    evaluate_runs,
    load_eval_questions,
)
from src.rag_chain import (
    PROMPT,
    format_docs,
    get_llm,
    get_selected_retriever,
    unique_sources,
)


DEFAULT_AGENT_EVAL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "agent_eval.csv"
DEFAULT_AGENT_EVAL_PARTIAL_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "agent_eval_partial.csv"
)
DEFAULT_AGENT_EVAL_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "agent_eval.md"
DEFAULT_TRAJECTORY_METRICS = ("answer_relevancy", "faithfulness")


@dataclass(frozen=True)
class TrajectoryEvalResult:
    """Paths and frames produced by trajectory evaluation."""

    results_df: Any
    summary_df: Any
    output_path: Path
    partial_output_path: Path
    markdown_path: Path | None


def run_trajectory_evaluation(
    dataset_path: Path,
    output_path: Path = DEFAULT_AGENT_EVAL_OUTPUT_PATH,
    partial_output_path: Path = DEFAULT_AGENT_EVAL_PARTIAL_OUTPUT_PATH,
    markdown_path: Path | None = DEFAULT_AGENT_EVAL_MARKDOWN_PATH,
    retriever_name: str = "hybrid",
    metrics: Sequence[str] = DEFAULT_TRAJECTORY_METRICS,
    limit: int | None = None,
    score_answers: bool = True,
) -> TrajectoryEvalResult:
    """Run baseline and agentic RAG on the same questions and write results."""

    pd = _import_pandas()
    questions = load_eval_questions(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than zero.")
        questions = questions[:limit]

    results_df = _load_existing_results(
        pd,
        partial_output_path=partial_output_path,
        output_path=output_path,
    )
    completed_keys = _completed_question_keys(
        results_df,
        retriever_name=retriever_name,
        metrics=metrics,
        score_answers=score_answers,
    )
    total_questions = len(questions)

    for index, item in enumerate(questions, start=1):
        print(f"Question {index}/{total_questions}: {item.question}", flush=True)
        question_key = _question_key(item, retriever_name)
        if question_key in completed_keys:
            print("Skipping existing completed result.", flush=True)
            continue

        try:
            baseline_row, baseline_run = _run_baseline_rag(item, retriever_name)
            agent_row, agent_run = _run_agentic_rag(item, retriever_name)
            question_df = _score_question_rows(
                pd,
                rows=[baseline_row, agent_row],
                rag_runs=[baseline_run, agent_run],
                output_path=output_path,
                metrics=metrics,
                score_answers=score_answers,
            )
            results_df = pd.concat([results_df, question_df], ignore_index=True)
            completed_keys.add(question_key)
            _write_results(results_df, partial_output_path)
        except Exception as exc:
            _write_results(results_df, partial_output_path)
            if _is_openai_api_error(exc):
                print(
                    "OpenAI evaluation call failed; completed rows were saved to "
                    f"{partial_output_path}. Error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            raise

    _write_results(results_df, partial_output_path)
    _write_results(results_df, output_path)

    summary_df = summarize_trajectory_results(results_df, metrics=metrics)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            dataframe_to_markdown(summary_df) + "\n",
            encoding="utf-8",
        )

    return TrajectoryEvalResult(
        results_df=results_df,
        summary_df=summary_df,
        output_path=output_path,
        partial_output_path=partial_output_path,
        markdown_path=markdown_path,
    )


def summarize_trajectory_results(
    results_df,
    metrics: Sequence[str] = DEFAULT_TRAJECTORY_METRICS,
):
    """Aggregate per-question trajectory rows into a README-friendly table."""

    pd = _import_pandas()
    summary_columns = [
        column
        for column in (
            *metrics,
            "tool_step_count",
            "query_rewrite_used",
            "retrieval_success_proxy",
            "retrieved_document_count",
            "web_search_used",
            "total_latency_s",
        )
        if column in results_df.columns
    ]
    if not summary_columns:
        return pd.DataFrame()

    return (
        results_df.groupby("system", as_index=False)[summary_columns]
        .mean(numeric_only=True)
        .sort_values("system")
        .round(3)
    )


def _score_question_rows(
    pd,
    *,
    rows: list[dict[str, Any]],
    rag_runs: list[RagRun],
    output_path: Path,
    metrics: Sequence[str],
    score_answers: bool,
):
    question_df = pd.DataFrame(rows)
    if score_answers and metrics:
        scored_df = evaluate_runs(
            rag_runs,
            output_path=output_path,
            metrics=metrics,
            write_output=False,
        )
        for metric in metrics:
            if metric in scored_df:
                question_df[metric] = scored_df[metric].tolist()
            else:
                question_df[metric] = None
    else:
        for metric in metrics:
            question_df[metric] = None

    return question_df


def _load_existing_results(pd, *, partial_output_path: Path, output_path: Path):
    for path in (partial_output_path, output_path):
        if not path.exists():
            continue
        try:
            return pd.read_csv(path)
        except Exception:
            continue

    return pd.DataFrame()


def _write_results(results_df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)


def _completed_question_keys(
    results_df,
    *,
    retriever_name: str,
    metrics: Sequence[str],
    score_answers: bool,
) -> set[str]:
    if results_df.empty:
        return set()
    if not {"question", "ground_truth", "system", "retriever"}.issubset(
        results_df.columns
    ):
        return set()

    completed_keys = set()
    required_metrics = tuple(metrics) if score_answers else ()
    for (question, ground_truth), group in results_df.groupby(
        ["question", "ground_truth"],
        dropna=False,
    ):
        if set(group.get("system", [])) != {"baseline_rag", "agentic_rag"}:
            continue
        if not all(
            str(value) == retriever_name for value in group.get("retriever", [])
        ):
            continue
        if required_metrics and not all(
            metric in group.columns and group[metric].notna().all()
            for metric in required_metrics
        ):
            continue
        completed_keys.add(
            _hash_key(str(question), str(ground_truth), retriever_name)
        )

    return completed_keys


def _run_baseline_rag(
    item: EvalQuestion,
    retriever_name: str,
) -> tuple[dict[str, Any], RagRun]:
    total_start = perf_counter()
    retriever = get_selected_retriever(retriever_name)

    retrieval_start = perf_counter()
    documents = list(retriever.invoke(item.question))
    retrieval_latency_s = perf_counter() - retrieval_start

    generation_start = perf_counter()
    answer_chain = PROMPT | get_llm() | StrOutputParser()
    answer = answer_chain.invoke(
        {
            "question": item.question,
            "context": format_docs(documents),
        }
    )
    generation_latency_s = perf_counter() - generation_start
    total_latency_s = perf_counter() - total_start
    sources = unique_sources(documents)

    row = _base_row(
        system="baseline_rag",
        retriever_name=retriever_name,
        item=item,
        answer=answer,
        sources=sources,
        documents=documents,
        total_latency_s=total_latency_s,
        retrieval_latency_s=retrieval_latency_s,
        generation_latency_s=generation_latency_s,
        trajectory=[],
    )
    row["tool_step_count"] = 2
    row["query_rewrite_used"] = False

    return row, _row_to_rag_run(row, item, documents)


def _run_agentic_rag(
    item: EvalQuestion,
    retriever_name: str,
) -> tuple[dict[str, Any], RagRun]:
    total_start = perf_counter()
    result = run_agent(item.question, retriever_name=retriever_name)
    total_latency_s = perf_counter() - total_start
    documents = list(result.get("documents", []))
    trajectory = list(result.get("trajectory", []))

    row = _base_row(
        system="agentic_rag",
        retriever_name=retriever_name,
        item=item,
        answer=str(result.get("answer", "")),
        sources=list(result.get("sources", [])),
        documents=documents,
        total_latency_s=total_latency_s,
        retrieval_latency_s=_trajectory_latency(trajectory, "document_retrieval"),
        generation_latency_s=_trajectory_latency(trajectory, "answer_generation"),
        trajectory=trajectory,
    )
    row["tool_step_count"] = len(trajectory)
    row["query_rewrite_used"] = _query_rewrite_used(trajectory)
    row["confidence"] = result.get("confidence")
    row["fallback"] = bool(result.get("fallback", False))
    row["fallback_reason"] = result.get("fallback_reason", "")
    row["web_search_used"] = bool(result.get("web_search_used", False))
    row["web_fallback_latency_s"] = _trajectory_latency(
        trajectory,
        "web_search_fallback",
    )

    return row, _row_to_rag_run(row, item, documents)


def _base_row(
    *,
    system: str,
    retriever_name: str,
    item: EvalQuestion,
    answer: str,
    sources: list[str],
    documents,
    total_latency_s: float,
    retrieval_latency_s: float | None,
    generation_latency_s: float | None,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "system": system,
        "retriever": retriever_name,
        "question": item.question,
        "ground_truth": item.ground_truth,
        "answer": answer,
        "sources": sources,
        "retrieved_document_count": len(documents),
        "retrieval_success_proxy": _retrieval_success_proxy(documents),
        "retrieval_latency_s": retrieval_latency_s,
        "generation_latency_s": generation_latency_s,
        "total_latency_s": total_latency_s,
        "tool_step_count": len(trajectory),
        "query_rewrite_used": False,
        "trajectory_steps": [step.get("step", "") for step in trajectory],
        "confidence": None,
        "fallback": False,
        "fallback_reason": "",
        "web_search_used": False,
        "web_fallback_latency_s": None,
    }


def _row_to_rag_run(row: dict[str, Any], item: EvalQuestion, documents) -> RagRun:
    return RagRun(
        chunking="agent_eval",
        retriever=f"{row['system']}+{row['retriever']}",
        question=item.question,
        ground_truth=item.ground_truth,
        answer=row["answer"],
        contexts=[document.page_content for document in documents],
        sources=list(row["sources"]),
        retrieval_latency_s=row["retrieval_latency_s"] or 0.0,
        generation_latency_s=row["generation_latency_s"] or 0.0,
        total_latency_s=row["total_latency_s"],
    )


def _question_key(item: EvalQuestion, retriever_name: str) -> str:
    return _hash_key(item.question, item.ground_truth, retriever_name)


def _hash_key(*parts: str) -> str:
    raw = "\n".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _retrieval_success_proxy(documents) -> bool:
    return bool(documents) and any(
        document.page_content.strip() for document in documents
    )


def _query_rewrite_used(trajectory: list[dict[str, Any]]) -> bool:
    for step in trajectory:
        if step.get("step") != "query_rewrite":
            continue
        metadata = step.get("metadata") or {}
        return bool(metadata.get("rewritten"))
    return False


def _trajectory_latency(
    trajectory: list[dict[str, Any]],
    step_name: str,
) -> float | None:
    for step in trajectory:
        if step.get("step") != step_name:
            continue
        metadata = step.get("metadata") or {}
        latency = metadata.get("latency_s")
        return float(latency) if latency is not None else None
    return None


def _is_openai_api_error(exc: Exception) -> bool:
    if _is_named_openai_error(exc):
        return True

    cause = exc.__cause__
    while cause is not None:
        if _is_named_openai_error(cause):
            return True
        cause = cause.__cause__

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "insufficient_quota",
            "rate limit",
            "ratelimiterror",
            "apiconnectionerror",
            "connection error",
        )
    )


def _is_named_openai_error(exc: BaseException) -> bool:
    error_name = type(exc).__name__
    return error_name in {
        "RateLimitError",
        "APIConnectionError",
        "APIStatusError",
        "OpenAIError",
    }


def dataframe_to_markdown(dataframe) -> str:
    """Render a small dataframe as a GitHub-flavored markdown table."""

    columns = list(dataframe.columns)
    rows = [columns]
    for _, row in dataframe.iterrows():
        rows.append([_format_markdown_cell(row[column]) for column in columns])

    widths = [
        max(len(str(row[index])) for row in rows)
        for index in range(len(columns))
    ]
    lines = [
        "| "
        + " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(rows[0])
        )
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rows[1:]:
        lines.append(
            "| "
            + " | ".join(
                str(value).ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )

    return "\n".join(lines)


def _format_markdown_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "The pandas package is required for evaluation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return pd
