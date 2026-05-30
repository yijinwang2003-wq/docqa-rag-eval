"""Evaluation helpers for comparing RAG configurations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import json
import re
import signal
import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.chunking import FIXED_CHUNKING, SEMANTIC_CHUNKING
from src.config import (
    OPENAI_EVAL_MODEL,
    PROJECT_ROOT,
    RETRIEVER_K,
    require_openai_api_key,
)
from src.rag_chain import HybridRetriever, PROMPT, format_docs, get_llm, unique_sources
from src.vectorstore import get_persisted_documents, get_vectorstore


DENSE_RETRIEVER = "dense"
HYBRID_RETRIEVER = "hybrid"
DEFAULT_CONFIGS = (
    (FIXED_CHUNKING, DENSE_RETRIEVER),
    (FIXED_CHUNKING, HYBRID_RETRIEVER),
    (SEMANTIC_CHUNKING, DENSE_RETRIEVER),
    (SEMANTIC_CHUNKING, HYBRID_RETRIEVER),
)
DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval_questions.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval_results.csv"
DEFAULT_PARTIAL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "eval_results_partial.csv"
ANSWER_CACHE_PATH = PROJECT_ROOT / "outputs" / "cache" / "eval_answers.json"
JUDGE_CACHE_PATH = PROJECT_ROOT / "outputs" / "cache" / "eval_judges.json"
LATENCY_COLUMNS = (
    "retrieval_latency_s",
    "generation_latency_s",
    "total_latency_s",
)
LLM_JUDGE_TIMEOUT_SECONDS = 60
LLM_JUDGE_MAX_ATTEMPTS = 3
_RAGAS_UNAVAILABLE = False


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    ground_truth: str


@dataclass(frozen=True)
class RagRun:
    chunking: str
    retriever: str
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    sources: list[str]
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float

    @property
    def configuration(self) -> str:
        return f"{self.chunking}+{self.retriever}"


def load_eval_questions(path: Path = DEFAULT_DATASET_PATH) -> list[EvalQuestion]:
    """Load evaluation questions from JSON."""

    with path.open(encoding="utf-8") as file:
        raw_questions = json.load(file)

    questions = []
    for index, item in enumerate(raw_questions, start=1):
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()

        if not question or not ground_truth:
            raise ValueError(
                f"Evaluation item {index} must include question and ground_truth."
            )

        questions.append(EvalQuestion(question=question, ground_truth=ground_truth))

    return questions


def run_rag_configurations(
    questions: Sequence[EvalQuestion],
    configurations: Sequence[tuple[str, str]] = DEFAULT_CONFIGS,
) -> list[RagRun]:
    """Run retrieval and answer generation for each evaluation configuration."""

    runs = []
    llm = get_llm()
    answer_chain = PROMPT | llm | StrOutputParser()
    total_questions = len(questions)

    for chunking, retriever_name in configurations:
        print(f"Current configuration: {chunking}+{retriever_name}", flush=True)
        retriever = build_configured_retriever(chunking, retriever_name)

        for question_index, item in enumerate(questions, start=1):
            print(
                f"Question {question_index}/{total_questions}: {item.question}",
                flush=True,
            )
            retrieval_start = time.perf_counter()
            documents = list(retriever.invoke(item.question))
            retrieval_latency_s = time.perf_counter() - retrieval_start
            generation_start = time.perf_counter()
            answer = answer_chain.invoke(
                {
                    "question": item.question,
                    "context": format_docs(documents),
                }
            )
            generation_latency_s = time.perf_counter() - generation_start
            runs.append(
                RagRun(
                    chunking=chunking,
                    retriever=retriever_name,
                    question=item.question,
                    ground_truth=item.ground_truth,
                    answer=answer,
                    contexts=[document.page_content for document in documents],
                    sources=unique_sources(documents),
                    retrieval_latency_s=retrieval_latency_s,
                    generation_latency_s=generation_latency_s,
                    total_latency_s=retrieval_latency_s + generation_latency_s,
                )
            )

    return runs


def build_configured_retriever(chunking: str, retriever_name: str) -> BaseRetriever:
    """Create a retriever scoped to one chunking strategy."""

    if retriever_name == DENSE_RETRIEVER:
        return _get_dense_retriever(chunking)
    if retriever_name == HYBRID_RETRIEVER:
        return HybridRetriever(
            dense_retriever=_get_dense_retriever(chunking),
            bm25_retriever=_get_bm25_retriever(chunking),
            k=RETRIEVER_K,
        )

    raise ValueError(f"Unsupported retriever: {retriever_name}")


def evaluate_runs(
    runs: Sequence[RagRun],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metrics: Sequence[str] = DEFAULT_METRICS,
    judge_cache: dict | None = None,
    write_output: bool = True,
):
    """Evaluate completed RAG runs and save per-row scores."""

    pd = _import_pandas()
    try:
        results_df = _evaluate_runs_with_ragas(runs, metrics, pd)
    except Exception:
        global _RAGAS_UNAVAILABLE
        _RAGAS_UNAVAILABLE = True
        print("RAGAS unavailable; using LLM fallback evaluator")
        results_df = _evaluate_runs_with_llm_fallback(
            runs,
            metrics,
            pd,
            judge_cache=judge_cache,
        )

    metadata_df = pd.DataFrame(_runs_to_records(runs))

    for column in metadata_df.columns:
        if column not in results_df.columns:
            results_df.insert(len(results_df.columns), column, metadata_df[column])

    ordered_columns = [
        "configuration",
        "chunking",
        "retriever",
        "question",
        "ground_truth",
        "answer",
        "contexts",
        "sources",
        *LATENCY_COLUMNS,
        *metrics,
    ]
    available_columns = [column for column in ordered_columns if column in results_df]
    results_df = results_df[available_columns]

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
    return results_df


def summarize_results(results_df, metrics: Sequence[str] = DEFAULT_METRICS):
    """Return mean metric values per configuration."""

    summary_columns = [
        column
        for column in (*metrics, *LATENCY_COLUMNS)
        if column in results_df.columns
    ]
    if not summary_columns:
        return results_df.groupby("configuration", as_index=False).size()

    return (
        results_df.groupby("configuration", as_index=False)[summary_columns]
        .mean(numeric_only=True)
        .sort_values("configuration")
    )


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    configurations: Sequence[tuple[str, str]] = DEFAULT_CONFIGS,
    metrics: Sequence[str] = DEFAULT_METRICS,
    limit: int | None = None,
    partial_output_path: Path = DEFAULT_PARTIAL_OUTPUT_PATH,
):
    """Load eval questions, run all configs, score, and summarize."""

    questions = load_eval_questions(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be greater than zero.")
        questions = questions[:limit]

    results_df = run_incremental_evaluation(
        questions=questions,
        configurations=configurations,
        output_path=output_path,
        partial_output_path=partial_output_path,
        metrics=metrics,
    )
    summary_df = summarize_results(results_df, metrics=metrics)
    return results_df, summary_df


def run_incremental_evaluation(
    questions: Sequence[EvalQuestion],
    configurations: Sequence[tuple[str, str]],
    output_path: Path,
    partial_output_path: Path,
    metrics: Sequence[str],
):
    """Run and score one config/question at a time, saving partial results."""

    pd = _import_pandas()
    existing_df = _load_partial_results(partial_output_path, pd)
    planned_keys = _planned_result_keys(questions, configurations)
    existing_df = _filter_partial_results(existing_df, planned_keys)
    existing_df = _filter_completed_results(existing_df, metrics, LATENCY_COLUMNS)
    completed_keys = _completed_result_keys(existing_df, metrics, LATENCY_COLUMNS)
    result_frames = [existing_df] if not existing_df.empty else []
    answer_cache = _load_json_cache(ANSWER_CACHE_PATH)
    judge_cache = _load_json_cache(JUDGE_CACHE_PATH)
    llm = get_llm()
    answer_chain = PROMPT | llm | StrOutputParser()
    total_questions = len(questions)

    for chunking, retriever_name in configurations:
        print(f"Current configuration: {chunking}+{retriever_name}", flush=True)
        retriever = build_configured_retriever(chunking, retriever_name)

        for question_index, item in enumerate(questions, start=1):
            print(
                f"Question {question_index}/{total_questions}: {item.question}",
                flush=True,
            )
            row_key = _result_key(chunking, retriever_name, item)
            if row_key in completed_keys:
                print("Skipping existing partial result.", flush=True)
                continue

            total_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            documents = list(retriever.invoke(item.question))
            retrieval_latency_s = time.perf_counter() - retrieval_start
            generation_start = time.perf_counter()
            answer = _get_cached_answer(
                answer_chain=answer_chain,
                cache=answer_cache,
                chunking=chunking,
                retriever_name=retriever_name,
                item=item,
                documents=documents,
            )
            generation_latency_s = time.perf_counter() - generation_start
            total_latency_s = time.perf_counter() - total_start
            run = RagRun(
                chunking=chunking,
                retriever=retriever_name,
                question=item.question,
                ground_truth=item.ground_truth,
                answer=answer,
                contexts=[document.page_content for document in documents],
                sources=unique_sources(documents),
                retrieval_latency_s=retrieval_latency_s,
                generation_latency_s=generation_latency_s,
                total_latency_s=total_latency_s,
            )
            result_frames.append(
                evaluate_runs(
                    [run],
                    output_path=partial_output_path,
                    metrics=metrics,
                    judge_cache=judge_cache,
                    write_output=False,
                )
            )
            partial_df = pd.concat(result_frames, ignore_index=True)
            partial_output_path.parent.mkdir(parents=True, exist_ok=True)
            partial_df.to_csv(partial_output_path, index=False)
            completed_keys.add(row_key)

    if result_frames:
        results_df = pd.concat(result_frames, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df


def _load_partial_results(path: Path, pd):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _planned_result_keys(
    questions: Sequence[EvalQuestion],
    configurations: Sequence[tuple[str, str]],
) -> set[str]:
    return {
        _result_key(chunking, retriever_name, item)
        for chunking, retriever_name in configurations
        for item in questions
    }


def _filter_partial_results(results_df, planned_keys: set[str]):
    if results_df.empty:
        return results_df

    return results_df[
        results_df.apply(lambda row: _result_key_from_row(row) in planned_keys, axis=1)
    ].copy()


def _completed_result_keys(
    results_df,
    metrics: Sequence[str],
    required_columns: Sequence[str] = (),
) -> set[str]:
    if results_df.empty:
        return set()

    completed_keys = set()
    for _, row in results_df.iterrows():
        if all(
            column in row and _has_value(row[column])
            for column in (*metrics, *required_columns)
        ):
            completed_keys.add(_result_key_from_row(row))

    return completed_keys


def _filter_completed_results(
    results_df,
    metrics: Sequence[str],
    required_columns: Sequence[str] = (),
):
    if results_df.empty:
        return results_df

    return results_df[
        results_df.apply(
            lambda row: all(
                column in row and _has_value(row[column])
                for column in (*metrics, *required_columns)
            ),
            axis=1,
        )
    ].copy()


def _has_value(value) -> bool:
    try:
        return not bool(_import_pandas().isna(value))
    except (TypeError, ValueError):
        return value is not None


def _result_key(chunking: str, retriever_name: str, item: EvalQuestion) -> str:
    return _hash_parts(chunking, retriever_name, item.question, item.ground_truth)


def _result_key_from_row(row) -> str:
    return _hash_parts(
        str(row.get("chunking", "")),
        str(row.get("retriever", "")),
        str(row.get("question", "")),
        str(row.get("ground_truth", "")),
    )


def _get_cached_answer(
    answer_chain,
    cache: dict,
    chunking: str,
    retriever_name: str,
    item: EvalQuestion,
    documents: Sequence[Document],
) -> str:
    context = format_docs(documents)
    cache_key = _answer_cache_key(
        chunking=chunking,
        retriever_name=retriever_name,
        item=item,
        documents=documents,
    )
    cached_answer = cache.get(cache_key)
    if cached_answer is not None:
        return str(cached_answer)

    answer = answer_chain.invoke(
        {
            "question": item.question,
            "context": context,
        }
    )
    cache[cache_key] = answer
    _write_json_cache(ANSWER_CACHE_PATH, cache)
    return answer


def _answer_cache_key(
    chunking: str,
    retriever_name: str,
    item: EvalQuestion,
    documents: Sequence[Document],
) -> str:
    return _hash_parts(
        chunking,
        retriever_name,
        item.question,
        item.ground_truth,
        _documents_fingerprint(documents),
    )


def _judge_cache_key(run: RagRun, metric: str, prompt: str) -> str:
    return _hash_parts(
        OPENAI_EVAL_MODEL,
        run.configuration,
        run.question,
        run.ground_truth,
        run.answer,
        metric,
        prompt,
    )


def _documents_fingerprint(documents: Sequence[Document]) -> str:
    document_keys = []
    for document in documents:
        metadata = document.metadata
        document_keys.append(
            _hash_parts(
                str(metadata.get("chroma_id", "")),
                str(metadata.get("source", "")),
                str(metadata.get("chunk_index", "")),
                document.page_content,
            )
        )

    return _hash_parts(*document_keys)


def _hash_parts(*parts: str) -> str:
    raw = "\n".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file)
        file.write("\n")


def _get_dense_retriever(chunking: str) -> BaseRetriever:
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": RETRIEVER_K,
            "filter": {"chunking_strategy": chunking},
        }
    )


def _get_bm25_retriever(chunking: str):
    from langchain_community.retrievers import BM25Retriever

    documents = _get_documents_for_chunking(chunking)
    if not documents:
        raise RuntimeError(
            f"No persisted documents found for chunking strategy '{chunking}'. "
            "Ingest docs with scripts/ingest_docs.py for each chunking strategy first."
        )

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = RETRIEVER_K
    return retriever


def _get_documents_for_chunking(chunking: str) -> list[Document]:
    return [
        document
        for document in get_persisted_documents()
        if document.metadata.get("chunking_strategy") == chunking
    ]


def _make_ragas_dataset(runs: Sequence[RagRun]):
    Dataset = _import_dataset()
    records = _runs_to_records(runs)
    return Dataset.from_list(records)


def _evaluate_runs_with_ragas(runs: Sequence[RagRun], metrics: Sequence[str], pd):
    global _RAGAS_UNAVAILABLE

    if _RAGAS_UNAVAILABLE:
        raise RuntimeError("RAGAS previously failed in this process.")

    dataset = _make_ragas_dataset(runs)
    ragas_metrics = _load_ragas_metrics(metrics)
    evaluate = _import_ragas_evaluate()
    for metric in metrics:
        print(f"Judging metric: {metric}", flush=True)

    try:
        result = evaluate(
            dataset,
            metrics=ragas_metrics,
            raise_exceptions=False,
            show_progress=True,
        )
        return _evaluation_result_to_dataframe(result, pd)
    except Exception:
        _RAGAS_UNAVAILABLE = True
        raise


def _evaluate_runs_with_llm_fallback(
    runs: Sequence[RagRun],
    metrics: Sequence[str],
    pd,
    judge_cache: dict | None = None,
):
    judge_cache = judge_cache if judge_cache is not None else _load_json_cache(JUDGE_CACHE_PATH)
    llm = get_eval_llm()
    rows = []

    for run in runs:
        row = {}
        for metric in metrics:
            print(f"Judging metric: {metric}", flush=True)
            row[metric] = _score_run_with_llm(llm, run, metric, judge_cache)
        rows.append(row)

    return pd.DataFrame(rows)


def get_eval_llm() -> ChatOpenAI:
    """Create the lower-cost model used for LLM fallback judging."""

    require_openai_api_key()
    return ChatOpenAI(model=OPENAI_EVAL_MODEL)


def _score_run_with_llm(
    llm,
    run: RagRun,
    metric: str,
    judge_cache: dict | None = None,
) -> float:
    prompt = _build_metric_judge_prompt(run, metric)
    cache_key = _judge_cache_key(run, metric, prompt)
    judge_cache = judge_cache if judge_cache is not None else _load_json_cache(JUDGE_CACHE_PATH)
    cached_response = judge_cache.get(cache_key)
    if cached_response is not None:
        return _parse_judge_score(cached_response)

    response = _invoke_llm_judge_with_retry(llm, prompt)
    response_text = _llm_response_text(response)
    judge_cache[cache_key] = response_text
    _write_json_cache(JUDGE_CACHE_PATH, judge_cache)
    return _parse_judge_score(response_text)


def _invoke_llm_judge_with_retry(llm, prompt: str):
    last_exc = None

    for attempt in range(1, LLM_JUDGE_MAX_ATTEMPTS + 1):
        try:
            return _invoke_with_timeout(
                lambda: llm.invoke(prompt),
                timeout_seconds=LLM_JUDGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= LLM_JUDGE_MAX_ATTEMPTS:
                break
            print(
                "LLM judge call failed "
                f"(attempt {attempt}/{LLM_JUDGE_MAX_ATTEMPTS}): "
                f"{type(exc).__name__}: {exc}. Retrying...",
                flush=True,
            )

    raise RuntimeError(
        f"LLM judge call failed after {LLM_JUDGE_MAX_ATTEMPTS} attempts."
    ) from last_exc


def _invoke_with_timeout(function, timeout_seconds: int):
    if not hasattr(signal, "SIGALRM"):
        return function()

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"LLM judge call exceeded {timeout_seconds} seconds.")

    try:
        previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    except ValueError:
        return function()
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_metric_judge_prompt(run: RagRun, metric: str) -> str:
    metric_instruction = _fallback_metric_instruction(metric)
    contexts = _format_judge_contexts(run.contexts)

    return f"""You are judging one RAG benchmark result.

Metric: {metric}
Scoring instruction: {metric_instruction}

Return only JSON in this exact schema:
{{"score": 0.0}}

Use a score from 0 to 1, where 1 is best. Use intermediate decimals when the result is partially correct.

Question:
{run.question}

Ground truth:
{run.ground_truth}

Answer:
{run.answer}

Retrieved contexts:
{contexts}
"""


def _fallback_metric_instruction(metric: str) -> str:
    if metric == "answer_relevancy":
        return (
            "Score whether the answer directly and completely addresses the "
            "question, regardless of whether it is factually supported."
        )
    if metric == "faithfulness":
        return (
            "Score whether the answer is supported by the retrieved contexts. "
            "Penalize claims that are missing from or contradicted by the contexts."
        )
    if metric == "context_precision":
        return (
            "Score whether the retrieved contexts contain clear evidence for the "
            "ground truth. Penalize irrelevant or insufficient contexts."
        )

    raise ValueError(f"Unsupported fallback metric: {metric}")


def _format_judge_contexts(contexts: Sequence[str]) -> str:
    if not contexts:
        return "(none)"

    return "\n\n".join(
        f"[Context {index}]\n{_truncate_for_judge(context)}"
        for index, context in enumerate(contexts, start=1)
    )


def _truncate_for_judge(value: str, max_chars: int = 2500) -> str:
    normalized = " ".join(str(value).strip().split())
    if len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rsplit(" ", maxsplit=1)[0]
    return f"{truncated}..."


def _llm_response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part)

    return str(content)


def _parse_judge_score(value: str) -> float:
    cleaned_value = _strip_markdown_fence(value.strip())
    try:
        parsed = json.loads(cleaned_value)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_extract_json_object(cleaned_value))
        except ValueError:
            parsed = _extract_first_float(cleaned_value)

    if isinstance(parsed, dict):
        score = parsed.get("score")
    else:
        score = parsed

    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LLM judge response did not include a numeric score: {value}"
        ) from exc


def _strip_markdown_fence(value: str) -> str:
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    return value


def _extract_json_object(value: str) -> str:
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM judge response did not contain a JSON object.")

    return value[start : end + 1]


def _extract_first_float(value: str) -> float:
    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", value)
    if not match:
        raise ValueError("LLM judge response did not contain a score.")

    return float(match.group(0))


def _runs_to_records(runs: Iterable[RagRun]) -> list[dict]:
    return [
        {
            "configuration": run.configuration,
            "chunking": run.chunking,
            "retriever": run.retriever,
            "question": run.question,
            "user_input": run.question,
            "ground_truth": run.ground_truth,
            "reference": run.ground_truth,
            "answer": run.answer,
            "response": run.answer,
            "contexts": run.contexts,
            "retrieved_contexts": run.contexts,
            "sources": run.sources,
            "retrieval_latency_s": run.retrieval_latency_s,
            "generation_latency_s": run.generation_latency_s,
            "total_latency_s": run.total_latency_s,
        }
        for run in runs
    ]


def _load_ragas_metrics(metric_names: Sequence[str]):
    ragas_metrics = importlib.import_module("ragas.metrics")

    metric_objects = []
    for metric_name in metric_names:
        try:
            metric_objects.append(getattr(ragas_metrics, metric_name))
        except AttributeError as exc:
            raise ValueError(f"Unsupported RAGAS metric: {metric_name}") from exc

    return metric_objects


def _evaluation_result_to_dataframe(result, pd):
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()

    scores = getattr(result, "scores", None)
    if scores is not None:
        return pd.DataFrame(scores)

    return pd.DataFrame(result)


def _import_dataset():
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for evaluation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return Dataset


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "The pandas package is required for evaluation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return pd


def _import_ragas_evaluate():
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "The ragas package is required for evaluation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return evaluate
