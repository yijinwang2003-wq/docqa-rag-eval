"""Generate and clean a synthetic RAG evaluation set."""

from pathlib import Path
import argparse
import inspect
import json
import math
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from src.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT  # noqa: E402
from src.loaders import CORE_DOC_URLS  # noqa: E402
from src.rag_chain import get_llm  # noqa: E402
from src.vectorstore import get_embeddings, get_persisted_documents  # noqa: E402


DEFAULT_RAW_PATH = CONFIG_PROJECT_ROOT / "data" / "eval_questions_raw.json"
DEFAULT_CLEAN_PATH = CONFIG_PROJECT_ROOT / "data" / "eval_questions.json"
DEFAULT_REVIEW_PATH = CONFIG_PROJECT_ROOT / "data" / "eval_questions_review.md"
DEFAULT_MAX_DOCS = 120
MIN_FINAL_QUESTIONS = 40
VAGUE_QUESTIONS = {
    "what is this",
    "what is this?",
    "what does this mean",
    "what does this mean?",
    "how does it work",
    "how does it work?",
    "what are they",
    "what are they?",
    "explain this",
    "explain this.",
}
CONCEPTUAL_PATH_HINTS = (
    "/rag",
    "/overview",
    "/prompt",
    "/messages",
    "/splitter",
    "/vectorstores/chroma",
    "/build-with-claude",
    "/api/messages",
)
FALLBACK_QUESTION_TYPES = (
    "definition",
    "comparison",
    "procedural",
    "implementation detail",
    "failure analysis",
    "prompt grounding",
    "API-specific",
)
FALLBACK_BATCH_SIZE = 6
MAX_FALLBACK_CONTEXT_CHARS = 12000
MAX_FALLBACK_DOCUMENT_CHARS = 1800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation questions from Chroma docs."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of synthetic questions to request.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help="Maximum persisted chunks to pass to generation.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path for raw generated records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CLEAN_PATH,
        help="Path for cleaned evaluation questions JSON.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=DEFAULT_REVIEW_PATH,
        help="Path for manual review markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = select_generation_documents(
        get_persisted_documents(),
        max_docs=args.max_docs,
    )
    if not documents:
        raise RuntimeError(
            "No persisted Chroma documents found. Run scripts/ingest_docs.py first."
        )

    print(f"Selected source chunks for generation: {len(documents)}")
    generated_records = generate_testset_records(documents, testset_size=args.n)
    raw_records = normalize_generated_records(generated_records)
    cleaned_records = clean_eval_records(raw_records, max_records=args.n)

    write_json(args.raw_output, raw_records)
    write_json(args.output, cleaned_records)
    write_review_markdown(args.review_output, cleaned_records)

    print(f"Raw generated records: {len(raw_records)} -> {args.raw_output}")
    print(f"Cleaned review-ready records: {len(cleaned_records)} -> {args.output}")
    print(f"Manual review checklist: {args.review_output}")

    if len(cleaned_records) < MIN_FINAL_QUESTIONS:
        print(
            "Warning: fewer than 40 questions survived filtering. "
            "Increase --n or review the raw output."
        )


def select_generation_documents(
    documents: list[Document],
    max_docs: int = DEFAULT_MAX_DOCS,
) -> list[Document]:
    """Prefer core conceptual documentation chunks for synthetic generation."""

    non_empty_documents = [
        document for document in documents if document.page_content.strip()
    ]
    ranked_documents = sorted(
        non_empty_documents,
        key=lambda document: _document_quality_rank(document),
    )
    selected = ranked_documents[:max_docs]

    return [_clean_generation_document(document) for document in selected]


def generate_testset_records(
    documents: list[Document],
    testset_size: int,
) -> list[dict]:
    """Generate raw eval records, preferring RAGAS and falling back to OpenAI."""

    try:
        records = _generate_ragas_testset_records(documents, testset_size)
    except Exception as exc:
        print("RAGAS generation unavailable; using OpenAI fallback generator")
        print(f"RAGAS failure: {type(exc).__name__}: {exc}")
        return generate_openai_fallback_records(documents, testset_size)

    print("Using RAGAS TestsetGenerator")
    return records


def _generate_ragas_testset_records(
    documents: list[Document],
    testset_size: int,
) -> list[dict]:
    """Generate raw records from RAGAS TestsetGenerator."""

    TestsetGenerator = _import_testset_generator()
    generator = _make_testset_generator(TestsetGenerator)
    generate_candidates = {
        "documents": documents,
        "testset_size": testset_size,
        "raise_exceptions": False,
    }
    generate_kwargs = _supported_kwargs(
        generator.generate_with_langchain_docs,
        generate_candidates,
    )

    if {"documents", "testset_size"}.issubset(generate_kwargs):
        testset = generator.generate_with_langchain_docs(**generate_kwargs)
    else:
        positional_kwargs = {
            key: value
            for key, value in generate_kwargs.items()
            if key not in {"documents", "testset_size"}
        }
        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size,
            **positional_kwargs,
        )

    return _testset_to_records(testset)


def generate_openai_fallback_records(
    documents: list[Document],
    testset_size: int,
) -> list[dict]:
    """Generate question/ground_truth pairs directly with the configured LLM."""

    llm = get_llm()
    records = []
    batches = _fallback_document_batches(documents)
    if not batches:
        return records

    records_per_batch = max(1, math.ceil(testset_size / len(batches)))

    for batch_index, batch in enumerate(batches):
        remaining = testset_size - len(records)
        if remaining <= 0:
            break

        batch_records = _generate_openai_batch_records(
            llm=llm,
            documents=batch,
            batch_index=batch_index,
            target_count=min(records_per_batch, remaining),
        )
        records.extend(batch_records)

    return records[:testset_size]


def _fallback_document_batches(documents: list[Document]) -> list[list[Document]]:
    return [
        documents[index : index + FALLBACK_BATCH_SIZE]
        for index in range(0, len(documents), FALLBACK_BATCH_SIZE)
    ]


def _generate_openai_batch_records(
    llm,
    documents: list[Document],
    batch_index: int,
    target_count: int,
) -> list[dict]:
    prompt = _build_openai_fallback_prompt(
        documents=documents,
        batch_index=batch_index,
        target_count=target_count,
    )
    response = llm.invoke(prompt)
    response_text = _llm_response_text(response)
    parsed_records = _parse_json_records(response_text)

    return [
        {
            "question": _normalize_whitespace(
                _extract_first(record, "question", "user_input", "query")
            ),
            "ground_truth": _normalize_whitespace(
                _extract_first(
                    record,
                    "ground_truth",
                    "reference",
                    "answer",
                    "expected_output",
                )
            ),
        }
        for record in parsed_records
    ]


def _build_openai_fallback_prompt(
    documents: list[Document],
    batch_index: int,
    target_count: int,
) -> str:
    question_types = _fallback_question_types_for_batch(batch_index, target_count)
    context = _format_fallback_context(documents)

    return f"""You are creating a synthetic RAG evaluation set from technical documentation chunks.

Generate exactly {target_count} question/ground_truth pairs from the provided context.
Use these question types in order, one per item: {", ".join(question_types)}.

Requirements:
- Questions must be answerable from the provided context only.
- Ground truths must be concise but complete, and must not mention "the context" or "the document".
- Prefer concrete technical questions over vague summary questions.
- Include a mix of API names, implementation details, procedural steps, comparisons, failure modes, and prompt-grounding behavior when supported by the context.
- Return only a JSON array with objects in this exact schema:
  [{{"question": "...", "ground_truth": "..."}}]

Context:
{context}
"""


def _fallback_question_types_for_batch(
    batch_index: int,
    target_count: int,
) -> list[str]:
    return [
        FALLBACK_QUESTION_TYPES[
            (batch_index * target_count + offset) % len(FALLBACK_QUESTION_TYPES)
        ]
        for offset in range(target_count)
    ]


def _format_fallback_context(documents: list[Document]) -> str:
    chunks = []
    total_chars = 0

    for index, document in enumerate(documents, start=1):
        content = _truncate_text(
            _normalize_whitespace(document.page_content),
            MAX_FALLBACK_DOCUMENT_CHARS,
        )
        metadata = document.metadata
        source = metadata.get("source") or metadata.get("url") or "unknown"
        title = metadata.get("title") or ""
        chunk = (
            f"[Chunk {index}]\n"
            f"Source: {source}\n"
            f"Title: {title}\n"
            f"Content: {content}"
        )

        if total_chars + len(chunk) > MAX_FALLBACK_CONTEXT_CHARS:
            break

        chunks.append(chunk)
        total_chars += len(chunk)

    return "\n\n".join(chunks)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    truncated = value[:max_chars].rsplit(" ", maxsplit=1)[0]
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


def _parse_json_records(value: str) -> list[dict]:
    cleaned_value = _strip_markdown_fence(value.strip())
    try:
        parsed = json.loads(cleaned_value)
    except json.JSONDecodeError:
        parsed = json.loads(_extract_json_array(cleaned_value))

    if not isinstance(parsed, list):
        raise ValueError("OpenAI fallback response must be a JSON array.")

    return [item for item in parsed if isinstance(item, dict)]


def _strip_markdown_fence(value: str) -> str:
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    return value


def _extract_json_array(value: str) -> str:
    start = value.find("[")
    end = value.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("OpenAI fallback response did not contain a JSON array.")

    return value[start : end + 1]


def clean_eval_records(records: list[dict], max_records: int = 50) -> list[dict]:
    """Filter duplicate, underspecified, or incomplete generated questions."""

    cleaned_records = []
    seen_questions = set()

    for record in records:
        question = _extract_first(record, "question", "user_input", "query")
        ground_truth = _extract_first(
            record,
            "ground_truth",
            "reference",
            "answer",
            "expected_output",
        )
        question = _normalize_whitespace(question)
        ground_truth = _normalize_whitespace(ground_truth)
        question_key = _question_key(question)

        if not question or not ground_truth:
            continue
        if question_key in seen_questions:
            continue
        if _is_too_short_question(question):
            continue
        if _is_vague_question(question):
            continue

        seen_questions.add(question_key)
        cleaned_records.append(
            {
                "question": question,
                "ground_truth": ground_truth,
            }
        )

        if len(cleaned_records) >= max_records:
            break

    return cleaned_records


def normalize_generated_records(records: list[dict]) -> list[dict]:
    """Convert RAGAS records into the project eval question schema."""

    normalized_records = []

    for record in records:
        question = _extract_first(record, "question", "user_input", "query")
        ground_truth = _extract_first(
            record,
            "ground_truth",
            "reference",
            "answer",
            "expected_output",
        )
        normalized_records.append(
            {
                "question": _normalize_whitespace(question),
                "ground_truth": _normalize_whitespace(ground_truth),
            }
        )

    return normalized_records


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_review_markdown(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Question Review",
        "",
        "Review each generated item before using it for portfolio results.",
        "",
    ]

    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index}. {record['question']}",
                "",
                f"Ground truth: {record['ground_truth']}",
                "",
                "- [ ] keep",
                "- [ ] revise",
                "- [ ] drop",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _document_quality_rank(document: Document) -> tuple[int, int, int, str]:
    metadata = document.metadata
    source = str(metadata.get("source", ""))
    source_without_fragment = source.split("#", maxsplit=1)[0]
    is_core = source_without_fragment in CORE_DOC_URLS
    is_conceptual = any(hint in source.lower() for hint in CONCEPTUAL_PATH_HINTS)
    content_length = len(document.page_content)

    return (
        0 if is_core else 1,
        0 if is_conceptual else 1,
        0 if 400 <= content_length <= 2500 else 1,
        source,
    )


def _clean_generation_document(document: Document) -> Document:
    metadata = dict(document.metadata)
    metadata.pop("chroma_id", None)
    return Document(page_content=document.page_content.strip(), metadata=metadata)


def _make_testset_generator(TestsetGenerator):
    from_langchain = TestsetGenerator.from_langchain
    kwargs = _supported_kwargs(
        from_langchain,
        {
            "llm": get_llm(),
            "embedding_model": get_embeddings(),
        },
    )
    return from_langchain(**kwargs)


def _testset_to_records(testset) -> list[dict]:
    to_pandas = getattr(testset, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas().to_dict(orient="records")

    to_dataset = getattr(testset, "to_dataset", None)
    if callable(to_dataset):
        dataset = to_dataset()
        to_list = getattr(dataset, "to_list", None)
        if callable(to_list):
            return to_list()
        return list(dataset)

    samples = getattr(testset, "samples", None)
    if samples is not None:
        return [_object_to_record(sample) for sample in samples]

    return [_object_to_record(item) for item in testset]


def _object_to_record(item) -> dict:
    if isinstance(item, dict):
        return dict(item)

    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    if hasattr(item, "__dict__"):
        return dict(item.__dict__)

    return {"value": str(item)}


def _supported_kwargs(function, candidates: dict) -> dict:
    signature = inspect.signature(function)
    return {
        name: value
        for name, value in candidates.items()
        if name in signature.parameters
    }


def _extract_first(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value)

    return ""


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _question_key(question: str) -> str:
    normalized = question.lower()
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return " ".join(normalized.split())


def _is_too_short_question(question: str) -> bool:
    return len(question.split()) < 5 or len(question) < 24


def _is_vague_question(question: str) -> bool:
    normalized = _question_key(question)
    if normalized in {_question_key(item) for item in VAGUE_QUESTIONS}:
        return True

    vague_markers = (
        "this document",
        "this page",
        "the text",
        "the context",
        "the information",
    )
    return any(marker in normalized for marker in vague_markers)


def _import_testset_generator():
    try:
        from ragas.testset import TestsetGenerator
    except ImportError as first_exc:
        try:
            from ragas.testset.generator import TestsetGenerator
        except ImportError as second_exc:
            raise RuntimeError(
                "RAGAS TestsetGenerator could not be imported. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from second_exc
        return TestsetGenerator
    except Exception:
        raise

    return TestsetGenerator


if __name__ == "__main__":
    main()
