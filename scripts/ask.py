"""Ask a question against the local documentation vector store."""

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.rag_chain import answer_question


RETRIEVER_MATCH_LABELS = {
    "dense": "dense",
    "bm25": "BM25",
    "both": "both",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a documentation question.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunk details for retrieval quality inspection.",
    )
    parser.add_argument(
        "--retriever",
        choices=("dense", "hybrid"),
        default="dense",
        help="Retriever strategy to use. Defaults to dense.",
    )
    parser.add_argument("question", nargs="+", help="Question to ask the RAG system.")
    return parser.parse_args()


def print_debug_info(documents: list[Document], retriever_name: str) -> None:
    """Print retrieved chunk details for debugging retrieval quality."""

    chunking_strategies = sorted(
        {
            document.metadata.get("chunking_strategy", "unknown")
            for document in documents
        }
    )
    chunking_strategy_label = (
        ", ".join(chunking_strategies) if chunking_strategies else "unknown"
    )

    print("\nDebug:")
    print(f"Retriever: {retriever_name}")
    print(f"Chunking strategy: {chunking_strategy_label}")
    print(f"Retrieved chunks: {len(documents)}")

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        preview = document.page_content[:500].strip()

        print(f"\nChunk {index}:")
        print(f"Source: {metadata.get('source', 'unknown')}")

        title = metadata.get("title")
        if title:
            print(f"Title: {title}")

        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            print(f"Chunk index: {chunk_index}")

        print(f"Chunking strategy: {metadata.get('chunking_strategy', 'unknown')}")

        retriever_match = metadata.get("retriever_match", retriever_name)
        retriever_match_label = RETRIEVER_MATCH_LABELS.get(
            retriever_match,
            retriever_match,
        )
        print(f"Retriever match: {retriever_match_label}")
        print("Preview:")
        print(preview)


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()

    result = answer_question(question, retriever_name=args.retriever)

    print("\nAnswer:\n")
    print(result.answer.strip())

    print("\nSources:")
    if result.sources:
        for source in result.sources:
            print(f"- {source}")
    else:
        print("- No sources retrieved. Run scripts/ingest_docs.py first.")

    if args.debug:
        print_debug_info(result.documents, result.retriever_name)


if __name__ == "__main__":
    main()
