"""Load documentation pages, chunk them, and persist embeddings in Chroma."""

from pathlib import Path
import argparse
from collections import Counter
import logging
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import CHUNKING_STRATEGIES, chunk_documents
from src.config import CHROMA_COLLECTION, CHROMA_DIR
from src.loaders import (
    CORE_DOC_URLS,
    DEFAULT_DOC_URLS,
    DEFAULT_SITEMAP_URLS,
    get_sitemap_urls,
    load_documents_with_failures,
)
from src.vectorstore import add_documents, reset_vectorstore


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest docs into local ChromaDB.")
    parser.add_argument(
        "--source",
        choices=("sample", "sitemap"),
        default="sample",
        help="Use the sample URL list or discover URLs from configured sitemaps.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum sitemap URLs to ingest when --source sitemap is used.",
    )
    parser.add_argument(
        "--sitemap-url",
        action="append",
        dest="sitemap_urls",
        help="Sitemap URL to ingest. Repeat for multiple sitemaps.",
    )
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="Documentation URL to ingest. Repeat for multiple URLs. Overrides --source.",
    )
    parser.add_argument(
        "--chunking",
        choices=CHUNKING_STRATEGIES,
        default="fixed",
        help="Chunking strategy to use before embedding. Defaults to fixed.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma collection before ingesting.",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _dedupe_urls(urls: list[str]) -> list[str]:
    unique_urls = []
    seen = set()

    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)

    return unique_urls


def resolve_urls(args: argparse.Namespace) -> list[str]:
    if args.urls:
        logger.info("URLs loaded from CLI arguments: %s", len(args.urls))
        return args.urls

    if args.source == "sample":
        logger.info("Using sample URL list: %s URLs", len(DEFAULT_DOC_URLS))
        return DEFAULT_DOC_URLS

    sitemap_urls = args.sitemap_urls or DEFAULT_SITEMAP_URLS
    discovered_urls = []

    for index, sitemap_url in enumerate(sitemap_urls):
        remaining = None
        if args.limit is not None:
            remaining_total = args.limit - len(discovered_urls)
            if remaining_total <= 0:
                break
            remaining_sitemaps = len(sitemap_urls) - index
            remaining = (remaining_total + remaining_sitemaps - 1) // remaining_sitemaps

        try:
            urls = get_sitemap_urls(sitemap_url, limit=remaining)
        except Exception as exc:
            logger.warning("Failed to parse sitemap %s: %s", sitemap_url, exc)
            continue

        logger.info("Sitemap URLs discovered from %s: %s", sitemap_url, len(urls))
        discovered_urls.extend(urls)

    discovered_urls = _dedupe_urls(discovered_urls)
    if args.limit is not None:
        discovered_urls = discovered_urls[: args.limit]

    logger.info("Total sitemap URLs discovered: %s", len(discovered_urls))
    logger.info("Core URLs added: %s", len(CORE_DOC_URLS))

    if not discovered_urls:
        logger.warning("No sitemap URLs discovered; using core URL list only.")

    merged_urls = _dedupe_urls([*CORE_DOC_URLS, *discovered_urls])
    logger.info("Total URLs after merging core + sitemap: %s", len(merged_urls))

    return merged_urls


def main() -> None:
    _configure_logging()
    args = parse_args()
    urls = resolve_urls(args)

    if args.reset:
        reset_vectorstore()

    load_result = load_documents_with_failures(urls)
    documents = load_result.documents
    loaded_urls = {document.metadata.get("source") for document in documents}

    logger.info("URLs loaded: %s/%s", len(loaded_urls), len(urls))
    logger.info("Skipped URLs after document cleaning: %s", len(load_result.skipped_urls))
    logger.info("Failed URLs: %s", len(load_result.failed_urls))
    for skipped_url in load_result.skipped_urls:
        logger.info("Skipped URL after cleaning: %s", skipped_url)
    for failed_url in load_result.failed_urls:
        logger.warning("Failed URL: %s", failed_url)

    logger.info("Chunking strategy selected: %s", args.chunking)
    chunks = chunk_documents(documents, strategy=args.chunking)
    chunk_counts = Counter(
        chunk.metadata.get("chunking_strategy", "unknown") for chunk in chunks
    )
    for strategy, count in sorted(chunk_counts.items()):
        logger.info("Chunk count for %s chunking: %s", strategy, count)

    add_documents(chunks)

    print(f"Loaded documents: {len(documents)}")
    print(f"Persisted chunks: {len(chunks)}")
    print(f"Chroma path: {CHROMA_DIR}")
    print(f"Collection: {CHROMA_COLLECTION}")


if __name__ == "__main__":
    main()
