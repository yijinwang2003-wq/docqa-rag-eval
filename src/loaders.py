"""Document loading utilities."""

from collections.abc import Sequence
from dataclasses import dataclass
import gzip
import logging
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

from src.config import USER_AGENT

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
import requests


CORE_DOC_URLS = [
    "https://docs.langchain.com/oss/python/langchain/rag",
    "https://docs.langchain.com/oss/python/langchain/overview",
    "https://docs.langchain.com/oss/python/integrations/vectorstores/chroma",
    "https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter",
    "https://platform.claude.com/docs/en/intro",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview",
    "https://platform.claude.com/docs/en/api/messages",
]

DEFAULT_DOC_URLS = CORE_DOC_URLS

DEFAULT_SITEMAP_URLS = [
    "https://docs.anthropic.com/sitemap.xml",
    "https://platform.claude.com/sitemap.xml",
    "https://docs.langchain.com/sitemap.xml",
]

ASSET_EXTENSIONS = (
    ".avif",
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".svg",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".zip",
)

EXCLUDED_PATH_PARTS = (
    "/api-reference/",
    "auth-service",
    "agent-connections",
)

PREFERRED_PATH_PARTS = (
    "/oss/python/langchain/",
    "/oss/python/integrations/",
    "/docs/",
    "/build-with-claude/",
    "/api/messages",
)

BOILERPLATE_LINE_EXACT = {
    "company",
    "contact sales",
    "github",
    "linkedin",
    "resources",
    "was this page helpful?",
    "youtube",
}

BOILERPLATE_LINE_CONTAINS = (
    "docs by langchain home page",
)

MIN_CLEANED_TEXT_LENGTH = 800

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    documents: list[Document]
    failed_urls: list[str]
    skipped_urls: list[str]


def infer_doc_site(url: str) -> str:
    """Infer the documentation site label from a URL."""

    netloc = urlparse(url).netloc.lower()

    if "anthropic" in netloc or "claude" in netloc:
        return "anthropic"
    if "langchain" in netloc:
        return "langchain"

    return "unknown"


def _clean_metadata(metadata: dict, fallback_url: str = "") -> dict:
    source = metadata.get("source") or metadata.get("url") or fallback_url
    title = metadata.get("title") or ""
    return {
        "source": str(source),
        "doc_site": infer_doc_site(str(source)),
        "title": str(title),
    }


def _clean_text(text: str) -> str:
    cleaned_lines = []
    previous_line = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized_line = _normalize_line(line)

        if not line or _is_boilerplate_line(normalized_line):
            continue
        if normalized_line == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = normalized_line

    return "\n".join(cleaned_lines)


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().lower().split())


def _is_boilerplate_line(normalized_line: str) -> bool:
    if normalized_line in BOILERPLATE_LINE_EXACT:
        return True

    return any(marker in normalized_line for marker in BOILERPLATE_LINE_CONTAINS)


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1].lower()


def _is_sitemap_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".xml", ".xml.gz")) and "sitemap" in path


def _is_changelog_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    parts = [part for part in path.split("/") if part]
    return any("changelog" in part or "release-notes" in part for part in parts)


def _is_supported_doc_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()

    if parsed.scheme not in {"http", "https"}:
        return False
    if path.endswith(ASSET_EXTENSIONS):
        return False
    if _is_changelog_url(url):
        return False
    if _is_sitemap_url(url):
        return False
    if any(part in path for part in EXCLUDED_PATH_PARTS):
        return False
    if not any(part in path for part in PREFERRED_PATH_PARTS):
        return False

    return infer_doc_site(url) in {"anthropic", "langchain"}


def _fetch_xml(url: str) -> ET.Element:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()

    content = response.content
    if urlparse(url).path.lower().endswith(".gz"):
        content = gzip.decompress(content)

    return ET.fromstring(content)


def _extract_loc_values(root: ET.Element) -> list[str]:
    return [
        (element.text or "").strip()
        for element in root.iter()
        if _tag_name(element.tag) == "loc" and (element.text or "").strip()
    ]


def _dedupe_urls(urls: Sequence[str]) -> list[str]:
    unique_urls = []
    seen = set()

    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)

    return unique_urls


def get_sitemap_urls(sitemap_url: str, limit: int | None = None) -> list[str]:
    """Parse a sitemap and return filtered documentation page URLs."""

    candidate_urls: list[str] = []
    discovered_urls: list[str] = []
    skipped_urls: set[str] = set()
    seen_candidate_urls: set[str] = set()
    seen_urls: set[str] = set()
    visited_sitemaps: set[str] = set()

    def add_candidate(url: str) -> None:
        if url in seen_candidate_urls:
            return
        seen_candidate_urls.add(url)
        candidate_urls.append(url)

    def add_url(url: str) -> None:
        if url in seen_urls:
            return
        seen_urls.add(url)
        discovered_urls.append(url)

    def visit(url: str) -> None:
        if url in visited_sitemaps:
            return
        if limit is not None and len(discovered_urls) >= limit:
            return

        visited_sitemaps.add(url)
        root = _fetch_xml(url)
        is_sitemap_index = _tag_name(root.tag) == "sitemapindex"

        for loc in _extract_loc_values(root):
            if limit is not None and len(discovered_urls) >= limit:
                return

            if is_sitemap_index or _is_sitemap_url(loc):
                visit(loc)
                continue

            add_candidate(loc)

            if _is_supported_doc_url(loc):
                add_url(loc)
            else:
                skipped_urls.add(loc)

    visit(sitemap_url)
    kept_urls = _dedupe_urls(discovered_urls[:limit])

    logger.info(
        "Discovered sitemap URLs before filtering from %s: %s",
        sitemap_url,
        len(candidate_urls),
    )
    logger.info(
        "Kept URLs after filtering from %s: %s",
        sitemap_url,
        len(kept_urls),
    )
    logger.info(
        "Skipped URL count from %s: %s",
        sitemap_url,
        len(skipped_urls),
    )

    return kept_urls


def load_documents_with_failures(urls: Sequence[str] | None = None) -> LoadResult:
    """Load documentation pages and report URLs that failed to load."""

    target_urls = list(urls or DEFAULT_DOC_URLS)
    documents = []
    failed_urls = []
    skipped_urls = []

    for url in target_urls:
        try:
            loader = WebBaseLoader(web_paths=[url], requests_per_second=2)
            loaded_documents = loader.load()
        except Exception as exc:
            logger.warning("Failed to load %s: %s", url, exc)
            failed_urls.append(url)
            continue

        cleaned_documents = []
        for document in loaded_documents:
            cleaned_text = _clean_text(document.page_content)
            if len(cleaned_text) < MIN_CLEANED_TEXT_LENGTH:
                continue

            cleaned_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=_clean_metadata(document.metadata, fallback_url=url),
                )
            )

        if cleaned_documents:
            documents.extend(cleaned_documents)
        else:
            skipped_urls.append(url)

    return LoadResult(
        documents=documents,
        failed_urls=failed_urls,
        skipped_urls=skipped_urls,
    )


def load_documents(urls: Sequence[str] | None = None) -> list[Document]:
    """Load public documentation pages into LangChain Documents."""

    return load_documents_with_failures(urls).documents
