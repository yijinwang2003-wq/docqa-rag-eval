"""Text splitting utilities."""

from collections.abc import Sequence
import logging
import math
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.vectorstore import get_embeddings


ChunkingStrategy = str
FIXED_CHUNKING = "fixed"
SEMANTIC_CHUNKING = "semantic"
CHUNKING_STRATEGIES = (FIXED_CHUNKING, SEMANTIC_CHUNKING)

logger = logging.getLogger(__name__)


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create the default recursive character splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )


def chunk_documents(
    documents: Sequence[Document],
    strategy: ChunkingStrategy = FIXED_CHUNKING,
) -> list[Document]:
    """Split documents into retrieval-sized chunks using the selected strategy."""

    if strategy == FIXED_CHUNKING:
        chunks = chunk_documents_fixed(documents)
    elif strategy == SEMANTIC_CHUNKING:
        chunks = chunk_documents_semantic(documents)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    _annotate_chunks(chunks, strategy)
    return chunks


def chunk_documents_fixed(documents: Sequence[Document]) -> list[Document]:
    """Split documents with RecursiveCharacterTextSplitter."""

    splitter = get_text_splitter()
    return splitter.split_documents(list(documents))


def chunk_documents_semantic(documents: Sequence[Document]) -> list[Document]:
    """Split documents with LangChain SemanticChunker or a local fallback."""

    semantic_chunks = _chunk_with_langchain_semantic_chunker(documents)
    if semantic_chunks is not None:
        logger.info("Semantic chunking implementation: LangChain SemanticChunker")
        return semantic_chunks

    logger.info("Semantic chunking implementation: local sentence-similarity fallback")
    return _chunk_with_sentence_similarity(documents)


def _chunk_with_langchain_semantic_chunker(
    documents: Sequence[Document],
) -> list[Document] | None:
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        return None

    splitter = SemanticChunker(get_embeddings())
    return splitter.split_documents(list(documents))


def _chunk_with_sentence_similarity(documents: Sequence[Document]) -> list[Document]:
    embeddings = get_embeddings()
    chunks = []

    for document in documents:
        sentences = _split_sentences(document.page_content)
        if not sentences:
            continue

        sentence_embeddings = embeddings.embed_documents(sentences)
        chunks.extend(
            _merge_sentences_by_similarity(
                document=document,
                sentences=sentences,
                sentence_embeddings=sentence_embeddings,
            )
        )

    return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = []

    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue

        block_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", block)
        sentences.extend(
            sentence.strip()
            for sentence in block_sentences
            if sentence.strip()
        )

    return sentences


def _merge_sentences_by_similarity(
    document: Document,
    sentences: Sequence[str],
    sentence_embeddings: Sequence[Sequence[float]],
) -> list[Document]:
    chunks = []
    current_sentences = [sentences[0]]
    current_length = len(sentences[0])
    start_sentence_index = 0

    for sentence_index in range(1, len(sentences)):
        sentence = sentences[sentence_index]
        previous_embedding = sentence_embeddings[sentence_index - 1]
        current_embedding = sentence_embeddings[sentence_index]
        similarity = _cosine_similarity(previous_embedding, current_embedding)
        candidate_length = current_length + 1 + len(sentence)

        should_break = (
            similarity < SEMANTIC_SIMILARITY_THRESHOLD
            and current_length >= SEMANTIC_MIN_CHUNK_SIZE
        )
        too_large = (
            candidate_length > SEMANTIC_MAX_CHUNK_SIZE
            and current_length >= SEMANTIC_MIN_CHUNK_SIZE
        )

        if should_break or too_large:
            chunks.append(
                _make_semantic_chunk(
                    document=document,
                    sentences=current_sentences,
                    start_sentence_index=start_sentence_index,
                    end_sentence_index=sentence_index - 1,
                )
            )
            current_sentences = [sentence]
            current_length = len(sentence)
            start_sentence_index = sentence_index
            continue

        current_sentences.append(sentence)
        current_length = candidate_length

    chunks.append(
        _make_semantic_chunk(
            document=document,
            sentences=current_sentences,
            start_sentence_index=start_sentence_index,
            end_sentence_index=len(sentences) - 1,
        )
    )

    return chunks


def _make_semantic_chunk(
    document: Document,
    sentences: Sequence[str],
    start_sentence_index: int,
    end_sentence_index: int,
) -> Document:
    metadata = dict(document.metadata)
    metadata["semantic_start_sentence"] = start_sentence_index
    metadata["semantic_end_sentence"] = end_sentence_index
    metadata["semantic_similarity_threshold"] = SEMANTIC_SIMILARITY_THRESHOLD

    return Document(
        page_content=" ".join(sentences),
        metadata=metadata,
    )


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _annotate_chunks(chunks: Sequence[Document], strategy: ChunkingStrategy) -> None:
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunking_strategy"] = strategy
