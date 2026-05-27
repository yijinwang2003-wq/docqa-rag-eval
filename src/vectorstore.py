"""ChromaDB vector store setup."""

from collections.abc import Sequence
import hashlib
import json

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    PROJECT_ROOT,
    OPENAI_EMBEDDING_MODEL,
    require_openai_api_key,
)


EMBEDDING_CACHE_PATH = PROJECT_ROOT / "outputs" / "cache" / "query_embeddings.json"


class CachedQueryEmbeddings(Embeddings):
    """Cache query embeddings while delegating document embeddings to OpenAI."""

    def __init__(self, embeddings: OpenAIEmbeddings, cache_path=EMBEDDING_CACHE_PATH):
        self.embeddings = embeddings
        self.cache_path = cache_path
        self.cache = _load_json_cache(cache_path)

    def __getattr__(self, name: str):
        return getattr(self.embeddings, name)

    def embed_query(self, text: str) -> list[float]:
        key = _cache_key(OPENAI_EMBEDDING_MODEL, text)
        cached_embedding = self.cache.get(key)
        if cached_embedding is not None:
            return cached_embedding

        embedding = self.embeddings.embed_query(text)
        self.cache[key] = embedding
        _write_json_cache(self.cache_path, self.cache)
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)


def get_embeddings() -> CachedQueryEmbeddings:
    """Create the OpenAI embedding client."""

    require_openai_api_key()
    return CachedQueryEmbeddings(OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL))


def _cache_key(*parts: str) -> str:
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_json_cache(path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json_cache(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file)
        file.write("\n")


def get_vectorstore() -> Chroma:
    """Create or load the persistent Chroma vector store."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def reset_vectorstore() -> None:
    """Delete the configured Chroma collection if it already exists."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise


def _document_id(document: Document) -> str:
    source = document.metadata.get("source", "")
    chunk_index = document.metadata.get("chunk_index", "")
    chunking_strategy = document.metadata.get("chunking_strategy", "")
    raw = f"{source}:{chunk_index}:{chunking_strategy}:{document.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def add_documents(documents: Sequence[Document]) -> Chroma:
    """Embed and upsert documents into Chroma."""

    vectorstore = get_vectorstore()
    document_list = list(documents)

    if document_list:
        vectorstore.add_documents(
            documents=document_list,
            ids=[_document_id(document) for document in document_list],
        )

    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        persist()

    return vectorstore


def get_persisted_documents() -> list[Document]:
    """Load all persisted Chroma chunks as LangChain Documents."""

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    data = collection.get(include=["documents", "metadatas"])

    documents = []
    ids = data.get("ids") or []
    page_contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    for index, page_content in enumerate(page_contents):
        if not page_content:
            continue

        metadata = dict(metadatas[index] or {})
        if index < len(ids):
            metadata["chroma_id"] = ids[index]

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents
