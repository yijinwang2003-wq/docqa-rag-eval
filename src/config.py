"""Configuration for the documentation QA pipeline."""

from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _path_from_env(name: str, default: str) -> Path:
    raw_path = Path(os.getenv(name, default)).expanduser()
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5")
OPENAI_EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-5-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

CHROMA_DIR = _path_from_env("DOCQA_CHROMA_DIR", ".chroma")
CHROMA_COLLECTION = os.getenv("DOCQA_CHROMA_COLLECTION", "technical_docs")

RETRIEVER_K = _int_from_env("DOCQA_RETRIEVER_K", 4)
CHUNK_SIZE = _int_from_env("DOCQA_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int_from_env("DOCQA_CHUNK_OVERLAP", 200)
SEMANTIC_SIMILARITY_THRESHOLD = _float_from_env(
    "DOCQA_SEMANTIC_SIMILARITY_THRESHOLD",
    0.78,
)
SEMANTIC_MIN_CHUNK_SIZE = _int_from_env("DOCQA_SEMANTIC_MIN_CHUNK_SIZE", 500)
SEMANTIC_MAX_CHUNK_SIZE = _int_from_env(
    "DOCQA_SEMANTIC_MAX_CHUNK_SIZE",
    CHUNK_SIZE * 2,
)

USER_AGENT = os.getenv("USER_AGENT", "docqa-rag-eval/0.1")
os.environ.setdefault("USER_AGENT", USER_AGENT)


def require_openai_api_key() -> None:
    """Fail fast with a clear setup message if no OpenAI key is configured."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
