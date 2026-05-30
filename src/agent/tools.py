"""Tool wrappers around the existing RAG implementation."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.agent.prompts import QUERY_REWRITE_PROMPT
from src.rag_chain import (
    PROMPT,
    format_docs,
    get_llm,
    get_selected_retriever,
    unique_sources,
)


def rewrite_query(question: str) -> str:
    """Rewrite a question for retrieval using the configured chat model."""

    chain = QUERY_REWRITE_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def retrieve_documents(question: str, retriever_name: str) -> list[Document]:
    """Retrieve documents through the existing retriever factory."""

    retriever = get_selected_retriever(retriever_name)
    return list(retriever.invoke(question))


def generate_answer(question: str, documents: list[Document]) -> str:
    """Generate an answer with the existing source-grounded RAG prompt."""

    answer_chain = PROMPT | get_llm() | StrOutputParser()
    return answer_chain.invoke(
        {
            "question": question,
            "context": format_docs(documents),
        }
    )


def document_sources(documents: list[Document]) -> list[str]:
    """Return unique source URLs from retrieved documents."""

    return unique_sources(documents)
