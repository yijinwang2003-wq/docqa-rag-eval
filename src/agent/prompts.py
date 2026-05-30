"""Prompts used by the single-agent RAG workflow."""

from langchain_core.prompts import ChatPromptTemplate


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the user's documentation question into one clear, standalone
search query. Preserve the user's intent and technical terms. Do not answer the
question. Return only the rewritten query.""",
        ),
        ("human", "{question}"),
    ]
)
