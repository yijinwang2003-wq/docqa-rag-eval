"""FastAPI query service for the documentation QA pipeline."""

from time import perf_counter
from typing import Literal

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.chunking import FIXED_CHUNKING, SEMANTIC_CHUNKING
from src.evaluation import DENSE_RETRIEVER, HYBRID_RETRIEVER, build_configured_retriever
from src.rag_chain import PROMPT, format_docs, get_llm, unique_sources


ChunkingName = Literal["fixed", "semantic"]
RetrieverName = Literal["dense", "hybrid"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retriever: RetrieverName = HYBRID_RETRIEVER
    chunking: ChunkingName = FIXED_CHUNKING


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_latency_s: float
    generation_latency_s: float
    total_latency_s: float


app = FastAPI(title="Documentation QA API")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    total_start = perf_counter()
    retriever = build_configured_retriever(request.chunking, request.retriever)

    retrieval_start = perf_counter()
    documents = list(retriever.invoke(request.question))
    retrieval_latency_s = perf_counter() - retrieval_start

    generation_start = perf_counter()
    answer_chain = PROMPT | get_llm() | StrOutputParser()
    answer = answer_chain.invoke(
        {
            "question": request.question,
            "context": format_docs(documents),
        }
    )
    generation_latency_s = perf_counter() - generation_start

    return QueryResponse(
        answer=answer,
        sources=unique_sources(documents),
        retrieval_latency_s=retrieval_latency_s,
        generation_latency_s=generation_latency_s,
        total_latency_s=perf_counter() - total_start,
    )
