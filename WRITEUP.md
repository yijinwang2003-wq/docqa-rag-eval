# WRITEUP: Technical Documentation QA with Configurable RAG Evaluation

This is an engineering analysis of the design decisions, experiments, and tradeoffs behind this project. The README covers what the system does and how to run it. This document covers why it was built this way.

---

## Motivation

Most public RAG demos are built around a UX outcome: a chatbot that answers questions. The infrastructure decisions—how documents are chunked, how retrieval is configured, how answer quality is measured—are treated as implementation details rather than the main subject.

This project inverts that. The central question is: **how much do chunking strategy and retrieval configuration actually affect answer quality, and can that be measured reproducibly?**

The documentation corpus (Anthropic and LangChain public docs) was chosen because it is technically dense, publicly available, and contains the kind of conceptual content that stresses retrieval: similar-sounding concepts, nested terminology, and topics where a partial chunk can look relevant but not actually answer the question.

---

## Problem: Retrieval Failure Before Any Optimization

The first observation after building the initial RAG scaffold was that dense retrieval over naively ingested sitemap URLs performed poorly on conceptual questions.

Queries like "how does hybrid retrieval combine BM25 and dense search?" consistently surfaced LangChain integration pages (e.g., Cohere, HuggingFace provider pages) rather than conceptual RAG documentation. The retrieved chunks had high embedding similarity to the query but were about configuration of a specific provider, not about retrieval mechanics.

Three root causes were identified:

1. **Noisy sitemap coverage.** Sitemaps include every indexed page: API references, changelogs, provider integration pages, auth guides. These pages share vocabulary with conceptual documentation but answer different questions.

2. **Insufficient conceptual corpus coverage.** The most important RAG and chunking documentation pages were not guaranteed to be included in a random sitemap sample. Without them, the retriever had no good candidates.

3. **Semantic drift in dense retrieval.** Dense retrieval finds semantically nearby content, not necessarily the best answer. When the corpus is noisy, "nearby" resolves to topically adjacent but irrelevant content.

These failures motivated three mitigations before any chunking or retrieval experiments could produce meaningful results.

---

## Mitigation 1: Corpus Quality

**Sitemap filtering.** URL patterns for API references, changelogs, auth-service pages, and agent-connection pages were excluded from sitemap ingestion. This reduced noise but also reduced coverage, which is why the next step was necessary.

**Boilerplate cleaning.** Many documentation pages include repeated navigation blocks, footer text, and cookie banners. These inflate chunk content with text that is not semantically useful. Pages with fewer than 800 characters after cleaning were dropped entirely.

**CORE_DOC_URLS allowlist.** A fixed set of the most important conceptual documentation pages (LangChain RAG guide, vector store docs, chunking guide, Anthropic Claude API docs) was added as a guaranteed-included set. Sitemap discovery appends to this list rather than replacing it, and the `--limit` flag applies only to sitemap-discovered URLs so core coverage is not squeezed out.

The result was a smaller but higher-quality corpus. Retrieval failure on conceptual queries dropped noticeably after this change, before any retrieval strategy tuning.

---

## Design Decision: What to Compare

The experiment compares two variables:

- **Chunking strategy**: `fixed` (RecursiveCharacterTextSplitter) vs `semantic` (SemanticChunker with sentence-similarity fallback)
- **Retrieval strategy**: `dense` (Chroma cosine similarity) vs `hybrid` (Chroma + BM25 with reciprocal rank fusion)

This gives four configurations: `fixed+dense`, `fixed+hybrid`, `semantic+dense`, `semantic+hybrid`.

These were chosen because they represent the first and most impactful decision point in a RAG pipeline. Chunking determines the text units that get embedded; retrieval determines how those units are ranked. Both affect answer quality before the LLM is ever called.

Variables held constant across all configurations: embedding model (OpenAI `text-embedding-3-small`), LLM (same model and prompt), ChromaDB persistence path, and evaluation question set.

---

## Design Decision: Evaluation Set Size

The initial evaluation set had 10 questions. This was insufficient for comparing four configurations.

The problem is statistical: with 10 questions, a single retrieval failure on one question can shift the mean faithfulness or context precision by 0.1 or more. The configurations differ in subtle ways, and the differences in mean metrics are small (1–4 percentage points). At n=10, these differences are noise. At n=42, the ranking of configurations stabilizes.

The evaluation set was expanded using RAGAS `TestsetGenerator` to generate synthetic candidates from the indexed corpus, then filtered automatically (duplicates, vague questions, empty ground truths) and reviewed manually against a markdown checklist. The final 42 questions cover LangChain RAG mechanics, chunking behavior, Chroma operations, BM25, embeddings, and the Anthropic Claude API.

The same reviewed question set is used for all four configurations, so the comparison is controlled.

---

## Evaluation Infrastructure

RAGAS is the preferred evaluator in the code path. In practice, RAGAS had dependency compatibility issues in the local environment (conflicts between `ragas`, `datasets`, and the installed LangChain version). All 42 reported results use an LLM fallback evaluator that implements the same three metrics—faithfulness, answer relevancy, context precision—using direct LLM scoring rather than the RAGAS framework.

The fallback evaluator produces scores on the same 0–1 scale. The scores are not directly comparable to RAGAS-scored results from other projects, but they are internally consistent: the four configurations are evaluated with the same evaluator under identical conditions, so relative differences are valid.

Evaluation runs are checkpointed incrementally to `outputs/eval_results_partial.csv`. Completed rows are not re-evaluated on resume. This was necessary because a full 4-configuration × 42-question run makes ~168 LLM calls for generation plus ~504 calls for evaluation (3 metrics × 168), and failures mid-run should not require restarting from zero.

---

## Results and Interpretation

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Retrieval Latency (s) |
|---|---:|---:|---:|---:|
| fixed+dense | 0.998 | 0.854 | 0.710 | 0.013 |
| fixed+hybrid | 0.993 | 0.886 | 0.720 | 0.004 |
| semantic+dense | 0.990 | 0.868 | 0.682 | 0.003 |
| semantic+hybrid | 0.995 | 0.890 | 0.690 | 0.004 |

**Faithfulness saturated near 1.0 across all configurations.** This means generated answers were consistently grounded in retrieved context regardless of chunking or retrieval strategy. Faithfulness is not a useful discriminating metric here. The model does not hallucinate beyond the retrieved context when the prompt is explicit about grounding.

**Answer relevancy separates hybrid from dense.** Both hybrid configurations outperformed their dense-only counterparts by 2–4 points. BM25 adds keyword matching to dense retrieval, which helps with exact technical terms like `chunk_overlap`, `RecursiveCharacterTextSplitter`, and `context_precision`. Dense retrieval can drift toward semantically adjacent concepts; BM25 anchors the retrieval to the specific term in the query.

**Semantic chunking did not improve context precision.** The hypothesis was that semantic chunks—grouped by meaning rather than character count—would produce more topically coherent retrieved context. The results did not support this. Semantic chunking actually reduced context precision compared to fixed chunking (0.682–0.690 vs 0.710–0.720).

The likely explanation: semantic chunking on technical documentation produces larger, less uniform chunks. A semantic chunk that covers a broad concept like "how vector search works" contains more text, but not all of it is relevant to a specific query about `k` selection or distance metrics. Fixed chunking at a consistent character count produces smaller, more precise units. For a corpus of structured technical documentation, fixed chunking is the better baseline.

**`fixed+hybrid` is the best overall configuration.** Highest context precision (0.720), second-highest answer relevancy (0.886, within 0.4 points of `semantic+hybrid`), and faster retrieval latency than `fixed+dense`.

**Retrieval latency is sub-10ms for all configurations.** At this scale the difference between 0.003s and 0.013s is not operationally significant. The latency numbers matter more as corpus size grows, where BM25 index construction and Chroma query time diverge.

---

## What Was Not Built

**No reranker.** Cross-encoder reranking is the natural next step: retrieve a larger candidate set, then rerank with a model that scores query-document pairs jointly. The hypothesis is that reranking would improve context precision more than chunking strategy did in these experiments. This is the next planned experiment.

**No recall@k or hit@k.** The current evaluation measures generated answer quality (faithfulness, relevancy, precision) but not raw retrieval quality. Adding ground-truth relevance labels to a subset of queries would enable direct retrieval evaluation independent of the LLM.

**No streaming.** The FastAPI endpoint returns synchronously. Total latency of ~16s on a cold request is dominated by LLM generation time. Streaming would improve perceived latency but not actual generation time.

**No frontend.** The system is intentionally CLI and API-first. A Streamlit or React frontend would make it more demonstrable but would not add to the retrieval engineering analysis.

---

## What This Project Is

This project is a reproducible retrieval benchmarking system. The core value is not the QA system itself—any RAG scaffold can answer questions—but the evaluation infrastructure that makes configuration tradeoffs visible and comparable.

The main claims supported by the experiments:
- Corpus quality (filtering, cleaning, guaranteed core coverage) matters more than retrieval strategy tuning on a noisy corpus.
- Hybrid retrieval consistently improves answer relevancy over dense-only retrieval on technical documentation.
- Fixed chunking outperforms semantic chunking on structured documentation where chunk size consistency matters more than semantic coherence.
- Faithfulness is a floor metric for this type of grounded QA; answer relevancy and context precision are the informative signals.
