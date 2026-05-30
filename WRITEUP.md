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
- **Retrieval strategy**: `dense` (Chroma cosine similarity) vs `hybrid` (dense retrieval augmented with BM25 retrieval, followed by result merging and deduplication)

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

## Evaluation Dataset

The final benchmark contains 42 reviewed questions covering:

- RAG concepts
- Chunking
- Retrieval
- Embeddings
- Vector stores
- Anthropic API usage

Questions were initially generated automatically and then filtered, deduplicated,
and reviewed before inclusion in the benchmark. The static classification in
`outputs/eval_question_classification.md` further labels the 42 questions by
predicted rewrite behavior, question type, and likely agent benefit.

---

## Evaluation Infrastructure

RAGAS is the preferred evaluator in the code path. When RAGAS is unavailable, the project falls back to an LLM judge that implements the same three metrics: faithfulness, answer relevancy, and context precision. The current saved outputs do not record evaluator-backend metadata, so future runs should persist whether each score came from RAGAS or fallback judging.

The fallback evaluator produces scores on the same 0–1 scale. These scores should not be treated as directly comparable to RAGAS-scored results from other projects, but they can still support controlled within-run comparisons when the same evaluator is used across all configurations.

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

**`fixed+hybrid` is the strongest balanced configuration if prioritizing context precision and stable retrieval behavior.** It achieved the highest context precision (0.720) while remaining within 0.004 answer-relevancy points of `semantic+hybrid`, the top relevancy configuration.

**Retrieval latency is sub-10ms for all configurations.** At this scale the difference between 0.003s and 0.013s is not operationally significant. The latency numbers matter more as corpus size grows, where BM25 index construction and Chroma query time diverge.

---

## What Was Not Built

**No reranker.** Cross-encoder reranking is the natural next step: retrieve a larger candidate set, then rerank with a model that scores query-document pairs jointly. The hypothesis is that reranking would improve context precision more than chunking strategy did in these experiments. This is the next planned experiment.

**No recall@k or hit@k.** The current evaluation measures generated answer quality (faithfulness, relevancy, precision) but not raw retrieval quality. Adding ground-truth relevance labels to a subset of queries would enable direct retrieval evaluation independent of the LLM.

**No streaming.** The FastAPI endpoint returns synchronously. Total latency of ~16s on a cold request is dominated by LLM generation time. Streaming would improve perceived latency but not actual generation time.

**No interactive RAG frontend.** The system is intentionally CLI and API-first. The repository may include static portfolio assets, but it does not include an interactive RAG frontend such as Streamlit, Gradio, or a React chat UI.

---

## Agentic RAG Evaluation

The later phase of the project adds a minimal agentic RAG workflow on top of the same retrieval and generation components. The purpose is not to claim that a more complex agent is automatically better. The purpose is to measure whether specific agent behaviors—query analysis, query rewriting, trajectory logging, and confidence scoring—change answer quality or operational cost.

### Why a Minimal Agent

Phase 1 uses a single LangGraph workflow, the existing retriever, and the existing grounded generation chain. This keeps the comparison controlled: the baseline and agent share the same corpus, vector store, retriever implementation, prompt, and evaluator. Any measured differences are therefore easier to attribute to the agent wrapper rather than to a new retrieval backend or a new tool stack.

The project intentionally does not add multi-agent orchestration, supervisor-worker routing, external web search, code execution, or MCP tool integrations in this phase. Those features would make the system more capable, but they would also introduce new failure modes and confound the evaluation. A small graph is enough to test the first agentic question: does query analysis and optional rewriting improve retrieval and answer quality enough to justify the additional steps?

### Findings

The 10-question pilot did not trigger query rewriting, so it mostly tested the LangGraph wrapper and trajectory instrumentation. Faithfulness remained unchanged at 1.000 for both baseline and agent, and retrieval success was also unchanged. This run did not provide evidence that the agent improved retrieval reliability.

The targeted 7-question rewrite subset produced a more informative result. Q17 is the clearest positive rewrite case in the subset. The original question was symptom-first: "My RAG answers keep cutting off mid-explanation. Could this be a chunking problem?" The rewrite shifted retrieval toward a more technical troubleshooting query, which produced a higher judged answer relevancy score: 0.350 for baseline versus 0.600 for the agent, a 71% relative gain, while faithfulness remained 1.000 for both. This should not be interpreted as proving that chunking was not involved; rather, it shows that query rewriting changed the retrieval direction and produced a more relevant judged answer in this case.

The same subset also shows that rewriting is not universally helpful. Question 41 was already a well-specified question about adversarial instructions in retrieved documents. Both systems answered with the same substantive mitigations, but the baseline relevancy score was 1.000 and the agent score was 0.900. I would not treat that 0.1 difference as strong evidence of degradation because LLM judge scores are noisy, but it is evidence that rewriting can be unnecessary when the original query is already precise.

Agent orchestration introduced measurable workflow overhead. In the rewrite subset, the agent used a mean of 5.0 tool steps versus 2.0 for the baseline, and mean latency increased from 6.037s to 7.098s. The observed 5.0 mean mainly reflects the fixed Phase 1 graph design: query analysis, optional rewrite, retrieval, generation, and confidence scoring. Confidence scoring currently runs on every query and records fallback metadata when confidence is low; it does not trigger a second retrieval or generation pass.

Fallback behavior in Phase 1 means flagging low-confidence outputs and recording a fallback reason for evaluation. It does not yet execute an alternate retrieval strategy, rerun generation, or call external tools.

### Interpretation

The agentic evaluation reinforces the main evaluation philosophy of the project: final-answer quality is not enough. Agent behavior needs to be measured independently because the intermediate path can add latency, unnecessary rewrites, or extra confidence steps without improving the final answer.

Rewrite usefulness is highly query-dependent. It adds value when the user query is underspecified, symptom-first, or does not map cleanly to documentation structure. It is less useful when the query already contains the right technical concepts. More orchestration is therefore not automatically better; it needs to be justified by measurable gains on the query types where it is expected to help.

### Limitations

The agentic evaluation is small. The 10-question pilot did not exercise rewriting, and the 7-question rewrite subset was selected specifically because those questions were predicted to trigger the rewrite heuristic. That selection is useful for stress-testing rewrite behavior, but it is not representative of the full 42-question benchmark.

The evaluator is also an LLM judge, so small score differences should be interpreted cautiously. The Q41 result is best treated as inconclusive rather than as proof that rewriting harms good queries. The more defensible conclusion is narrower: the current rewrite heuristic can fire on already precise queries and should be made more selective.

Confidence and fallback calibration is another open issue. Future experiments should test whether confidence scoring should be conditionally invoked, and whether fallback thresholds correlate with retrieval-success signals, retrieved-document counts, answer relevancy, faithfulness, and latency.

---

## Cost Considerations

The full retrieval benchmark evaluates 4 configurations across 42 questions, producing 168 generated answers. With three answer-quality metrics per generated answer, a full run can require roughly 504 evaluation judgments in addition to generation calls. This is why the project separates generation and evaluation models, supports resumable partial CSV outputs, and uses cache-aware evaluation logic. The design allows failed or interrupted runs to resume without discarding completed generations and judgments.

---

## Threats to Validity

**LLM judge variability.** The evaluation uses LLM-scored metrics when RAGAS is unavailable. Scores are useful for controlled comparison within the same run, but small differences should not be overinterpreted.

**Dataset size.** The 42-question benchmark is large enough to compare the four retrieval configurations more reliably than the initial 10-question set, but it is still a small benchmark.

**Corpus scope.** The corpus is limited to selected public technical documentation, so results may not generalize to noisy enterprise corpora, long PDFs, or multimodal data.

**Rewrite subset selection.** The 7-question rewrite subset was intentionally selected to stress query rewriting, so it should not be treated as representative of all user questions.

**Evaluator metadata.** Saved outputs should record whether each score came from RAGAS or the fallback LLM judge to make future audits easier.

---

## What This Project Is

This project is a reproducible retrieval benchmarking system. The core value is not the QA system itself—any RAG scaffold can answer questions—but the evaluation infrastructure that makes configuration tradeoffs visible and comparable.

The main claims supported by the experiments:
- During development, corpus-quality improvements appeared to have a larger practical impact on retrieval failures than retrieval-strategy tuning. This observation was not evaluated through a dedicated ablation study and should be treated as an engineering observation rather than a measured experimental result.
- Hybrid retrieval consistently improves answer relevancy over dense-only retrieval on this technical documentation benchmark.
- Fixed chunking outperforms semantic chunking on context precision in this structured documentation corpus.
- Faithfulness is a floor metric for this type of grounded QA; answer relevancy and context precision are the informative signals.
- Agentic behavior needs separate evaluation: query rewriting can help underspecified diagnostic questions, but it can also add steps or fire unnecessarily on already precise queries.
