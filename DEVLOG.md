# Devlog

Running notes for the portfolio writeup.

## 2026-05-26

### Week 1 RAG Scaffold

- Created the initial LangChain RAG project structure.
- Added loading for a small hardcoded set of Claude/Anthropic and LangChain documentation URLs.
- Added recursive character chunking, OpenAI embeddings, local persistent ChromaDB, and a CLI QA flow.
- Added README setup instructions and environment configuration.

### RAG CLI Debug Mode

- Added `--debug` to `scripts/ask.py` for inspecting retrieval quality.
- Debug output reports retrieved chunk count, source URL, title when present, `chunk_index` when present, and a 500-character chunk preview.
- Kept normal CLI output unchanged so demos and screenshots remain stable.

### Sitemap Ingestion

- Added XML sitemap parsing in `src/loaders.py` with URL filtering for assets, images, PDFs, and obvious changelog pages.
- Added `--source sample`, `--source sitemap`, `--limit`, and `--sitemap-url` options to `scripts/ingest_docs.py`.
- Added ingestion logging for sitemap discovery counts, loaded URL counts, and failed URLs.
- Split sitemap limits across configured sitemaps so default discovery samples both documentation sources.

### Sitemap Filtering And Cleaning

- Tightened sitemap URL filtering to focus on conceptual docs and tutorials.
- Excluded LangChain API reference, auth-service, and agent-connections pages from sitemap ingestion.
- Added boilerplate line removal and skipped pages with less than 800 characters after cleaning.
- Added logging for pre-filter sitemap candidates, kept URLs, skipped URL counts, and pages skipped after cleaning.

### Hybrid Retrieval

- Added a hybrid retriever that combines Chroma dense retrieval with BM25 keyword retrieval.
- BM25 now builds from the same persisted Chroma chunks used by dense retrieval.
- Added `--retriever dense` and `--retriever hybrid` to `scripts/ask.py`, keeping dense as the default.
- Debug output now reports the retriever strategy and whether each returned chunk came from dense, BM25, or both.

### Core Document Coverage

- Added `CORE_DOC_URLS` for the most important LangChain RAG and Claude conceptual pages.
- Sitemap ingestion now preserves core URLs first, then appends deduplicated sitemap URLs.
- Kept `--limit` scoped to sitemap-discovered URLs so core coverage is not squeezed out.

### Chunking Experiments

- Added `--chunking fixed` and `--chunking semantic` to ingestion.
- Kept fixed chunking on `RecursiveCharacterTextSplitter` for predictable baseline experiments.
- Added semantic chunking that prefers LangChain `SemanticChunker` when installed and otherwise falls back to sentence splitting plus adjacent-sentence embedding similarity.
- Stored `chunking_strategy` in chunk metadata and surfaced it in ask debug output.
- Documented why chunking affects retrieval quality: chunk boundaries define the exact text units embedded and returned as context.

### Retrieval Failure Analysis

Initial dense retrieval over sitemap-ingested docs frequently surfaced irrelevant integration/provider pages for conceptual RAG queries.

Root causes:
1. noisy sitemap coverage
2. insufficient conceptual corpus coverage
3. semantic drift in dense retrieval

Mitigations:
- sitemap filtering
- boilerplate cleaning
- CORE_DOC_URLS allowlist
- hybrid retrieval (BM25 + dense)

### RAGAS Evaluation Pipeline

- Added `src/evaluation.py` as a modular evaluation layer for loading questions, running RAG configurations, scoring with RAGAS, saving CSV output, and summarizing metrics.
- Added `scripts/run_eval.py` to compare `fixed + dense`, `fixed + hybrid`, `semantic + dense`, and `semantic + hybrid`.
- Added `data/eval_questions.json` with 10 focused technical questions covering LangChain RAG, chunking, Chroma, embeddings, Claude Messages API, prompt grounding, dense retrieval, BM25, and hybrid retrieval.
- Added `ragas`, `datasets`, and `pandas` to `requirements.txt`.
- The evaluator stores per-question scores in `outputs/eval_results.csv` and prints mean `faithfulness`, `answer_relevancy`, and `context_precision` by configuration.

Why this matters:

- Retrieval quality sets the evidence boundary for the generator. Bad retrieval can produce weak answers even with a strong model.
- `faithfulness` checks whether generated answers are supported by retrieved context.
- `answer_relevancy` checks whether generated answers address the question being asked.
- `context_precision` checks whether useful contexts are ranked ahead of less useful retrieved text.

Observed and expected configuration tradeoffs:

- Fixed chunking is stable and easy to reproduce, but can cut through conceptual explanations.
- Semantic chunking can keep related ideas together, but costs more during ingestion and can produce less uniform chunks.
- Dense retrieval handles semantic paraphrases well, but can drift toward nearby concepts.
- Hybrid retrieval combines dense search with BM25 keyword matching, which is usually better for exact technical terms but may bring back more heterogeneous context.

Validation:

- `python -m compileall src scripts`
- `python scripts/run_eval.py --help`

### Evaluation Set Generation And Visualization

- Added `scripts/generate_eval_questions.py` to build a larger synthetic eval set from persisted Chroma chunks with RAGAS `TestsetGenerator`.
- The generator prefers chunks from `CORE_DOC_URLS` and other conceptual documentation so questions are based on the most important RAG, chunking, vector store, Claude, and prompting material.
- Added filters for duplicate questions, very short questions, empty ground truths, and vague prompts such as "What is this?"
- Added `data/eval_questions_raw.json` for raw generated records, `data/eval_questions.json` for cleaned review-ready records, and `data/eval_questions_review.md` for manual keep/revise/drop review.
- Added `scripts/plot_eval_results.py` to read `outputs/eval_results.csv`, compute mean `faithfulness`, `answer_relevancy`, and `context_precision` by configuration, and save `outputs/eval_metrics.png`.
- Added `matplotlib` to `requirements.txt`.

Why the 10-question set was insufficient:

- Ten questions are fine for a smoke test, but a single retrieval miss can move the mean too much.
- The configurations differ in subtle ways: chunk boundaries, semantic similarity, keyword matching, and mixed evidence all need repeated examples before the tradeoffs are meaningful.
- A 40-50 question reviewed set is still small enough to inspect manually, while giving the writeup a more defensible comparison.

Synthetic generation plus manual review workflow:

- Use RAGAS to generate candidate questions from the indexed documentation.
- Filter obvious low-quality candidates automatically.
- Review the markdown checklist to keep, revise, or drop each item before publishing results.
- Run the same reviewed dataset across all configurations so the comparison is controlled.

Visualization step:

- The CSV is the audit artifact for per-question behavior.
- The PNG bar chart is the portfolio artifact for quickly comparing mean metric values across `fixed+dense`, `fixed+hybrid`, `semantic+dense`, and `semantic+hybrid`.

Validation:

- `python -m compileall src scripts`
- `python scripts/generate_eval_questions.py --help`
- `python scripts/plot_eval_results.py --help`
