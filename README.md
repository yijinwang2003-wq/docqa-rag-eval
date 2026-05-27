# Technical Documentation QA with Configurable RAG Evaluation and Retrieval Benchmarking

An engineering-focused RAG experimentation system over public AI and technical documentation. It benchmarks retrieval behavior across chunking strategies, compares dense and hybrid retrieval, evaluates answer quality automatically, and tracks latency so quality/performance tradeoffs are visible.

The project is designed as a reproducible evaluation pipeline rather than a chatbot demo: ingestion, chunking, retrieval, generation, evaluation, visualization, caching, resumability, and FastAPI serving are all implemented as separable components.

## Architecture

```text
Public AI Docs (Anthropic, LangChain)
                ↓
       Sitemap Crawling
                ↓
      Cleaning + Filtering
                ↓
     Fixed / Semantic Chunking
                ↓
 OpenAI Embeddings + ChromaDB
                ↓
 Dense / Hybrid Retrieval (BM25)
                ↓
      LLM Answer Generation

                ├── Evaluation Pipeline
                └── FastAPI Serving API
```

Evaluation and serving are independent downstream paths: the same retrieval and answer-generation components can be benchmarked offline or served through the API.

## Why This Project?

Many public RAG demos focus mainly on chatbot UX. This project focuses on retrieval engineering and evaluation: how chunking, retrieval strategy, and latency affect answer quality.

The goal is to study system tradeoffs reproducibly by comparing fixed vs semantic chunking, dense vs hybrid retrieval, checkpointed evaluation runs, and latency measurements across configurations.

## Tech Stack

- LangChain
- ChromaDB
- OpenAI Embeddings
- BM25
- FastAPI
- Matplotlib
- RAGAS-compatible evaluation

## Features

- Sitemap-based ingestion from public documentation.
- Boilerplate cleaning and URL filtering.
- Fixed and semantic chunking strategies.
- Dense vector retrieval and hybrid dense + BM25 retrieval.
- Local ChromaDB vector store.
- Source-grounded answers with returned source URLs.
- Automated evaluation over 42 curated documentation QA questions.
- Metrics: faithfulness, answer relevancy, and context precision.
- Retrieval, generation, and total latency tracking.
- Resumable evaluation with partial CSV checkpoints.
- FastAPI `/query` endpoint for serving the RAG pipeline.

## Evaluation Methodology

The benchmark uses 42 manually curated documentation QA questions and compares four RAG configurations:

- `fixed+dense`
- `fixed+hybrid`
- `semantic+dense`
- `semantic+hybrid`

RAGAS remains the preferred evaluator in the code path, but all 42 reported questions were evaluated using the LLM fallback evaluator because RAGAS had dependency compatibility issues in the local environment. The fallback evaluator preserves the same evaluation schema and metrics: faithfulness, answer relevancy, and context precision.

Evaluation runs can be interrupted and resumed because completed rows are checkpointed incrementally to avoid redundant LLM calls. Results are saved to `outputs/eval_results.csv`, and the chart is saved to `outputs/eval_metrics.png`.

## Results

The benchmark compares retrieval quality and latency tradeoffs across four RAG configurations.

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Retrieval Latency (s) |
|---|---:|---:|---:|---:|
| fixed+dense | 0.998 | 0.854 | 0.710 | 0.013 |
| fixed+hybrid | 0.993 | 0.886 | 0.720 | 0.004 |
| semantic+dense | 0.990 | 0.868 | 0.682 | 0.003 |
| semantic+hybrid | 0.995 | 0.890 | 0.690 | 0.004 |

The chart below summarizes retrieval quality and latency tradeoffs across all evaluated configurations.

![Evaluation Results](outputs/eval_metrics.png)

## Key Findings

- Hybrid retrieval consistently improved answer relevancy over dense-only retrieval by roughly 2-3 percentage points.
- `fixed+hybrid` had the best overall balance of answer relevancy and context precision.
- Semantic chunking did not improve context precision in this setup, likely because broader semantic chunks introduced more irrelevant context.
- Faithfulness scores saturated near 1.0, indicating that generated answers generally remained well-grounded in retrieved context. As a result, answer relevancy and context precision became the more informative metrics for distinguishing retrieval configurations.
- Cached generation makes generation latency near-zero in benchmark reruns, so retrieval latency is the more meaningful latency signal.

For design decisions, experiment analysis, and result interpretation, see [WRITEUP.md](WRITEUP.md).

## How To Run

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Add your OpenAI API key to `.env`.

Ingest documentation with both chunking strategies:

```bash
python scripts/ingest_docs.py --source sitemap --limit 50 --chunking fixed --reset
python scripts/ingest_docs.py --source sitemap --limit 50 --chunking semantic
```

Ask a CLI question:

```bash
python scripts/ask.py --retriever hybrid --debug "What does chunk_overlap do?"
```

Run evaluation and generate the results chart:

```bash
python scripts/run_eval.py --configs all
python scripts/plot_eval_results.py
```

Run the API:

```bash
uvicorn src.api:app --reload
```

## API Usage

Interactive Swagger UI is available at `http://127.0.0.1:8000/docs`.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does chunk_overlap do?", "retriever": "hybrid", "chunking": "fixed"}'
```

Response fields:

- `answer`
- `sources`
- `retrieval_latency_s`
- `generation_latency_s`
- `total_latency_s`

## Project Structure

```text
docqa-rag-eval/
  README.md
  requirements.txt
  .env.example
  src/
    api.py
    chunking.py
    config.py
    evaluation.py
    loaders.py
    rag_chain.py
    vectorstore.py
  scripts/
    ask.py
    generate_eval_questions.py
    ingest_docs.py
    plot_eval_results.py
    run_eval.py
  data/
    eval_questions.json
    eval_questions_raw.json
    eval_questions_review.md
  outputs/
    eval_results.csv
    eval_results_partial.csv
    eval_metrics.png
```

## Limitations

- The corpus is limited to selected public documentation.
- The LLM fallback evaluator is not a perfect substitute for human evaluation.
- No reranker is included yet.
- There is no frontend UI.
- The API does not include production authentication or rate limiting.

## Future Work

The next priority is adding a reranker to test whether post-retrieval ordering improves context precision and answer relevancy.

Future improvements include human evaluation, LangSmith tracing, recall@k and hit@k retrieval metrics, a frontend UI, and API deployment.
