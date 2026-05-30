# Agent Evaluation Analysis

Command requested:

```bash
.venv/bin/python scripts/run_agent_eval.py --retriever hybrid --limit 10
```

The first attempt resumed from existing checkpoints but failed on question 4 with
an OpenAI `APIConnectionError` caused by network/DNS access. The same command was
rerun with network access and resumed successfully from
`outputs/agent_eval_partial.csv`.

Completed sample:

| Item | Value |
|---|---:|
| Questions | 10 |
| Rows | 20 |
| Systems compared | `baseline_rag`, `agentic_rag` |

## Summary

| Metric | Baseline RAG | Agentic RAG | Agentic - Baseline |
|---|---:|---:|---:|
| Answer relevancy | 0.900 | 1.000 | +0.100 |
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Total latency (s) | 5.236 | 4.392 | -0.844 |
| Retrieval success rate | 1.000 | 1.000 | +0.000 |
| Query rewrite rate | 0.000 | 0.000 | +0.000 |

## Rewrite Trigger Rate

Query rewriting did not trigger in this 10-question sample.

| System | Rewrite Trigger Rate |
|---|---:|
| Agentic RAG | 0.000 |

Questions where rewrite was triggered:

| Question |
|---|
| None |

## Per-Question Results

| Question | Baseline Relevancy | Agent Relevancy | Baseline Faithfulness | Agent Faithfulness | Baseline Latency (s) | Agent Latency (s) | Rewrite Used |
|---|---:|---:|---:|---:|---:|---:|---|
| What does the chunk_overlap parameter do in RecursiveCharacterTextSplitter? | 1.000 | 1.000 | 1.000 | 1.000 | 11.581 | 1.732 | False |
| How does RecursiveCharacterTextSplitter.split_text differ from create_documents? | 1.000 | 1.000 | 1.000 | 1.000 | 3.810 | 4.735 | False |
| What failure modes are avoided by splitting a large document before embedding and retrieval? | 1.000 | 1.000 | 1.000 | 1.000 | 3.958 | 4.733 | False |
| How does chunking a document help ground generation at run time in the RAG workflow? | 1.000 | 1.000 | 1.000 | 1.000 | 4.481 | 4.844 | False |
| Which LangChain splitter API is used for generic text, and what configuration is shown for splitting a loaded document? | 1.000 | 1.000 | 1.000 | 1.000 | 3.492 | 3.452 | False |
| What is a VectorStore in the LangChain RAG pipeline? | 1.000 | 1.000 | 1.000 | 1.000 | 3.364 | 3.894 | False |
| How do you initialize a local Chroma vector store with data persistence using OpenAI embeddings? | 1.000 | 1.000 | 1.000 | 1.000 | 2.959 | 2.293 | False |
| How do you create a LangChain Chroma vector store from an existing chromadb client after creating or accessing a collection? | 0.000 | 1.000 | 1.000 | 1.000 | 12.432 | 11.198 | False |
| A local Chroma server is running with `chroma run`, but the code uses persist_directory and does not connect to the server. What is the likely issue and what should be used instead? | 1.000 | 1.000 | 1.000 | 1.000 | 3.917 | 4.304 | False |
| Which Chroma vector store API searches using an embedding vector directly, and how is the query embedding produced? | 1.000 | 1.000 | 1.000 | 1.000 | 2.365 | 2.733 | False |

## Interpretation

Agentic RAG had higher average answer relevancy in this sample, but the
difference is driven by one baseline answer receiving a relevancy score of 0.000.
Faithfulness was tied at 1.000 for both systems.

Retrieval success was also tied at 100%, so this run does not show a reliability
improvement from the agentic workflow on retrieval success.

Agentic RAG was faster on average by 0.844 seconds in this run. This should be
interpreted cautiously because LLM latency varies and query rewriting did not
trigger, so the agentic workflow did not exercise its extra rewrite call.

Query rewriting was not used for any of the 10 completed questions. This means
the run mainly compares baseline retrieve-generate behavior against the
agentic graph path without rewrite intervention.
