# Evaluation Question Classification

This is a static analysis of `data/eval_questions.json`. No evaluation was run.

Rewrite prediction uses the current `src.agent.nodes.analyze_query` heuristic:

- trigger rewrite if the question has fewer than 5 word tokens, or
- trigger rewrite if it contains one of: `it`, `this`, `that`, `they`, `those`, `these`

Question type is a qualitative estimate:

- `simple retrieval`: answer likely lives in one focused retrieved chunk
- `multi-hop`: answer likely requires comparing, combining, or sequencing multiple facts
- `low-confidence`: higher risk that retrieval returns incomplete context or that the answer depends on narrow/version-specific details

## Summary

| Category | Count |
|---|---:|
| Total questions | 42 |
| Predicted rewrite triggers | 7 |
| Simple retrieval | 15 |
| Multi-hop | 21 |
| Low-confidence | 6 |
| High likely agent benefit | 12 |
| Medium likely agent benefit | 13 |
| Low likely agent benefit | 17 |

## Questions Most Likely To Benefit From Agent Behavior

These questions are most likely to benefit from query analysis, rewrite/fallback behavior, or multi-step reasoning because they are comparative, procedural, ambiguous, or risk incomplete retrieval.

| # | Why It May Benefit |
|---:|---|
| 9 | Requires detecting a configuration mismatch and recommending the correct alternative. |
| 13 | Asks for multiple failure modes/tradeoffs of agentic RAG behavior. |
| 16 | Requires explaining a subtle retrieval failure mode and BM25 mitigation. |
| 17 | User-style diagnostic query; rewrite is predicted and the answer requires interpreting symptoms. |
| 20 | Rewrite is predicted because of `it`; question needs tool implementation details and return shape. |
| 21 | Requires fallback behavior and prompt-injection handling. |
| 25 | Asks for a recommendation/setup for inspecting multi-step chains or agents. |
| 35 | Rewrite is predicted because of `that`; asks about a non-obvious billing/token accounting distinction. |
| 37 | Rewrite is predicted because of `this`; asks for a tradeoff between RAG agent and two-step chain. |
| 39 | Requires assembling an end-to-end pipeline from multiple steps. |
| 40 | Combines indirect prompt injection with agentic RAG and mitigations. |
| 41 | Conversational/adversarial-risk question; rewrite is predicted and likely benefits from clearer retrieval framing. |

## Full Classification

| # | Rewrite Predicted | Type | Likely Agent Benefit | Rationale |
|---:|---|---|---|---|
| 1 | No | simple retrieval | Low | Direct definition of a named parameter. |
| 2 | No | simple retrieval | Low | Direct API comparison between two named methods. |
| 3 | No | multi-hop | Medium | Requires listing multiple failure modes avoided by chunking. |
| 4 | No | multi-hop | Medium | Connects chunking, embedding/vector storage, retrieval, and grounding. |
| 5 | No | multi-hop | Medium | Requires identifying an API plus several configuration values. |
| 6 | No | simple retrieval | Low | Direct definition of VectorStore. |
| 7 | No | multi-hop | Medium | Procedural setup question with multiple initialization components. |
| 8 | No | multi-hop | Medium | Requires sequencing client, collection, optional add, and Chroma wrapper steps. |
| 9 | No | multi-hop | High | Requires diagnosing local persistence vs server connection and recommending alternatives. |
| 10 | No | multi-hop | Medium | Requires API identification plus how the embedding is produced. |
| 11 | No | simple retrieval | Low | Direct definition of indirect prompt injection. |
| 12 | No | multi-hop | Medium | Requires contrasting two mitigation approaches. |
| 13 | No | multi-hop | High | Requires multiple tradeoffs of agent-controlled search behavior. |
| 14 | No | multi-hop | Medium | Requires comparing Chroma and Milvus local storage configuration. |
| 15 | No | multi-hop | Medium | Requires enumerating prompt-grounding rules. |
| 16 | No | multi-hop | High | Requires nuanced explanation of dense retrieval failure and keyword mitigation. |
| 17 | Yes | low-confidence | High | User-style symptom query; rewrite should clarify the technical target and retrieval may need diagnostic context. |
| 18 | Yes | simple retrieval | Medium | Rewrite predicted by `it`; otherwise a direct definition/value question. |
| 19 | No | multi-hop | Medium | Requires ordered steps for creating a minimal RAG agent. |
| 20 | Yes | multi-hop | High | Rewrite predicted by `it`; answer needs implementation details and return values. |
| 21 | No | multi-hop | High | Requires fallback behavior plus instruction-handling behavior. |
| 22 | No | multi-hop | Medium | Requires describing a two-step chain with middleware/prompt injection flow. |
| 23 | No | multi-hop | Medium | Requires contrasting RAG agent and two-step RAG chain. |
| 24 | No | multi-hop | Medium | Requires procedural provider setup and provider swapping rationale. |
| 25 | No | low-confidence | High | Recommendation-style observability setup for multi-step chains/agents. |
| 26 | No | simple retrieval | Low | Direct question about LangSmith purpose. |
| 27 | No | simple retrieval | Low | Direct API/middleware return-shape question with named method. |
| 28 | Yes | simple retrieval | Medium | Rewrite predicted by `it`; direct definition/use-case question. |
| 29 | No | simple retrieval | Low | Direct checklist-style prompt engineering prerequisite question. |
| 30 | No | multi-hop | Medium | Requires explaining when prompt engineering is not the right lever. |
| 31 | No | simple retrieval | Low | Direct definition of adaptive thinking. |
| 32 | No | multi-hop | Medium | Requires comparing adaptive and manual thinking across model versions. |
| 33 | No | simple retrieval | Low | Direct troubleshooting instruction for `max_tokens` stop reason. |
| 34 | No | low-confidence | Medium | Version-specific error cause involving Claude Opus 4.7 thinking settings. |
| 35 | Yes | low-confidence | High | Rewrite predicted by `that`; billing vs visible-token accounting is narrow and easy to miss. |
| 36 | No | multi-hop | Medium | Requires combining adaptive thinking behavior with ZDR data handling. |
| 37 | Yes | multi-hop | High | Rewrite predicted by `this`; requires cost/speed tradeoff between agent and two-step chain. |
| 38 | No | multi-hop | Medium | Requires explaining chunking quality interaction and semantic chunking tradeoff. |
| 39 | No | multi-hop | High | End-to-end pipeline question requiring many ordered components. |
| 40 | No | multi-hop | High | Combines indirect prompt injection, agentic RAG risk, and mitigations. |
| 41 | Yes | low-confidence | High | Conversational security-risk question; rewrite should improve retrieval focus and fallback matters. |
| 42 | No | low-confidence | Medium | Requires nuanced recommendation involving Message Batches, ZDR, and real-time constraints. |

## Notes

- The current rewrite heuristic triggers on vague-reference tokens, so some long questions trigger rewrite solely because they contain `it`, `this`, or `that`.
- The first 10 questions used in the recent `--limit 10` run have no predicted rewrite triggers, which explains why rewrite usage was 0% in that evaluation sample.
- The most useful future evaluation slice for testing agent behavior should include the predicted rewrite-triggering questions: 17, 18, 20, 28, 35, 37, and 41.
