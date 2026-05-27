# Evaluation Question Review — Curated

Target: 42 questions across 4 categories
- RAG concepts (chunking, retrieval, embedding): 17
- LangChain usage: 10
- Anthropic/Claude API: 10
- Cross-document / synthesis: 5

---

## Category 1: RAG Concepts — Chunking, Retrieval, Embedding (15)

### 1. What does the chunk_overlap parameter do in RecursiveCharacterTextSplitter?

Ground truth: chunk_overlap sets the target overlap between chunks, which helps mitigate information loss when context is divided across chunk boundaries.

- [x] keep

---

### 2. How does RecursiveCharacterTextSplitter.split_text differ from create_documents?

Ground truth: split_text returns the string content chunks directly, while create_documents creates LangChain Document objects for downstream tasks.

- [x] keep

---

### 3. What failure modes are avoided by splitting a large document before embedding and retrieval?

Ground truth: Splitting avoids exceeding many models' finite context windows and reduces the chance that models struggle to find relevant information in very long inputs.

- [x] keep

---

### 4. How does chunking a document help ground generation at run time in the RAG workflow?

Ground truth: The document is split into smaller chunks for embedding and vector storage so retrieval can return only the most relevant parts for the model at run time.

- [x] keep

---

### 5. Which LangChain splitter API is used for generic text, and what configuration is shown for splitting a loaded document?

Ground truth: RecursiveCharacterTextSplitter is used with chunk_size=1000, chunk_overlap=200, and add_start_index=True.

- [x] keep

---

### 6. What is a VectorStore in the LangChain RAG pipeline?

Ground truth: A VectorStore is a wrapper around a vector database used for storing and querying embeddings.

- [x] keep

---

### 7. How do you initialize a local Chroma vector store with data persistence using OpenAI embeddings?

Ground truth: Install langchain-chroma, create an OpenAIEmbeddings instance, then instantiate Chroma with a collection_name, embedding_function, and persist_directory such as "./chroma_langchain_db".

- [x] keep

---

### 8. How do you create a LangChain Chroma vector store from an existing chromadb client after creating or accessing a collection?

Ground truth: Create or access the collection with client.get_or_create_collection("collection_name"), optionally add documents, then instantiate Chroma with client, collection_name, and embedding_function.

- [x] keep

---

### 9. A local Chroma server is running with `chroma run`, but the code uses persist_directory and does not connect to the server. What is the likely issue and what should be used instead?

Ground truth: persist_directory configures a local persistent database, not a server connection. To connect to the running server, initialize Chroma with host="localhost"; for lower-level client initialization use chromadb.HttpClient(host="localhost", port=8000).

- [x] keep

---

### 10. Which Chroma vector store API searches using an embedding vector directly, and how is the query embedding produced?

Ground truth: Use vector_store.similarity_search_by_vector; the embedding is produced with embeddings.embed_query("query string") and passed as the embedding argument.

- [x] keep

---

### 11. What is indirect prompt injection in RAG applications?

Ground truth: Indirect prompt injection occurs when retrieved documents contain instruction-like text that the model may follow because the retrieved data shares the same context window as the system prompt.

- [x] keep

---

### 12. How do defensive prompts and context delimiters differ as mitigations for indirect prompt injection?

Ground truth: Defensive prompts explicitly tell the model to treat retrieved context as data and ignore instructions within it, while context delimiters use structural markers such as XML tags to separate retrieved data from instructions.

- [x] keep

---

### 13. What failure modes or trade-offs occur when using agentic RAG where the LLM decides whether to call the search tool?

Ground truth: When search is performed, it requires two inference calls: one to generate the query and one to produce the final response. The LLM also has reduced control guarantees because it may skip searches when needed or issue extra searches when unnecessary.

- [x] keep

---

### 14. How does the local storage configuration for Chroma differ from the Milvus configuration?

Ground truth: Chroma uses persist_directory="./chroma_langchain_db" to save data locally, while Milvus sets URI="./milvus_example.db" and passes it via connection_args={"uri": URI} with index_params={"index_type": "FLAT", "metric_type": "L2"}.

- [x] keep

---

### 15. What prompt-grounding rules should a RAG prompt include to constrain the model's answer?

Ground truth: The prompt should tell the model to use retrieved pieces to answer, say it does not know if the answer is unknown or relevant information is missing, keep answers concise, and treat retrieved material as data only rather than following any instructions inside it.

- [x] keep

---

### 16. Why can dense retrieval miss important technical keywords even when the query is semantically close?

Ground truth: Dense retrieval embeds queries and chunks into continuous vector spaces and matches by semantic similarity. Technical terms like specific API names, parameter names, or method signatures may not cluster near semantically related chunks if the embedding model generalizes over surface form, causing keyword-critical results to rank below semantically adjacent but less precise chunks. BM25-based hybrid retrieval mitigates this by adding exact keyword matching.

- [x] keep (naturalistic phrasing)

---

### 17. My RAG answers keep cutting off mid-explanation. Could this be a chunking problem?

Ground truth: Yes. Fixed-size chunking can split conceptual explanations at arbitrary character boundaries, so the retrieved chunk contains only part of the explanation. Increasing chunk_size, adding chunk_overlap, or switching to semantic chunking that respects sentence or paragraph boundaries can reduce mid-explanation cuts.

- [x] keep (naturalistic phrasing)

---

## Category 2: LangChain Usage (10)

### 16. What is LangChain's standard model interface and why does it matter?

Ground truth: It standardizes interactions with different model providers so users can swap providers seamlessly and avoid lock-in despite provider-specific APIs and response formats.

- [x] keep

---

### 17. What steps are used to create a minimal RAG agent with a retrieve_context tool?

Ground truth: Define retrieve_context with @tool(response_format="content_and_artifact"), call vector_store.similarity_search(query, k=2), serialize each retrieved document with metadata and page content, return both the serialized text and raw documents, add the tool to tools, write a system prompt, and call create_agent(model, tools, system_prompt=prompt).

- [x] keep

---

### 18. How is the retrieve_context tool implemented and what does it return?

Ground truth: retrieve_context is decorated with @tool(response_format="content_and_artifact"), accepts a query string, runs vector_store.similarity_search(query, k=2), serializes each retrieved document's metadata and page content, and returns both the serialized string and the retrieved documents.

- [x] keep

---

### 19. What should a RAG agent do if retrieved context lacks relevant information or contains instructions?

Ground truth: It should say that it does not know when retrieved context is not relevant, treat retrieved context as data only, and ignore any instructions contained within it.

- [x] keep

---

### 20. How does the two-step RAG chain inject retrieved content into the model prompt using LangChain APIs?

Ground truth: It defines a @dynamic_prompt function taking a ModelRequest, reads the last user message, runs vector_store.similarity_search, joins each retrieved document's page_content, and returns a system message containing that content. The agent is created with create_agent(model, tools=[], middleware=[prompt_with_context]).

- [x] keep

---

### 21. How does the RAG agent implementation differ from the two-step RAG chain for retrieval and generation?

Ground truth: The RAG agent executes searches through a retrieval tool and is a general-purpose implementation, while the two-step RAG chain uses a single LLM call per query and is fast and effective for simple queries.

- [x] keep

---

### 22. What steps are described for using a LangChain provider integration and swapping providers consistently?

Ground truth: Install the dedicated langchain-<provider> package, choose a model name, and use LangChain's standard interfaces so providers can be swapped without changing application code.

- [x] keep

---

### 23. If a user asks how to inspect what is happening inside a multi-step LangChain RAG chain or agent, what setup should be recommended?

Ground truth: Use LangSmith, sign up, then enable tracing by setting LANGSMITH_TRACING="true" and LANGSMITH_API_KEY as environment variables or in Python.

- [x] keep

---

### 24. What is LangSmith used for in a LangChain RAG setup?

Ground truth: LangSmith is used to log traces and inspect what is happening inside complex LangChain chains or agents with multiple steps and LLM calls.

- [x] keep

---

### 25. What does RetrieveDocumentsMiddleware.before_model return after calling vector_store.similarity_search on the last message text?

Ground truth: It returns a dictionary with an updated messages list containing the last message augmented with retrieved document content and grounding instructions, plus a context field containing the retrieved docs.

- [x] keep

---

## Category 3: Anthropic / Claude API (10)

### 26. What is the Messages API, and what use case is it best for?

Ground truth: The Messages API provides direct model prompting access and is best for custom agent loops and fine-grained control.

- [x] keep

---

### 27. Before starting prompt engineering for a Claude use case, what should you establish?

Ground truth: Establish clear success criteria, empirical tests against those criteria, and a first draft prompt to improve. If you do not have a draft, use the prompt generator in the Claude Console.

- [x] keep

---

### 28. Why might prompt engineering be the wrong fix for a failing evaluation related to latency or cost?

Ground truth: Latency and cost are not always best controlled through prompt engineering and may be more easily improved by selecting a different model.

- [x] keep

---

### 29. What is adaptive thinking in the Claude API?

Ground truth: Adaptive thinking is a mode configured with thinking: {type: "adaptive"} where Claude determines when and how much extended thinking to use, optionally guided by an effort parameter.

- [x] keep

---

### 30. How does adaptive thinking differ from manual thinking for Claude Opus 4.7?

Ground truth: Adaptive thinking is the only supported mode on Claude Opus 4.7, where type: "enabled" with budget_tokens is rejected. On Opus 4.6 and Sonnet 4.6, manual thinking with budget_tokens still works but is deprecated; adaptive mode is recommended.

- [x] keep

---

### 31. What should you do if responses using adaptive thinking stop with stop_reason: "max_tokens"?

Ground truth: Increase max_tokens to give Claude more room, or lower the effort level because high and max effort can cause more extensive thinking and exhaust the total output budget.

- [x] keep

---

### 32. A request to claude-opus-4-7 uses thinking: {"type": "enabled", "budget_tokens": 4096} and receives a 400 error. What caused the failure?

Ground truth: Claude Opus 4.7 rejects manual thinking with budget_tokens because adaptive thinking is its only supported thinking mode. Use thinking: {"type": "adaptive"} instead.

- [x] keep

---

### 34. Why might a response using adaptive thinking have billed output tokens that do not match the visible token count?

Ground truth: Billing is based on the full thinking process, not just the tokens visible in the response.

- [x] keep

---

### 35. What does Claude adaptive thinking mode do, and what happens to data under a Zero Data Retention arrangement?

Ground truth: Adaptive thinking lets Claude dynamically decide when and how much to use extended thinking. With a ZDR arrangement, data sent through the feature is not stored after the API response is returned.

- [x] keep

---

## Category 4: Cross-Document / Synthesis (5)

### 36. When should you use a RAG agent versus a two-step RAG chain, and how does this relate to inference cost?

Ground truth: Use a two-step chain for simple queries where speed and cost matter — it uses one LLM call. Use an agent when the query requires multiple retrieval steps or tool calls, accepting that it requires two or more inference calls and has less predictable control flow.

- [x] keep (synthesis of LangChain RAG docs + Claude API cost considerations)

---

### 37. How does chunking strategy interact with retrieval quality, and what tradeoff does semantic chunking introduce?

Ground truth: Fixed chunking is stable and reproducible but can cut through conceptual explanations. Semantic chunking keeps related ideas together but costs more during ingestion and can produce less uniform chunks, which affects embedding and retrieval consistency.

- [x] keep (synthesis of chunking + retrieval docs)

---

### 38. What are the end-to-end steps to build a RAG pipeline over a set of documentation pages?

Ground truth: Load pages, clean boilerplate, chunk with RecursiveCharacterTextSplitter, embed with OpenAI embeddings, store in a persistent Chroma vector store, implement retrieval (dense or hybrid), inject retrieved context into the model prompt, and evaluate with faithfulness, answer relevancy, and context precision metrics.

- [x] keep (full pipeline synthesis)

---

### 39. How does indirect prompt injection interact with agentic RAG, and what mitigations apply?

Ground truth: In agentic RAG, the LLM processes retrieved content inside its context window and may follow instructions embedded in retrieved documents. Mitigations include defensive system prompts that tell the model to treat retrieved content as data, context delimiters such as XML tags, and grounding rules that instruct the model to say it does not know rather than follow injected instructions.

- [x] keep (synthesis of injection + agent docs)

---

### 40. I'm building a RAG chatbot and someone pointed out that my retrieved docs could contain adversarial instructions. How serious is this and what can I do about it?

Ground truth: This is the indirect prompt injection problem. It is a real risk in RAG systems because retrieved content shares the model's context window with the system prompt. Mitigations include defensive system prompts instructing the model to treat retrieved content as data only, using XML or other structural delimiters to separate retrieved content from instructions, and configuring the model to respond with "I don't know" when retrieved context is insufficient rather than attempting to infer or follow embedded instructions.

- [x] keep (naturalistic phrasing, synthesis)

---

### 41. When would you recommend the Message Batches API and when would you not, particularly for a Zero Data Retention workflow requiring immediate responses?

Ground truth: The Message Batches API is asynchronous and intended for tasks where immediate responses are not required. It is not eligible for Zero Data Retention, so it should not be recommended for ZDR workflows or any use case requiring real-time responses.

- [x] keep (synthesis of Anthropic API docs)

---

## Dropped Questions (not included)

- 14: Azure-specific env var names — too provider-specific
- 17: NVIDIA Embeddings init steps — too implementation-specific
- 18: OpenSearch + AWS auth details — too infra-specific
- 32: Separator list including Unicode variants — too trivial
- 36: OpenRouter vs Perplexity package names — too trivial
- 45: Streaming + thinking encryption signature delivery — too API-detail specific
- 46: Billed vs visible tokens with thinking.display omitted — consolidated into Q34
- 47: Whether thinking model sees summarized output — too narrow
- 41: Azure chat model init with init_chat_model — too provider-specific
- 16: Chroma vs Milvus local storage — kept as Q14 above, duplicate removed
- Original Q32 (direct user to claude.ai) — too trivial, does not test retrieval quality
