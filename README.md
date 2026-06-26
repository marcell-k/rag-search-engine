# RAG Engine

A Retrieval Augmented Generation pipeline for searching and querying a movie catalogue, built in Python with a hybrid search backend and an optional agentic loop.

## Capabilities

### Search
- **Keyword search** — BM25 inverted index for exact term matching
- **Semantic search** — embedding-based chunked search for themes and concepts
- **Hybrid search** — combines both via Reciprocal Rank Fusion (RRF)
- **Reranking** — optional LLM-based reranking in individual or batch mode

### RAG CLI (`rag`)
Four generation modes over hybrid search results:

| Command | Description |
|---|---|
| `rag "query" rag` | Direct answer from retrieved documents |
| `rag "query" summarize` | Summarise the top results |
| `rag "query" citations` | Answer with inline citations |
| `rag "query" question` | Detailed answer with citations |

### Agentic CLI (`agent`)
The LLM drives its own search loop, picking tools based on what it finds:

- Chooses from keyword, semantic, regex, genre, and actor search
- Accumulates and deduplicates results across iterations
- Stops when it has enough or hits `--max-iterations`
- Produces a final cited answer from everything collected

```
agent "scary bear movies set in a forest"
agent "Hugh Jackman survival films" --limit 8 --max-iterations 4
agent "horror twist endings" --debug
```

## Stack

- **LLM** — Gemini (via `GEMINI_API_KEY`)
- **Embeddings** — HuggingFace sentence transformers
- **Keyword index** — custom BM25 inverted index
- **Package manager** — `uv`

## Roadmap

- [ ] WebSocket-based UI (framework TBD)
- [ ] Migrate data source to K10 documents
- [ ] Rewrite ingestion and retrieval pipeline for K10 schema
