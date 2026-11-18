# System Patterns

## Architecture (all logic in `rag_skeleton.py`, UI in `app.py`)
```
docs/ → list_pages → per page:
          pdf_text_layer (free)  ──┐
          else transcript cache  ──┼→ Document(source, page, method)
          else vision LLM (paid) ──┘
      → split (Recursive, 800/150) → chunk_id
      → HybridRetriever: InMemoryVectorStore (embeddings) + BM25 → RRF (1/(60+rank)), TOP_K=5
      → answer(): ANSWER_PROMPT with "[file, עמוד N]" context → reply + hits
      → evaluate(): per case: retrieved? (source in hits) / answered? (expected substring in reply)
```

## Key patterns
- **Budget object**: every API call goes through `budget.spend(what)`; raises `BudgetExceeded` at `MAX_API_CALLS`. Shared across Streamlit reruns via `st.cache_resource`.
- **Stop on first error**: `load_documents` returns `(docs, error)` and breaks on the first API failure — never "try the next page". `max_retries=1`.
- **Disk caches in `transcripts/`**:
  - `{filename}_p{N}.json` → `{"text", "method": "text"|"vision"}` (missing method = vision, legacy).
  - `embcache_{model}.json` → sha1(chunk) → vector; saved after every batch so a 429 mid-way resumes.
  - No index file: the vector store is rebuilt in memory from `embcache` (only new chunks hit the API).
- **App auto-load**: on open, app.py builds the index from free sources (text layer + cached transcripts, `load_documents(..., vision_llm=None)`) when `embeddings.missing(...) == 0`; the build button is disabled when nothing is new.
- **Text layer beats vision**: a PDF page with a usable text layer (≥50 letters and ≥50% of visible chars are letters) overwrites any older vision transcript. `_fix_hebrew_pdf_text` repairs reversed parentheses.
- **Embedding batching**: the free tier counts each chunk as a request (100/min) → batches of 90 + 62 s wait. No wait for `ollama:` or FAKE.
- **Chat rate limiting**: `InMemoryRateLimiter` at 1 call / 4.5 s (skipped for ollama).
- **Models via `init_chat_model` / `init_embeddings`** with `"provider:model"` strings, overridable by env vars.
- **FAKE mode** (`RAG_FAKE=1`): fake chat + deterministic fake embeddings, zero API calls; separate caches.
- **UI safety**: all model/document text passes through `esc()` before HTML; citations `(file, עמוד N)` are regex-rendered as tags.
