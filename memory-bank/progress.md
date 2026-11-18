# Progress

## Works
- [x] Transcription: text layer (free) → cache → vision LLM, with page cap and stop-on-error
- [x] Uncertainty markers in transcripts, highlighted in the UI
- [x] Chunking with source/page/method metadata
- [x] Hybrid retrieval (vector + BM25, RRF), persisted vector index
- [x] Grounded answers with citations + "לא נמצא במסמכים"
- [x] Eval: retrieval vs answer scored separately (CLI + UI tab)
- [x] Budget guard, rate limiting, embedding batching, disk caches, FAKE mode
- [x] Streamlit RTL UI: Q&A history, transcript-vs-image review, eval, API meter
- [x] Explainer + interview-prep HTML pages

## Not done
- [ ] Query decomposition for multi-question messages
- [ ] Archive / understanding answer modes with routing
- [ ] Add `pymupdf` (and `rank_bm25`) to `requirements.txt`
- [ ] Metadata enrichment (year, doc type, language) — placeholder comment in `load_documents`

## Lessons learned (interview material)
- The page cap silently left only 11/79 pages of the book indexed → "not found" for an obvious question. Fix: free text-layer extraction; the cap now applies only to scanned pages.
- The vision model misread Rashi script and merged two columns without marking uncertainty → prefer the text layer whenever it exists.
- 517 chunks sent to embeddings at once → 429 on the free tier (each chunk = 1 request, 100/min) → batches of 90 + 62 s wait + per-batch cache.
