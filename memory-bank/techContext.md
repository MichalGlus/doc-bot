# Tech Context

## Stack
- Python, venv in `.venv/` (Windows).
- LangChain 1.x (`langchain`, `langchain_core`, `langchain_community` BM25Retriever, `langchain_text_splitters`, `langchain-google-genai`), `google-genai` client.
- `pymupdf` for PDF text + page rendering (200 dpi). **Installed in the venv but missing from `requirements.txt`** (it is in `requirements1.txt`, a full pip freeze). BM25 also needs `rank_bm25`.
- Streamlit UI (`st.html`, `st.tabs`, `st.form`, `width="stretch"`).

## Models (defaults, override via env)
| Role | Env var | Default |
|---|---|---|
| Vision transcription | `RAG_VISION_MODEL` | `google_genai:gemini-3.5-flash-lite` |
| Answer | `RAG_ANSWER_MODEL` | `google_genai:gemini-3.5-flash-lite` |
| Embeddings | `RAG_EMBEDDING_MODEL` | `google_genai:gemini-embedding-2` |

Requires `GOOGLE_API_KEY`. `ollama:` models are supported (no rate-limit waits).

## Other env knobs
`RAG_FAKE`, `RAG_MAX_API_CALLS` (60), `RAG_MAX_PAGES` (30), `RAG_SECONDS_BETWEEN_CALLS` (4.5), `RAG_EMBED_BATCH` (90), `RAG_EMBED_WAIT_SECONDS` (62).

## Commands
- `python check_models.py` — list visible models + 2 tiny test calls (run before a real session).
- `python rag_skeleton.py docs/` — CLI: build, run eval, interactive Q&A.
- `streamlit run app.py` — UI.
- `python make_samples.py` — generates synthetic "aged" sample scans into `docs/` (needs pillow, python-bidi; uses Linux font paths).

## Files
- `docs/` — source documents (synthetic samples: letter_lodz_1939, residents_list_1941, bescheinigung_1942, karta_1945, testimony_318; plus real Hebrew PDFs/images).
- `transcripts/` — all caches (deleting them = paying again).
- `eval_set.json` — 8 cases `{question, expected, source}`; one is a deliberate "not found" case.
- `how_it_works.html`, `interview_prep.html` — standalone Hebrew explainer pages.
- Not a git repo.
