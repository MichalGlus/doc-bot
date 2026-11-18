# Project Brief — doc-agent

## What
A RAG agent that answers questions about **old scanned archive documents** (Hebrew + multilingual: Yiddish, German, Polish), with every fact cited back to (file, page).

## Why
Built as a portfolio / job-interview project (see `interview_prep.html`). Must be explainable end-to-end and run on the **Google Gemini free tier** without blowing quota.

## Core requirements
1. Transcribe documents: free text-layer extraction for digital PDFs; vision LLM only for scanned pages/images.
2. Answer **only** from retrieved chunks, cite `(file, page)` after each fact, say "לא נמצא במסמכים" when absent — never guess names/dates/numbers.
3. Hybrid retrieval (vector + BM25) because names/places need exact match.
4. Hard cost guards: capped API calls per run, capped new pages per run, everything cached to disk.
5. Measurable quality: eval set scoring retrieval and answer separately.
6. Streamlit UI in Hebrew (RTL) with an archive look.

## Out of scope (for now)
Agentic tool use, query decomposition, multi-mode answering (see `progress.md`).
