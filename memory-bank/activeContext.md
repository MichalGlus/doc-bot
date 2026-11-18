# Active Context

_Last updated: 2026-11-17_

## Current focus
Preparing for a job interview around this project (`interview_prep.html`, 2026-11-17). The core pipeline works; remaining work is the quality improvements below.

## Recent changes
- Free PDF text-layer extraction; the text layer overrides old vision transcripts; a UI tag shows where each page's text came from ("נשלף מהקובץ" vs "תועתק במודל").
- Embedding batching + per-batch disk cache to survive 429 RESOURCE_EXHAUSTED.
- `needs_vision()` so the sidebar cost estimate counts only scanned, untranscribed pages.

## Known open issues
1. Two questions in one message → a single search, only one gets answered, and the whole reply is flagged "not found".
2. Conceptual questions ("what is X?") get "not found": strict prompt + only 5 chunks.
3. `pymupdf` missing from `requirements.txt`.

## Next steps (candidates)
- Query decomposition (LLM splits a message into sub-questions, each retrieved + answered separately).
- Two answer modes with routing: archive mode (strict, cited) vs understanding mode (more chunks, explanatory prompt).
