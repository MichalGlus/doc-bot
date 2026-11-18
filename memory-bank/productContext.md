# Product Context

## Users / use case
Researchers or archivists asking factual questions over a folder of scanned historical documents (letters, resident lists, work permits, registration cards, testimonies) and also long digital books (e.g. a 79-page Hebrew halachic PDF, "ספר בעלי הנפש").

## Problems it solves
- Old scans are hard to read and multilingual → vision-model transcription that preserves the original language and marks `[לא קריא]` (unreadable) and `[?word?]` (unsure).
- Trust → answers cite source + page; the UI shows retrieved chunks and the original image next to the transcript so a human can verify.
- Cost anxiety → the UI shows, *before* clicking, how many API calls a build will send, plus a live API-call meter.

## UX principles
- Hebrew, RTL, auto-direction per text (Latin-heavy text → LTR).
- No API call ever happens on page render — only inside explicit button handlers (Streamlit reruns the whole script on every interaction).
- Three tabs: שאלות (Q&A history cards), בדיקת תעתוק (image vs transcript), הערכה (eval run with two meters).
- Visual theme: "paper / ink / stamp-red" archive look (`.streamlit/config.toml` + `style.css`; colors are duplicated in both — change them together).
