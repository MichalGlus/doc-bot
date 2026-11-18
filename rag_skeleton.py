"""
שלד RAG למסמכים סרוקים ישנים (עברית / רב-לשוני) - LangChain 1.x

שלבים:
  1. תעתוק   - PDF שנוצר במחשב: שליפת הטקסט המובנה (בחינם, בלי API).
               עמוד סרוק / תמונה: מודל vision קורא את התמונה (עם cache לדיסק)
  2. Chunking - חלוקה לקטעים, כל קטע שומר מקור + עמוד
  3. אינדקס  - וקטורי (embeddings, נשמר לדיסק) + BM25 (מילות מפתח)
  4. אחזור   - היברידי, מיזוג בשיטת RRF
  5. תשובה   - רק מתוך הקטעים, עם ציטוט מקור, מותר לומר "לא נמצא"
  6. הערכה   - שאלות עם תשובה ידועה (מהקובץ eval_set.json)

הגנות עלות (ראי Budget למטה):
  - תקרה קשיחה למספר קריאות API בריצה אחת, ותקרה למספר עמודים.
  - אין אף לולאה שקוראת ל-API בלי גבול: כל הלולאות רצות על רשימה סופית.
  - שגיאת API עוצרת מיד (לא ממשיכים "לנסות את הדף הבא"), ומקסימום ניסיון חוזר אחד.
  - מה שכבר תועתק / אונדקס נשמר לדיסק ולא נשלח שוב.

הרצה:
  python rag_skeleton.py docs/          # שורת פקודה
  streamlit run app.py                  # ממשק
  RAG_FAKE=1 ...                        # ריצת ניסיון בלי שום קריאת API (מודלים מדומים)
"""

import base64
import hashlib
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# טעינת מפתחות והגדרות מקובץ .env שליד הקוד (GOOGLE_API_KEY, LANGSMITH_*, RAG_*).
# נטען לפני LangChain כדי שהגדרות הניטור של LangSmith יהיו זמינות כבר בייבוא.
# משתנה סביבה שכבר מוגדר במחשב גובר על הקובץ.
load_dotenv(Path(__file__).with_name(".env"))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- הגדרות
# פורמט: "ספק:שם_מודל". אפשר לשנות כאן או דרך משתני סביבה, בלי לגעת בקוד.
VISION_MODEL = os.getenv("RAG_VISION_MODEL", "google_genai:gemini-3.5-flash-lite")
ANSWER_MODEL = os.getenv("RAG_ANSWER_MODEL", "google_genai:gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "google_genai:gemini-embedding-2")

FAKE = os.getenv("RAG_FAKE") == "1"   # מודלים מדומים: אפס קריאות API, אפס עלות

# --- הגנות עלות ---
MAX_API_CALLS = int(os.getenv("RAG_MAX_API_CALLS", "60"))   # סך כל הקריאות בריצה אחת
MAX_PAGES = int(os.getenv("RAG_MAX_PAGES", "30"))           # עמודים חדשים לתעתוק בריצה
SECONDS_BETWEEN_CALLS = float(os.getenv("RAG_SECONDS_BETWEEN_CALLS", "4.5"))  # ~13 בדקה
# embeddings: בשכבה החינמית כל קטע = בקשה, ותקרה של 100 בדקה. 90 בקבוצה + דקה המתנה = בטוח.
EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "90"))
EMBED_WAIT_SECONDS = float(os.getenv("RAG_EMBED_WAIT_SECONDS", "62"))

CACHE_DIR = Path("transcripts")   # כדי לא לשלם פעמיים על אותו דף
EVAL_FILE = Path("eval_set.json")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
NOT_FOUND = "לא נמצא במסמכים"
MIN_TEXT_LETTERS = 50   # פחות אותיות מזה בעמוד PDF = נחשב סרוק, ונשלח למודל התמונה

TRANSCRIBE_PROMPT = """תעתק את הטקסט שבתמונה בדיוק כפי שהוא כתוב.
- שמור על השפה המקורית (עברית, יידיש, גרמנית, פולנית וכו'). אל תתרגם.
- אל תתקן כתיב ואל תשלים מילים חסרות.
- מילה שאינה קריאה סמן [לא קריא]. מילה שאינך בטוח בה סמן [?מילה?].
- טבלה או רשימה - שורה לכל רשומה.
החזר את התעתוק בלבד."""

ANSWER_PROMPT = """ענה על השאלה אך ורק על סמך הקטעים שלמטה.
- אחרי כל עובדה ציין מקור בסוגריים: (קובץ, עמוד).
- אם התשובה אינה מופיעה בקטעים, כתוב: "לא נמצא במסמכים".
- אל תנחש שמות, תאריכים או מספרים.

קטעים:
{context}

שאלה: {question}"""


# ---------------------------------------------------------------- הגנת עלות
class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """מונה קריאות API. כל קריאה עוברת דרך spend(); מעבר לתקרה - עוצרים בשגיאה."""

    def __init__(self, limit: int = MAX_API_CALLS):
        self.limit = limit
        self.used = 0

    def spend(self, what: str):
        if self.used >= self.limit:
            raise BudgetExceeded(
                f"נעצר: הגעת לתקרה של {self.limit} קריאות API בריצה זו (בקריאה: {what}). "
                "כדי להמשיך - הפעילי מחדש, או העלי את RAG_MAX_API_CALLS."
            )
        self.used += 1

    @property
    def left(self) -> int:
        return self.limit - self.used


class CountingEmbeddings(Embeddings):
    """עוטף את מודל ה-embeddings: סופר קריאות בתקציב, שולח בקבוצות, ושומר כל קטע לדיסק.

    בשכבה החינמית של Google כל קטע נספר כבקשה נפרדת (תקרה: 100 בדקה).
    לכן שולחים קבוצות של EMBED_BATCH קטעים עם המתנה ביניהן. כל קבוצה נשמרת
    מיד לדיסק, כך ששגיאה באמצע לא מוחקת את מה שכבר שולם - הבנייה הבאה ממשיכה מאותה נקודה.
    """

    def __init__(self, inner: Embeddings, budget: Budget):
        self.inner, self.budget = inner, budget
        local = FAKE or EMBEDDING_MODEL.startswith("ollama:")
        self.wait = 0 if local else EMBED_WAIT_SECONDS
        self.progress = None   # אפשר להציב פונקציה (done, total) להצגת התקדמות
        safe_model = re.sub(r"\W", "_", EMBEDDING_MODEL)
        self._cache_file = CACHE_DIR / f"embcache_{safe_model}{'_fake' if FAKE else ''}.json"
        self._cache: dict | None = None

    def _load_cache(self) -> dict:
        if self._cache is None:
            try:
                self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                self._cache = {}
        return self._cache

    def _save_cache(self):
        CACHE_DIR.mkdir(exist_ok=True)
        self._cache_file.write_text(json.dumps(self._cache), encoding="utf-8")

    def missing(self, texts) -> int:
        """כמה מהקטעים עוד אין להם embedding שמור (= יישלחו ל-API). בלי שום קריאה."""
        cache = self._load_cache()
        return len({k for k in (hashlib.sha1(t.encode()).hexdigest() for t in texts) if k not in cache})

    def embed_documents(self, texts):
        cache = self._load_cache()
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = list({k: t for k, t in zip(keys, texts) if k not in cache}.items())
        for start in range(0, len(todo), EMBED_BATCH):
            if start > 0 and self.wait:
                time.sleep(self.wait)   # מגבלת הבקשות-לדקה של השכבה החינמית
            batch = todo[start:start + EMBED_BATCH]
            self.budget.spend(f"embeddings ל-{len(batch)} קטעים")
            vectors = self.inner.embed_documents([t for _, t in batch])
            cache.update(zip((k for k, _ in batch), vectors))
            self._save_cache()
            if self.progress:
                self.progress(start + len(batch), len(todo))
        return [cache[k] for k in keys]

    def embed_query(self, text):
        self.budget.spend("embedding לשאלה")
        return self.inner.embed_query(text)


def build_models(budget: Budget):
    """מחזיר (vision_llm, answer_llm, embeddings). ב-RAG_FAKE=1 - מודלים מדומים."""
    if FAKE:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.language_models import FakeListChatModel

        llm = FakeListChatModel(responses=["[מודל מדומה] טקסט לדוגמה: רשימת שמות, ורשה 1941."])
        return llm, llm, CountingEmbeddings(DeterministicFakeEmbedding(size=64), budget)

    from langchain.chat_models import init_chat_model
    from langchain.embeddings import init_embeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter

    # מרווח בין קריאות: שומר על מגבלת הבקשות-לדקה של השכבה החינמית
    def chat(name: str):
        # מודל מקומי (ollama) לא צריך המתנה בין קריאות; גם לא כש-RAG_SECONDS_BETWEEN_CALLS=0
        local = name.startswith("ollama:")
        limiter = None
        if SECONDS_BETWEEN_CALLS > 0 and not local:
            limiter = InMemoryRateLimiter(
                requests_per_second=1 / SECONDS_BETWEEN_CALLS, check_every_n_seconds=0.1, max_bucket_size=1
            )
        # max_retries=1: בלי זה ספריות מנסות שוב ושוב בשקט, וכל ניסיון נספר במכסה
        return init_chat_model(name, temperature=0, max_retries=1, rate_limiter=limiter)

    vision_llm = chat(VISION_MODEL)
    llm = chat(ANSWER_MODEL)
    embeddings = CountingEmbeddings(init_embeddings(EMBEDDING_MODEL), budget)
    return vision_llm, llm, embeddings


# ---------------------------------------------------------------- 1. תעתוק
def list_pages(folder: Path) -> list[tuple[Path, int]]:
    """כל (קובץ, מספר עמוד) בתיקייה - בלי לקרוא לשום מודל."""
    if not folder.is_dir():
        raise FileNotFoundError(f"התיקייה '{folder}' לא קיימת")
    pages = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(path) as pdf:
                pages += [(path, i) for i in range(1, len(pdf) + 1)]
        else:
            pages.append((path, 1))
    return pages


def _fix_hebrew_pdf_text(text: str) -> str:
    """תיקונים קלים לטקסט עברי שנשלף מ-PDF.
    השליפה הופכת לפעמים סוגריים בסוף שורה: 'ביבמות)(סב ב' -> 'ביבמות (סב ב)'."""
    text = re.sub(r"\)\(([^()\n]*?)\s*$", r" (\1)", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)                       # רווחים כפולים
    return "\n".join(line.strip() for line in text.splitlines()).strip()


@lru_cache(maxsize=4096)
def _pdf_text_layer_cached(path_str: str, mtime: float, page_no: int) -> str | None:
    import pymupdf

    with pymupdf.open(path_str) as pdf:
        raw = pdf[page_no - 1].get_text()
    letters = len(re.findall(r"[א-תA-Za-z]", raw))
    visible = len(re.findall(r"\S", raw))
    # מעט מדי אותיות = עמוד סרוק (תמונה בלבד) או ריק.
    # רוב התווים אינם אותיות = גופן עם קידוד שבור (ג'יבריש) - עדיף לתעתק מהתמונה.
    if letters < MIN_TEXT_LETTERS or letters < visible * 0.5:
        return None
    return _fix_hebrew_pdf_text(raw)


def pdf_text_layer(path: Path, page_no: int) -> str | None:
    """הטקסט המובנה של עמוד PDF (קובץ שנוצר במחשב) - בחינם, בלי API.
    מחזיר None אם אין טקסט שמיש, ואז העמוד יישלח למודל התמונה."""
    if path.suffix.lower() != ".pdf":
        return None
    return _pdf_text_layer_cached(str(path), path.stat().st_mtime, page_no)


def needs_vision(path: Path, page_no: int) -> bool:
    """האם העמוד הזה יעלה קריאת API (אין לו תעתוק שמור וגם לא טקסט מובנה)."""
    return read_cache(path, page_no) is None and pdf_text_layer(path, page_no) is None


def page_image(path: Path, page_no: int) -> tuple[bytes, str]:
    """bytes + mime של עמוד אחד."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(path) as pdf:
            return pdf[page_no - 1].get_pixmap(dpi=200).tobytes("png"), "image/png"
    return path.read_bytes(), "image/png" if suffix == ".png" else "image/jpeg"


def cache_path(path: Path, page_no: int) -> Path:
    # שם הקובץ המלא (כולל סיומת) - כדי ש-a.jpg ו-a.pdf לא ידרסו זה את זה
    return CACHE_DIR / f"{path.name}_p{page_no}.json"


def _read_cache_record(path: Path, page_no: int) -> dict | None:
    cache = cache_path(path, page_no)
    if not cache.exists():
        return None
    try:
        record = json.loads(cache.read_text(encoding="utf-8"))
        record["text"]   # חייב להיות
        return record
    except (json.JSONDecodeError, KeyError, TypeError):
        return None   # קובץ cache פגום - נתעתק מחדש


def read_cache(path: Path, page_no: int) -> str | None:
    record = _read_cache_record(path, page_no)
    return record["text"] if record else None


def read_cache_method(path: Path, page_no: int) -> str | None:
    """'text' = נשלף מהקובץ, 'vision' = תועתק במודל. קבצי cache ישנים (בלי שדה) = 'vision'."""
    record = _read_cache_record(path, page_no)
    return record.get("method", "vision") if record else None


def write_cache(path: Path, page_no: int, text: str, method: str):
    cache_path(path, page_no).write_text(
        json.dumps({"text": text, "method": method}, ensure_ascii=False), encoding="utf-8"
    )


def transcribe(image: bytes, mime: str, vision_llm, budget: Budget) -> str:
    budget.spend("תעתוק עמוד")
    b64 = base64.b64encode(image).decode()
    msg = HumanMessage(
        content=[
            {"type": "text", "text": TRANSCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    return vision_llm.invoke([msg]).text


def load_documents(folder: Path, vision_llm, budget: Budget, progress=None):
    """מחזיר (docs, error). error הוא None אם הכול עבר, אחרת הודעה על מה שעצר.

    עוצר בשגיאה הראשונה ולא ממשיך לדפים הבאים: אם המכסה נגמרה, אין טעם
    לשלוח עוד בקשות. מה שכבר תועתק שמור ב-cache וייטען בפעם הבאה בחינם.

    vision_llm=None: רק מה שבחינם (טקסט מובנה + תעתוקים שמורים). עמוד סרוק שעוד
    לא תועתק מדולג, בלי קריאת API.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    docs, error, new_pages = [], None, 0
    pages = list_pages(folder)
    for n, (path, page_no) in enumerate(pages, start=1):
        # 1. טקסט מובנה ב-PDF: בחינם ומדויק. גובר גם על תעתוק ישן מהמודל של אותו עמוד.
        digital = pdf_text_layer(path, page_no)
        if digital is not None:
            text, method = digital, "text"
            if read_cache(path, page_no) != digital or read_cache_method(path, page_no) != "text":
                write_cache(path, page_no, digital, "text")
        else:
            text, method = read_cache(path, page_no), "vision"
        # 2. אין טקסט מובנה ואין תעתוק שמור: שולחים את התמונה למודל (עולה קריאת API)
        if text is None:
            if vision_llm is None:
                continue
            if new_pages >= MAX_PAGES:
                error = f"נעצר אחרי {MAX_PAGES} עמודים חדשים (RAG_MAX_PAGES). הריצי שוב להמשך."
                break
            try:
                image, mime = page_image(path, page_no)
                text = transcribe(image, mime, vision_llm, budget)
            except Exception as e:  # מכסה, רשת, קובץ פגום - עוצרים, לא מנסים שוב
                error = f"נעצר ב-{path.name} עמוד {page_no}: {type(e).__name__}: {e}"
                break
            new_pages += 1
            if text.strip():   # לא שומרים תעתוק ריק - אחרת הוא "יינעל" ב-cache
                write_cache(path, page_no, text, "vision")
        if progress:
            progress(n, len(pages), path.name)
        if text.strip():
            # כאן המקום להוסיף מטא-דאטה: שנה, סוג מסמך, שפה
            docs.append(Document(
                page_content=text, metadata={"source": path.name, "page": page_no, "method": method}
            ))
    return docs, error


# ---------------------------------------------------------------- 2. Chunking
def split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


# ---------------------------------------------------------------- 3+4. אינדקס ואחזור היברידי
def tokenize(text: str) -> list[str]:
    """ל-BM25: מילים בלי סימני פיסוק (ברירת המחדל מפצלת רק לפי רווח, ואז 'ורשה,' != 'ורשה')."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """וקטורי (משמעות) + BM25 (מילים מדויקות: שמות, מקומות), מיזוג ב-RRF."""

    def __init__(self, chunks: list[Document], embeddings):
        if not chunks:
            raise ValueError("אין טקסט לאנדקס: לא נמצאו מסמכים, או שכל התעתוקים ריקים.")
        from langchain_community.retrievers import BM25Retriever

        # אין צורך לשמור את האינדקס עצמו לדיסק: כל embedding כבר שמור לפי הקטע
        # (CountingEmbeddings), כך שבנייה מחדש שולחת ל-API רק קטעים חדשים.
        self.vector = InMemoryVectorStore.from_documents(chunks, embeddings)
        self.bm25 = BM25Retriever.from_documents(chunks, k=TOP_K * 2, preprocess_func=tokenize)

    def search(self, query: str, k: int = TOP_K) -> list[Document]:
        ranked_lists = [
            self.vector.similarity_search(query, k=k * 2),
            self.bm25.invoke(query),
        ]
        # Reciprocal Rank Fusion: ציון = סכום של 1/(60+דירוג) בכל רשימה
        scores, by_id = {}, {}
        for results in ranked_lists:
            for rank, doc in enumerate(results):
                cid = doc.metadata["chunk_id"]
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
                by_id[cid] = doc
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [by_id[cid] for cid in best]


# ---------------------------------------------------------------- 5. תשובה
def answer(question: str, retriever: HybridRetriever, llm, budget: Budget):
    hits = retriever.search(question)
    if not hits:
        return NOT_FOUND, []
    context = "\n\n".join(
        f"[{d.metadata['source']}, עמוד {d.metadata['page']}]\n{d.page_content}" for d in hits
    )
    budget.spend("תשובה לשאלה")
    reply = llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).text
    return reply, hits


# ---------------------------------------------------------------- 6. הערכה
def load_eval_set() -> list[dict]:
    """eval_set.json: [{"question": "...", "expected": "מילה שחייבת להופיע", "source": "file.jpg"}]"""
    if not EVAL_FILE.exists():
        return []
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    return [c for c in cases if {"question", "expected", "source"} <= set(c)]


def evaluate(retriever: HybridRetriever, llm, budget: Budget, cases: list[dict]) -> list[dict]:
    """מודד בנפרד: האם האחזור הביא את המסמך הנכון, והאם התשובה נכונה.
    כל שאלה = 2 קריאות API (embedding + תשובה). הלולאה סופית: עוברת פעם אחת על cases."""
    rows = []
    for case in cases:
        reply, hits = answer(case["question"], retriever, llm, budget)
        rows.append(
            {
                "question": case["question"],
                "retrieved": any(h.metadata["source"] == case["source"] for h in hits),
                "answered": case["expected"] in reply,
                "reply": reply,
            }
        )
    return rows


# ---------------------------------------------------------------- main
def main(folder: str):
    budget = Budget()
    vision_llm, llm, embeddings = build_models(budget)

    docs, error = load_documents(
        Path(folder), vision_llm, budget, progress=lambda n, total, name: print(f"  {n}/{total} {name}")
    )
    if error:
        print("!!", error)
    if not docs:
        sys.exit("אין מסמכים מתועתקים - אין מה לאנדקס.")
    chunks = split(docs)
    print(f"{len(docs)} עמודים, {len(chunks)} קטעים")
    retriever = HybridRetriever(chunks, embeddings)

    cases = load_eval_set()
    if cases:
        rows = evaluate(retriever, llm, budget, cases)
        for r in rows:
            print(f"{'V' if r['retrieved'] else 'X'} אחזור | {'V' if r['answered'] else 'X'} תשובה | {r['question']}")
        print(f"\nאחזור: {sum(r['retrieved'] for r in rows)}/{len(rows)}   "
              f"תשובה: {sum(r['answered'] for r in rows)}/{len(rows)}")

    # הלולאה הזו ממתינה להקלדה שלך בכל סיבוב - היא לא יכולה לרוץ לבד
    while True:
        try:
            q = input(f"\nשאלה (ריק ליציאה, נותרו {budget.left} קריאות): ").strip()
        except EOFError:
            break
        if not q:
            break
        reply, hits = answer(q, retriever, llm, budget)
        print(reply)
        print("מקורות שאוחזרו:", [(h.metadata["source"], h.metadata["page"]) for h in hits])


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else "docs")
    except (BudgetExceeded, FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
