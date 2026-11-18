# פרומפט לבנייה מחדש של doc-agent (להדבקה בצ'אט חדש)

---

אתה מהנדס AI בכיר. בנה איתי מאפס, מקצה לקצה, פרויקט RAG בשם **doc-agent**: סוכן שעונה על שאלות על **מסמכי ארכיון סרוקים ישנים** (בני כ-100 שנה, איכות סריקה ירודה, עברית + יידיש + גרמנית + פולנית). כל עובדה בתשובה חייבת להיות מצוטטת ל-(קובץ, עמוד).

הפרויקט משמש להכנה לראיון עבודה ולמבחן מעשי, ולכן:
- הקוד צריך להיות **קצר, קריא, וניתן להסבר שורה-שורה**. בלי הפשטות מיותרות ובלי פריימוורקים כבדים.
- כל החלטה תכנונית צריכה הערה קצרה בקוד שמסבירה **למה** (הערות בעברית).
- הוא חייב לרוץ על **השכבה החינמית של Google Gemini** בלי לשרוף מכסה.

עבוד בשלבים: בנה קובץ, הסבר בקצרה מה בנית ולמה, ואז המשך. בסוף כל שלב תן לי פקודה לבדוק שהוא עובד (עדיף במצב FAKE, בלי עלות).

## 0. סביבה וסטאק

- Python 3.11 ומעלה, venv ב-`.venv/` (Windows, PowerShell).
- LangChain 1.x: `langchain`, `langchain-core`, `langchain-community` (BM25Retriever), `langchain-text-splitters`, `langchain-google-genai`, `google-genai`.
- `pymupdf` (שליפת טקסט מ-PDF ורינדור עמוד לתמונה), `rank_bm25`, `python-dotenv`, `streamlit`.
- `requirements.txt` נקי שמכיל **רק** את מה שבאמת בשימוש, כולל `pymupdf` ו-`rank_bm25`. (בגרסה הקודמת הם חסרו, וזה באג.)
- `.env` עם `GOOGLE_API_KEY` (וגם אופציונלית `LANGSMITH_*`), שנטען עם `load_dotenv(Path(__file__).with_name(".env"))` **לפני** ייבוא LangChain. משתנה סביבה שכבר מוגדר גובר על הקובץ. `.gitignore`: `.env`, `.venv/`, `__pycache__/`.

## 1. מבנה קבצים

```
rag_skeleton.py      # כל הלוגיקה + CLI
app.py               # ממשק Streamlit (תצוגה בלבד, קורא ל-rag_skeleton)
style.css            # עיצוב הרכיבים שבונים ב-HTML
.streamlit/config.toml  # ערכת צבעים וגופנים של Streamlit
check_models.py      # בדיקת מודלים/מפתח לפני ריצה אמיתית
make_samples.py      # מייצר מסמכים "ישנים" מלאכותיים לתרגול
eval_set.json        # שאלות עם תשובה ידועה
docs/                # מסמכי המקור
transcripts/         # כל ה-caches (מחיקה = לשלם שוב)
```

## 2. הגדרות (ב-rag_skeleton.py, כולן ניתנות לדריסה ממשתני סביבה)

| קבוע | env | ברירת מחדל |
|---|---|---|
| VISION_MODEL | RAG_VISION_MODEL | `google_genai:gemini-3.5-flash-lite` |
| ANSWER_MODEL | RAG_ANSWER_MODEL | `google_genai:gemini-3.5-flash-lite` |
| EMBEDDING_MODEL | RAG_EMBEDDING_MODEL | `google_genai:gemini-embedding-2` |
| FAKE | RAG_FAKE=1 | כבוי |
| MAX_API_CALLS | RAG_MAX_API_CALLS | 60 |
| MAX_PAGES | RAG_MAX_PAGES | 30 (עמודים **חדשים לתעתוק** בריצה) |
| SECONDS_BETWEEN_CALLS | RAG_SECONDS_BETWEEN_CALLS | 4.5 (בערך 13 בדקה) |
| EMBED_BATCH | RAG_EMBED_BATCH | 90 |
| EMBED_WAIT_SECONDS | RAG_EMBED_WAIT_SECONDS | 62 |

קבועים נוספים: `CACHE_DIR=Path("transcripts")`, `EVAL_FILE=Path("eval_set.json")`, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`, `TOP_K=5`, `SUFFIXES={.pdf,.png,.jpg,.jpeg}`, `NOT_FOUND="לא נמצא במסמכים"`, `MIN_TEXT_LETTERS=50`.

שמות מודלים בפורמט `"ספק:מודל"` דרך `init_chat_model` / `init_embeddings`, כך שאפשר להחליף ל-`ollama:...` או לספק אחר בלי לשנות קוד.

## 3. הפרומפטים (מילה במילה)

TRANSCRIBE_PROMPT:
```
תעתק את הטקסט שבתמונה בדיוק כפי שהוא כתוב.
- שמור על השפה המקורית (עברית, יידיש, גרמנית, פולנית וכו'). אל תתרגם.
- אל תתקן כתיב ואל תשלים מילים חסרות.
- מילה שאינה קריאה סמן [לא קריא]. מילה שאינך בטוח בה סמן [?מילה?].
- טבלה או רשימה - שורה לכל רשומה.
החזר את התעתוק בלבד.
```

ANSWER_PROMPT:
```
ענה על השאלה אך ורק על סמך הקטעים שלמטה.
- אחרי כל עובדה ציין מקור בסוגריים: (קובץ, עמוד).
- אם התשובה אינה מופיעה בקטעים, כתוב: "לא נמצא במסמכים".
- אל תנחש שמות, תאריכים או מספרים.

קטעים:
{context}

שאלה: {question}
```
ה-context בנוי כך: לכל קטע שורת כותרת `[{source}, עמוד {page}]` ואחריה הטקסט, וקטעים מופרדים בשורה ריקה.

## 4. הגנות עלות (העיקרון המרכזי של הפרויקט)

- `class BudgetExceeded(RuntimeError)`.
- `class Budget`: `limit`, `used`, `spend(what)` שזורק `BudgetExceeded` עם הודעה בעברית שמציינת את התקרה ואת שם הקריאה, ו-`left` כ-property. **כל** קריאת API עוברת דרך `spend()`.
- `max_retries=1` בכל מודל צ'אט, כי ספריות מנסות שוב בשקט וכל ניסיון נספר במכסה.
- `InMemoryRateLimiter(requests_per_second=1/SECONDS_BETWEEN_CALLS, check_every_n_seconds=0.1, max_bucket_size=1)`. לא מופעל עבור `ollama:` או כש-SECONDS=0.
- `temperature=0`.
- אין לולאה שקוראת ל-API בלי גבול: כל לולאה רצה על רשימה סופית.
- **עצירה בשגיאה הראשונה**: לא ממשיכים "לנסות את הדף הבא".
- כל מה ששולם נשמר לדיסק ולא נשלח שוב.

### CountingEmbeddings(Embeddings)
עוטף את מודל ה-embeddings:
- cache לדיסק: `transcripts/embcache_{re.sub(r"\W","_",EMBEDDING_MODEL)}{"_fake" if FAKE}.json` בצורה של מילון `sha1(text) -> vector`. נטען בעצלתיים.
- `missing(texts)`: כמה טקסטים ייחודיים עוד אין להם וקטור, **בלי** קריאת API (משמש להערכת עלות ב-UI).
- `embed_documents`: מסנן מה שכבר שמור, מוריד כפילויות, שולח בקבוצות של EMBED_BATCH, עם `time.sleep(EMBED_WAIT_SECONDS)` **בין** קבוצות (לא לפני הראשונה). `budget.spend` לכל קבוצה, ו**שמירה לדיסק אחרי כל קבוצה**, כך ש-429 באמצע לא מוחק את מה ששולם. callback אופציונלי `self.progress(done, total)`. אין המתנה במצב FAKE או `ollama:`.
- `embed_query`: `spend` ואז קריאה ישירה, בלי cache.
- הרקע: בשכבה החינמית כל chunk נספר כבקשה (100 בדקה). כש-517 chunks נשלחו בבת אחת התקבל 429 RESOURCE_EXHAUSTED.

### build_models(budget) -> (vision_llm, answer_llm, embeddings)
במצב FAKE: `FakeListChatModel(responses=["[מודל מדומה] טקסט לדוגמה: רשימת שמות, ורשה 1941."])` לשניהם, ו-`DeterministicFakeEmbedding(size=64)` עטוף ב-CountingEmbeddings. כל הייבואים הכבדים נעשים בתוך הפונקציה.

## 5. שלב 1: טעינה ותעתוק

- `list_pages(folder)`: רשימה של `(path, page_no)` לכל קובץ נתמך, ממוינת, כשכל עמוד PDF נספר לחוד. זורקת `FileNotFoundError` בעברית אם התיקייה לא קיימת. **בלי** קריאה למודל.
- **שכבת טקסט של PDF קודמת לכול (בחינם):** `pdf_text_layer(path, page_no)` מחזירה None אם הקובץ אינו PDF. הפונקציה הפנימית `@lru_cache(4096) _pdf_text_layer_cached(path_str, mtime, page_no)` (ה-mtime נכלל במפתח כדי שקובץ שהשתנה ייקרא מחדש) מריצה `page.get_text()`, סופרת אותיות `[א-תA-Za-z]` ותווים גלויים `\S`, ומחזירה None אם יש פחות מ-50 אותיות (עמוד סרוק) **או** אם האותיות הן פחות מ-50% מהתווים הגלויים (גופן עם קידוד שבור/ג'יבריש).
- `_fix_hebrew_pdf_text`: תיקון סוגריים הפוכים בסוף שורה (`re.sub(r"\)\(([^()\n]*?)\s*$", r" (\1)", text, flags=re.M)`: `ביבמות)(סב ב` הופך ל-`ביבמות (סב ב)`), כיווץ רווחים, ו-strip לכל שורה.
- `page_image(path, page_no)`: עבור PDF רינדור ב-200dpi ל-PNG, עבור תמונה הבתים עצמם, ומחזיר גם mime.
- **cache תעתוק לכל עמוד:** `transcripts/{path.name}_p{page_no}.json` (שם מלא עם סיומת, כדי ש-a.jpg ו-a.pdf לא יידרסו), בתוכן `{"text":..., "method": "text"|"vision"}`. קובץ ישן בלי `method` נחשב `vision`. קובץ פגום (JSON שבור / חסר text) נחשב כאילו אינו קיים. נכתב עם `ensure_ascii=False`. הפונקציות: `_read_cache_record`, `read_cache`, `read_cache_method`, `write_cache`.
- `needs_vision(path, page)`: True רק אם אין cache וגם אין שכבת טקסט. משמש להערכת עלות.
- `transcribe(image, mime, vision_llm, budget)`: `spend`, ואז `HumanMessage` עם content שמכיל טקסט + `image_url` כ-data URL ב-base64. מחזירה `.text`.
- `load_documents(folder, vision_llm, budget, progress=None) -> (docs, error)`:
  1. אם יש שכבת טקסט: משתמשים בה (method="text"), והיא **דורסת** cache ישן של vision לאותו עמוד, כי vision קרא כתב רש"י לא נכון ומיזג שתי עמודות בלי לסמן אי-ודאות.
  2. אחרת קוראים את ה-cache.
  3. אחרת, אם `vision_llm is None` מדלגים (מצב "רק מה שבחינם"). אם `new_pages >= MAX_PAGES` עוצרים עם error. אחרת מתעתקים, ו-`try/except Exception` מחזיר error ועוצר (break). **לא שומרים תעתוק ריק**, כי אחרת הוא "יינעל" ב-cache.
  4. `progress(n, total, name)`.
  5. `Document(page_content=text, metadata={"source": path.name, "page": page_no, "method": method})` רק אם הטקסט לא ריק. השאר הערה: "כאן המקום להוסיף מטא-דאטה: שנה, סוג מסמך, שפה".
  - חשוב: תקרת MAX_PAGES חלה **רק על עמודים סרוקים**. בגרסה הראשונה היא חלה על כל העמודים, ורק 11 מתוך 79 עמודי ספר נכנסו לאינדקס. התוצאה הייתה "לא נמצא" על שאלה ברורה.

## 6. שלב 2: Chunking
`RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n","\n",". "," "])`. ה-metadata עוברת לכל chunk, ומוסיפים `chunk_id` רץ.

## 7. שלבים 3+4: אחזור היברידי
- `tokenize(text) = re.findall(r"\w+", text.lower())`, כי ברירת המחדל של BM25 מפצלת רק לפי רווח, ואז "ורשה," שונה מ-"ורשה".
- `HybridRetriever(chunks, embeddings)`: `ValueError` בעברית אם אין chunks. `InMemoryVectorStore.from_documents` (אין שמירת אינדקס לדיסק, כי ה-embcache הופך בנייה מחדש לחינמית) ו-`BM25Retriever.from_documents(chunks, k=TOP_K*2, preprocess_func=tokenize)`.
- `search(query, k=TOP_K)`: וקטורי עם `k*2` + BM25, מיזוג **RRF**: `score += 1/(60+rank)` לפי `chunk_id`, ממוינים, ומחזירים את k הראשונים.
- למה היברידי: שמות, מקומות ומספרים דורשים התאמה מדויקת (BM25), ושאלות ניסוח חופשי דורשות משמעות (וקטור).

## 8. שלב 5: תשובה
`answer(question, retriever, llm, budget) -> (reply, hits)`: אם אין hits מחזירים `NOT_FOUND, []`. אחרת בונים context, `spend("תשובה לשאלה")` ו-`llm.invoke(...).text`. כל שאלה = 2 קריאות (embedding לשאלה + תשובה).

## 9. שלב 6: הערכה
- `eval_set.json` הוא רשימה של `{question, expected, source}`. `load_eval_set` מסנן רשומות חסרות.
- `evaluate(...)` מחזיר שורות `{question, retrieved, answered, reply}`, כאשר `retrieved` = האם אחד ה-hits מהקובץ הנכון, ו-`answered` = האם `expected` מופיע כתת-מחרוזת בתשובה.
- **מודדים בנפרד אחזור ותשובה**: אם האחזור נכשל, הבעיה באינדקס/chunking. אם רק התשובה נכשלה, הבעיה בתעתוק או בפרומפט.
- 8 מקרים, כולל מקרה "לא נמצא" מכוון (expected="לא נמצא") שבודק שהמודל לא ממציא.

## 10. CLI (main)
`python rag_skeleton.py docs/`: בונה מודלים, טוען (עם הדפסת התקדמות), מדפיס "N עמודים, M קטעים", מריץ הערכה (V/X לאחזור ולתשובה + סיכום), ואז לולאת `input()` שמציגה כמה קריאות נותרו. קלט ריק או EOF מסיימים. `BudgetExceeded / FileNotFoundError / ValueError` נתפסים ומסתיימים ב-`sys.exit(str(e))`.

## 11. check_models.py
מוודא שיש `GOOGLE_API_KEY`, מדפיס מודלים זמינים דרך `genai.Client(api_key=key).models.list()` (מסנן flash/embedding/gemma; הלקוח חייב להישמר במשתנה כדי שלא ייסגר) ושולח בדיוק 2 קריאות בדיקה (`max_retries=0`): צ'אט ו-embedding. מדפיס V/X עם שגיאה מקוצרת ל-400 תווים. מריצים לפני כל סשן אמיתי.

## 12. make_samples.py
מייצר עם Pillow + python-bidi חמישה מסמכים בדויים במראה סריקה ישנה: רקע נייר (236,224,196), דיו לא אחיד, 14 כתמים מטושטשים, קו קיפול, 9000 פיקסלי רעש, סיבוב קל וטשטוש גאוסי. גודל A4 ב-150dpi (1240×1754). על כל דף חותמת "דוגמה מלאכותית לתרגול - כל הפרטים בדויים". נתיבי הגופנים צריכים להתאים גם ל-Windows (למשל `C:/Windows/Fonts/`) עם fallback.
- `letter_lodz_1939.jpg`: מכתב בעברית עם תאריך עברי (כ"ג באדר תרצ"ט) לשמואל ויינברג בתל אביב, חתום "חיים".
- `residents_list_1941.png`: טבלת דיירים (שם, שנת לידה, מקצוע). מנחם גולדשטיין – חייט, מרים זילברברג – 1925.
- `bescheinigung_1942.jpg`: אישור עבודה בגרמנית (LTR) של חיים ויינברג, מספר 4471.
- `karta_1945.png`: כרטיס רישום בפולנית של רבקה רוזנצווייג, מספר 2086.
- `testimony_318.pdf`: עדות בשני עמודים (העדה עלתה באניה "עצמאות", האב נפטר בגטו ב-1942).
ושמונה שאלות eval שמתאימות בדיוק לפרטים האלה.

## 13. ממשק Streamlit (app.py)

**כלל-על:** Streamlit מריץ את כל הקובץ מחדש בכל אינטראקציה. לכן **אף קריאת API לא נמצאת ברמה העליונה**, רק בתוך `if button`.

- `st.set_page_config(page_title="שאלות על מסמכי הארכיון", page_icon="📜", layout="wide")` ו-`st.html(Path(__file__).parent / "style.css")`.
- `@st.cache_resource shared()` מחזיק `{"budget": Budget(), "models": None}`, משותף בין reruns. `models()` בונה את המודלים בעצלתיים.
- `session_state`: `retriever`, `docs`, `history`, `loaded_key`, `eval_rows`.
- `run_safely(fn)` תופס `BudgetExceeded` ו-`Exception` ומציג `st.error` במקום לקרוס.
- עזרי תצוגה שמחזירים רק HTML: `esc()` (`html.escape`, **כל** טקסט ממסמך או ממודל עובר דרכו), `text_dir()` (יותר אותיות לטיניות `[A-Za-zÀ-ɏ]` מעבריות `[֐-׿]` = ltr), `source_tag()` עם `<bdi>`, `doc_text()` שצובע `[לא קריא]` כ-`mark.unread` ו-`[?...?]` כ-`mark.unsure`, `answer_block()` שהופך ציטוטים `(קובץ, עמוד N)` לתגיות עם regex ועוטף בשורות ריקות כדי ש-Markdown ימשיך לעבוד, `meter()` עם פס ואדום מעל 80%, ו-`result_tag()`.
- **סרגל צד:** מיתוג "ארכיון המסמכים", חותמת מצב FAKE, מד קריאות API (placeholder שמתמלא בסוף הקובץ), שדה תיקייה (ברירת מחדל `docs`), העלאת קבצים (שמירה לדיסק בלבד ורק אם לא קיים), ספירת עמודים ועמודים סרוקים שטרם תועתקו, ו**הערכת עלות לפני הלחיצה**: עד `min(todo, MAX_PAGES)` קריאות תעתוק + embeddings ל-N קטעים חדשים (`ceil(N/EMBED_BATCH)` קריאות), או "הכול שמור: בנייה לא תשלח אף קריאת API".
- **טעינה אוטומטית בחינם:** `load_documents(folder, None, budget)` ואז split. אם `embeddings.missing(...) == 0` בונים את האינדקס מהדיסק. `loaded_key=(folder, len(chunks))` מונע בנייה חוזרת בכל rerun. אם בניית המודלים נכשלה (למשל מפתח חסר) אין טעינה אוטומטית.
- כפתור "תעתוק ובניית אינדקס" (primary) שמושבת כשאין עמודים או כשאין שום דבר חדש. יש בו progress bar לתעתוק ו-progress bar ל-embeddings ("דקת המתנה בין קבוצות - מגבלת השכבה החינמית"), ו-`finally` שמנקה את ה-callback.
- תצוגת שמות המודלים.
- **גוף:** masthead, שורת סטטוס (אינדקס מוכן / אין אינדקס + קריאות API), ושלוש לשוניות:
  1. **שאלות:** `st.form` עם `clear_on_submit` (כדי שלא תישלח שאלה בכל הקלדה). היסטוריה, החדש למעלה, בכרטיסים `st.container(border=True)`: מספר שאלה, שאלה, תשובה, שורת "מקורות שאוחזרו" (ייחודיים; מקור שמוזכר בתשובה מודגש; תגית אזהרה אם "לא נמצא"), ו-expander עם הקטעים שאוחזרו ממוספרים לפי דירוג, לבדיקה אם התשובה באמת כתובה שם. מצב ריק מסביר מה לעשות.
  2. **בדיקת תעתוק:** selectbox של עמודים מתועתקים. שתי עמודות: תמונת המקור מימין והתעתוק משמאל, עם תגיות "נשלף מהקובץ (בחינם)" / "תועתק במודל" ומונה `[לא קריא]` / `[?`. המטרה: לראות אם הבעיה בתעתוק לפני שמאשימים את האחזור.
  3. **הערכה:** מספר השאלות ועלות (`2×N` קריאות), כפתור הרצה, שני מדים גדולים (אחזור / תשובה), וכרטיס לכל מקרה עם צפוי, מקור, ✓/✗ ותשובת המודל.
- בסוף הקובץ ממלאים את מד הקריאות ואת שורת הסטטוס, כדי לכלול גם את הקריאות של ה-rerun הנוכחי.

**עיצוב "ארכיון": נייר, דיו, חותמת אדומה.** RTL מלא. `config.toml`: primary `#8B3A2B`, bg `#F5F0E6`, secondary `#FCFAF5`, text `#2A2520`, border `#D5CAB4`, sidebar `#ECE4D2`, גופן Heebo וכותרות Frank Ruhl Libre (Google Fonts), `toolbarMode="minimal"`. ב-`style.css` אותם צבעים כמשתני `:root` (עם הערה שצריך לשנות בשני המקומות), `.doc-text` בגופן serif עם `white-space: pre-wrap` ו-`text-align: start`, ו-`section[data-testid="stSidebar"]{overflow:hidden}` (תיקון לסרגל צד סגור ב-RTL).

## 14. בדיקות קבלה
1. `RAG_FAKE=1 python rag_skeleton.py docs/` רץ מקצה לקצה בלי אף קריאת API.
2. ריצה שנייה אמיתית שולחת 0 קריאות תעתוק ו-0 embeddings למסמכים (הכול מה-cache).
3. PDF דיגיטלי של 79 עמודים נכנס כולו לאינדקס בלי קריאות vision.
4. הקטנת `RAG_MAX_API_CALLS=3` עוצרת בהודעה ברורה ולא קורסת.
5. ניתוק רשת באמצע תעתוק: נעצר, ומה שתועתק נשמר.
6. ב-eval: אחזור 8/8, תשובה 7–8/8, והמקרה "לא נמצא" עובר.

## 15. אחרי שהבסיס עובד (שדרוגים, כל אחד בנפרד ובאישורי)
1. **Query decomposition:** הודעה עם שתי שאלות מקבלת היום חיפוש אחד, ורק אחת נענית. צריך פיצול לתת-שאלות ואחזור ותשובה לכל אחת.
2. **שני מצבי תשובה עם ניתוב:** מצב ארכיון (קפדני, מצוטט) מול מצב הבנה ("מה זה X?": יותר קטעים ופרומפט מסביר). היום שאלות מושגיות מקבלות "לא נמצא".
3. **נרמול לאחזור:** הסרת ניקוד, אחידות אותיות סופיות, `ſ→s`, `ä/ae`, וגרסאות איות של שמות (Weinberg / Vainberg / ווינברג / ווייןברג).
4. **Reranker רב-לשוני** (למשל `bge-reranker-v2-m3`) אחרי RRF.
5. **מטא-דאטה** (שנה, שפה, סוג מסמך) שמחולצת בזמן התעתוק, וסינון לפיה.
6. **OCR חלופי/משלים** (Tesseract עם `heb`/`deu`/`deu_frak`, או Transkribus לכתב יד) והשוואת CER מול vision LLM.
