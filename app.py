"""
ממשק Streamlit לשלד ה-RAG.   הרצה:  streamlit run app.py

חשוב להבין על Streamlit: כל לחיצה או הקלדה מריצה את כל הקובץ הזה מחדש מלמעלה.
לכן הכלל כאן: שום קריאת API לא נמצאת ברמה העליונה של הקובץ - רק בתוך
"if כפתור נלחץ". המודלים, התקציב והאינדקס נשמרים בין הריצות ולא נבנים שוב.

העיצוב מחולק לשלושה מקומות:
  .streamlit/config.toml - צבעים וגופנים של הרכיבים המובנים של Streamlit
  style.css              - העיצוב של הרכיבים שאנחנו בונים כאן ב-HTML (כרטיסים, תגיות, מדים)
  app.py (הקובץ הזה)     - מה מוצג ומתי. פונקציות העזר למטה מחזירות מחרוזות HTML קטנות.
"""

import html
import re
from pathlib import Path

import streamlit as st

import rag_skeleton as rag

st.set_page_config(page_title="שאלות על מסמכי הארכיון", page_icon="📜", layout="wide")
st.html(Path(__file__).parent / "style.css")   # קובץ CSS: Streamlit עוטף אותו ב-<style> לבד


@st.cache_resource   # נוצר פעם אחת לכל הפעלה של האפליקציה, לא בכל לחיצה
def shared():
    return {"budget": rag.Budget(), "models": None}


S = shared()
budget: rag.Budget = S["budget"]
ss = st.session_state
ss.setdefault("retriever", None)
ss.setdefault("docs", [])
ss.setdefault("history", [])


def models():
    if S["models"] is None:
        S["models"] = rag.build_models(budget)
    return S["models"]


def run_safely(fn):
    """מריץ פעולה שקוראת ל-API ומציג שגיאה במקום לקרוס."""
    try:
        return fn()
    except rag.BudgetExceeded as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
    return None


# ---------------------------------------------------------------- עזרי תצוגה
# כל הפונקציות כאן רק בונות מחרוזת HTML. אף אחת מהן לא קוראת ל-API.
# כל טקסט שמגיע ממסמך או ממודל עובר קודם דרך esc(), כדי שתווים כמו < לא יתפרשו כ-HTML.
def esc(text) -> str:
    return html.escape(str(text), quote=False)


def text_dir(text: str) -> str:
    """כיוון הטקסט לפי התוכן: סופרים אותיות עבריות מול לטיניות (גרמנית, פולנית)."""
    hebrew = len(re.findall(r"[֐-׿]", text))
    latin = len(re.findall(r"[A-Za-zÀ-ɏ]", text))
    return "ltr" if latin > hebrew else "rtl"


def source_tag(source: str, page: int, extra_class: str = "") -> str:
    """תגית מקור: שם הקובץ + עמוד. <bdi> שומר על שם קובץ לטיני שלם בתוך שורה עברית."""
    return f'<span class="tag {extra_class}"><bdi>{esc(source)}</bdi> · עמ׳ {page}</span>'


def doc_text(text: str) -> str:
    """טקסט של מסמך בכיוון הנכון, עם סימוני אי-הוודאות של התעתוק צבועים."""
    out = esc(text)
    out = out.replace("[לא קריא]", '<mark class="unread">[לא קריא]</mark>')
    out = re.sub(r"\[\?.+?\?\]", lambda m: f'<mark class="unsure">{m.group(0)}</mark>', out)
    return f'<div class="doc-text" dir="{text_dir(text)}">{out}</div>'


def answer_block(reply: str) -> str:
    """התשובה של המודל. ציטוט בצורת (קובץ, עמוד 3) הופך לתגית בולטת.
    השורות הריקות סביב התשובה מאפשרות ל-Markdown שבתוכה (הדגשות, רשימות) להמשיך לעבוד."""
    body = re.sub(
        r"\(([^()\n]+?),\s*עמוד\s*(\d+)\)",
        lambda m: f'<span class="tag cited"><bdi>{m.group(1)}</bdi> · עמ׳ {m.group(2)}</span>',
        esc(reply),
    )
    return f'<div class="answer" dir="{text_dir(reply)}">\n\n{body}\n\n</div>'


def meter(title: str, used: int, total: int, note: str = "", warn_when_high: bool = False, big: bool = False) -> str:
    """מד עם פס התקדמות. warn_when_high צובע באדום כשעוברים 80% (מתאים לתקציב)."""
    share = used / total if total else 0
    classes = "meter" + (" big" if big else "") + (" high" if warn_when_high and share >= 0.8 else "")
    return (
        f'<div class="{classes}">'
        f'<div class="meter-top"><span>{title}</span><b dir="ltr">{used} / {total}</b></div>'
        f'<div class="bar"><i style="width: {min(share, 1) * 100:.0f}%"></i></div>'
        + (f"<small>{note}</small>" if note else "")
        + "</div>"
    )


def result_tag(ok: bool, label: str) -> str:
    return f'<span class="tag {"ok" if ok else "bad"}">{"✓" if ok else "✗"} {label}</span>'


# ---------------------------------------------------------------- סרגל צד
with st.sidebar:
    st.html('<div class="brand"><b>ארכיון המסמכים</b><span>תעתוק, אחזור ותשובות עם מקור</span></div>')
    if rag.FAKE:
        st.html('<div class="fake-stamp"><b>מצב ניסיון</b> (RAG_FAKE=1): מודלים מדומים, אין קריאות API.</div>')
    meter_box = st.empty()   # מתמלא בסוף הקובץ, אחרי שהקריאות של הריצה הזו נספרו

    st.html('<div class="side-title">מסמכים</div>')
    folder = Path(st.text_input("תיקיית מסמכים", "docs"))
    uploads = st.file_uploader(
        "או העלי קבצים", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
    )
    for f in uploads or []:
        folder.mkdir(exist_ok=True)
        target = folder / Path(f.name).name
        if not target.exists():   # שמירה לדיסק בלבד, בלי קריאת API
            target.write_bytes(f.getvalue())

    try:
        pages = rag.list_pages(folder)
    except FileNotFoundError:
        pages = []
    # עמוד שיש לו טקסט מובנה ב-PDF לא עולה כלום - רק עמודים סרוקים בלי תעתוק שמור נספרים כאן
    todo = [p for p in pages if rag.needs_vision(*p)]

    # מה שכבר מוכן בחינם: טקסט מובנה + תעתוקים שמורים (vision_llm=None = בלי קריאת API)
    free_docs = rag.load_documents(folder, None, budget)[0] if pages else []
    free_chunks = rag.split(free_docs)
    try:
        embeddings = models()[2]
        new_chunks = embeddings.missing([c.page_content for c in free_chunks])
    except Exception:   # למשל מפתח חסר - אין טעינה אוטומטית, הכפתור יציג את השגיאה
        embeddings, new_chunks = None, len(free_chunks)

    # טעינה אוטומטית: אם לכל מה שמוכן כבר יש embedding שמור, האינדקס נבנה מהדיסק בחינם.
    # ss.loaded_key מונע בנייה חוזרת בכל לחיצה; משתנה רק כשנוספו עמודים מוכנים.
    free_key = (str(folder), len(free_chunks))
    if embeddings and free_chunks and new_chunks == 0 and ss.get("loaded_key") != free_key:
        with st.spinner("טוען אינדקס שמור..."):
            ss.retriever = run_safely(lambda: rag.HybridRetriever(free_chunks, embeddings))
        if ss.retriever is not None:
            ss.docs, ss.loaded_key = free_docs, free_key

    # כמה קריאות יישלחו - לפני שלוחצים על הכפתור
    emb_calls = -(-new_chunks // rag.EMBED_BATCH)   # עיגול כלפי מעלה: קריאה לכל קבוצה
    if not todo and not new_chunks:
        cost = "הכול שמור: בנייה לא תשלח אף קריאת API."
    else:
        cost = (
            f"בנייה תשלח עד <b>{min(len(todo), rag.MAX_PAGES)}</b> קריאות תעתוק"
            f" + embeddings ל-<b>{new_chunks}</b> קטעים חדשים (<b>{emb_calls}</b> קריאות)"
            + (" + הקטעים של העמודים שיתועתקו." if todo else ".")
        )
    st.html(
        f'<div class="note"><b>{len(pages)}</b> עמודים בתיקייה, מתוכם <b>{len(todo)}</b> סרוקים שעוד לא תועתקו'
        " (עמודי PDF עם טקסט מובנה נשלפים בחינם).</div>"
        f'<div class="note cost-note">{cost}</div>'
    )

    nothing_new = not todo and not new_chunks and ss.retriever is not None
    if st.button("תעתוק ובניית אינדקס", type="primary", disabled=not pages or nothing_new, width="stretch"):
        def build():
            vision_llm, _, embeddings = models()
            bar = st.progress(0.0, "מתעתק...")
            docs, error = rag.load_documents(
                folder, vision_llm, budget,
                progress=lambda n, total, name: bar.progress(n / total, f"{n}/{total} {name}"),
            )
            bar.empty()
            if error:
                st.warning(error)
            ss.docs = docs
            emb_bar = st.progress(0.0, "יוצר embeddings...")
            embeddings.progress = lambda done, total: emb_bar.progress(
                done / total, f"embeddings: {done}/{total} קטעים (דקת המתנה בין קבוצות - מגבלת השכבה החינמית)"
            )
            try:
                chunks = rag.split(docs)
                ss.retriever = rag.HybridRetriever(chunks, embeddings) if docs else None
                ss.loaded_key = (str(folder), len(chunks))
            finally:
                embeddings.progress = None
                emb_bar.empty()
            return len(docs)

        n = run_safely(build)
        if n:
            st.success(f"אינדקס מוכן: {n} עמודים.")

    st.html(
        '<div class="side-title">מודלים</div><div class="models">'
        f"תעתוק <bdi>{esc(rag.VISION_MODEL)}</bdi>"
        f"תשובה <bdi>{esc(rag.ANSWER_MODEL)}</bdi>"
        f"Embeddings <bdi>{esc(rag.EMBEDDING_MODEL)}</bdi></div>"
    )

# ---------------------------------------------------------------- גוף
st.html(
    '<div class="masthead"><div class="eyebrow">ארכיון מסמכים סרוקים</div>'
    "<h1>שאלות על מסמכי הארכיון</h1>"
    "<p>התשובות נכתבות רק מתוך המסמכים, וכל עובדה מצוינת עם הקובץ והעמוד שממנו נלקחה.</p></div>"
)
status_box = st.empty()   # מצב האינדקס + מונה הקריאות; מתמלא בסוף הקובץ
tab_ask, tab_text, tab_eval = st.tabs(["שאלות", "בדיקת תעתוק", "הערכה"])

with tab_ask:
    if ss.retriever is None:
        st.html(
            '<div class="empty"><b>עדיין אין אינדקס</b>'
            "כדי לשאול שאלות צריך קודם לתעתק את המסמכים ולבנות מהם אינדקס."
            "<br><ol><li>פתחי את סרגל הצד (בחלון צר: החץ שבראש העמוד).</li>"
            "<li>בדקי שם כמה קריאות API הבנייה תשלח.</li>"
            "<li>לחצי על ״תעתוק ובניית אינדקס״.</li></ol></div>"
        )
    else:
        # form: השאלה נשלחת רק בלחיצה על הכפתור, לא בכל הקלדה
        with st.form("ask", clear_on_submit=True):
            q = st.text_input("שאלה", placeholder="למשל: מה המקצוע של מנחם גולדשטיין?")
            sent = st.form_submit_button("שאלי", type="primary")
        if sent and q.strip():
            _, llm, _ = models()
            with st.spinner("מחפש ועונה..."):
                result = run_safely(lambda: rag.answer(q.strip(), ss.retriever, llm, budget))
            if result:
                ss.history.insert(0, (q.strip(), *result))
        if not ss.history:
            st.html(
                '<div class="empty"><b>האינדקס מוכן</b>'
                "כתבי שאלה למעלה. כל שאלה = 2 קריאות API (embedding + תשובה).</div>"
            )

        # היסטוריה: החדשה ביותר למעלה. כל שאלה היא "כרטיס" עם מסגרת.
        for i, (question, reply, hits) in enumerate(ss.history):
            number = len(ss.history) - i
            with st.container(border=True, key=f"qa_{number}"):
                st.html(
                    f'<div class="qa-num">שאלה {number}</div>'
                    f'<p class="qa-question" dir="{text_dir(question)}">{esc(question)}</p>'
                )
                st.markdown(answer_block(reply), unsafe_allow_html=True)

                # שורת מקורות: כל (קובץ, עמוד) שאוחזר, פעם אחת. מקור שהתשובה מזכירה - מודגש.
                sources = dict.fromkeys((h.metadata["source"], h.metadata["page"]) for h in hits)
                tags = "".join(source_tag(s, p, "cited" if s in reply else "") for s, p in sources)
                if rag.NOT_FOUND in reply:
                    tags = '<span class="tag warn">התשובה לא נמצאה במסמכים</span>' + tags
                st.html(f'<span class="label">מקורות שאוחזרו:</span>{tags}')

                with st.expander(f"{len(hits)} קטעים שאוחזרו (לבדיקה: האם התשובה באמת כתובה שם?)"):
                    st.html("".join(
                        f'<div class="chunk"><div class="chunk-head"><span class="rank">{rank}</span>'
                        f'{source_tag(h.metadata["source"], h.metadata["page"])}</div>'
                        f"{doc_text(h.page_content)}</div>"
                        for rank, h in enumerate(hits, start=1)
                    ))

with tab_text:
    st.caption("תמונת המקור מול התעתוק - כאן רואים אם הבעיה בתעתוק, לפני שמאשימים את האחזור.")
    done = [p for p in pages if rag.read_cache(*p) is not None]
    if not done:
        st.html(
            '<div class="empty"><b>אין עדיין תעתוקים</b>'
            "אחרי בניית האינדקס יופיע כאן כל עמוד לצד התעתוק שלו.</div>"
        )
    else:
        choice = st.selectbox("עמוד", done, format_func=lambda p: f"{p[0].name} - עמוד {p[1]}")
        text = rag.read_cache(*choice)
        unread, unsure = text.count("[לא קריא]"), text.count("[?")
        right, left = st.columns(2, gap="large")   # בפריסה מימין לשמאל העמודה הראשונה היא הימנית
        with right:
            st.html('<div class="pane-title">תמונת המקור</div>')
            image, _ = rag.page_image(*choice)   # קריאה מהדיסק, לא API
            st.container(key="scan").image(image)
        with left:
            st.html('<div class="pane-title">התעתוק</div>')
            st.html(
                '<div class="sheet"><div class="sheet-head">'
                + source_tag(choice[0].name, choice[1])
                + ('<span class="tag ok">נשלף מהקובץ (בחינם)</span>'
                   if rag.read_cache_method(*choice) == "text"
                   else '<span class="tag">תועתק במודל</span>')
                + f'<span class="tag {"bad" if unread else ""}">{unread} לא קריא</span>'
                + f'<span class="tag {"warn" if unsure else ""}">{unsure} לא בטוח</span>'
                + f"</div>{doc_text(text)}</div>"
            )

with tab_eval:
    cases = run_safely(rag.load_eval_set) or []
    st.html(
        f'<div class="note"><b>{len(cases)}</b> שאלות עם תשובה ידועה בקובץ <bdi>{rag.EVAL_FILE}</bdi>.</div>'
        f'<div class="note cost-note">הרצה = <b>{len(cases) * 2}</b> קריאות API.</div>'
    )
    if ss.retriever is None:
        st.caption("כדי להריץ הערכה צריך קודם לבנות אינדקס (בסרגל הצד).")
    if st.button("הרצת הערכה", type="primary", disabled=not cases or ss.retriever is None):
        _, llm, _ = models()
        with st.spinner("מריץ..."):
            ss.eval_rows = run_safely(lambda: rag.evaluate(ss.retriever, llm, budget, cases))
    rows = ss.get("eval_rows") or []

    if rows:
        # שני מדדים נפרדים: אם האחזור נכשל - הבעיה באינדקס; אם רק התשובה נכשלה - בתעתוק או במודל
        st.html(
            '<div class="scores">'
            + meter("אחזור: המסמך הנכון הגיע", sum(r["retrieved"] for r in rows), len(rows),
                    note="אחד מהקטעים שאוחזרו הוא מהקובץ שבו נמצאת התשובה.", big=True)
            + meter("תשובה: הערך הצפוי מופיע", sum(r["answered"] for r in rows), len(rows),
                    note="המילה או המספר הצפויים מופיעים בתשובת המודל.", big=True)
            + "</div>"
        )

    # שורה לכל שאלה. לפני הרצה רואים רק את השאלה והתשובה הצפויה; אחריה גם את התוצאה.
    results = {r["question"]: r for r in rows}
    case_cards = ""
    for case in cases:
        r = results.get(case["question"])
        case_cards += (
            f'<div class="case"><div class="case-q" dir="{text_dir(case["question"])}">{esc(case["question"])}</div>'
            + (result_tag(r["retrieved"], "אחזור") + result_tag(r["answered"], "תשובה") if r else "")
            + f'<span class="label">צפוי:</span><span class="tag">{esc(case["expected"])}</span>'
            + f'<span class="label">מקור:</span><span class="tag"><bdi>{esc(case["source"])}</bdi></span>'
            + (f'<div class="case-reply"><span class="label">תשובת המודל</span>{doc_text(r["reply"])}</div>' if r else "")
            + "</div>"
        )
    if case_cards:
        st.html(case_cards)

# ---------------------------------------------------------------- מונה קריאות
# ממולא כאן, בסוף, כדי לכלול גם קריאות שנשלחו בריצה הנוכחית של הקובץ.
# מופיע פעמיים: בסרגל הצד (עם פס), ובראש העמוד - שם הוא נראה גם כשסרגל הצד סגור.
meter_box.html(meter("קריאות API בהפעלה זו", budget.used, budget.limit,
                     note=f"נותרו {budget.left} עד התקרה.", warn_when_high=True))
if ss.retriever is None:
    index_tag = '<span class="tag warn">אין אינדקס</span>'
else:
    index_tag = f'<span class="tag ok">אינדקס מוכן: {len(ss.docs)} עמודים</span>'
status_box.html(
    f'<div class="status">{index_tag}'
    f'<span class="tag">קריאות API: <bdi>{budget.used} / {budget.limit}</bdi></span></div>'
)
