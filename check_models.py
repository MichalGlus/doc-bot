"""
בדיקת מודלים לפני הרצה אמיתית.   הרצה:  python check_models.py

1. מדפיס אילו מודלים המפתח שלך רואה (רשימה בלבד - לא נספר במכסה).
2. שולח בדיוק 2 קריאות קטנות: אחת למודל הצ'אט ואחת ל-embeddings שהוגדרו ב-rag_skeleton.py.
אם שם מודל שגוי או שאין לו שכבה חינמית - תראי את זה כאן ולא באמצע המבחן.
"""

import os
import sys

import rag_skeleton as rag

key = os.getenv("GOOGLE_API_KEY")
if not key:
    sys.exit("חסר משתנה הסביבה GOOGLE_API_KEY")

from google import genai

print("מודלים זמינים למפתח שלך:")
client = genai.Client(api_key=key)   # חייב להישמר במשתנה - אחרת הלקוח נסגר לפני הבקשה
for m in client.models.list():
    name = m.name.removeprefix("models/")
    if any(s in name for s in ("flash", "embedding", "gemma")):
        print("  ", name)

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

for label, fn in [
    (rag.ANSWER_MODEL, lambda: init_chat_model(rag.ANSWER_MODEL, max_retries=0).invoke("אמור שלום").text),
    (rag.EMBEDDING_MODEL, lambda: f"וקטור באורך {len(init_embeddings(rag.EMBEDDING_MODEL).embed_query('שלום'))}"),
]:
    try:
        print(f"\nV {label}: {fn()}")
    except Exception as e:
        print(f"\nX {label}: {type(e).__name__}: {str(e)[:400]}")
