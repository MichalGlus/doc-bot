"""
יוצר מסמכים מלאכותיים לתרגול, במראה של סריקות ישנות.   הרצה:  python make_samples.py
כל השמות, המקומות והפרטים בדויים. על כל דף מודפס "דוגמה מלאכותית לתרגול".
(נדרש: pip install pillow python-bidi pymupdf)
"""

import random
from pathlib import Path

from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path("docs")
FONTS = "/usr/share/fonts/truetype/freefont/"
W, H = 1240, 1754   # A4 ב-150dpi
STAMP = "דוגמה מלאכותית לתרגול - כל הפרטים בדויים"


def font(name, size):
    return ImageFont.truetype(FONTS + name, size, layout_engine=ImageFont.Layout.BASIC)


def page(lines, fnt, rtl=True, seed=0, blur=0.6, tilt=0.8, line_gap=1.55):
    """lines: רשימת שורות. מחזיר תמונה 'מיושנת'."""
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), (236, 224, 196))
    d = ImageDraw.Draw(img)
    y = 130
    step = int(fnt.size * line_gap)
    for line in lines:
        ink = rnd.randint(25, 70)   # דיו לא אחיד
        text = get_display(line) if rtl else line
        if rtl:
            d.text((W - 120, y), text, font=fnt, fill=(ink, ink, ink + 10), anchor="ra")
        else:
            d.text((120, y), text, font=fnt, fill=(ink, ink, ink + 10))
        y += step
    small = font("FreeSans.ttf", 20)
    d.text((W // 2, H - 60), get_display(STAMP), font=small, fill=(120, 60, 60), anchor="ma")
    # כתמים, קפל ורעש
    for _ in range(14):
        x, yy, r = rnd.randint(0, W), rnd.randint(0, H), rnd.randint(15, 90)
        blot = Image.new("RGB", (W, H), (200, 180, 140))
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse((x - r, yy - r, x + r, yy + r), fill=rnd.randint(25, 70))
        img.paste(blot, mask=mask.filter(ImageFilter.GaussianBlur(r / 2)))
    ImageDraw.Draw(img).line((0, H // 2, W, H // 2 + 6), fill=(205, 190, 160), width=3)
    px = img.load()
    for _ in range(9000):
        x, yy = rnd.randrange(W), rnd.randrange(H)
        v = rnd.randint(90, 170)
        px[x, yy] = (v, v - 8, v - 25)
    img = img.rotate(rnd.uniform(-tilt, tilt), fillcolor=(236, 224, 196), resample=Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(blur))


SERIF, MONO = font("FreeSerif.ttf", 34), font("FreeMono.ttf", 28)
OUT.mkdir(exist_ok=True)

# 1. מכתב בעברית, תאריך עברי
page([
    "ב\"ה, לודז', כ\"ג באדר תרצ\"ט",
    "",
    "לכבוד אחי היקר מר שמואל ויינברג, תל אביב",
    "",
    "שלום וברכה,",
    "מכתבך מיום ט\"ו בשבט הגיע לידינו בשבוע שעבר ושמחנו בו מאד.",
    "אבא שיחיה חזר לעבודתו בבית המלאכה לנגרות ברחוב זגרסקה 14,",
    "אך הפרנסה דחוקה והמסים כבדים מיום ליום.",
    "אחותנו רבקה נישאה בחודש כסלו לבחור מפיוטרקוב, ושמו",
    "יעקב רוזנצווייג, סופר סת\"ם בן טובים.",
    "אנו מבקשים ממך לברר במשרד העליה בדבר רשיון בשביל ההורים.",
    "שלחנו את התעודות בדואר רשום ביום ראשון שעבר.",
    "",
    "דרישת שלום לכל המשפחה, ונזכה לבשורות טובות.",
    "",
    "אחיך המצפה לתשובתך,",
    "חיים ויינברג",
], SERIF, seed=1).save(OUT / "letter_lodz_1939.jpg", quality=80)

# 2. רשימת שמות (טבלה) בעברית - שורה לרשומה
rows = [
    "מס'   שם משפחה ופרטי        שנת לידה   מקצוע       כתובת",
    "--------------------------------------------------------------",
    "1     וינברג חיים             1908       נגר         זגרסקה 14",
    "2     וינברג לאה              1912       תופרת       זגרסקה 14",
    "3     רוזנצויג יעקב           1905       סופר        פיוטרקובסקה 88",
    "4     רוזנצויג רבקה           1914       -           פיוטרקובסקה 88",
    "5     גולדשטיין מנחם          1899       חייט        נובומייסקה 3",
    "6     גולדשטיין אסתר          1901       -           נובומייסקה 3",
    "7     קפלן דוד                1921       תלמיד       וולבורסקה 27",
    "8     פרידמן שרה              1917       אחות        לימנובסקיגו 41",
    "9     הורוביץ אברהם           1890       מלמד        ברזינסקה 9",
    "10    זילברברג מרים           1925       תלמידה      ברזינסקה 9",
    "11    לוין יצחק               1910       אופה        מלינרסקה 5",
    "12    שפירא חנה               1903       מורה        פרנצישקנסקה 30",
]
page(["רשימת דיירים - בית ועד מס' 7, לודז'", "נערכה ביום 14 במרץ 1941", ""] + rows,
     font("FreeMono.ttf", 24), seed=2, blur=0.8).save(OUT / "residents_list_1941.png")

# 3. מכתב רשמי בגרמנית (אותיות לטיניות, שמות בתעתיק אחר)
page([
    "Der Aelteste der Juden in Litzmannstadt",
    "Arbeitsamt                         Litzmannstadt, den 3. Juni 1942",
    "",
    "Bescheinigung Nr. 4471",
    "",
    "Es wird hiermit bescheinigt, dass der Tischler",
    "Chaim WAJNBERG, geboren am 12. April 1908 in Lodz,",
    "wohnhaft Hohensteinerstrasse 14, in der",
    "Tischlerei-Abteilung, Betrieb Nr. 12, beschaeftigt ist.",
    "",
    "Der Genannte ist als Facharbeiter eingestuft und",
    "erhaelt die Zusatzverpflegung der Gruppe B.",
    "",
    "Diese Bescheinigung ist gueltig bis zum 30. September 1942.",
    "",
    "",
    "Der Leiter des Arbeitsamtes            (Stempel)",
    "gez. B. Fuchs",
], font("FreeMono.ttf", 30), rtl=False, seed=3).save(OUT / "bescheinigung_1942.jpg", quality=75)

# 4. כרטיס רישום בפולנית
page([
    "KARTA REJESTRACYJNA  Nr 2086",
    "Centralny Komitet Zydow w Polsce - Oddzial w Lodzi",
    "",
    "Nazwisko:           ROZENCWAJG",
    "Imie:               Rywka",
    "Nazwisko panienskie: Wajnberg",
    "Data urodzenia:     5 listopada 1914",
    "Miejsce urodzenia:  Lodz",
    "Imiona rodzicow:    Mordechaj i Fajga",
    "Zawod:              krawcowa",
    "Adres przed wojna:  Lodz, ul. Piotrkowska 88",
    "Pobyt w czasie wojny: getto Lodz, oboz Ravensbrueck",
    "Adres obecny:       Lodz, ul. Narutowicza 32",
    "Poszukuje:          meza Jakuba, brata Chaima",
    "",
    "Lodz, dnia 19 sierpnia 1945 r.",
    "Podpis rejestrujacego: M. Kowalska",
], font("FreeMono.ttf", 30), rtl=False, seed=4, blur=0.9).save(OUT / "karta_1945.png")

# 5. עדות בעברית - PDF של שני עמודים (טקסט רציף)
p1 = page([
    "עדות מס' 318                                   נגבתה בחיפה, 2 במאי 1958",
    "העדה: רבקה רוזנצווייג לבית ויינברג, ילידת לודז', 1914",
    "",
    "נולדתי בלודז' למשפחה של נגרים. אבי מרדכי החזיק בית מלאכה",
    "ברחוב זגרסקה. היינו ארבעה ילדים: שמואל, חיים, אני ובלומה",
    "הקטנה. שמואל עלה לארץ ישראל בשנת 1935 וגר בתל אביב.",
    "בשנת 1938 נישאתי ליעקב, ועברנו לגור ברחוב פיוטרקובסקה.",
    "",
    "בפברואר 1940 נצטווינו לעבור לגטו. קיבלנו חדר אחד ברחוב",
    "ברזינסקה, ובו גרנו שבע נפשות. בעלי עבד בבית המלאכה לנייר,",
    "ואני תפרתי מדים בבית החרושת של הגטו, שתים עשרה שעות ביום.",
    "מנת הלחם היתה כיכר אחת לשמונה ימים.",
    "",
    "אחי חיים עבד בנגריה והיתה לו תעודת עובד מומחה, ובזכותה",
    "קיבל תוספת מזון שחילק עם ההורים. אבי נפטר בגטו בחורף 1942",
    "ממחלת ריאות. את אמי לקחו בספטמבר של אותה שנה ולא ראינו",
    "אותה עוד.",
], SERIF, seed=5)
p2 = page([
    "- 2 -",
    "",
    "באוגוסט 1944 חוסל הגטו. נלקחנו ברכבות, ושם הפרידו ביני",
    "ובין בעלי. אותי העבירו אחרי כמה שבועות למחנה רוונסבריק,",
    "ושם עבדתי בבית מלאכה לתפירה עד האביב.",
    "",
    "שוחררתי בסוף אפריל 1945. חזרתי ברגל וברכבות ללודז'",
    "לחפש את בני משפחתי, ונרשמתי בועד היהודי. חיכיתי שם",
    "שנה שלמה. מכל המשפחה שנשארה בפולין לא חזר איש מלבדי.",
    "",
    "בשנת 1946 יצאתי מפולין, שהיתי במחנה עקורים ליד מינכן,",
    "ובאוקטובר 1948 עליתי לארץ באניה \"עצמאות\". אחי שמואל",
    "חיכה לי בנמל חיפה. לא ראינו זה את זה שלוש עשרה שנה.",
    "",
    "כיום אני גרה בחיפה ועובדת כתופרת.",
    "",
    "העדות נרשמה מפי העדה ואושרה בחתימתה.",
    "רשם: א. בן-דוד",
], SERIF, seed=6)
p1.save(OUT / "testimony_318.pdf", save_all=True, append_images=[p2], resolution=150)

print("נוצרו:", ", ".join(sorted(p.name for p in OUT.iterdir())))
