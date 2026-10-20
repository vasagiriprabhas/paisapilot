"""Day 2: put every transaction into a category (Food, Groceries, Rent ...).

Order of trust, highest first:
    1. YOU      - a correction you made ("SWIGGY INSTAMART is Groceries"). Always wins.
    2. RULES    - keyword lists written by hand ("SWIGGY" -> Food). Precise and easy to explain.
    3. MODEL    - a small text classifier (TF-IDF + logistic regression) for merchants no rule knows.
                  It learns from the rows the rules were sure about, your corrections and a small built-in
                  list of well-known merchants. It only answers when it is reasonably confident.
    4. else     - "Uncategorised". Better to say "I don't know" than to guess wrong and mislead you.

Every row gets: category, cat_source (you / rule / model / default / none) and cat_confidence (0 to 1),
so the app can show WHY a row has its category and let you fix it.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

UNKNOWN = "Uncategorised"
MODEL_MIN_CONFIDENCE = 0.45      # below this the model stays quiet
MIN_TRAIN_ROWS = 15

# Order matters: the first category whose keyword matches wins, so specific ones come before general ones
# (AMAZON PRIME is a subscription, plain AMAZON is shopping; SWIGGY INSTAMART is groceries, plain SWIGGY is food).
RULES = [
    ("Income", ["SALARY", "PAYROLL", "STIPEND", "INTEREST PAID", "INT PD", "DIVIDEND", "CASHBACK", "REFUND"]),
    ("Rent", ["RENT", "LANDLORD", "HOUSE OWNER", "PG RENT", "HOSTEL FEE"]),
    ("EMI & Loans", ["EMI", "LOAN", "BAJAJ FINSERV", "HOME CREDIT", "CREDIT CARD", "KREDITBEE", "MONEYVIEW", "NACH LOAN"]),
    ("Investments", ["MUTUAL FUND", "SIP", "ZERODHA", "GROWW", "UPSTOX", "SMALLCASE", "PPF", "NPS", "ANGEL ONE",
                     "KUVERA", "COIN BY ZERODHA", "FIXED DEPOSIT", "RECURRING DEPOSIT", "GOLD BEES"]),
    ("Subscriptions", ["NETFLIX", "SPOTIFY", "YOUTUBE PREMIUM", "AMAZON PRIME", "PRIME VIDEO", "HOTSTAR", "DISNEY",
                       "GOOGLE ONE", "GOOGLE STORAGE", "ICLOUD", "SONYLIV", "ZEE5", "JIOCINEMA", "CHATGPT", "OPENAI",
                       "MICROSOFT 365", "ADOBE", "LINKEDIN PREMIUM", "AUDIBLE", "GAANA", "WYNK", "CANVA"]),
    ("Education", ["UDEMY", "COURSERA", "AMAZON BOOKS", "BOOKS", "BYJU", "UNACADEMY", "EDX", "SCHOOL", "COLLEGE",
                   "TUITION", "UNIVERSITY", "EXAM FEE", "NPTEL", "SKILLSHARE", "PHYSICSWALLAH", "VEDANTU"]),
    ("Groceries", ["BIGBASKET", "DMART", "BLINKIT", "ZEPTO", "INSTAMART", "GROCER", "GROCERY", "SUPERMARKET",
                   "RELIANCE FRESH", "RELIANCE SMART", "MORE RETAIL", "SPENCER", "KIRANA", "BIG BAZAAR",
                   "NATURES BASKET", "JIOMART", "MILK", "VEGETABLE", "FRUITS", "STAR BAZAAR"]),
    ("Food", ["SWIGGY", "ZOMATO", "DOMINO", "PIZZA", "MCDONALD", "KFC", "BURGER", "CAFE", "CHAI", "COFFEE",
              "STARBUCKS", "RESTAURANT", "HOTEL", "BIRYANI", "BAKERY", "BARBEQUE", "DINING", "BHAVAN", "TIFFIN",
              "SWEETS", "FOOD", "EATSURE", "FAASOS", "SUBWAY", "DHABA", "MESS", "ICE CREAM", "JUICE", "KITCHEN"]),
    ("Travel", ["IRCTC", "UBER", "OLA", "REDBUS", "RAPIDO", "PETROL", "FUEL", "INDIAN OIL", "IOCL", "HPCL", "BPCL",
                "SHELL", "FASTAG", "MAKEMYTRIP", "GOIBIBO", "INDIGO", "AIR INDIA", "AKASA", "SPICEJET", "METRO",
                "CLEARTRIP", "YATRA", "ABHIBUS", "TOLL", "PARKING", "TAXI", "CABS", "RAILWAY", "APSRTC", "TSRTC"]),
    ("Bills & Utilities", ["ELECTRICITY", "TNEB", "BESCOM", "APSPDCL", "TSSPDCL", "WATER", "BWSSB", "BROADBAND",
                           "RECHARGE", "AIRTEL", "JIO", "VODAFONE", "BSNL", "GAS", "DTH", "TATA PLAY", "INSURANCE",
                           "LIC", "POSTPAID", "BILLDESK", "BILL PAY", "ACT FIBERNET", "MUNICIPAL", "PROPERTY TAX"]),
    ("Health", ["PHARMACY", "APOLLO", "1MG", "MEDPLUS", "HOSPITAL", "CLINIC", "DIAGNOSTIC", "PATH LAB", "CULT FIT",
                "GYM", "PHARMEASY", "DENTAL", "NETMEDS", "PRACTO", "MEDICAL", "MEDICALS", "DOCTOR", "OPTICAL"]),
    ("Entertainment", ["BOOKMYSHOW", "PVR", "INOX", "CINEMA", "STEAM", "DREAM11", "GAMES", "GAMING", "PLAYSTATION",
                       "MOVIE", "TICKETNEW", "MPL", "ZUPEE"]),
    ("Shopping", ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "MEESHO", "NYKAA", "DECATHLON", "LENSKART", "IKEA", "CROMA",
                  "RELIANCE DIGITAL", "APPLE STORE", "TATA CLIQ", "SHOPPERS STOP", "LIFESTYLE", "WESTSIDE",
                  "MAX FASHION", "PANTALOONS", "ZUDIO", "BATA", "SHOPPING"]),
]
_RULE_RES = [(cat, re.compile(r"(?<![A-Z0-9])(?:" + "|".join(re.escape(k) for k in kws) + r")S?(?![A-Z0-9])"))
             for cat, kws in RULES]

# words that show a name belongs to a business, so it is NOT a person we sent money to
BUSINESS_WORDS = {"LTD", "PVT", "LIMITED", "PRIVATE", "STORE", "STORES", "SHOP", "TRADERS", "TRADING", "ENTERPRISES",
                  "SERVICES", "MART", "FOODS", "AGENCY", "AGENCIES", "CENTRE", "CENTER", "COMPANY", "CO", "INDIA",
                  "TECHNOLOGIES", "SOLUTIONS", "SYSTEMS", "MOTORS", "ELECTRONICS", "SUPPLIERS", "WORKS", "GARAGE",
                  "VENTURES", "INDUSTRIES", "ASSOCIATES", "MARKETING", "DISTRIBUTORS", "BROTHERS", "BROS", "SONS",
                  "CORPORATION", "CORP", "GROUP", "HOLDINGS", "LABS", "LAB", "FASHION", "TEXTILES", "JEWELLERS",
                  "FURNITURE", "ENTERPRISE", "TRADERS", "PHARMA", "AUTOMOBILES", "CONSULTANCY", "CONSULTING", "INFOTECH"}

# well-known merchants the keyword rules do NOT cover on purpose: they give the model something to learn from
SEED = {
    "Food": ["PIZZA HUT", "BURGER KING", "HALDIRAMS", "A2B ADYAR ANANDA BHAVAN", "PUNJABI DHABA", "BEHROUZ BIRYANI",
             "TEA TIME", "WAFFLE HOUSE", "NATURALS ICE CREAM", "PARADISE BIRYANI", "CAFE COFFEE DAY", "SAGAR RATNA"],
    "Groceries": ["NILGIRIS", "LULU HYPERMARKET", "FRESHTOHOME", "LICIOUS", "COUNTRY DELIGHT", "SUPER MARKET",
                  "GREEN BASKET", "DAILY NEEDS STORE", "SRI LAKSHMI PROVISIONS"],
    "Shopping": ["RELIANCE TRENDS", "MAX LIFESTYLE", "SNITCH", "BEWAKOOF", "BOAT LIFESTYLE", "FIRSTCRY", "PEPPERFRY",
                 "URBAN LADDER", "TATA NEU", "VIJAY SALES", "SAMSUNG SHOP"],
    "Travel": ["NAMMA YATRI", "BLUSMART", "AIR ASIA", "VISTARA", "ZOOMCAR", "SRS TRAVELS", "KSRTC BUS", "VRL TRAVELS",
               "PETRO PUMP", "BHARAT PETROLEUM", "NEXA SERVICE"],
    "Bills & Utilities": ["TANGEDCO", "MSEDCL", "ADANI GAS", "MAHANAGAR GAS", "HATHWAY", "SUN DIRECT", "PAYTM POSTPAID",
                          "BHARTI AIRTEL", "VI RECHARGE", "TORRENT POWER", "SOCIETY MAINTENANCE"],
    "Health": ["MAX HOSPITAL", "FORTIS", "MANIPAL", "SRL DIAGNOSTICS", "DR LAL PATHLABS", "WELLNESS FOREVER",
               "ONEMG", "ANYTIME FITNESS", "TATA HEALTH"],
    "Entertainment": ["BOOK MY SHOW", "SONY PICTURES", "BOWLING ALLEY", "FUN CITY", "XBOX", "EPIC GAMES", "TICKETS"],
    "Education": ["GREAT LEARNING", "SIMPLILEARN", "CODECADEMY", "INTERNSHALA", "UPGRAD", "ALLEN CAREER", "LEETCODE"],
    "Subscriptions": ["APPLE MUSIC", "TINDER", "NOTION", "DROPBOX", "GITHUB", "MEDIUM", "PRIME LITE", "YOUTUBE MUSIC"],
    "Investments": ["ICICI PRUDENTIAL MF", "HDFC MF", "AXIS MF", "NIPPON SIP", "PARAG PARIKH", "STOCK BROKING"],
    "EMI & Loans": ["TATA CAPITAL", "LOAN REPAYMENT", "HDFC LOAN", "SBI CARD", "CARD OUTSTANDING"],
}


def _blob(row) -> str:
    return f'{row["merchant"]} {row["text"]}'.upper()


def _rule_category(row):
    """Returns (category, confidence) from the keyword rules, or (None, 0)."""
    blob = _blob(row)
    if row["channel"] == "atm":
        return "Cash", 0.95
    for cat, rx in _RULE_RES:
        if rx.search(blob):
            if cat == "Income" and row["direction"] != "credit":
                continue                       # "REFUND" or "CASHBACK" only means income when money comes IN
            return cat, 0.9
    return None, 0.0


def _looks_like_person(row) -> bool:
    """An UPI payment to a plain 1-3 word name with nothing business-like in it is almost always a person."""
    if row["channel"] not in ("upi", "imps"):
        return False
    words = re.findall(r"[A-Z]+", str(row["merchant"]).upper())
    return 1 <= len(words) <= 3 and not (set(words) & BUSINESS_WORDS) and not re.search(r"\d", str(row["merchant"]))


def _model_text(merchant, text) -> str:
    return f"{merchant} {text}".upper()


def _train(df, labelled):
    """labelled: list of (text, category). Returns a fitted pipeline or None if there is too little to learn from."""
    if len(labelled) < MIN_TRAIN_ROWS or len({c for _, c in labelled}) < 2:
        return None
    texts, cats = zip(*labelled)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)
    clf = LogisticRegression(C=20, max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(texts), cats)
    return vec, clf


def _seed_rows():
    return [(name.upper(), cat) for cat, names in SEED.items() for name in names]


def load_corrections(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return {str(k).upper(): v for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}


def save_corrections(corrections: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2, ensure_ascii=False)


def categorise(df: pd.DataFrame, corrections: dict = None, use_rules: bool = True, use_model: bool = True) -> pd.DataFrame:
    """Adds category, cat_source and cat_confidence. `df` needs merchant, text, channel, direction (see merchants.py).

    corrections: {"MERCHANT NAME": "Category"} - what the person told us; they always win.
    use_rules / use_model exist so we can measure each layer separately.
    """
    corrections = {str(k).upper(): v for k, v in (corrections or {}).items()}
    out = df.copy().reset_index(drop=True)
    n = len(out)
    cat, src, conf = [UNKNOWN] * n, ["none"] * n, [0.0] * n

    for i, row in out.iterrows():
        m = str(row["merchant"]).upper()
        if m in corrections:
            cat[i], src[i], conf[i] = corrections[m], "you", 1.0
            continue
        if use_rules:
            c, p = _rule_category(row)
            if c:
                cat[i], src[i], conf[i] = c, "rule", p
                continue
            if row["direction"] == "debit" and _looks_like_person(row):
                cat[i], src[i], conf[i] = "Transfers", "rule", 0.7
                continue

    if use_model:
        # learn from: built-in examples + rows already settled by you or the rules + your corrections
        learn = _seed_rows()
        learn += [(_model_text(out.at[i, "merchant"], out.at[i, "text"]), cat[i])
                  for i in range(n) if src[i] in ("you", "rule") and cat[i] != UNKNOWN]
        learn += [(k, v) for k, v in corrections.items()]
        fitted = _train(out, learn)
        todo = [i for i in range(n) if src[i] == "none"]
        if fitted and todo:
            vec, clf = fitted
            probs = clf.predict_proba(vec.transform([_model_text(out.at[i, "merchant"], out.at[i, "text"]) for i in todo]))
            for i, p in zip(todo, probs):
                k = int(np.argmax(p))
                if p[k] >= MODEL_MIN_CONFIDENCE:
                    cat[i], src[i], conf[i] = clf.classes_[k], "model", round(float(p[k]), 3)

    # money coming in that nothing recognised: UPI = a person paying you back, anything else = income
    for i, row in out.iterrows():
        if src[i] == "none" and row["direction"] == "credit":
            cat[i], src[i], conf[i] = ("Transfers" if row["channel"] == "upi" else "Income"), "default", 0.5

    out["category"], out["cat_source"], out["cat_confidence"] = cat, src, conf
    return out


def evaluate(categorised: pd.DataFrame, truth: pd.Series) -> dict:
    """Compare against the answer key. Only the sample statement has one."""
    truth = truth.reset_index(drop=True)
    right = (categorised["category"].reset_index(drop=True) == truth)
    answered = categorised["category"].reset_index(drop=True) != UNKNOWN
    return {"rows": len(truth), "accuracy": float(right.mean()),
            "answered": float(answered.mean()),
            "accuracy_when_answered": float(right[answered].mean()) if answered.any() else 0.0}
