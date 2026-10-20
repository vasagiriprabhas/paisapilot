"""Day 5, step 1: the FACTS. Code computes every number; the AI never does arithmetic.

build_facts(df, question) returns a plain dict:
    overview, months, spending by category and month, top merchants, repeating payments, double charges,
    unusual payments, the forecast, and a "focus" section that digs into whatever the question mentions
    (a month, a category or a merchant).

All amounts are whole rupees, percentages one decimal. The AI is only allowed to quote these numbers, and
ask.py checks that it did.

Privacy: with anonymise=True, people's names (transfers), landlords and employers are hidden before anything is
sent to an AI service. Raw bank narrations, reference numbers and balances are never part of the facts at all.
"""
import re

import numpy as np
import pandas as pd

import forecast as F
import insights as I

MONTH_NAMES = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
               "september": 9, "october": 10, "november": 11, "december": 12,
               "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
               "oct": 10, "nov": 11, "dec": 12}
CATEGORY_WORDS = {
    "Food": ["food", "eating", "restaurant", "restaurants", "dining", "swiggy", "zomato", "takeaway", "eat"],
    "Groceries": ["groceries", "grocery", "vegetables", "supermarket"],
    "Shopping": ["shopping", "shop", "clothes", "online shopping"],
    "Travel": ["travel", "transport", "cab", "cabs", "fuel", "petrol", "commute", "trip", "trips"],
    "Bills & Utilities": ["bills", "bill", "utilities", "electricity", "recharge", "broadband", "water"],
    "Health": ["health", "medical", "medicine", "medicines", "pharmacy", "doctor", "gym", "fitness"],
    "Entertainment": ["entertainment", "movies", "movie", "games", "gaming", "cinema"],
    "Education": ["education", "courses", "course", "learning", "books", "study"],
    "Subscriptions": ["subscriptions", "subscription", "ott", "streaming"],
    "Rent": ["rent", "landlord"],
    "EMI & Loans": ["emi", "loan", "loans"],
    "Investments": ["investment", "investments", "invest", "sip", "mutual"],
    "Cash": ["cash", "atm", "withdrawal", "withdrawals"],
    "Transfers": ["transfer", "transfers", "friends", "family", "sent money"],
}


def whole(x) -> int:
    return int(round(float(x)))


# ---------------------------------------------------------------------------------------------------------------
def anonymise(df: pd.DataFrame) -> pd.DataFrame:
    """Hide names of people, landlords and employers. Shops and apps stay, the AI needs them."""
    out = df.copy()
    out.loc[out["category"] == "Transfers", "merchant"] = "A PERSON"
    out.loc[out["category"] == "Rent", "merchant"] = "LANDLORD"
    out.loc[out["category"] == "Income", "merchant"] = "EMPLOYER"
    return out


def parse_month(question: str, months) -> str:
    """'march', 'april 2026', '2026-04', 'last month'  ->  '2026-03'. Empty string if no month is mentioned.
    A month name that exists in two years means the most recent one."""
    q = question.lower()
    keys = [str(m) for m in months]
    found = re.search(r"\b(20\d{2})-(\d{2})\b", q)
    if found and f"{found.group(1)}-{found.group(2)}" in keys:
        return f"{found.group(1)}-{found.group(2)}"
    if re.search(r"\b(last|previous) month\b", q) and keys:
        return keys[-1]
    for word in re.findall(r"[a-z]+", q):
        number = MONTH_NAMES.get(word)
        if not number:
            continue
        if word == "may" and not re.search(r"\b(in|for|of|during|since|until|by|last|this)\s+may\b|\bmay\s+20\d{2}", q):
            continue                                   # "may I know ..." is not the month
        year = re.search(rf"{word}\s*,?\s*(20\d{{2}})", q)
        options = [k for k in keys if int(k[5:]) == number and (not year or k[:4] == year.group(1))]
        if options:
            return options[-1]
    return ""


def parse_category(question: str) -> str:
    q = " " + re.sub(r"[^a-z& ]", " ", question.lower()) + " "
    for cat, words in CATEGORY_WORDS.items():
        if any(f" {w} " in q for w in words):
            return cat
    for cat in CATEGORY_WORDS:                                    # the full category name, e.g. "bills & utilities"
        if f" {cat.lower()} " in q:
            return cat
    return ""


def parse_merchant(question: str, df: pd.DataFrame) -> str:
    """A merchant is mentioned when its whole name (or its first word, for names longer than 4 letters) appears."""
    q = " " + re.sub(r"[^a-z0-9 ]", " ", question.lower()) + " "
    best = ""
    for m in df["merchant"].dropna().unique():
        name = re.sub(r"[^a-z0-9 ]", " ", str(m).lower()).strip()
        first = name.split()[0] if name else ""
        if name and (f" {name} " in q or (len(first) > 4 and f" {first} " in q)) and len(name) > len(best):
            best = str(m)
    return best


# ---------------------------------------------------------------------------------------------------------------
def _spend_table(df: pd.DataFrame):
    d = I._debits(df)
    d = d[d["category"] != "Income"].copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return d


def build_facts(df: pd.DataFrame, question: str = "", hide_names: bool = False) -> dict:
    if hide_names:
        df = anonymise(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    complete, _ = F._complete_months(df)
    complete = [str(m) for m in complete]
    all_months = sorted(df["date"].dt.to_period("M").astype(str).unique())
    spend = _spend_table(df)
    cat_month = (spend.pivot_table(index="month", columns="category", values="out", aggfunc="sum", fill_value=0.0)
                 .reindex(all_months, fill_value=0.0))
    summary = I.monthly_summary(df).set_index("month")
    use = [m for m in complete if m in summary.index] or list(summary.index)

    facts = {"overview": {
        "statement_from": str(df["date"].min().date()), "statement_to": str(df["date"].max().date()),
        "transactions": int(len(df)), "full_months": len(complete), "months_in_statement": all_months,
        "total_money_in": whole(df.loc[df["amount"] > 0, "amount"].sum()),
        "total_money_out": whole(-df.loc[df["amount"] < 0, "amount"].sum()),
        "average_monthly_spending": whole(summary.loc[use, "money_out"].mean()) if use else 0,
        "average_monthly_income": whole(summary.loc[use, "money_in"].mean()) if use else 0,
        "average_monthly_saving": whole(summary.loc[use, "saved"].mean()) if use else 0,
    }}
    facts["months"] = [{"month": m, "full_month": m in complete, "money_in": whole(r["money_in"]),
                        "money_out": whole(r["money_out"]), "saved": whole(r["saved"]),
                        "saved_pct": None if pd.isna(r["saved_pct"]) else round(float(r["saved_pct"]), 1)}
                       for m, r in summary.iterrows()]
    facts["spending_by_category_and_month"] = {
        m: {c: whole(v) for c, v in cat_month.loc[m].items() if v > 0} for m in all_months}
    typical = cat_month.loc[[m for m in use if m in cat_month.index]].mean() if use else pd.Series(dtype=float)
    facts["average_month_by_category"] = {c: whole(v) for c, v in typical.sort_values(ascending=False).items() if v > 0}

    top = (spend.groupby("merchant")["out"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(10))
    facts["top_merchants"] = [{"merchant": m, "total": whole(r["sum"]), "payments": int(r["count"])} for m, r in top.iterrows()]

    biggest = spend.sort_values("out", ascending=False).head(5)
    facts["largest_payments"] = [{"date": str(r["date"].date()), "merchant": r["merchant"], "category": r["category"],
                                  "amount": whole(r["out"])} for _, r in biggest.iterrows()]

    rec = I.recurring_payments(df)
    facts["repeating_payments"] = [{"merchant": r["merchant"], "kind": r["kind"], "every": r["every"],
                                    "typical_amount": whole(r["typical_amount"]), "per_month": whole(r["monthly_cost"]),
                                    "status": r["status"], "note": r["note"]} for _, r in rec.iterrows()]
    subs = rec[rec["kind"] == "Subscription"]
    facts["subscriptions_cost"] = {"per_month": whole(subs["monthly_cost"].sum()), "per_year": whole(subs["yearly_cost"].sum()),
                                   "count": int(len(subs))}
    dup = I.duplicate_charges(df)
    facts["possible_double_charges"] = [{"date": str(r["date"]), "merchant": r["merchant"], "amount": whole(r["amount"]),
                                         "times": int(r["times"]), "likelihood": r["likelihood"],
                                         "extra_paid": whole(r["extra_paid"])} for _, r in dup.iterrows()]
    un = I.unusual_spends(df)
    facts["unusual_payments"] = [{"date": str(r["date"]), "merchant": r["merchant"], "category": r["category"],
                                  "amount": whole(r["amount"]), "typical_for_category": whole(r["typical"]),
                                  "times_typical": whole(r["times_typical"])} for _, r in un.iterrows()]
    fc = F.forecast(df)
    if fc.total:
        facts["forecast"] = {"month": fc.next_month, "expected_spending": whole(fc.total), "likely_low": whole(fc.low),
                             "likely_high": whole(fc.high), "method": fc.method, "confidence": fc.confidence,
                             "expected_income": whole(fc.expected_income), "expected_saving": whole(fc.expected_saving),
                             "typical_error_pct": None if np.isnan(fc.typical_error_pct) else round(fc.typical_error_pct, 1),
                             "by_category": {r["category"]: whole(r["expected"]) for _, r in fc.by_category.iterrows()}}

    facts["focus"] = _focus(df, spend, cat_month, summary, use, question)
    return facts


def _focus(df, spend, cat_month, summary, use, question) -> dict:
    """Extra detail for whatever the question is about."""
    month = parse_month(question, list(cat_month.index)) if question else ""
    cat = parse_category(question) if question else ""
    merchant = parse_merchant(question, spend) if question else ""
    focus = {}
    if month:
        row = summary.loc[month]
        others = [m for m in use if m != month]
        typical = cat_month.loc[others].mean() if others else pd.Series(dtype=float)
        this = cat_month.loc[month]
        extra = (this - typical.reindex(this.index).fillna(0)).sort_values(ascending=False)
        typical_total = whole(summary.loc[others, "money_out"].mean()) if others else 0
        focus["month"] = {
            "month": month, "money_out": whole(row["money_out"]), "money_in": whole(row["money_in"]),
            "typical_month_spending_other_months": typical_total,
            "more_than_typical_by": whole(row["money_out"] - typical_total),
            "biggest_increases_vs_typical": [{"category": c, "this_month": whole(this[c]),
                                              "typical": whole(typical.get(c, 0)), "extra": whole(v)}
                                             for c, v in extra.head(3).items() if v > 0],
            "biggest_payments": [{"date": str(r["date"].date()), "merchant": r["merchant"], "category": r["category"],
                                  "amount": whole(r["out"])}
                                 for _, r in spend[spend["month"] == month].sort_values("out", ascending=False).head(5).iterrows()],
        }
    if cat:
        series = cat_month[cat] if cat in cat_month.columns else pd.Series(0.0, index=cat_month.index)
        sub = spend[spend["category"] == cat]
        top = sub.groupby("merchant")["out"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(5)
        total_all = spend["out"].sum()
        focus["category"] = {
            "category": cat, "by_month": {m: whole(v) for m, v in series.items()},
            "average_per_month": whole(series.loc[[m for m in use if m in series.index]].mean()) if use else 0,
            "total": whole(sub["out"].sum()), "share_of_all_spending_pct": round(float(sub["out"].sum() / total_all * 100), 1) if total_all else 0.0,
            "top_merchants": [{"merchant": m, "total": whole(r["sum"]), "payments": int(r["count"])} for m, r in top.iterrows()],
        }
    if merchant:
        sub = spend[spend["merchant"] == merchant]
        if len(sub):
            focus["merchant"] = {"merchant": merchant, "total": whole(sub["out"].sum()), "payments": int(len(sub)),
                                 "average_payment": whole(sub["out"].mean()), "biggest_payment": whole(sub["out"].max()),
                                 "by_month": {m: whole(v) for m, v in sub.groupby("month")["out"].sum().items()},
                                 "first": str(sub["date"].min().date()), "last": str(sub["date"].max().date()),
                                 "category": str(sub["category"].mode().iloc[0])}
    return focus


def numbers_in(obj) -> list:
    """Every number inside a facts dict (used to check that an AI answer only quotes real numbers)."""
    found = []
    if isinstance(obj, bool) or obj is None:
        return found
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return [float(obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += numbers_in(v)
    elif isinstance(obj, str):                       # only amounts written inside text, e.g. "price went up from Rs 199 to Rs 649"
        found += [float(x.replace(",", "")) for x in re.findall(r"Rs\s*(\d[\d,]*\.?\d*)", obj)]
    return found
