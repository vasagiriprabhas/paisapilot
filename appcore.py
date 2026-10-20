"""Day 6: everything the web page needs, without any web code, so it can be tested without a browser.

app.py only draws things; the decisions live here: loading a statement, applying the person's corrections,
computing every report once, and limiting AI use.
"""
import json
import threading
import time
from dataclasses import dataclass, field

import pandas as pd

import forecast as F
import generate_sample as gs
import insights as I
from categoriser import UNKNOWN, categorise
from merchants import add_merchants
from statement import StatementProblem, parse_statement

CATEGORIES = ["Food", "Groceries", "Shopping", "Travel", "Bills & Utilities", "Health", "Entertainment", "Education",
              "Subscriptions", "Rent", "EMI & Loans", "Investments", "Cash", "Transfers", "Income", UNKNOWN]
DAILY_AI_LIMIT = 150            # AI questions for ALL visitors together, per day
VISIT_AI_LIMIT = 10             # AI questions for one visit
MAX_CORRECTIONS_FILE_KB = 200


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Loaded:
    df: pd.DataFrame            # parsed rows with merchant / text / channel (not yet categorised)
    info: dict
    source: str                 # file name, or "sample"


def load_statement(data: bytes, name: str) -> Loaded:
    """Raises StatementProblem with a plain-English message if the file cannot be read."""
    result = parse_statement(data, name)
    return Loaded(add_merchants(result.df), result.info, name)


def load_sample() -> Loaded:
    stmt, _ = gs.generate()
    return load_statement(stmt.to_csv(index=False).encode(), "sample")


# ---------------------------------------------------------------------------------------------------------------
def clean_corrections(raw: dict) -> dict:
    """Keep only {merchant: known category}. Anything else in an uploaded file is ignored."""
    return {str(k).strip().upper(): v for k, v in raw.items() if isinstance(v, str) and v in CATEGORIES and str(k).strip()}


def corrections_from_bytes(data: bytes) -> dict:
    if len(data) > MAX_CORRECTIONS_FILE_KB * 1024:
        raise ValueError("That corrections file is too large.")
    try:
        raw = json.loads(data.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError):
        raise ValueError("That file is not a Paisapilot corrections file.")
    if not isinstance(raw, dict):
        raise ValueError("That file is not a Paisapilot corrections file.")
    return clean_corrections(raw)


def corrections_to_bytes(corrections: dict) -> bytes:
    return json.dumps(corrections, indent=2, ensure_ascii=False).encode("utf-8")


def with_correction(corrections: dict, merchant: str, category: str) -> dict:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    new = dict(corrections)
    new[str(merchant).strip().upper()] = category
    return new


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Report:
    df: pd.DataFrame
    monthly: pd.DataFrame
    categories: pd.DataFrame            # category, total, share_pct (money out)
    recurring: pd.DataFrame
    duplicates: pd.DataFrame
    unusual: pd.DataFrame
    forecast: F.Forecast
    numbers: dict = field(default_factory=dict)
    to_review: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_report(loaded: Loaded, corrections: dict = None) -> Report:
    df = categorise(loaded.df, corrections=corrections or {})
    monthly = I.monthly_summary(df)
    complete, _ = F._complete_months(df)
    full = monthly[monthly["month"].isin([str(m) for m in complete])] if len(complete) else monthly
    out = df[(df["amount"] < 0) & (df["category"] != "Income")]
    cats = out.groupby("category")["amount"].sum().mul(-1).sort_values(ascending=False).rename("total").reset_index()
    cats["share_pct"] = (cats["total"] / cats["total"].sum() * 100).round(1) if len(cats) else []
    numbers = {
        "date_from": loaded.info["date_from"], "date_to": loaded.info["date_to"], "rows": int(len(df)),
        "money_in": float(df.loc[df["amount"] > 0, "amount"].sum()), "money_out": float(-df.loc[df["amount"] < 0, "amount"].sum()),
        "avg_spending": float(full["money_out"].mean()) if len(full) else 0.0,
        "avg_income": float(full["money_in"].mean()) if len(full) else 0.0,
        "avg_saving": float(full["saved"].mean()) if len(full) else 0.0,
        "full_months": int(len(full)),
        "balance_check": loaded.info["balance_check"], "warnings": list(loaded.info.get("warnings", [])),
    }
    numbers["saving_rate_pct"] = round(numbers["avg_saving"] / numbers["avg_income"] * 100, 1) if numbers["avg_income"] else None
    return Report(df=df, monthly=monthly, categories=cats, recurring=I.recurring_payments(df),
                  duplicates=I.duplicate_charges(df), unusual=I.unusual_spends(df), forecast=F.forecast(df),
                  numbers=numbers, to_review=merchants_to_review(df))


def merchants_to_review(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Merchants worth a human look: Uncategorised first, then the ones the model guessed, biggest spend first."""
    d = df[(df["amount"] < 0) & (df["cat_source"].isin(["none", "model", "default"]))]
    if d.empty:
        return pd.DataFrame(columns=["merchant", "category", "rows", "total", "why"])
    g = d.groupby("merchant").agg(category=("category", "first"), rows=("amount", "size"), total=("amount", lambda s: -s.sum()),
                                  src=("cat_source", "first")).reset_index()
    g["why"] = g["src"].map({"none": "no rule matched", "model": "guessed by the model", "default": "assumed"})
    g["order"] = (g["category"] != UNKNOWN).astype(int)
    return g.sort_values(["order", "total"], ascending=[True, False]).head(limit).drop(columns=["src", "order"]).reset_index(drop=True)


def category_breakdown(df: pd.DataFrame, month: str = "") -> pd.DataFrame:
    d = df[(df["amount"] < 0) & (df["category"] != "Income")].copy()
    if month:
        d = d[pd.to_datetime(d["date"]).dt.to_period("M").astype(str) == month]
    out = d.groupby("category")["amount"].sum().mul(-1).sort_values(ascending=False).rename("total").reset_index()
    return out


def transaction_view(df: pd.DataFrame, category: str = "", text: str = "") -> pd.DataFrame:
    d = df
    if category:
        d = d[d["category"] == category]
    if text.strip():
        t = text.strip().upper()
        d = d[d["merchant"].str.upper().str.contains(t, regex=False) | d["description"].str.upper().str.contains(t, regex=False)]
    out = d[["date", "merchant", "amount", "category", "cat_source", "cat_confidence"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values("date", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
class DailyGuard:
    """One shared counter for every visitor, so a public demo cannot use up the free AI quota in a day."""

    def __init__(self, limit: int = DAILY_AI_LIMIT, today=None):
        self.limit = limit
        self._today = today or (lambda: time.strftime("%Y-%m-%d", time.gmtime()))
        self._day, self.used = self._today(), 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self._today() != self._day:
                self._day, self.used = self._today(), 0
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


class Budget:
    """AI calls for one visit, also counted against the shared daily guard. ask.ask_money calls take()."""

    def __init__(self, limit: int = VISIT_AI_LIMIT, shared: DailyGuard = None):
        self.limit, self.used, self.shared = limit, 0, shared

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        if self.shared is not None and not self.shared.take():
            return False
        self.used += 1
        return True
