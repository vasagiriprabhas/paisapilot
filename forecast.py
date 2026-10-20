"""Day 4: predict next month's spending, and measure how wrong that kind of prediction has been.

How it works (no AI, no black box):
    1. Add up spending per month and category. A half-finished first or last month is dropped, not guessed.
    2. One-off purchases (the 64,900 laptop) are taken out of the history: they will not repeat next month.
    3. Try four simple rules for "next month" on every past month, using only the months before it:
           last month | average of last 3 | middle of last 3 | average of last 6
       and keep the rule that would have been closest. This is the BACKTEST.
    4. Predict next month with that rule, and give a range as wide as the typical past mistake.

If there are fewer than 4 full months there is nothing to test the rule on, so the answer is marked "low confidence".
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import insights as I

METHODS = {
    "average of the last 3 months": lambda v: float(np.mean(v[-3:])),
    "middle of the last 3 months": lambda v: float(np.median(v[-3:])),
    "average of the last 6 months": lambda v: float(np.mean(v[-6:])),
    "same as last month": lambda v: float(v[-1]),
}
MIN_BACKTEST_HISTORY = 3        # a month is only predicted once at least 3 earlier months exist
SKIP_CATEGORIES = {"Income"}


@dataclass
class Forecast:
    next_month: str = ""
    total: float = 0.0
    low: float = 0.0
    high: float = 0.0
    method: str = ""
    confidence: str = "none"                       # high / medium / low / none
    by_category: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)       # month, actual, predicted, error_pct
    typical_error_pct: float = float("nan")
    baseline_error_pct: float = float("nan")       # error of the plain "same as last month" rule
    expected_income: float = 0.0
    expected_saving: float = 0.0
    one_offs_removed: pd.DataFrame = field(default_factory=pd.DataFrame)
    months_used: int = 0
    notes: list = field(default_factory=list)


def _complete_months(df: pd.DataFrame):
    """Returns (list of complete month periods, notes). Drops a first month that started late and a last month that
    ended early, because their totals are small for the wrong reason."""
    d = pd.to_datetime(df["date"])
    first, last = d.min(), d.max()
    months = pd.period_range(first.to_period("M"), last.to_period("M"), freq="M")
    notes = []
    if len(months) and first.day > 7:
        notes.append(f"{months[0]} left out: the statement starts on day {first.day}, so that month is incomplete.")
        months = months[1:]
    if len(months) and last.day < last.days_in_month - 6:
        notes.append(f"{months[-1]} left out: the statement ends on day {last.day}, so that month is incomplete.")
        months = months[:-1]
    return list(months), notes


def _monthly_by_category(spend: pd.DataFrame, months) -> pd.DataFrame:
    table = spend.pivot_table(index="month", columns="category", values="out", aggfunc="sum", fill_value=0.0)
    return table.reindex(months, fill_value=0.0)


def _predict(table: pd.DataFrame, upto: int, method) -> pd.Series:
    """Prediction for the month at position `upto`, using only the months before it."""
    return table.iloc[:upto].apply(lambda col: method(col.to_numpy()))


def forecast(df: pd.DataFrame) -> Forecast:
    out = Forecast()
    if df is None or df.empty or (df["amount"] < 0).sum() == 0:
        out.notes.append("There is no spending in this statement to forecast from.")
        return out
    months, notes = _complete_months(df)
    out.notes += notes
    if len(months) < 2:
        out.notes.append("I need at least 2 full months of history to forecast.")
        return out

    spend = I._debits(df)
    spend = spend[~spend["category"].isin(SKIP_CATEGORIES)].copy()
    spend["month"] = spend["date"].dt.to_period("M")
    spend = spend[spend["month"].isin(months)].copy()

    ones = I.unusual_spends(df)
    if len(ones):
        key = set(zip(pd.to_datetime(ones["date"]).dt.date, ones["merchant"], ones["amount"].round(2)))
        mask = [(d.date(), m, round(a, 2)) in key for d, m, a in zip(spend["date"], spend["merchant"], spend["out"])]
        out.one_offs_removed = spend[mask][["date", "merchant", "category", "out"]].rename(columns={"out": "amount"}).reset_index(drop=True)
        spend = spend[[not x for x in mask]]
        if len(out.one_offs_removed):
            out.notes.append(f"{len(out.one_offs_removed)} one-off purchase(s) were left out of the history "
                             f"(they will not repeat), see one_offs_removed.")

    table = _monthly_by_category(spend, months)
    out.months_used = len(table)
    out.next_month = str(months[-1] + 1)

    # -- backtest every rule ---------------------------------------------------------------------------------
    results = {}
    if len(table) > MIN_BACKTEST_HISTORY:
        actual = table.sum(axis=1)
        for name, fn in METHODS.items():
            rows = []
            for t in range(MIN_BACKTEST_HISTORY, len(table)):
                pred = float(_predict(table, t, fn).sum())
                if actual.iloc[t] > 0:                       # an empty month has no percentage error to measure
                    rows.append({"month": str(table.index[t]), "actual": float(actual.iloc[t]), "predicted": pred})
            bt = pd.DataFrame(rows, columns=["month", "actual", "predicted"])
            bt["error_pct"] = (bt["predicted"] - bt["actual"]) / bt["actual"] * 100
            results[name] = bt
        if any(len(b) == 0 for b in results.values()):
            results = {}
    if results:
        best = min(results, key=lambda n: results[n]["error_pct"].abs().mean())
        out.method, out.backtest = best, results[best]
        out.typical_error_pct = float(results[best]["error_pct"].abs().mean())
        out.baseline_error_pct = float(results["same as last month"]["error_pct"].abs().mean())
        band = max(0.05, float(np.percentile(results[best]["error_pct"].abs(), 80)) / 100)
        out.confidence = "high" if len(results[best]) >= 6 and out.typical_error_pct <= 10 else "medium"
    else:
        out.method = "average of the last 3 months"
        out.confidence = "low"
        band = 0.25
        out.notes.append("Fewer than 4 full months, so this could not be tested against the past. Treat it as a rough guide.")

    by_cat = _predict(table, len(table), METHODS[out.method])
    by_cat = by_cat[by_cat > 0].sort_values(ascending=False)
    out.by_category = by_cat.round(0).rename("expected").reset_index().rename(columns={"index": "category"})
    out.total = float(by_cat.sum())
    out.low, out.high = out.total * (1 - band), out.total * (1 + band)

    # -- income and what is left -----------------------------------------------------------------------------
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    income = d[(d["amount"] > 0) & d["month"].isin(months)].groupby("month")["amount"].sum().reindex(months, fill_value=0.0)
    out.expected_income = float(np.median(income.to_numpy()[-3:]))
    out.expected_saving = out.expected_income - out.total
    return out
