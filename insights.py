"""Day 3: find what a person would want to know about their statement, with a reason for every finding.

    recurring_payments(df)  -> subscriptions, bills and fixed payments that repeat (and ones that look stopped or got dearer)
    duplicate_charges(df)   -> the same merchant charging the same amount twice on the same day
    unusual_spends(df)      -> a single payment far bigger than that category normally is
    monthly_summary(df)     -> money in, money out and savings rate per month

Everything here is plain arithmetic on the table, no AI, so every number can be checked by hand. `df` is the output of
categorise() (needs date, amount, merchant, text, category). Money out is negative, money in is positive.
Amounts are written "Rs 1,234" because the Windows terminal cannot always print the rupee sign.
"""
import numpy as np
import pandas as pd

BILL_CATEGORIES = {"Bills & Utilities", "Subscriptions", "EMI & Loans", "Rent", "Investments", "Education"}
NOT_UNUSUAL = {"Rent", "EMI & Loans", "Investments", "Income", "Cash"}   # large by nature, so "unusual" would be noise


def rs(x) -> str:
    return f"Rs {x:,.0f}"


def _debits(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["amount"] < 0].copy()
    d["out"] = -d["amount"]
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date")


# ---------------------------------------------------------------------------------------------------------------
def recurring_payments(df: pd.DataFrame, min_times: int = 3) -> pd.DataFrame:
    """Merchants paid again and again on a steady rhythm (about monthly, or weekly at a fixed amount)."""
    d = _debits(df)
    if d.empty:
        return _empty(["merchant", "category", "kind", "every", "typical_amount", "times", "first", "last",
                       "monthly_cost", "yearly_cost", "status", "note"])
    end = pd.to_datetime(df["date"]).max()
    rows = []
    for merchant, g in d.groupby("merchant"):
        if len(g) < min_times:
            continue
        gaps = g["date"].diff().dt.days.dropna().to_numpy()
        gap = float(np.median(gaps))
        amounts = g["out"].to_numpy()
        spread = (amounts.max() - amounts.min()) / np.median(amounts)
        months = g["date"].dt.to_period("M")
        distinct = months.nunique()
        span = (months.max() - months.min()).n + 1
        days = g["date"].dt.day.to_numpy()
        monthly = (distinct >= min_times and distinct / span >= 0.7 and len(g) / distinct <= 1.5
                   and np.median(np.abs(days - np.median(days))) <= 4)      # same few days each month
        if monthly:
            every, per_month = "monthly", float(max(1, round(len(g) / distinct)))
        elif 6 <= gap <= 8 and np.mean((gaps >= 5) & (gaps <= 9)) >= 0.7 and spread <= 0.1:
            every, per_month = "weekly", 52 / 12            # weekly needs a fixed amount, or a busy shop looks like a bill
        else:
            continue

        recent = float(np.median(amounts[-3:]))
        fixed = spread <= 0.1
        category = g["category"].mode().iloc[0] if "category" in g else ""
        if category not in BILL_CATEGORIES and (not fixed or every == "weekly"):
            continue                       # a restaurant you happen to visit monthly is a habit, not a bill
        notes = []
        price_up = len(amounts) >= 4 and abs(amounts[-1] - amounts[-2]) < 0.01 and amounts[-1] > amounts[0] * 1.03 \
            and amounts[0] == amounts[1]
        if price_up:
            notes.append(f"price went up from {rs(amounts[0])} to {rs(amounts[-1])}")
        if not fixed and not price_up:
            notes.append(f"amount changes ({rs(amounts.min())} to {rs(amounts.max())})")
        since_last = (end - g["date"].iloc[-1]).days
        stopped = since_last > (50 if every == "monthly" else 15)
        status = "may have stopped" if stopped else "active"
        if stopped:
            notes.append(f"last paid {since_last} days before the end of the statement")

        kind = "Subscription" if category == "Subscriptions" else ("Regular payment" if fixed else "Regular bill")
        rows.append({"merchant": merchant, "category": category, "kind": kind, "every": every,
                     "typical_amount": round(recent, 2), "times": len(g), "first": g["date"].iloc[0].date(),
                     "last": g["date"].iloc[-1].date(), "monthly_cost": round(recent * per_month, 2),
                     "yearly_cost": round(recent * per_month * 12, 2), "status": status, "note": "; ".join(notes)})
    out = pd.DataFrame(rows)
    return out.sort_values("monthly_cost", ascending=False).reset_index(drop=True) if len(out) else _empty(list(out.columns))


# ---------------------------------------------------------------------------------------------------------------
def duplicate_charges(df: pd.DataFrame, min_amount: float = 100) -> pd.DataFrame:
    """Same merchant, same amount, same day. Small amounts and cash are ignored (two teas can be two teas)."""
    d = _debits(df)
    d = d[(d["out"] >= min_amount) & (d["category"] != "Cash")] if len(d) else d
    cols = ["date", "merchant", "amount", "times", "likelihood", "reason", "extra_paid"]
    if d.empty:
        return _empty(cols)
    rows = []
    for (day, merchant, amt), g in d.groupby([d["date"].dt.date, "merchant", d["out"].round(2)]):
        if len(g) < 2:
            continue
        same_text = g["description"].nunique() == 1 if "description" in g else False
        likelihood = "very likely" if same_text else "possible"
        why = (f"{merchant} charged {rs(amt)} {len(g)} times on {day}"
               + (", with the exact same bank entry each time." if same_text else ". The bank entries differ slightly, so it may be two real purchases."))
        rows.append({"date": day, "merchant": merchant, "amount": amt, "times": len(g), "likelihood": likelihood,
                     "reason": why, "extra_paid": round(amt * (len(g) - 1), 2)})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("extra_paid", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
def unusual_spends(df: pd.DataFrame, min_amount: float = 1500, min_history: int = 8) -> pd.DataFrame:
    """A payment is unusual when it is far above that category's normal range AND at least 2.5x its typical size."""
    d = _debits(df)
    cols = ["date", "merchant", "category", "amount", "typical", "times_typical", "reason"]
    if d.empty:
        return _empty(cols)
    rows = []
    for cat, g in d.groupby("category"):
        if cat in NOT_UNUSUAL or len(g) < min_history:
            continue
        med = float(g["out"].median())
        q1, q3 = g["out"].quantile([0.25, 0.75])
        limit = max(q3 + 3 * (q3 - q1), 2.5 * med, min_amount)
        for _, r in g[g["out"] > limit].iterrows():
            own = g[g["merchant"] == r["merchant"]]["out"]
            if len(own) >= 3 and r["out"] <= 2.5 * own.median():
                continue                      # big for the category, but normal for this merchant (a favourite restaurant)
            rows.append({"date": r["date"].date(), "merchant": r["merchant"], "category": cat, "amount": r["out"],
                         "typical": round(med, 2), "times_typical": round(r["out"] / med, 1),
                         "reason": f"{rs(r['out'])} at {r['merchant']} is {r['out'] / med:.0f}x your usual {cat} payment "
                                   f"(typical {rs(med)}, from {len(g)} {cat} payments)."})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("amount", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    out = d.groupby("month")["amount"].agg(money_in=lambda s: s[s > 0].sum(), money_out=lambda s: -s[s < 0].sum()).reset_index()
    out["saved"] = out["money_in"] - out["money_out"]
    out["saved_pct"] = np.where(out["money_in"] > 0, out["saved"] / out["money_in"] * 100, np.nan).round(1)
    return out


def _empty(cols):
    return pd.DataFrame(columns=cols)
