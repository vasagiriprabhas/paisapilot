"""Day 4 tests: forecast. Run:  python test_day4.py"""
import numpy as np
import pandas as pd

import forecast as F
import generate_sample as gs
from categoriser import categorise
from merchants import add_merchants
from statement import parse_statement


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        raise SystemExit(f"\nFAILED: {name}")


def make(months, amount_fn, start=(2026, 1), income=60000, last_day=28, first_day=1):
    """A tiny statement: rent, food, and salary every month. amount_fn(i) is that month's food spend."""
    rows = []
    for i in range(months):
        y, m = start[0] + (start[1] - 1 + i) // 12, (start[1] - 1 + i) % 12 + 1
        d0 = first_day if i == 0 else 1
        d1 = last_day if i == months - 1 else 28
        rows.append((pd.Timestamp(y, m, d0), "SALARY CO", income, "Income"))
        rows.append((pd.Timestamp(y, m, 2), "LANDLORD", -10000, "Rent"))
        rows.append((pd.Timestamp(y, m, d1), "ZOMATO", -amount_fn(i), "Food"))
    start_day = pd.Timestamp(start[0], start[1], first_day)
    rows = [r for r in rows if r[0] >= start_day]            # nothing can be dated before the statement starts
    return pd.DataFrame([{"date": d, "merchant": mer, "description": mer, "amount": a, "category": c} for d, mer, a, c in rows])


stmt, labels = gs.generate()
df = categorise(add_merchants(parse_statement(stmt.to_csv(index=False).encode(), "s.csv").df))
f = F.forecast(df)

check("sample: forecast is for the month after the statement", f.next_month == "2026-10")
check("sample: 12 full months used", f.months_used == 12)
check("sample: total equals the sum of the categories", abs(f.total - f.by_category["expected"].sum()) < len(f.by_category))
check("sample: low < total < high", f.low < f.total < f.high)
check("sample: the 64,900 one-off is taken out of the history", len(f.one_offs_removed) == 1
      and f.one_offs_removed.iloc[0]["amount"] == 64900)
check("sample: forecast is not inflated by the one-off (under Rs 60,000)", f.total < 60000)
check("sample: backtest covers 9 months", len(f.backtest) == 9)
check("sample: typical error is reported and under 25%", 0 < f.typical_error_pct < 25)
check("sample: chosen method is no worse than 'same as last month'", f.typical_error_pct <= f.baseline_error_pct + 1e-9)
check("sample: expected saving = income - spending", abs(f.expected_saving - (f.expected_income - f.total)) < 1e-6)

# perfectly steady spending: the forecast should be exact
steady = F.forecast(make(10, lambda i: 5000))
check("steady spending: forecast is exactly 15,000", abs(steady.total - 15000) < 1)
check("steady spending: backtest error is zero", steady.typical_error_pct < 1e-9)
check("steady spending: high confidence", steady.confidence == "high")

# rising spending: a rule that follows recent months should beat 'same as last month' being noisy
rising = F.forecast(make(10, lambda i: 4000 + 300 * i))
check("rising spending: forecast is above the old average", rising.total > 10000 + 4000 + 300 * 4)

# too little history
short = F.forecast(make(3, lambda i: 5000))
check("3 months: forecast given but marked low confidence", short.confidence == "low" and short.total > 0)
one = F.forecast(make(1, lambda i: 5000))
check("1 month: says it cannot forecast", one.total == 0 and any("2 full months" in n for n in one.notes))

# partial months are dropped, not used
late = F.forecast(make(8, lambda i: 5000, last_day=10))
check("statement ending on day 10: last month left out", late.months_used == 7 and any("left out" in n for n in late.notes))
start_late = F.forecast(make(8, lambda i: 5000, first_day=20))
check("statement starting on day 20: first month left out", start_late.months_used == 7)

# a month with no spending at all (found by the Day 7 end-to-end test): no crash, no warnings
import warnings
gap_rows = make(8, lambda i: 5000)
gap_rows = gap_rows[~((gap_rows["date"].dt.month == 4) & (gap_rows["amount"] < 0))]     # April: salary only
with warnings.catch_warnings():
    warnings.simplefilter("error")
    gap = F.forecast(gap_rows)
check("a month with no spending does not break the backtest", gap.total > 0)

# empty and no-spending
check("empty input does not crash", F.forecast(df.head(0)).total == 0)
check("income-only statement does not crash", F.forecast(df[df["amount"] > 0]).total == 0)
before = df.copy()
F.forecast(df)
check("input table is not modified", df.equals(before))

print("\nALL TESTS PASSED")
