"""Day 4 report: next month's spending, and how trustworthy that prediction is.

    python day4.py                     (fictional sample statement)
    python day4.py my_statement.csv    (your own file: CSV, Excel or PDF)
"""
import sys

import pandas as pd

import forecast as F
from categoriser import categorise
from day3 import load
from merchants import add_merchants
from statement import StatementProblem


def main(arg=None):
    df = categorise(add_merchants(load(arg).df))
    f = F.forecast(df)
    print("=" * 70)
    for note in f.notes:
        print("Note:", note)
    if not f.next_month or f.total == 0:
        return
    print(f"\nExpected spending in {f.next_month}: Rs {f.total:,.0f}   (likely between Rs {f.low:,.0f} and Rs {f.high:,.0f})")
    print(f"Method: {f.method}  |  based on {f.months_used} full months  |  confidence: {f.confidence}")
    print(f"Expected income Rs {f.expected_income:,.0f}  ->  expected saving about Rs {f.expected_saving:,.0f}")

    print("\nWhere it is expected to go:")
    for _, r in f.by_category.iterrows():
        print(f"  {r['category']:20} Rs {r['expected']:>9,.0f}")

    if len(f.backtest):
        print("\n" + "=" * 70)
        print("HOW GOOD IS THIS METHOD? Each past month, predicted using only the months before it:")
        bt = f.backtest.copy()
        bt["actual"], bt["predicted"] = bt["actual"].round(0), bt["predicted"].round(0)
        bt["error_pct"] = bt["error_pct"].round(1)
        print(bt.to_string(index=False))
        close = int((f.backtest["error_pct"].abs() <= 10).sum())
        print(f"\nTypical miss: {f.typical_error_pct:.1f}%.  Within 10%: {close} of {len(f.backtest)} months.")
        print(f"The plain rule 'same as last month' missed by {f.baseline_error_pct:.1f}%,", end=" ")
        print("so the chosen method is better." if f.typical_error_pct < f.baseline_error_pct else "so a simple rule does as well.")
        print("Spending is partly random (a trip, a gift), so no method can be exact. The range above shows that.")


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
    except StatementProblem as problem:
        sys.exit(f"Problem: {problem}")
