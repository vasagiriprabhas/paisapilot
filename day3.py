"""Day 3 report: what Paisapilot finds in a statement.

    python day3.py                     (fictional sample statement)
    python day3.py my_statement.csv    (your own file: CSV, Excel or PDF)
"""
import sys

import pandas as pd

import insights as I
from categoriser import categorise
from merchants import add_merchants
from statement import StatementProblem, parse_statement
import generate_sample as gs


def load(arg):
    if arg is None:
        stmt, _ = gs.generate()
        return parse_statement(stmt.to_csv(index=False).encode(), "sample.csv")
    with open(arg, "rb") as f:
        return parse_statement(f.read(), arg)


def main(arg=None):
    df = categorise(add_merchants(load(arg).df))
    pd.set_option("display.width", 160, "display.max_colwidth", 60)

    rec = I.recurring_payments(df)
    print("=" * 70, "\n1. REPEATING PAYMENTS")
    subs = rec[rec["kind"] == "Subscription"]
    if len(rec):
        print(rec[["merchant", "kind", "typical_amount", "times", "status"]].to_string(index=False))
        if len(subs):
            print(f"\nSubscriptions cost you Rs {subs['monthly_cost'].sum():,.0f} a month, "
                  f"Rs {subs['yearly_cost'].sum():,.0f} a year.")
        for _, r in rec[rec["note"] != ""].iterrows():
            print(f"  - {r['merchant']}: {r['note']}")
    else:
        print("none found (need a payment on a steady rhythm for at least 3 months)")

    dup = I.duplicate_charges(df)
    print("\n" + "=" * 70, "\n2. POSSIBLE DOUBLE CHARGES")
    for _, r in dup.iterrows():
        print(f"  [{r['likelihood']}] {r['reason']}  Extra paid: Rs {r['extra_paid']:,.0f}")
    if dup.empty:
        print("none found")

    un = I.unusual_spends(df)
    print("\n" + "=" * 70, "\n3. UNUSUALLY LARGE PAYMENTS")
    for _, r in un.iterrows():
        print(f"  {r['date']}: {r['reason']}")
    if un.empty:
        print("none found")

    print("\n" + "=" * 70, "\n4. MONTH BY MONTH")
    print(I.monthly_summary(df).to_string(index=False))


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
    except StatementProblem as problem:
        sys.exit(f"Problem: {problem}")
