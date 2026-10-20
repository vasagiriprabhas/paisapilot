"""Day 1 demo: read a bank statement and show what Paisapilot understood.

Usage:
    python day1.py sample                    (make and read a fictional statement)
    python day1.py my_statement.csv          (CSV, Excel or PDF)
"""
import sys

import pandas as pd

from merchants import add_merchants
from statement import StatementProblem, parse_statement


def show(result):
    info, df = result.info, add_merchants(result.df)
    print("=" * 66)
    print(f"Read {info['rows']} transactions, {info['date_from']} to {info['date_to']}")
    print(f"Money in:  {info['money_in']:>12,.2f}\nMoney out: {info['money_out']:>12,.2f}")
    check = info["balance_check"]
    if check["status"] == "ok":
        print(f"Balance check: OK ({check['matched']} of {check['checked']} rows add up)")
    elif check["status"] == "mismatch":
        print(f"Balance check: PROBLEM ({check['matched']} of {check['checked']} rows add up)")
    else:
        print("Balance check: not possible (no running balance in this statement)")
    if info["skipped_lines"]:
        print(f"Skipped {info['skipped_lines']} non-transaction lines (headers, totals, notes)")
    for w in info["warnings"]:
        print("Note:", w)
    print("\nTop 10 places money went:")
    spend = df[df["amount"] < 0].groupby("merchant")["amount"].agg(["sum", "count"]).sort_values("sum").head(10)
    for name, row in spend.iterrows():
        print(f"  {name[:34]:34} {-row['sum']:>11,.2f}   ({int(row['count'])}x)")
    print("\nFirst 8 rows, as understood:")
    pd.set_option("display.width", 140, "display.max_colwidth", 44)
    print(df[["date", "amount", "channel", "merchant"]].head(8).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        if sys.argv[1] == "sample":
            import generate_sample
            statement, _ = generate_sample.generate()
            statement.to_csv("sample_statement.csv", index=False)
            path = "sample_statement.csv"
            print("Wrote sample_statement.csv (fictional data)")
        else:
            path = sys.argv[1]
        with open(path, "rb") as f:
            show(parse_statement(f.read(), path))
    except StatementProblem as problem:
        sys.exit(f"Problem: {problem}")
