"""Day 2 tests: the categoriser. Run:  python test_day2.py"""
import os
import tempfile

import pandas as pd

import generate_sample as gs
from categoriser import UNKNOWN, categorise, evaluate, load_corrections, save_corrections
from merchants import add_merchants
from statement import parse_statement


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        raise SystemExit(f"\nFAILED: {name}")


def frame(desc, amount):
    """One-row statement run through the real reader and cleaner."""
    csv = f"Date,Narration,Debit,Credit,Balance\n01/01/2026,{desc},{amount if amount > 0 else ''},{-amount if amount < 0 else ''},1000\n"
    # amount > 0 means money OUT here
    return add_merchants(parse_statement(csv.encode(), "t.csv").df)


stmt, labels = gs.generate()
df = add_merchants(parse_statement(stmt.to_csv(index=False).encode(), "s.csv").df)
truth = labels["category"]

full = categorise(df)
r = evaluate(full, truth)
check("full system is at least 95% right on the sample", r["accuracy"] >= 0.95)
check("every row gets a category and a reason", full["category"].notna().all() and (full["cat_source"] != "").all())
check("plural names match (MCDONALDS is Food)", (full[full["merchant"] == "MCDONALDS"]["category"] == "Food").all())
check("AMAZON PRIME is a subscription, plain AMAZON is shopping",
      (full[full["merchant"].str.contains("AMAZON PRIME")]["category"] == "Subscriptions").all()
      and (full[full["merchant"] == "AMAZON PAY INDIA"]["category"] == "Shopping").all())
check("ATM withdrawals are Cash", (full[full["channel"] == "atm"]["category"] == "Cash").all())
check("salary credit is Income", (full[full["description"].str.contains("SALARY")]["category"] == "Income").all())
check("a friend paid by UPI is Transfers", (full[full["merchant"].isin(gs.PEOPLE)]["category"] == "Transfers").all())

# corrections always win and are remembered by merchant name
fixed = categorise(df, corrections={"SWIGGY": "Groceries"})
sw = fixed[fixed["merchant"] == "SWIGGY"]
check("a correction beats the rules", (sw["category"] == "Groceries").all() and (sw["cat_source"] == "you").all())
check("correction is case-insensitive", (categorise(df, corrections={"swiggy": "Health"})
                                         .query("merchant == 'SWIGGY'")["category"] == "Health").all())

# the model learns from one correction: an unknown merchant, once corrected, is recognised next time
odd = frame("UPI/DR/123456789012/ZXQ VENTURES/ybl/zxq@ybl/Payment", 500)
check("unknown business is not guessed wildly", odd["merchant"].iloc[0] == "ZXQ VENTURES")
first = categorise(odd)
check("unknown merchant is honestly 'Uncategorised' or low-trust", first["category"].iloc[0] == UNKNOWN
      or first["cat_confidence"].iloc[0] < 0.9)
learned = categorise(odd, corrections={"ZXQ VENTURES": "Shopping"})
check("after a correction it is Shopping", learned["category"].iloc[0] == "Shopping")

# credits that nothing recognises
refund = frame("UPI/CR/123456789012/ZXQ VENTURES/ybl/zxq@ybl/Payment", -300)
check("unknown money coming in is not called spending", categorise(refund)["category"].iloc[0] in ("Transfers", "Income"))
check("REFUND only counts as income when money comes in", categorise(frame("AMAZON REFUND", 200))["category"].iloc[0] != "Income")
check("REFUND credit is income", categorise(frame("AMAZON REFUND", -200))["category"].iloc[0] == "Income")

# input is not changed, empty input is fine
before = df.copy()
categorise(df)
check("input table is not modified", df.equals(before))
check("empty table gives empty result", len(categorise(df.head(0))) == 0)

# corrections file round trip, bad file is ignored
path = os.path.join(tempfile.mkdtemp(), "c.json")
save_corrections({"swiggy": "Food", "Zepto": "Groceries"}, path)
check("corrections save and load", load_corrections(path) == {"SWIGGY": "Food", "ZEPTO": "Groceries"})
check("missing corrections file gives {}", load_corrections(path + ".nope") == {})
open(path, "w").write("{not json")
check("broken corrections file gives {}", load_corrections(path) == {})

print("\nALL TESTS PASSED")
