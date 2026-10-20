"""Day 3 tests: insights. Run:  python test_day3.py"""
import pandas as pd

import generate_sample as gs
import insights as I
from categoriser import categorise
from merchants import add_merchants
from statement import parse_statement


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        raise SystemExit(f"\nFAILED: {name}")


def make(rows):
    """rows: (date, merchant, amount, category). Money out is negative."""
    return pd.DataFrame([{"date": pd.Timestamp(d), "merchant": m, "description": f"UPI/{m}", "amount": a,
                          "category": c} for d, m, a, c in rows])


def monthly(merchant, amounts, cat="Subscriptions", day=10, start=(2026, 1)):
    out = []
    for i, a in enumerate(amounts):
        y, m = start[0] + (start[1] - 1 + i) // 12, (start[1] - 1 + i) % 12 + 1
        out.append((f"{y}-{m:02d}-{day:02d}", merchant, -a, cat))
    return out


stmt, labels = gs.generate()
df = categorise(add_merchants(parse_statement(stmt.to_csv(index=False).encode(), "s.csv").df))

rec = I.recurring_payments(df)
subs = set(rec[rec["kind"] == "Subscription"]["merchant"])
check("finds all 5 planted subscriptions", subs == {"NETFLIX", "SPOTIFY", "YOUTUBE PREMIUM", "AMAZON PRIME", "GOOGLE ONE"})
check("finds rent, EMI and SIP as regular payments",
      {"LANDLORD SURESH", "HDFC BANK LTD", "SBI MUTUAL FUND SIP"} <= set(rec["merchant"]))
check("electricity (amount changes) is a regular bill", "TNEB ELECTRICITY" in set(rec[rec["kind"] == "Regular bill"]["merchant"]))
check("food places you just visit are NOT called bills", not ({"CHAI POINT", "IRCTC", "SWIGGY"} & set(rec["merchant"])))
check("Google One started late: counted from when it began", int(rec[rec["merchant"] == "GOOGLE ONE"]["times"].iloc[0]) == 4)
check("Netflix costs 649 a month, 7788 a year",
      rec[rec["merchant"] == "NETFLIX"][["monthly_cost", "yearly_cost"]].iloc[0].tolist() == [649.0, 7788.0])

dup = I.duplicate_charges(df)
check("finds the one planted double charge (Amazon, 1799)", len(dup) == 1 and dup.iloc[0]["merchant"] == "AMAZON PAY INDIA"
      and dup.iloc[0]["amount"] == 1799 and dup.iloc[0]["extra_paid"] == 1799)
un = I.unusual_spends(df)
check("finds the one planted huge purchase (Apple, 64900) and nothing else", list(un["merchant"]) == ["APPLE STORE"])
check("every finding has a plain-English reason", dup["reason"].str.len().gt(20).all() and un["reason"].str.len().gt(20).all())

# --- hand-made cases ---
price = make(monthly("NETFLIX", [199, 199, 199, 199, 649, 649]))
r = I.recurring_payments(price)
check("price rise is reported", len(r) == 1 and "price went up" in r.iloc[0]["note"])

stopped = make(monthly("HOTSTAR", [299] * 4) + monthly("NETFLIX", [649] * 9))
r = I.recurring_payments(stopped).set_index("merchant")
check("a subscription that stopped months ago is flagged", r.loc["HOTSTAR", "status"] == "may have stopped"
      and r.loc["NETFLIX", "status"] == "active")

check("only 2 payments is not enough to call it recurring", I.recurring_payments(make(monthly("NETFLIX", [649, 649]))).empty)

busy = make([(f"2026-0{m}-{d:02d}", "SWIGGY", -300, "Food") for m in range(1, 7) for d in (3, 9, 15, 21, 27)])
check("a busy shop (5 orders a month) is not a subscription", I.recurring_payments(busy).empty)

tea = make([("2026-03-05", "CHAI POINT", -60, "Food")] * 2)
check("two cheap identical purchases are not flagged as double charge", I.duplicate_charges(tea).empty)
two_cabs = make([("2026-03-05", "UBER INDIA", -250, "Travel"), ("2026-03-06", "UBER INDIA", -250, "Travel")])
check("same amount on different days is not a double charge", I.duplicate_charges(two_cabs).empty)
twice = make([("2026-03-05", "UBER INDIA", -250, "Travel")] * 2)
r = I.duplicate_charges(twice)
check("same merchant, amount and day is flagged, extra = one charge", len(r) == 1 and r.iloc[0]["extra_paid"] == 250)

fav = make([(f"2026-01-{d:02d}", "SWIGGY", -300, "Food") for d in range(1, 21)]
           + [(f"2026-02-{d:02d}", "BARBEQUE NATION", -2200, "Food") for d in (3, 10, 17, 24)])
check("a favourite expensive restaurant is not 'unusual'", I.unusual_spends(fav).empty)
spike = make([(f"2026-01-{d:02d}", "SWIGGY", -300, "Food") for d in range(1, 21)] + [("2026-02-01", "NEW PLACE", -5000, "Food")])
check("a one-off big payment in a normal category is unusual", list(I.unusual_spends(spike)["merchant"]) == ["NEW PLACE"])
check("big rent is never 'unusual'", I.unusual_spends(make(monthly("LANDLORD", [9500] * 9, "Rent") + [("2026-10-01", "LANDLORD", -40000, "Rent")])).empty)

ms = I.monthly_summary(df)
check("monthly summary adds up to the statement", abs(ms["money_in"].sum() - df[df["amount"] > 0]["amount"].sum()) < 0.01
      and abs(ms["money_out"].sum() + df[df["amount"] < 0]["amount"].sum()) < 0.01)

empty = df.head(0)
check("empty input does not crash", I.recurring_payments(empty).empty and I.duplicate_charges(empty).empty
      and I.unusual_spends(empty).empty)
before = df.copy()
I.recurring_payments(df); I.duplicate_charges(df); I.unusual_spends(df)
check("input table is not modified", df.equals(before))

print("\nALL TESTS PASSED")
