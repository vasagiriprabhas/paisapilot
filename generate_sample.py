"""Day 1: make a realistic but FICTIONAL bank statement, so Paisapilot can be built, tested and demoed
without anyone's real data.

    python generate_sample.py                 -> sample_statement.csv, sample_labels.csv (the right category of every row)
    python generate_sample.py --style split   -> Debit / Credit columns (default)
    python generate_sample.py --style single  -> one Amount column with a Dr/Cr flag
    python generate_sample.py --pdf           -> also sample_statement.pdf

Everything is random but seeded, so you always get the same statement. The "labels" file is the answer key we use
on Day 2 to measure how accurate the categoriser is.
"""
import argparse
import random
from datetime import date, timedelta

import pandas as pd

# (merchant text as it appears in a bank narration, category, typical amount range)
MERCHANTS = {
    "Food": [("SWIGGY", 150, 650), ("ZOMATO", 150, 700), ("DOMINOS PIZZA", 250, 700), ("MCDONALDS", 120, 450),
             ("CHAI POINT", 60, 220), ("BARBEQUE NATION", 900, 2200), ("HOTEL SARAVANA BHAVAN", 150, 600)],
    "Groceries": [("BIGBASKET", 400, 2200), ("DMART", 600, 3000), ("BLINKIT", 150, 900), ("ZEPTO", 150, 800),
                  ("RELIANCE FRESH", 300, 1800)],
    "Shopping": [("AMAZON PAY INDIA", 300, 4500), ("FLIPKART", 400, 5000), ("MYNTRA", 600, 3500), ("AJIO", 500, 2500)],
    "Travel": [("IRCTC", 300, 2200), ("UBER INDIA", 90, 450), ("OLA CABS", 80, 400), ("REDBUS", 400, 1500),
               ("INDIAN OIL PETROL", 500, 1800), ("RAPIDO", 40, 220)],
    "Bills & Utilities": [("TNEB ELECTRICITY", 600, 1600), ("JIO PREPAID RECHARGE", 299, 749),
                          ("AIRTEL BROADBAND", 799, 1179), ("BWSSB WATER", 150, 400)],
    "Health": [("APOLLO PHARMACY", 150, 1800), ("1MG", 200, 1500), ("CULT FIT", 999, 1999)],
    "Entertainment": [("BOOKMYSHOW", 200, 900), ("PVR CINEMAS", 250, 800), ("STEAM GAMES", 300, 1800)],
    "Education": [("UDEMY", 449, 699), ("COURSERA", 2000, 4000), ("AMAZON BOOKS", 200, 900)],
}
SUBSCRIPTIONS = [("NETFLIX", "Subscriptions", 649), ("SPOTIFY", "Subscriptions", 119),
                 ("YOUTUBE PREMIUM", "Subscriptions", 129), ("AMAZON PRIME", "Subscriptions", 299),
                 ("GOOGLE ONE", "Subscriptions", 130)]
PEOPLE = ["RAVI KUMAR", "SANDHYA REDDY", "ANIL VARMA", "MEGHANA P", "KIRAN TEJA"]
CATS_BY_WEIGHT = ["Food"] * 5 + ["Groceries"] * 4 + ["Shopping"] * 3 + ["Travel"] * 4 + ["Bills & Utilities"] * 1 + \
                 ["Health"] * 1 + ["Entertainment"] * 2 + ["Education"] * 1


def _upi(merchant: str, rng: random.Random) -> str:
    """Narrations from different banks look different. Mix a few common styles."""
    ref = "".join(rng.choice("0123456789") for _ in range(12))
    handle = merchant.lower().replace(" ", "")[:12]
    style = rng.choice(["a", "b", "c", "d"])
    if style == "a":
        return f"UPI/DR/{ref}/{merchant}/yesb/{handle}@yesb/UPI"
    if style == "b":
        return f"UPI-{merchant}-{handle}@icici-ICIC0000011-{ref}-UPI"
    if style == "c":
        return f"UPI/{ref}/{merchant}/Payment from PhonePe"
    return f"POS {rng.randint(400000, 499999)}XXXXXX{rng.randint(1000, 9999)} {merchant}"


def generate(months: int = 12, seed: int = 7, salary: int = 60000, start: date = None):
    """Returns (statement_df, labels_df). Columns: Date, Narration, Ref No, Debit, Credit, Balance."""
    rng = random.Random(seed)
    start = start or date(2025, 10, 1)
    rows = []      # (date, narration, debit, credit, category)

    def add(d, narration, debit=0.0, credit=0.0, category=""):
        rows.append((d, narration, round(debit, 2), round(credit, 2), category))

    for m in range(months):
        first = date(start.year + (start.month - 1 + m) // 12, (start.month - 1 + m) % 12 + 1, 1)
        nxt = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
        days = (nxt - first).days
        day = lambda n: first + timedelta(days=min(n, days) - 1)

        add(day(1), "NEFT-CR-TECHNOVA SOLUTIONS PVT LTD-SALARY", credit=salary, category="Income")
        add(day(2), f"UPI/DR/{rng.randint(10**11, 10**12 - 1)}/LANDLORD SURESH/okaxis/suresh@okaxis/Rent", debit=9500, category="Rent")
        add(day(5), f"ACH D- HDFC BANK LTD-LOAN EMI-{rng.randint(10**7, 10**8 - 1)}", debit=4200, category="EMI & Loans")
        add(day(7), f"ACH D- SBI MUTUAL FUND SIP-{rng.randint(10**7, 10**8 - 1)}", debit=3000, category="Investments")
        for name, cat, amt in SUBSCRIPTIONS[: 3 + (m >= 4) + (m >= 8)]:     # new subscriptions appear over the year
            add(day(10 + SUBSCRIPTIONS.index((name, cat, amt)) * 3), _upi(name, rng), debit=amt, category=cat)
        add(day(12), _upi("JIO PREPAID RECHARGE", rng), debit=rng.choice([299, 599, 749]), category="Bills & Utilities")
        add(day(rng.randint(15, 20)), _upi("TNEB ELECTRICITY", rng), debit=rng.randint(600, 1600), category="Bills & Utilities")

        for _ in range(rng.randint(16, 24)):                         # everyday spending
            cat = rng.choice(CATS_BY_WEIGHT)
            name, lo, hi = rng.choice(MERCHANTS[cat])
            add(day(rng.randint(1, days)), _upi(name, rng), debit=round(rng.uniform(lo, hi) / 5) * 5, category=cat)
        for _ in range(rng.randint(1, 3)):
            add(day(rng.randint(1, days)), f"ATM WDL {rng.randint(100000, 999999)} ANNA NAGAR CHENNAI", debit=rng.choice([500, 1000, 2000]), category="Cash")
        for _ in range(rng.randint(1, 3)):                           # sending money to friends and family
            person = rng.choice(PEOPLE)
            add(day(rng.randint(1, days)), f"UPI/DR/{rng.randint(10**11, 10**12 - 1)}/{person}/ybl/{person.split()[0].lower()}@ybl/Payment", debit=rng.choice([200, 300, 500, 800, 1500]), category="Transfers")
        if rng.random() < 0.5:
            person = rng.choice(PEOPLE)
            add(day(rng.randint(1, days)), f"UPI/CR/{rng.randint(10**11, 10**12 - 1)}/{person}/ybl/{person.split()[0].lower()}@ybl/Payment", credit=rng.choice([300, 500, 1000]), category="Transfers")

    # Two planted problems for the insights day to find:
    dup_day = start + timedelta(days=120)
    add(dup_day, _upi("AMAZON PAY INDIA", rng), debit=1799, category="Shopping")
    add(dup_day, rows[-1][1], debit=1799, category="Shopping")                          # charged twice, same day
    add(start + timedelta(days=200), _upi("APPLE STORE", rng), debit=64900, category="Shopping")   # one very large purchase

    rows.sort(key=lambda r: r[0])
    balance, out, labels = 38000.0, [], []
    for i, (d, narration, debit, credit, cat) in enumerate(rows):
        balance = round(balance + credit - debit, 2)
        ref = f"S{rng.randint(10**7, 10**8 - 1)}"
        out.append({"Date": d.strftime("%d/%m/%Y"), "Narration": narration, "Ref No": ref,
                    "Debit": debit or "", "Credit": credit or "", "Balance": balance})
        labels.append({"row": i, "category": cat})
    return pd.DataFrame(out), pd.DataFrame(labels)


def to_single_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Some banks give one Amount column and a Dr/Cr flag instead of two columns."""
    out = df.copy()
    out["Amount"] = [d if d != "" else c for d, c in zip(df["Debit"], df["Credit"])]
    out["Dr/Cr"] = ["DR" if d != "" else "CR" for d in df["Debit"]]
    return out[["Date", "Narration", "Ref No", "Amount", "Dr/Cr", "Balance"]]


def write_pdf(df: pd.DataFrame, path: str):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()
    cells = [list(df.columns)] + [[Paragraph(str(v), styles["BodyText"]) if c == "Narration" else str(v)
                                   for c, v in row.items()] for _, row in df.iterrows()]
    table = Table(cells, repeatRows=1, colWidths=[62, 280, 70, 60, 60, 70])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=20, rightMargin=20).build(
        [Paragraph("Sample Bank - Statement of Account (FICTIONAL)", styles["Title"]), Spacer(1, 8), table])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["split", "single"], default="split")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--months", type=int, default=12)
    args = ap.parse_args()
    statement, labels = generate(months=args.months)
    if args.style == "single":
        statement = to_single_amount(statement)
    statement.to_csv("sample_statement.csv", index=False)
    labels.to_csv("sample_labels.csv", index=False)
    print(f"Wrote sample_statement.csv ({len(statement)} rows) and sample_labels.csv")
    if args.pdf:
        write_pdf(statement.head(120), "sample_statement.pdf")
        print("Wrote sample_statement.pdf (first 120 rows)")
