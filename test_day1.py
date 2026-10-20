"""Self-test for Day 1 (no internet, no API key, no real bank data needed).

Run:  python test_day1.py        Every line should say ok.
"""
import io

import pandas as pd

import generate_sample as gs
from merchants import add_merchants, channel_of, clean_description
from statement import StatementProblem, parse_number, parse_statement

failures = 0


def total(column):
    return pd.to_numeric(column, errors="coerce").fillna(0).sum()


def check(name, condition):
    global failures
    failures += not condition
    print(("ok   " if condition else "FAIL ") + name)


def problem(data, name=""):
    try:
        parse_statement(data, name)
    except StatementProblem as e:
        return str(e)
    return None


def as_csv(df, **kw):
    return df.to_csv(index=False, **kw).encode()


stmt, labels = gs.generate()
spent = total(stmt["Debit"])
earned = total(stmt["Credit"])

# --- the sample data itself
check("sample has about a year of transactions", 300 < len(stmt) < 700 and len(labels) == len(stmt))
check("sample balance never goes negative", (stmt["Balance"] >= 0).all())
check("sample is repeatable (same seed, same data)", gs.generate()[0].equals(stmt))

# --- reading the same statement in different layouts
a = parse_statement(as_csv(stmt), "a.csv")
check("split Debit/Credit CSV: every row read", a.info["rows"] == len(stmt))
check("totals match the file exactly", abs(a.info["money_out"] - spent) < 0.01 and abs(a.info["money_in"] - earned) < 0.01)
check("running balance check passes", a.info["balance_check"]["status"] == "ok")
check("money out is negative, money in positive", (a.df.loc[a.df["direction"] == "debit", "amount"] < 0).all())

single = gs.to_single_amount(stmt)
b = parse_statement(as_csv(single), "b.csv")
check("single Amount + Dr/Cr column gives the same totals", abs(b.info["money_out"] - spent) < 0.01 and abs(b.info["money_in"] - earned) < 0.01)

unsigned = single.drop(columns=["Dr/Cr"])
c = parse_statement(as_csv(unsigned), "c.csv")
check("unsigned amounts are signed from the running balance", abs(c.info["money_out"] - spent) < 1 and c.info["warnings"])

no_flag_no_balance = unsigned.drop(columns=["Balance"])
check("unsigned amounts with no balance: clear message", "money in from money out" in (problem(as_csv(no_flag_no_balance)) or ""))

preamble = ("HDFC BANK\nStatement of account for A B\nAddress: 12 Some Street, Tirupati\nAccount No: XXXXXX1234\n\n").encode() \
    + as_csv(stmt) + b"\n,,,Page total,,\n"
d = parse_statement(preamble, "d.csv")
check("lines before the table and a totals line are ignored", d.info["rows"] == len(stmt) and d.info["skipped_lines"] >= 1)

newest_first = stmt.iloc[::-1].reset_index(drop=True)
e = parse_statement(as_csv(newest_first), "e.csv")
check("newest-first statement is put in date order", e.df["date"].is_monotonic_increasing and e.info["rows"] == len(stmt))
check("newest-first still passes the balance check", e.info["balance_check"]["status"] == "ok")

semi = stmt.to_csv(index=False, sep=";").encode()
check("semicolon-separated file works", parse_statement(semi, "f.csv").info["rows"] == len(stmt))
check("UTF-8 with BOM and latin-1 both work", parse_statement(b"\xef\xbb\xbf" + as_csv(stmt)).info["rows"] == len(stmt)
      and parse_statement(as_csv(stmt).replace(b"TNEB", b"T\xe9NEB")).info["rows"] == len(stmt))

renamed = stmt.rename(columns={"Narration": "Particulars", "Debit": "Withdrawal Amt.", "Credit": "Deposit Amt.", "Date": "Txn Date"})
check("other banks' column names are recognised", parse_statement(as_csv(renamed)).info["rows"] == len(stmt))

xl = io.BytesIO()
stmt.to_excel(xl, index=False)
check("Excel file works", parse_statement(xl.getvalue(), "g.xlsx").info["rows"] == len(stmt))

import os, tempfile
_tmp = tempfile.mkdtemp()
TMP_TABLE = os.path.join(_tmp, "table.pdf")
TMP_TEXT = os.path.join(_tmp, "text.pdf")
pdf_part = stmt.head(60)
gs.write_pdf(pdf_part, TMP_TABLE)
p = parse_statement(open(TMP_TABLE, "rb").read(), "h.pdf")
check("PDF statement: every row read", p.info["rows"] == len(pdf_part))
check("PDF statement: amounts match", abs(p.info["money_out"] - total(pdf_part["Debit"])) < 0.01)

# PDF with no drawn table lines: read from the text of each line instead
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
head = stmt.head(40)
cv = canvas.Canvas(TMP_TEXT, pagesize=A4)
cv.setFont("Helvetica", 6.5)
cv.drawString(20, 810, "Statement of Account   Date  Narration  Withdrawal  Deposit  Balance")
for n, (_, r) in enumerate(head.iterrows()):
    amount = r["Debit"] if r["Debit"] != "" else r["Credit"]
    cv.drawString(20, 795 - n * 11, f'{r["Date"]} {r["Narration"][:55]} {float(amount):,.2f} {float(r["Balance"]):,.2f}')
cv.save()
q = parse_statement(open(TMP_TEXT, "rb").read(), "t.pdf")
check("PDF without table lines: rows and totals match",
      q.info["rows"] == len(head) and abs(q.info["money_out"] - total(head["Debit"])) < 0.01)

# --- number and date oddities
check("Indian commas", parse_number("1,23,456.78") == (123456.78, None))
check("brackets mean negative", parse_number("(500.00)") == (-500.0, None))
check("Dr / Cr suffix", parse_number("500.00 Dr") == (500.0, "dr") and parse_number("1,000.00CR") == (1000.0, "cr"))
check("rupee sign and Rs.", parse_number("₹ 2,500") == (2500.0, None) and parse_number("Rs. 99.50") == (99.5, None))
check("blank and dash are empty", parse_number("") == (None, None) and parse_number("-") == (None, None))
dates = pd.DataFrame({"Date": ["05-Mar-25", "06-Mar-25"], "Narration": ["UPI SWIGGY", "UPI ZOMATO"], "Debit": ["100", "200"], "Credit": ["", ""]})
check("05-Mar-25 style dates", list(parse_statement(as_csv(dates)).df["date"].dt.day) == [5, 6])
dm = pd.DataFrame({"Date": ["05/03/2025", "13/03/2025"], "Narration": ["A", "B"], "Debit": ["1", "2"], "Credit": ["", ""]})
check("dates are read day-first (05/03 is 5 March)", parse_statement(as_csv(dm)).df["date"].dt.month.tolist() == [3, 3])

# --- bad input gets a friendly message, never a crash
check("empty file", problem(b"") is not None)
check("random text", problem(b"hello there, this is not a statement\nat all") is not None)
check("binary junk", problem(bytes(range(256)) * 20) is not None)
check("huge file", problem(b"x" * (6 * 1024 * 1024)) is not None)
check("broken PDF", problem(b"%PDF-1.4 nope", "x.pdf") is not None)
check("table with a header but no rows", problem(b"Date,Narration,Debit,Credit\n") is not None)

# --- merchant cleaning
expect = {
    "UPI/DR/412345678901/SWIGGY/yesb/swiggy@yesb/UPI": "SWIGGY",
    "UPI-ZOMATO-zomato@icici-ICIC0000011-401234567890-UPI": "ZOMATO",
    "POS 412345XXXXXX1234 DOMINOS PIZZA": "DOMINOS PIZZA",
    "UPI/412345678901/AMAZON PAY INDIA/Payment from PhonePe": "AMAZON PAY INDIA",
    "ACH D- HDFC BANK LTD-LOAN EMI-12345678": "HDFC BANK LTD",
    "NEFT-CR-TECHNOVA SOLUTIONS PVT LTD-SALARY": "TECHNOVA SOLUTIONS PVT LTD",
    "ATM WDL 123456 ANNA NAGAR CHENNAI": "ATM CASH",
    "UPI/DR/123456789012/RAVI KUMAR/ybl/ravi@ybl/Payment": "RAVI KUMAR",
}
for raw, merchant in expect.items():
    check(f"merchant: {merchant}", clean_description(raw)["merchant"] == merchant)
check("the text keeps words the categoriser needs (EMI, SALARY, RENT)",
      "LOAN EMI" in clean_description("ACH D- HDFC BANK LTD-LOAN EMI-123")["text"]
      and "RENT" in clean_description("UPI/DR/123456789012/LANDLORD SURESH/okaxis/s@okaxis/Rent")["text"])
check("channels", channel_of("UPI/DR/1/X") == "upi" and channel_of("POS 1234XXXXXX5678 SHOP") == "card"
      and channel_of("NEFT-CR-X") == "neft" and channel_of("ATM WDL 1") == "atm")
cleaned = add_merchants(a.df)
check("no merchant in the whole sample contains a long number or a payment ID",
      not cleaned["merchant"].str.contains(r"\d{5,}|@").any())
check("the sample has a manageable number of distinct merchants", cleaned["merchant"].nunique() < 60)

print(f"\n{'ALL TESTS PASSED' if not failures else str(failures) + ' TEST(S) FAILED'}")
raise SystemExit(1 if failures else 0)
