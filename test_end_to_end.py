"""Day 7: the whole pipeline, from file bytes to answer, for every statement style and file type.
Run:  python test_end_to_end.py"""
import io
import os
import tempfile

import pandas as pd

import appcore as A
import ask
import generate_sample as gs
from categoriser import evaluate


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        raise SystemExit(f"\nFAILED: {name}")


stmt, labels = gs.generate()
truth = labels["category"]
QUESTIONS = ["How much did I spend on food in March?", "Why was April so high?", "What subscriptions do I pay?",
             "Was I charged twice for anything?", "What will I spend next month?", "which month did I spend the most?",
             "how much did I save?", "hello"]


def run_through(data, name, rows):
    loaded = A.load_statement(data, name)
    report = A.build_report(loaded)
    answers = [ask.ask_money(q, report.df, ai=False) for q in QUESTIONS]
    return loaded, report, answers


for style in ("split", "single"):
    df = stmt if style == "split" else gs.to_single_amount(stmt)
    files = {"csv": df.to_csv(index=False).encode()}
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    files["xlsx"] = buf.getvalue()
    for kind, data in files.items():
        loaded, report, answers = run_through(data, f"s.{kind}", len(stmt))
        label = f"{style} / {kind}"
        check(f"{label}: all 416 rows read, balance adds up", report.numbers["rows"] == 416
              and report.numbers["balance_check"]["status"] == "ok")
        check(f"{label}: totals match the data", abs(report.numbers["money_out"] - pd.to_numeric(stmt["Debit"], errors="coerce").fillna(0).sum()) < 1
              and abs(report.numbers["money_in"] - pd.to_numeric(stmt["Credit"], errors="coerce").fillna(0).sum()) < 1)
        check(f"{label}: categorisation is at least 95% right", evaluate(report.df, truth)["accuracy"] >= 0.95)
        check(f"{label}: the planted findings are found",
              len(report.recurring) == 10 and len(report.duplicates) == 1 and len(report.unusual) == 1)
        check(f"{label}: forecast is produced", report.forecast.total > 0 and report.forecast.next_month == "2026-10")
        check(f"{label}: every question gets a non-empty answer", all(len(a.text) > 20 for a in answers))
        check(f"{label}: the April answer mentions the Apple purchase", "APPLE STORE" in answers[1].text)

# PDF (first 120 rows only, as the sample PDF is)
pdf_rows = stmt.head(120)
path = os.path.join(tempfile.mkdtemp(), "s.pdf")
gs.write_pdf(pdf_rows, path)
with open(path, "rb") as f:
    loaded, report, answers = run_through(f.read(), "s.pdf", 120)
check("pdf: 120 rows read", report.numbers["rows"] == 120)
check("pdf: money out matches", abs(report.numbers["money_out"] - pd.to_numeric(pdf_rows["Debit"], errors="coerce").fillna(0).sum()) < 1)
check("pdf: every question gets an answer", all(len(a.text) > 20 for a in answers))

# a statement with only a few months and a messy preamble still works end to end
short = gs.generate(months=3)[0]
text = "Account Statement\nCustomer: TEST USER\nPeriod: Q4\n\n" + short.to_csv(index=False)
loaded, report, answers = run_through(text.encode(), "short.csv", len(short))
check("3-month statement with a preamble: reads, forecasts with low confidence", report.numbers["rows"] == len(short)
      and report.forecast.confidence in ("low", "none"))
check("3-month statement: answers do not crash", all(len(a.text) > 10 for a in answers))

print("\nALL TESTS PASSED")
