"""Day 1: read a bank statement (CSV, Excel or PDF) into one clean table, whatever the bank's layout.

Banks all export differently: a few lines of address before the table, "Debit / Credit" columns or one "Amount"
column with Dr/Cr, dates like 05/03/2025 or 05-Mar-25, amounts like 1,23,456.78 or (500.00). This module finds the
header row, recognises the columns by name, and returns:

    date | description | amount | balance | direction
    amount is NEGATIVE for money out and POSITIVE for money in.

Nothing is written to disk: files are read from bytes in memory.
"""
import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_FILE_MB = 5
MAX_PDF_PAGES = 60
MAX_ROWS = 20000


class StatementProblem(Exception):
    """Something the visitor can fix. The message is shown to them as it is."""


@dataclass
class Statement:
    df: pd.DataFrame
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- column recognition
def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# Exact names first, then "contains" fallbacks. Order matters: the first role to match wins.
EXACT = {
    "date": ["transactiondate", "txndate", "trandate", "date", "postingdate", "valuedate", "valdt", "txndt"],
    "description": ["narration", "description", "particulars", "transactiondetails", "transactionremarks", "remarks",
                    "details", "transactiondescription", "transactionparticulars"],
    "debit": ["debit", "debits", "withdrawal", "withdrawals", "withdrawalamt", "withdrawalamount", "debitamount",
              "dr", "paidout", "moneyout", "withdrawalamtinr", "debitinr"],
    "credit": ["credit", "credits", "deposit", "deposits", "depositamt", "depositamount", "creditamount", "cr",
               "paidin", "moneyin", "depositamtinr", "creditinr"],
    "amount": ["amount", "transactionamount", "amountinr", "txnamount", "amt"],
    "drcr": ["drcr", "crdr", "type", "transactiontype", "debitcredit", "creditdebit", "txntype", "cr/dr"],
    "balance": ["balance", "closingbalance", "runningbalance", "availablebalance", "balanceinr", "closebalance",
                "balanceamt"],
}
CONTAINS = [("balance", "balance"), ("debit", "debit"), ("withdraw", "debit"), ("credit", "credit"),
            ("deposit", "credit"), ("narration", "description"), ("particular", "description"),
            ("descript", "description"), ("remark", "description")]


def map_columns(header: list) -> dict:
    """header cells -> {role: column index}. Roles not found are simply absent."""
    names = [_norm(h) for h in header]
    found = {}
    for role, options in EXACT.items():
        for option in options:
            o = _norm(option)
            if o in names and role not in found and names.index(o) not in found.values():
                found[role] = names.index(o)
                break
    for key, role in CONTAINS:
        if role in found:
            continue
        for i, n in enumerate(names):
            if key in n and i not in found.values():
                found[role] = i
                break
    return found


def looks_like_header(cells: list) -> bool:
    found = map_columns(cells)
    has_money = "amount" in found or "debit" in found or "credit" in found
    return "date" in found and "description" in found and has_money


# ---------------------------------------------------------------- numbers and dates
_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*|[-+]?\.\d+")


def parse_number(text):
    """'1,23,456.78' -> (123456.78, None);  '(500.00)' -> (-500.0, None);  '500.00 Dr' -> (500.0, 'dr')."""
    if text is None:
        return None, None
    if isinstance(text, (int, float, np.integer, np.floating)):
        return (None, None) if pd.isna(text) else (float(text), None)
    s = str(text).strip()
    if not s or s in {"-", "--", "nan", "NaN", "None"}:
        return None, None
    flag = None
    m = re.search(r"(?<![A-Za-z])(dr|cr|debit|credit)\.?\s*$", s, re.IGNORECASE)
    if m:
        flag = "dr" if m.group(1).lower() in ("dr", "debit") else "cr"
        s = s[: m.start()].strip()
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("−")
    s = re.sub(r"[₹$₹]|rs\.?|inr", "", s, flags=re.IGNORECASE)
    s = s.replace("−", "-").strip("() ")
    match = _NUM.search(s.replace(" ", ""))
    if not match:
        return None, None
    try:
        value = float(match.group().replace(",", ""))
    except ValueError:
        return None, None
    value = abs(value)
    return (-value if negative else value), flag


DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y", "%d-%b-%y",
                "%d %b %y", "%d %b, %Y", "%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%d %B %Y", "%d-%B-%Y"]


def parse_dates(values: pd.Series) -> pd.Series:
    """Try each common format and keep the one that reads the most rows (so 05/03/2025 is read day-first)."""
    cleaned = values.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?$", "", regex=True, case=False)
    best, best_count = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]"), -1
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
        count = int(parsed.notna().sum())
        if count > best_count:
            best, best_count = parsed, count
    return best


# ---------------------------------------------------------------- turning rows into a table
def _rows_from_csv(text: str) -> list:
    try:
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        return [[c.strip() for c in row] for row in csv.reader(text.splitlines(), dialect)]
    except csv.Error:
        raise StatementProblem("I could not read that file as a table. Upload a CSV, Excel or PDF statement.")


def _rows_from_excel(data: bytes) -> list:
    try:
        frame = pd.read_excel(io.BytesIO(data), header=None, dtype=object)
    except ImportError:
        raise StatementProblem("Excel files need the openpyxl package. Export the statement as CSV or PDF instead.")
    except Exception:
        raise StatementProblem("I could not open that Excel file. Export the statement as CSV or PDF instead.")
    return [["" if pd.isna(c) else str(c).strip() for c in row] for row in frame.values.tolist()]


_PDF_LINE = re.compile(
    r"^(?P<date>\d{1,2}[-/. ][A-Za-z0-9]{1,3}[-/. ]\d{2,4})\s+(?P<text>.+?)\s+"
    r"(?P<amount>\(?[\d,]+\.\d{2}\)?(?:\s?(?:Dr|Cr))?)\s+(?P<balance>[\d,]+\.\d{2}(?:\s?(?:Dr|Cr))?)$", re.IGNORECASE)


def _rows_from_pdf(data: bytes) -> list:
    import pdfplumber
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception:
        raise StatementProblem("I could not open that PDF. If it is password-protected, remove the password first.")
    with pdf:
        if len(pdf.pages) > MAX_PDF_PAGES:
            raise StatementProblem(f"That PDF has {len(pdf.pages)} pages (the limit is {MAX_PDF_PAGES}). "
                                   "Upload a shorter period.")
        table_rows, text_lines = [], []
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    table_rows.append([re.sub(r"\s+", " ", (c or "")).strip() for c in row])
            text_lines += (page.extract_text() or "").splitlines()
    if any(looks_like_header(r) for r in table_rows):
        return table_rows
    # No proper table: read lines that look like  "05/03/2025  UPI-SWIGGY...  450.00  12,300.50"
    rows, header = [], ["Date", "Narration", "Amount", "Balance"]
    for line in text_lines:
        line = line.strip()
        m = _PDF_LINE.match(line)
        if m:
            rows.append([m["date"], m["text"], m["amount"], m["balance"]])
        elif rows and line and not re.match(r"^\d{1,2}[-/. ]", line):
            rows[-1][1] += " " + line                      # a narration that continues on the next line
    return [header] + rows if rows else []


def read_rows(data: bytes, filename: str = "") -> list:
    if not data:
        raise StatementProblem("That file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise StatementProblem(f"That file is larger than {MAX_FILE_MB} MB.")
    name = (filename or "").lower()
    if data.startswith(b"%PDF") or name.endswith(".pdf"):
        return _rows_from_pdf(data)
    if data[:2] == b"PK" or name.endswith((".xlsx", ".xls")):
        return _rows_from_excel(data)
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):                 # only trust UTF-16 when the file says so
        encodings = ("utf-16", "utf-8-sig", "cp1252")
    else:
        encodings = ("utf-8-sig", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            return _rows_from_csv(data.decode(encoding))
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise StatementProblem("I could not read that file as text. Upload a CSV, Excel or PDF statement.")


def parse_statement(data: bytes, filename: str = "") -> Statement:
    rows = read_rows(data, filename)
    header_at = next((i for i, r in enumerate(rows[:60]) if looks_like_header(r)), None)
    if header_at is None:
        raise StatementProblem(
            "I could not find the transaction table. I look for a header row with a Date, a Narration/Description "
            "and Debit/Credit or Amount columns. Export a plain statement from net banking (CSV is best).")
    cols = map_columns(rows[header_at])
    body = [r for r in rows[header_at + 1:] if any(c.strip() for c in r)]
    if len(body) > MAX_ROWS:
        raise StatementProblem(f"That statement has more than {MAX_ROWS} rows. Upload a shorter period.")

    def cell(row, role):
        i = cols.get(role)
        return row[i] if i is not None and i < len(row) else ""

    raw = pd.DataFrame({
        "date_text": [cell(r, "date") for r in body],
        "description": [re.sub(r"\s+", " ", cell(r, "description")).strip() for r in body],
        "debit": [cell(r, "debit") for r in body], "credit": [cell(r, "credit") for r in body],
        "amount": [cell(r, "amount") for r in body], "drcr": [cell(r, "drcr") for r in body],
        "balance": [cell(r, "balance") for r in body],
    })
    raw["date"] = parse_dates(raw["date_text"])
    dropped_rows = int(raw["date"].isna().sum())                  # repeated headers, page totals, notes
    raw = raw[raw["date"].notna()].reset_index(drop=True)
    if raw.empty:
        raise StatementProblem("I found the table but could not read any dates in it.")

    balance = raw["balance"].map(lambda v: parse_number(v)[0])
    amounts, warnings = pd.Series(np.nan, index=raw.index), []
    if "debit" in cols or "credit" in cols:
        out = raw["debit"].map(lambda v: parse_number(v)[0]).fillna(0).abs()
        inc = raw["credit"].map(lambda v: parse_number(v)[0]).fillna(0).abs()
        amounts = inc - out
        if "amount" in cols and not (out.any() or inc.any()):
            amounts = pd.Series(np.nan, index=raw.index)
    if amounts.isna().all() and "amount" in cols:
        parsed = raw["amount"].map(parse_number)
        value = parsed.map(lambda p: p[0])
        flag = parsed.map(lambda p: p[1])
        drcr = raw["drcr"].astype(str).str.strip().str.lower()
        flag = flag.where(flag.notna(), drcr.map(lambda s: "dr" if s.startswith(("dr", "d", "debit", "w")) else
                                                 ("cr" if s.startswith(("cr", "c", "credit")) else None)))
        if flag.notna().any():
            sign = flag.map({"dr": -1, "cr": 1})
            amounts = value.abs() * sign
        elif (value < 0).any():
            amounts = value                                       # the bank already signs its amounts
        elif balance.notna().sum() > 1:
            amounts = _amounts_from_balance(value.abs(), balance, raw["description"],
                                            raw["date"].iloc[0] > raw["date"].iloc[-1])
            warnings.append("This statement has one unsigned Amount column, so I worked out money in and money "
                            "out from the running balance.")
        else:
            raise StatementProblem("I cannot tell money in from money out: there is no Debit/Credit column, no "
                                   "Dr/Cr flag and no running balance. Export a statement that includes one.")
    if amounts.isna().all():
        raise StatementProblem("I could not read any amounts from that statement.")

    df = pd.DataFrame({"date": raw["date"], "description": raw["description"], "amount": amounts.round(2),
                       "balance": balance})
    ignore = df["description"].str.contains(r"opening balance|closing balance|brought forward|carried forward",
                                            case=False, regex=True)
    df = df[~ignore & df["amount"].notna() & (df["amount"] != 0)].reset_index(drop=True)
    dropped_rows += int(ignore.sum())
    if df.empty:
        raise StatementProblem("I found the table but no transactions in it.")

    if df["date"].iloc[0] > df["date"].iloc[-1]:                  # newest-first statements: put oldest first
        df = df.iloc[::-1].reset_index(drop=True)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["direction"] = np.where(df["amount"] < 0, "debit", "credit")

    check = balance_check(df)
    if check["status"] == "mismatch":
        warnings.append("The running balance does not add up for many rows, so some transactions may be missing "
                        "or misread. Check the totals below against your bank.")
    info = {"rows": len(df), "date_from": df["date"].min().date(), "date_to": df["date"].max().date(),
            "money_in": round(float(df.loc[df["amount"] > 0, "amount"].sum()), 2),
            "money_out": round(float(-df.loc[df["amount"] < 0, "amount"].sum()), 2),
            "balance_check": check, "skipped_lines": dropped_rows, "warnings": warnings, "columns": cols}
    return Statement(df=df, info=info)


CREDIT_WORDS = re.compile(r"salary|refund|interest|cashback|reversal|\bcr\b|credited|deposit|received", re.IGNORECASE)


def _amounts_from_balance(amount: pd.Series, balance: pd.Series, descriptions: pd.Series, newest_first: bool):
    """Unsigned amounts + a running balance: a row is money IN if the balance went up since the row before it
    (in time order). The very first row has nothing before it, so we guess from its words."""
    order = list(amount.index)[::-1] if newest_first else list(amount.index)
    out = pd.Series(np.nan, index=amount.index)
    for k, i in enumerate(order):
        if k == 0:
            out.loc[i] = amount.loc[i] if CREDIT_WORDS.search(descriptions.loc[i]) else -amount.loc[i]
            continue
        before, now = balance.loc[order[k - 1]], balance.loc[i]
        if pd.notna(before) and pd.notna(now) and now != before:
            out.loc[i] = amount.loc[i] if now > before else -amount.loc[i]
    return out


def balance_check(df: pd.DataFrame) -> dict:
    """Does previous balance + this amount = this balance? If yes for most rows, we read the statement correctly."""
    if df["balance"].notna().sum() < 3:
        return {"status": "unknown", "matched": 0, "checked": 0}
    prev = df["balance"].shift(1)
    ok = (prev + df["amount"] - df["balance"]).abs() <= 0.51
    valid = prev.notna() & df["balance"].notna()
    checked, matched = int(valid.sum()), int((ok & valid).sum())
    status = "ok" if checked and matched / checked >= 0.9 else "mismatch"
    return {"status": status, "matched": matched, "checked": checked}
