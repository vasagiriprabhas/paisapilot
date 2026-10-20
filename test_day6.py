"""Day 6 tests: the logic behind the page (appcore.py), and a walk through app.py with a stand-in for Streamlit.
The stand-in proves the page code runs without errors in each state. Your browser is the real check.
Run:  python test_day6.py"""
import json
import runpy
import sys
import types
from unittest.mock import MagicMock

import pandas as pd

import appcore as A
import generate_sample as gs
from statement import StatementProblem


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        raise SystemExit(f"\nFAILED: {name}")


# ---------------------------------------------------------------------------------------------------------------
loaded = A.load_sample()
report = A.build_report(loaded)
n = report.numbers
check("sample loads: 416 rows, balance check ok", n["rows"] == 416 and n["balance_check"]["status"] == "ok")
check("report: money in/out match the statement", abs(n["money_out"] - 575155) < 1 and abs(n["money_in"] - 724100) < 1)
check("report: average saving = income - spending", abs(n["avg_saving"] - (n["avg_income"] - n["avg_spending"])) < 1e-6)
check("report: category totals add up to spending (no income)",
      abs(report.categories["total"].sum() - (-report.df[(report.df["amount"] < 0) & (report.df["category"] != "Income")]["amount"].sum())) < 1)
check("report: shares add to about 100%", abs(report.categories["share_pct"].sum() - 100) < 0.5)
check("report: findings are all there", len(report.recurring) == 10 and len(report.duplicates) == 1 and len(report.unusual) == 1
      and report.forecast.total > 0)

# corrections
fixed = A.build_report(loaded, {"apple store": "Education"})
apple = fixed.df[fixed.df["merchant"] == "APPLE STORE"]
check("a correction changes the category everywhere", (apple["category"] == "Education").all() and (apple["cat_source"] == "you").all())
check("the correction changes the totals",
      fixed.categories.set_index("category").loc["Education", "total"] > report.categories.set_index("category").loc["Education", "total"])
c = A.with_correction({}, " swiggy ", "Groceries")
check("corrections: key is tidied to upper case", c == {"SWIGGY": "Groceries"})
try:
    A.with_correction({}, "X", "Not a category")
    check("corrections: unknown category is refused", False)
except ValueError:
    check("corrections: unknown category is refused", True)
data = A.corrections_to_bytes({"SWIGGY": "Food", "ZEPTO": "Groceries"})
check("corrections: save and load round trip", A.corrections_from_bytes(data) == {"SWIGGY": "Food", "ZEPTO": "Groceries"})
check("corrections: unknown categories and junk values are dropped",
      A.corrections_from_bytes(json.dumps({"A": "Food", "B": "Nonsense", "C": 5, "": "Food"}).encode()) == {"A": "Food"})
for bad in (b"not json", b"[1, 2]", b"\xff\xfe\x00"):
    try:
        A.corrections_from_bytes(bad)
        check("corrections: bad file refused", False)
    except ValueError:
        pass
check("corrections: bad files are refused", True)
try:
    A.corrections_from_bytes(b"{" + b" " * (300 * 1024) + b"}")
    check("corrections: oversized file refused", False)
except ValueError:
    check("corrections: oversized file refused", True)

# review list: something unknown shows up
odd = pd.DataFrame({"Date": ["01/01/2026", "02/01/2026", "03/01/2026"] * 6,
                    "Narration": ["UPI/DR/123456789012/ZXQ VENTURES/ybl/zxq@ybl/Payment", "NEFT-CR-ACME LTD-SALARY",
                                  "UPI/DR/223456789012/SWIGGY/yesb/sw@yesb/UPI"] * 6,
                    "Debit": [500, "", 300] * 6, "Credit": ["", 50000, ""] * 6, "Balance": [""] * 18})
odd["Balance"] = None
odd_loaded = A.load_statement(odd.to_csv(index=False).encode(), "odd.csv")
odd_report = A.build_report(odd_loaded)
check("review list: an unknown merchant is listed first", len(odd_report.to_review) >= 1 and odd_report.to_review.iloc[0]["merchant"] == "ZXQ VENTURES")
check("review list: a statement without balance is handled", odd_report.numbers["balance_check"]["status"] == "unknown")
check("review list: shrinks after a correction",
      "ZXQ VENTURES" not in list(A.build_report(odd_loaded, {"ZXQ VENTURES": "Shopping"}).to_review["merchant"]))

# views
check("category breakdown: one month is smaller than all", A.category_breakdown(report.df, "2026-03")["total"].sum()
      < A.category_breakdown(report.df)["total"].sum())
view = A.transaction_view(report.df, "Food", "swiggy")
check("transaction view: filters by category and text", len(view) > 0 and (view["category"] == "Food").all()
      and view["merchant"].str.contains("SWIGGY").all())
check("transaction view: newest first", list(view["date"]) == sorted(view["date"], reverse=True))
check("transaction view: no match gives an empty table", len(A.transaction_view(report.df, "", "zzzzzz")) == 0)

# bad files
for name, data in (("empty.csv", b""), ("junk.csv", b"\x00\x01\x02garbage"), ("x.pdf", b"not a pdf")):
    try:
        A.load_statement(data, name)
        check(f"bad file {name} is refused", False)
    except StatementProblem as e:
        check(f"bad file {name} gives a plain message", len(str(e)) > 10)

# AI limits
day = {"d": "2026-01-01"}
guard = A.DailyGuard(limit=2, today=lambda: day["d"])
b1, b2 = A.Budget(limit=5, shared=guard), A.Budget(limit=5, shared=guard)
check("daily guard: shared across visitors", b1.take() and b2.take() and not b1.take() and b1.used == 1)
day["d"] = "2026-01-02"
check("daily guard: resets the next day", b1.take())
solo = A.Budget(limit=2)
check("visit budget: stops at the limit", solo.take() and solo.take() and not solo.take() and solo.remaining == 0)


# ---------------------------------------------------------------------------------------------------------------
# Walk through app.py with a pretend Streamlit
# ---------------------------------------------------------------------------------------------------------------
class Rerun(Exception):
    pass


class State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Upload:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


class Col(MagicMock):
    """A column or tab: widgets drawn inside it behave like the page-level ones."""
    page = None

    def selectbox(self, *a, **kw):
        return self.page.selectbox(*a, **kw)

    def text_input(self, *a, **kw):
        return self.page.text_input(*a, **kw)

    def button(self, *a, **kw):
        return self.page.button(*a, **kw)


def fake_streamlit(state, press=(), uploads=None, typed="", form_press=()):
    st = MagicMock(name="streamlit")
    st.session_state = state
    st.cache_resource = lambda f: f
    st.stop = MagicMock(side_effect=SystemExit("stopped"))
    st.rerun = MagicMock(side_effect=Rerun())
    st.button = lambda label, **kw: label in press
    st.form_submit_button = lambda label, **kw: label in form_press
    st.file_uploader = lambda label, **kw: (uploads or {}).get(kw.get("key", "main"))
    st.selectbox = lambda label, options, index=0, **kw: options[index] if len(options) else None
    st.text_input = lambda label, **kw: typed
    st.checkbox = lambda label, value=False, **kw: value
    Col.page = st
    st.tabs = lambda labels: [Col() for _ in labels]
    st.columns = lambda spec: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    return st


def run_app(state, **kw):
    """Runs app.py from the top. A rerun request simply runs it again, like the real thing."""
    errors = []
    for _ in range(4):
        sys.modules["streamlit"] = fake_streamlit(state, **kw)
        try:
            runpy.run_path("app.py")
            return sys.modules["streamlit"]
        except Rerun:
            kw["press"] = ()
            kw["form_press"] = ()
            continue
        except SystemExit as e:
            if str(e) == "stopped":
                return sys.modules["streamlit"]
            raise
    raise RuntimeError("too many reruns")


S0 = State()
st = run_app(S0)
check("app: first visit shows the landing page and stops", st.stop.called and S0["loaded"] is None)

S1 = State()
st = run_app(S1, press=("Try with a sample statement",))
check("app: sample loads and the full page draws", S1["loaded"] is not None and not st.stop.called and st.tabs is not None)
check("app: balance success message shown", any("running balance adds up" in str(c) for c in st.success.call_args_list))
check("app: sample warning shown", any("fictional" in str(c) for c in st.warning.call_args_list))

st = run_app(S1, form_press=("Apply",))
check("app: applying a correction stores it and rebuilds the report", S1["corrections"] != {} and S1["report"] is not None)

S2 = State()
run_app(S2, press=("Try with a sample statement",))
run_app(S2, typed="Was I charged twice?", form_press=("Ask",))
check("app: asking a question stores the answer", len(S2["history"]) == 1 and "AMAZON PAY INDIA" in S2["history"][0][1].text)

S3 = State()
st = run_app(S3, uploads={"main": Upload("bad.csv", b"garbage")})
check("app: a bad upload shows an error and keeps going", S3["problem"] != "" and st.error.called and st.stop.called)

csv = gs.generate()[0].to_csv(index=False).encode()
S4 = State()
run_app(S4, uploads={"main": Upload("my.csv", csv)})
check("app: a real upload loads", S4["loaded"] is not None and S4["loaded"].source == "my.csv")

S5 = State()
run_app(S5, uploads={"corr_upload": Upload("c.json", json.dumps({"SWIGGY": "Groceries"}).encode())},
        press=("Try with a sample statement",))
check("app: saved corrections are loaded", S5["corrections"].get("SWIGGY") == "Groceries")

S6 = State()
st = run_app(S6, press=("Try with a sample statement",))
run_app(S6, press=("Clear my data",))
check("app: 'Clear my data' wipes the session", S6["loaded"] is None)

print("\nALL TESTS PASSED")
