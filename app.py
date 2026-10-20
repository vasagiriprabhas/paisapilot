"""Paisapilot web app (Day 6).   Run:  streamlit run app.py

Upload a bank statement -> see where the money goes, repeating payments, double charges, next month's forecast,
and ask questions in plain English. All real work lives in appcore.py, ask.py and the modules before it (tested
without a browser). This file only draws the page.
"""
import streamlit as st

import appcore as A
import ask
from llm import provider
from statement import StatementProblem

st.set_page_config(page_title="Paisapilot", page_icon="🧭", layout="wide")

CHART_NOTE = "Only money that went out is counted as spending. Salary and other money in is shown separately."
EXAMPLES = ["What subscriptions do I pay?", "Was I charged twice for anything?", "What will I spend next month?"]


def rs(x) -> str:
    return f"Rs {x:,.0f}"


# ------------------------------------------------------------------ state
@st.cache_resource
def daily_guard():
    """One counter shared by every visitor of this server."""
    return A.DailyGuard()


S = st.session_state
S.setdefault("loaded", None)          # the parsed statement
S.setdefault("corrections", {})       # {"MERCHANT": "Category"} that the person set
S.setdefault("report", None)          # computed once, thrown away whenever the data or corrections change
S.setdefault("history", [])           # questions asked and their answers
S.setdefault("budget", A.Budget(shared=daily_guard()))
S.setdefault("file_sig", None)
S.setdefault("corr_sig", None)
S.setdefault("problem", "")


def new_data(loaded):
    S.loaded, S.report, S.history, S.problem = loaded, None, [], ""


def get_report() -> A.Report:
    if S.report is None:
        S.report = A.build_report(S.loaded, S.corrections)
    return S.report


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Paisapilot")
    st.caption("Your bank statement, explained.")
    upload = st.file_uploader("Your statement (CSV, Excel or PDF)", type=["csv", "xlsx", "xls", "pdf"])
    if upload is not None:
        sig = (upload.name, upload.size)
        if sig != S.file_sig:
            S.file_sig = sig
            try:
                new_data(A.load_statement(upload.getvalue(), upload.name))
            except StatementProblem as problem:
                S.problem = str(problem)
            except Exception:
                S.problem = "I could not read that file. Export a plain statement from net banking (CSV is best)."
    if st.button("Try with a sample statement"):
        new_data(A.load_sample())
        S.file_sig = None
        st.rerun()
    if S.problem:
        st.error(S.problem)

    saved = st.file_uploader("Load my saved corrections (optional)", type=["json"], key="corr_upload")
    if saved is not None and (saved.name, saved.size) != S.corr_sig:
        S.corr_sig = (saved.name, saved.size)
        try:
            S.corrections = {**S.corrections, **A.corrections_from_bytes(saved.getvalue())}
            S.report = None
            st.success("Corrections loaded.")
        except ValueError as problem:
            st.error(str(problem))

    st.divider()
    ai_ready = bool(provider())
    use_ai = st.checkbox("Use AI to word answers", value=ai_ready, disabled=not ai_ready,
                         help="Numbers always come from your statement. The AI only words the answer, and every number it "
                              "writes is checked.")
    if ai_ready:
        st.caption(f"AI answers left this visit: {S.budget.remaining}")
    else:
        st.caption("No AI key on this server, so answers are the built-in kind.")

    with st.expander("Privacy: is my statement stored?"):
        st.write("No. Your file is read in memory during this visit and is not saved. When AI is on, only computed totals "
                 "are sent, with names of people, landlords and employers hidden. Raw bank lines, reference numbers and "
                 "balances are never sent. Close the tab or press the button below to wipe everything.")
    if st.button("Clear my data"):
        for key in list(S.keys()):
            del S[key]
        st.rerun()

# ------------------------------------------------------------------ landing page
st.title("Paisapilot")
if S.loaded is None:
    st.subheader("Understand your money from one bank statement")
    st.markdown(
        "- **See where it goes:** categories, months, biggest payments\n"
        "- **Catch problems:** forgotten subscriptions, double charges, unusually large payments\n"
        "- **Look ahead:** next month's spending, with an honest margin of error\n"
        "- **Ask in plain English:** every number in an answer is checked against your statement")
    st.info("Upload a statement in the sidebar, or press **Try with a sample statement** (fictional data) to look around.")
    st.caption("How to get a statement: in net banking, open Account Statement, choose a period of 6 to 12 months, and "
               "download CSV or Excel. PDF works too, but CSV is the most reliable.")
    st.stop()

report = get_report()
n = report.numbers
df = report.df

if S.loaded.source == "sample":
    st.warning("You are looking at a fictional sample statement.")
check = n["balance_check"]
if check["status"] == "ok":
    st.success(f"Read {n['rows']} transactions from {n['date_from']} to {n['date_to']}. "
               f"The running balance adds up ({check['matched']} of {check['checked']} rows), so nothing was misread.")
elif check["status"] == "mismatch":
    st.error(f"Read {n['rows']} transactions, but the running balance does not add up on "
             f"{check['checked'] - check['matched']} rows. Some amounts may have been misread, so check the numbers.")
else:
    st.info(f"Read {n['rows']} transactions from {n['date_from']} to {n['date_to']}. "
            "This statement has no running balance, so I could not double-check the totals.")
for warning in n["warnings"]:
    st.caption(f"Note: {warning}")

tab_overview, tab_tx, tab_insights, tab_forecast, tab_ask = st.tabs(
    ["Overview", "Transactions", "Insights", "Forecast", "Ask your money"])

# ------------------------------------------------------------------ overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Money in", rs(n["money_in"]))
    c2.metric("Money out", rs(n["money_out"]))
    c3.metric("Average monthly spending", rs(n["avg_spending"]))
    c4.metric("Average monthly saving", rs(n["avg_saving"]),
              None if n["saving_rate_pct"] is None else f"{n['saving_rate_pct']}% of income", delta_color="off")
    st.subheader("Month by month")
    st.bar_chart(report.monthly.set_index("month")[["money_out"]].rename(columns={"money_out": "Spending"}))
    st.caption("Each bar is one month of spending (money out). A tall bar usually means one big purchase: "
               "check the Insights tab.")
    st.subheader("Where the money went")
    months = ["All months"] + list(report.monthly["month"])
    choice = st.selectbox("Period", months)
    breakdown = A.category_breakdown(df, "" if choice == "All months" else choice)
    if len(breakdown):
        st.bar_chart(breakdown.set_index("category")["total"])
        st.caption(CHART_NOTE)
    else:
        st.info("No spending in that period.")

# ------------------------------------------------------------------ transactions and corrections
with tab_tx:
    review = report.to_review
    if len(review):
        st.warning(f"{len(review)} merchants need a look. I was unsure about these, biggest spending first.")
        st.dataframe(review.rename(columns={"rows": "payments", "total": "total spent"}), hide_index=True)
    else:
        st.success("Every merchant was categorised with confidence.")

    with st.form("fix"):
        st.markdown("**Fix a category.** Pick a merchant and the right category. It applies to all their payments, "
                    "and I use it to categorise similar merchants next time.")
        merchants = list(review["merchant"]) + sorted(set(df["merchant"]) - set(review["merchant"]))
        merchant = st.selectbox("Merchant", merchants)
        current = df.loc[df["merchant"] == merchant, "category"].iloc[0] if merchant else A.CATEGORIES[0]
        category = st.selectbox("Correct category", A.CATEGORIES,
                                index=A.CATEGORIES.index(current) if current in A.CATEGORIES else 0)
        if st.form_submit_button("Apply"):
            S.corrections = A.with_correction(S.corrections, merchant, category)
            S.report = None
            st.rerun()
    if S.corrections:
        st.caption(f"You have made {len(S.corrections)} correction(s). They are lost when you close the tab, "
                   "unless you download them and load them next time.")
        st.download_button("Download my corrections", A.corrections_to_bytes(S.corrections),
                           file_name="paisapilot_corrections.json")

    st.subheader("All transactions")
    f1, f2 = st.columns(2)
    pick = f1.selectbox("Category", ["All"] + A.CATEGORIES)
    text = f2.text_input("Search merchant or description", max_chars=60)
    view = A.transaction_view(df, "" if pick == "All" else pick, text)
    st.caption(f"{len(view)} transactions. Negative amounts are money out. 'Source' says why a category was chosen: "
               "you, rule, model or default.")
    st.dataframe(view, hide_index=True)

# ------------------------------------------------------------------ insights
with tab_insights:
    st.subheader("Repeating payments")
    rec = report.recurring
    if len(rec):
        subs = rec[rec["kind"] == "Subscription"]
        if len(subs):
            a, b = st.columns(2)
            a.metric("Subscriptions per month", rs(subs["monthly_cost"].sum()))
            b.metric("Subscriptions per year", rs(subs["yearly_cost"].sum()))
        st.dataframe(rec[["merchant", "kind", "every", "typical_amount", "times", "status", "note"]], hide_index=True)
        st.caption("A payment counts as repeating when it happens on a steady rhythm for at least 3 months. "
                   "'May have stopped' means it has not appeared for a while: check you cancelled it.")
    else:
        st.info("No repeating payments found. I need a payment on a steady rhythm for at least 3 months.")

    st.subheader("Possible double charges")
    if len(report.duplicates):
        for _, row in report.duplicates.iterrows():
            st.warning(f"{row['reason']} Possible extra payment: {rs(row['extra_paid'])}.")
    else:
        st.success("No payment was charged twice on the same day.")

    st.subheader("Unusually large payments")
    if len(report.unusual):
        for _, row in report.unusual.iterrows():
            st.info(f"{row['date']}: {row['reason']}")
    else:
        st.success("Nothing stood out as unusually large for its category.")

# ------------------------------------------------------------------ forecast
with tab_forecast:
    fc = report.forecast
    if fc.total:
        st.subheader(f"Expected spending in {fc.next_month}")
        a, b, c = st.columns(3)
        a.metric("Expected", rs(fc.total))
        b.metric("Likely range", f"{rs(fc.low)} to {rs(fc.high)}")
        c.metric("Expected saving", rs(fc.expected_saving), f"income about {rs(fc.expected_income)}", delta_color="off")
        st.caption(f"Method: {fc.method}. Based on {fc.months_used} full months. Confidence: {fc.confidence}.")
        for note in fc.notes:
            st.caption(f"Note: {note}")
        st.markdown("**Where it is expected to go**")
        st.bar_chart(fc.by_category.set_index("category")["expected"])
        if len(fc.backtest):
            st.subheader("How good is this method?")
            st.write(f"I tested it on months I already know, predicting each one using only the months before it. "
                     f"It missed by {fc.typical_error_pct:.1f}% on average. The simple rule 'same as last month' missed by "
                     f"{fc.baseline_error_pct:.1f}%.")
            st.line_chart(fc.backtest.set_index("month")[["actual", "predicted"]])
            st.caption("Spending is partly random (a trip, a gift), so no method can be exact. That is why a range is shown.")
    else:
        for note in fc.notes:
            st.info(note)

# ------------------------------------------------------------------ ask
with tab_ask:
    st.subheader("Ask your money")
    st.caption("Examples: 'How much did I spend on food in March?', 'Why was April so high?', 'How much on Swiggy?'")
    top_month = report.monthly.sort_values("money_out").iloc[-1]["month"]
    examples = [f"Why was {ask.label(top_month)} so high?"] + EXAMPLES
    cols = st.columns(len(examples))
    clicked = None
    for col, example in zip(cols, examples):
        if col.button(example, key=f"ex_{example}"):
            clicked = example
    with st.form("ask_form"):
        typed = st.text_input("Your question", max_chars=300)
        submitted = st.form_submit_button("Ask")
    question = clicked or (typed.strip() if submitted and typed.strip() else None)
    if question:
        answer = ask.ask_money(question, df, ai=use_ai and ai_ready, budget=S.budget)
        S.history.insert(0, (question, answer))
    for q, answer in S.history[:8]:
        st.markdown(f"**{q}**")
        if answer.source == "ai":
            st.caption("AI wording. Every number was checked against your statement.")
        else:
            st.caption("Built-in answer (numbers come straight from your statement).")
        st.write(answer.text.replace("\n", "  \n"))          # keep line breaks (one subscription per line)
        if answer.note:
            st.caption(f"Note: {answer.note}")
        if answer.source == "ai":
            with st.expander("Numbers behind this answer"):
                st.text(answer.evidence)
        st.divider()

st.caption("Paisapilot explains your own statement. It is not financial, tax or investment advice.")
