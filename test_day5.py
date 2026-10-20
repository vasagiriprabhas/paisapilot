"""Day 5 tests: facts, answer checking, built-in answers, and the AI flow (with a pretend AI, so no key is needed).
Run:  python test_day5.py"""
import json

import ask
import facts as Fa
import generate_sample as gs
import insights as I
from categoriser import categorise
from merchants import add_merchants
from statement import parse_statement


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        raise SystemExit(f"\nFAILED: {name}")


class FakeLLM:
    """Stands in for the real AI. `replies` are returned one by one; an Exception in the list is raised."""

    class LLMUnavailable(Exception):
        pass

    def __init__(self, replies, key=True):
        self.replies, self.key, self.prompts, self.calls = list(replies), key, [], 0

    def provider(self):
        return "fake" if self.key else ""

    def ask(self, prompt, system=""):
        self.prompts.append(prompt)
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else self.replies_default
        if isinstance(reply, Exception):
            raise reply
        return reply


stmt, labels = gs.generate()
df = categorise(add_merchants(parse_statement(stmt.to_csv(index=False).encode(), "s.csv").df))

# ---- facts come from the data -----------------------------------------------------------------------------
f = Fa.build_facts(df, "")
check("facts: total money out matches the statement", f["overview"]["total_money_out"] == round(-df[df["amount"] < 0]["amount"].sum()))
check("facts: total money in matches the statement", f["overview"]["total_money_in"] == round(df[df["amount"] > 0]["amount"].sum()))
ms = I.monthly_summary(df).set_index("month")
check("facts: every month matches the monthly summary",
      all(m["money_out"] == round(ms.loc[m["month"], "money_out"]) for m in f["months"]))
food_march = Fa.build_facts(df, "food in march")["focus"]["category"]["by_month"]["2026-03"]
manual = -df[(df["category"] == "Food") & (df["date"].dt.strftime("%Y-%m") == "2026-03")]["amount"].sum()
check("facts: Food in March equals a hand calculation", food_march == round(manual))
check("facts: subscriptions cost matches insights", f["subscriptions_cost"]["per_month"] == 1326)

# ---- reading the question ---------------------------------------------------------------------------------
months = ["2025-10", "2026-03", "2026-04", "2026-05", "2026-09"]
check("month: name", Fa.parse_month("spend in March?", months) == "2026-03")
check("month: 'last month' is the latest full month", Fa.parse_month("what about last month", months) == "2026-09")
check("month: year given", Fa.parse_month("october 2025 spending", months) == "2025-10")
check("month: none mentioned", Fa.parse_month("how much did I spend", months) == "")
check("month: 'may I know' is not May, 'spend in May' is",
      Fa.parse_month("may I know my total", months) == "" and Fa.parse_month("what did I spend in May?", months) == "2026-05")
check("category: word", Fa.parse_category("how much on groceries") == "Groceries")
check("category: Swiggy means Food", Fa.parse_category("swiggy spending") == "Food")
check("category: none", Fa.parse_category("what is my balance") == "")
check("merchant: found by name", Fa.parse_merchant("how much on swiggy?", df) == "SWIGGY")
check("merchant: not found", Fa.parse_merchant("how much on nothing", df) == "")

# ---- checking numbers -------------------------------------------------------------------------------------
good = "Your Netflix costs Rs 649 a month and all subscriptions cost Rs 1,326 a month."
check("verify: real numbers pass", ask.unverified_numbers(good, f) == [])
check("verify: invented amount is caught", ask.unverified_numbers("You spent Rs 12,345 on food.", f) == ["Rs 12,345"])
check("verify: '11.6k' style is refused", ask.unverified_numbers("Shopping was about Rs 11.6k.", f) != [])
check("verify: years, dates and small counts are ignored",
      ask.unverified_numbers("In April 2026 (2026-04-19) you made 3 payments over 12 months.", f) == [])
check("verify: invented percentage is caught", ask.unverified_numbers("That is 73% of your spending.", f) == ["73%"])
check("verify: real percentage passes", ask.unverified_numbers("Food was 6.3% of spending.", Fa.build_facts(df, "food")) == [])
check("verify: a number the user typed is allowed", ask.unverified_numbers("Rs 777 is small.", f, "is Rs 777 a lot?") == [])
check("verify: rounding by one rupee is allowed", ask.unverified_numbers("Rs 1,325.6 a month", f) == [])

# ---- built-in answers: every number in them must be a real fact ----------------------------------------
questions = ["How much did I spend on food in March?", "Why was April so high?", "What subscriptions do I pay?",
             "Was I charged twice for anything?", "What was my biggest payment?", "How much did I save?",
             "What will I spend next month?", "how much on swiggy", "swiggy in august", "where did my money go",
             "what is my income", "how was march", "how much did I spend in total", "rent"]
for q in questions:
    a = ask.ask_money(q, df, ai=False)
    local = Fa.build_facts(df, q)
    check(f"built-in: '{q}' only uses real numbers", ask.unverified_numbers(a.text, local, q) == [] and a.source == "rules")
check("built-in: an unknown question shows the help", ask.ask_money("hello there", df, ai=False).text == ask.HELP)
check("built-in: double charge answer is right", "AMAZON PAY INDIA" in ask.ask_money("charged twice?", df, ai=False).text
      and "Rs 1,799" in ask.ask_money("charged twice?", df, ai=False).text)

# ---- the AI flow ------------------------------------------------------------------------------------------
q = "How much do my subscriptions cost?"
ok_llm = FakeLLM(["Your subscriptions cost Rs 1,326 a month, which is Rs 15,912 a year."])
a = ask.ask_money(q, df, llm=ok_llm)
check("ai: a correct answer is shown as AI and verified", a.source == "ai" and a.verified and ok_llm.calls == 1)

fix_llm = FakeLLM(["You pay Rs 9,999 a month.", "Subscriptions cost Rs 1,326 a month."])
a = ask.ask_money(q, df, llm=fix_llm)
check("ai: a wrong number gets one rewrite, then passes", a.source == "ai" and fix_llm.calls == 2 and "Rs 9,999" in fix_llm.prompts[1])

bad_llm = FakeLLM(["You pay Rs 9,999 a month.", "You pay Rs 8,888 a month."])
a = ask.ask_money(q, df, llm=bad_llm)
check("ai: still wrong after rewrite -> built-in answer used", a.source == "rules" and bad_llm.calls == 2
      and "could not verify" in a.note and "Rs 1,326" in a.text and a.unverified == ["Rs 8,888"])

a = ask.ask_money(q, df, llm=FakeLLM([], key=False))
check("ai: no key -> built-in answer and a note", a.source == "rules" and "No AI key" in a.note)

down = FakeLLM([])
down.replies = [FakeLLM.LLMUnavailable("Free quota reached.")]
a = ask.ask_money(q, df, llm=down)
check("ai: API error -> built-in answer, no crash", a.source == "rules" and "Free quota" in a.note)

budget = ask.AiBudget(1)
ask.ask_money(q, df, llm=FakeLLM(["Rs 1,326 a month."]), budget=budget)
a = ask.ask_money(q, df, llm=FakeLLM(["Rs 1,326 a month."]), budget=budget)
check("ai: call budget is enforced", a.source == "rules" and "limit" in a.note)
check("ai: ai=False never calls the AI", FakeLLM(["x"]).calls == 0 and ask.ask_money(q, df, ai=False, llm=FakeLLM(["x"])).source == "rules")
check("ai: empty statement is handled", ask.ask_money(q, df.head(0), llm=FakeLLM([])).source == "rules")

# ---- the AI only gets what it needs (free plans reject long messages) -------------------------------------
for qq in ["which month i have lot of waste expenses", "What subscriptions do I pay?", "Why was April so high?",
           "Was I charged twice?", "what will I spend next month", "compare each month", "hello", "how much on food in March?"]:
    sent_facts = ask.compact(ask.facts_for_ai(Fa.build_facts(df, qq, hide_names=True), qq))
    check(f"ai size: '{qq}' sends under {ask.AI_FACTS_LIMIT + 800} characters", len(sent_facts) <= ask.AI_FACTS_LIMIT + 800)
check("ai size: a subscriptions question includes the subscriptions facts",
      "subscriptions_cost" in ask.facts_for_ai(Fa.build_facts(df, "subscriptions", hide_names=True), "subscriptions"))
check("ai size: a double-charge question includes double charges",
      "possible_double_charges" in ask.facts_for_ai(Fa.build_facts(df, "charged twice", hide_names=True), "charged twice"))
check("ai size: unrelated sections are left out", "forecast" not in ask.facts_for_ai(Fa.build_facts(df, "charged twice", hide_names=True), "charged twice"))

# ---- the question that failed on your laptop -----------------------------------------------------------------
a = ask.ask_money("which month i have lot of waste expenses", df, ai=False)
check("built-in: 'waste' explains what it can and cannot say", "cannot tell what is wasteful" in a.text and "Shopping" in a.text)
a = ask.ask_money("which month did I spend the most?", df, ai=False)
check("built-in: 'which month' gives the highest month", "April 2026" in a.text and "Rs 106,695" in a.text)
check("built-in: highest-month answer uses real numbers",
      ask.unverified_numbers(a.text, Fa.build_facts(df, "which month did I spend the most?"), "") == [])

# ---- privacy: what actually leaves the computer ----------------------------------------------------------
spy = FakeLLM(["Subscriptions cost Rs 1,326 a month."])
ask.ask_money("who did I send money to, where did my money go, and what about rent?", df, llm=spy)
sent = spy.prompts[0]
check("privacy: no personal names are sent", not any(p in sent for p in gs.PEOPLE) and "LANDLORD SURESH" not in sent
      and "TECHNOVA" not in sent)
check("privacy: no raw bank lines, handles, or reference numbers are sent",
      "UPI/" not in sent and "@" not in sent and "NEFT" not in sent and
      not any(r in sent for r in stmt["Ref No"]))
check("privacy: no running balances are sent", "balance" not in json.dumps(json.loads(sent.split("(JSON):\n", 1)[1])).lower())
check("privacy: shops and categories are still sent", "APPLE STORE" in sent and "Food" in sent)

input_copy = df.copy()
ask.ask_money(q, df, ai=False)
check("input table is not modified", df.equals(input_copy))

print("\nALL TESTS PASSED")
