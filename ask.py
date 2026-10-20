"""Day 5, step 2: "Ask your money". Code finds the numbers. The AI only puts them into words. Then code checks the words.

    question --> build_facts()  (code: every number comes from your statement)
             --> AI writes a short answer using ONLY those facts
             --> verify: every amount and percentage in the answer must exist in the facts
             --> ok: show it (marked "AI, numbers checked")
                 not ok: ask once more; still not ok: show the built-in answer instead

The built-in (rule-based) answer works with no AI key and no internet, and its numbers are right by construction.
Nothing is sent to an AI service except the computed facts, with people's names hidden: never raw bank lines.
"""
import re
from dataclasses import dataclass, field

import pandas as pd

import facts as Fa
import insights as I

rs = I.rs
YEAR_LIKE = range(1990, 2101)
SYSTEM = ("You are Paisapilot, a careful personal finance assistant. You are given FACTS computed from the user's "
          "bank statement. Answer the QUESTION using only those facts.\n"
          "Rules: (1) Quote amounts exactly as they appear in the facts, as full numbers like Rs 11,597. Never write "
          "11.6k or 'about 12 thousand'. (2) Never calculate a new number: no sums, differences, averages or "
          "percentages of your own. If the answer needs a number that is not in the facts, say you do not have it. "
          "(3) Be brief: at most 5 short sentences, plain language, no markdown tables. (4) If the facts cannot "
          "answer the question, say so. (5) Never give investment, tax or legal advice.")


@dataclass
class Answer:
    text: str
    source: str                    # "ai" or "rules"
    verified: bool = True
    note: str = ""                 # why the built-in answer was used, or what was fixed
    evidence: str = ""             # the built-in answer's numbers, always available to check an AI answer against
    unverified: list = field(default_factory=list)


class AiBudget:
    """A cap on AI calls per session, so a free key cannot be used up by accident."""

    def __init__(self, max_calls: int = 20):
        self.max_calls, self.used = max_calls, 0

    def take(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True


# ---------------------------------------------------------------------------------------------------------------
# Sending the AI only what the question needs (free API plans reject long messages)
# ---------------------------------------------------------------------------------------------------------------
AI_FACTS_LIMIT = 3500                      # characters of facts sent to the AI
TOPICS = [                                 # (facts section, words in the question that make it relevant)
    ("repeating_payments", r"subscri|recurring|regular|repeat|netflix|spotify|prime|bill|emi|rent|stopped|cancel|price"),
    ("subscriptions_cost", r"subscri|recurring|regular|repeat|netflix|spotify|prime|ott|streaming|cancel"),
    ("possible_double_charges", r"twice|double|duplicate|charged|refund|mistake|wrong"),
    ("unusual_payments", r"unusual|large|big|expens|strange|suspicious|spike|waste|why|high|anomal"),
    ("largest_payments", r"biggest|largest|highest|big|expens|waste|single|why|high"),
    ("forecast", r"next|forecast|predict|expect|future|will i|going to|budget|afford"),
    ("top_merchants", r"where|top|most|which|merchant|shop|waste|spend"),
    ("average_month_by_category", r"typical|average|usual|normal|category|categories|waste|breakdown|where|most|compare"),
    ("spending_by_category_and_month", r"compare|each month|every month|by month|trend|over time|month by month"),
]


def facts_for_ai(facts: dict, question: str) -> dict:
    """The sections of the facts this question needs. Always: overview, months, and the focus on whatever was mentioned."""
    q = question.lower()
    out = {"overview": facts["overview"],
           "months": [{k: m[k] for k in ("month", "money_in", "money_out", "saved")} for m in facts["months"]]}
    if facts.get("focus"):
        out["focus"] = facts["focus"]
    chosen = [name for name, pattern in TOPICS if name in facts and re.search(pattern, q)]
    if not chosen and not facts.get("focus"):
        chosen = ["average_month_by_category", "top_merchants", "largest_payments"]
    for name in chosen:
        out[name] = facts[name]
    # still too long? drop the bulkiest optional parts first
    for name in ("spending_by_category_and_month", "repeating_payments", "top_merchants", "average_month_by_category"):
        if len(compact(out)) <= AI_FACTS_LIMIT:
            break
        if name == "repeating_payments" and name in out:
            out[name] = out[name][:6]
        else:
            out.pop(name, None)
    return out


def compact(obj) -> str:
    return __import__("json").dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------------------------
# Checking the AI's numbers
# ---------------------------------------------------------------------------------------------------------------
_MONEY = re.compile(r"(?:Rs\.?|INR|₹)\s*(\d[\d,]*(?:\.\d+)?)(?:\s*(k|thousand|lakhs?|lacs?|crores?|cr|m|million)\b)?", re.IGNORECASE)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_TIMES = re.compile(r"(\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE)
_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_BARE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d{4,})(?![\w])")


def numbers_to_check(answer: str) -> list:
    """(text as written, value, kind) for every amount, percentage and multiple in an answer."""
    text = _DATE.sub(" ", answer)
    found = []
    for m in _MONEY.finditer(text):
        value = float(m.group(1).replace(",", ""))
        found.append((m.group(0).strip(), None if m.group(2) else value, "money"))      # "11.6k" can never be checked
    text = _MONEY.sub(" ", text)
    found += [(m.group(0).strip(), float(m.group(1)), "percent") for m in _PERCENT.finditer(text)]
    found += [(m.group(0).strip(), float(m.group(1)), "times") for m in _TIMES.finditer(text)]
    text = _PERCENT.sub(" ", _TIMES.sub(" ", text))
    for m in _BARE.finditer(text):
        value = float(m.group(1).replace(",", ""))
        if "," not in m.group(1) and int(value) in YEAR_LIKE:
            continue                                            # 2026 is a year, not an amount
        found.append((m.group(1), value, "money"))
    return found


def unverified_numbers(answer: str, facts: dict, question: str = "") -> list:
    """Numbers in the answer that are NOT in the facts (or the question). Empty list = every number checks out."""
    known = Fa.numbers_in(facts) + [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", question)
                                    if x.replace(",", "").replace(".", "").isdigit()]
    known.append(0.0)                                   # "you spent Rs 0 there" is always true to say
    bad = []
    for written, value, kind in numbers_to_check(answer):
        if value is None:
            bad.append(written)
            continue
        tolerance = 0.06 if kind in ("percent", "times") else max(0.51, value * 0.002)
        if not any(abs(value - k) <= tolerance for k in known):
            bad.append(written)
    return bad


# ---------------------------------------------------------------------------------------------------------------
# The built-in answer (no AI)
# ---------------------------------------------------------------------------------------------------------------
HELP = ("I can answer questions like:\n"
        "  - How much did I spend on food in March?\n"
        "  - Why was April so high?\n"
        "  - What subscriptions do I pay for?\n"
        "  - Was I charged twice for anything?\n"
        "  - What was my biggest payment?\n"
        "  - How much did I save?  /  What will I spend next month?")


def label(month: str) -> str:
    """'2026-04' -> 'April 2026'"""
    return pd.Period(month).strftime("%B %Y")


def _has(q: str, *words) -> bool:
    return any(re.search(rf"\b{w}", q) for w in words)


def rule_answer(question: str, f: dict) -> str:
    q, focus, ov = question.lower(), f["focus"], f["overview"]
    month, cat, mer = focus.get("month"), focus.get("category"), focus.get("merchant")

    if _has(q, "waste", "wasteful", "unnecessary", "overspend", "overspending", "cut down", "reduce spending"):
        avg = f["average_month_by_category"]
        wants = [(c, avg[c]) for c in ("Shopping", "Food", "Entertainment", "Subscriptions", "Travel") if c in avg]
        text = ("I cannot tell what is wasteful for you, because that depends on your choices. What I can show is where the "
                "flexible money goes in a typical month: " + ", ".join(f"{c} {rs(v)}" for c, v in wants) + ".")
        if month and month.get("chosen_as_highest_spending_month"):
            text += f" Your highest-spending month was {label(month['month'])} ({rs(month['money_out'])} out)."
        if f["unusual_payments"]:
            u = f["unusual_payments"][0]
            text += f" One payment stands out: {u['merchant']} {rs(u['amount'])} on {u['date']}, far above your usual {u['category']} payment."
        return text
    if month and month.get("chosen_as_highest_spending_month"):
        up = month["biggest_increases_vs_typical"]
        text = (f"Your highest-spending month was {label(month['month'])}: {rs(month['money_out'])} went out, against a typical "
                f"{rs(month['typical_month_spending_other_months'])}.")
        if up:
            text += f" The biggest reason was {up[0]['category']} ({rs(up[0]['this_month'])} against a typical {rs(up[0]['typical'])})."
        if month["biggest_payments"]:
            b = month["biggest_payments"][0]
            text += f" Its largest payment was {rs(b['amount'])} to {b['merchant']} on {b['date']}."
        return text
    if _has(q, "duplicate", "double", "twice", "charged two"):
        items = f["possible_double_charges"]
        if not items:
            return "I found no payments where the same shop charged the same amount twice on the same day."
        return "\n".join(f"{i['merchant']} charged {rs(i['amount'])} {i['times']} times on {i['date']} ({i['likelihood']} a double charge). "
                         f"If it was a mistake, {rs(i['extra_paid'])} may be refundable." for i in items)
    if _has(q, "subscription", "recurring", "repeating", "netflix", "spotify", "regular payment") and not month:
        subs = [r for r in f["repeating_payments"] if r["kind"] == "Subscription"]
        if not subs:
            return "I found no subscriptions (a payment repeating monthly for at least 3 months)."
        lines = [f"{r['merchant']}: {rs(r['typical_amount'])} a month" + (f" ({r['status']})" if r["status"] != "active" else "") for r in subs]
        c = f["subscriptions_cost"]
        return (f"You have {c['count']} subscriptions costing {rs(c['per_month'])} a month, {rs(c['per_year'])} a year.\n"
                + "\n".join(lines))
    if _has(q, "next month", "forecast", "predict", "expect", "will i spend", "going to spend"):
        fc = f.get("forecast")
        if not fc:
            return "I need at least 2 full months of spending to forecast."
        return (f"I expect about {rs(fc['expected_spending'])} of spending in {fc['month']} (likely between {rs(fc['likely_low'])} "
                f"and {rs(fc['likely_high'])}), using the {fc['method']}. With income near {rs(fc['expected_income'])}, "
                f"that leaves about {rs(fc['expected_saving'])}. Confidence: {fc['confidence']}.")
    if month and _has(q, "why", "high", "more", "increase", "spike", "jump", "so much", "unusual"):
        text = (f"In {label(month['month'])} you spent {rs(month['money_out'])}, which is {rs(abs(month['more_than_typical_by']))} "
                f"{'more' if month['more_than_typical_by'] >= 0 else 'less'} than a typical month ({rs(month['typical_month_spending_other_months'])}).")
        up = month["biggest_increases_vs_typical"]
        if up:
            text += f" The biggest jump was {up[0]['category']}: {rs(up[0]['this_month'])} against a typical {rs(up[0]['typical'])}."
        if month["biggest_payments"]:
            b = month["biggest_payments"][0]
            text += f" Your largest single payment was {rs(b['amount'])} to {b['merchant']} on {b['date']}."
        return text
    if mer and month:
        value = mer["by_month"].get(month["month"], 0)
        return f"You spent {rs(value)} at {mer['merchant']} in {label(month['month'])} (whole statement: {rs(mer['total'])} over {mer['payments']} payments)."
    if mer:
        return (f"{mer['merchant']}: {rs(mer['total'])} over {mer['payments']} payments (average {rs(mer['average_payment'])}, "
                f"biggest {rs(mer['biggest_payment'])}), from {mer['first']} to {mer['last']}.")
    if cat and month:
        value = cat["by_month"].get(month["month"], 0)
        return f"You spent {rs(value)} on {cat['category']} in {label(month['month'])} (average month: {rs(cat['average_per_month'])})."
    if cat:
        tops = ", ".join(f"{m['merchant']} {rs(m['total'])}" for m in cat["top_merchants"][:3])
        return (f"{cat['category']}: {rs(cat['total'])} in total, about {rs(cat['average_per_month'])} a month "
                f"({cat['share_of_all_spending_pct']}% of your spending). Most of it went to {tops}.")
    if month:
        return (f"In {label(month['month'])} you spent {rs(month['money_out'])} and received {rs(month['money_in'])}. "
                f"A typical month is {rs(month['typical_month_spending_other_months'])} of spending.")
    if _has(q, "unusual", "biggest", "largest", "big", "expensive", "anomal", "strange", "suspicious", "highest"):
        un, big = f["unusual_payments"], f["largest_payments"]
        text = "Largest payments: " + "; ".join(f"{b['merchant']} {rs(b['amount'])} on {b['date']}" for b in big[:3]) + "."
        if un:
            text += " Unusually large for their category: " + "; ".join(
                f"{u['merchant']} {rs(u['amount'])} (typical {u['category']} payment {rs(u['typical_for_category'])})" for u in un[:3]) + "."
        return text
    if _has(q, "save", "saving", "saved", "savings", "left over"):
        return (f"On average you save {rs(ov['average_monthly_saving'])} a month "
                f"(income {rs(ov['average_monthly_income'])}, spending {rs(ov['average_monthly_spending'])}).")
    if _has(q, "income", "salary", "earn"):
        return f"Your average monthly income is {rs(ov['average_monthly_income'])}; total received was {rs(ov['total_money_in'])}."
    if _has(q, "where", "most", "top", "which"):
        return "Top places by spending: " + "; ".join(f"{m['merchant']} {rs(m['total'])}" for m in f["top_merchants"][:5]) + "."
    if _has(q, "total", "spend", "spent", "spending", "how much"):
        top = list(f["average_month_by_category"].items())[:3]
        return (f"From {ov['statement_from']} to {ov['statement_to']} you spent {rs(ov['total_money_out'])}, about "
                f"{rs(ov['average_monthly_spending'])} a month. Biggest categories in a typical month: "
                + ", ".join(f"{c} {rs(v)}" for c, v in top) + ".")
    return HELP


# ---------------------------------------------------------------------------------------------------------------
# The full flow
# ---------------------------------------------------------------------------------------------------------------
def ask_money(question: str, df, ai: bool = True, llm=None, budget: AiBudget = None) -> Answer:
    llm = llm or __import__("llm")
    if df is None or len(df) == 0:
        return Answer("There are no transactions to answer from.", "rules")
    local = Fa.build_facts(df, question)
    built_in = rule_answer(question, local)
    base = Answer(built_in, "rules", True, evidence=built_in)
    if not ai:
        return base
    if not llm.provider():
        base.note = "No AI key set, so this is the built-in answer."
        return base

    shared = Fa.build_facts(df, question, hide_names=True)
    prompt = f"QUESTION: {question}\n\nFACTS (JSON):\n{compact(facts_for_ai(shared, question))}"
    bad = []
    for attempt in range(2):
        if budget is not None and not budget.take():
            base.note = "AI limit for this session reached, so this is the built-in answer."
            return base
        try:
            text = llm.ask(prompt if not bad else prompt + "\n\nYour previous answer used numbers that are not in the facts: "
                           + ", ".join(bad) + ". Rewrite it using only numbers copied from the facts.", system=SYSTEM).strip()
        except llm.LLMUnavailable as error:
            base.note = f"AI unavailable ({error}) so this is the built-in answer."
            return base
        bad = unverified_numbers(text, shared, question)
        if not bad:
            return Answer(text, "ai", True, note="" if attempt == 0 else "Rewritten once after a number did not check out.",
                          evidence=built_in)
    base.note = "The AI's answer used numbers I could not verify (" + ", ".join(bad) + "), so this is the built-in answer."
    base.unverified = bad
    return base
