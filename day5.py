"""Day 5: Ask your money.

    python day5.py                       ask questions about the fictional sample statement
    python day5.py my_statement.csv      ask about your own statement
    python day5.py --no-ai               built-in answers only (no AI key needed)

To turn the AI on, set a free key in the same terminal first (never put it in a file you upload):
    Windows (cmd):  set GEMINI_API_KEY=your_key_here        or   set GROQ_API_KEY=your_key_here
Type a blank line to quit.
"""
import sys

import ask
import llm
from categoriser import categorise
from day3 import load
from merchants import add_merchants
from statement import StatementProblem


def main(argv):
    use_ai = "--no-ai" not in argv
    paths = [a for a in argv if not a.startswith("--")]
    df = categorise(add_merchants(load(paths[0] if paths else None).df))
    print(f"Loaded {len(df)} transactions.")
    if use_ai and llm.provider():
        print(f"AI is ON ({llm.provider()}). Only computed totals are sent, with people's names hidden.")
    else:
        print("AI is OFF: using the built-in answers." + ("" if not use_ai else " (set GEMINI_API_KEY or GROQ_API_KEY to turn it on)"))
    print(ask.HELP, "\n")
    budget = ask.AiBudget(20)
    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer = ask.ask_money(question, df, ai=use_ai, budget=budget)
        label = "AI, every number checked against your statement" if answer.source == "ai" else "built-in answer"
        print(f"\n[{label}]\n{answer.text}")
        if answer.note:
            print(f"(note: {answer.note})")
        if answer.source == "ai":
            print("\nNumbers behind this answer, from your statement:\n  " + answer.evidence.replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except StatementProblem as problem:
        sys.exit(f"Problem: {problem}")
