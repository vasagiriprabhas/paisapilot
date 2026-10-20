"""Day 2 report: how good is the categoriser on the sample statement? (We have the answer key for it.)

    python day2.py
"""
import io

import numpy as np
import pandas as pd

import generate_sample as gs
from categoriser import UNKNOWN, categorise, evaluate
from merchants import add_merchants
from statement import parse_statement

stmt, labels = gs.generate()
df = add_merchants(parse_statement(stmt.to_csv(index=False).encode(), "sample.csv").df)
truth = labels["category"]

print("== 1. How much does each layer help? (all 'known' merchants) ==")
for name, kw in [("rules only", dict(use_model=False)), ("model only", dict(use_rules=False)),
                 ("rules + model", {})]:
    r = evaluate(categorise(df, **kw), truth)
    print(f"{name:14s} accuracy {r['accuracy']:.1%}   answered {r['answered']:.1%}   "
          f"correct when it answered {r['accuracy_when_answered']:.1%}")

print("\n== 2. The hard test: merchants the system has never seen ==")
print("Hide each merchant in turn (in 5 groups). The model must guess it from the other merchants only,")
print("with NO keyword rules. This shows how well it learns from corrections.")
merchants = sorted(df["merchant"].unique())
rng = np.random.RandomState(0)
rng.shuffle(merchants)
folds = [merchants[i::5] for i in range(5)]
right = answered = total = 0
for held in folds:
    mask = df["merchant"].isin(held)
    known = df[~mask]
    corrections = {m: truth[known.index[known["merchant"] == m][0]] for m in known["merchant"].unique()}
    pred = categorise(df[mask], corrections=corrections, use_rules=False)
    t = truth[mask].reset_index(drop=True)
    right += int((pred["category"] == t).sum())
    answered += int((pred["category"] != UNKNOWN).sum())
    total += len(t)
print(f"unseen merchants: accuracy {right / total:.1%}   answered {answered / total:.1%}   "
      f"correct when it answered {right / max(answered, 1):.1%}")

print("\n== 3. What the full system still gets wrong ==")
full = categorise(df)
bad = full[full["category"] != truth.reset_index(drop=True)]
print(bad.groupby(["merchant", "category"]).size().rename("rows").reset_index().head(15).to_string(index=False)
      if len(bad) else "nothing")
print("\nNote: the sample statement was written by me, so the rules look better here than they will on a real")
print("statement. The honest number comes from correcting your own statement and counting the corrections.")
