# Paisapilot: personal finance copilot

Upload a bank or UPI statement. Paisapilot sorts every transaction into a category, finds subscriptions, duplicate
charges and unusual spending, forecasts the month, and answers questions about your money.
Your statement is read in memory and is never stored. Public demos use fictional data only.

## Progress
- [x] Day 1: Statement reader (CSV, Excel, PDF), merchant cleaner, fictional sample data
- [ ] Day 2: Categoriser (rules + ML, accuracy measured on held-out data)
- [ ] Day 3: Insights (subscriptions, duplicates, anomalies with reasons)
- [ ] Day 4: Spending forecast
- [ ] Day 5: "Ask your money" AI layer (code computes the numbers, AI only explains them)
- [ ] Day 6: Streamlit app
- [ ] Day 7: Testing, deployment, README with screenshots

## Setup
```
pip install -r requirements.txt
python test_day1.py        # every line should say ok
python day1.py sample      # make and read a fictional statement
python day1.py my_statement.csv     # your own file (CSV, Excel or PDF)
```

## What Day 1 does
- `statement.py`: finds the header row, recognises columns by name (Date, Narration/Particulars, Debit/Withdrawal,
  Credit/Deposit, Amount, Dr/Cr, Balance), reads Indian-style numbers (1,23,456.78), several date styles, and
  statements that are newest-first. For one unsigned Amount column it works out money in/out from the running balance.
  A balance check tells you whether every row was read correctly.
- `merchants.py`: turns `UPI/DR/412345678901/SWIGGY/yesb/swiggy@yesb/UPI` into `SWIGGY`, and notes the channel (UPI, card, ATM, NEFT...).
- `generate_sample.py`: a realistic fictional year of transactions, plus an answer key of categories for measuring accuracy on Day 2.

## Privacy
Never commit a real statement. `.gitignore` blocks PDFs and files named `my_statement*` / `real_*`.
Test on your own statement privately; use the fictional sample for screenshots, demos and README images.
