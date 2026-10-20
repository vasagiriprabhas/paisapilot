# Paisapilot: personal finance copilot

Upload a bank or UPI statement. Paisapilot sorts every transaction into a category, finds subscriptions, duplicate
charges and unusual spending, forecasts the month, and answers questions about your money.
Your statement is read in memory and is never stored. Public demos use fictional data only.

## Progress
- [x] Day 1: Statement reader (CSV, Excel, PDF), merchant cleaner, fictional sample data
- [x] Day 2: Categoriser (rules + ML, accuracy measured on held-out data)
- [x] Day 3: Insights (subscriptions, duplicates, anomalies with reasons)
- [x] Day 4: Spending forecast
- [x] Day 5: "Ask your money" AI layer (code computes the numbers, AI only explains them)
- [x] Day 6: Streamlit app
- [ ] Day 7: Testing, deployment, README with screenshots

## Setup
```
pip install -r requirements.txt
python test_day1.py        # every line should say ok
python day1.py sample      # make and read a fictional statement
python test_day2.py        # categoriser tests
python day2.py             # accuracy report on the sample
python test_day3.py        # insights tests
python day3.py             # subscriptions, double charges, unusual payments
python test_day4.py        # forecast tests
python day4.py             # next month's spending + honest backtest
python test_day5.py        # facts, answer checking, AI flow (pretend AI, no key needed)
python day5.py --no-ai     # ask questions, built-in answers
python day5.py             # same, with AI if GEMINI_API_KEY or GROQ_API_KEY is set
python test_day6.py        # page logic + a walk through app.py
streamlit run app.py       # the web app
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

## Ask your money (Day 5): how the AI is kept honest

1. **Code computes every number** (`facts.py`): totals, months, categories, subscriptions, forecast.
2. **The AI only writes the answer** from those facts (`ask.py`). It is told not to calculate anything.
3. **Code checks the answer**: every amount and percentage must exist in the facts. If not, it is asked once to rewrite; if it still fails, the built-in answer is shown instead.
4. **No AI key, no internet, quota used up?** The built-in rule-based answer still works.
5. **Privacy:** only computed totals are sent, with names of people, landlords and employers hidden. Raw bank lines, reference numbers and balances are never sent. The question you type is sent as written, so do not type personal details into it.
