"""Day 1: turn a messy bank narration into the merchant's name.

    UPI/DR/412345678901/SWIGGY/yesb/swiggy@yesb/UPI        ->  SWIGGY
    UPI-ZOMATO-zomato@icici-ICIC0000011-401234567890-UPI   ->  ZOMATO
    POS 412345XXXXXX1234 DOMINOS PIZZA                     ->  DOMINOS PIZZA
    ACH D- HDFC BANK LTD-LOAN EMI-12345678                 ->  HDFC BANK LTD   (text: HDFC BANK LTD LOAN EMI)

Two outputs per row:
    merchant : the first meaningful part, used to group spending ("how much on SWIGGY?")
    text     : everything meaningful joined, used by the categoriser on Day 2 ("LOAN EMI", "SALARY", "RENT")
    channel  : how the money moved (upi, card, atm, neft, imps, ach, cheque, other)
"""
import re

import pandas as pd

NOISE = {"UPI", "DR", "CR", "IMPS", "NEFT", "RTGS", "POS", "ATM", "ACH", "NACH", "ECS", "PAYMENT", "FROM", "TO",
         "PHONEPE", "GPAY", "PAYTM", "BHIM", "WDL", "TXN", "TRANSFER", "PURCHASE", "INB", "MB", "BY", "D", "C", "REF",
         "NO", "ONLINE", "VIA", "THRU", "MOB", "MOBILE", "BANKING", "INT", "PYMT", "TRF", "TFR", "DEBIT", "CREDIT",
         "SENT", "RECEIVED", "OF", "IN", "PAID"}
PSP_HANDLES = {"YESB", "YBL", "OKAXIS", "OKSBI", "OKHDFCBANK", "OKICICI", "IBL", "AXL", "PTYES", "PTSBI", "PTAXIS",
               "PAYTM", "APL", "FBL", "AXISB", "SBI", "HDFCBANK", "ICICI", "KOTAK", "UPI"}
IFSC = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
CARD_MASK = re.compile(r"^\d{4,6}[X*]{2,}\d{2,4}$|^[X*]{4,}\d{3,4}$")
CHANNELS = [("upi", r"^UPI|/UPI|UPI/|@"), ("card", r"^POS\b|\bPOS\b|DEBIT CARD|CARD PURCHASE"), ("atm", r"\bATM\b|\bNWD\b|CASH WDL"),
            ("neft", r"\bNEFT\b|\bRTGS\b"), ("imps", r"\bIMPS\b"), ("ach", r"\bACH\b|\bNACH\b|\bECS\b|AUTO DEBIT|MANDATE"),
            ("cheque", r"\bCHQ\b|\bCHEQUE\b|\bCLG\b")]


def channel_of(description: str) -> str:
    text = str(description).upper()
    for name, pattern in CHANNELS:
        if re.search(pattern, text):
            return name
    return "other"


def _useful(token: str) -> bool:
    t = token.strip(".,:;()[]{}#-")
    if not t or t in NOISE:
        return False
    if re.fullmatch(r"\d+", t) or CARD_MASK.match(t):          # reference numbers and masked card numbers
        return False
    if IFSC.match(t) or re.fullmatch(r"[A-Z0-9]{10,}", t) and re.search(r"\d", t):   # bank codes, long ids
        return False
    return True


def clean_description(description: str) -> dict:
    text = str(description or "").upper()
    channel = channel_of(text)
    if channel == "atm":
        place = " ".join(t for t in re.split(r"\s+", re.sub(r"\d+", " ", text)) if _useful(t) and t != "NWD")
        return {"merchant": "ATM CASH", "text": f"ATM CASH {place}".strip(), "channel": channel}
    segments = []
    for part in re.split(r"[/|]|\s-\s|-(?=[A-Z@])|(?<=[A-Z0-9])-(?=\d)|(?<=\d)-(?=[A-Z])", text):
        part = part.strip()
        if not part or "@" in part or part in PSP_HANDLES:       # payment IDs like swiggy@icici, bank handles like YBL
            continue
        words = [w for w in re.split(r"\s+", part) if _useful(w)]
        if words:
            segments.append(" ".join(words))
    if not segments:                                              # nothing left: fall back to the payment ID's name
        vpa = re.search(r"([A-Z0-9._]+)@", text)
        segments = [vpa.group(1).replace(".", " ").replace("_", " ")] if vpa else [text.strip()[:40] or "UNKNOWN"]
    return {"merchant": segments[0], "text": " ".join(segments), "channel": channel}


def add_merchants(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["description"].map(clean_description)
    out = df.copy()
    out["merchant"] = parts.map(lambda p: p["merchant"])
    out["text"] = parts.map(lambda p: p["text"])
    out["channel"] = parts.map(lambda p: p["channel"])
    return out
