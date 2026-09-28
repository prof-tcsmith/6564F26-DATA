#!/usr/bin/env python
"""Generate the Week 6 synthetic customer-support ticket corpus.

ISM 6564 — Text Analytics, Fall 2026.
Northwind Outfitters, an online outdoor retailer, is the running example for
Weeks 6–10.

WHY SYNTHETIC?
Week 6 needs a corpus with four properties at once, and no convenient small
public dataset has all four:

1. **A house-specific label space.** The point of the fine-tuning half of the
   week is that a base model cannot guess *your* internal routing codes
   (`SHIP-DELAY`, `RET-RMA`, ...).  A public dataset labelled with ordinary
   English category names lets the base model succeed for the wrong reason.
2. **A learnable escalation rule.**  `needs_human` is a deterministic function
   of things actually visible in the ticket, so "the model got it wrong" is a
   real error and not label noise.
3. **Redistributable.**  Real support mail is PII-laden and licence-encumbered.
4. **An honest test set.**  See below.

THE TEMPLATE-DISJOINT SPLIT — the part that matters
Template-generated corpora leak.  If train and test draw from the same
templates, a bigram TF-IDF model memorises one template per label and scores
100%, which tells you nothing.  So the split here is **by template, not by
row**: templates 0–6 of each category generate the training set, templates
7–10 generate the test set, and the two never overlap.  Every phrasing in the
test set is one the model has never seen.  That is the production question —
new customers write in new words — and it is the only split under which the
Week 3 baseline and a language model can be compared fairly.

The generator is seeded, so this script reproduces both CSVs exactly.

Run:  uv run python week-06-prompting-apis-finetuning/data/make_support_tickets.py
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

SEED = 6564
N_TRAIN = 720
N_TEST = 180
N_TRAIN_TEMPLATES = 7          # templates 0..6 → train, 7..10 → test

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# The house label space.  These codes are internal to Northwind Outfitters and
# appear nowhere in any pretraining corpus.
# ---------------------------------------------------------------------------
CATEGORIES = {
    "SHIP-DELAY": "shipping status, late or missing delivery, tracking",
    "RET-RMA": "returns, exchanges, refund requests for delivered goods",
    "BILL-DISP": "charges, invoices, duplicate or unexpected payments",
    "PROD-DEF": "an item arrived damaged, broken, or defective",
    "ACCT-SEC": "login, password, account lockout, suspicious activity",
    "PROD-INFO": "pre-purchase questions about sizing, materials, stock",
}

PRODUCTS = [
    "trail runner GTX", "packable down jacket", "merino base layer",
    "40L hauler duffel", "insulated flask", "canvas field pants",
    "wool crew socks (3-pack)", "storm shell", "camp chair", "headlamp 400",
    "climbing chalk bag", "waxed cotton cap",
]
SIZES = ["XS", "S", "M", "L", "XL", "US 9", "US 10.5", "EU 42", "one size"]
COLORS = ["slate", "moss", "clay", "black", "dune", "navy"]

OPENERS = [
    "Hi,", "Hello,", "Hi there,", "Good morning,", "Hey,",
    "To whom it may concern,", "", "Hi team,", "Afternoon,",
]
CLOSERS = [
    "Thanks.", "Thanks in advance.", "Please advise.", "Appreciate the help.",
    "", "Regards,\nA. Customer", "Let me know.", "Thank you!",
]

# ---------------------------------------------------------------------------
# Body templates: (text, flags).  Flags drive the escalation rule, so the rule
# does not depend on brittle substring matching and survives new phrasings.
#   "unauth"     — the customer reports account activity they did not authorise
#   "chargeback" — the customer disputes or does not recognise a charge
# Indices 0..6 are the TRAIN templates; 7..10 are the held-out TEST templates.
# ---------------------------------------------------------------------------
BODIES: dict[str, list[tuple[str, frozenset]]] = {
    "SHIP-DELAY": [
        ("Order {order} was supposed to arrive {days} days ago and the tracking hasn't moved since it left the warehouse.", frozenset()),
        ("I placed order {order} on the {dom}th and it still says 'label created'. Has it actually shipped?", frozenset()),
        ("The tracking for {order} shows delivered but nothing is at my door and my neighbours don't have it either.", frozenset()),
        ("Any update on {order}? It's now {days} days past the estimated delivery window.", frozenset()),
        ("My {prod} ({order}) is stuck in transit. Carrier says they're waiting on the shipper.", frozenset()),
        ("I paid for two-day shipping on {order} and it's been {days} days.", frozenset()),
        ("Where is order {order}? The status page has said 'in transit' all week.", frozenset()),
        # --- held out for test ---
        ("The courier left a card for {order} but never actually knocked, and now the parcel is back at the depot.", frozenset()),
        ("Half of {order} turned up — the {prod} is here, the rest of the box is apparently still in a warehouse somewhere.", frozenset()),
        ("Your site says {order} shipped {days} days ago but the carrier has no record of receiving it.", frozenset()),
        ("I've had no despatch email for {order} and the account page just shows a spinner.", frozenset()),
    ],
    "RET-RMA": [
        ("I'd like to return the {prod} from order {order}. Wrong size, I need {size} instead of {size2}.", frozenset()),
        ("Can I exchange the {color} {prod} on {order} for the {color2} one? Unworn, tags on.", frozenset()),
        ("Order {order} — the {prod} doesn't fit. What's the return process?", frozenset()),
        ("I want a refund for {order}. The {prod} isn't what I expected from the photos.", frozenset()),
        ("How do I send back the {prod} from {order}? It's within the 30-day window.", frozenset()),
        ("Returning {order}. Do I need to pay for the return label or is it prepaid?", frozenset()),
        ("Ordered two sizes of the {prod} on {order} to compare. Sending {size2} back.", frozenset()),
        # --- held out for test ---
        ("The {prod} on {order} is lovely but far too warm for what I need. Can I swap it for something lighter?", frozenset()),
        ("Bought the {prod} ({order}) as a gift and it wasn't wanted. Is a gift return possible without the recipient knowing the price?", frozenset()),
        ("{order}: I've changed my mind on the {color} {prod}. Never taken out of the bag. What now?", frozenset()),
        ("Is there any chance of store credit rather than a card refund for the {prod} on {order}?", frozenset()),
    ],
    "BILL-DISP": [
        ("I was charged {amt} twice for order {order}. Both charges cleared my card.", frozenset()),
        ("There's a charge of {amt} on my statement from you and I don't recognise it. No order number I can find.", frozenset({"chargeback"})),
        ("My refund for {order} was supposed to be {amt} but only {amt2} came back.", frozenset()),
        ("You've billed me {amt} for order {order} but the checkout total was {amt2}.", frozenset()),
        ("I cancelled order {order} and the {amt} charge is still pending after {days} days.", frozenset()),
        ("Invoice for {order} shows a shipping fee of {amt2} that wasn't on the order page.", frozenset()),
        ("My card was charged {amt} for an order I never placed. This is not my account activity.", frozenset({"chargeback"})),
        # --- held out for test ---
        ("A {amt} payment to Northwind showed up on my statement this morning and there is no matching order anywhere in my history.", frozenset({"chargeback"})),
        ("The {amt2} promo code on {order} came off the subtotal but the card was still hit for the full {amt}.", frozenset()),
        ("You've taken a {amt} subscription fee I never signed up for. I want it reversed.", frozenset({"chargeback"})),
        ("Order {order} was split into two shipments and I appear to have been charged shipping of {amt2} on each.", frozenset()),
    ],
    "PROD-DEF": [
        ("The {prod} from order {order} arrived with a torn seam along the left side.", frozenset()),
        ("My {prod} ({order}) showed up with the zip pull snapped clean off.", frozenset()),
        ("Order {order} — the {prod} has a hole in the packaging and a deep scratch across the front.", frozenset()),
        ("The {color} {prod} in {order} is defective: the buckle won't latch at all.", frozenset()),
        ("Received {order} today. The {prod} leaks from the base seam the moment you fill it.", frozenset()),
        ("The stitching on the {prod} from {order} came apart the first time I wore it.", frozenset()),
        ("{prod} from order {order} arrived cracked. Box looks fine so it was packed that way.", frozenset()),
        # --- held out for test ---
        ("There's a manufacturing fault on the {prod} from {order} — the two halves of the shell don't line up at all.", frozenset()),
        ("The {color} {prod} in order {order} has dye coming off on everything it touches.", frozenset()),
        ("Waterproofing on the {prod} ({order}) failed within an hour. It wet through completely.", frozenset()),
        ("One of the clips on the {prod} from {order} sheared off under normal use on day two.", frozenset()),
    ],
    "ACCT-SEC": [
        ("I can't log in. Password reset emails never arrive and I've checked spam {days} times.", frozenset()),
        ("My account is locked after too many attempts. Can you unlock it?", frozenset()),
        ("I got an email saying my password changed. I didn't change it.", frozenset({"unauth"})),
        ("There's an order in my history I didn't place. I think someone has access to my account.", frozenset({"unauth"})),
        ("Two-factor is texting a phone number that isn't mine anymore. I can't get in.", frozenset()),
        ("Signing in redirects me back to the login page every time. Cleared cookies already.", frozenset()),
        ("Someone tried to log into my account from another country according to your alert email.", frozenset({"unauth"})),
        # --- held out for test ---
        ("My saved delivery address has been changed to somewhere I've never lived and it wasn't me.", frozenset({"unauth"})),
        ("The site keeps telling me my email is already registered, but the password I set does not work.", frozenset()),
        ("A device I don't own is listed under active sessions on my profile.", frozenset({"unauth"})),
        ("Single sign-on with my work email throws an error page every time and I can no longer reach my account.", frozenset()),
    ],
    "PROD-INFO": [
        ("Does the {prod} run true to size? I'm normally a {size}.", frozenset()),
        ("Is the {color} {prod} coming back in stock? It's been sold out for weeks.", frozenset()),
        ("What's the shell fabric on the {prod}? Looking for something actually waterproof, not just resistant.", frozenset()),
        ("Is the {prod} machine washable or does it need special care?", frozenset()),
        ("Do you ship the {prod} to Canada, and roughly what's the duty on it?", frozenset()),
        ("How does the {prod} compare to the {prod2} for cold weather?", frozenset()),
        ("Before I order — is the {prod} in {size} cut the same as last season's?", frozenset()),
        # --- held out for test ---
        ("What's the packed weight of the {prod}? Trying to keep the whole kit under a limit.", frozenset()),
        ("Would the {prod} work as a layer under the {prod2}, or is that going to be too bulky?", frozenset()),
        ("Any chance of a restock alert for the {color2} {prod} in {size2}?", frozenset()),
        ("Is the {prod} covered by the lifetime repair scheme or only the {prod2}?", frozenset()),
    ],
}

# Secondary-intent sentences.  About a fifth of tickets get one drawn from a
# *different* category, because real support mail is rarely single-intent, and
# because it stops a bag-of-words model from solving the task by memorising one
# vocabulary per label.  The label is the PRIMARY intent.
SECONDARY = {
    "SHIP-DELAY": ["Also, is there a way to watch the carrier tracking directly?",
                   "Separately, the delivery estimate on the site never updated."],
    "RET-RMA": ["If it can't be sorted I'd rather just send it back for a refund.",
                "Also, how long is the exchange window on something like this?"],
    "BILL-DISP": ["And can you confirm I wasn't billed twice for it?",
                  "Also the invoice total doesn't match what I remember at checkout."],
    "PROD-DEF": ["The outer box was pretty beaten up too.",
                 "Also there's a scuff on the front that was clearly there before shipping."],
    "ACCT-SEC": ["Separately, my password reset link expired before I could use it.",
                 "Also I had to sign in three times to write this."],
    "PROD-INFO": ["Also, does that model come in a taller cut?",
                  "And while I'm here — is the same fabric used on the lined version?"],
}

# Soft time references that are NOT high urgency.  Without these, any mention
# of time at all is a perfect predictor of urgency=high.
SOFT_TIME = [
    "Would be good to get this settled.",
    "Hoping for an update this week if possible.",
    "I'd like to sort this out soon-ish.",
    "Happy to wait a couple of days for a reply.",
]

URGENCY_MARKERS = [
    "I need this resolved today.",
    "This is time sensitive — I fly out on Friday.",
    "It's for a gift and I'm running out of time.",
    "Third time I've written about this.",
    "I've been waiting on a reply for over a week.",
]
ANGRY_MARKERS = [
    "This is completely unacceptable.",
    "I'm about to dispute this with my bank.",
    "If this isn't fixed I'm done shopping here.",
    "Frankly this has been a shambles.",
]


# ---------------------------------------------------------------------------
# The escalation rule.  needs_human is True when ANY of:
#   1. ACCT-SEC  and the customer reports unauthorised account activity
#   2. BILL-DISP and the customer disputes or does not recognise the charge
#   3. the ticket carries an angry marker
#   4. urgency is high and the category is not PROD-INFO
# ---------------------------------------------------------------------------
def needs_human(category: str, flags: frozenset, angry: bool, urgency: str) -> bool:
    if category == "ACCT-SEC" and "unauth" in flags:
        return True
    if category == "BILL-DISP" and "chargeback" in flags:
        return True
    if angry:
        return True
    return urgency == "high" and category != "PROD-INFO"


def make_ticket(rng: random.Random, idx: int, split: str) -> dict:
    category = rng.choice(list(CATEGORIES))
    pool = range(N_TRAIN_TEMPLATES) if split == "train" else range(N_TRAIN_TEMPLATES, len(BODIES[category]))
    t_idx = rng.choice(list(pool))
    template, flags = BODIES[category][t_idx]

    prod, prod2 = rng.sample(PRODUCTS, 2)
    size, size2 = rng.sample(SIZES, 2)
    color, color2 = rng.sample(COLORS, 2)
    order = f"NW-{rng.randint(10000, 99999)}"
    amt = f"${rng.randint(24, 340)}.{rng.randint(0, 99):02d}"
    amt2 = f"${rng.randint(9, 300)}.{rng.randint(0, 99):02d}"

    body = template.format(
        order=order, prod=prod, prod2=prod2, size=size, size2=size2,
        color=color, color2=color2, amt=amt, amt2=amt2,
        days=rng.randint(2, 21), dom=rng.randint(2, 27),
    )
    parts = [body]

    if rng.random() < 0.22:
        other = rng.choice([c for c in CATEGORIES if c != category])
        parts.append(rng.choice(SECONDARY[other]))

    angry = False
    roll = rng.random()
    if roll < 0.22:
        urgency = "high"
        parts.append(rng.choice(URGENCY_MARKERS))
        if rng.random() < 0.45:
            parts.append(rng.choice(ANGRY_MARKERS))
            angry = True
    elif roll < 0.40:
        urgency = "high"
        parts.append(rng.choice(ANGRY_MARKERS))
        angry = True
    elif roll < 0.70:
        urgency = "normal"
        if rng.random() < 0.40:
            parts.append(rng.choice(SOFT_TIME))
    else:
        urgency = "low"
        parts.append(rng.choice(["No rush.", "Whenever you get a chance.", "Not urgent."]))

    opener, closer = rng.choice(OPENERS), rng.choice(CLOSERS)
    text = " ".join(p for p in ([opener] + parts + [closer]) if p).strip()

    # Light, realistic surface noise on ~8% of tickets.
    if rng.random() < 0.08:
        text = text.replace("the ", "teh ", 1)
    if rng.random() < 0.05:
        text = text.lower()

    order_id = order if order.lower() in text.lower() else ""
    return {
        "ticket_id": f"T{idx:05d}",
        "text": text,
        "category": category,
        "urgency": urgency,
        "needs_human": str(needs_human(category, flags, angry, urgency)).lower(),
        "order_id": order_id,
        "template_id": f"{category}:{t_idx}",
    }


def build(split: str, n: int, start: int, rng: random.Random) -> list[dict]:
    rows, seen, idx = [], set(), start
    while len(rows) < n:
        t = make_ticket(rng, idx, split)
        idx += 1
        if t["text"] in seen:
            continue
        seen.add(t["text"])
        rows.append(t)
    return rows


def main() -> None:
    rng = random.Random(SEED)
    train = build("train", N_TRAIN, 0, rng)
    test = build("test", N_TEST, 100000, rng)
    rng.shuffle(train)
    rng.shuffle(test)

    fields = ["ticket_id", "text", "category", "urgency", "needs_human", "order_id", "template_id"]
    for name, part in (("train", train), ("test", test)):
        path = HERE / f"support_tickets_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(part)
        print(f"wrote {path.name}: {len(part)} rows")

    both = train + test
    print("\ncategory:   ", dict(Counter(r["category"] for r in both)))
    print("urgency:    ", dict(Counter(r["urgency"] for r in both)))
    print("needs_human:", dict(Counter(r["needs_human"] for r in both)))
    print("has order_id:", sum(bool(r["order_id"]) for r in both), "/", len(both))
    tr_t = {r["template_id"] for r in train}
    te_t = {r["template_id"] for r in test}
    print(f"templates: {len(tr_t)} train, {len(te_t)} test, overlap = {len(tr_t & te_t)}")


if __name__ == "__main__":
    main()
