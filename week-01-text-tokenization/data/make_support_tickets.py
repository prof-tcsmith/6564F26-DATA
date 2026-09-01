"""Generate `support_tickets.jsonl` — a small synthetic customer-support corpus.

Why synthetic: every public support-ticket dataset of a usable size is either
licence-encumbered or full of real customers' personal data. This generator
produces text with the *surface features that break tokenizers* — order IDs,
SKUs, tracking numbers, URLs, e-mail addresses, prices, dates, emoji, ALL CAPS,
and a few non-English messages — without shipping anyone's real complaint.

It is deterministic: same seed, same file. Run from the repo root:

    uv run python week-01-text-tokenization/data/make_support_tickets.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 6564
N_TICKETS = 400

PRODUCTS = [
    "Aeronaut 45L travel pack", "Corsair wireless earbuds", "Halcyon standing desk",
    "Meridian espresso machine", "Nightjar running shoes", "Overlook 4K webcam",
    "Pinnacle chef's knife", "Quarry ceramic mug set", "Riverstone yoga mat",
    "Solstice down jacket", "Tessellate desk lamp", "Vantage mechanical keyboard",
]
CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]

REFUND = [
    "I returned the {product} on {date} using the prepaid label but my refund of ${price} still hasn't shown up. Order {order}.",
    "Requesting a refund for order {order}. The {product} arrived and it is not what was pictured on {url}.",
    "It has been 14 days since {carrier} confirmed delivery of my return ({tracking}) and I have heard NOTHING. Refund me ${price} please.",
    "Cancelled order {order} within the hour and was still charged ${price}. Please reverse the charge to the card ending 4417.",
    "Hi — I'd like to return the {product} ({sku}). It's unopened. What's the process? Order {order}.",
]
SHIPPING = [
    "Where is my order?? {order} was supposed to arrive {date}. {carrier} tracking {tracking} hasn't updated in 6 days.",
    "Tracking {tracking} says delivered but nothing was at my door. The {product} was a gift and I need it by {date}.",
    "Can I change the delivery address on order {order}? I moved. New address is on file under {email}.",
    "Order {order} shipped in two boxes and only one arrived. Missing: {product} ({sku}).",
    "Is expedited shipping available for {sku}? I need it before {date} and I'll pay the ${price} surcharge.",
]
DEFECT = [
    "The {product} I received is defective — it powers on, clicks twice, then dies. Order {order}, SKU {sku}.",
    "Third {product} in a row with the same fault 😤 . Order {order}. I've attached a video at {url}.",
    "My {product} arrived cracked. Photos attached. Bought it {date} for ${price}, order {order}.",
    "The stitching on the {product} came apart after two weeks of normal use. {sku}. Is this covered under warranty?",
    "Firmware update v2.4.1 bricked my {product}. Rolling back to v2.3.9 doesn't help. Serial on the box is {sku}.",
]
ACCOUNT = [
    "I can't log in. Password reset e-mails to {email} never arrive — I've checked spam. Please help.",
    "Two-factor is stuck in a loop on my account ({email}). The code from the app is always rejected.",
    "Please merge my two accounts — {email} and the older one under the same name. Order history is split across both.",
    "How do I delete my account and all associated data? I'd like written confirmation sent to {email}.",
    "My loyalty points went from 4,280 to 0 overnight. Account {email}. What happened?",
]
BILLING = [
    "I was charged ${price} twice for order {order}. Same card, same timestamp, two line items on the statement.",
    "The invoice for order {order} shows ${price} but the checkout page said ${price2}. Which is right?",
    "Sales tax on order {order} looks wrong — I'm in a state with no clothing tax and I was charged ${price}.",
    "Please send a VAT-compliant invoice for order {order} to {email}. I need it for expenses by {date}.",
    "My subscription renewed at ${price} but I cancelled on {date}. Confirmation was sent to {email}.",
]
NON_ENGLISH = [
    ("es", "Mi pedido {order} llegó dañado. El {product} tiene una grieta en la carcasa. ¿Puedo solicitar un reembolso de ${price}?"),
    ("de", "Die Lieferung zu Bestellung {order} ist unvollständig. Der Artikel {sku} fehlt komplett. Bitte um Rücksendeetikett."),
    ("fr", "Bonjour, le suivi {tracking} indique « livré » mais je n'ai rien reçu. Commande {order}."),
    ("es", "Necesito cambiar la dirección de envío del pedido {order}. Mi correo es {email}."),
    ("de", "Rückerstattung für Bestellung {order} über ${price} steht seit dem {date} aus. Bitte prüfen."),
]

CATEGORIES = {
    "refund": REFUND, "shipping": SHIPPING, "defect": DEFECT,
    "account": ACCOUNT, "billing": BILLING,
}
CHANNELS = ["email", "web_form", "chat"]

OPENERS = ["", "Hi, ", "Hello, ", "Hey — ", "Good morning. ", "To whom it may concern: "]
CLOSERS = ["", " Thanks.", " Thanks!", " Please advise.", " Appreciate any help 🙏", " -- sent from my phone"]


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for i in range(N_TICKETS):
        if rng.random() < 0.08:
            lang, template = rng.choice(NON_ENGLISH)
            category = "other"
        else:
            lang = "en"
            category = rng.choice(list(CATEGORIES))
            template = rng.choice(CATEGORIES[category])

        price = rng.randrange(1200, 89900) / 100
        text = template.format(
            product=rng.choice(PRODUCTS),
            sku=f"SKU-{rng.choice('ABCDEFGHJKLMNPQRSTVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTVWXYZ')}"
                f"{rng.randrange(10, 100)}-{rng.randrange(1000, 10000)}",
            order=f"ORD-2026-{rng.randrange(100000, 1000000)}",
            tracking=f"1Z{rng.randrange(10**11, 10**12)}",
            carrier=rng.choice(CARRIERS),
            email=f"{rng.choice(['j.okafor', 'm.rivera', 'a.petrov', 's.nakamura', 'l.dubois', 'k.osei'])}"
                  f"{rng.randrange(10, 100)}@example.com",
            url=f"https://shop.example.com/p/{rng.randrange(10000, 100000)}?ref=support",
            date=f"2026-{rng.randrange(1, 9):02d}-{rng.randrange(1, 29):02d}",
            price=f"{price:,.2f}",
            price2=f"{price * rng.choice([0.9, 1.1]):,.2f}",
        )
        if lang == "en":
            text = rng.choice(OPENERS) + text + rng.choice(CLOSERS)

        rows.append({
            "ticket_id": f"T-{100000 + i}",
            "category": category,
            "language": lang,
            "channel": rng.choice(CHANNELS),
            "text": text,
        })

    out = HERE / "support_tickets.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    chars = sum(len(r["text"]) for r in rows)
    print(f"support_tickets.jsonl  {len(rows)} tickets, {chars:,} chars, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
