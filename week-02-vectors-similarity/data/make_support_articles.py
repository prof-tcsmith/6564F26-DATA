"""Generate the Week 2 synthetic help-centre corpus.

ISM 6564 — Text Analytics, Fall 2026.

Why synthetic? Week 2 needs a corpus with three properties that no convenient
small public dataset has all at once:

1. **Short, readable documents** — students must be able to eyeball a retrieval
   result and judge for themselves whether it is any good.
2. **Known topic labels** — so a k-means / SVD demo has something to be scored
   against.
3. **Planted near-duplicates** — pairs whose *degree* of overlap we control, so
   the Jaccard-on-shingles demo has a documented ground truth instead of
   "these two look similar to me".

The generator is fully deterministic: it takes a hand-written pool of base
articles and applies three documented edit operations to selected articles to
manufacture near-duplicates. No randomness is used at all, so re-running this
script byte-for-byte reproduces `support_articles.csv`.

Usage
-----
    uv run python week-02-vectors-similarity/data/make_support_articles.py

Writes `support_articles.csv` next to this file with columns:

    doc_id, category, title, body, near_dup_of, edit_kind

`near_dup_of` is empty for base articles and holds the `doc_id` of the source
article for manufactured duplicates. `edit_kind` names which operation produced
it. Together those two columns are the ground truth for the near-duplicate
detection exercise.
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).with_name("support_articles.csv")

# --------------------------------------------------------------------------
# Base articles. Hand-written. Five categories, deliberately overlapping in
# vocabulary at the edges (a "refund" shows up in both returns and payments)
# so that clustering is not trivially separable.
# --------------------------------------------------------------------------

BASE: list[tuple[str, str, str]] = [
    # ---------------- shipping ----------------
    (
        "shipping",
        "How long does standard shipping take?",
        "Standard shipping usually arrives within four to six business days after "
        "your order leaves our warehouse. Orders placed after 2 PM ship the next "
        "business day. Delivery estimates do not include weekends or public "
        "holidays. You will receive a shipping confirmation email with a tracking "
        "number as soon as the parcel is handed to the carrier.",
    ),
    (
        "shipping",
        "Tracking my parcel",
        "Every shipment gets a tracking number, which appears in your shipping "
        "confirmation email and on the order page in your account. Tracking can "
        "take up to twenty four hours to show its first scan after the carrier "
        "collects the parcel. If tracking has not updated for three business days, "
        "contact support and we will open an investigation with the carrier.",
    ),
    (
        "shipping",
        "Do you ship internationally?",
        "We ship to most countries in Europe, North America, and Asia Pacific. "
        "International orders are sent by a partner carrier and typically arrive "
        "in seven to fourteen business days. Import duties and customs charges are "
        "set by the destination country and are the responsibility of the "
        "recipient. We cannot estimate duties before the parcel clears customs.",
    ),
    (
        "shipping",
        "My parcel says delivered but I did not receive it",
        "Carriers sometimes mark a parcel delivered a little before it reaches the "
        "door. Check with neighbours and look for a safe place or parcel locker "
        "note. If the parcel has not appeared after two business days, report it "
        "through the order page and we will start a lost parcel claim with the "
        "carrier. Claims are usually resolved within ten business days.",
    ),
    (
        "shipping",
        "Changing the delivery address after ordering",
        "You can change the delivery address from the order page as long as the "
        "order has not yet been packed. Once the warehouse prints a label the "
        "address is locked and we cannot edit it. If the order has already "
        "shipped, most carriers let the recipient redirect the parcel using the "
        "tracking number on the carrier website.",
    ),
    (
        "shipping",
        "Expedited and next day delivery",
        "Expedited delivery arrives in two business days and next day delivery "
        "arrives the following business day for orders placed before 2 PM. Both "
        "options are shown at checkout with their price. Next day delivery is not "
        "available for oversized items, for addresses outside the mainland, or on "
        "public holidays.",
    ),
    (
        "shipping",
        "Shipping costs and free shipping threshold",
        "Standard shipping is free on orders over fifty dollars. Below that the "
        "flat rate is four dollars and ninety five cents. The free shipping "
        "threshold is calculated on the order total after discounts and before "
        "tax. Expedited and next day delivery are charged separately and are never "
        "included in the free shipping offer.",
    ),
    (
        "shipping",
        "Split shipments",
        "Large orders are sometimes split across several parcels so that in stock "
        "items are not delayed by a backordered one. Each parcel gets its own "
        "tracking number and its own shipping confirmation email. You are only "
        "charged shipping once no matter how many parcels the order becomes.",
    ),
    # ---------------- returns ----------------
    (
        "returns",
        "How do I return an item?",
        "Start a return from the order page in your account within thirty days of "
        "delivery. Print the prepaid return label, pack the item in its original "
        "packaging, and drop it at any carrier location. Refunds are issued once "
        "the warehouse inspects the item, which usually takes three to five "
        "business days after it arrives.",
    ),
    (
        "returns",
        "What can I not return?",
        "Personalised items, gift cards, opened software, and perishable goods "
        "cannot be returned. Underwear and swimwear can only be returned with the "
        "hygiene seal intact. Items marked final sale on the product page are not "
        "eligible for a return or an exchange. Everything else is returnable "
        "within thirty days of delivery.",
    ),
    (
        "returns",
        "Exchanging for a different size",
        "We do not process direct exchanges. Return the item you have for a refund "
        "and place a new order for the size you want. Placing the new order right "
        "away is the fastest route because popular sizes sell out while a return "
        "is in transit. Both the refund and the new order appear separately on "
        "your statement.",
    ),
    (
        "returns",
        "Return shipping costs",
        "Return shipping is free when you use the prepaid label from the order "
        "page. If you post the item back yourself we cannot reimburse the postage "
        "and we cannot investigate a parcel that was not sent on our label. "
        "Returns from outside the country are charged a flat handling fee that is "
        "deducted from the refund.",
    ),
    (
        "returns",
        "My item arrived damaged",
        "Photograph the damage and the outer packaging before you throw anything "
        "away, then report it from the order page within seven days of delivery. "
        "Damaged items are replaced at no cost, or refunded in full if a "
        "replacement is out of stock. You do not need to return a damaged item "
        "unless we specifically ask for it.",
    ),
    (
        "returns",
        "When will my refund appear?",
        "Refunds are issued to the original payment method once the warehouse "
        "inspects the returned item. The refund leaves us within one business day "
        "of inspection, and the bank then takes a further three to five business "
        "days to post it to your statement. Refunds to a credit card can take one "
        "full billing cycle to appear.",
    ),
    (
        "returns",
        "Returning a gift",
        "A gift can be returned with the gift receipt number printed on the packing "
        "slip. Gift returns are refunded as store credit to the recipient rather "
        "than to the purchaser's card, so the buyer is never notified. Store credit "
        "does not expire and can be combined with other payment methods at "
        "checkout.",
    ),
    # ---------------- payments ----------------
    (
        "payments",
        "Which payment methods do you accept?",
        "We accept Visa, Mastercard, and American Express, plus PayPal, Apple Pay, "
        "and Google Pay. Store credit and gift cards can be combined with any card "
        "at checkout. We do not accept cheques, bank transfers, or cash on "
        "delivery for online orders.",
    ),
    (
        "payments",
        "My card was declined",
        "A decline is nearly always a decision made by your bank rather than by us. "
        "Check that the billing address matches your statement exactly, that the "
        "card has not expired, and that the card is enabled for online purchases. "
        "If it still declines, call the number on the back of the card. Repeated "
        "attempts can trigger a temporary block on the card.",
    ),
    (
        "payments",
        "Why do I see a pending charge for an order I cancelled?",
        "When you place an order we ask your bank to authorise the amount, which "
        "shows on your statement as a pending charge. Cancelling releases the "
        "authorisation, but banks can take up to five business days to drop the "
        "pending line. No money leaves your account for a cancelled order even "
        "while the pending charge is visible.",
    ),
    (
        "payments",
        "When am I actually charged?",
        "The card is authorised when you place the order and captured when the "
        "parcel ships. For a split shipment each parcel is captured separately, so "
        "one order can produce several smaller charges on your statement that add "
        "up to the order total. Backordered items are only captured when they "
        "ship.",
    ),
    (
        "payments",
        "Using a promotional code",
        "Enter the promotional code in the box at checkout before you pay. Only one "
        "promotional code can be applied per order and codes cannot be applied "
        "after the order is placed. Codes have an expiry date and some exclude sale "
        "items, which is stated in the terms shown with the code.",
    ),
    (
        "payments",
        "Sales tax and duties",
        "Sales tax is calculated at checkout from the delivery address and is shown "
        "before you confirm the order. International orders may also attract import "
        "duty, which is charged by the destination country when the parcel clears "
        "customs and is not collected by us.",
    ),
    (
        "payments",
        "Gift cards and store credit",
        "Gift cards are delivered by email within fifteen minutes of purchase and "
        "never expire. Store credit is applied automatically at checkout before any "
        "card is charged. Neither gift cards nor store credit can be converted back "
        "into cash, and a lost gift card code cannot be replaced.",
    ),
    (
        "payments",
        "Is my card data stored securely?",
        "Card numbers never touch our servers. Payment details go directly to our "
        "payment processor over an encrypted connection and we only keep a token "
        "and the last four digits so you can recognise a saved card. You can delete "
        "a saved card from the payment methods page at any time.",
    ),
    # ---------------- account ----------------
    (
        "account",
        "Resetting your password",
        "Use the forgot password link on the sign in page. The reset email arrives "
        "within a few minutes and the link inside it is valid for one hour. If the "
        "email does not arrive, check the spam folder and confirm you are using the "
        "address the account was created with. Requesting a second reset email "
        "invalidates the first link.",
    ),
    (
        "account",
        "Changing your email address",
        "Change the email address on the profile page in account settings. We send "
        "a confirmation link to the new address and the change only takes effect "
        "once that link is clicked. Order history, store credit, and saved "
        "addresses all move with the account and are not affected.",
    ),
    (
        "account",
        "Turning on two factor authentication",
        "Two factor authentication is enabled from the security page in account "
        "settings. Scan the code with any authenticator app and save the ten "
        "recovery codes somewhere safe. Recovery codes are the only way back into "
        "the account if you lose the phone, and each one works exactly once.",
    ),
    (
        "account",
        "Deleting your account",
        "Account deletion is requested from the privacy page in account settings "
        "and takes effect after a seven day grace period, during which signing in "
        "cancels the request. Deletion removes your profile, saved addresses, and "
        "saved payment tokens. Order records are retained for the period required "
        "by tax law and cannot be deleted on request.",
    ),
    (
        "account",
        "Managing marketing emails",
        "Marketing preferences live on the notifications page in account settings, "
        "where each list can be turned off independently. Unsubscribing from "
        "marketing does not stop transactional email such as order and shipping "
        "confirmations, which are part of the service and cannot be turned off "
        "while you have open orders.",
    ),
    (
        "account",
        "Saved addresses and default shipping address",
        "You can store as many delivery addresses as you like on the addresses page "
        "and mark one as the default. The default address is preselected at "
        "checkout but can be changed on the order. Editing a saved address does not "
        "change the address on an order that has already been placed.",
    ),
    (
        "account",
        "I did not get my order confirmation email",
        "Order confirmations are sent within a few minutes of checkout. If one is "
        "missing, check the spam folder first, then confirm the address on the "
        "profile page is correct and current. The confirmation is always available "
        "on the order page in your account whether or not the email arrived.",
    ),
    # ---------------- product care ----------------
    (
        "product_care",
        "Caring for leather goods",
        "Wipe leather with a barely damp cloth and let it dry away from direct "
        "heat. Condition twice a year with a neutral cream and never use household "
        "detergent, which strips the finish. Leather darkens with use and small "
        "variations in grain are a feature of the material rather than a defect.",
    ),
    (
        "product_care",
        "Washing merino wool",
        "Wash merino on a wool cycle at thirty degrees with a wool safe detergent, "
        "or by hand in cool water. Do not tumble dry. Press the water out without "
        "wringing and dry flat, because hanging a wet wool garment stretches the "
        "shoulders permanently. Merino only needs washing every few wears.",
    ),
    (
        "product_care",
        "Battery care for cordless devices",
        "Lithium batteries last longest when kept between twenty and eighty percent "
        "charge and away from heat. Storing a device fully discharged for months "
        "can put the cell into a state the charger will not recover. Run the "
        "battery down fully once every few months so the charge gauge stays "
        "accurate.",
    ),
    (
        "product_care",
        "Cleaning a non stick pan",
        "Let the pan cool before washing, then use a soft sponge and warm soapy "
        "water. Metal utensils, abrasive pads, and the dishwasher all shorten the "
        "life of a non stick coating. Heating an empty non stick pan on a high "
        "burner is the fastest way to ruin it.",
    ),
    (
        "product_care",
        "Storing seasonal clothing",
        "Wash everything before storing, because body oils attract moths and set "
        "into stains over a season. Fold knitwear rather than hanging it and use "
        "breathable cotton bags instead of plastic, which traps moisture. Cedar "
        "blocks deter moths and should be sanded lightly each year to refresh the "
        "scent.",
    ),
    (
        "product_care",
        "Warranty coverage and what voids it",
        "Products carry a two year warranty against manufacturing defects from the "
        "delivery date. Normal wear, accidental damage, and damage caused by "
        "unauthorised repair are not covered. Register the product in your account "
        "so the warranty start date is on file, then claim from the order page with "
        "photographs of the fault.",
    ),
]

# --------------------------------------------------------------------------
# Near-duplicate manufacture.
#
# Three documented edit operations, applied to named base articles. These are
# the kinds of duplication a real help centre accumulates: a page copied for a
# regional site, a lightly re-edited version left un-deleted after a rewrite,
# and a short excerpt syndicated onto a landing page.
#
#   "synonym_swap"  — same sentences, a handful of words substituted.
#                     Very high overlap: the pair a duplicate detector must find.
#   "reorder_trim"  — sentences reordered and the last one dropped, plus a new
#                     opening sentence. Moderate overlap: found by word-level
#                     Jaccard, easily missed by character shingles.
#   "excerpt"       — the first two sentences only, with a new closing line.
#                     Low overlap: the hard case, and the one that shows why the
#                     threshold you pick is a business decision.
# --------------------------------------------------------------------------

SYNONYM_TABLE = {
    "usually": "typically",
    "arrives": "shows up",
    "business days": "working days",
    "parcel": "package",
    "warehouse": "fulfilment centre",
    "contact support": "get in touch with support",
    "refund": "reimbursement",
    "delivery": "shipment",
}


def synonym_swap(body: str) -> str:
    """Substitute a fixed synonym table. Longest keys first so that
    'business days' is replaced before 'days' would be."""
    out = body
    for src, dst in sorted(SYNONYM_TABLE.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(src, dst)
    return out


def split_sentences(body: str) -> list[str]:
    """Naive sentence split. The corpus is written with no abbreviations, so
    splitting on '. ' is exact here — which is the point of a generator you
    control."""
    parts = [s.strip() for s in body.split(". ") if s.strip()]
    return [s if s.endswith(".") else s + "." for s in parts]


def reorder_trim(body: str, opener: str) -> str:
    sents = split_sentences(body)
    kept = sents[:-1]
    reordered = kept[1:] + kept[:1]
    return " ".join([opener] + reordered)


def excerpt(body: str, closer: str) -> str:
    sents = split_sentences(body)
    return " ".join(sents[:2] + [closer])


# (source title, edit kind, new title, extra argument)
DUP_PLAN: list[tuple[str, str, str, str]] = [
    (
        "How long does standard shipping take?",
        "synonym_swap",
        "Standard delivery times",
        "",
    ),
    (
        "How do I return an item?",
        "synonym_swap",
        "Starting a return",
        "",
    ),
    (
        "My card was declined",
        "synonym_swap",
        "Payment declined at checkout",
        "",
    ),
    (
        "Tracking my parcel",
        "reorder_trim",
        "Where is my order?",
        "This article explains how to follow an order once it has left us.",
    ),
    (
        "Resetting your password",
        "reorder_trim",
        "I cannot sign in",
        "Locked out of your account? Start here.",
    ),
    (
        "When will my refund appear?",
        "excerpt",
        "Refund timing at a glance",
        "See the full returns policy for the details.",
    ),
    (
        "Washing merino wool",
        "excerpt",
        "Merino wash instructions",
        "The care label on the garment always takes precedence.",
    ),
    (
        "Which payment methods do you accept?",
        "excerpt",
        "Accepted cards and wallets",
        "Payment options are confirmed again on the checkout page.",
    ),
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_title: dict[str, dict[str, str]] = {}

    for i, (category, title, body) in enumerate(BASE, start=1):
        row = {
            "doc_id": f"KB{i:03d}",
            "category": category,
            "title": title,
            "body": " ".join(body.split()),
            "near_dup_of": "",
            "edit_kind": "",
        }
        rows.append(row)
        by_title[title] = row

    next_id = len(BASE) + 1
    for src_title, kind, new_title, arg in DUP_PLAN:
        src = by_title[src_title]
        if kind == "synonym_swap":
            new_body = synonym_swap(src["body"])
        elif kind == "reorder_trim":
            new_body = reorder_trim(src["body"], arg)
        elif kind == "excerpt":
            new_body = excerpt(src["body"], arg)
        else:  # pragma: no cover
            raise ValueError(kind)

        rows.append(
            {
                "doc_id": f"KB{next_id:03d}",
                "category": src["category"],
                "title": new_title,
                "body": " ".join(new_body.split()),
                "near_dup_of": src["doc_id"],
                "edit_kind": kind,
            }
        )
        next_id += 1

    return rows


def main() -> None:
    rows = build_rows()
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["doc_id", "category", "title", "body", "near_dup_of", "edit_kind"]
        )
        writer.writeheader()
        writer.writerows(rows)

    n_dup = sum(1 for r in rows if r["near_dup_of"])
    print(f"wrote {OUT} — {len(rows)} articles ({len(rows) - n_dup} base, {n_dup} near-duplicates)")
    for cat in sorted({r["category"] for r in rows}):
        print(f"  {cat:<14} {sum(1 for r in rows if r['category'] == cat)}")


if __name__ == "__main__":
    main()
