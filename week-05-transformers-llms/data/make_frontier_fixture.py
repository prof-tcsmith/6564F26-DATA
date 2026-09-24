"""Write `frontier_cached.json` — the stand-in frontier-model transcript.

The Week 5 assignment compares a locally-run small language model against a
frontier API on the same 60 tickets. Students do not have API keys until Week 6,
so the assignment ships a **fixture**: the reply strings are authored, not
sampled from a live model.

Everything in the file that *can* be computed is computed. Token counts are real
`tiktoken` counts over the exact prompt and reply strings the notebook builds,
so the cost arithmetic in the assignment is arithmetic on real numbers. Nothing
in the file claims to be a latency measurement, because none of it is one.

Rebuild:

    uv run python week-05-transformers-llms/data/make_frontier_fixture.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import tiktoken

HERE = Path(__file__).parent
TICKETS = HERE / "support_tickets.csv"
OUT = HERE / "frontier_cached.json"

CATEGORIES = ["Shipping", "Billing", "Returns", "Defect", "Account", "Other"]

SYSTEM_PROMPT = (
    "You are a support-ticket triage assistant for Northwind Outfitters, an online "
    "outdoor-gear retailer.\n"
    "Read the customer message and choose the single best category.\n\n"
    "Categories:\n"
    "- Shipping: where is my order, delivery delays, lost or misdelivered packages, "
    "tracking, delivery address.\n"
    "- Billing: charges, refunds not yet received, invoices, payment methods, promo "
    "codes, tax.\n"
    "- Returns: starting a return or exchange, return labels, return policy, wrong "
    "item or size sent.\n"
    "- Defect: the product arrived damaged or failed in use; quality complaints.\n"
    "- Account: login, password, email address, profile, loyalty points, newsletter "
    "subscription.\n"
    "- Other: anything that does not fit the categories above.\n\n"
    "Answer with exactly one category word and nothing else."
)

# Tickets the fixture answers with something other than the gold label, and what
# it answers instead. Seven of the eighteen `ambiguous` tickets plus one `clear`
# one: a frontier model that is perfect on every clear ticket is not a realistic
# fixture, and the assignment's error analysis needs at least one clear miss to
# chew on.
DISAGREEMENTS = {
    "T008": "Billing",   # charged for expedited shipping, wants the fee back
    "T010": "Defect",    # box crushed, contents apparently fine
    "T016": "Account",   # clear: switching the card on a membership
    "T019": "Account",   # membership auto-renewed after profile change
    "T038": "Returns",   # stove failed, customer wants money back
    "T048": "Billing",   # loyalty points vanished
    "T058": "Returns",   # out-of-warranty repair pricing
    "T060": "Returns",   # size chart contradicts the garment tag
}

# Formatting deviations. The task says "one category word and nothing else";
# real frontier replies still drift, and the assignment's parser has to cope.
FORMATS = {
    "T003": "{label}.",
    "T022": "Category: {label}",
    "T035": "{lower}",
    "T047": "**{label}**",
    "T053": "{label} ",
}


def main() -> None:
    rows = list(csv.DictReader(TICKETS.open(encoding="utf-8")))
    assert len(rows) == 60, len(rows)
    enc = tiktoken.get_encoding("o200k_base")

    sys_tokens = len(enc.encode(SYSTEM_PROMPT))
    responses = {}
    n_agree = 0
    for r in rows:
        label = DISAGREEMENTS.get(r["id"], r["category"])
        n_agree += label == r["category"]
        template = FORMATS.get(r["id"], "{label}")
        reply = template.format(label=label, lower=label.lower())
        # Chat-completions billing counts the system message once per request
        # plus the user message; the small per-message framing overhead the
        # provider adds is not something we can count from outside, so this is
        # a floor, and the assignment says so.
        prompt_tokens = sys_tokens + len(enc.encode(r["text"]))
        completion_tokens = len(enc.encode(reply))
        responses[r["id"]] = {
            "reply": reply,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    payload = {
        "meta": {
            "kind": "fixture",
            "model_label": "frontier-mini (stand-in)",
            "note": (
                "Authored reply strings, NOT sampled from a live model. Token counts "
                "are real tiktoken o200k_base counts over the exact strings above. "
                "No latency is recorded because none was measured. Set OPENAI_API_KEY "
                "and re-run the notebook's live cell to replace this with a real "
                "transcript."
            ),
            "encoding": "o200k_base",
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "system_prompt_tokens": sys_tokens,
            "n_tickets": len(rows),
            "n_agreeing_with_gold": n_agree,
        },
        "responses": responses,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  system prompt: {sys_tokens} tokens (o200k_base)")
    print(f"  fixture agrees with gold on {n_agree}/{len(rows)} = {n_agree / len(rows):.4f}")
    tot_in = sum(v["prompt_tokens"] for v in responses.values())
    tot_out = sum(v["completion_tokens"] for v in responses.values())
    print(f"  total prompt tokens {tot_in:,}, completion tokens {tot_out:,}")


if __name__ == "__main__":
    main()
