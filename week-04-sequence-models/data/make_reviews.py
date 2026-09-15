"""Build `reviews.csv` — the Week 4 (Sequence Models) product-review corpus.

Why this corpus is synthetic
----------------------------
Week 4 makes one empirical claim: *word order carries label information that a
bag of words cannot represent.* To let students **measure** that claim instead
of taking it on faith, we need a corpus where the order-sensitive portion is
labelled and isolated, and where the *distance* between the two words that
jointly determine the label is a controlled variable. No off-the-shelf review
dataset ships that annotation, so we generate one from clause banks with a
fixed seed.

Everything else about the corpus is deliberately ordinary: retailer product
reviews, balanced labels, 10-50 tokens, a vocabulary of a few hundred types.

The four slices
---------------
`simple`        One clause. The polarity words alone decide the label, so a
                unigram bag of words already has everything it needs. This is
                the control: whatever a sequence model scores here, the cheap
                baseline should match.

`contrast`      "<clause A> but <clause B>", labelled by the polarity of B.
                Generated as **matched pairs** — both orderings of the same two
                clauses appear, with opposite labels. The two members have an
                identical unigram multiset, so any unigram model must score
                them identically and can be right on exactly one: a provable
                50% ceiling. Bigrams *do* crack this slice, because the
                connective sits directly beside the deciding clause. That is
                the point of including it — n-grams buy local order cheaply.

`scope_short`   A polarity-neutral reporting frame ("i can honestly say that" /
                "i can **not** honestly say that") followed immediately by a
                polarity clause. The label is the clause polarity XOR the
                frame's negation. The negation cue and the polarity words are
                ~4 tokens apart, so no n-gram window of 2 or 3 ever contains
                both. A bag of n-grams gets one weight for "not", and needs it
                to push negative beside a positive clause and positive beside a
                negative one. One weight cannot do both.

`scope_long`    The same construction with two polarity-neutral adverbial
                bridges wedged between the frame and the clause, pushing the
                cue-to-content distance to roughly 15-20 tokens. Identical
                task, longer dependency. The gap between `scope_short` and
                `scope_long` is what "long-range dependence" means numerically.

Clause banks are shared across splits. That is deliberate: this corpus measures
*composition*, not vocabulary generalisation. Both members of a contrast pair
always land in the same split, so the 50% unigram ceiling holds within each
split, not just overall.

Usage
-----
    uv run python week-04-sequence-models/data/make_reviews.py
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 6564
OUT = Path(__file__).parent / "reviews.csv"

# Draw counts, before de-duplication. The clause banks are finite, so sampling
# with replacement collides; `dedupe` below drops any group whose text has
# already been emitted, which costs roughly 10-20% of the `simple` draws and
# almost none of the others.
N_SIMPLE = 1500
N_CONTRAST_PAIRS = 620          # -> up to 1240 rows
N_SCOPE_SHORT = 1300
N_SCOPE_LONG = 1250

# --- polarity-bearing clauses ---------------------------------------------

POSITIVE_CLAUSES = [
    "the screen is bright and crisp",
    "the battery easily lasts two days",
    "delivery arrived a day early",
    "the price is unbeatable for this spec",
    "support answered within an hour",
    "the packaging was neat and fully recyclable",
    "the fit is exactly as described",
    "setup took under five minutes",
    "the build quality feels premium",
    "the return label was already in the box",
    "the sound is surprisingly rich",
    "it pairs with my laptop instantly",
    "the printed instructions were clear and short",
    "customer service refunded me the same day",
    "the fabric held up after ten washes",
    "the companion app is responsive and clean",
    "it charges fully in about an hour",
    "the colour matches the product photos",
    "the hinge still feels tight after a year",
    "the seller threw in a spare cable",
    "the carrying case is genuinely useful",
    "it survived a drop onto concrete",
    "the firmware update installed without a hitch",
    "the keys have a satisfying travel",
    "the warranty claim took three minutes online",
    "the weight is far lower than i expected",
    "the stitching is even all the way round",
    "the mounting bracket lines up perfectly",
]

NEGATIVE_CLAUSES = [
    "the screen developed a dead pixel in week two",
    "the battery drains overnight while it is off",
    "delivery was nine days late with no notice",
    "the price went up the day after i ordered",
    "support never replied to my ticket",
    "the packaging arrived crushed and open",
    "the fit runs two sizes small",
    "setup took me most of an afternoon",
    "the build quality feels cheap in the hand",
    "the return label cost me twelve dollars",
    "the sound is tinny at any volume",
    "it disconnects from my laptop constantly",
    "the printed instructions were one unreadable page",
    "customer service put me on hold for an hour",
    "the fabric pilled after a single wash",
    "the companion app crashes on every launch",
    "it stopped charging after three weeks",
    "the colour is nothing like the product photos",
    "the hinge went loose within a month",
    "the seller shipped the wrong variant",
    "the carrying case tore at the seam",
    "it cracked from a knock off the desk",
    "the firmware update bricked the unit",
    "the keys started sticking almost immediately",
    "the warranty claim has been open for six weeks",
    "the weight makes it useless for travel",
    "the stitching came apart at the corner",
    "the mounting bracket snapped under load",
]

# --- reporting frames -----------------------------------------------------
# Each pair is (affirming, negating). The two members differ only by the
# inserted negation word, and neither carries polarity of its own.

FRAME_PAIRS = [
    ("i can honestly say that", "i can not honestly say that"),
    ("i would tell a friend that", "i would never tell a friend that"),
    ("it is true that", "it is not true that"),
    ("i will confirm that", "i will not confirm that"),
    ("the other reviews were correct that", "the other reviews were not correct that"),
    ("my experience matched the claim that", "my experience did not match the claim that"),
]

# --- polarity-neutral adverbial bridges (used only by scope_long) ---------

BRIDGES = [
    "after two weeks of daily use",
    "having tried it in a home office and on the train",
    "compared with the previous model i owned",
    "even allowing for the most recent firmware update",
    "across three different laptops and one tablet",
    "in the two months since the box arrived",
    "once i had worked through the printed setup guide",
    "for the money and for the use case i bought it for",
    "going by what the product listing actually says",
    "after swapping in the cable that came with it",
]

# --- polarity-free filler, to vary length without touching the label ------

PREFIXES = [
    "", "", "", "",
    "ordered this in june", "second one i have bought", "for reference",
    "buying for a home office", "shipped to a uk address",
    "i have used it daily since march", "replacing a five year old unit",
    "bought during the autumn sale",
]

SUFFIXES = [
    "", "", "", "",
    "posting this after a month of use", "happy to answer questions",
    "updating this review later", "model number lc four seven five",
    "bought the black version", "will report back after the winter",
    "ordered through the app",
]


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    counter = {"id": 0}

    def emit(core: str, label: int, slice_: str, group: int,
             pre: str | None = None, suf: str | None = None) -> None:
        pre = rng.choice(PREFIXES) if pre is None else pre
        suf = rng.choice(SUFFIXES) if suf is None else suf
        text = " ".join(p for p in (pre, core, suf) if p).strip()
        rows.append({"id": counter["id"], "text": text, "label": label,
                     "slice": slice_, "group": group})
        counter["id"] += 1

    group = 0

    # -- simple ------------------------------------------------------------
    for _ in range(N_SIMPLE):
        pos = rng.random() < 0.5
        clause = rng.choice(POSITIVE_CLAUSES if pos else NEGATIVE_CLAUSES)
        emit(clause, int(pos), "simple", group)
        group += 1

    # -- contrast: matched "A but B" / "B but A" permutation pairs ---------
    for _ in range(N_CONTRAST_PAIRS):
        a = rng.choice(POSITIVE_CLAUSES)
        b = rng.choice(NEGATIVE_CLAUSES)
        # Same filler on both members so the pair is an exact unigram
        # permutation of itself.
        pre, suf = rng.choice(PREFIXES), rng.choice(SUFFIXES)
        emit(f"{a} but {b}", 0, "contrast", group, pre, suf)
        emit(f"{b} but {a}", 1, "contrast", group, pre, suf)
        group += 1

    # -- scope_short / scope_long -----------------------------------------
    for slice_, n_rows, n_bridges in (("scope_short", N_SCOPE_SHORT, 0),
                                      ("scope_long", N_SCOPE_LONG, 2)):
        for _ in range(n_rows):
            affirm, negate = rng.choice(FRAME_PAIRS)
            negated = rng.random() < 0.5
            frame = negate if negated else affirm

            clause_pos = rng.random() < 0.5
            clause = rng.choice(POSITIVE_CLAUSES if clause_pos else NEGATIVE_CLAUSES)

            bridges = rng.sample(BRIDGES, n_bridges) if n_bridges else []
            core = " ".join([frame, *bridges, clause])

            label = int(clause_pos != negated)      # XOR: negation flips it
            emit(core, label, slice_, group)
            group += 1

    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Drop any group containing a text we have already emitted.

    Without this, ~11% of test rows have their exact text somewhere in train --
    the clause banks are finite and the draws are with replacement. That is
    ordinary train/test leakage and it would quietly inflate every number in the
    assignment, most of all on the `simple` slice.

    De-duplication happens at *group* level so a contrast pair is never split in
    half: either both orderings survive or neither does.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    by_group: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    for g in sorted(by_group):
        members = by_group[g]
        texts = [m["text"] for m in members]
        if len(set(texts)) != len(texts) or any(t in seen for t in texts):
            continue
        seen.update(texts)
        kept.extend(members)

    for i, r in enumerate(kept):        # renumber so `id` stays contiguous
        r["id"] = i
    return kept


def assign_splits(rows: list[dict]) -> None:
    """Split by `group`, so both members of a contrast pair land together."""
    rng = random.Random(SEED + 1)
    groups = sorted({r["group"] for r in rows})
    rng.shuffle(groups)
    n_train, n_val = int(0.65 * len(groups)), int(0.10 * len(groups))
    lookup = {
        g: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, g in enumerate(groups)
    }
    for r in rows:
        r["split"] = lookup[r["group"]]


def check(rows: list[dict]) -> None:
    """Assert the properties the assignment relies on."""
    by_group: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["slice"] == "contrast":
            by_group[r["group"]].append(r)
    for g, pair in by_group.items():
        assert len(pair) == 2, g
        a, b = pair
        assert Counter(a["text"].split()) == Counter(b["text"].split()), g
        assert a["label"] != b["label"], g
        assert a["split"] == b["split"], g

    # No slice may be label-imbalanced enough to make a majority baseline
    # look good.
    for sl in ("simple", "contrast", "scope_short", "scope_long"):
        sub = [r for r in rows if r["slice"] == sl]
        rate = sum(r["label"] for r in sub) / len(sub)
        assert 0.45 < rate < 0.55, (sl, rate)

    # No text may appear twice anywhere in the corpus, so there is no
    # train/test leakage and no label ambiguity.
    texts = [r["text"] for r in rows]
    assert len(set(texts)) == len(texts), "duplicate text survived de-duplication"


def main() -> None:
    rows = dedupe(build())
    assign_splits(rows)
    check(rows)

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "split", "slice", "label", "text"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("id", "split", "slice", "label", "text")})

    slices = ("simple", "contrast", "scope_short", "scope_long")
    print(f"wrote {OUT}  ({len(rows)} rows)")
    lengths = [len(r["text"].split()) for r in rows]
    vocab = {t for r in rows for t in r["text"].split()}
    print(f"  vocabulary: {len(vocab)} types   "
          f"length: min={min(lengths)} median={sorted(lengths)[len(lengths) // 2]} "
          f"max={max(lengths)}")
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        parts = ", ".join(f"{sl}={sum(1 for r in sub if r['slice'] == sl)}"
                          for sl in slices)
        pos = sum(r["label"] for r in sub) / len(sub)
        print(f"  {split:<6} n={len(sub):<6} {parts}   positive={pos:.3f}")


if __name__ == "__main__":
    main()
