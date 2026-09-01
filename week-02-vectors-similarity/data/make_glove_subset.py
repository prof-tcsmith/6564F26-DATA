"""Build the vendored GloVe subset used by the Week 2 lecture notebook.

ISM 6564 — Text Analytics, Fall 2026.

The full `glove-wiki-gigaword-100` release is 400,000 words × 100 dimensions,
a 128 MB download. That is a long time to sit and watch in a 12:30 class, and
it is a hard dependency on the network for anyone running the notebook on
campus wifi. So we ship a subset instead:

* the **top `TOP_N` most frequent words** — GloVe's vocabulary file is already
  ordered by corpus frequency, so `index_to_key[:TOP_N]` is exactly "the most
  common English words", which is what the analogy and nearest-neighbour demos
  search over; plus
* **every word that occurs in `support_articles.csv`**, so the document-vector
  demo has full coverage of the corpus it is run on.

Vectors are stored as `float32` (GloVe is distributed at that precision
anyway), which keeps the file comfortably under the repository's 10 MB
per-file guideline.

Requires the full model, so this script needs network access on first run:

    uv run python week-02-vectors-similarity/data/make_glove_subset.py

Writes `glove_6b_100d_subset.npz` next to this file, with two arrays:

    words    : (V,)      unicode  — vocabulary, most-frequent first
    vectors  : (V, 100)  float32  — row i is the vector for words[i]

Licence: GloVe is released by Stanford NLP under the Public Domain Dedication
and License (PDDL) v1.0, which permits redistribution of derived subsets.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "glove_6b_100d_subset.npz"
ARTICLES = HERE / "support_articles.csv"

TOP_N = 15_000
TOKEN_RE = re.compile(r"[a-z]+")

# Words the lecture demos name explicitly. Almost all are inside the top-N
# already; listing them makes the dependency explicit rather than lucky.
DEMO_WORDS = [
    "king", "queen", "man", "woman", "monarch", "throne",
    "paris", "france", "italy", "rome", "milan", "berlin", "germany",
    "walk", "walking", "swim", "swimming", "surfing",
    "big", "bigger", "small", "smaller", "larger",
    "doctor", "nurse", "physician", "engineer", "teacher",
    "bank", "banks", "river", "riverbank", "financial", "credit", "loan",
    "shore", "stream", "deposit", "teller",
    "apple", "microsoft", "orange", "fruit", "banana",
    "refund", "shipping", "parcel", "package", "delivery", "carrier",
    "warranty", "wool", "leather", "battery", "password", "checkout",
]


def corpus_words() -> list[str]:
    if not ARTICLES.exists():
        raise SystemExit(
            f"{ARTICLES} not found — run make_support_articles.py first."
        )
    seen: list[str] = []
    with ARTICLES.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for field in ("title", "body"):
                seen.extend(TOKEN_RE.findall(row[field].lower()))
    return seen


def main() -> None:
    import gensim.downloader as api

    print("loading glove-wiki-gigaword-100 (128 MB on first run)...")
    kv = api.load("glove-wiki-gigaword-100")
    print(f"  full model: {len(kv.index_to_key):,} words x {kv.vector_size}d")

    keep: dict[str, None] = {}          # dict preserves insertion order
    for w in kv.index_to_key[:TOP_N]:
        keep[w] = None

    added_demo = 0
    for w in DEMO_WORDS:
        if w in kv.key_to_index and w not in keep:
            keep[w] = None
            added_demo += 1

    missing_corpus, added_corpus = set(), 0
    for w in corpus_words():
        if w in keep:
            continue
        if w in kv.key_to_index:
            keep[w] = None
            added_corpus += 1
        else:
            missing_corpus.add(w)

    words = np.array(list(keep), dtype=object)
    vectors = np.stack([kv[w] for w in words]).astype(np.float32)

    np.savez_compressed(OUT, words=words, vectors=vectors)

    size_mb = OUT.stat().st_size / 1e6
    print(f"  kept top {TOP_N:,} + {added_demo} demo + {added_corpus} corpus words")
    print(f"  out-of-GloVe corpus words: {sorted(missing_corpus) or 'none'}")
    print(f"wrote {OUT.name}: {vectors.shape[0]:,} x {vectors.shape[1]} float32, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
