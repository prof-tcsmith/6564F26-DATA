"""Build `glove_20news_100d.npz` — the GloVe subset Week 3 needs.

Week 3 §4.2 represents a document as the *average* of its word vectors and
compares that against TF-IDF. To do that honestly the vectors have to cover the
corpus we are actually classifying, which is the four-group 20 Newsgroups slice
used throughout the Week 3 lecture notebook.

The full `glove-wiki-gigaword-100` release is 400,000 words and a 128 MB
download. Shipping that is a hard network dependency and far too slow to watch
in class. Instead this script keeps only the words that

  1. occur at least `MIN_COUNT` times in the Week 3 corpus (train + test,
     `remove=("headers", "footers", "quotes")`, `CountVectorizer` defaults), and
  2. exist in GloVe's vocabulary.

That is ~17.5k words: about 7 MB as `float32`, and it covers ~96% of the corpus
*tokens* (not types — the long tail of one-off tokens is what is dropped, and
the notebook reports the coverage number it measures rather than this comment).

Output arrays in the `.npz`:
    words    (N,)      object array, corpus-frequency order, most frequent first
    vectors  (N, 100)  float32; row i is the vector for words[i]

Deterministic: no RNG anywhere. Re-running reproduces the file.

Requires network access on first run (downloads the full GloVe model to
`~/gensim-data/`). After that it is offline.

    uv run python week-03-text-classification/data/make_glove_subset.py
"""

from pathlib import Path

import numpy as np

MIN_COUNT = 2
CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "sci.med",
    "soc.religion.christian",
]
OUT = Path(__file__).parent / "glove_20news_100d.npz"


def main() -> None:
    import gensim.downloader as api
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    print("fetching 20 newsgroups ...")
    train = fetch_20newsgroups(
        subset="train", categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    test = fetch_20newsgroups(
        subset="test", categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    docs = list(train.data) + list(test.data)
    print(f"  {len(docs)} documents")

    counter = CountVectorizer()
    counts = counter.fit_transform(docs)
    vocab = counter.get_feature_names_out()
    totals = np.asarray(counts.sum(axis=0)).ravel()
    print(f"  {len(vocab)} distinct tokens, {totals.sum()} token occurrences")

    print("loading glove-wiki-gigaword-100 (128 MB on first run) ...")
    glove = api.load("glove-wiki-gigaword-100")

    # corpus-frequency order, most frequent first, ties broken alphabetically
    order = sorted(range(len(vocab)), key=lambda j: (-totals[j], vocab[j]))
    keep = [j for j in order if totals[j] >= MIN_COUNT and vocab[j] in glove.key_to_index]

    words = np.array([vocab[j] for j in keep], dtype=object)
    vectors = np.stack([glove[w] for w in words]).astype(np.float32)

    covered = totals[keep].sum() / totals.sum()
    print(f"  kept {len(words)} words "
          f"({len(words) / len(vocab):.1%} of types, {covered:.1%} of tokens)")

    np.savez_compressed(OUT, words=words, vectors=vectors)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
