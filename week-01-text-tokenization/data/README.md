# Week 1 data

Everything here is public domain or synthetic. Nothing is scraped, nothing
needs an API key, and no real customer text is included.

| File | Size | What it is |
|---|---:|---|
| `moby_dick.txt` | ~1.2 MB | Melville, *Moby-Dick* (1851). Public domain. |
| `udhr_parallel.json` | ~74 KB | The Universal Declaration of Human Rights in five languages. Public domain. |
| `support_tickets.jsonl` | ~98 KB | 400 synthetic customer-support tickets. |
| `build_data.py` | — | Regenerates `moby_dick.txt` and `udhr_parallel.json` from NLTK. |
| `make_support_tickets.py` | — | Regenerates `support_tickets.jsonl`. Deterministic. |

---

## `moby_dick.txt`

Melville's *Moby-Dick*, from NLTK's `gutenberg` corpus. Line endings
normalised to `\n`; otherwise byte-for-byte the NLTK text. Long enough for a
vocabulary-growth curve, and old enough that its vocabulary looks nothing like
a modern tokenizer's training distribution.

## `udhr_parallel.json`

The UDHR in **English, German, Spanish, Hindi, and Chinese (Simplified)**,
from the Unicode UDHR project via NLTK's `udhr2` (UTF-8) corpus.

```json
{"English": {"code": "eng", "script": "Latin",
             "source_file": "eng.txt", "text": "..."}, ...}
```

This is a **parallel** corpus: every entry says the same thing. Any difference
in token count is therefore a property of the tokenizer, not of the content —
which is what makes it usable as evidence about token cost across languages.

The five languages span the axes that matter for tokenization: Latin-script
analytic (English), Latin-script compounding (German), Latin-script inflecting
(Spanish), a Devanagari abugida (Hindi), and a logographic script with no
whitespace word boundaries (Chinese).

The UDHR is published by the United Nations and is free to reproduce.

Regenerate with:

```bash
uv run python -m nltk.downloader gutenberg udhr2
uv run python week-01-text-tokenization/data/build_data.py
```

## `support_tickets.jsonl`

400 synthetic customer-support tickets, one JSON object per line:

| Field | Values |
|---|---|
| `ticket_id` | `T-100000` … |
| `category` | shipping, defect, billing, … |
| `language` | `en` (375), `de` (12), `es` (7), `fr` (6) |
| `channel` | chat, email, … |
| `text` | the ticket body |

**Synthetic on purpose.** Every public support-ticket dataset of a usable size
is either licence-encumbered or full of real customers' personal data. The
generator produces the surface features that break tokenizers — order IDs,
SKUs, tracking numbers, URLs, e-mail addresses, prices, dates, emoji, ALL
CAPS, and some non-English messages — without shipping anyone's real
complaint.

Deterministic: same seed, same file.

```bash
uv run python week-01-text-tokenization/data/make_support_tickets.py
```

---

## 20 Newsgroups — fetched, not vendored

The corpus behind **Assignment 1** is not stored here. It is fetched at
runtime:

```python
sklearn.datasets.fetch_20newsgroups(
    subset="train",
    categories=["comp.sys.mac.hardware", "rec.sport.baseball", "sci.med", "sci.space"],
    remove=("headers", "footers"),
    random_state=0,
    shuffle=False,
)
```

scikit-learn downloads it once (~14 MB) and caches it in
`~/scikit_learn_data/`; subsequent runs are offline. Headers and footers are
stripped, but **quoted reply text is deliberately kept** — `>` -quoting,
signatures, e-mail addresses, and URLs are exactly the surface noise that
makes tokenization decisions matter.

The 20 Newsgroups collection was assembled by Ken Lang and is redistributed
freely for research and teaching.
