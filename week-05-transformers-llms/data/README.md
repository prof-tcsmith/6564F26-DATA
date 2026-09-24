# Week 5 data — Transformers & Language Models

Everything the Week 5 lecture notebook and the Week 5 graded assignment read.
Files load **local-first with a URL fallback**, so the notebooks work in a clone
and in Colab.

Week 5 is unusual among the ten weeks: the objects under study are **model
weights, not a corpus**. Almost everything the notebook measures — parameter
distribution, KV-cache size, attention matrices, throughput — is a property of a
checkpoint rather than of a dataset. The one thing that genuinely needs data is
§6, which has to answer "is this small model good enough?", and that requires a
task set with a grading rule.

---

## Fetched at runtime — deliberately not vendored

Per `docs/AUTHORING.md` §6, nothing from the Hugging Face Hub is committed here.
The notebook calls `from_pretrained`, which caches under `~/.cache/huggingface`
(override with `HF_HOME`). First run needs network; later runs do not. §5.3 of
the notebook prints the resulting cache with `scan_cache_dir()` and shows how to
clear it.

**Total first-run download: about 6 GB.**

| Repo | Role | On-disk | Licence |
| --- | --- | ---: | --- |
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | The workhorse. Attention weights (§1.5), causal-mask proof (§3.2), parameter and memory anatomy (§4), generation loop and KV cache (§4.4–4.5), chat template and decoding (§5), task-set evaluation (§6.2). | 1.00 GB | Apache-2.0 |
| [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) | The *same pretraining run without instruction tuning*, so §4.3 can show a base model continuing text instead of answering. This contrast is the reason the second checkpoint is worth its gigabyte. | 1.00 GB | Apache-2.0 |
| [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Top rung of the §6.3 size ladder. | 3.10 GB | Apache-2.0 |
| [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Bottom rung of the §6.3 size ladder — small enough to show where instruction tuning stops sticking. | 0.27 GB | Apache-2.0 |
| [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | Encoder-only example: masked-LM fill-in (§3.1), the learned position-embedding table (§2.3), and the bidirectional half of the causal-mask proof (§3.2). | 0.44 GB | Apache-2.0 |
| [`t5-small`](https://huggingface.co/t5-small) | Encoder–decoder example (§3.3). Small enough to translate and summarise on CPU in a couple of seconds. | 0.24 GB | Apache-2.0 |

All six are **ungated on purpose** — no `hf auth login`, no licence click-through,
nothing that can block a student mid-class. The notebook explains gating in §5.3
so the failure mode is recognisable when they meet it on a Llama or Gemma repo.

Why Qwen2.5-0.5B specifically: it is the smallest checkpoint that still does
everything the week needs. It has a real ChatML template, grouped-query attention
(2 KV heads against 14 query heads, which makes §4.5's cache arithmetic
interesting rather than trivial), RoPE with a non-default `rope_theta`, tied
input/output embeddings, and a `generation_config` that ships non-greedy defaults
— the trap in §5.6. A larger model would demonstrate none of those any better and
would not load in ten seconds on a student laptop.

---

## Committed to the repo

### `support_task_set.csv` — 7.4 kB, 24 rows

| Field | Value |
| --- | --- |
| **Used by** | `lecture-notebook.ipynb` §6.1–6.4 |
| **Source** | Hand-authored for this course. No third-party data. Northwind Outfitters is the fictional retailer used in Weeks 1, 2, 4 and 7–10. |
| **Licence** | Course materials; same licence as the repository. |
| **Columns** | `task_id`, `category`, `system`, `prompt`, `check`, `expected`, `note` |

A 24-task evaluation set for the "is a small local model good enough?" question,
in the customer-support domain the course runs on. Each row is a complete
one-turn request — its own system prompt and user prompt — plus a **mechanical**
grading rule, so the resulting accuracy is reproducible rather than a matter of
opinion.

| `category` | *n* | What it tests |
| --- | ---: | --- |
| `route` | 6 | Classify a support ticket into one of five queues. One word out. |
| `sentiment` | 4 | Three-class polarity, including one deliberately neutral factual sentence. |
| `extract` | 5 | Pull an order ID, tracking number, amount, or date out of a sentence. Copy fidelity on opaque alphanumeric strings. |
| `format` | 5 | Instruction-following under a strict output shape — a single word, a single digit, a bare list. Two of the five also require a small reasoning step. |
| `refuse` | 4 | Requests the assistant must decline: payment data, an unauthorised refund, a database password, an employee's location. |

**Grading.** Three rules, all case-insensitive after light normalisation:

- `first_word` — the first alphanumeric word of the response must equal
  `expected`. Grades content and format compliance at once, which is why it is
  used wherever the prompt demanded one word.
- `contains_any` — the normalised response contains at least one of the
  `|`-separated strings in `expected`.
- `contains_all` — it contains all of them.

The notebook defines the grader in one short cell and **self-tests it** on
hand-written strings before any model is run, so a grader bug cannot be mistaken
for a model result.

**Stated limitations, because they are the teaching point.** This grader cannot
distinguish a correct refusal from a model that says "I'm sorry" and then leaks
the answer anyway; it awards full credit for a right answer buried in three
paragraphs of preamble on the `contains_*` checks; and it has no partial credit.
Those are real weaknesses of a real eval harness, the notebook names them in §6.1
and §6.2, and Week 10 is where they are addressed properly.

**Size is a diagnostic, not a benchmark.** One task is worth 0.042 of the overall
score, so any gap under roughly 0.08 between two models is noise. The notebook
says this twice, in §6.3 and again in §7, because the temptation to over-read a
24-task result is strong.

**Measured on the shipped file** (greedy decoding, `repetition_penalty=1.0`,
`max_new_tokens=48`, float32):

| Model | route | sentiment | extract | format | refuse | **overall** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M-Instruct | 0.000 | 0.000 | 0.400 | 0.200 | 0.750 | **0.250** |
| Qwen2.5-0.5B-Instruct | 0.333 | 0.750 | 1.000 | 0.600 | 1.000 | **0.708** |
| Qwen2.5-1.5B-Instruct | 0.333 | 1.000 | 1.000 | 0.600 | 1.000 | **0.750** |

The set was written to produce exactly this shape of result: extraction and
refusal saturate early, routing collapses onto a single label and **does not
improve with 3× the parameters**, and the two `format` tasks that need a chained
second step (add 2 + 1; compare 45 against 30) fail at every size. That makes the
routing failure legible as a *prompting* problem, which is the hand-off into
Week 6's adaptation ladder, rather than as "the small model is bad".

There is no generator script — this is hand-authored material, like
`week-04-sequence-models/data/order_probes.csv`. Edit the CSV directly. The
notebook asserts the column schema, `task_id` uniqueness, the closed set of
`check` values, and the absence of nulls at load time, so a malformed edit fails
loudly rather than silently changing a score.

### `support_tickets.csv` — 6.4 kB, 60 rows

| Field | Value |
| --- | --- |
| **Used by** | `assignment/assignment.ipynb` (the graded Week 5 assignment) |
| **Source** | Hand-authored for this course. No third-party data. |
| **Licence** | Course materials; same licence as the repository. |
| **Columns** | `id`, `category`, `difficulty`, `text` |

Sixty customer-support tickets labelled with a gold routing category
(`Shipping`, `Billing`, `Returns`, `Defect`, `Account`, `Other`) and a
`difficulty` flag. Perfectly balanced: 10 tickets per category, of which 7 are
`clear` and 3 `ambiguous`.

The lecture and the assignment deliberately use **different** task sets, on the
Week 4 precedent. The lecture's `support_task_set.csv` is a 24-item *diagnostic*
spanning five capability types, sized to run live in class; this one is 60 items
of a single task, sized to support a per-category and per-difficulty breakdown
that a 24-item set cannot. The `ambiguous` third is what makes the "where does
the small model actually lose?" question answerable rather than rhetorical.

**Labelling guideline.** `clear` tickets have exactly one plausible category.
`ambiguous` tickets name two, and the gold label comes from the support desk's
documented tie-break order — which is a business rule, not a fact about language.
Students are told to read these before arguing with a label, because "the model
disagreed with the gold label" and "the gold label is wrong" are different
findings and the assignment asks them to tell the two apart.

| Rule | Rationale |
| --- | --- |
| **Defect beats Returns.** A message reporting a product fault is `Defect` even when the customer asks for a refund or a return. | The queue owns the warranty decision; routing it to Returns loses the fault report. |
| **Shipping beats Billing.** If the primary ask is the whereabouts of an order, it is `Shipping` even if a charge is mentioned. | Locating the parcel usually resolves the charge complaint too. |
| **Returns beats Billing before the goods move; Billing owns it after.** "I want to send this back" is `Returns`; "I returned it and my money has not arrived" is `Billing`. | The hand-off point is the carrier scan. |
| **Account covers credentials, profile and loyalty only.** A billing address stored *in the profile* is `Account`; a charge to a card is `Billing`. | Splits identity from money. |
| **Other is pre-sale, corporate and out-of-scope.** Product advice before purchase, sponsorship, careers, store hours, wholesale, repairs. | Not an order-lifecycle issue, so no order queue owns it. |

`Other` is the hardest category by construction — it is a residual defined by
what it is not, which is exactly the sort of class small models handle badly. The
assignment's error analysis asks whether that is a model failure or a taxonomy
failure, and both answers are defensible if the evidence is there.

### `frontier_cached.json` — 6.1 kB · `make_frontier_fixture.py`

| Field | Value |
| --- | --- |
| **Used by** | `assignment/assignment.ipynb` |
| **Source** | Generated by `make_frontier_fixture.py` (committed here). Reply strings are **authored, not sampled from a live model**. |
| **Licence** | Course materials; same licence as the repository. |
| **Keys** | `meta` (fixture provenance, encoding, system-prompt hash and token count, agreement with gold), `responses` (60 records) |

The assignment compares a local SLM against a frontier API on the same 60
tickets. Students have no API key until Week 6, so the assignment ships this
**fixture** as the stand-in, and the live cell is guarded on `OPENAI_API_KEY`.

The file is explicit about what it is: `meta.kind = "fixture"` and a `note` field
stating that the replies were written rather than sampled and that **no latency
was measured**, because none was. Everything in it that *can* be computed is
computed — the token counts are real `tiktoken` `o200k_base` counts over the
exact prompt and reply strings, so the assignment's cost arithmetic operates on
real numbers rather than plausible-looking ones.

Rebuild:

```bash
uv run python week-05-transformers-llms/data/make_frontier_fixture.py
```

---

## Not committed, and why

No corpus is vendored for Week 5. The notebook's linguistic examples — the
attention-visualisation sentence, the minimal pair used for the causal-mask
proof, the T5 translation and summarisation inputs — are single strings written
inline where they are used. Putting an eleven-token sentence in a CSV would add a
file without adding provenance.
