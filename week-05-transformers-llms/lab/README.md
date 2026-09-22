# Week 5 Lab — One model of each shape

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prof-tcsmith/6564F26-DATA/blob/main/week-05-transformers-llms/lab/lab-slm-by-shape.ipynb)

**What it is.** A 40-minute hands-on that follows the slide *Today's small language
models, by shape*. One model of each shape, loaded and used for the job it is built for:

| Station | Shape | Model | You do |
| --- | --- | --- | --- |
| 1 | encoder-only | `bert-base-uncased` | fill a blank; prove it read the words *after* the blank |
| 2 | encoder-only | `all-MiniLM-L6-v2` | embed five tickets; read the similarity table; find the odd one out |
| 3 | decoder-only | `Qwen2.5-0.5B` and `-Instruct` | base continues, instruct answers; temperature 0 / 0.7 / 1.5; tokens per second on your machine |
| 4 | encoder–decoder | `t5-small` | translate and summarise with the task named in the prompt; try a language it never saw |
| 5 | all three | — | print what each model calls itself and count its learned numbers |

Then a results sheet and four discussion questions.

**Not marked. Nothing to submit.** Bring the results sheet to the discussion.

## Two ways to run it

### A. Google Colab (recommended for the lab)

1. Click the badge above. You need a Google account. (Or open
   [colab.research.google.com](https://colab.research.google.com), choose *GitHub*,
   and paste the notebook's URL.)
2. *Runtime → Change runtime type → T4 GPU → Save.* If Colab offers no GPU right
   now, stay on CPU: everything runs, Station 3 is just slower.
3. *Runtime → Run all*, or run cell by cell. The first two cells install two
   libraries and download the five models, about a minute on Colab's connection.
   This repeats in each new session because Colab's disk is wiped.
4. To keep your filled-in results sheet: *File → Save a copy in Drive*.

### B. Your own machine

1. Set up once with the [getting-started guide](../../docs/getting-started-local.md):
   Miniconda, VS Code, and your GPU if you have one.
2. Download `lab-slm-by-shape.ipynb` (*Raw → Save*), open it in VS Code, and
   select the `Python (ism6564)` kernel.
3. Run the first two cells **at home** before class. The five models are about
   2.7 GB; on classroom Wi-Fi with thirty laptops that download does not finish.

## Timing in class

| Minutes | |
| --- | --- |
| 0–5 | open the notebook, pick the runtime, run Setup and Fetch (instant if you fetched at home) |
| 5–12 | Station 1 |
| 12–20 | Station 2 |
| 20–30 | Station 3 |
| 30–36 | Station 4 |
| 36–40 | Station 5 and the results sheet |

## If something breaks

- **No GPU on Colab** → CPU works; expect a few tokens per second at Station 3.
- **`ModuleNotFoundError` locally** → the kernel is not the `ism6564` environment; re-select it (top right in VS Code).
- **Downloads stall** → run the Fetch cell again; downloads resume where they stopped.
- **Out of memory at Station 3** → restart the kernel and run Setup, Fetch and Station 3 only. The cell frees the base model after use.
- **Windows warning about symlinks** from Hugging Face → harmless; the guide shows how to silence it.

## What it connects to

Slides 25–32 (the three shapes) and 33–37 (a language model on your laptop), and
the companion pages for the same slides. The practice assignment takes the decoder
apart. Assignment 5 measures three decoders on your machine. Week 7's retrieval
system runs on Station 2's model.
