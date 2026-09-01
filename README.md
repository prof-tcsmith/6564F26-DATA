# ISM 6564 — Text Analytics (Fall 2026) · Course Data

Data and Python environment for ISM 6564 at USF.

This repository holds the **datasets** the course's notebooks and assignments
use, plus the pinned Python environment to run them in. Weeks appear here as
they are released — one folder per released week, each containing only that
week's `data/`.

Everything else — slides, lecture notebooks, readings, assignments, practice
work, quizzes, and submissions — lives in **Canvas**. If you are looking for
course material and it is not in Canvas, it has not been released yet.

> **Before your first submission**, read §7 of the syllabus (Working with AI,
> in Canvas) and the AI Usage Report template in
> [templates/ai-usage-report.md](templates/ai-usage-report.md). The report is
> required with every assignment.

---

## Setup

The course notebooks you download from Canvas are meant to be **run from a
clone of this repository**: they read their data from the week folders here
(with a URL fallback to this repo on GitHub, so they also work in Colab), and
they expect this repo's `uv` environment.

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: brew install uv
# Windows (PowerShell): powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone this repository and enter it
git clone https://github.com/prof-tcsmith/6564F26-DATA.git
cd 6564F26-DATA

# 3. Install Python 3.12 + every package the course uses, into ./.venv
uv sync

# 4. Download the language data the early labs need
uv run python -m spacy download en_core_web_sm
uv run python -m nltk.downloader punkt punkt_tab stopwords wordnet averaged_perceptron_tagger_eng

# 5. Verify
uv run python -c "import torch, transformers, sklearn, tiktoken; print('torch', torch.__version__, '| MPS', torch.backends.mps.is_available())"
```

First run takes ~3–5 minutes (PyTorch is large); later syncs take seconds.
**Step 4 is not optional** — Week 1's notebook raises an error on a fresh
install without it.

**Working on a notebook from Canvas:** save it into the matching week folder
of this clone (for example `week-02-vectors-similarity/`), open it in VS Code,
and pick the `.venv` interpreter this repo's `uv sync` created. Run
`git pull` at the start of each week to receive newly released data.

## What's here

| Path | Contents |
| --- | --- |
| `week-NN-…/data/` | That week's datasets, with a `README.md` describing each file and its provenance |
| `templates/ai-usage-report.md` | The AI Usage Report you submit with every assignment |
| `pyproject.toml`, `uv.lock`, `.python-version` | The pinned course environment (`uv sync` builds it) |
| `.vscode/` | Editor settings so the notebook kernel picker finds `./.venv` |

Data files load **local-first with a URL fallback**: run a notebook inside
this clone and it reads from disk; run it elsewhere (e.g. Colab) and it
fetches the same files from this repository's `main` branch.

## Questions

Email [smith515@usf.edu](mailto:smith515@usf.edu) with a subject line starting `ISM 6564`. Weekday
replies within 24 hours, weekends within 48.
