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

## Setup — three steps

The course notebooks you download from Canvas are meant to be **run from a
clone of this repository**: they read their data from the week folders here,
and they expect this repo's pinned Python environment.

**Step 0 — get a terminal with `git` and `bash`:**

- **macOS:** nothing to install. Open **Terminal**. (The first time you run
  `git`, macOS may offer to install its command-line tools — accept and wait.)
- **Windows:** install **Git for Windows** from
  [gitforwindows.org](https://gitforwindows.org/) with the default options,
  then do everything below in **Git Bash** (it is installed alongside Git and
  provides the `bash` that the setup script needs).

**Step 1 — clone this repository:**

```bash
git clone https://github.com/prof-tcsmith/6564F26-DATA.git
cd 6564F26-DATA
```

**Step 2 — run the setup script:**

```bash
./setup.sh
```

That's it. The script installs [`uv`](https://docs.astral.sh/uv/) if you don't
have it, builds the exact pinned course environment into `./.venv` (Python
3.12 and every package, first run 3–5 minutes), downloads the spaCy/NLTK
language data the early notebooks require, and verifies the result. It is
**safe to re-run at any time** — if anything ever looks broken, delete the
`.venv` folder and run `./setup.sh` again.

> If the script says uv was installed but cannot be seen yet, close the
> window, open a fresh Terminal / Git Bash, and run `./setup.sh` once more.

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
