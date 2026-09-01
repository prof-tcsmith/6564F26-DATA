#!/usr/bin/env bash
# ISM 6564 — one-shot environment setup.
#
#   macOS   : run in Terminal        ->  ./setup.sh
#   Windows : run in **Git Bash**    ->  ./setup.sh
#             (Git Bash comes with Git for Windows — gitforwindows.org.
#              You already have it if you were able to `git clone`.)
#
# Safe to re-run at any time; it only does work that still needs doing.
set -euo pipefail

say() { printf '\n== %s\n' "$*"; }

case "$(uname -s)" in
  Darwin*)              OS=mac ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *)                    OS=linux ;;
esac

# Always operate on the repo this script lives in.
cd "$(dirname "$0")"
if [ ! -f pyproject.toml ]; then
  echo "ERROR: run this from inside your 6564F26-DATA clone." >&2
  exit 1
fi

# --- 1. uv — the tool that installs Python and every course package ----------
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (one time)"
  if [ "$OS" = windows ]; then
    # uv's Windows installer is a PowerShell script; call it from Git Bash.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
      "irm https://astral.sh/uv/install.ps1 | iex"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi
# The installer configures future shells; make uv visible in THIS one too.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is installed but this window cannot see it yet." >&2
  echo "Close this window, open a new Terminal / Git Bash, and run ./setup.sh again." >&2
  exit 1
fi
say "Using $(uv --version)"

# --- 2. Python 3.12 + all course packages, exact pinned versions -> ./.venv --
say "Installing the course environment (first run 3-5 minutes; later runs take seconds)"
uv sync

# --- 3. Language data the early weeks' notebooks require ---------------------
say "Downloading spaCy and NLTK language data"
uv run python -m spacy download en_core_web_sm
uv run python -m nltk.downloader punkt punkt_tab stopwords wordnet averaged_perceptron_tagger_eng

# --- 4. Verify ---------------------------------------------------------------
say "Verifying the install"
uv run python -c "import torch, transformers, sklearn, tiktoken, spacy, nltk, gensim; print('   OK — torch', torch.__version__)"

say "Done. When you open a notebook in VS Code, pick the interpreter at ./.venv"
