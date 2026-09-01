"""Regenerate the two derived corpora in this folder from NLTK's distribution.

Both source corpora are public domain. Nothing here is scraped, and nothing
here needs a key. Run from the repo root:

    uv run python week-01-text-tokenization/data/build_data.py

Requires the NLTK data packages `gutenberg` and `udhr2`:

    uv run python -m nltk.downloader gutenberg udhr2

Outputs
-------
moby_dick.txt        Melville, *Moby-Dick* (1851). Public domain. Line endings
                     normalised to \\n; otherwise byte-for-byte the NLTK text.
udhr_parallel.json   The Universal Declaration of Human Rights in five
                     languages, from the Unicode UDHR project via NLTK's
                     `udhr2` (UTF-8) corpus. Public domain.
"""

from __future__ import annotations

import json
from pathlib import Path

from nltk.corpus import gutenberg, udhr2

HERE = Path(__file__).parent

# Five languages chosen to span the axes that matter for tokenization:
# Latin-script analytic (English), Latin-script compounding (German),
# Latin-script inflecting (Spanish), Devanagari abugida (Hindi), and a
# logographic script with no whitespace word boundaries (Chinese).
LANGUAGES = [
    ("English", "eng", "eng.txt", "Latin"),
    ("German", "deu", "deu.txt", "Latin"),
    ("Spanish", "spa", "spa.txt", "Latin"),
    ("Hindi", "hin", "hin.txt", "Devanagari"),
    ("Chinese (Simplified)", "cmn-Hans", "cmn_hans.txt", "Han (Simplified)"),
]


def clean(text: str) -> str:
    """Normalise line endings and strip the fixed left indent NLTK ships."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def main() -> None:
    moby = gutenberg.raw("melville-moby_dick.txt").replace("\r\n", "\n")
    (HERE / "moby_dick.txt").write_text(moby, encoding="utf-8")
    print(f"moby_dick.txt        {len(moby):>9,} chars")

    parallel = {}
    for name, code, fileid, script in LANGUAGES:
        text = clean(udhr2.raw(fileid))
        parallel[name] = {"code": code, "script": script, "source_file": fileid, "text": text}
        print(f"  {name:<22} {len(text):>6,} chars")

    out = HERE / "udhr_parallel.json"
    out.write_text(json.dumps(parallel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"udhr_parallel.json   {out.stat().st_size:>9,} bytes")


if __name__ == "__main__":
    main()
