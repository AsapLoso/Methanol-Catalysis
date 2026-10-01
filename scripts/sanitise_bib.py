"""Strip Zotero `file = {...}` fields out of references.bib.

Better BibTeX auto-export emits an absolute local path per attachment. On this
machine those paths embed the Windows profile leaf, the OneDrive organisation
name and the folder layout of a separate PRIVATE project-management repo. The
repo is public, so any such field that reaches `main` is a disclosure.

The durable fix is to uncheck the `file` field in the Better BibTeX export map
(`.agents/runbooks/zotero-bib-export.md`). This script is the fallback for
exports produced before that change, and for any re-export that reintroduces
the field. It is therefore idempotent by construction: a run that finds nothing
to remove rewrites the file byte-for-byte identically.

Run with:  pixi run sanitise-bib
"""

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Resolve against the repo root so the script works from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
BIB_PATH = REPO_ROOT / "references.bib"

# One `file = {…}` field, anchored to the start of a line and indented like the
# other fields Better BibTeX emits. The value is brace-delimited and the paths
# it contains escape colons and backslashes (`C\:\\Users\\...`), but they never
# contain an unbalanced brace, so a non-greedy match to the first `}` is safe.
# `re.MULTILINE` is required: without it `^` would only ever match string start.
FILE_FIELD_RE = re.compile(
    r"^[ \t]*file[ \t]*=[ \t]*\{[^{}]*\}[ \t]*\r?\n?", re.MULTILINE
)


def main() -> int:
    if not BIB_PATH.exists():
        print(f"ERROR: {BIB_PATH} does not exist")
        return 1

    # newline="" keeps the on-disk line endings untouched, so the rewrite is a
    # pure deletion of the matched lines and nothing else drifts.
    original = BIB_PATH.read_text(encoding="utf-8")
    sanitised, removed = FILE_FIELD_RE.subn("", original)

    if sanitised != original:
        with BIB_PATH.open("w", encoding="utf-8", newline="") as handle:
            handle.write(sanitised)

    print(f"references.bib: removed {removed} 'file' field(s)")
    print(f"remaining 'file' fields: {len(FILE_FIELD_RE.findall(sanitised))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
