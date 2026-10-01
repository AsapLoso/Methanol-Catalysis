"""Cross-check this repo's references.bib against the private PM citation manifest.

REPORT ONLY. This script NEVER modifies references.bib. It joins the two
sources ON DOI and reports the gaps in both directions. It never falls back to
title matching: a title match is a guess, and a guess in a citation audit is
worse than a reported "no DOI".

Both input locations are resolved through the stable domain IDs declared in
BEP_SOURCE_MAP.md -> .agents/local_paths.json, so this script contains no
absolute local path and no knowledge of the private repository's layout.

Run with:  pixi run python scripts/bib_crosscheck.py
"""

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
BIB_PATH = REPO_ROOT / "references.bib"
LOCAL_PATHS = REPO_ROOT / ".agents" / "local_paths.json"
REPORT_PATH = REPO_ROOT / "bib_crosscheck_report.md"

# --- domain-ID resolution ----------------------------------------------------


def resolve(domains: dict, root: Path, domain_id: str):
    """Resolve a domain ID from .agents/local_paths.json to a concrete path."""
    if domain_id not in domains:
        raise SystemExit(
            f"FATAL: domain ID '{domain_id}' is not in {LOCAL_PATHS}. "
            f"BEP_SOURCE_MAP.md and .agents/local_paths.json are out of sync."
        )
    value = domains[domain_id]
    return [root / value] if isinstance(value, str) else [root / v for v in value]


def load_local_paths():
    if not LOCAL_PATHS.exists():
        raise SystemExit(
            f"FATAL: {LOCAL_PATHS} not found. It is gitignored and local-only; "
            f"see BEP_SOURCE_MAP.md for the domain IDs it is expected to define."
        )
    data = json.loads(LOCAL_PATHS.read_text(encoding="utf-8"))
    return Path(data["pm_root"]), data["domains"]


# --- DOI normalisation -------------------------------------------------------

DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "doi:",
    "doi ",
)


def normalise_doi(raw):
    """Lower-case, strip whitespace, strip any known DOI resolver prefix.

    Returns None for an absent/empty DOI. Never returns a guess.
    """
    if raw is None:
        return None
    value = str(raw).strip().lower().strip('"').strip("{}")
    # BibTeX/Wikipedia sometimes carry a trailing footnote or comma.
    value = value.split(",")[0].strip()
    for prefix in DOI_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    value = value.rstrip(". ;")
    return value or None


# --- BibTeX parsing (brace-balanced, no third-party dependency) --------------

# re.MULTILINE is required: without it `^` only ever matches string start, so
# only the first entry of a 48-entry file would ever be found.
ENTRY_START = re.compile(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
FIELD_START = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*")


def parse_bib(text: str):
    """Return a list of {key, type, doi, year, title} dicts.

    Scans @type{key, ... } by tracking brace depth, so a DOI or abstract
    containing a '}' cannot desynchronise the parse.
    """
    entries = []
    pos = 0
    length = len(text)
    while True:
        at = text.find("@", pos)
        if at == -1:
            break
        header = ENTRY_START.match(text, at)
        if not header:
            pos = at + 1
            continue
        depth = 1
        i = header.end()
        while i < length and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[header.end() : i - 1]
        entries.append(
            {
                "type": header.group(1).lower(),
                "key": header.group(2),
                "fields": _split_fields(body),
            }
        )
        pos = i
    return entries


def _split_fields(body: str):
    """Extract `name = value` pairs with a single forward scan.

    Values may be brace-delimited, quote-delimited, or bare. A brace-delimited
    value is consumed by brace matching, so a value that itself contains
    `year = {` cannot be mistaken for a new field, and a `}` inside a quoted
    string cannot desynchronise the scan.
    """
    fields = {}
    pos = 0
    length = len(body)
    while pos < length:
        match = FIELD_START.search(body, pos)
        if match is None:
            break
        j = match.end()
        if j >= length:
            break
        opener = body[j]
        if opener == "{":
            depth = 0
            k = j
            while k < length:
                char = body[k]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            fields[match.group(1).lower()] = body[j + 1 : k]
            pos = k + 1
        elif opener == '"':
            k = body.find('"', j + 1)
            if k == -1:
                break
            fields[match.group(1).lower()] = body[j + 1 : k]
            pos = k + 1
        else:
            k = j
            while k < length and body[k] not in "\r\n":
                k += 1
            # A bare value runs to end of line, and the field separator is part of
            # that line: `year = 2008,` scanned verbatim yields "2008,", which then
            # fails every downstream string comparison against a manifest "2008".
            # Strip surrounding whitespace, then the one trailing separator comma.
            fields[match.group(1).lower()] = body[j:k].strip().rstrip(",").strip()
            pos = k
    return fields


# --- _consensus.md presence --------------------------------------------------

CONSENSUS_NAME = re.compile(r"^(?P<stem>.+?)_consensus\.md$", re.IGNORECASE)
DOI_IN_TEXT = re.compile(r"10\.\d{4,9}/[^\s`\"'<>)\]]+", re.IGNORECASE)


def build_consensus_index(directory: Path):
    """Map normalised DOI -> filename for every _consensus.md, plus name+year set.

    Two independent, explicitly-labelled lookups, because neither is complete:
      * DOI found in the report body. Measured on 2026-09-30 against the corpus
        resolved through domain `consensus_reports`: 36 `_consensus.md` files
        exist and 33 of them print a DOI. Re-measure before repeating that pair
        of numbers -- the corpus lives in the private repository and moves.
      * filename stem starting with a lower-cased first-author surname and
        ending with the 4-digit year (the remaining few; 34 of 36 stems carry
        such a year, so this lookup is not the limiting one).
    """
    by_doi = {}
    by_name_year = {}
    if not directory.exists():
        return by_doi, by_name_year
    for path in sorted(directory.glob("*_consensus.md")):
        stem = CONSENSUS_NAME.match(path.name)
        stem = stem.group("stem") if stem else path.stem
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in DOI_IN_TEXT.findall(text):
            doi = normalise_doi(raw)
            if doi:
                by_doi.setdefault(doi, path.name)
        years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", stem)
        lowered = re.sub(r"[^a-z]", "", stem.lower())
        for year in years:
            by_name_year.setdefault((lowered, year), path.name)
    return by_doi, by_name_year


def lookup_consensus(doi, bibkey, year, by_doi, by_name_year):
    """Return (filename_or_empty, how) for a DOI-matched bib entry.

    Two named lookups, in order: the DOI printed inside the report body, then a
    filename heuristic (surname + year). Neither is complete, so the caller
    reports which one hit rather than presenting a bare yes/no as proof.
    """
    hit = by_doi.get(doi)
    if hit:
        return hit, "DOI in body"
    key = (re.sub(r"[^a-z]", "", str(bibkey).lower()), str(year))
    hit = by_name_year.get(key)
    if hit:
        return hit, "filename surname+year"
    return "", ""


def surname_of(first_author: str) -> str:
    token = re.sub(r"[^A-Za-z\- ]", " ", str(first_author)).strip()
    if not token:
        return ""
    return token.split()[0].lower()


# --- report ------------------------------------------------------------------


def md_escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def main() -> int:
    if not BIB_PATH.exists():
        print(f"FATAL: {BIB_PATH} not found")
        return 1

    pm_root, domains = load_local_paths()
    manifest_path = resolve(domains, pm_root, "citations_manifest")[0]
    consensus_dirs = resolve(domains, pm_root, "consensus_reports")

    if not manifest_path.exists():
        print(
            f"FATAL: manifest not found at resolved path for domain "
            f"'citations_manifest': {manifest_path}"
        )
        return 1

    raw_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8", errors="replace")
    )
    manifest = []
    for prop_name, value in raw_manifest.items():
        if not isinstance(value, dict):
            continue
        manifest.append(
            {
                "ref_num": value.get("ref_num", prop_name),
                "doi": normalise_doi(value.get("doi")),
                "doi_raw": (value.get("doi") or "").strip(),
                "year": str(value.get("year") or ""),
                "first_author": str(value.get("first_author") or ""),
            }
        )
    manifest.sort(key=lambda r: str(r["ref_num"]).zfill(3))

    bib_entries = parse_bib(BIB_PATH.read_text(encoding="utf-8", errors="replace"))
    for entry in bib_entries:
        fields = entry["fields"]
        entry["doi"] = normalise_doi(fields.get("doi"))
        entry["year"] = str(fields.get("year") or "").strip()
        entry["title"] = re.sub(r"\s+", " ", str(fields.get("title") or "")).strip()

    consensus_by_doi = {}
    consensus_by_name_year = {}
    consensus_dirs_found = []
    for directory in consensus_dirs:
        if directory.exists():
            consensus_dirs_found.append(directory)
            a, b = build_consensus_index(directory)
            consensus_by_doi.update(a)
            consensus_by_name_year.update(b)

    # --- join on DOI ---------------------------------------------------------
    manifest_by_doi = {r["doi"]: r for r in manifest if r["doi"]}
    bib_by_doi = {e["doi"]: e for e in bib_entries if e["doi"]}

    matched, bib_only, manifest_only = [], [], []
    for entry in bib_entries:
        if entry["doi"] and entry["doi"] in manifest_by_doi:
            matched.append((entry, manifest_by_doi[entry["doi"]]))
        else:
            bib_only.append(entry)
    for record in manifest:
        if record["doi"] and record["doi"] in bib_by_doi:
            continue
        if not record["doi"]:
            manifest_only.append((record, "no DOI in manifest"))
        else:
            manifest_only.append((record, "no bib entry with this DOI"))

    lines = []
    lines.append("# references.bib x citations_manifest cross-check")
    lines.append("")
    lines.append(
        "Generated by `scripts/bib_crosscheck.py` (report only; "
        "`references.bib` is not modified by this script)."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("| Side | Resolved via | Count |")
    lines.append("|---|---|---|")
    lines.append(
        f"| `references.bib` | local file | {len(bib_entries)} entries "
        f"({sum(1 for e in bib_entries if e['doi'])} with a DOI) |"
    )
    lines.append(
        f"| citation manifest | domain `citations_manifest` | "
        f"{len(manifest)} records "
        f"({sum(1 for r in manifest if r['doi'])} with a DOI) |"
    )
    lines.append("")
    lines.append(
        "Join key: **DOI**, case-insensitive, with or without a "
        "`https://doi.org/` prefix. Title matching is deliberately "
        "not used: a title match is a guess. Records with no DOI on "
        "either side are reported as unmatched, never fuzzy-matched."
    )
    lines.append("")
    lines.append(
        f"Matched on DOI: **{len(matched)}**. Bib entries with no "
        f"manifest record: **{len(bib_only)}**. Manifest records with "
        f"no bib key: **{len(manifest_only)}**."
    )
    lines.append("")

    # --- matched table -------------------------------------------------------
    lines.append("## 1. DOI matches, and whether a consensus report exists")
    lines.append("")
    lines.append(
        "`consensus?` is resolved by DOI found inside the "
        "`_consensus.md` body, then by a filename heuristic "
        "(first-author surname + year). `by` names which lookup hit; "
        "an empty `by` means neither matched."
    )
    lines.append("")
    lines.append("| ref | bibkey | year | first author | DOI | consensus? | by |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry, record in sorted(matched, key=lambda p: str(p[1]["ref_num"]).zfill(3)):
        name = md_escape(record["first_author"])
        found, how = lookup_consensus(
            entry["doi"],
            entry["key"],
            record["year"],
            consensus_by_doi,
            consensus_by_name_year,
        )
        lines.append(
            f"| {md_escape(record['ref_num'])} | `{entry['key']}` | "
            f"{md_escape(record['year'])} | {name} | `{entry['doi']}` | "
            f"{('yes: `' + found + '`') if found else '**no**'} | {how or '-'} |"
        )
    lines.append("")

    # --- bib only ------------------------------------------------------------
    lines.append("## 2. Bib entries with no manifest record")
    lines.append("")
    lines.append(
        "These keys exist in `references.bib` but have no DOI-matched "
        "record in the manifest. That is not automatically a defect: "
        "the manifest is the reference list of the *source* thesis "
        "document, and this repository's bibliography may legitimately "
        "be wider."
    )
    lines.append("")
    if bib_only:
        lines.append("| bibkey | year | DOI | note |")
        lines.append("|---|---|---|---|")
        for entry in bib_only:
            note = (
                "**no DOI in bib entry - cannot join**"
                if not entry["doi"]
                else "DOI absent from manifest"
            )
            lines.append(
                f"| `{entry['key']}` | {md_escape(entry['year'])} | "
                f"{('`' + entry['doi'] + '`') if entry['doi'] else '-'} | {note} |"
            )
    else:
        lines.append("None.")
    lines.append("")

    # --- manifest only -------------------------------------------------------
    lines.append("## 3. Manifest records with no bib key")
    lines.append("")
    lines.append(
        "These are referenced by the source document but have no "
        "DOI-matched entry in this repository's `references.bib`."
    )
    lines.append("")
    if manifest_only:
        lines.append("| ref | year | first author | DOI | reason |")
        lines.append("|---|---|---|---|---|")
        for record, reason in manifest_only:
            doi_cell = f"`{record['doi']}`" if record["doi"] else "**none recorded**"
            lines.append(
                f"| {md_escape(record['ref_num'])} | "
                f"{md_escape(record['year'])} | "
                f"{md_escape(record['first_author'])} | {doi_cell} | {reason} |"
            )
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 4. Do not read section 3 as 'nothing exists'")
    lines.append("")
    lines.append(
        "A manifest record with no bib key means the *bibliography* has "
        "no entry for it. It does not mean the reference was never "
        "analysed. The following unmatched records DO have a "
        "`_consensus.md`, located by the `ref_NN_` filename convention "
        "— that is a filename match, **not** a DOI match, so it is "
        "reported separately and separately caveated. Re-verify by "
        "DOI inside the report before relying on it."
    )
    lines.append("")
    # Section 4 globs the consensus reports directly instead of going through
    # `consensus_by_*`, so it needs the resolved paths as a list. Previously it
    # used the loop variable `directory` leaked from the earlier
    # `for directory in consensus_dirs:` loop: correct only while that list
    # happened to hold exactly one existing directory (NameError on an empty
    # list, and silently only the LAST directory on a multi-entry list, while
    # the DOI re-read below walked every one of them -- two halves of one
    # section disagreeing). Resolve explicitly, and aggregate across all of
    # them.
    consensus_paths = []
    for resolved_dir in consensus_dirs_found:
        consensus_paths.extend(sorted(resolved_dir.glob("*_consensus.md")))
    consensus_path_by_name = {}
    for candidate in consensus_paths:
        consensus_path_by_name.setdefault(candidate.name, candidate)

    found_by_name = []
    for record, _reason in manifest_only:
        token = re.sub(r"[^0-9]", "", str(record["ref_num"]))
        surname = re.sub(r"[^a-z]", "", surname_of(record["first_author"]))
        year = str(record["year"])
        for path in consensus_paths:
            stem = path.name.lower()
            if f"ref_{token}_" in stem or (
                surname and surname in re.sub(r"[^a-z]", "", stem) and year in stem
            ):
                found_by_name.append((record, path.name))
                break
    if found_by_name:
        lines.append(
            "| ref | first author | year | consensus report (filename match) | DOI verified inside? |"
        )
        lines.append("|---|---|---|---|---|")
        for record, name in found_by_name:
            body = consensus_path_by_name.get(name)
            text = ""
            if body is not None:
                try:
                    text = body.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
            doi_ok = (
                "no"
                if not record["doi"]
                else ("yes" if record["doi"] in text.lower() else "no")
            )
            lines.append(
                f"| {md_escape(record['ref_num'])} | "
                f"{md_escape(record['first_author'])} | "
                f"{md_escape(record['year'])} | `{name}` | {doi_ok} |"
            )
    elif not consensus_paths:
        lines.append(
            "No `*_consensus.md` file was found under the resolved "
            "`consensus_reports` domain, so no filename match was "
            "attempted. That is a missing input, not a clean result."
        )
    else:
        lines.append(
            "No unmatched manifest record was located by the "
            "`ref_NN_` / surname+year filename convention."
        )
    lines.append("")

    lines.append("## Notes and limits")
    lines.append("")
    lines.append(
        "- A `consensus?` value of `**no**` means neither the DOI-in-body "
        "nor the filename heuristic found a report. It does not prove "
        "the report is absent; the heuristics are documented above and "
        "neither is complete."
    )
    lines.append(
        "- No data here is inferred. A record whose DOI is missing on "
        "either side is reported as unmatched rather than matched on "
        "title or author string."
    )
    lines.append(
        "- This report is local output and is `.gitignore`d. It names "
        "neither the private repository's path nor any person."
    )
    lines.append("")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")

    # --- stdout table --------------------------------------------------------
    out = []
    out.append(
        f"references.bib: {len(bib_entries)} entries "
        f"({sum(1 for e in bib_entries if e['doi'])} with DOI)"
    )
    out.append(
        f"citations_manifest: {len(manifest)} records "
        f"({sum(1 for r in manifest if r['doi'])} with DOI)"
    )
    out.append(
        f"matched on DOI: {len(matched)} | bib-only: {len(bib_only)} "
        f"| manifest-only: {len(manifest_only)}"
    )
    out.append("")
    out.append("| ref | bibkey | year | first author | DOI | consensus? | by |")
    out.append("|---|---|---|---|---|---|---|")
    for entry, record in sorted(matched, key=lambda p: str(p[1]["ref_num"]).zfill(3)):
        found, how = lookup_consensus(
            entry["doi"],
            entry["key"],
            record["year"],
            consensus_by_doi,
            consensus_by_name_year,
        )
        out.append(
            f"| {record['ref_num']} | {entry['key']} | {record['year']} | "
            f"{md_escape(record['first_author'])} | {entry['doi']} | "
            f"{(found or 'NO')} | {how or '-'} |"
        )
    out.append("")
    out.append("BIB ENTRIES WITH NO MANIFEST RECORD")
    for entry in bib_only:
        out.append(f"  {entry['key']}  doi={entry['doi'] or 'NONE (cannot join)'}")
    if not bib_only:
        out.append("  (none)")
    out.append("")
    out.append("MANIFEST RECORDS WITH NO BIB KEY")
    for record, reason in manifest_only:
        out.append(
            f"  ref {record['ref_num']}  {record['first_author']} "
            f"{record['year']}  doi={record['doi'] or 'NONE RECORDED'}  [{reason}]"
        )
    if not manifest_only:
        out.append("  (none)")
    out.append("")
    out.append(f"report written: {REPORT_PATH.name}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
