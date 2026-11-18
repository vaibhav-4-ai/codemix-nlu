"""Stage 1b: check the Hinglish rewrites and build the final test + dev files.

Drafts live in data/annotation/drafts/*.tsv, one per line:  id <TAB> hinglish with [slot : value]
    5902	check karo kya [time : subah 6 baje] ka alarm laga hai

Rule: the Hinglish must keep exactly the same slot TYPES as the Hindi original (same labels,
only the writing changes). Rows that break the rule are listed and the build stops.
"""

import json

from cmx.data import DATA, save_jsonl
from cmx.massive import slot_types

ANNOTATION = DATA / "annotation"


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_drafts():
    """All draft lines from every .tsv file -> {id: hinglish_annot}."""
    drafts = {}
    for path in sorted((ANNOTATION / "drafts").glob("*.tsv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row_id, text = line.split("\t", 1)
            if row_id in drafts:
                raise SystemExit(f"{path.name}: id {row_id} appears twice")
            drafts[row_id] = text.strip()
    return drafts


def problems(hindi_annot, hinglish_annot):
    """'' if the slot types match, else a short description of what differs."""
    if hinglish_annot.count("[") != hinglish_annot.count("]"):
        return "unbalanced brackets"
    want = slot_types(hindi_annot)
    got = slot_types(hinglish_annot)
    if want != got:
        return f"slot types differ: Hindi {want} vs Hinglish {got}"
    return ""


def build():
    drafts = read_drafts()
    for row in read_jsonl(ANNOTATION / "seed_50_hinglish.jsonl"):
        drafts[row["id"]] = row["hinglish_annot"]
    bad = 0
    missing = 0
    for name in ["test_set", "dev_set"]:
        out = []
        for row in read_jsonl(ANNOTATION / f"{name}.jsonl"):
            if row["id"] not in drafts:
                missing += 1
                continue
            issue = problems(row["hindi"], drafts[row["id"]])
            if issue:
                bad += 1
                print(f"{name} id {row['id']} ({row['intent']}): {issue}\n    {drafts[row['id']]}")
                continue
            out.append(
                {
                    "id": row["id"],
                    "intent": row["intent"],
                    "hinglish_annot": drafts[row["id"]],
                    "drafted_by": "claude",
                    "edited": False,
                }
            )
        save_jsonl(ANNOTATION / f"{name}_hinglish.jsonl", out)
        print(f"{name}: {len(out)} rows ready")
    print(f"{bad} rows with problems, {missing} not drafted yet")
    if bad:
        raise SystemExit(1)
