"""Stage 1b: check the Hinglish rewrites and build the final test + dev files.

Drafts live in data/annotation/drafts/*.tsv, one per line:  id <TAB> hinglish with [slot : value]
    5902	check karo kya [time : subah 6 baje] ka alarm laga hai

Rule: the Hinglish must keep exactly the same slot TYPES as the Hindi original (same labels,
only the writing changes). Rows that break the rule are listed and the build stops.
"""

import json
import random

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


REVIEW = ANNOTATION / "review_100.tsv"
REVIEW_SEED = 13
# Who checked the 100. Kept honest on purpose: a second AI pass is not a human review.
REVIEWER = "claude, second pass (not a human)"


def make_review():
    """Pick 100 random rewrites (fixed seed) into review_100.tsv for a second check."""
    english_by_id = {}
    for row in read_jsonl(ANNOTATION / "test_set.jsonl") + read_jsonl(ANNOTATION / "dev_set.jsonl"):
        english_by_id[row["id"]] = row["english"]
    rows = read_jsonl(ANNOTATION / "test_set_hinglish.jsonl") + read_jsonl(ANNOTATION / "dev_set_hinglish.jsonl")
    ids = []
    text_by_id = {}
    for row in rows:
        ids.append(row["id"])
        text_by_id[row["id"]] = row["hinglish_annot"]
    picked = random.Random(REVIEW_SEED).sample(sorted(ids), 100)
    lines = ["n\tid\tenglish (the meaning)\thinglish (drafted)\tyour_version (leave empty if fine)\n"]
    for n, row_id in enumerate(picked, start=1):
        lines.append(f"{n}\t{row_id}\t{english_by_id[row_id]}\t{text_by_id[row_id]}\t\n")
    with open(REVIEW, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"wrote {REVIEW} - 100 sentences to review")


def read_review():
    """-> (ids that were reviewed, {id: corrected hinglish} for the ones the reviewer changed)."""
    reviewed = set()
    changed = {}
    if not REVIEW.exists():
        return reviewed, changed
    for line in REVIEW.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        reviewed.add(parts[1])
        if len(parts) >= 5 and parts[4].strip() and parts[4].strip() != parts[3].strip():
            changed[parts[1]] = parts[4].strip()
    return reviewed, changed


def build():
    drafts = read_drafts()
    for row in read_jsonl(ANNOTATION / "seed_50_hinglish.jsonl"):
        drafts[row["id"]] = row["hinglish_annot"]
    reviewed, changed = read_review()
    for row_id, text in changed.items():
        drafts[row_id] = text  # the reviewer's version wins
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
                    "checked": row["id"] in reviewed,  # one of the random 100 (see REVIEWER)
                    "edited": row["id"] in changed,
                }
            )
        save_jsonl(ANNOTATION / f"{name}_hinglish.jsonl", out)
        print(f"{name}: {len(out)} rows ready")
    print(f"{bad} rows with problems, {missing} not drafted yet")
    if reviewed:
        summary = {"checked": len(reviewed), "edited": len(changed), "reviewer": REVIEWER,
                   "edit_rate": round(len(changed) / len(reviewed), 3)}  # fmt: skip
        (ANNOTATION / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"check by {REVIEWER}: {len(changed)} of {len(reviewed)} sentences changed")
    if bad:
        raise SystemExit(1)
