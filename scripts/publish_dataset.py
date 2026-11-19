"""Publish the 952 paired sentences as a Hugging Face dataset with a full data card.

Each row: the MASSIVE id + intent, the sentence in Devanagari Hindi, English and Hinglish (all with
[slot : value] marks), and the Hinglish sentence's code-mixing numbers.
Runs only in CI, only with the HF_TOKEN secret (it creates the dataset repo if it is missing).
"""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from cmx.data import load
from cmx.examples import read_jsonl
from cmx.features import band, build_word_lists, sentence_features
from cmx.massive import parse_annot

REPO = "vaibhav-4-ai/hinglish-massive-paired"
OUT = Path("artifacts") / "dataset"
ANNOTATION = Path("data") / "annotation"

CARD = """---
license: cc-by-4.0
language:
- hi
- en
task_categories:
- text-classification
- token-classification
tags:
- code-mixing
- hinglish
- intent-classification
- slot-filling
- massive
size_categories:
- n<1K
pretty_name: Hinglish MASSIVE (paired)
---

# Hinglish MASSIVE, paired

{n_test} test + {n_dev} dev voice-assistant requests from Amazon's MASSIVE, each in **three forms with the
same labels**: the original Devanagari Hindi, the original English, and a romanized, code-mixed
**Hinglish** rewrite ("kal subah 7 baje ka alarm laga do"). Made for measuring how much accuracy a Hindi
model loses when the same request arrives as Hinglish. Project and results:
https://github.com/vaibhav-4-ai/codemix-nlu

## How the Hinglish was made (please read)

- **Drafted by an AI model (Claude, Anthropic)**, in an informal WhatsApp style, one sentence at a time
  from the Hindi and English originals. Names and places were sometimes swapped for Indian ones
  (e.g. "Logan" -> "Lagaan"); the intent and slot types never change.
- **Automatic check on every row:** the Hinglish must carry exactly the same slot types as the Hindi
  original, and brackets must balance. All {n_total} rows pass.
- **Second check on a random 100 (fixed seed 13): done by a second AI pass, not by a person.**
  {edited} of 100 were corrected ({rate}). The corrections are in `review_100.tsv`.
- **Not reviewed line by line by a native speaker.** Treat it as a strong synthetic benchmark, not as
  human-written Hinglish. For human-written Hinglish, see Google's Hinglish-TOP (the project also
  reports results on it).

## Fields

| field | meaning |
|---|---|
| id, intent | MASSIVE's sentence id and intent (60 intents) |
| hinglish, hinglish_annot | the rewrite, plain and with `[slot : value]` marks |
| devanagari_annot, english_annot | MASSIVE's own Hindi (hi-IN) and English (en-US) versions |
| cmi | code-mixing index of the Hinglish, 0 = one language, 50 = half and half |
| cmi_band | none / light (0-20) / medium (20-35) / heavy (35-50) |
| edited_in_check | true for the rows corrected in the 100-sentence check |

Splits: `test` comes from MASSIVE's test split (up to 10 per intent), `dev` from its dev split
(up to 7 per intent). Never train on `test`.

## Licence and credit

CC BY 4.0, the same as MASSIVE. The Devanagari and English sentences and all labels are from
**MASSIVE** (FitzGerald et al., 2022, "MASSIVE: A 1M-Example Multilingual Natural Language
Understanding Dataset with 51 Typologically-Diverse Languages", arXiv 2204.08582), CC BY 4.0.
The Hinglish rewrites are new.
"""


def rows_for(name, hindi_by_id, english_by_id, lists):
    out = []
    for row in read_jsonl(ANNOTATION / f"{name}_hinglish.jsonl"):
        words, _ = parse_annot(row["hinglish_annot"])
        cmi = sentence_features(words, lists)["cmi"]
        out.append({
            "id": row["id"],
            "intent": row["intent"],
            "hinglish": " ".join(words),
            "hinglish_annot": row["hinglish_annot"],
            "devanagari_annot": hindi_by_id[row["id"]]["annot_utt"],
            "english_annot": english_by_id[row["id"]]["annot_utt"],
            "cmi": cmi,
            "cmi_band": band(cmi),
            "edited_in_check": row.get("edited", False),
        })  # fmt: skip
    return out


def main():
    hindi_by_id = {}
    for row in load("hi-IN"):
        hindi_by_id[row["id"]] = row
    english_by_id = {}
    for row in load("en-US"):
        english_by_id[row["id"]] = row
    lists = build_word_lists()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, split in [("test_set", "test"), ("dev_set", "dev")]:
        rows = rows_for(name, hindi_by_id, english_by_id, lists)
        counts[split] = len(rows)
        lines = []
        for row in rows:
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(lines)
    summary = json.loads((ANNOTATION / "review_summary.json").read_text(encoding="utf-8"))
    card = CARD.format(
        n_test=counts["test"],
        n_dev=counts["dev"],
        n_total=counts["test"] + counts["dev"],
        edited=summary["edited"],
        rate=f"{100 * summary['edit_rate']:.0f}%",
    )
    (OUT / "README.md").write_text(card, encoding="utf-8")
    (OUT / "review_100.tsv").write_text((ANNOTATION / "review_100.tsv").read_text(encoding="utf-8"),
                                        encoding="utf-8")
    print(f"built {counts} in {OUT}")

    if not os.environ.get("HF_TOKEN"):
        print("no HF_TOKEN secret - dataset built but not uploaded")
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=REPO, repo_type="dataset",
                      commit_message="CI: paired Hinglish MASSIVE sentences + data card")
    print(f"uploaded to https://huggingface.co/datasets/{REPO}")


main()
