"""Build the sentences the real model trains and tests on. No torch here, so tests stay fast.

One example looks like:
    {"id": "3761", "intent": "alarm_set",
     "words": ["mujhe", "8", "baje", "utha", "dena"],
     "tags":  ["O", "B-time", "I-time", "O", "O"]}
"""

import json

from cmx.data import DATA, load
from cmx.massive import parse_annot
from cmx.quick import romanize


def make_example(row_id, intent, annot):
    words, tags = parse_annot(annot)
    return {"id": row_id, "intent": intent, "words": words, "tags": tags}


def romanized_copy(example):
    """Same sentence, each Devanagari word turned into English letters. Tags stay the same."""
    words = []
    for word in example["words"]:
        words.append(romanize(word))
    return {"id": example["id"], "intent": example["intent"], "words": words, "tags": example["tags"]}


def training_examples(recipe):
    """recipe 'plain' = Devanagari only. recipe 'fixed' = + auto-romanized copy + English."""
    hindi = load("hi-IN")
    english = load("en-US")
    examples = []
    for row in hindi:
        if row["partition"] == "train":
            example = make_example(row["id"], row["intent"], row["annot_utt"])
            examples.append(example)
            if recipe == "fixed":
                examples.append(romanized_copy(example))
    if recipe == "fixed":
        for row in english:
            if row["partition"] == "train":
                examples.append(make_example(row["id"], row["intent"], row["annot_utt"]))
    return examples


def test_sets():
    """Every set the model is scored on, by name."""
    hindi = load("hi-IN")
    dev = []
    test = []
    hindi_by_id = {}
    for row in hindi:
        hindi_by_id[row["id"]] = row
        if row["partition"] == "dev":
            dev.append(make_example(row["id"], row["intent"], row["annot_utt"]))
        if row["partition"] == "test":
            test.append(make_example(row["id"], row["intent"], row["annot_utt"]))

    path = DATA / "annotation" / "seed_50_hinglish.jsonl"
    hinglish_50 = []
    devanagari_50 = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hinglish_50.append(make_example(row["id"], row["intent"], row["hinglish_annot"]))
        original = hindi_by_id[row["id"]]
        devanagari_50.append(make_example(row["id"], row["intent"], original["annot_utt"]))

    return {"dev": dev, "test": test, "devanagari_50": devanagari_50, "hinglish_50": hinglish_50}


def align_tags(word_ids, tags, tag_ids):
    """Give each word-piece a tag number. Only a word's FIRST piece is scored; the rest get -100.

    The model splits words into pieces: "baje" -> "ba", "##je". word_ids says which word each
    piece came from (None = a special token like [CLS]). -100 means "ignore this piece".
    """
    labels = []
    previous = None
    for word_id in word_ids:
        if word_id is None or word_id == previous:
            labels.append(-100)
        else:
            labels.append(tag_ids.get(tags[word_id], tag_ids["O"]))
        previous = word_id
    return labels
