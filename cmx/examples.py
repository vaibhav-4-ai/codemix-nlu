"""Build the sentences the real model trains and tests on. No torch here, so tests stay fast.

One example looks like:
    {"id": "3761", "intent": "alarm_set",
     "words": ["mujhe", "8", "baje", "utha", "dena"],
     "tags":  ["O", "B-time", "I-time", "O", "O"]}

Training recipes (each one changes ONE thing, so its effect can be measured):
    plain      Hindi train set, Devanagari only
    romanized  plain + the same sentences auto-converted to English letters
    english    plain + the English train set
    fixed      plain + romanized + English              (the main model)
    noisy      fixed, but the romanized copies get Hinglish-style spelling noise
"""

import json
import random

from cmx.data import DATA, load
from cmx.massive import parse_annot
from cmx.quick import romanize

RECIPES = ["plain", "romanized", "english", "fixed", "noisy"]
NOISE_SEED = 13


def make_example(row_id, intent, annot):
    words, tags = parse_annot(annot)
    return {"id": row_id, "intent": intent, "words": words, "tags": tags}


def clean_word(word):
    """Keep letters and digits only: ITRANS marks like '.n' or '~' never appear in typed Hinglish."""
    kept = ""
    for ch in word:
        if ch.isalnum():
            kept += ch
    if kept:
        return kept
    return word


def romanized_copy(example):
    """Same sentence, each Devanagari word turned into English letters. Tags stay the same."""
    words = []
    for word in example["words"]:
        words.append(clean_word(romanize(word)))
    return {"id": example["id"], "intent": example["intent"], "words": words, "tags": example["tags"]}


def add_spelling_noise(word, rng):
    """Make a mechanical romanization look more like how people type.

    'kamare' -> 'kamre' (drop a middle 'a'), 'kii' -> 'ki', 'baatcheet' -> 'batcheet',
    'vaalaa' -> 'wala'. Each rule fires at random, so one word gets many spellings.
    """
    if rng.random() < 0.7:
        word = word.replace("aa", "a")
    if rng.random() < 0.5:
        word = word.replace("ii", "i")
    elif rng.random() < 0.5:
        word = word.replace("ii", "ee")
    if rng.random() < 0.5:
        word = word.replace("uu", "u")
    elif rng.random() < 0.5:
        word = word.replace("uu", "oo")
    if rng.random() < 0.4:
        word = word.replace("v", "w")
    if len(word) > 4 and rng.random() < 0.5:
        # schwa deletion: drop one 'a' that sits between two consonants in the middle of the word
        vowels = "aeiou"
        for i in range(2, len(word) - 2):
            if word[i] == "a" and word[i - 1] not in vowels and word[i + 1] not in vowels:
                word = word[:i] + word[i + 1 :]
                break
    return word


def noisy_copy(example, rng):
    copy = romanized_copy(example)
    words = []
    for word in copy["words"]:
        words.append(add_spelling_noise(word, rng))
    copy["words"] = words
    return copy


def training_examples(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe}")
    rng = random.Random(NOISE_SEED)
    hindi = load("hi-IN")
    english = load("en-US")
    examples = []
    for row in hindi:
        if row["partition"] == "train":
            example = make_example(row["id"], row["intent"], row["annot_utt"])
            examples.append(example)
            if recipe in ("romanized", "fixed"):
                examples.append(romanized_copy(example))
            if recipe == "noisy":
                examples.append(noisy_copy(example, rng))
    if recipe in ("english", "fixed", "noisy"):
        for row in english:
            if row["partition"] == "train":
                examples.append(make_example(row["id"], row["intent"], row["annot_utt"]))
    return examples


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paired_sets(name, hindi_by_id):
    """The Hinglish rows of one annotation file + the same ids in Devanagari."""
    hinglish = []
    devanagari = []
    for row in read_jsonl(DATA / "annotation" / f"{name}.jsonl"):
        hinglish.append(make_example(row["id"], row["intent"], row["hinglish_annot"]))
        original = hindi_by_id[row["id"]]
        devanagari.append(make_example(row["id"], row["intent"], original["annot_utt"]))
    return devanagari, hinglish


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

    sets = {"dev": dev, "test": test}
    for file_name, short in [
        ("seed_50_hinglish", "50"),
        ("test_set_hinglish", "paired_test"),
        ("dev_set_hinglish", "paired_dev"),
    ]:
        devanagari, hinglish = paired_sets(file_name, hindi_by_id)
        sets[f"devanagari_{short}"] = devanagari
        sets[f"hinglish_{short}"] = hinglish
    return sets


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
