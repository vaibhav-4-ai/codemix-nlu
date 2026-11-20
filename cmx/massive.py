"""Read MASSIVE's labelled sentences.

MASSIVE marks slots inside the sentence itself:
    "kal [time : subah 7 baje] ka alarm laga do"
parse_annot turns that into words + one tag per word:
    words: kal  subah   7       baje    ka alarm laga do
    tags:  O    B-time  I-time  I-time  O  O     O    O
(B- = first word of a slot, I- = the rest of it, O = not part of any slot.)
"""

import re

SLOT = re.compile(r"\[\s*([^\]:]+?)\s*:\s*([^\]]*?)\s*\]")


def parse_annot(annot: str) -> tuple[list[str], list[str]]:
    """Split a bracketed sentence into words and their slot tags."""
    words, tags = [], []
    pos = 0
    for m in SLOT.finditer(annot):
        for w in annot[pos : m.start()].split():
            words.append(w)
            tags.append("O")
        for i, w in enumerate(m.group(2).split()):
            words.append(w)
            tags.append(("B-" if i == 0 else "I-") + m.group(1))
        pos = m.end()
    for w in annot[pos:].split():
        words.append(w)
        tags.append("O")
    return words, tags


def slot_spans(tags):
    """BIO tags -> set of slots as (first word, last word, type).

    ["B-time", "I-time", "O", "B-date"] -> {(0, 1, "time"), (3, 3, "date")}
    An I- tag with no matching slot open starts a new slot (the same rule seqeval uses).
    """
    spans = set()
    start = None
    kind = None
    padded = list(tags) + ["O"]
    for i, tag in enumerate(padded):
        continues = tag.startswith("I-") and tag[2:] == kind
        if start is not None and not continues:
            spans.add((start, i - 1, kind))
            start = None
            kind = None
        if tag.startswith("B-") or (tag.startswith("I-") and not continues):
            start = i
            kind = tag[2:]
    return spans


def slot_types(annot: str) -> list[str]:
    """The slot names in a sentence, sorted: '[time : 7 baje] [date : kal]' -> ['date', 'time']."""
    return sorted(m.group(1).strip() for m in SLOT.finditer(annot))


def slots(annot: str) -> list[tuple[str, str]]:
    """The (slot name, words) pairs in order: [('time', '7 baje'), ...]."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in SLOT.finditer(annot)]
