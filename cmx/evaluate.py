"""Stage 7: the final numbers, worked out from the saved predictions (CPU only, no torch).

Kaggle trained 2 models x 3 seeds and uploaded every prediction to Hugging Face. CI downloads
them into artifacts/hf_runs/ first. This file then reports, each with a 95% range:
  - intent accuracy and slot F1 on the same 50 sentences, in Devanagari and in Hinglish
  - the drop (Devanagari minus Hinglish) for each model
  - how much the fix wins back on Hinglish (fixed minus plain)

"Paired": every comparison uses the SAME 50 sentences, so each resample picks one set of
sentence ids and uses it for every model and both scripts. That makes the ranges for the
differences much narrower than comparing two separate accuracies would.
"""

import json
import random
from pathlib import Path

from cmx.quick import notice

RECIPES = ["plain", "fixed"]
SEEDS = [13, 42, 1337]
SETS = ["test", "devanagari_50", "hinglish_50"]
RUNS = Path("artifacts") / "hf_runs"
OUT = Path("artifacts") / "evaluate"
DRAWS = 2000
SEED = 13

KEYS_50 = ["plain devanagari_50", "plain hinglish_50", "fixed devanagari_50", "fixed hinglish_50"]
DIFFERENCES = [
    ("plain: Devanagari minus Hinglish", "plain devanagari_50", "plain hinglish_50"),
    ("fixed: Devanagari minus Hinglish", "fixed devanagari_50", "fixed hinglish_50"),
    ("Hinglish: fixed minus plain", "fixed hinglish_50", "plain hinglish_50"),
]


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


def load_predictions(recipe, seed, name):
    path = RUNS / f"{recipe}_seed{seed}" / f"predictions_{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} - CI downloads it from Hugging Face before this step")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def per_sentence(recipe, name):
    """For every sentence id: how many seeds got the intent right, plus slot counts over all seeds."""
    table = {}
    for seed in SEEDS:
        for row in load_predictions(recipe, seed, name):
            if row["id"] not in table:
                table[row["id"]] = {"right": 0, "hits": 0, "true": 0, "guessed": 0}
            entry = table[row["id"]]
            if row["predicted"] == row["intent"]:
                entry["right"] += 1
            true_spans = slot_spans(row["tags"])
            guessed_spans = slot_spans(row["predicted_tags"])
            entry["hits"] += len(true_spans & guessed_spans)
            entry["true"] += len(true_spans)
            entry["guessed"] += len(guessed_spans)
    return table


def summarize(entries, n_seeds):
    """-> (intent accuracy, slot F1), both pooled over all seeds."""
    right = 0
    hits = 0
    total = 0
    for entry in entries:
        right += entry["right"]
        hits += entry["hits"]
        total += entry["true"] + entry["guessed"]
    accuracy = right / (len(entries) * n_seeds)
    slot_f1 = 2 * hits / total if total else 0.0
    return accuracy, slot_f1


def interval(values):
    """The middle 95% of the resampled values."""
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def pct(x):
    return f"{100 * x:.1f}"


def run():
    tables = {}
    for recipe in RECIPES:
        for name in SETS:
            tables[f"{recipe} {name}"] = per_sentence(recipe, name)

    ids = sorted(tables["plain hinglish_50"])
    for key in KEYS_50:
        if sorted(tables[key]) != ids:
            raise SystemExit(f"{key} does not have the same 50 sentence ids - can't pair them")

    # Point estimates on the real 50 sentences.
    point = {}
    for key in KEYS_50:
        entries = []
        for i in ids:
            entries.append(tables[key][i])
        point[key] = summarize(entries, len(SEEDS))

    # Paired bootstrap: one random draw of sentence ids, used for every table.
    draws = {}
    for key in KEYS_50:
        draws[key] = {"intent": [], "slot_f1": []}
    for label, _, _ in DIFFERENCES:
        draws[label] = {"intent": [], "slot_f1": []}
    rng = random.Random(SEED)
    for _ in range(DRAWS):
        sample = [rng.choice(ids) for _ in ids]
        scores = {}
        for key in KEYS_50:
            entries = []
            for i in sample:
                entries.append(tables[key][i])
            scores[key] = summarize(entries, len(SEEDS))
            draws[key]["intent"].append(scores[key][0])
            draws[key]["slot_f1"].append(scores[key][1])
        for label, a, b in DIFFERENCES:
            draws[label]["intent"].append(scores[a][0] - scores[b][0])
            draws[label]["slot_f1"].append(scores[a][1] - scores[b][1])

    results = {"seeds": SEEDS, "n_50": len(ids), "draws": DRAWS, "test": {}, "sets_50": {}, "differences": {}}
    for recipe in RECIPES:
        accuracy, slot_f1 = summarize(list(tables[f"{recipe} test"].values()), len(SEEDS))
        results["test"][recipe] = {"intent": accuracy, "slot_f1": slot_f1, "n": len(tables[f"{recipe} test"])}
    for key in KEYS_50:
        results["sets_50"][key] = {
            "intent": point[key][0],
            "intent_95": interval(draws[key]["intent"]),
            "slot_f1": point[key][1],
            "slot_f1_95": interval(draws[key]["slot_f1"]),
        }
    for label, a, b in DIFFERENCES:
        results["differences"][label] = {
            "intent": point[a][0] - point[b][0],
            "intent_95": interval(draws[label]["intent"]),
            "slot_f1": point[a][1] - point[b][1],
            "slot_f1_95": interval(draws[label]["slot_f1"]),
        }

    lines = [
        f"Final numbers: hing-mbert, seeds {SEEDS}, pooled. 95% ranges: paired bootstrap, {DRAWS} draws.",
        "",
        "| Model | Sentences | Intent accuracy % | Slot F1 % |",
        "|---|---|---|---|",
    ]
    for recipe in RECIPES:
        r = results["test"][recipe]
        lines.append(f"| {recipe} | MASSIVE test, Devanagari (n={r['n']}) | {pct(r['intent'])} | {pct(r['slot_f1'])} |")
    for key in KEYS_50:
        r = results["sets_50"][key]
        recipe, name = key.split(" ")
        lines.append(
            f"| {recipe} | {name} | {pct(r['intent'])} ({pct(r['intent_95'][0])} to {pct(r['intent_95'][1])})"
            f" | {pct(r['slot_f1'])} ({pct(r['slot_f1_95'][0])} to {pct(r['slot_f1_95'][1])}) |"
        )
    lines += ["", "| Difference (points) | Intent | Slot F1 |", "|---|---|---|"]
    for label, _, _ in DIFFERENCES:
        r = results["differences"][label]
        lines.append(
            f"| {label} | {pct(r['intent'])} ({pct(r['intent_95'][0])} to {pct(r['intent_95'][1])})"
            f" | {pct(r['slot_f1'])} ({pct(r['slot_f1_95'][0])} to {pct(r['slot_f1_95'][1])}) |"
        )
        notice(
            label,
            f"intent {pct(r['intent'])} points (95%: {pct(r['intent_95'][0])} to {pct(r['intent_95'][1])}); "
            f"slot F1 {pct(r['slot_f1'])} (95%: {pct(r['slot_f1_95'][0])} to {pct(r['slot_f1_95'][1])})",
        )
    for key in KEYS_50:
        r = results["sets_50"][key]
        notice(
            key,
            f"intent {pct(r['intent'])}% (95%: {pct(r['intent_95'][0])} to {pct(r['intent_95'][1])}); "
            f"slot F1 {pct(r['slot_f1'])}%",
        )
    for recipe in RECIPES:
        r = results["test"][recipe]
        notice(f"{recipe} test (n={r['n']})", f"intent {pct(r['intent'])}%; slot F1 {pct(r['slot_f1'])}%")

    table = "\n".join(lines) + "\n"
    print(table)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "results.md").write_text(table, encoding="utf-8")
    print(f"saved {OUT}/results.json and results.md")
