"""Look at the data before training on it. Writes artifacts/eda/: report.md, features.csv, 4 charts.

1. sizes          how many sentences in each split
2. class balance  sentences per intent (is any intent rare?)
3. slot tail      how often each slot type appears (a few common, many rare)
4. lengths        words per sentence: Devanagari vs Hinglish vs English
5. code-mixing    CMI, I-Index, script, spelling variants - per set, plus the 4 CMI bands
6. leakage        does any test sentence also sit in the training data?
"""

import json
from pathlib import Path

from cmx.charts import COLOURS, GRID, INK, MUTED, SURFACE, new_chart, save_chart
from cmx.data import load
from cmx.examples import make_example, romanized_copy, test_sets
from cmx.features import BANDS, FEATURE_NAMES, band, build_word_lists, sentence_features
from cmx.quick import notice

OUT = Path("artifacts") / "eda"
FEATURES = FEATURE_NAMES


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def by_size(pair):
    """Sort helper: ("alarm_set", 120) -> 120."""
    return pair[1]


def count_by(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def class_balance(train, lines):
    counts = count_by([r["intent"] for r in train])
    ordered = sorted(counts.items(), key=by_size, reverse=True)
    biggest = ordered[0]
    smallest = ordered[-1]
    lines.append("## 2. Class balance (training sentences per intent)\n")
    lines.append(f"{len(counts)} intents. Most: `{biggest[0]}` {biggest[1]}. Fewest: `{smallest[0]}` "
                 f"{smallest[1]}. Ratio {biggest[1] / smallest[1]:.0f} to 1.")
    lines.append("So accuracy alone can hide rare intents failing: the results also report macro-F1, "
                 "which gives every intent the same weight.\n")

    fig, ax = new_chart(7, 9, "Training sentences per intent")
    names = []
    sizes = []
    for name, size in reversed(ordered):
        names.append(name)
        sizes.append(size)
    ax.barh(names, sizes, color=COLOURS[0], height=0.7)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlabel("sentences", color=MUTED)
    save_chart(fig, OUT / "intents.png")
    return {"intents": len(counts), "most": biggest, "fewest": smallest}


def slot_tail(train_examples, lines):
    counts = {}
    for example in train_examples:
        for tag in example["tags"]:
            if tag.startswith("B-"):
                counts[tag[2:]] = counts.get(tag[2:], 0) + 1
    ordered = sorted(counts.items(), key=by_size, reverse=True)
    rare = []
    for name, size in ordered:
        if size < 50:
            rare.append(name)
    top = []
    for name, size in ordered[:5]:
        top.append(f"`{name}` {size}")
    lines.append("## 3. Slot long tail (slots in the training sentences)\n")
    lines.append(f"{len(counts)} slot types. Top 5: {', '.join(top)}.")
    lines.append(f"{len(rare)} types appear fewer than 50 times - the model sees few examples of these, "
                 "which is why tuning may weight rare slots higher.\n")

    fig, ax = new_chart(9, 4, "Slot types, most to least common (log scale)")
    names = []
    sizes = []
    for name, size in ordered:
        names.append(name)
        sizes.append(size)
    ax.bar(range(len(names)), sizes, color=COLOURS[0], width=0.7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle="--")
    ax.text(len(names) - 1, 55, "50 examples", color=MUTED, fontsize=8, ha="right")
    ax.set_ylabel("times in training", color=MUTED)
    save_chart(fig, OUT / "slots.png")
    return {"slot_types": len(counts), "rare_types": len(rare)}


def lengths(sets, english_test, lines):
    groups = [
        ("Devanagari (paired test)", [len(e["words"]) for e in sets["devanagari_paired_test"]]),
        ("Hinglish (paired test)", [len(e["words"]) for e in sets["hinglish_paired_test"]]),
        ("English (test)", [len(e["words"]) for e in english_test]),
    ]
    lines.append("## 4. Sentence length (words)\n")
    lines.append("| set | mean | median | longest |")
    lines.append("|---|---|---|---|")
    result = {}
    for name, sizes in groups:
        lines.append(f"| {name} | {mean(sizes):.1f} | {median(sizes)} | {max(sizes)} |")
        result[name] = round(mean(sizes), 2)
    lines.append("\nAll short (voice-assistant commands), so 64 word-pieces per sentence is plenty.\n")

    fig, ax = new_chart(7, 3.6, "Words per sentence")
    for i, (name, sizes) in enumerate(groups):
        counts = count_by(sizes)
        xs = list(range(1, 26))
        ys = []
        for x in xs:
            ys.append(100 * counts.get(x, 0) / len(sizes))
        ax.plot(xs, ys, color=COLOURS[i], linewidth=2, label=name)
    ax.set_xlabel("words", color=MUTED)
    ax.set_ylabel("% of sentences", color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save_chart(fig, OUT / "lengths.png")
    return result


def code_mixing(sets, english_test, romanized_test, lists, lines, rows):
    groups = [
        ("Hinglish paired test", sets["hinglish_paired_test"]),
        ("Hinglish paired dev", sets["hinglish_paired_dev"]),
        ("Devanagari paired test", sets["devanagari_paired_test"]),
        ("Auto-romanized Hindi test", romanized_test),
        ("English test", english_test),
    ]
    lines.append("## 5. Code-mixing features\n")
    lines.append("CMI: 0 = one language, 50 = half and half. I-Index: how often the language switches. "
                 "Latin: share of words in English letters. Variant / unknown: share of English-letter "
                 "words that are a new spelling of a known Hindi word / in no word list.\n")
    lines.append("| set | sentences | CMI | I-Index | English words | Latin script | variant | unknown |")
    lines.append("|---|---|---|---|---|---|---|---|")
    result = {}
    cmi_values = {}
    for name, examples in groups:
        per_feature = {}
        for feature in FEATURES:
            per_feature[feature] = []
        for example in examples:
            values = sentence_features(example["words"], lists)
            for feature in FEATURES:
                per_feature[feature].append(values[feature])
            if name.startswith("Hinglish"):
                set_name = "test" if name.endswith("test") else "dev"
                row = [example["id"], set_name]
                for feature in FEATURES:
                    row.append(str(values[feature]))
                row.append(band(values["cmi"]))
                rows.append(row)
        means = {}
        for feature in FEATURES:
            means[feature] = round(mean(per_feature[feature]), 3)
        result[name] = means
        cmi_values[name] = per_feature["cmi"]
        lines.append(
            f"| {name} | {len(examples)} | {means['cmi']:.1f} | {means['i_index']:.2f} | "
            f"{means['english_share']:.0%} | {means['script_latin']:.0%} | {means['variant']:.0%} | "
            f"{means['oov']:.0%} |"
        )

    lines.append("\nWord-tagger check: on English sentences nearly every word should be tagged English, and "
                 "on auto-romanized Hindi nearly every word Hindi. The rows above show how close it gets "
                 "(it is a simple word-list tagger, so some short words go the wrong way).\n")

    lines.append("### CMI bands (the results are sliced by these)\n")
    lines.append("| band | CMI range | Hinglish test | Hinglish dev |")
    lines.append("|---|---|---|---|")
    band_counts = {}
    for name, low, high in BANDS:
        test_n = 0
        dev_n = 0
        for value in cmi_values["Hinglish paired test"]:
            if band(value) == name:
                test_n += 1
        for value in cmi_values["Hinglish paired dev"]:
            if band(value) == name:
                dev_n += 1
        band_counts[name] = {"test": test_n, "dev": dev_n}
        if name == "none":
            span = "exactly 0"
        elif high > 50:
            span = f"{low:g} to 50"
        else:
            span = f"{low:g} to {high:g}"
        lines.append(f"| {name} | {span} | {test_n} | {dev_n} |")
    lines.append("")

    fig, ax = new_chart(7, 3.6, "How mixed is each Hinglish test sentence? (CMI)")
    values = cmi_values["Hinglish paired test"]
    edges = list(range(0, 52, 4))  # bars 4 CMI points wide: 0-4, 4-8, ...
    ax.hist(values, bins=edges, color=COLOURS[0], edgecolor=SURFACE, linewidth=2)
    for band_row in BANDS[1:]:
        ax.axvline(band_row[1], color=MUTED, linewidth=1, linestyle="--")
    tops = ax.get_ylim()[1]
    for name, low, high in BANDS:
        middle = (low + min(high, 50)) / 2
        if name == "none":
            middle = 1
        ax.text(middle, tops * 0.93, name, color=MUTED, fontsize=9, ha="center")
    ax.set_xlabel("CMI (0 = all Hindi or all English, 50 = half and half)", color=MUTED)
    ax.set_ylabel("sentences", color=MUTED)
    save_chart(fig, OUT / "cmi.png")
    return {"means": result, "bands": band_counts}


def leakage(hindi, sets, lines):
    lines.append("## 6. Leakage checks\n")
    by_partition = {"train": set(), "dev": set(), "test": set()}
    partition_of = {}
    for row in hindi:
        by_partition[row["partition"]].add(row["utt"].strip())
        partition_of[row["id"]] = row["partition"]
    train_test = by_partition["train"] & by_partition["test"]
    train_dev = by_partition["train"] & by_partition["dev"]
    test_ids = set()
    for e in sets["hinglish_paired_test"]:
        test_ids.add(e["id"])
    dev_ids = set()
    for e in sets["hinglish_paired_dev"]:
        dev_ids.add(e["id"])
    wrong_test = 0
    for row_id in test_ids:
        if partition_of[row_id] != "test":
            wrong_test += 1
    wrong_dev = 0
    for row_id in dev_ids:
        if partition_of[row_id] != "dev":
            wrong_dev += 1
    hinglish_texts_test = set()
    for e in sets["hinglish_paired_test"]:
        hinglish_texts_test.add(" ".join(e["words"]).lower())
    shared_text = 0
    for e in sets["hinglish_paired_dev"]:
        if " ".join(e["words"]).lower() in hinglish_texts_test:
            shared_text += 1
    paired_in_train = 0
    for e in sets["devanagari_paired_test"]:
        if " ".join(e["words"]) in by_partition["train"]:
            paired_in_train += 1

    checks = [
        ("Hindi sentences that appear word-for-word in both train and test", len(train_test)),
        ("Hindi sentences in both train and dev", len(train_dev)),
        ("Paired test sentences (of 560) whose Hindi text is also a training sentence", paired_in_train),
        ("Hinglish test sentences whose id is not in MASSIVE's test split", wrong_test),
        ("Hinglish dev sentences whose id is not in MASSIVE's dev split", wrong_dev),
        ("Ids in both the Hinglish test and dev sets", len(test_ids & dev_ids)),
        ("Hinglish sentences with the same text in test and dev", shared_text),
    ]
    lines.append("| check | count |")
    lines.append("|---|---|")
    for name, value in checks:
        lines.append(f"| {name} | {value} |")
    examples = "; ".join(sorted(train_test)[:3])
    if examples:
        lines.append(f"\nExamples of train/test repeats: {examples}. MASSIVE ships them this way; they "
                     "are kept so the numbers stay comparable with other MASSIVE results.")
    lines.append("\nThe word lists for the features and the tuning only ever use training and dev "
                 "sentences, never test.\n")
    result = {}
    for name, value in checks:
        result[name] = value
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    hindi = load("hi-IN")
    english = load("en-US")
    sets = test_sets()
    lists = build_word_lists()

    train = []
    for row in hindi:
        if row["partition"] == "train":
            train.append(row)
    train_examples = []
    for row in train:
        train_examples.append(make_example(row["id"], row["intent"], row["annot_utt"]))
    english_test = []
    for row in english:
        if row["partition"] == "test":
            english_test.append(make_example(row["id"], row["intent"], row["annot_utt"]))
    romanized_test = []
    for example in sets["test"]:
        romanized_test.append(romanized_copy(example))

    lines = ["# Data analysis (written by `python -m cmx eda`)\n"]
    lines.append("## 1. Sizes\n")
    lines.append("| split | Hindi | English |")
    lines.append("|---|---|---|")
    for partition in ["train", "dev", "test"]:
        n_hi = 0
        for row in hindi:
            if row["partition"] == partition:
                n_hi += 1
        n_en = 0
        for row in english:
            if row["partition"] == partition:
                n_en += 1
        lines.append(f"| {partition} | {n_hi:,} | {n_en:,} |")
    lines.append(f"\nHinglish rewrites (drafted with AI help, see the README's data section): "
                 f"{len(sets['hinglish_paired_test'])} test, {len(sets['hinglish_paired_dev'])} dev - "
                 "each paired with the same sentence in Devanagari.\n")

    rows = []
    summary = {
        "balance": class_balance(train, lines),
        "slots": slot_tail(train_examples, lines),
        "lengths": lengths(sets, english_test, lines),
        "code_mixing": code_mixing(sets, english_test, romanized_test, lists, lines, rows),
        "leakage": leakage(hindi, sets, lines),
    }

    header = ["id", "set", *FEATURES, "band"]  # *FEATURES = all 6 feature names, in order
    csv_lines = [",".join(header) + "\n"]
    for row in rows:
        csv_lines.append(",".join(row) + "\n")
    with open(OUT / "features.csv", "w", encoding="utf-8") as f:
        f.writelines(csv_lines)
    (OUT / "eda.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = "\n".join(lines)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(report)

    means = summary["code_mixing"]["means"]
    notice("CMI", f"Hinglish test mean CMI {means['Hinglish paired test']['cmi']:.1f}, "
                  f"English test {means['English test']['cmi']:.1f}, "
                  f"romanized Hindi {means['Auto-romanized Hindi test']['cmi']:.1f}")
    bands = summary["code_mixing"]["bands"]
    text = []
    for band_row in BANDS:
        text.append(f"{band_row[0]} {bands[band_row[0]]['test']}")
    notice("CMI bands (test)", ", ".join(text))
    print(f"\nwrote {OUT}/report.md, features.csv, eda.json and 4 charts")
