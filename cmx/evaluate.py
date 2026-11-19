"""Stage 7: the final numbers, worked out from the saved predictions (CPU only, no torch).

Kaggle trained 6 models x 3 seeds and uploaded every prediction to Hugging Face
(runs_v2/<recipe>_<backbone>_seed<seed>/). scripts/fetch_runs.py downloads them first.

    1. headline   every model: full Hindi test, 560 paired sentences in Devanagari and Hinglish
    2. the drop   Devanagari minus Hinglish, per model, with a 95% range
    3. ablations  which part of the fix matters (each recipe vs plain), HingBERT vs MuRIL
    4. by CMI     the drop sliced by how mixed the Hinglish sentence is (the degradation curve)
    5. checks     seed-to-seed spread; the drop without the 23 sentences that also sit in training
    6. curves     training loss and Hinglish dev accuracy per epoch
    7. errors     real sentences the model gets wrong, written to errors.md

"Paired": every comparison uses the SAME sentences, so each resample picks one set of sentence
ids and uses it for every model and both scripts. The ranges for differences come out narrower
than comparing two separate accuracies would give.
"""

import csv
import json
import random
from pathlib import Path

from cmx.charts import COLOURS, MUTED, new_chart, new_pair, save_chart
from cmx.data import load
from cmx.features import BANDS, band, build_word_lists, sentence_features
from cmx.quick import notice

SEEDS = [13, 42, 1337]
MAIN = "hing-mbert"
MURIL = "muril-base-cased"
MODELS = [
    ("plain", MAIN),
    ("romanized", MAIN),
    ("english", MAIN),
    ("fixed", MAIN),
    ("noisy", MAIN),
    ("fixed", MURIL),
]
SETS = ["test", "devanagari_paired_test", "hinglish_paired_test"]
RUNS = Path("artifacts") / "hf" / "runs_v2"
OUT = Path("artifacts") / "evaluate"
DRAWS = 2000
SEED = 13


def model_name(recipe, backbone):
    return f"{recipe} {backbone}"


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


def run_folder(recipe, backbone, seed):
    return RUNS / f"{recipe}_{backbone}_seed{seed}"


def is_complete(recipe, backbone):
    """True when all 3 seeds of this model are downloaded."""
    for seed in SEEDS:
        if not (run_folder(recipe, backbone, seed) / "metrics.json").exists():
            return False
    return True


def load_predictions(recipe, backbone, seed, name):
    path = run_folder(recipe, backbone, seed) / f"predictions_{name}.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def per_sentence(recipe, backbone, name):
    """For every sentence id: how many seeds got the intent right, slot counts, and each guess."""
    table = {}
    for seed in SEEDS:
        for row in load_predictions(recipe, backbone, seed, name):
            if row["id"] not in table:
                table[row["id"]] = {
                    "right": 0, "hits": 0, "true": 0, "guessed": 0, "intent": row["intent"],
                    "words": row["words"], "tags": row["tags"], "guesses": [], "guessed_tags": [],
                }  # fmt: skip
            entry = table[row["id"]]
            if row["predicted"] == row["intent"]:
                entry["right"] += 1
            true_spans = slot_spans(row["tags"])
            guessed_spans = slot_spans(row["predicted_tags"])
            entry["hits"] += len(true_spans & guessed_spans)
            entry["true"] += len(true_spans)
            entry["guessed"] += len(guessed_spans)
            entry["guesses"].append(row["predicted"])
            entry["guessed_tags"].append(row["predicted_tags"])
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


def macro_f1(entries):
    """Intent F1 for each intent, then the plain average: a rare intent counts as much as a common one."""
    counts = {}
    for entry in entries:
        for guess in entry["guesses"]:
            for name in [entry["intent"], guess]:
                if name not in counts:
                    counts[name] = {"hit": 0, "true": 0, "guessed": 0}
            counts[entry["intent"]]["true"] += 1
            counts[guess]["guessed"] += 1
            if guess == entry["intent"]:
                counts[guess]["hit"] += 1
    scores = []
    for value in counts.values():
        if value["true"] > 0:
            scores.append(2 * value["hit"] / (value["true"] + value["guessed"]))
    return sum(scores) / len(scores)


def interval(values):
    """The middle 95% of the resampled values."""
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def pct(x):
    return f"{100 * x:.1f}"


def with_range(value, low_high):
    return f"{pct(value)} ({pct(low_high[0])} to {pct(low_high[1])})"


def bootstrap(tables, keys, ids, seed):
    """Point estimate + 2,000 paired resamples of the same ids, for every key.

    Returns {key: {"point": (acc, f1), "intent": [acc per draw], "slot_f1": [f1 per draw]}}.
    """
    result = {}
    for key in keys:
        entries = []
        for i in ids:
            entries.append(tables[key][i])
        result[key] = {"point": summarize(entries, len(SEEDS)), "intent": [], "slot_f1": []}
    rng = random.Random(seed)
    for _ in range(DRAWS):
        sample = []
        for _ in ids:
            sample.append(rng.choice(ids))
        for key in keys:
            entries = []
            for i in sample:
                entries.append(tables[key][i])
            accuracy, slot_f1 = summarize(entries, len(SEEDS))
            result[key]["intent"].append(accuracy)
            result[key]["slot_f1"].append(slot_f1)
    return result


def difference(boot, key_a, key_b):
    """a minus b, with the 95% range from the paired draws."""
    intent_draws = []
    slot_draws = []
    for k in range(DRAWS):
        intent_draws.append(boot[key_a]["intent"][k] - boot[key_b]["intent"][k])
        slot_draws.append(boot[key_a]["slot_f1"][k] - boot[key_b]["slot_f1"][k])
    return {
        "intent": boot[key_a]["point"][0] - boot[key_b]["point"][0],
        "intent_95": interval(intent_draws),
        "slot_f1": boot[key_a]["point"][1] - boot[key_b]["point"][1],
        "slot_f1_95": interval(slot_draws),
    }


def most_common(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    best = None
    for item, count in counts.items():
        if best is None or count > counts[best]:
            best = item
    return best


# ---------- the sections ----------


def headline(models, tables, boot, lines, results):
    lines.append("## 1. Every model (3 seeds pooled; 95% ranges from a paired bootstrap)\n")
    lines.append("| model | Hindi test (2,974): intent / macro-F1 / slot F1 | Devanagari paired (560): "
                 "intent / slot F1 | Hinglish paired (560): intent / slot F1 |")
    lines.append("|---|---|---|---|")
    results["models"] = {}
    for recipe, backbone in models:
        name = model_name(recipe, backbone)
        test_entries = list(tables[f"{name}|test"].values())
        test_acc, test_f1 = summarize(test_entries, len(SEEDS))
        test_macro = macro_f1(test_entries)
        row = {"test": {"intent": test_acc, "macro_f1": test_macro, "slot_f1": test_f1}}
        cells = [f"{pct(test_acc)} / {pct(test_macro)} / {pct(test_f1)}"]
        for set_name in ["devanagari_paired_test", "hinglish_paired_test"]:
            b = boot[f"{name}|{set_name}"]
            row[set_name] = {
                "intent": b["point"][0], "intent_95": interval(b["intent"]),
                "slot_f1": b["point"][1], "slot_f1_95": interval(b["slot_f1"]),
                "macro_f1": macro_f1(list(tables[f"{name}|{set_name}"].values())),
            }  # fmt: skip
            cells.append(f"{with_range(b['point'][0], interval(b['intent']))} / "
                         f"{with_range(b['point'][1], interval(b['slot_f1']))}")
        results["models"][name] = row
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")


def differences(models, boot, lines, results):
    names = []
    for recipe, backbone in models:
        names.append(model_name(recipe, backbone))
    rows = []
    for name in names:
        rows.append((f"Drop for {name}: Devanagari minus Hinglish",
                     f"{name}|devanagari_paired_test", f"{name}|hinglish_paired_test"))
    plain = model_name("plain", MAIN)
    for recipe in ["romanized", "english", "fixed", "noisy"]:
        name = model_name(recipe, MAIN)
        if name in names and plain in names:
            rows.append((f"On Hinglish: {recipe} minus plain",
                         f"{name}|hinglish_paired_test", f"{plain}|hinglish_paired_test"))
    fixed = model_name("fixed", MAIN)
    noisy = model_name("noisy", MAIN)
    muril = model_name("fixed", MURIL)
    if fixed in names and noisy in names:
        rows.append(("On Hinglish: noisy minus fixed (does spelling noise help?)",
                     f"{noisy}|hinglish_paired_test", f"{fixed}|hinglish_paired_test"))
    if fixed in names and muril in names:
        for set_name in ["hinglish_paired_test", "devanagari_paired_test"]:
            rows.append((f"{set_name.split('_')[0].title()}: HingBERT minus MuRIL (both fixed)",
                         f"{fixed}|{set_name}", f"{muril}|{set_name}"))

    lines.append("## 2-3. The drop and the ablations (points; 95% range)\n")
    lines.append("| comparison | intent | slot F1 |")
    lines.append("|---|---|---|")
    results["differences"] = {}
    for label, key_a, key_b in rows:
        d = difference(boot, key_a, key_b)
        results["differences"][label] = d
        lines.append(f"| {label} | {with_range(d['intent'], d['intent_95'])} | "
                     f"{with_range(d['slot_f1'], d['slot_f1_95'])} |")
        notice(label, f"intent {with_range(d['intent'], d['intent_95'])}; "
                      f"slot F1 {with_range(d['slot_f1'], d['slot_f1_95'])}")
    lines.append("\nA range that does not include 0 means the difference is real, not luck of which "
                 "sentences were picked.\n")


def by_band(models, tables, ids, bands, lines, results):
    lines.append("## 4. The degradation curve: accuracy by how mixed the sentence is (CMI band)\n")
    lines.append("| model | band | sentences | Devanagari intent | Hinglish intent | drop |")
    lines.append("|---|---|---|---|---|---|")
    results["bands"] = {}
    chart = {}
    for band_index, band_row in enumerate(BANDS):
        band_name = band_row[0]
        band_ids = []
        for i in ids:
            if bands[i] == band_name:
                band_ids.append(i)
        if len(band_ids) < 2:
            continue
        keys = []
        for recipe, backbone in models:
            name = model_name(recipe, backbone)
            keys.append(f"{name}|devanagari_paired_test")
            keys.append(f"{name}|hinglish_paired_test")
        boot = bootstrap(tables, keys, band_ids, SEED + band_index)
        for recipe, backbone in models:
            name = model_name(recipe, backbone)
            dev_key = f"{name}|devanagari_paired_test"
            hin_key = f"{name}|hinglish_paired_test"
            d = difference(boot, dev_key, hin_key)
            entry = {
                "n": len(band_ids),
                "devanagari": boot[dev_key]["point"][0], "devanagari_95": interval(boot[dev_key]["intent"]),
                "hinglish": boot[hin_key]["point"][0], "hinglish_95": interval(boot[hin_key]["intent"]),
                "drop": d["intent"], "drop_95": d["intent_95"],
            }  # fmt: skip
            results["bands"][f"{name}|{band_name}"] = entry
            if name not in chart:
                chart[name] = []
            chart[name].append((band_index, entry))
            lines.append(f"| {name} | {band_name} | {len(band_ids)} | {pct(entry['devanagari'])} | "
                         f"{with_range(entry['hinglish'], entry['hinglish_95'])} | "
                         f"{with_range(entry['drop'], entry['drop_95'])} |")
    lines.append("")

    fig, ax = new_chart(7, 4, "Intent accuracy on Hinglish, by how mixed the sentence is")
    shown = [(model_name("plain", MAIN), "plain (Devanagari only)", 0),
             (model_name("fixed", MAIN), "fixed (+ romanized + English)", 1)]
    for name, label, colour in shown:
        if name not in chart:
            continue
        xs = []
        ys = []
        lows = []
        highs = []
        for band_index, entry in chart[name]:
            xs.append(band_index)
            ys.append(100 * entry["hinglish"])
            lows.append(100 * (entry["hinglish"] - entry["hinglish_95"][0]))
            highs.append(100 * (entry["hinglish_95"][1] - entry["hinglish"]))
        ax.errorbar(xs, ys, yerr=[lows, highs], color=COLOURS[colour], linewidth=2, marker="o",
                    markersize=6, capsize=3, label=label)
    plain = model_name("plain", MAIN)
    if plain in chart:
        xs = []
        ys = []
        for band_index, entry in chart[plain]:
            xs.append(band_index)
            ys.append(100 * entry["devanagari"])
        ax.plot(xs, ys, color=MUTED, linewidth=1.5, linestyle="--", label="plain, same sentences in Devanagari")
    ax.set_xticks(range(len(BANDS)))
    names = []
    for band_row in BANDS:
        names.append(band_row[0])
    ax.set_xticklabels(names)
    ax.set_xlabel("code-mixing band (CMI)", color=MUTED)
    ax.set_ylabel("intent accuracy %", color=MUTED)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    save_chart(fig, OUT / "degradation.png")


def checks(models, tables, ids, lines, results):
    lines.append("## 5. Checks\n")
    lines.append("**Seed to seed** (Hinglish paired intent accuracy for each seed):\n")
    lines.append("| model | " + " | ".join(f"seed {s}" for s in SEEDS) + " |")
    lines.append("|---|" + "---|" * len(SEEDS))
    results["seeds"] = {}
    for recipe, backbone in models:
        cells = []
        values = []
        for seed in SEEDS:
            metrics = json.loads((run_folder(recipe, backbone, seed) / "metrics.json").read_text(encoding="utf-8"))
            value = metrics["hinglish_paired_test"]["intent_accuracy"]
            values.append(value)
            cells.append(pct(value))
        results["seeds"][model_name(recipe, backbone)] = values
        lines.append(f"| {model_name(recipe, backbone)} | " + " | ".join(cells) + " |")

    train_texts = set()
    for row in load("hi-IN"):
        if row["partition"] == "train":
            train_texts.add(row["utt"].strip())
    clean_ids = []
    first = model_name(models[0][0], models[0][1])
    for i in ids:
        if " ".join(tables[f"{first}|devanagari_paired_test"][i]["words"]) not in train_texts:
            clean_ids.append(i)
    lines.append(f"\n**Without the {len(ids) - len(clean_ids)} paired sentences whose Hindi text also "
                 f"appears in training** ({len(clean_ids)} left):\n")
    lines.append("| model | Devanagari intent | Hinglish intent | drop |")
    lines.append("|---|---|---|---|")
    results["without_train_repeats"] = {}
    for recipe, backbone in models:
        name = model_name(recipe, backbone)
        dev_entries = []
        hin_entries = []
        for i in clean_ids:
            dev_entries.append(tables[f"{name}|devanagari_paired_test"][i])
            hin_entries.append(tables[f"{name}|hinglish_paired_test"][i])
        dev_acc = summarize(dev_entries, len(SEEDS))[0]
        hin_acc = summarize(hin_entries, len(SEEDS))[0]
        results["without_train_repeats"][name] = {"devanagari": dev_acc, "hinglish": hin_acc}
        lines.append(f"| {name} | {pct(dev_acc)} | {pct(hin_acc)} | {pct(dev_acc - hin_acc)} |")
    lines.append("")


def curves(models, lines):
    """Average the per-epoch numbers over the 3 seeds; draw loss and Hinglish dev accuracy."""
    fig, left, right = new_pair(10, 3.8, "Training loss", "Hinglish dev intent accuracy %")
    lines.append("## 6. Learning curves (mean of 3 seeds)\n")
    lines.append("| model | epochs | last train loss | dev intent | Hinglish dev intent |")
    lines.append("|---|---|---|---|---|")
    for index, (recipe, backbone) in enumerate(models):
        sums = {}
        for seed in SEEDS:
            path = run_folder(recipe, backbone, seed) / "curves.csv"
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    epoch = int(row["epoch"])
                    if epoch not in sums:
                        sums[epoch] = {"loss": 0.0, "dev": 0.0, "hinglish": 0.0}
                    sums[epoch]["loss"] += float(row["train_loss"]) / len(SEEDS)
                    sums[epoch]["dev"] += float(row["dev_intent_accuracy"]) / len(SEEDS)
                    sums[epoch]["hinglish"] += float(row["hinglish_dev_intent_accuracy"]) / len(SEEDS)
        epochs = sorted(sums)
        losses = []
        hinglish = []
        for epoch in epochs:
            losses.append(sums[epoch]["loss"])
            hinglish.append(100 * sums[epoch]["hinglish"])
        name = model_name(recipe, backbone)
        left.plot(epochs, losses, color=COLOURS[index], linewidth=2, marker="o", markersize=4, label=name)
        right.plot(epochs, hinglish, color=COLOURS[index], linewidth=2, marker="o", markersize=4, label=name)
        last = sums[epochs[-1]]
        lines.append(f"| {name} | {len(epochs)} | {last['loss']:.3f} | {pct(last['dev'])} | "
                     f"{pct(last['hinglish'])} |")
    left.set_xlabel("epoch", color=MUTED)
    right.set_xlabel("epoch", color=MUTED)
    right.set_ylim(0, 100)
    right.legend(frameon=False, fontsize=7)
    save_chart(fig, OUT / "curves.png")
    lines.append("")


def errors(tables, bands, ids):
    """Real sentences: what goes wrong, written for a person to read."""
    plain = model_name("plain", MAIN)
    fixed = model_name("fixed", MAIN)
    lines = ["# Error analysis (written by `python -m cmx evaluate`)\n"]
    lines.append("Majority vote over the 3 seeds. 'Right' = at least 2 of 3 seeds got it.\n")
    for name in [plain, fixed]:
        if f"{name}|hinglish_paired_test" not in tables:
            continue
        dev = tables[f"{name}|devanagari_paired_test"]
        hin = tables[f"{name}|hinglish_paired_test"]
        only_hinglish = []
        for i in ids:
            if dev[i]["right"] >= 2 and hin[i]["right"] < 2:
                only_hinglish.append(i)
        lines.append(f"## {name}: right in Devanagari, wrong in Hinglish ({len(only_hinglish)} of {len(ids)})\n")
        lines.append("| CMI band | Hinglish sentence | true intent | guessed |")
        lines.append("|---|---|---|---|")
        for i in only_hinglish[:12]:
            wrong_guesses = []
            for guess in hin[i]["guesses"]:
                if guess != hin[i]["intent"]:
                    wrong_guesses.append(guess)
            lines.append(f"| {bands[i]} | {' '.join(hin[i]['words'])} | {hin[i]['intent']} | "
                         f"{most_common(wrong_guesses)} |")

        confusions = {}
        guessed_when_wrong = {}
        for i in ids:
            for guess in hin[i]["guesses"]:
                if guess != hin[i]["intent"]:
                    pair = f"{hin[i]['intent']} -> {guess}"
                    confusions[pair] = confusions.get(pair, 0) + 1
                    guessed_when_wrong[guess] = guessed_when_wrong.get(guess, 0) + 1
        lines.append(f"\n**Most common mix-ups on Hinglish ({name}, counted over 3 seeds):**\n")
        for pair, count in sorted(confusions.items(), key=by_count, reverse=True)[:8]:
            lines.append(f"- {pair}: {count}")
        lines.append("\n**When wrong, what does it guess most?**\n")
        for guess, count in sorted(guessed_when_wrong.items(), key=by_count, reverse=True)[:5]:
            lines.append(f"- {guess}: {count}")

        # Slot recall per slot type: Devanagari vs Hinglish
        found = {}
        for set_name, table in [("devanagari", dev), ("hinglish", hin)]:
            for i in ids:
                entry = table[i]
                true_spans = slot_spans(entry["tags"])
                for guessed_tags in entry["guessed_tags"]:
                    guessed_spans = slot_spans(guessed_tags)
                    for span in true_spans:
                        kind = span[2]
                        if kind not in found:
                            found[kind] = {"devanagari": [0, 0], "hinglish": [0, 0]}
                        found[kind][set_name][1] += 1
                        if span in guessed_spans:
                            found[kind][set_name][0] += 1
        gaps = []
        for kind, value in found.items():
            if value["hinglish"][1] >= 15 and value["devanagari"][1] > 0:
                dev_recall = value["devanagari"][0] / value["devanagari"][1]
                hin_recall = value["hinglish"][0] / value["hinglish"][1]
                gaps.append((kind, dev_recall, hin_recall, value["hinglish"][1] // len(SEEDS)))
        gaps.sort(key=by_gap, reverse=True)
        lines.append("\n**Slot types that suffer most in Hinglish (share of slots found):**\n")
        lines.append("| slot | slots in test | Devanagari | Hinglish |")
        lines.append("|---|---|---|---|")
        for kind, dev_recall, hin_recall, n in gaps[:8]:
            lines.append(f"| {kind} | {n} | {pct(dev_recall)} | {pct(hin_recall)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def by_count(pair):
    return pair[1]


def by_gap(row):
    return row[1] - row[2]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    models = []
    missing = []
    for recipe, backbone in MODELS:
        if is_complete(recipe, backbone):
            models.append((recipe, backbone))
        else:
            missing.append(model_name(recipe, backbone))
    if not models:
        print("No finished runs downloaded yet (Kaggle still training?) - nothing to evaluate.")
        notice("evaluate", "no finished runs yet")
        return
    if missing:
        print(f"Not finished yet, left out: {missing}")
        notice("evaluate", f"left out (not finished on Kaggle yet): {', '.join(missing)}")

    tables = {}
    for recipe, backbone in models:
        for set_name in SETS:
            tables[f"{model_name(recipe, backbone)}|{set_name}"] = per_sentence(recipe, backbone, set_name)

    first = model_name(models[0][0], models[0][1])
    ids = sorted(tables[f"{first}|hinglish_paired_test"])
    keys = []
    for recipe, backbone in models:
        for set_name in ["devanagari_paired_test", "hinglish_paired_test"]:
            key = f"{model_name(recipe, backbone)}|{set_name}"
            if sorted(tables[key]) != ids:
                raise SystemExit(f"{key} does not have the same sentence ids - can't pair them")
            keys.append(key)

    lists = build_word_lists()
    bands = {}
    for i in ids:
        cmi = sentence_features(tables[f"{first}|hinglish_paired_test"][i]["words"], lists)["cmi"]
        bands[i] = band(cmi)

    print(f"bootstrap: {DRAWS} draws over {len(ids)} paired sentences, {len(models)} models ...", flush=True)
    boot = bootstrap(tables, keys, ids, SEED)

    results = {"seeds": SEEDS, "draws": DRAWS, "paired_sentences": len(ids), "missing": missing}
    lines = ["# Results (written by `python -m cmx evaluate`)\n"]
    headline(models, tables, boot, lines, results)
    differences(models, boot, lines, results)
    by_band(models, tables, ids, bands, lines, results)
    checks(models, tables, ids, lines, results)
    curves(models, lines)

    report = "\n".join(lines) + "\n"
    print(report)
    (OUT / "results.md").write_text(report, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    error_report = errors(tables, bands, ids)
    (OUT / "errors.md").write_text(error_report, encoding="utf-8")
    print(error_report)
    print(f"saved {OUT}/results.md, results.json, errors.md, degradation.png, curves.png")
