"""Stage 6: is the model's confidence honest? (CPU only, no torch.)

"Calibrated" = when the model says 90% sure, it is right about 90% of the time.
ECE (expected calibration error): sort answers into 10 confidence bins (0-10%, ..., 90-100%);
in each bin compare average confidence with actual accuracy; average the gaps, weighted by bin size.
0 = perfectly honest. 0.20 = typically 20 points over- or under-confident.

Temperature scaling (the standard fix): one number T. p -> p^(1/T), then renormalise.
T > 1 softens over-confident answers. T is fitted on Devanagari test sentences only
(never the 50 paired ones), then applied unchanged to the 50 Devanagari and 50 Hinglish.
On the fitting data itself we cross-fit: fit on half A, measure on half B, and the reverse.
"""

import json
import math
import random
from pathlib import Path

from cmx.evaluate import RECIPES, RUNS, SEEDS, load_predictions
from cmx.quick import notice

BINS = 10
TEMPERATURES = [round(0.5 + 0.05 * k, 2) for k in range(51)]  # 0.5, 0.55, ..., 3.0
FLOOR = 1e-6  # probabilities were saved rounded to 5 decimals, so some are 0
SEED = 13
OUT = Path("artifacts") / "calibrate"


def ece(confidences, correct):
    """Expected calibration error with 10 equal-width bins."""
    totals = [0] * BINS
    conf_sums = [0.0] * BINS
    right_sums = [0] * BINS
    for i, value in enumerate(confidences):
        b = min(int(value * BINS), BINS - 1)
        totals[b] += 1
        conf_sums[b] += value
        if correct[i]:
            right_sums[b] += 1
    error = 0.0
    for b in range(BINS):
        if totals[b]:
            gap = abs(right_sums[b] / totals[b] - conf_sums[b] / totals[b])
            error += gap * totals[b] / len(confidences)
    return error


def to_case(row, names):
    """One prediction -> log-probabilities, index of the right intent, and whether it was right."""
    logs = []
    for p in row["probs"]:
        logs.append(math.log(max(p, FLOOR)))
    return {"logs": logs, "answer": names.index(row["intent"]), "right": row["predicted"] == row["intent"]}


def confidence(case, temperature):
    """Top probability after temperature scaling."""
    biggest = max(case["logs"])
    total = 0.0
    for value in case["logs"]:
        total += math.exp((value - biggest) / temperature)
    return 1.0 / total  # the top answer's share: exp(0) / total


def nll(cases, temperature):
    """Average negative log-likelihood of the right intent: lower = better fit."""
    loss = 0.0
    for case in cases:
        biggest = max(case["logs"])
        total = 0.0
        for value in case["logs"]:
            total += math.exp((value - biggest) / temperature)
        loss += -(case["logs"][case["answer"]] - biggest) / temperature + math.log(total)
    return loss / len(cases)


def fit_temperature(cases):
    best = TEMPERATURES[0]
    best_loss = nll(cases, best)
    for temperature in TEMPERATURES[1:]:
        loss = nll(cases, temperature)
        if loss < best_loss:
            best = temperature
            best_loss = loss
    return best


def describe(cases, temperature):
    confidences = []
    correct = []
    for case in cases:
        confidences.append(confidence(case, temperature))
        correct.append(case["right"])
    return confidences, correct


def run():
    report = {"temperatures": {}, "rows": {}}
    lines = [
        "Calibration: hing-mbert, 3 seeds pooled. T fitted on Devanagari test sentences (not the 50).",
        "",
        "| Model | Sentences | Accuracy % | Avg confidence % (before -> after) | ECE (before -> after) |",
        "|---|---|---|---|---|",
    ]
    for recipe in RECIPES:
        pooled = {}
        for name in ["test", "devanagari_50", "hinglish_50"]:
            pooled[name] = {"before": [], "after": [], "right": []}
        temperatures = []
        for seed in SEEDS:
            metrics = json.loads((RUNS / f"{recipe}_seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
            names = metrics["intent_names"]
            paired = {}
            for name in ["devanagari_50", "hinglish_50"]:
                paired[name] = []
                for row in load_predictions(recipe, seed, name):
                    paired[name].append(to_case(row, names))
            paired_ids = set()
            for row in load_predictions(recipe, seed, "hinglish_50"):
                paired_ids.add(row["id"])
            pool = []
            for row in load_predictions(recipe, seed, "test"):
                if row["id"] not in paired_ids:
                    pool.append(to_case(row, names))

            # Cross-fit on the pool: fit on one half, measure on the other.
            random.Random(SEED).shuffle(pool)
            half = len(pool) // 2
            part_a = pool[:half]
            part_b = pool[half:]
            for fit_on, measure_on in [(part_a, part_b), (part_b, part_a)]:
                temperature = fit_temperature(fit_on)
                before, right = describe(measure_on, 1.0)
                after, _ = describe(measure_on, temperature)
                pooled["test"]["before"] += before
                pooled["test"]["after"] += after
                pooled["test"]["right"] += right

            # One T from the whole pool, applied unchanged to the 50 + 50.
            temperature = fit_temperature(pool)
            temperatures.append(temperature)
            for name in ["devanagari_50", "hinglish_50"]:
                before, right = describe(paired[name], 1.0)
                after, _ = describe(paired[name], temperature)
                pooled[name]["before"] += before
                pooled[name]["after"] += after
                pooled[name]["right"] += right

        report["temperatures"][recipe] = temperatures
        for name in ["test", "devanagari_50", "hinglish_50"]:
            p = pooled[name]
            accuracy = sum(p["right"]) / len(p["right"])
            row = {
                "accuracy": accuracy,
                "confidence_before": sum(p["before"]) / len(p["before"]),
                "confidence_after": sum(p["after"]) / len(p["after"]),
                "ece_before": ece(p["before"], p["right"]),
                "ece_after": ece(p["after"], p["right"]),
            }
            report["rows"][f"{recipe} {name}"] = row
            label = "MASSIVE test, Devanagari (cross-fit)" if name == "test" else name
            lines.append(
                f"| {recipe} | {label} | {100 * accuracy:.1f}"
                f" | {100 * row['confidence_before']:.1f} -> {100 * row['confidence_after']:.1f}"
                f" | {row['ece_before']:.3f} -> {row['ece_after']:.3f} |"
            )
            notice(
                f"calibration {recipe} {name}",
                f"accuracy {100 * accuracy:.1f}%; confidence {100 * row['confidence_before']:.1f}% ->"
                f" {100 * row['confidence_after']:.1f}%; ECE {row['ece_before']:.3f} -> {row['ece_after']:.3f}",
            )
        notice(f"temperature {recipe}", f"T per seed {temperatures}")
    lines.append("")
    for recipe in RECIPES:
        lines.append(f"T per seed ({recipe}): {report['temperatures'][recipe]}")
    table = "\n".join(lines) + "\n"
    print(table)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "results.md").write_text(table, encoding="utf-8")
    print(f"saved {OUT}/results.json and results.md")
