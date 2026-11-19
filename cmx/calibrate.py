"""Stage 6: is the model's confidence honest? (CPU only, no torch.)

"Calibrated" = when the model says 90% sure, it is right about 90% of the time.
ECE (expected calibration error): sort answers into 10 confidence bins (0-10%, ..., 90-100%);
in each bin compare average confidence with actual accuracy; average the gaps, weighted by bin size.
0 = perfectly honest. 0.20 = typically 20 points over- or under-confident.

Temperature scaling (the standard fix): one number T. p -> p^(1/T), then renormalise.
T > 1 softens over-confident answers. T is fitted on DEV sentences only, two ways:
    T(Devanagari dev)      what you would do if you never looked at Hinglish
    T(dev + Hinglish dev)  the same, but the 392 Hinglish dev sentences join in
Both are then applied unchanged to the test sentences, and ECE is reported per CMI band.

Abstention: below a confidence cut-off the assistant should ask "did you mean ...?" instead of
acting. The cut-off is picked on dev (the lowest one that keeps answered dev sentences 90% right),
then we report on test how many sentences get answered and how many of those are right.
"""

import json
import math
from pathlib import Path

from cmx.charts import COLOURS, MUTED, new_pair, save_chart
from cmx.evaluate import (
    MAIN,
    MODELS,
    SEEDS,
    is_complete,
    load_predictions,
    model_name,
    run_folder,
)
from cmx.features import BANDS, band, build_word_lists, sentence_features
from cmx.quick import notice

BINS = 10
TEMPERATURES = [round(0.5 + 0.05 * k, 2) for k in range(51)]  # 0.5, 0.55, ..., 3.0
FLOOR = 1e-6  # probabilities were saved rounded to 5 decimals, so some are 0
TARGET = 0.90  # answered sentences should be right at least this often
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


def reliability(confidences, correct):
    """Per bin: (average confidence, accuracy, count) - the points of a reliability diagram."""
    points = []
    for b in range(BINS):
        conf_sum = 0.0
        right = 0
        count = 0
        for i, value in enumerate(confidences):
            if min(int(value * BINS), BINS - 1) == b:
                conf_sum += value
                count += 1
                if correct[i]:
                    right += 1
        if count:
            points.append((conf_sum / count, right / count, count))
    return points


def to_case(row, names):
    """One prediction -> log-probabilities, index of the right intent, and whether it was right."""
    logs = []
    for p in row["probs"]:
        logs.append(math.log(max(p, FLOOR)))
    return {"id": row["id"], "logs": logs, "answer": names.index(row["intent"]),
            "right": row["predicted"] == row["intent"], "words": row["words"]}  # fmt: skip


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


def pick_cutoff(confidences, correct):
    """The lowest confidence cut-off whose answered sentences are at least TARGET right."""
    for step in range(100):
        cutoff = step / 100
        answered = 0
        right = 0
        for i, value in enumerate(confidences):
            if value >= cutoff:
                answered += 1
                if correct[i]:
                    right += 1
        if answered and right / answered >= TARGET:
            return cutoff
    return 1.0


def answered_share(confidences, correct, cutoff):
    """-> (share answered, accuracy of the answered ones)."""
    answered = 0
    right = 0
    for i, value in enumerate(confidences):
        if value >= cutoff:
            answered += 1
            if correct[i]:
                right += 1
    if answered == 0:
        return 0.0, 0.0
    return answered / len(confidences), right / answered


def pct(x):
    return f"{100 * x:.1f}"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lists = build_word_lists()
    shown = []
    for recipe, backbone in MODELS:
        if is_complete(recipe, backbone):
            shown.append((recipe, backbone))
    if not shown:
        print("No finished runs downloaded yet - nothing to calibrate.")
        return

    report = {}
    lines = ["# Calibration (written by `python -m cmx calibrate`)\n"]
    lines.append("3 seeds pooled. T fitted on dev only; test sentences never touch the fit.\n")
    lines.append("| model | test set | accuracy | ECE before | ECE, T from Devanagari dev | "
                 "ECE, T from dev + Hinglish dev |")
    lines.append("|---|---|---|---|---|---|")
    diagram = {}
    abstain_lines = []
    for recipe, backbone in shown:
        name = model_name(recipe, backbone)
        groups = {}  # group name -> {"before": [], "dev_t": [], "mixed_t": [], "right": []}
        temperatures = []
        for seed in SEEDS:
            metrics = json.loads((run_folder(recipe, backbone, seed) / "metrics.json").read_text(encoding="utf-8"))
            names = metrics["intent_names"]
            cases = {}
            for set_name in ["dev", "hinglish_paired_dev", "test", "devanagari_paired_test",
                             "hinglish_paired_test"]:
                cases[set_name] = []
                for row in load_predictions(recipe, backbone, seed, set_name):
                    cases[set_name].append(to_case(row, names))
            t_dev = fit_temperature(cases["dev"])
            t_mixed = fit_temperature(cases["dev"] + cases["hinglish_paired_dev"])
            temperatures.append((t_dev, t_mixed))

            # cut-off for "ask instead of act", picked on both dev sets with the mixed T
            dev_conf, dev_right = describe(cases["dev"] + cases["hinglish_paired_dev"], t_mixed)
            cutoff = pick_cutoff(dev_conf, dev_right)

            targets = [("Hindi test", cases["test"]),
                       ("Devanagari paired", cases["devanagari_paired_test"]),
                       ("Hinglish paired", cases["hinglish_paired_test"])]
            for band_row in BANDS:
                in_band = []
                for case in cases["hinglish_paired_test"]:
                    if band(sentence_features(case["words"], lists)["cmi"]) == band_row[0]:
                        in_band.append(case)
                targets.append((f"Hinglish, {band_row[0]} mixing", in_band))
            for group, group_cases in targets:
                if group not in groups:
                    groups[group] = {"before": [], "dev_t": [], "mixed_t": [], "right": [], "cutoffs": []}
                before, right = describe(group_cases, 1.0)
                with_dev_t, _ = describe(group_cases, t_dev)
                with_mixed_t, _ = describe(group_cases, t_mixed)
                groups[group]["before"] += before
                groups[group]["dev_t"] += with_dev_t
                groups[group]["mixed_t"] += with_mixed_t
                groups[group]["right"] += right
                groups[group]["cutoffs"].append(cutoff)

        report[name] = {"temperatures (Devanagari dev, dev + Hinglish dev)": temperatures, "groups": {}}
        for group, g in groups.items():
            if not g["right"]:
                continue
            row = {
                "n": len(g["right"]) // len(SEEDS),
                "accuracy": sum(g["right"]) / len(g["right"]),
                "ece_before": ece(g["before"], g["right"]),
                "ece_dev_t": ece(g["dev_t"], g["right"]),
                "ece_mixed_t": ece(g["mixed_t"], g["right"]),
            }
            # abstention: each seed's own cut-off, pooled
            cutoff = sum(g["cutoffs"]) / len(g["cutoffs"])
            share, answered_accuracy = answered_share(g["mixed_t"], g["right"], cutoff)
            row["cutoff"] = cutoff
            row["answered_share"] = share
            row["answered_accuracy"] = answered_accuracy
            report[name]["groups"][group] = row
            lines.append(f"| {name} | {group} (n={row['n']}) | {pct(row['accuracy'])} | "
                         f"{row['ece_before']:.3f} | {row['ece_dev_t']:.3f} | {row['ece_mixed_t']:.3f} |")
            abstain_lines.append(f"| {name} | {group} | {cutoff:.2f} | {pct(share)} | {pct(answered_accuracy)} |")
            if group == "Hinglish paired" and recipe in ["plain", "fixed"] and backbone == MAIN:
                diagram[recipe] = {
                    "before": reliability(g["before"], g["right"]),
                    "after": reliability(g["mixed_t"], g["right"]),
                }
                notice(f"calibration {name} Hinglish",
                       f"accuracy {pct(row['accuracy'])}%; ECE {row['ece_before']:.3f} -> "
                       f"{row['ece_dev_t']:.3f} (T from Devanagari dev) / {row['ece_mixed_t']:.3f} "
                       f"(T from dev + Hinglish dev); answers {pct(share)}% at {pct(answered_accuracy)}% right")

    lines.append("\nT per seed, (from Devanagari dev, from dev + Hinglish dev):\n")
    for name, value in report.items():
        lines.append(f"- {name}: {value['temperatures (Devanagari dev, dev + Hinglish dev)']}")
    lines.append(f"\n## Ask instead of act: confidence cut-off picked on dev to keep answers "
                 f"{pct(TARGET)}% right\n")
    lines.append("| model | test set | cut-off | share answered | answered right % |")
    lines.append("|---|---|---|---|---|")
    lines += abstain_lines

    if diagram:
        fig, left, right = new_pair(10, 4.2, "plain: Hinglish paired test", "fixed: Hinglish paired test")
        for recipe, ax in [("plain", left), ("fixed", right)]:
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="honest")
            if recipe in diagram:
                for index, (key, label) in enumerate([("before", "before"), ("after", "after scaling")]):
                    xs = []
                    ys = []
                    for conf, accuracy, _ in diagram[recipe][key]:
                        xs.append(conf)
                        ys.append(accuracy)
                    ax.plot(xs, ys, color=COLOURS[index], linewidth=2, marker="o", markersize=5, label=label)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("how sure the model said it was", color=MUTED)
            ax.set_ylabel("how often it was right", color=MUTED)
            ax.legend(frameon=False, fontsize=8)
        save_chart(fig, OUT / "reliability.png")

    table = "\n".join(lines) + "\n"
    print(table)
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "results.md").write_text(table, encoding="utf-8")
    print(f"saved {OUT}/results.json, results.md and reliability.png")
