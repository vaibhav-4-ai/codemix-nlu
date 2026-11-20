"""Stage 7b: does the model work on Hinglish written by OTHER people, for a DIFFERENT dataset?

Google's Hinglish-TOP (Agarwal et al. 2022, Apache-2.0): requests to a voice assistant, written
by people in English AND in Hinglish. Only the human-written files are used (the dataset also has
170K machine-generated sentences, which are left out on purpose).

Its labels are not MASSIVE's labels, so each Hinglish-TOP intent is mapped to a MASSIVE intent
ONLY when the meaning is the same (checked against how MASSIVE labels similar English requests).
Everything else is dropped, and every dropped intent is listed with its reason.

Scored, all on the same mapped sentences:
    floor     always guess the most common intent
    TF-IDF    the simple baseline, plain and fixed recipe
    int8      the shrunk main model (artifacts/export, made by `python -m cmx export`)
    ceiling   the int8 model on MASSIVE's own Hinglish test, restricted to the same intents
Each in Hinglish and in English, so the English -> Hinglish drop is measured on new sentences.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy

from cmx.examples import test_sets, training_examples
from cmx.export import OnnxModel
from cmx.features import BANDS, band, build_word_lists, sentence_features
from cmx.quick import make_model, notice

COMMIT = "fdd3998a6573130659bfa1ce4b1ebe698df2bf3a"  # pinned, so the file can never change under us
URL = (
    "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/"
    f"{COMMIT}/Dataset/Human%20Annotated%20Data/test.tsv"
)
SHA256 = "9f7bd4f80b15956143b4188a113981735264944ac825d6231b60cc2444533791"
OUT = Path("artifacts") / "transfer"
EXPORTED = Path("artifacts") / "export"  # made by `python -m cmx export`
MAIN_MODEL = "main model (fixed, hing-mbert), int8"
DRAWS = 2000
SEED = 13

MAPPING = {
    "GET_WEATHER": "weather_query",
    "GET_SUNSET": "weather_query",  # MASSIVE labels "when is sunset" as weather_query
    "CREATE_ALARM": "alarm_set",
    "GET_ALARM": "alarm_query",
    "DELETE_ALARM": "alarm_remove",
    "CREATE_REMINDER": "calendar_set",  # MASSIVE puts "remind me to ..." under calendar
    "GET_REMINDER": "calendar_query",
    "DELETE_REMINDER": "calendar_remove",
    "GET_EVENT": "recommendation_events",  # "events happening in Phoenix", not my calendar
    "GET_INFO_TRAFFIC": "transport_traffic",
    "GET_DIRECTIONS": "transport_query",
    "GET_ESTIMATED_DURATION": "transport_query",
    "PLAY_MUSIC": "play_music",
    "LIKE_MUSIC": "music_likeness",
    "DISLIKE_MUSIC": "music_dislikeness",
    "START_SHUFFLE_MUSIC": "music_settings",
    "LOOP_MUSIC": "music_settings",
}


def drop_reason(intent):
    """Why a Hinglish-TOP intent has no MASSIVE partner."""
    if intent.startswith("UNSUPPORTED"):
        return "marked unsupported by the dataset itself"
    if "TIMER" in intent:
        return "MASSIVE has no timer intent"
    if "MESSAGE" in intent:
        return "MASSIVE splits messages across email and social intents"
    return "no single MASSIVE intent means the same"


def words_of(text):
    """'Kya hawaii me raining ho rahi hai?' -> ['kya', 'hawaii', 'me', 'raining', 'ho', 'rahi', 'hai']"""
    words = []
    for word in text.lower().split():
        word_clean = word.strip(".,?!'\"")
        if word_clean:
            words.append(word_clean)
    return words


def load_rows():
    with urllib.request.urlopen(URL) as response:
        blob = response.read()
    if hashlib.sha256(blob).hexdigest() != SHA256:
        raise SystemExit("Hinglish-TOP test file does not match the pinned checksum")
    rows = []
    for line in blob.decode("utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 5:
            top_intent = parts[3].split()[0].replace("[IN:", "")
            rows.append({"english": parts[0], "hinglish": parts[1], "top_intent": top_intent})
    return rows


def accuracy(guesses, truth):
    right = 0
    for k, guess in enumerate(guesses):
        if guess == truth[k]:
            right += 1
    return right / len(truth)


def right_flags(guesses, truth):
    """1 for each sentence the model got right, 0 for each it got wrong."""
    flags = []
    for k, guess in enumerate(guesses):
        flags.append(1 if guess == truth[k] else 0)
    return flags


def macro_f1(guesses, truth):
    """F1 for each true intent, then the plain average (a small intent counts as much as a big one)."""
    counts = {}
    for k, guess in enumerate(guesses):
        for name in [truth[k], guess]:
            if name not in counts:
                counts[name] = {"hit": 0, "true": 0, "guessed": 0}
        counts[truth[k]]["true"] += 1
        counts[guess]["guessed"] += 1
        if guess == truth[k]:
            counts[guess]["hit"] += 1
    scores = []
    for value in counts.values():
        if value["true"] > 0:
            scores.append(2 * value["hit"] / (value["true"] + value["guessed"]))
    return sum(scores) / len(scores)


def middle_95(values):
    ordered = sorted(values)
    return [float(ordered[int(0.025 * len(ordered))]), float(ordered[int(0.975 * len(ordered)) - 1])]


def paired_ranges(english_flags, hinglish_flags):
    """95% ranges for English accuracy, Hinglish accuracy and the drop, from 2,000 paired resamples.

    Each resample picks n sentence positions at random (with repeats) and uses the SAME positions
    for both columns, so the range of the drop is honest and narrow.
    numpy does the counting: `flags[picks]` looks up the 0/1 of every picked sentence at once.
    """
    english = numpy.array(english_flags)
    hinglish = numpy.array(hinglish_flags)
    rng = numpy.random.default_rng(SEED)
    picks = rng.integers(0, len(english), size=(DRAWS, len(english)))
    english_means = english[picks].mean(axis=1)
    hinglish_means = hinglish[picks].mean(axis=1)
    return {
        "english_95": middle_95(english_means),
        "hinglish_95": middle_95(hinglish_means),
        "drop_95": middle_95(english_means - hinglish_means),
    }


def pct(x):
    return f"{100 * x:.1f}"


def with_range(value, low_high):
    """'85.3 (84.2 to 86.4)', or just '85.3' when there is no range."""
    if low_high is None:
        return pct(value)
    return f"{pct(value)} ({pct(low_high[0])} to {pct(low_high[1])})"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    kept = []
    dropped = {}
    kept_counts = {}
    for row in rows:
        if row["top_intent"] in MAPPING:
            row["intent"] = MAPPING[row["top_intent"]]
            kept.append(row)
            kept_counts[row["top_intent"]] = kept_counts.get(row["top_intent"], 0) + 1
        else:
            dropped[row["top_intent"]] = dropped.get(row["top_intent"], 0) + 1
    truth = []
    for row in kept:
        truth.append(row["intent"])

    scores = {}
    # floor: always the most common answer
    counts = {}
    for label in truth:
        counts[label] = counts.get(label, 0) + 1
    common = None
    for label, count in counts.items():
        if common is None or count > counts[common]:
            common = label
    scores["floor: always guess " + common] = {"hinglish": counts[common] / len(truth),
                                                "english": counts[common] / len(truth)}  # fmt: skip

    guesses_by_model = {}
    # TF-IDF baselines
    for recipe in ["plain", "fixed"]:
        train_texts = []
        train_labels = []
        for example in training_examples(recipe):
            train_texts.append(" ".join(example["words"]))
            train_labels.append(example["intent"])
        model = make_model().fit(train_texts, train_labels)
        name = f"TF-IDF + logistic regression, {recipe}"
        guesses_by_model[name] = {}
        for side in ["hinglish", "english"]:
            texts = []
            for row in kept:
                texts.append(" ".join(words_of(row[side])))
            guesses_by_model[name][side] = list(model.predict(texts))

    # the shrunk main model
    small = OnnxModel(EXPORTED)
    guesses_by_model[MAIN_MODEL] = {}
    for side in ["hinglish", "english"]:
        guesses = []
        for row in kept:
            guesses.append(small.predict(words_of(row[side]))["intent"])
        guesses_by_model[MAIN_MODEL][side] = guesses
    guesses_hinglish = guesses_by_model[MAIN_MODEL]["hinglish"]

    # accuracy, macro-F1 and paired 95% ranges for every model
    for name, by_side in guesses_by_model.items():
        result = {}
        flags = {}
        for side in ["hinglish", "english"]:
            result[side] = accuracy(by_side[side], truth)
            result[f"{side}_macro_f1"] = macro_f1(by_side[side], truth)
            flags[side] = right_flags(by_side[side], truth)
        ranges = paired_ranges(flags["english"], flags["hinglish"])
        result.update(ranges)
        scores[name] = result

    # ceiling: same model, MASSIVE's own Hinglish test, same intents only
    mapped_intents = set(MAPPING.values())
    in_domain = []
    for example in test_sets()["hinglish_paired_test"]:
        if example["intent"] in mapped_intents:
            in_domain.append(example)
    ceiling_guesses = []
    ceiling_truth = []
    for example in in_domain:
        ceiling_guesses.append(small.predict(example["words"])["intent"])
        ceiling_truth.append(example["intent"])
    ceiling = accuracy(ceiling_guesses, ceiling_truth)
    ceiling_flags = right_flags(ceiling_guesses, ceiling_truth)
    ceiling_95 = paired_ranges(ceiling_flags, ceiling_flags)["english_95"]  # one column: both sides the same

    # by CMI band, Hinglish side
    lists = build_word_lists()
    band_scores = {}
    for band_row in BANDS:
        band_scores[band_row[0]] = [0, 0]
    for k, row in enumerate(kept):
        name = band(sentence_features(words_of(row["hinglish"]), lists)["cmi"])
        band_scores[name][1] += 1
        if guesses_hinglish[k] == truth[k]:
            band_scores[name][0] += 1

    lines = ["# Transfer: Google's Hinglish-TOP, human-written test file (written by `python -m cmx transfer`)\n"]
    lines.append(f"{len(rows):,} sentences in the file; {len(kept):,} kept ({len(MAPPING)} intents mapped), "
                 f"{len(rows) - len(kept):,} dropped. Intent only: Hinglish-TOP's slot labels are a "
                 "different scheme.\n")
    lines.append("Intent accuracy, with 95% ranges from 2,000 paired resamples of the sentences; macro-F1 "
                 "(every intent counts the same) in the last two columns.\n")
    lines.append("| model | English | Hinglish | drop | macro-F1 English | macro-F1 Hinglish |")
    lines.append("|---|---|---|---|---|---|")
    for name, result in scores.items():
        cells = []
        for side in ["english", "hinglish"]:
            cells.append(with_range(result[side], result.get(f"{side}_95")))
        cells.append(with_range(result["english"] - result["hinglish"], result.get("drop_95")))
        for side in ["english", "hinglish"]:
            value = result.get(f"{side}_macro_f1")
            cells.append("-" if value is None else pct(value))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append(f"\nCeiling - the same int8 model on MASSIVE's own Hinglish test, same {len(mapped_intents)} "
                 f"intents ({len(in_domain)} sentences): {with_range(ceiling, ceiling_95)}. The gap to the "
                 "Hinglish column is the cost of new writers, new topics and a different labelling style.\n")
    lines.append("**Main model on Hinglish-TOP, by CMI band:**\n")
    lines.append("| band | sentences | intent accuracy |")
    lines.append("|---|---|---|")
    for name, (right, total) in band_scores.items():
        if total:
            lines.append(f"| {name} | {total} | {pct(right / total)} |")
    lines.append("\n## Intent mapping\n")
    lines.append("| Hinglish-TOP intent | MASSIVE intent | sentences |")
    lines.append("|---|---|---|")
    for top_intent, massive_intent in MAPPING.items():
        lines.append(f"| {top_intent} | {massive_intent} | {kept_counts.get(top_intent, 0)} |")
    lines.append("\n## Dropped intents (every one, with the reason)\n")
    lines.append("| Hinglish-TOP intent | sentences | reason |")
    lines.append("|---|---|---|")
    for top_intent, count in sorted(dropped.items(), key=by_count, reverse=True):
        lines.append(f"| {top_intent} | {count} | {drop_reason(top_intent)} |")
    lines.append("\nSource: Agarwal et al. 2022, CST5: Data Augmentation for Code-Switched Semantic Parsing "
                 f"(arXiv 2211.07514). File pinned to commit {COMMIT[:7]}, checksum checked.")

    report = "\n".join(lines) + "\n"
    print(report)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    results = {"kept": len(kept), "dropped": len(rows) - len(kept), "scores": scores, "ceiling": ceiling,
               "ceiling_95": ceiling_95, "ceiling_sentences": len(in_domain), "bands": band_scores}  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    main = scores[MAIN_MODEL]
    notice("Hinglish-TOP transfer", f"int8 main model: English {with_range(main['english'], main['english_95'])}"
                                    f"; Hinglish {with_range(main['hinglish'], main['hinglish_95'])}"
                                    f"; macro-F1 Hinglish {pct(main['hinglish_macro_f1'])}"
                                    f"; ceiling {with_range(ceiling, ceiling_95)}; n={len(kept)}")


def by_count(pair):
    return pair[1]
