"""The quick version: a first rough number in seconds, on CPU.

Question: a model trained on Hindi in Devanagari - how much worse is it at guessing the
intent when the same 50 sentences arrive as Hinglish?

Three simple models (TF-IDF over character pieces + logistic regression), trained on:
  1. Devanagari only            (what most Hindi models see)
  2. + the same sentences auto-converted to English letters   (the cheap fix)
  3. + the English versions too  (Hinglish borrows many English words)
"""

import json
import math
import os
import random
from pathlib import Path

from indic_transliteration import sanscript
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from cmx.data import DATA, load

SEED = 13


def romanize(text: str) -> str:
    """Devanagari -> English letters, mechanically: 'कमरे की बत्ती' -> 'kamare kii battii'."""
    return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


def bootstrap_ci(correct: list[bool], n: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """95% range for an accuracy: resample the sentences 2000 times, take the middle 95%."""
    rng = random.Random(seed)
    k = len(correct)
    accs = sorted(sum(rng.choice(correct) for _ in range(k)) / k for _ in range(n))
    return accs[int(0.025 * n)], accs[int(0.975 * n) - 1]


def make_model():
    # max_features keeps the model small enough (~4 MB) to run inside a web page.
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=10000
        ),
        LogisticRegression(max_iter=2000),
    )


def export(model, path: Path) -> None:
    """Save a model as plain numbers the demo page can use (it redoes the maths in JavaScript)."""
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    data = {
        "intents": [str(c) for c in clf.classes_],
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": [round(float(x), 4) for x in vec.idf_],
        "coef": [[round(float(x), 4) for x in row] for row in clf.coef_],
        "intercept": [round(float(x), 4) for x in clf.intercept_],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def predict_from_export(data: dict, text: str) -> str:
    """The same steps the demo page runs in JavaScript, in Python, so a test can check them.

    1. cut each word (padded with spaces) into 2-4 letter pieces: " room " -> " r", "ro", ...
    2. weight each known piece: (1 + log(count)) * idf, then scale the list to length 1
    3. score each intent: sum(weight * coef) + intercept; the highest score wins
    """
    counts: dict[int, int] = {}
    for word in text.lower().split():
        w = f" {word} "
        for n in range(2, 5):
            last = max(0, len(w) - n)
            for start in range(last + 1):
                idx = data["vocab"].get(w[start : start + n])
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            if last == 0:  # word shorter than n: scikit-learn counts it once and stops
                break
    weights = {i: (1 + math.log(c)) * data["idf"][i] for i, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
    scores = [
        b + sum(v / norm * row[i] for i, v in weights.items())
        for row, b in zip(data["coef"], data["intercept"], strict=True)
    ]
    return data["intents"][scores.index(max(scores))]


def notice(title: str, message: str) -> None:
    """On GitHub, also post the line as a note on the run page (Claude can read these)."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title={title}::{message}")


def run(export_dir: str | None = None) -> None:
    hindi = load("hi-IN")
    english = {r["id"]: r for r in load("en-US")}
    train = [r for r in hindi if r["partition"] == "train"]
    path = DATA / "annotation" / "seed_50_hinglish.jsonl"
    yours = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    hindi_by_id = {r["id"]: r for r in hindi}

    test_dev = [hindi_by_id[r["id"]]["utt"] for r in yours]
    test_hing = [r["hinglish"] for r in yours]
    truth = [r["intent"] for r in yours]

    x_dev = [r["utt"] for r in train]
    y = [r["intent"] for r in train]
    x_rom = [romanize(t) for t in x_dev]
    x_en = [english[r["id"]]["utt"] for r in train]
    recipes = {
        "1. Devanagari only": (x_dev, y),
        "2. + auto-romanized": (x_dev + x_rom, y + y),
        "3. + auto-romanized + English": (x_dev + x_rom + x_en, y + y + y),
    }

    print(f"Trained on {len(train):,} Hindi sentences. Tested on your {len(yours)} sentences, two ways.\n")
    print(f"{'model':32} {'Devanagari':>18} {'Hinglish':>18}")
    results, models = {}, {}
    for name, (x, labels) in recipes.items():
        model = make_model().fit(x, labels)
        models[name] = model
        row = {}
        for script, texts in (("devanagari", test_dev), ("hinglish", test_hing)):
            correct = [p == t for p, t in zip(model.predict(texts), truth, strict=True)]
            acc = sum(correct) / len(correct)
            lo, hi = bootstrap_ci(correct)
            row[script] = {"accuracy": acc, "ci95": [lo, hi]}
        results[name] = row
        d, h = row["devanagari"], row["hinglish"]
        line = (
            f"Devanagari {d['accuracy']:.0%} [{d['ci95'][0]:.0%}-{d['ci95'][1]:.0%}]"
            f" | Hinglish {h['accuracy']:.0%} [{h['ci95'][0]:.0%}-{h['ci95'][1]:.0%}]"
        )
        print(f"{name:32} {line}")
        notice(f"Quick result {name}", line)

    print("\n[x-y] = 95% range. With only 50 sentences the ranges are wide - this is a first look.\n")
    print("Examples (your Hinglish -> what models 1 and 3 guessed):")
    first, last = models["1. Devanagari only"], models["3. + auto-romanized + English"]
    for text, t in list(zip(test_hing, truth, strict=True))[:6]:
        a, b = first.predict([text])[0], last.predict([text])[0]
        print(f"  {text}\n      truth {t} | model 1: {a} {'✓' if a == t else '✗'} | model 3: {b} {'✓' if b == t else '✗'}")

    out = Path("artifacts") / "quick.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if export_dir:
        folder = Path(export_dir)
        folder.mkdir(parents=True, exist_ok=True)
        export(first, folder / "model_plain.json")
        export(last, folder / "model_fixed.json")
        (folder / "results.json").write_text(json.dumps(results), encoding="utf-8")
        print(f"\nexported both models for the demo page into {folder}/")
