"""Simple models first: if these already did well, the big model would not be needed.

1. TF-IDF + logistic regression (intent)      letter pieces -> one weight per piece per intent
2. LightGBM on TF-IDF + code-mixing features (intent)   trees instead of weights
3. CRF (slots)                                 tags each word, looking at its neighbours
4. Nested cross-validation leakage check       how much a "pick the best setting, then report
                                               its score" number flatters itself

Every model is trained twice, on the same two recipes as the big model:
    plain  Hindi training sentences (Devanagari)
    fixed  plain + the same sentences romanized + the English versions
and scored on the full Hindi test set and the 560 paired sentences (Devanagari and Hinglish).
Writes artifacts/baselines/report.md and results.json.
"""

import json
import os
import random
import time
from pathlib import Path

import lightgbm
import sklearn_crfsuite
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from cmx.evaluate import slot_spans
from cmx.examples import RECIPES, test_sets, training_examples
from cmx.features import FEATURE_NAMES, build_word_lists, sentence_features
from cmx.quick import make_model

OUT = Path("artifacts") / "baselines"
SCORED = ["test", "devanagari_paired_test", "hinglish_paired_test"]
BASELINE_RECIPES = ["plain", "fixed"]
SEED = 13


def notice(title, message):
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title={title}::{message}")


def texts_and_intents(examples):
    texts = []
    intents = []
    for example in examples:
        texts.append(" ".join(example["words"]))
        intents.append(example["intent"])
    return texts, intents


def intent_scores(truth, guesses):
    return {
        "accuracy": round(accuracy_score(truth, guesses), 4),
        "macro_f1": round(f1_score(truth, guesses, average="macro", zero_division=0), 4),
    }


# ---------- 1. TF-IDF + logistic regression ----------


def tfidf_logreg(train, sets):
    texts, intents = texts_and_intents(train)
    model = make_model().fit(texts, intents)
    result = {}
    for name in SCORED:
        test_texts, truth = texts_and_intents(sets[name])
        result[name] = intent_scores(truth, list(model.predict(test_texts)))
    return result


# ---------- 2. LightGBM ----------


def feature_matrix(vectorizer, texts, examples, lists):
    """TF-IDF letter pieces + the 6 code-mixing numbers, side by side."""
    pieces = vectorizer.transform(texts)
    extra = []
    for example in examples:
        values = sentence_features(example["words"], lists)
        row = []
        for name in FEATURE_NAMES:
            row.append(values[name])
        extra.append(row)
    return hstack([pieces, csr_matrix(extra)]).tocsr()


def lightgbm_intent(train, sets, lists):
    texts, intents = texts_and_intents(train)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=3000
    )
    vectorizer.fit(texts)
    model = lightgbm.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=5,
        colsample_bytree=0.5,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(feature_matrix(vectorizer, texts, train, lists), intents)
    result = {}
    for name in SCORED:
        test_texts, truth = texts_and_intents(sets[name])
        guesses = model.predict(feature_matrix(vectorizer, test_texts, sets[name], lists))
        result[name] = intent_scores(truth, list(guesses))
    return result


# ---------- 3. CRF for slots ----------


def word_features(words, i):
    """What the CRF knows about word i: the word, its ending, its neighbours."""
    word = words[i].lower()
    features = {
        "word": word,
        "last3": word[-3:],
        "last2": word[-2:],
        "first2": word[:2],
        "digit": word.isdigit(),
        "latin": word.isascii(),
        "length": str(min(len(word), 8)),  # a label ("5"), not a number to multiply by
    }
    if i > 0:
        features["previous"] = words[i - 1].lower()
    else:
        features["start"] = True
    if i < len(words) - 1:
        features["next"] = words[i + 1].lower()
    else:
        features["end"] = True
    return features


def sentence_to_features(words):
    rows = []
    for i in range(len(words)):
        rows.append(word_features(words, i))
    return rows


def slot_f1(truth_tags, guessed_tags):
    """Span F1: a slot counts only if its first word, last word and type are all right."""
    hits = 0
    guessed = 0
    real = 0
    for truth, guess in zip(truth_tags, guessed_tags):
        truth_spans = slot_spans(truth)
        guess_spans = slot_spans(guess)
        hits += len(truth_spans & guess_spans)
        guessed += len(guess_spans)
        real += len(truth_spans)
    if hits == 0:
        return 0.0
    precision = hits / guessed
    recall = hits / real
    return round(2 * precision * recall / (precision + recall), 4)


def crf_slots(train, sets):
    x = []
    y = []
    for example in train:
        x.append(sentence_to_features(example["words"]))
        y.append(example["tags"])
    model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100)
    model.fit(x, y)
    result = {}
    for name in SCORED:
        test_x = []
        truth = []
        for example in sets[name]:
            test_x.append(sentence_to_features(example["words"]))
            truth.append(example["tags"])
        result[name] = {"slot_f1": slot_f1(truth, model.predict(test_x))}
    return result


# ---------- 4. Nested cross-validation ----------


def small_sample(train):
    """Up to 50 sentences per intent (intents with fewer than 10 are left out), fixed seed."""
    rng = random.Random(SEED)
    by_intent = {}
    for example in train:
        if example["intent"] not in by_intent:
            by_intent[example["intent"]] = []
        by_intent[example["intent"]].append(example)
    picked = []
    for intent in sorted(by_intent):
        group = by_intent[intent]
        if len(group) >= 10:
            rng.shuffle(group)
            picked += group[:50]
    return picked


def nested_cv(train):
    """Two ways to report "how good is my best setting", on the same sentences.

    Leaky:  try 4 settings with 5-fold CV, report the best one's CV score.
            The same folds both PICKED the winner and SCORED it, so the score flatters it.
    Nested: an outer 5-fold loop holds out a fold; inside, a 3-fold search picks the setting
            using only the other 4 folds; the held-out fold scores it. Honest.
    """
    texts, intents = texts_and_intents(small_sample(train))
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )
    grid = {"logisticregression__C": [0.3, 1.0, 3.0, 10.0]}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    leaky = GridSearchCV(pipeline, grid, cv=outer, scoring="accuracy").fit(texts, intents)
    search = GridSearchCV(pipeline, grid, cv=inner, scoring="accuracy")
    nested_scores = cross_val_score(search, texts, intents, cv=outer, scoring="accuracy")

    # The mistake people make most: learn the TF-IDF vocabulary on ALL sentences before
    # splitting, so the held-out folds leak into the features.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    all_features = vectorizer.fit_transform(texts)
    early = GridSearchCV(
        LogisticRegression(max_iter=2000), {"C": [0.3, 1.0, 3.0, 10.0]}, cv=outer, scoring="accuracy"
    ).fit(all_features, intents)

    nested_mean = sum(nested_scores) / len(nested_scores)
    squares = 0.0
    for score in nested_scores:
        squares += (score - nested_mean) ** 2
    spread = (squares / len(nested_scores)) ** 0.5
    return {
        "sentences": len(texts),
        "intents": len(set(intents)),
        "leaky_best_cv": round(float(leaky.best_score_), 4),
        "leaky_best_C": leaky.best_params_["logisticregression__C"],
        "tfidf_fit_before_split": round(float(early.best_score_), 4),
        "nested": round(float(nested_mean), 4),
        "nested_fold_sd": round(float(spread), 4),
    }


# ---------- report ----------


def pct(x):
    return f"{100 * x:.1f}"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = test_sets()
    lists = build_word_lists()
    results = {}
    timings = {}
    for recipe in BASELINE_RECIPES:
        if recipe not in RECIPES:
            raise SystemExit(f"unknown recipe {recipe}")
        train = training_examples(recipe)
        print(f"recipe {recipe}: {len(train):,} training sentences", flush=True)
        for model_name in ["tfidf_logreg", "lightgbm", "crf"]:
            if model_name == "lightgbm" and recipe == "plain":
                continue  # trees are slow; the fixed recipe is the one that matters for Hinglish
            start = time.time()
            if model_name == "tfidf_logreg":
                scores = tfidf_logreg(train, sets)
            elif model_name == "lightgbm":
                scores = lightgbm_intent(train, sets, lists)
            else:
                scores = crf_slots(train, sets)
            results[f"{model_name} {recipe}"] = scores
            timings[f"{model_name} {recipe}"] = round(time.time() - start, 1)
            print(f"  {model_name}: {scores}  ({timings[f'{model_name} {recipe}']}s)", flush=True)

    print("nested cross-validation ...", flush=True)
    nested = nested_cv(training_examples("plain"))
    results["nested_cv"] = nested

    lines = ["# Classical baselines (written by `python -m cmx baselines`)\n"]
    lines.append("## Intent (accuracy / macro-F1, %)\n")
    lines.append("| model | recipe | Hindi test (2,974) | Devanagari paired (560) | Hinglish paired (560) |")
    lines.append("|---|---|---|---|---|")
    for key, scores in results.items():
        if key.startswith("tfidf") or key.startswith("lightgbm"):
            model_name, recipe = key.split(" ")
            cells = []
            for name in SCORED:
                cells.append(f"{pct(scores[name]['accuracy'])} / {pct(scores[name]['macro_f1'])}")
            lines.append(f"| {model_name} | {recipe} | " + " | ".join(cells) + " |")
    lines.append("\n## Slots (span F1, %)\n")
    lines.append("| model | recipe | Hindi test | Devanagari paired | Hinglish paired |")
    lines.append("|---|---|---|---|---|")
    for key, scores in results.items():
        if key.startswith("crf"):
            model_name, recipe = key.split(" ")
            cells = []
            for name in SCORED:
                cells.append(pct(scores[name]["slot_f1"]))
            lines.append(f"| {model_name} | {recipe} | " + " | ".join(cells) + " |")

    gap = nested["leaky_best_cv"] - nested["nested"]
    lines.append("\n## Nested cross-validation: how much does picking-then-reporting flatter a score?\n")
    lines.append(f"TF-IDF + logistic regression, {nested['sentences']:,} training sentences "
                 f"({nested['intents']} intents, up to 50 each), 4 values of C.\n")
    lines.append("| how the score was measured | accuracy % |")
    lines.append("|---|---|")
    lines.append(f"| Leaky: best of 4 settings, scored on the folds that picked it | "
                 f"{pct(nested['leaky_best_cv'])} |")
    lines.append(f"| Leakier: also learned the TF-IDF vocabulary on all folds first | "
                 f"{pct(nested['tfidf_fit_before_split'])} |")
    lines.append(f"| Honest (nested): setting picked inside, scored on an untouched fold | "
                 f"{pct(nested['nested'])} (fold-to-fold spread {pct(nested['nested_fold_sd'])}) |")
    lines.append(f"\nGap leaky minus honest: {100 * gap:+.1f} points. The big model avoids this the same "
                 "way: Optuna only ever sees the dev sets, and the test sets are scored once at the end.")
    lines.append(f"\nSeconds per model: {timings}")

    report = "\n".join(lines)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(report)

    for key in ["tfidf_logreg plain", "tfidf_logreg fixed", "lightgbm fixed"]:
        scores = results[key]
        notice(f"Baseline {key}", f"intent acc: Devanagari paired {pct(scores['devanagari_paired_test']['accuracy'])}"
                                  f" | Hinglish paired {pct(scores['hinglish_paired_test']['accuracy'])}")
    for key in ["crf plain", "crf fixed"]:
        scores = results[key]
        notice(f"Baseline {key}", f"slot F1: Devanagari paired {pct(scores['devanagari_paired_test']['slot_f1'])}"
                                  f" | Hinglish paired {pct(scores['hinglish_paired_test']['slot_f1'])}")
    notice("Nested CV", f"leaky {pct(nested['leaky_best_cv'])} vs nested {pct(nested['nested'])} "
                        f"(gap {100 * gap:+.1f} points)")
