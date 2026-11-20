"""Benchmark any model on the paired Hinglish set - the part of this project others can reuse.

Three steps, no torch needed:

    python -m cmx benchmark-inputs inputs.jsonl          1. the sentences to predict
    (your model)  or  python -m cmx predict-file inputs.jsonl predictions.jsonl
    python -m cmx benchmark predictions.jsonl            3. the scores

inputs.jsonl, one line per sentence and form (560 sentences x 2 forms):
    {"id": "123", "form": "hinglish", "words": ["kal", "subah", "7", "baje", ...]}
predictions.jsonl, the same lines with your answers ("tags" optional, one BIO tag per word):
    {"id": "123", "form": "hinglish", "intent": "alarm_set", "tags": ["B-date", "B-time", ...]}

The scores: intent accuracy, macro-F1 and slot F1 for the Devanagari and the Hinglish form of the
SAME requests, the drop between them with a paired 95% range, and Hinglish accuracy per CMI band.
"""

import json
import random
import urllib.request
from pathlib import Path

from cmx.massive import parse_annot, slot_spans

DATASET = "https://huggingface.co/datasets/vaibhav-4-ai/hinglish-massive-paired/resolve/main/"
SERVING = "https://huggingface.co/vaibhav-4-ai/codemix-nlu/resolve/main/serving/"
CACHE = Path.home() / ".cache" / "codemix-nlu"
FORMS = ["devanagari", "hinglish"]
BAND_ORDER = ["none", "light", "medium", "heavy"]
MAX_PIECES = 64
DRAWS = 2000
SEED = 13


def download(url, path):
    """Download once into the cache folder; later calls reuse the file."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


def paired_test():
    """The 560 paired test requests: id, intent, CMI band, and words + tags in both forms."""
    path = download(DATASET + "test.jsonl", CACHE / "test.jsonl")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hinglish_words, hinglish_tags = parse_annot(row["hinglish_annot"])
        devanagari_words, devanagari_tags = parse_annot(row["devanagari_annot"])
        rows.append({
            "id": str(row["id"]), "intent": row["intent"], "band": row["cmi_band"],
            "hinglish": {"words": hinglish_words, "tags": hinglish_tags},
            "devanagari": {"words": devanagari_words, "tags": devanagari_tags},
        })  # fmt: skip
    return rows


def read_lines(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_lines(path, rows):
    lines = []
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def write_inputs(path):
    rows = []
    for item in paired_test():
        for form in FORMS:
            rows.append({"id": item["id"], "form": form, "words": item[form]["words"]})
    write_lines(path, rows)
    print(f"wrote {len(rows)} sentences to predict into {path}")


# ---------- scoring ----------


def intent_scores(items, answers, form):
    """-> (accuracy, macro-F1) for one form. `answers` maps (id, form) -> prediction."""
    right = 0
    counts = {}
    for item in items:
        guess = answers[(item["id"], form)]["intent"]
        truth = item["intent"]
        if guess == truth:
            right += 1
        for name in [truth, guess]:
            if name not in counts:
                counts[name] = {"hit": 0, "true": 0, "guessed": 0}
        counts[truth]["true"] += 1
        counts[guess]["guessed"] += 1
        if guess == truth:
            counts[guess]["hit"] += 1
    f1s = []
    for value in counts.values():
        if value["true"] > 0:
            f1s.append(2 * value["hit"] / (value["true"] + value["guessed"]))
    return right / len(items), sum(f1s) / len(f1s)


def slot_f1(items, answers, form):
    """Span F1: a slot counts only if its type and every one of its words are right. None if no tags."""
    hits = 0
    total = 0
    for item in items:
        tags = answers[(item["id"], form)].get("tags")
        if tags is None:
            return None
        true_spans = slot_spans(item[form]["tags"])
        guessed_spans = slot_spans(tags)
        hits += len(true_spans & guessed_spans)
        total += len(true_spans) + len(guessed_spans)
    return 2 * hits / total if total else 0.0


def middle_95(values):
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]


def drop_range(items, answers):
    """95% range of (Devanagari accuracy - Hinglish accuracy), resampling the same requests for both."""
    flags = []
    for item in items:
        dev_right = 1 if answers[(item["id"], "devanagari")]["intent"] == item["intent"] else 0
        hin_right = 1 if answers[(item["id"], "hinglish")]["intent"] == item["intent"] else 0
        flags.append(dev_right - hin_right)
    rng = random.Random(SEED)
    drops = []
    for _ in range(DRAWS):
        total = 0
        for _ in flags:
            total += flags[rng.randrange(len(flags))]
        drops.append(total / len(flags))
    return middle_95(drops)


def score(items, predictions):
    """All the numbers. Raises an error naming the first missing prediction."""
    answers = {}
    for row in predictions:
        answers[(str(row["id"]), row["form"])] = row
    for item in items:
        for form in FORMS:
            if (item["id"], form) not in answers:
                raise SystemExit(f"no prediction for id {item['id']}, form {form}")
    result = {"requests": len(items)}
    for form in FORMS:
        accuracy, macro = intent_scores(items, answers, form)
        result[form] = {"intent_accuracy": accuracy, "intent_macro_f1": macro, "slot_f1": slot_f1(items, answers, form)}
    result["drop"] = result["devanagari"]["intent_accuracy"] - result["hinglish"]["intent_accuracy"]
    result["drop_95"] = drop_range(items, answers)
    result["hinglish_by_band"] = {}
    for name in BAND_ORDER:
        band_items = []
        for item in items:
            if item["band"] == name:
                band_items.append(item)
        if band_items:
            accuracy = intent_scores(band_items, answers, "hinglish")[0]
            result["hinglish_by_band"][name] = {"requests": len(band_items), "intent_accuracy": accuracy}
    return result


def pct(x):
    if x is None:
        return "-"
    return f"{100 * x:.1f}"


def report(result):
    lines = [f"Paired Hinglish benchmark: {result['requests']} requests, each in Devanagari and in Hinglish.\n"]
    lines.append("| form | intent accuracy | intent macro-F1 | slot F1 |")
    lines.append("|---|---|---|---|")
    for form in FORMS:
        r = result[form]
        lines.append(f"| {form} | {pct(r['intent_accuracy'])} | {pct(r['intent_macro_f1'])} | {pct(r['slot_f1'])} |")
    low, high = result["drop_95"]
    lines.append(f"\nDrop, Devanagari - Hinglish: {pct(result['drop'])} points (95% range {pct(low)} to {pct(high)}).\n")
    lines.append("| Hinglish CMI band | requests | intent accuracy |")
    lines.append("|---|---|---|")
    for name, r in result["hinglish_by_band"].items():
        lines.append(f"| {name} | {r['requests']} | {pct(r['intent_accuracy'])} |")
    lines.append("\nThe same table for this project's models is in the README "
                 "(https://github.com/vaibhav-4-ai/codemix-nlu).")
    return "\n".join(lines)


def benchmark(predictions_path):
    # notice() prints a note on the GitHub run page; it does nothing elsewhere.
    from cmx.quick import notice

    result = score(paired_test(), read_lines(predictions_path))
    print(report(result))
    notice("Benchmark tool", f"Devanagari {pct(result['devanagari']['intent_accuracy'])}; Hinglish "
                             f"{pct(result['hinglish']['intent_accuracy'])}; Hinglish slot F1 "
                             f"{pct(result['hinglish']['slot_f1'])}; drop {pct(result['drop'])}")
    return result


# ---------- this project's model (int8 ONNX, needs: pip install "codemix-nlu[model]") ----------


class Model:
    """The served model: fixed recipe, hing-mbert, int8 ONNX. Downloaded once (~180 MB) into the cache."""

    def __init__(self):
        import onnxruntime  # imported here: only this class needs the [model] extra
        from tokenizers import Tokenizer

        folder = CACHE / "serving"
        download(SERVING + "model.int8.onnx", folder / "model.int8.onnx")
        download(SERVING + "labels.json", folder / "labels.json")
        download(SERVING + "tokenizer/tokenizer.json", folder / "tokenizer.json")
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        self.intent_names = labels["intent_names"]
        self.tag_names = labels["tag_names"]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_PIECES)
        self.session = onnxruntime.InferenceSession(str(folder / "model.int8.onnx"))

    def predict(self, words):
        """['kal', 'subah', '7', 'baje', 'alarm', 'laga', 'do'] -> intent, confidence, one tag per word."""
        import numpy

        enc = self.tokenizer.encode(words, is_pretokenized=True)
        ids = numpy.array([enc.ids], dtype=numpy.int64)
        mask = numpy.ones_like(ids)
        intent_logits, slot_logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})
        scores = intent_logits[0] - intent_logits[0].max()
        probs = numpy.exp(scores) / numpy.exp(scores).sum()
        best = int(probs.argmax())
        tags = ["O"] * len(words)
        previous = None
        for position, word_id in enumerate(enc.word_ids):
            if word_id is not None and word_id != previous:
                tags[word_id] = self.tag_names[int(slot_logits[0][position].argmax())]
            previous = word_id
        return {"intent": self.intent_names[best], "confidence": round(float(probs[best]), 3), "tags": tags}


def predict_text(text):
    words = text.split()
    result = Model().predict(words)
    result["words"] = words
    print(json.dumps(result, ensure_ascii=False, indent=2))


def predict_file(inputs_path, outputs_path):
    model = Model()
    rows = []
    for row in read_lines(inputs_path):
        guess = model.predict(row["words"])
        rows.append({"id": row["id"], "form": row["form"], "intent": guess["intent"], "tags": guess["tags"]})
    write_lines(outputs_path, rows)
    print(f"wrote {len(rows)} predictions into {outputs_path}")
