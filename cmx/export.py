"""Stage 8: make the main model small and fast enough for a cheap CPU server.

1. download the trained main model (fixed recipe, hing-mbert, seed 13) from Hugging Face
2. ONNX: save the model as a plain graph of maths steps that runs without PyTorch
3. int8: store each weight as an 8-bit integer instead of a 32-bit decimal -> about 4x smaller
4. check what that cost: accuracy, how often int8 and full-size agree, size, speed (p50 / p95)

Writes artifacts/export/: model.int8.onnx, labels.json, tokenizer/, report.md, results.json.
"""

import json
import os
import time
from pathlib import Path

import numpy
import onnxruntime
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from cmx.evaluate import slot_spans
from cmx.examples import test_sets
from cmx.quick import notice
from cmx.train import BACKBONE, MAX_PIECES, JointModel

REPO = "vaibhav-4-ai/codemix-nlu"
RUN = "runs_v2/fixed_hing-mbert_seed13"
OUT = Path("artifacts") / "export"
CHECK_SETS = ["devanagari_paired_test", "hinglish_paired_test"]


class OnnxModel:
    """The exported model: words in, intent + probabilities + one tag per word out."""

    def __init__(self, folder):
        folder = Path(folder)
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        self.intent_names = labels["intent_names"]
        self.tag_names = labels["tag_names"]
        self.tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2  # a small server has 1-2 CPUs
        self.session = onnxruntime.InferenceSession(str(folder / "model.int8.onnx"), options)

    def predict(self, words):
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_PIECES)
        ids = numpy.array([enc["input_ids"]], dtype=numpy.int64)
        mask = numpy.ones_like(ids)
        intent_logits, slot_logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})
        scores = intent_logits[0] - intent_logits[0].max()
        probs = numpy.exp(scores) / numpy.exp(scores).sum()
        best = int(probs.argmax())
        tags = ["O"] * len(words)
        previous = None
        for position, word_id in enumerate(enc.word_ids()):
            if word_id is not None and word_id != previous:
                tags[word_id] = self.tag_names[int(slot_logits[0][position].argmax())]
            previous = word_id
        return {"intent": self.intent_names[best], "confidence": float(probs[best]), "tags": tags}


def build_full_model(folder):
    """Rebuild the trained PyTorch model from model.pt (same class as training)."""
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    model = JointModel(BACKBONE, len(metrics["intent_names"]), len(metrics["tag_names"]))
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    model.eval()  # no dropout when predicting
    return model, metrics


def score(predictions, examples):
    right = 0
    hits = 0
    total = 0
    for k, example in enumerate(examples):
        if predictions[k]["intent"] == example["intent"]:
            right += 1
        true_spans = slot_spans(example["tags"])
        guessed_spans = slot_spans(predictions[k]["tags"])
        hits += len(true_spans & guessed_spans)
        total += len(true_spans) + len(guessed_spans)
    return right / len(examples), (2 * hits / total if total else 0.0)


def full_model_predict(model, tokenizer, metrics, words):
    """The same steps as OnnxModel.predict, with the full-size PyTorch model."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_PIECES)
    ids = torch.tensor([enc["input_ids"]])
    with torch.no_grad():
        intent_logits, slot_logits = model(ids, torch.ones_like(ids))
    best = int(intent_logits[0].argmax())
    tags = ["O"] * len(words)
    previous = None
    for position, word_id in enumerate(enc.word_ids()):
        if word_id is not None and word_id != previous:
            tags[word_id] = metrics["tag_names"][int(slot_logits[0][position].argmax())]
        previous = word_id
    return {"intent": metrics["intent_names"][best], "tags": tags}


def export_onnx(model, example, path, use_new_exporter):
    """Save the model as ONNX. PyTorch has two exporters; the newer one copes with newer models.

    dynamic_axes: batch size and sentence length can change at run time.
    """
    torch.onnx.export(
        model,
        (example["input_ids"], example["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "pieces"},
            "attention_mask": {0: "batch", 1: "pieces"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch", 1: "pieces"},
        },
        opset_version=18,
        dynamo=use_new_exporter,
    )


def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    folder = Path(snapshot_download(REPO, allow_patterns=[f"{RUN}/metrics.json", f"{RUN}/tokenizer/*"],
                                    local_dir="artifacts/hf"))
    folder = folder / RUN
    hf_hub_download(REPO, f"{RUN}/model.pt", local_dir="artifacts/hf")
    model, metrics = build_full_model(folder)
    tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")

    # 1. ONNX export. dynamic_axes: batch size and sentence length can change at run time.
    example = tokenizer(["alarm", "set", "karo"], is_split_into_words=True, return_tensors="pt")
    full_path = OUT / "model.onnx"
    try:
        export_onnx(model, example, full_path, use_new_exporter=False)
        exporter = "classic (TorchScript)"
    except Exception as error:  # noqa: BLE001 - any failure: try the other exporter
        print(f"classic exporter failed ({error}); trying the new one")
        export_onnx(model, example, full_path, use_new_exporter=True)
        exporter = "new (torch.export)"
    print(f"exported with the {exporter} exporter")
    # 2. int8: weights of the matrix multiplications become 8-bit integers.
    # per_channel: each row of a weight matrix gets its own scale, instead of one scale for the
    # whole matrix - small extra size, usually much less accuracy lost.
    quantize_dynamic(str(full_path), str(OUT / "model.int8.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True)
    tokenizer.save_pretrained(OUT / "tokenizer")
    labels = {"intent_names": metrics["intent_names"], "tag_names": metrics["tag_names"]}
    (OUT / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    # 3. what did shrinking cost?
    small = OnnxModel(OUT)
    sets = test_sets()
    results = {
        "size_mb": {
            "pytorch_fp32": round(os.path.getsize(folder / "model.pt") / 1e6, 1),
            "onnx_int8": round(os.path.getsize(OUT / "model.int8.onnx") / 1e6, 1),
        },
        "sets": {},
    }
    timings = []
    for name in CHECK_SETS:
        full_guesses = []
        small_guesses = []
        agree = 0
        for item in sets[name]:
            full_guess = full_model_predict(model, tokenizer, metrics, item["words"])
            started = time.perf_counter()
            small_guess = small.predict(item["words"])
            timings.append(1000 * (time.perf_counter() - started))
            full_guesses.append(full_guess)
            small_guesses.append(small_guess)
            if full_guess["intent"] == small_guess["intent"]:
                agree += 1
        full_acc, full_f1 = score(full_guesses, sets[name])
        small_acc, small_f1 = score(small_guesses, sets[name])
        results["sets"][name] = {
            "full_intent": full_acc, "int8_intent": small_acc, "full_slot_f1": full_f1,
            "int8_slot_f1": small_f1, "same_intent_share": agree / len(full_guesses),
        }  # fmt: skip
    results["ms_per_sentence"] = {"p50": round(percentile(timings, 0.5), 1),
                                  "p95": round(percentile(timings, 0.95), 1)}  # fmt: skip
    os.remove(full_path)  # the full-size ONNX copy is not needed any more

    lines = ["# Export: int8 ONNX for CPU serving (written by `python -m cmx export`)\n"]
    lines.append(f"Model: fixed recipe, hing-mbert, seed 13. Size: PyTorch {results['size_mb']['pytorch_fp32']} MB"
                 f" -> int8 ONNX {results['size_mb']['onnx_int8']} MB.\n")
    lines.append("| set | intent full | intent int8 | slot F1 full | slot F1 int8 | same intent |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in results["sets"].items():
        lines.append(f"| {name} | {100 * r['full_intent']:.1f} | {100 * r['int8_intent']:.1f} | "
                     f"{100 * r['full_slot_f1']:.1f} | {100 * r['int8_slot_f1']:.1f} | "
                     f"{100 * r['same_intent_share']:.1f}% |")
    lines.append(f"\nSpeed on the GitHub runner CPU, one sentence at a time, 2 threads: "
                 f"p50 {results['ms_per_sentence']['p50']} ms, p95 {results['ms_per_sentence']['p95']} ms.\n")
    report = "\n".join(lines)
    print(report)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    hinglish = results["sets"]["hinglish_paired_test"]
    notice("int8 export", f"{results['size_mb']['pytorch_fp32']} MB -> {results['size_mb']['onnx_int8']} MB; "
                          f"Hinglish intent {100 * hinglish['full_intent']:.1f} -> {100 * hinglish['int8_intent']:.1f};"
                          f" p95 {results['ms_per_sentence']['p95']} ms")
