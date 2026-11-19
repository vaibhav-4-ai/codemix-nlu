"""Does the int8 model give the same answers the way the live server runs it?

The live server (deferral-curve) uses the light `tokenizers` library instead of `transformers`, and
two memory-saving ONNX Runtime options. This runs the Hinglish test three ways and reports each:
    A  transformers tokenizer, default options        (how `cmx export` checks the model)
    B  tokenizers library, default options
    C  tokenizers library + the two memory options    (exactly what the live server does)
"""

import json
import os
from pathlib import Path

import numpy
import onnxruntime
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from cmx.examples import test_sets
from cmx.train import MAX_PIECES

FOLDER = Path("artifacts") / "export"


def session(memory_options):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    if memory_options:
        options.add_session_config_entry("session.disable_prepacking", "1")
        options.enable_cpu_mem_arena = False
    return onnxruntime.InferenceSession(str(FOLDER / "model.int8.onnx"), options)


def piece_ids(way, words, hf_tokenizer, light_tokenizer):
    if way == "A":
        return hf_tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_PIECES)["input_ids"]
    return light_tokenizer.encode(words, is_pretokenized=True).ids


def main():
    labels = json.loads((FOLDER / "labels.json").read_text(encoding="utf-8"))
    hf_tokenizer = AutoTokenizer.from_pretrained(FOLDER / "tokenizer")
    light_tokenizer = Tokenizer.from_file(str(FOLDER / "tokenizer" / "tokenizer.json"))
    light_tokenizer.enable_truncation(MAX_PIECES)
    examples = test_sets()["hinglish_paired_test"]
    sessions = {"A": session(False), "B": session(False), "C": session(True)}
    lines = []
    for way, run in sessions.items():
        right = 0
        same_ids = 0
        for example in examples:
            ids = piece_ids(way, example["words"], hf_tokenizer, light_tokenizer)
            if ids == piece_ids("A", example["words"], hf_tokenizer, light_tokenizer):
                same_ids += 1
            batch = numpy.array([ids], dtype=numpy.int64)
            intent_logits, _ = run.run(None, {"input_ids": batch, "attention_mask": numpy.ones_like(batch)})
            if labels["intent_names"][int(intent_logits[0].argmax())] == example["intent"]:
                right += 1
        line = (f"way {way}: Hinglish intent {100 * right / len(examples):.1f}%, "
                f"piece ids same as transformers for {same_ids}/{len(examples)}")
        print(line)
        lines.append(line)
    if os.environ.get("GITHUB_ACTIONS"):
        print("::notice title=serving check::" + " | ".join(lines))


main()
