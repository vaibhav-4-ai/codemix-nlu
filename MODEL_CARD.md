---
license: apache-2.0
language:
  - hi
  - en
base_model: l3cube-pune/hing-mbert
datasets:
  - AmazonScience/massive
  - vaibhav-4-ai/hinglish-massive-paired
tags:
  - intent-classification
  - slot-filling
  - code-mixing
  - hinglish
  - onnx
  - int8
---

# codemix-nlu: models, runs and reports

Joint intent + slot models (an encoder with an intent head on `[CLS]` and a BIO slot head on each word), trained on
MASSIVE Hindi to measure how much accuracy romanized, code-mixed Hinglish costs, and what fixes it. Code, method
and the full write-up: https://github.com/vaibhav-4-ai/codemix-nlu. Live demo:
https://deferral-curve.onrender.com/codemix

## What's here

| folder | contents |
|---|---|
| `serving/` | **the served model**: fixed recipe, hing-mbert, seed 13, int8 ONNX (178 MB), `labels.json`, `tokenizer/` |
| `runs_v2/<recipe>_<backbone>_seed<seed>/` | 18 training runs (5 recipes on hing-mbert + the fixed recipe on MuRIL, 3 seeds each): `metrics.json`, `curves.csv`, every prediction with intent probabilities |
| `runs_v2/fixed_hing-mbert_seed*/model.pt` | full PyTorch weights for `JointModel` in `cmx/train.py` (not a `transformers` checkpoint) |
| `tune/` | Optuna: the 12 trials and the chosen settings |
| `charts/` | every report and chart CI produces (data analysis, results, calibration, transfer) |
| `runs/` | the first, smaller version (2 recipes, 50 test sentences), kept for the history |

To use `serving/` without PyTorch: tokenize the words with `tokenizers` (`is_pretokenized=True`), run
`model.int8.onnx` with onnxruntime (inputs `input_ids`, `attention_mask`; outputs `intent_logits`, `slot_logits`),
and read the names from `labels.json`. See `dc/codemix.py` in [deferral-curve](https://github.com/vaibhav-4-ai/deferral-curve) for a complete example.

## Results (3 seeds pooled, 557 paired test requests, 95% ranges)

| model | Devanagari | Hinglish (same requests) | Hindi test (2,974) |
|---|---|---|---|
| plain (Devanagari only) | 82.0% intent | **48.3%** (44.8–51.7) | 82.9% |
| fixed (+ romanized + English) | 85.4% | **85.8%** (83.2–88.4) | 84.7% |
| served int8 (fixed, seed 13) | 82.5% | 83.2% | |

On Google's human-written Hinglish-TOP (4,262 requests, 17 mapped intents) the served int8 model gets 88.2% in
English and 83.8% in Hinglish.

## Limitations

- The paired Hinglish test set was drafted by an AI model and checked automatically and by a second AI pass (4 of
  100 corrected), not by a native speaker. The Hinglish-TOP result is the check on real writers.
- int8 costs about 3 intent points against the full model.
- For research and demos, not production: 60 MASSIVE intents only, English/Hindi only.

## Licences

Code and models: Apache 2.0. MASSIVE: CC BY 4.0, © Amazon.com, Inc. (FitzGerald et al., 2022). Hinglish-TOP:
Apache-2.0, © Google (Agarwal et al., 2022).
