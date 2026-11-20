# Changelog

Versions follow [semantic versioning](https://semver.org): a new MAJOR number would mean a change that breaks
existing use.

## 1.1.0 (2026-11-20)

- **Benchmark tool:** `codemix-nlu benchmark-inputs`, `benchmark` and `predict-file` score any model on the 560
  paired requests (intent accuracy, macro-F1, slot F1, the Devanagari − Hinglish drop with a paired 95% range, and
  accuracy by CMI band).
- **The model from the command line:** `codemix-nlu predict "..."`, using the published int8 ONNX model
  (`pip install "codemix-nlu[model]"`).
- `codemix-nlu cmi "..."` prints the code-mixing numbers for one sentence.
- New results: slot PR-AUC for rare vs common slot types, accuracy by unfamiliar spellings, slot F1 by slot
  rarity, and 95% ranges + macro-F1 on the Hinglish-TOP transfer test.
- Training now records library versions, the GPU and its peak memory in `metrics.json`.

## 1.0.0 (2026-11-20)

- First release: `cmx.features.code_mixing()` (CMI, I-Index, script, spelling variants, OOV) with the word lists
  built from MASSIVE's training data.
