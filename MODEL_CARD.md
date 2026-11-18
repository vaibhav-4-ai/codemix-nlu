---
license: apache-2.0
language:
  - hi
  - en
base_model: l3cube-pune/hing-mbert
datasets:
  - AmazonScience/massive
tags:
  - intent-classification
  - slot-filling
  - code-mixing
  - hinglish
---

# codemix-nlu: training runs

Joint intent + slot models (hing-mbert encoder with an intent head and a slot head) trained on MASSIVE Hindi, used
to measure how much accuracy romanized, code-mixed Hinglish costs. Code, method and full results:
https://github.com/vaibhav-4-ai/codemix-nlu

## What's here

`runs/{plain,fixed}_seed{13,42,1337}/`: six runs. Each has:

- `model.pt`: a PyTorch `state_dict` for `JointModel` in `cmx/train.py`. It isn't a `transformers` checkpoint.
  To load it, build `JointModel("l3cube-pune/hing-mbert", 60, n_tags)` and call `load_state_dict`. The intent and
  tag names are in `metrics.json`.
- `tokenizer/`, `metrics.json`, `curves.csv` (per-epoch dev scores)
- `predictions_{test,devanagari_50,hinglish_50}.jsonl`: every prediction, with intent probabilities

**plain** = trained on MASSIVE hi-IN train (Devanagari). **fixed** = plain + auto-romanized copies (ITRANS) +
MASSIVE en-US train.

## Results (3 seeds pooled)

| Model | 50 paired sentences, Devanagari | Same 50, Hinglish | MASSIVE hi-IN test |
|---|---|---|---|
| plain | 72.0% intent | **38.0%** | 79.2% intent / 50.6 slot F1 |
| fixed | 77.3% | **82.7%** | 83.8% / 66.6 |

## Limitations

The Hinglish test set has 50 sentences, drafted by an AI assistant and reviewed by the author (0 edited). The
settings were not tuned. These checkpoints are for research and reproduction, not production.

## Data licence

MASSIVE: CC BY 4.0, © Amazon.com, Inc. (FitzGerald et al., 2022).
