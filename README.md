# codemix-nlu

**How much worse does a Hindi voice assistant understand you when you type in Hinglish, and what fixes it?**

Most Indians type Hindi in English letters, mixed with English words: *"kal subah 7 baje ka alarm laga do"*.
Hindi NLU models are usually trained on Devanagari. This project measures what that mismatch costs on a real
intent + slot task, finds out which part of the fix matters for which kind of Hinglish, and serves the fixed
model on a 512 MB CPU server.

| | |
|---|---|
| **The drop** | A hing-mbert model trained on Devanagari Hindi gets the intent right for **82.0%** of 557 test requests, but only **48.3%** of **the same requests** written in Hinglish: **−33.7 points** (95% range 29.9 to 37.5). |
| **The fix** | Training on Devanagari + auto-romanized Hindi + English: Hinglish **85.8%**, Devanagari **85.4%**. The gap is gone (−0.4, range −2.8 to 2.0). |
| **Why both halves** | Romanized data fixes Hinglish with *no* English words; English data fixes *heavily* mixed Hinglish. Neither alone closes the gap ([the curve](#3-the-degradation-curve)). |
| **On strangers' Hinglish** | On Google's human-written Hinglish-TOP (4,262 requests, different writers and topics), the served int8 model scores **85.3%**, matching its 85.4% on our own test set. |
| **Honest confidence** | Temperature scaling cuts calibration error on Hinglish from 0.076 to 0.017; an "ask instead of act" cut-off answers 90% of Hinglish requests at 91% accuracy. |
| **Served** | 712 MB → **178 MB** int8 ONNX. Live on a 512 MB, 0.1-CPU server: **442 MB** peak, **p95 147 ms**: **[deferral-curve.onrender.com/codemix](https://deferral-curve.onrender.com/codemix)** |

Every number on this page is recomputed by CI on each push, from predictions saved on
[Hugging Face](https://huggingface.co/vaibhav-4-ai/codemix-nlu). The paired test set is published as
[`vaibhav-4-ai/hinglish-massive-paired`](https://huggingface.co/datasets/vaibhav-4-ai/hinglish-massive-paired).

---

## The task

NLU turns a request into something an app can act on: an **intent** (60 kinds) and **slots** (the key details).

| Input | Intent | Slots |
|---|---|---|
| `kal subah 7 baje ka alarm laga do` | `alarm_set` | `date = kal`, `time = subah 7 baje` |

## Data

- **[MASSIVE](https://github.com/alexa/massive)** (Amazon, CC BY 4.0): the same 16,521 requests in Hindi
  (`hi-IN`, Devanagari) and English (`en-US`), with identical labels. Train 11,514 / dev 2,033 / test 2,974.
- **Paired Hinglish set (new):** 560 test + 392 dev requests, up to 10 / 7 per intent, each rewritten in
  romanized, code-mixed Hinglish with **the same labels**. Only the writing changes, so any accuracy difference
  is caused by the script and the mixing.
  - **How it was made:** drafted by an AI model (Claude) in an informal WhatsApp style. Every row passes an
    automatic check (same slot types as the Hindi original). A random 100 got a second check, **also by an AI
    pass, not a person**: 4 of 100 were corrected. It has not been reviewed line by line by a native speaker,
    which is why the transfer test on Google's human-written Hinglish-TOP matters (it agrees).
- **Data analysis** (`python -m cmx eda`): 60 intents, the largest 202× the smallest, so macro-F1 is reported too.
  55 slot types, 19 of them seen fewer than 50 times. 115 Hindi sentences appear in both MASSIVE's train and test
  splits; 23 of them are in the paired set, and dropping them changes the drop by 1.2 points (checked in CI).
  [Report](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/charts/eda/report.md).

### Measuring "how mixed" (CMI)

Each word is tagged Hindi or English, using word lists built only from training data, and then:
**CMI = 100 × (1 − biggest language / tagged words)**: 0 = one language, 50 = half and half.
`kal subah 7 baje alarm set kar do` → 5 Hindi, 2 English → CMI 28.6. The test set splits into four bands: none
(54), light 0–20 (92), medium 20–35 (222), heavy 35–50 (192). Mean CMI: Hinglish 28.1, English 8.8,
auto-romanized Hindi 2.0 (the last two are a check on the tagger).

![CMI of the Hinglish test sentences](https://huggingface.co/vaibhav-4-ai/codemix-nlu/resolve/main/charts/eda/cmi.png)

## Model and training

- **Backbone:** [`l3cube-pune/hing-mbert`](https://huggingface.co/l3cube-pune/hing-mbert), with two heads: intent
  on `[CLS]`, and one BIO slot tag per word (on its first word-piece). Loss = intent loss + w × slot loss.
- **Training loop written by hand in PyTorch** (`cmx/train.py`): AdamW, linear warm-up, fp16 autocast with a
  gradient scaler, optional gradient accumulation, on a Kaggle T4.
- **Tuning:** Optuna (TPE sampler + median pruner), 12 trials, scored **only on the dev sets** (Devanagari dev +
  Hinglish dev). Chosen: lr 5.0e-5, 6 epochs, batch 32, slot weight 1.49, warm-up 15%.
  [Trials](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/tune/trials.csv).
- **Recipes**, each changing one thing (the code counts changes and refuses a run that changes two):

| recipe | training data |
|---|---|
| plain | MASSIVE Hindi train, Devanagari (11,514) |
| romanized | plain + the same sentences auto-romanized (ITRANS) |
| english | plain + the English train set |
| **fixed** | plain + romanized + English (34,542) |
| noisy | fixed, with Hinglish-style spelling noise on the romanized copies (*kamare* → *kamre*) |

- **Backbone ablation:** the fixed recipe on [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased).
- **3 seeds** (13, 42, 1337) for each of the 6 models: 18 runs.

## Results

All numbers pool the 3 seeds. 95% ranges come from a **paired bootstrap** (2,000 draws): each draw resamples the
sentences once and uses that draw for every model and both scripts, so differences get honest, narrow ranges.
Three sentences corrected after training are left out (557 scored).
[Full report](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/charts/evaluate/results.md).

### 1. Every model

| model | Hindi test (2,974): intent / macro-F1 / slot F1 | Devanagari paired: intent | **Hinglish paired: intent** | Hinglish slot F1 |
|---|---|---|---|---|
| plain | 82.9 / 77.4 / 64.5 | 82.0 (79.1–84.9) | **48.3** (44.8–51.7) | 43.1 |
| romanized | 84.1 / 79.9 / 68.8 | 84.5 | 73.3 (70.1–76.5) | 61.2 |
| english | 84.4 / 80.5 / 70.2 | 85.0 | 81.4 (78.4–84.3) | 65.1 |
| **fixed** | 84.7 / 80.5 / 70.1 | 85.4 (82.7–88.0) | **85.8** (83.2–88.4) | 68.8 |
| noisy | 84.7 / 80.7 / 70.7 | 85.1 | 86.8 (84.1–89.3) | 69.0 |
| fixed, MuRIL | 86.6 / 78.2 / 61.4 | 85.6 | 83.4 (80.7–86.1) | 52.8 |

### 2. The drop, and which part of the fix matters

| comparison (points) | intent | slot F1 |
|---|---|---|
| plain: Devanagari − Hinglish | **33.7** (29.9 to 37.5) | 18.1 (14.5 to 21.9) |
| fixed: Devanagari − Hinglish | **−0.4** (−2.8 to 2.0) | −0.9 (−3.8 to 2.1) |
| Hinglish gain over plain: + romanized only | +25.0 (21.7 to 28.4) | +18.1 |
| Hinglish gain over plain: + English only | +33.1 (29.8 to 36.6) | +22.0 |
| Hinglish gain over plain: + both (fixed) | **+37.5** (34.0 to 41.2) | +25.7 |
| spelling noise on top of fixed | +1.0 (−0.3 to 2.3): not significant | +0.2 |
| HingBERT − MuRIL on Hinglish (both fixed) | +2.4 (0.3 to 4.5) | **+16.0** (12.5 to 19.4) |

The fix also helps ordinary Devanagari (82.9 → 84.7 on the full Hindi test). MuRIL matches HingBERT on intent
but is far worse on slots. Seed-to-seed spread on Hinglish intent is at most 4 points (plain) and under
1.5 for the fixed model.

### 3. The degradation curve

![Intent accuracy on Hinglish by CMI band](https://huggingface.co/vaibhav-4-ai/codemix-nlu/resolve/main/charts/evaluate/degradation.png)

| Hinglish intent accuracy | none (54) | light (91) | medium (221) | heavy (191) |
|---|---|---|---|---|
| plain | 37.7 | 48.7 | 48.4 | 51.0 |
| + romanized only | **71.0** | 79.9 | 73.6 | 70.5 |
| + English only | 53.7 | 80.2 | 86.4 | **83.9** |
| fixed (both) | 72.2 | 87.5 | 89.6 | 84.3 |

**The hardest Hinglish has no English words at all**: pure romanized Hindi ("kal subah saat baje utha dena") gives
the model nothing familiar. Romanized training data is what fixes that band, while English training data is
what fixes the heavily mixed bands. That is why the fix needs both.

### 4. Is the confidence honest?

Temperature scaling (one number T, fitted on the **dev** sets only) per CMI band.
[Full report](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/charts/calibrate/results.md).

| fixed model, Hinglish test | accuracy | ECE before | ECE after |
|---|---|---|---|
| all | 85.7 | 0.076 | **0.017** |
| none / light / medium / heavy | 72.2 / 87.7 / 89.3 / 84.4 | 0.159 / 0.051 / 0.058 / 0.095 | 0.076 / 0.061 / 0.036 / 0.027 |

![Reliability diagrams](https://huggingface.co/vaibhav-4-ai/codemix-nlu/resolve/main/charts/calibrate/reliability.png)

**Ask instead of act:** pick the confidence cut-off on dev so that answered requests are 90% right, then apply it
to test. The fixed model answers **89.8%** of Hinglish requests at **90.9%** accuracy. The plain model's cut-off,
tuned the same way, answers only 33.6% of Hinglish requests, **and those are only 77.8% right**: a safety net
tuned on Devanagari leaks on Hinglish. (Fitting T with the Hinglish dev set included gave the same T for the
fixed model, 1.45.)

### 5. Simple models first

`python -m cmx baselines`, in CI. Same recipes, same paired sentences.

| model | Devanagari paired | Hinglish paired |
|---|---|---|
| TF-IDF char n-grams + logistic regression, plain (intent) | 73.2 | 12.1 |
| TF-IDF + logistic regression, fixed (intent) | 73.2 | 69.6 |
| LightGBM on SVD(TF-IDF) + 6 code-mixing features, fixed (intent) | 57.3 | 48.6 |
| CRF, plain (slot F1) | 68.1 | 14.8 |
| CRF, fixed (slot F1) | 68.3 | 48.8 |

The fine-tuned model beats the best simple model by 16 points on Hinglish intent and 20 on Hinglish slots. A CRF
matches it on **Devanagari** slots. **Nested cross-validation check:** picking the best of 4 settings and reporting
that same CV score gave 74.8%; the honest nested estimate gave 74.7%. The gap is tiny here (one parameter, up to 50
sentences per intent), and the BERT pipeline avoids the problem the same way: Optuna only sees the dev sets.

### 6. Transfer: Google's human-written Hinglish-TOP

[Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset) (Agarwal et al., 2022,
arXiv 2211.07514, Apache-2.0). Only the human-annotated test file is used (pinned commit, checksum checked).
Its intents are mapped to MASSIVE's only where the meaning is the same (17 intents, 4,262 requests). Every
dropped intent is listed with its reason in the
[report](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/charts/transfer/report.md), e.g. timers
(MASSIVE has none) and "send a message" (MASSIVE splits it across email and social).

| model | English | Hinglish |
|---|---|---|
| always guess the most common intent | 21.6 | 21.6 |
| TF-IDF, plain | 15.9 | 8.7 |
| TF-IDF, fixed | 77.0 | 70.1 |
| **fixed hing-mbert, int8 (the served model)** | **89.1** | **85.3** |

Ceiling: the same int8 model on our own Hinglish test, same intents: 85.4.

### 7. Serving

`python -m cmx export`: ONNX export, then dynamic int8 quantization with unsigned 8-bit weights:
**712 MB → 178 MB**. It keeps 93% of intent decisions identical to the full model; Hinglish intent 86.2 → 83.2,
slot F1 69.6 → 68.2. Signed 8-bit weights gave garbage on the live server's older CPU (no VNNI instructions)
while working on GitHub's; `scripts/check_serving.py` now runs the model exactly as the server does, in CI. A safety gate stops the upload if int8 loses more than 10 points (added after
a per-channel export silently broke the model, 1.6% accuracy, once). The served page runs this model next to the TF-IDF plain model inside the
[deferral-curve](https://github.com/vaibhav-4-ai/deferral-curve) container (512 MB, 0.1 CPU) with onnxruntime +
tokenizers only (no torch): **[/codemix](https://deferral-curve.onrender.com/codemix)**. Measured live (2026-11-20, 12
requests): **442 MB peak memory, p50 104 ms, p95 147 ms** on 0.1 CPU; current numbers on
[/stats](https://deferral-curve.onrender.com/stats).

### 8. Errors

Real examples are in [errors.md](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/charts/evaluate/errors.md).
The fixed model gets 30 of 557 requests right in Devanagari but wrong in Hinglish. Its most common Hinglish
mistake is **mixing up opposites when the direction word is Hindi**:

| Hinglish | true | guessed |
|---|---|---|
| hallway me lights **tez** karo ("brighter") | `iot_hue_lightup` | `iot_hue_lightdim` |
| please speaker ka volume **dheere** karo ("quieter") | `audio_volume_down` | `audio_volume_up` |

Opposite pairs (up↔down, on↔off) are 3 of its 8 most common mix-ups. In slots, **date** suffers most among the
common types (90.8% of date slots found in Devanagari, 65.7% in Hinglish). A next step would be targeted
augmentation of Hindi direction words and dates.

## Limitations

- The paired Hinglish is **AI-drafted and AI-checked**, not written by many real users. Hinglish-TOP (human-written)
  agrees with it (85.3 vs 85.4), but slang, typos and heavy abbreviation ("kl", "krna") are under-represented.
- The CMI tagger is a word-list tagger: short words like "to" / "me" are ambiguous, and unknown words default to
  Hindi.
- The "none" band has only 54 sentences, so its ranges are wide.
- Slot F1 on Hinglish-TOP is not reported: its slot labels are a different scheme.
- The served model is one seed (13), int8; the tables use the full-size models.
- **Change from the original plan:** HingBERT (`l3cube-pune/hing-bert`) has only 70 Devanagari tokens in its
  vocabulary and can't read the Devanagari training data, so hing-mbert is used.

## Prior work

- **Hinglish-TOP / CST5** (Agarwal et al., 2022, arXiv 2211.07514): 10k human-written Hinglish requests with
  semantic parses. Used here as the transfer test.
- arXiv [2204.12753](https://arxiv.org/abs/2204.12753): *A Comprehensive Understanding of Code-mixed Language
  Semantics using Hierarchical Transformer*.
- arXiv [1812.02914](https://arxiv.org/abs/1812.02914): *Intent Detection for code-mix utterances in task oriented
  dialogue systems*.
- Krishnan et al., 2021: multilingual code-switching augmentation for zero-shot cross-lingual intent and slot
  prediction.

As far as I found, none of these measures **the same sentences with identical labels in Devanagari and in
Hinglish**, sliced by how mixed they are, with calibration.

## Use it

```bash
pip install codemix-nlu
```
```python
from cmx.features import code_mixing
code_mixing("kal subah 7 baje alarm set kar do")
# {'cmi': 28.57, 'i_index': 0.333, 'english_share': 0.286, 'script_latin': 1.0, 'variant': 0.143,
#  'oov': 0.0, 'band': 'medium'}
```
The word lists ship with the package. To run the model itself, use `serving/` on
[Hugging Face](https://huggingface.co/vaibhav-4-ai/codemix-nlu) (int8 ONNX, see the model card).

## Reproduce

```bash
pip install -e ".[dev,baselines]"
bash scripts/reproduce.sh       # everything that runs on a CPU, from the saved predictions (~30 min)
```

The GPU part (tuning + 18 training runs, ~6 h on a Kaggle T4) is one script: in a Kaggle notebook with a
`HF_TOKEN` secret, run `pip install -e ".[train]" && python scripts/kaggle_all.py`. It uploads each run as it
finishes and skips finished runs if restarted.

| stage | command | where it runs |
|---|---|---|
| data analysis + CMI | `python -m cmx eda` | CI |
| simple models + nested CV | `python -m cmx baselines` | CI |
| tuning, training | `scripts/kaggle_all.py` | Kaggle T4 |
| results, ablations, curve | `python -m cmx evaluate` | CI |
| calibration | `python -m cmx calibrate` | CI |
| int8 export | `python -m cmx export` | CI |
| Hinglish-TOP transfer | `python -m cmx transfer` | CI |

## License

Code: Apache 2.0. MASSIVE: CC BY 4.0, © Amazon.com, Inc. (FitzGerald et al., 2022). The paired Hinglish set:
CC BY 4.0. Hinglish-TOP: Apache-2.0, © Google.
