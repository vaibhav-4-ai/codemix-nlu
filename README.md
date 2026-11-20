# codemix-nlu

**A Hindi voice-assistant model misroutes 52% of requests typed in Hinglish. Retraining it on the same data in
two more forms cuts that to 14%: about 3 in 4 misrouted requests fixed, with no new labelled data.**

**[Try it live](https://deferral-curve.onrender.com/codemix)** · `pip install codemix-nlu` ([use it](#use-it)) ·
[limitations](#limitations)

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
- **Rare slot types:** 19 of 55 slot types appear fewer than 50 times in training. The loop can weight each tag's
  loss by 1/√(its count). Optuna had it on in 8 of its 12 trials; the best of those scored 0.720 on dev against
  0.776 for the best trial, which had it **off**. Other settings changed at the same time, so this is a hint, not
  a clean test. How rare types fare is measured in [Robustness](#9-robustness-spelling-and-rare-slots).
- **Loss weighting:** loss = intent loss + 1.49 × slot loss. Optuna chose the 1.49 on dev (searched 0.5 to 2),
  not by hand.

### GPU memory, worked out (Kaggle T4, 16 GB)

hing-mbert has about **178 M parameters** (the saved fp32 weights are 712 MB at 4 bytes each). Half of them are
the word-piece table: 119,547 pieces × 768 numbers ≈ 92 M. Training with AdamW needs, per parameter:

| what | bytes per parameter | total |
|---|---|---|
| weights (fp32) | 4 | 0.71 GB |
| gradients (fp32) | 4 | 0.71 GB |
| AdamW's two running averages (fp32) | 8 | 1.42 GB |
| **subtotal** | 16 | **2.85 GB** |

Activations, for a batch of 32 sentences of up to 64 word-pieces in fp16 (autocast) through 12 layers, come to
at most ~0.7 GB (rough rule: sequence × batch × hidden × ~39 bytes per layer). Total **≈ 3.6 GB of 16 GB**, so
gradient accumulation (in the loop, set to 1) and gradient checkpointing (re-computing activations to save
memory, ~30% slower) weren't needed. This is an estimate: the 18 runs didn't record peak memory. The loop now
records it (`gpu_peak_memory_gb` in `metrics.json`) for future runs.

At 8 GPUs, ZeRO stage 1 would split the 1.42 GB of optimizer state across them (≈ 0.18 GB each), stage 2 the
gradients too, and stage 3 / FSDP the weights as well. I haven't run multi-GPU training; at this model size one
GPU is enough.

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

Intent accuracy with 95% ranges (2,000 paired resamples), then macro-F1 (every intent counts the same):

| model | English | Hinglish | macro-F1 English / Hinglish |
|---|---|---|---|
| always guess the most common intent | 21.6 | 21.6 | - |
| TF-IDF, plain | 15.9 (14.9–17.0) | 8.7 (7.9–9.5) | 12.6 / 8.6 |
| TF-IDF, fixed | 77.0 (75.7–78.3) | 70.1 (68.7–71.4) | 69.1 / 63.5 |
| **fixed hing-mbert, int8 (the served model)** | **89.1** (88.2–90.0) | **85.3** (84.3–86.3) | 83.8 / 79.2 |

Ceiling: the same int8 model on our own Hinglish test, same intents: 85.4 (79.2–91.5; only 130 sentences, so the
range is wide). English → Hinglish costs the served model 3.8 points (2.9–4.6) on these new writers. The test file has 6,513
human-written rows, and 4,262 fall in the 17 mapped intents. No Hinglish-TOP data is ever used for training.
The int8 model can give a different answer on a handful of sentences on different CPU types, so the last digit of
these ranges can move by 0.1 between CI runs.

### 7. Serving

`python -m cmx export`: ONNX export, then dynamic int8 quantization with unsigned 8-bit weights:
**712 MB → 178 MB**. It keeps 93% of intent decisions identical to the full model; Hinglish intent 86.2 → 83.2,
slot F1 69.6 → 68.2. Signed 8-bit weights gave garbage on the live server's older CPU (no VNNI instructions)
while working on GitHub's ([postmortem](https://github.com/vaibhav-4-ai/deferral-curve/blob/main/POSTMORTEM-01.md));
`scripts/check_serving.py` now runs the model exactly as the server does, in CI, and a daily job checks the live
answers. A safety gate stops the upload if int8 loses more than 10 points (added after
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

### 9. Robustness: spelling and rare slots

**Unfamiliar spellings.** Each Hinglish sentence is grouped by the share of its English-letter words that
training never showed in that spelling (*kamre* when training only had *kamare*, or not in any word list).
[Report](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/charts/evaluate/results.md), section 7.

| Hinglish intent accuracy | all familiar (171) | some unfamiliar, up to 20% (137) | many unfamiliar, over 20% (249) |
|---|---|---|---|
| plain | 49.5 (43.1–55.8) | 54.0 (47.0–61.1) | 44.3 (39.2–49.5) |
| fixed | 86.9 (82.3–91.2) | 88.8 (83.9–93.2) | 83.3 (78.8–87.3) |
| noisy (fixed + spelling noise) | 88.7 (84.2–92.8) | 89.8 (84.9–94.2) | 83.8 (79.7–87.7) |

The fixed model loses about 3.6 points on heavily misspelled sentences, and the ranges overlap, so spelling
isn't its main weakness. The spelling-noise recipe, built for exactly these sentences, doesn't help them
(83.8 vs 83.3).

**Rare slot types.** 19 of 55 slot types appear fewer than 50 times in training. Slot F1 on the paired test,
counted per group of types (46 / 358 / 242 slots):

| slot F1 | rare (under 50) | medium (50–499) | common (500+) |
|---|---|---|---|
| plain, Hinglish | 16.9 (4.1–30.0) | 38.3 (33.7–42.7) | 52.5 (47.5–57.6) |
| fixed, Devanagari | 53.2 (39.0–65.9) | 68.5 (64.2–72.8) | 69.3 (64.2–74.5) |
| fixed, Hinglish | 57.8 (42.7–70.4) | 71.1 (66.8–75.0) | 67.0 (61.9–71.9) |

**Slot PR-AUC** (full-size fixed model, seed 13, word level: P(B-type) + P(I-type) against "is this word part of
that slot"), averaged over slot types so a rare type counts as much as a common one:

| set | rare types | common types |
|---|---|---|
| full Hindi test (2,974) | **0.476** (17 types) | 0.778 (36 types) |
| Hinglish paired test (560) | 0.649 (14 types) | 0.795 (36 types) |

**Rare slot types are the model's clearest weak spot**, in Devanagari as much as in Hinglish: 13–15 points of
slot F1 below the medium types, and 0.30 lower PR-AUC on the full Hindi test. Only 46 rare slots are in the paired
test, so their ranges are wide; the PR-AUC on 2,974 sentences confirms the gap. Weighting rare tags in the loss
was tried during tuning (see Model and training) and wasn't chosen.

## Limitations

- The paired Hinglish is **AI-drafted and AI-checked**, not written by many real users. Hinglish-TOP (human-written)
  agrees with it (85.3 vs 85.4), but slang, typos and heavy abbreviation ("kl", "krna") are under-represented.
- The CMI tagger is a word-list tagger: short words like "to" / "me" are ambiguous, and unknown words default to
  Hindi.
- The "none" band has only 54 sentences, so its ranges are wide.
- Slot types that are rare in training are found much less often (slot PR-AUC 0.48 vs 0.78 on the Hindi test).
- Slot F1 on Hinglish-TOP is not reported: its slot labels are a different scheme.
- The served model is one seed (13), int8; the tables use the full-size models.
- **Change from the original plan:** HingBERT (`l3cube-pune/hing-bert`) has only 70 Devanagari tokens in its
  vocabulary and can't read the Devanagari training data, so hing-mbert is used.
- **Planned but not done:**
  - Human annotation of the Hinglish with a second annotator and an agreement score (Cohen's κ).
  - A slice of real users' Hinglish requests from public sources, with a consent note.
  - The reverse transfer direction (train on Hinglish-TOP, test on this set).
## Prior work

**The closest: Hinglish-TOP / CST5** (Agarwal et al., 2022, [arXiv 2211.07514](https://arxiv.org/abs/2211.07514)).
About 10k human-written Hinglish requests with intent + slot labels, paired with **English**, on TOP's
hierarchical parses, in 9 assistant domains. This project pairs Hinglish with **Devanagari Hindi** instead, on
MASSIVE's flat 60-intent / 55-slot labels, sliced by CMI and with calibration. Hinglish-TOP is used here as the
transfer test.

Data and benchmarks:
- **MASSIVE** (FitzGerald et al., 2022, [arXiv 2204.08582](https://arxiv.org/abs/2204.08582)): the source of
  every label and the Devanagari and English sentences.
- **GLUECoS** (Khanuja et al., ACL 2020, [arXiv 2004.12376](https://arxiv.org/abs/2004.12376)): the code-switched
  benchmark (language ID, POS, NER, sentiment, QA, NLI). It has no intent or slot task.
- **Naamapadam / IndicNER** (Mhaske et al., ACL 2023, [arXiv 2212.10168](https://arxiv.org/abs/2212.10168)):
  named entities for 11 Indian languages in their **native scripts**, not romanized.
- **CodeMixBench**: two different papers share the name. One tests LLMs on code-mixed text in 18 languages
  (Yang & Chai, EMNLP 2025, [arXiv 2507.18791](https://arxiv.org/abs/2507.18791)); the other tests code
  generation from code-mixed prompts, including Hinglish (Sheokand & Sawant, 2025,
  [arXiv 2505.05063](https://arxiv.org/abs/2505.05063)).
- **CALICO** ([arXiv 2412.05388](https://arxiv.org/abs/2412.05388)): uses an LLM to localize intent + slot data
  (MultiATIS++) into other languages, close in spirit to how the Hinglish here was drafted.

Models:
- **L3Cube-HingCorpus and HingBERT** (Nayak & Joshi, 2022, [arXiv 2204.08398](https://arxiv.org/abs/2204.08398)):
  the Hinglish encoders, including the hing-mbert used here.
- **MuRIL** (Khanuja et al., 2021, [arXiv 2103.10730](https://arxiv.org/abs/2103.10730)): Indian-language BERT that
  already trains on transliterated text, so the romanization idea is not new in pre-training; here it is
  measured as fine-tuning data.
- **RomanSetu** (Husain et al., ACL 2024, [arXiv 2401.14280](https://arxiv.org/abs/2401.14280)): romanized text
  as a way into LLMs for Indian languages.
- **Code Mixologist** (Gupta et al., 2026, [arXiv 2602.11181](https://arxiv.org/abs/2602.11181)): a practitioner's
  guide to building code-mixed LLMs.

Measuring code-mixing: **CMI** (Das & Gambäck, 2014, *Identifying languages at the word level in code-mixed
Indian social media text*) and the **I-Index** (Guzmán et al., 2017, *Metrics for modeling code-switching
across corpora*).

Earlier code-mixed intent work: arXiv [2204.12753](https://arxiv.org/abs/2204.12753) (*A Comprehensive
Understanding of Code-mixed Language Semantics using Hierarchical Transformer*), arXiv
[1812.02914](https://arxiv.org/abs/1812.02914) (*Intent Detection for code-mix utterances in task oriented
dialogue systems*), and Krishnan et al., 2021 (multilingual code-switching augmentation for zero-shot
cross-lingual intent and slot prediction).

As far as I found, none of these measures **the same sentences with identical labels in Devanagari and in
Hinglish**, sliced by how mixed they are, with calibration.

## Use it

```bash
pip install codemix-nlu            # code-mixing numbers + the benchmark (no torch)
pip install "codemix-nlu[model]"   # + this project's int8 model (onnxruntime, ~180 MB download on first use)
```

**How mixed is a sentence?**
```python
from cmx.features import code_mixing
code_mixing("kal subah 7 baje alarm set kar do")
# {'cmi': 28.57, 'i_index': 0.333, 'english_share': 0.286, 'script_latin': 1.0, 'variant': 0.143,
#  'oov': 0.0, 'band': 'medium'}
```

**Benchmark your own model** on the 560 paired requests (Devanagari vs Hinglish, same labels):
```bash
codemix-nlu benchmark-inputs inputs.jsonl     # the 1,120 sentences to predict: {"id", "form", "words"}
# ... your model writes predictions.jsonl:     {"id", "form", "intent", "tags" (optional)}
codemix-nlu benchmark predictions.jsonl       # accuracy, macro-F1, slot F1, the drop with a 95% range, by CMI band
```
To see the format, run this project's model through the same steps:
`codemix-nlu predict-file inputs.jsonl predictions.jsonl`. One sentence: `codemix-nlu predict "kal subah 7 baje
ka alarm laga do"`. CI runs all of these on every push, the way a stranger would. Changes:
[CHANGELOG.md](https://github.com/vaibhav-4-ai/codemix-nlu/blob/main/CHANGELOG.md).

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

**What ran where:**

| | |
|---|---|
| training | Kaggle T4 (16 GB), 2026-11-19, seeds 13 / 42 / 1337. Library versions weren't recorded for these 18 runs (Kaggle's default image that day); the loop records them from now on |
| everything else | GitHub Actions, `ubuntu-24.04`, Python 3.11, CPU only, on every push. Exact library versions: end of the [export report](https://huggingface.co/vaibhav-4-ai/codemix-nlu/blob/main/serving/report.md) |
| live demo | Render free plan: 512 MB, 0.1 CPU, no VNNI instructions |
| data | MASSIVE 1.1 tarball and the Hinglish-TOP test file, both pinned by SHA-256 checksum |
| ranges | paired bootstrap, 2,000 draws, fixed seed 13 |

## License

Code: Apache 2.0. MASSIVE: CC BY 4.0, © Amazon.com, Inc. (FitzGerald et al., 2022). The paired Hinglish set:
CC BY 4.0. Hinglish-TOP: Apache-2.0, © Google.
