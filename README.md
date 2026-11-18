# codemix-nlu

**How much worse does a Hindi assistant understand you when you type in Hinglish, and what fixes it?**

Most Indians type Hindi in English letters, mixed with English words ("kal subah 7 baje ka alarm laga do").
Hindi language models are usually trained on Hindi in Devanagari. This project measures the cost of that mismatch
on a real NLU task (intent + slots) and tests a cheap fix.

**Result (3 training runs, 95% ranges from a paired bootstrap):**

- A model trained the usual way (Devanagari only) gets the intent right for **72%** of test sentences in
  Devanagari, but only **38%** of **the same sentences** typed in Hinglish: a drop of **34 points** (22 to 47).
- Adding auto-romanized Hindi and English to training raises Hinglish to **83%** (+45 points, 35 to 55). The gap
  between the two scripts disappears (−5 points, range −15 to +4, which includes zero).
- The same fix also helps on ordinary Devanagari: on MASSIVE's full Hindi test set (2,974 sentences), intent
  accuracy goes **79.2% → 83.8%** and slot F1 **50.6% → 66.6%**.

🔗 **[Try the demo](https://vaibhav-4-ai.github.io/codemix-nlu/)**: type Hinglish and see both models side by side
(the demo runs the small TF-IDF version in your browser, not the BERT model below).

---

## What the task is

NLU turns a sentence into a note an app can act on:

| Input | Intent | Slots |
|---|---|---|
| `kal subah 7 baje utha dena` | `alarm_set` | `date = kal subah`, `time = 7 baje` |

It's not translation: the output is a label plus the key details, not another sentence.

## The experiment

1. **Data:** [MASSIVE](https://github.com/alexa/massive) (Amazon, CC BY 4.0), Hindi (`hi-IN`) and English
   (`en-US`), 60 intents, with slots marked in BIO tags.
2. **Paired test set:** 50 Hindi test sentences, one per intent, each with at least one slot. Each one is rewritten
   in Hinglish with **identical labels**, so the only thing that changes is how the sentence is written.
3. **Model:** [`l3cube-pune/hing-mbert`](https://huggingface.co/l3cube-pune/hing-mbert) with two heads: intent on
   `[CLS]`, and one slot tag per word (first word-piece). Trained jointly with a hand-written PyTorch loop: AdamW,
   lr 3e-5, 4 epochs, batch 32, fp16 on a Kaggle T4.
4. **Two training recipes:**
   - **plain**: MASSIVE Hindi train set (11,514 sentences, Devanagari).
   - **fixed**: plain + the same sentences auto-romanized (ITRANS) + the English train set (34,542 sentences).
5. **Three seeds** (13, 42, 1337) for each recipe. All predictions are saved on
   [Hugging Face](https://huggingface.co/vaibhav-4-ai/codemix-nlu/tree/main/runs), and CI recomputes every number
   below from them.

## Results

Intent accuracy and slot F1, pooled over 3 seeds. 95% ranges come from a **paired** bootstrap with 2,000 draws:
each draw resamples the 50 sentences and uses the same draw for both scripts and both models.

| Model | Test sentences | Intent accuracy % | Slot F1 % |
|---|---|---|---|
| plain | MASSIVE test, Devanagari (n=2,974) | 79.2 | 50.6 |
| fixed | MASSIVE test, Devanagari (n=2,974) | 83.8 | 66.6 |
| plain | 50 paired, Devanagari | 72.0 (60.0 to 83.3) | 42.5 (30.9 to 55.3) |
| plain | 50 paired, **Hinglish** | **38.0** (27.3 to 48.7) | **17.1** (8.4 to 26.2) |
| fixed | 50 paired, Devanagari | 77.3 (66.0 to 88.0) | 60.4 (49.0 to 72.8) |
| fixed | 50 paired, **Hinglish** | **82.7** (72.7 to 90.7) | **69.2** (58.9 to 79.5) |

| Difference (points) | Intent | Slot F1 |
|---|---|---|
| plain: Devanagari − Hinglish | **34.0** (22.0 to 46.7) | 25.4 (12.3 to 39.7) |
| fixed: Devanagari − Hinglish | −5.3 (−14.7 to 4.0) | −8.8 (−18.5 to 1.3) |
| Hinglish: fixed − plain | **44.7** (34.7 to 54.7) | 52.1 (39.2 to 65.4) |

Across the three seeds, full-test accuracy varied by under 1.5 points per recipe.

### Is the confidence honest? (calibration)

When a model says it's "90% sure", is it right 90% of the time? ECE measures the average gap between confidence
and accuracy: 0 means honest. Temperature scaling fits one number T on Devanagari test sentences (cross-fitted,
never the 50 paired ones) and applies it unchanged everywhere. 3 seeds pooled; `python -m cmx calibrate`.

| Model | Sentences | Accuracy % | Avg confidence % (before → after) | ECE (before → after) |
|---|---|---|---|---|
| plain | MASSIVE test, Devanagari | 79.3 | 81.6 → 81.6 | 0.024 → 0.024 |
| plain | 50 paired, Hinglish | 38.0 | 38.5 → 38.5 | 0.097 → 0.097 |
| fixed | MASSIVE test, Devanagari | 83.9 | 91.3 → 86.0 | 0.074 → 0.021 |
| fixed | 50 paired, Hinglish | 82.7 | 89.8 → 83.2 | 0.101 → 0.067 |

- **The plain model fails honestly on Hinglish:** its average confidence falls to about 38%, which matches its
  accuracy. That makes the failure detectable: a system can send low-confidence requests to a fallback (the idea
  behind my other project, [deferral-curve](https://github.com/vaibhav-4-ai/deferral-curve)).
- **The fixed model is overconfident** (91% sure, 84% right). One temperature (T = 1.25, the same for all 3 seeds)
  brings the gap down: ECE drops from 0.074 to 0.021. It was fitted on Devanagari and still helps on Hinglish.
- ECE on 150 answers (50 sentences × 3 seeds) is noisy, so treat the Hinglish rows as indicative.

**A first, simpler version** (TF-IDF character n-grams + logistic regression, in CI on every push) shows the same
pattern: Devanagari-only training scores 12% on Hinglish, and adding romanized + English data raises it to 68%.

## Limitations: read before quoting the numbers

- **The Hinglish test set is small (50 sentences)**, which is why the ranges are wide. The direction of the
  result is clear; its exact size is not.
- **The Hinglish sentences were drafted by an AI assistant (Claude)** and reviewed by me; I edited 0 of 50.
  Real users' Hinglish may be messier (more spelling variation, more English). Annotations I type myself are
  planned as a second test set.
- **No hyper-parameter tuning yet.** Settings were chosen once by hand. Slot F1 was still rising at epoch 4, so
  these slot numbers are likely below what this model can reach.
- **The fixed recipe adds two things at once** (romanized Hindi and English). They haven't been separated yet.
- **Changes from the original plan:** the plan used HingBERT (`l3cube-pune/hing-bert`), but its vocabulary has
  only 70 Devanagari tokens, so it can't read the Devanagari training data. hing-mbert (1,852) is used instead. The
  plan also called for a ~600-sentence hand-typed test set; this version uses 50.

## Prior work

- **Hinglish-TOP** (Google, [`WillHeld/hinglish_top`](https://huggingface.co/datasets/WillHeld/hinglish_top)):
  a human-written Hinglish semantic-parsing set. It's the closest existing resource, and a transfer evaluation on
  it is planned.
- arXiv [2204.12753](https://arxiv.org/abs/2204.12753): joint intent + slot on Hindi code-mixed text.
- arXiv [1812.02914](https://arxiv.org/abs/1812.02914): a code-mixed English-Hindi intent dataset.
- Krishnan et al., 2021: multilingual code-switching augmentation for zero-shot cross-lingual intent and slot
  prediction.
- Recent Hinglish intent datasets on Hugging Face (`Sachin-099/hinglish-support-intents-india`,
  `Hari5115/hinglish-retail-intent-dataset`) are LLM-generated, intent-only, and have no Devanagari pairing.

As far as I found, none of these measures the same sentences, with identical labels, in Devanagari and in Hinglish.

## Reproduce

```bash
pip install -e ".[dev]"
python -m cmx fetch-data        # download MASSIVE (hi-IN, en-US)
python -m cmx quick             # the TF-IDF version, CPU, ~1 minute
pip install -e ".[train]"
python -m cmx train --recipe plain --seed 13   # GPU: ~4 minutes on a T4
python -m cmx train --recipe fixed --seed 13   # GPU: ~9 minutes on a T4
python -m cmx evaluate          # reads predictions from artifacts/hf_runs/ (CI downloads them)
python -m cmx calibrate         # ECE + temperature scaling from the same predictions
```

Layout: `cmx/data.py` (download), `cmx/massive.py` (slot markup → BIO tags), `cmx/quick.py` (TF-IDF version and
demo export), `cmx/examples.py` (training sets, tag alignment), `cmx/train.py` (model and training loop),
`cmx/evaluate.py` (paired bootstrap), `cmx/calibrate.py` (ECE, temperature scaling), `docs/` (demo page).

## Status and future work

**Complete as of November 2026:** data pipeline, a paired Hinglish test set, the TF-IDF baseline with a live demo,
joint BERT training (2 recipes × 3 seeds), paired-bootstrap evaluation, and calibration. Every number on this page
is recomputed in CI from the saved predictions.

**Not done (cut for time):** transfer to Hinglish-TOP, serving the BERT model (an int8 export), a bigger
hand-typed test set, calibration per code-mixing level, hyper-parameter tuning, and separating the two parts of
the fix.

## License

Code: Apache 2.0. MASSIVE data: CC BY 4.0, © Amazon.com, Inc. (FitzGerald et al., 2022).
