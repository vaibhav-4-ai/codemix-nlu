# Build log

A short record of the decisions I made while building codemix-nlu, and why, including what went wrong. The README
has the results.

## Decisions

- **Licence:** Apache 2.0 for the code. The paired Hinglish set is CC BY 4.0, because it is built from MASSIVE
  (CC BY 4.0).
- **Where things run:** training and tuning on a Kaggle T4 (one unattended script, `scripts/kaggle_all.py`, which
  uploads each run to Hugging Face as it finishes and skips finished runs on restart). Everything else (data
  analysis, simple models, every result table, calibration, int8 export, the transfer test) runs on GitHub Actions
  on every push, from the saved predictions. Nothing heavy runs on my laptop.
- **Backbone:** `l3cube-pune/hing-mbert`, not `hing-bert` as first planned: hing-bert's vocabulary has only 70
  Devanagari tokens, so it can't read the Devanagari training data. MuRIL was trained as the comparison.
- **Paired test set, not a separate Hinglish dataset:** every Hinglish sentence keeps the exact labels of a MASSIVE
  Hindi sentence, so a difference in accuracy can only come from the script and the mixing.
- **Who wrote the Hinglish:** an AI model (Claude) drafted all 952 sentences; I chose this to finish in time. Every
  row is checked automatically (same slot types as the Hindi), and a random 100 were checked again, also by AI
  (4 corrected). Because it is not human-written, I added a transfer test on Google's human-written Hinglish-TOP;
  the model scores 85.3% there against 85.4% on my set, so the AI-drafted set isn't flattering the model.
- **One change at a time:** five training recipes, each adding one thing. `train.py` refuses a run that changes
  more than one setting from the main run.
- **Tuning only on dev:** Optuna's score is the mean of intent accuracy and slot F1 on the Devanagari dev set and
  the 392-sentence Hinglish dev set. The test sets are only scored at the end.
- **Paired bootstrap** for every range: one resample of sentence ids is used for every model and both scripts, so
  differences between them get honest, narrow ranges.
- **Serving inside deferral-curve:** Render's free hours cover one always-on service, so the int8 model runs in
  deferral-curve's existing 512 MB container (onnxruntime + tokenizers, no PyTorch) instead of a second server.

## Things that went wrong, and what I changed

- **The LightGBM baseline first scored 1.8%**, which means it guessed one intent every time. Trees split badly on
  3,000 sparse TF-IDF columns; compressing them to 200 dense SVD columns fixed it (98% on its own training data).
  It now also reports its training accuracy, so a broken run is obvious.
- **Per-channel int8 quantization broke the model** (Hinglish accuracy 1.6%) and CI uploaded it before I noticed.
  I reverted it and added a gate: the export fails, and nothing is uploaded, if int8 loses more than 10 points.
- **The live page first answered "calendar_query" for everything**, while the same model scored 83% in CI.
  `scripts/check_serving.py` ran the model exactly as the server does (light tokenizer, memory options) and
  still got 83%, so the code was fine and the difference was the **CPU**: the server reports no VNNI
  instructions, and ONNX Runtime's signed 8-bit path can overflow on such CPUs. Unsigned 8-bit weights (QUInt8)
  fixed it: 12 of 12 test requests right on the live page, and int8 slot F1 went up (63.7 → 68.2).
- **4 of the 100 checked sentences were wrong** (a wrong time, a slot on the wrong word, two garbled word orders).
  They were fixed after training, so the results script leaves those sentences out instead of comparing old
  predictions with new answers.
- **Ruff's default rule set is large** (407 rules); CI now prints lint errors as annotations so they can be read
  without the full log.

## Results I didn't expect

- The hardest Hinglish has **no English words at all** (pure romanized Hindi), not the most mixed.
- Romanized training data fixes that band; English training data fixes the heavily mixed bands.
- Spelling noise on top of the fix adds +1.0 point, not significant.
- MuRIL matches HingBERT on intent but is 16 points worse on Hinglish slots.
- A Devanagari-only model's "ask instead of act" cut-off leaks on Hinglish: tuned for 90% right, it gets 77.8%.
- Adding the code-mixing index to deferral-curve's hand-off calibrator didn't help (0 of 3 seeds), because that
  bot isn't worse on mixed questions.

## Status

Done: data pipeline, paired Hinglish set (published on Hugging Face with a data card), data analysis, simple
models with nested CV, tuning, 6 models × 3 seeds, the degradation curve, calibration per band, error analysis,
int8 export, Hinglish-TOP transfer (with 95% ranges and macro-F1), live demo, one-command reproduce, the pip
package with a benchmark tool, slot PR-AUC and robustness slices, and a postmortem of the live incident.

Not done, on purpose or for lack of people: human annotation with an agreement score, a slice of real users'
requests, and the reverse transfer direction. They are listed in the README's limitations.
