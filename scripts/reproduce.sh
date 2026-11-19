#!/usr/bin/env bash
# Every number in the README that runs on a CPU, in order. Needs: pip install -e ".[dev,baselines]"
# (and ".[train,serve]" plus CPU torch for the last two steps). The GPU part is scripts/kaggle_all.py.
set -euo pipefail

python -m cmx quick                 # 1 min: TF-IDF first look
python -m cmx eda                   # 1 min: data analysis, CMI, charts
python -m cmx baselines             # ~20 min: TF-IDF, LightGBM, CRF, nested CV
python scripts/fetch_runs.py        # the 18 Kaggle runs' predictions from Hugging Face (no token needed)
python -m cmx evaluate              # ~3 min: drop, ablations, CMI curve, errors
python -m cmx calibrate             # ~3 min: ECE per CMI band, ask-instead-of-act cut-off
python -m cmx export                # ~5 min: int8 ONNX (downloads the 712 MB model)
python -m cmx transfer              # ~5 min: Google's Hinglish-TOP

echo "done - reports are in artifacts/*/"
