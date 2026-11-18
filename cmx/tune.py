"""Stage 5: let Optuna choose the training settings, so nothing is hand-tuned.

Each trial trains the main model (fixed recipe, hing-mbert, seed 13) with settings Optuna picks,
and is scored on the DEV sets only (Devanagari dev + the Hinglish dev set) - never on test.
Trials that look hopeless after an epoch are stopped early (pruning) to save GPU time.

Output: configs/best_params.json (every later training run reads it) + artifacts/tune/trials.csv
"""

import json
from pathlib import Path

import optuna

from cmx.train import BACKBONE, BEST_PARAMS, DEFAULT_PARAMS, REFERENCE_RECIPE, train

SEED = 13
OUT = Path("artifacts") / "tune"


def choose_params(trial):
    params = dict(DEFAULT_PARAMS)
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True)
    params["epochs"] = trial.suggest_int("epochs", 3, 6)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
    params["slot_weight"] = trial.suggest_float("slot_weight", 0.5, 2.0)
    params["rare_slot_weighting"] = trial.suggest_categorical("rare_slot_weighting", [False, True])
    params["warmup"] = trial.suggest_float("warmup", 0.0, 0.2)
    return params


def objective(trial):
    params = choose_params(trial)
    result = train(REFERENCE_RECIPE, SEED, BACKBONE, params=params, out=OUT / f"trial_{trial.number}", trial=trial)
    if result is None:
        raise optuna.TrialPruned()
    return result


def run(trials):
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=trials)

    best = dict(DEFAULT_PARAMS)
    best.update(study.best_params)
    BEST_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "best_params.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")

    lines = ["trial,state,score,learning_rate,epochs,batch_size,slot_weight,rare_slot_weighting,warmup\n"]
    for t in study.trials:
        p = t.params
        lines.append(
            f"{t.number},{t.state.name},{t.value},{p.get('learning_rate')},{p.get('epochs')},"
            f"{p.get('batch_size')},{p.get('slot_weight')},{p.get('rare_slot_weighting')},{p.get('warmup')}\n"
        )
    (OUT / "trials.csv").write_text("".join(lines), encoding="utf-8")
    print(f"best score {study.best_value:.4f} with {best}")
    print(f"saved {BEST_PARAMS} and {OUT}/trials.csv")
