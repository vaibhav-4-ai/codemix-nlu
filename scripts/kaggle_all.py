"""Everything that needs a GPU, in one go. Run inside the Kaggle notebook (see README).

1. Optuna tuning (12 trials)            -> configs/best_params.json
2. 5 recipes x 3 seeds on hing-mbert    -> the degradation result + "which part of the fix matters"
3. fixed recipe x 3 seeds on MuRIL      -> backbone ablation
Each finished run is uploaded to Hugging Face right away, so a crash never loses finished work.
Runs that are already on Hugging Face are skipped, so re-running continues where it stopped.
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

REPO = "vaibhav-4-ai/codemix-nlu"
SEEDS = [13, 42, 1337]
RECIPES = ["plain", "romanized", "english", "fixed", "noisy"]
MAIN = "l3cube-pune/hing-mbert"
MURIL = "google/muril-base-cased"

api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
done = set()
for path in api.list_repo_files(REPO):
    parts = path.split("/")
    if parts[0] == "runs_v2" and len(parts) > 2 and parts[2] == "metrics.json":
        done.add(parts[1])
print(f"already on Hugging Face: {sorted(done)}")


def run(command):
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


# 1. tuning
if api.file_exists(REPO, "tune/best_params.json"):
    Path("configs").mkdir(exist_ok=True)
    api.hf_hub_download(REPO, "tune/best_params.json", local_dir="artifacts_hf")
    Path("configs/best_params.json").write_text(
        Path("artifacts_hf/tune/best_params.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    print("tuning already done - using tune/best_params.json from Hugging Face")
else:
    run([sys.executable, "-m", "cmx", "tune", "--trials", "12"])
    api.upload_file(path_or_fileobj="artifacts/tune/best_params.json", path_in_repo="tune/best_params.json",
                    repo_id=REPO)
    api.upload_file(path_or_fileobj="artifacts/tune/trials.csv", path_in_repo="tune/trials.csv", repo_id=REPO)

# 2 + 3. every training run
jobs = []
for seed in SEEDS:
    for recipe in RECIPES:
        jobs.append((recipe, MAIN, seed))
    jobs.append(("fixed", MURIL, seed))

for recipe, backbone, seed in jobs:
    name = f"{recipe}_{backbone.split('/')[-1]}_seed{seed}"
    if name in done:
        print(f"skip {name} (already uploaded)")
        continue
    run([sys.executable, "-m", "cmx", "train", "--recipe", recipe, "--seed", str(seed), "--backbone", backbone])
    api.upload_folder(folder_path=f"artifacts/train/{name}", repo_id=REPO, path_in_repo=f"runs_v2/{name}")
    print(f"uploaded {name}", flush=True)

print("\nALL DONE")
