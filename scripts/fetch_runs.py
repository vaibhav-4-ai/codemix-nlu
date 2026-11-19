"""Download the finished Kaggle runs from Hugging Face into artifacts/hf/ (no token needed).

Only the small files: predictions, metrics, curves, tuning results - never the 700 MB model weights.
"""

from huggingface_hub import snapshot_download

REPO = "vaibhav-4-ai/codemix-nlu"

folder = snapshot_download(
    repo_id=REPO,
    local_dir="artifacts/hf",
    allow_patterns=["runs_v2/*/*.jsonl", "runs_v2/*/*.json", "runs_v2/*/*.csv", "tune/*"],
)
print(f"downloaded into {folder}")
