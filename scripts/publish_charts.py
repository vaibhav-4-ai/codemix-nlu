"""Copy every chart CI drew (artifacts/<stage>/*.png) to Hugging Face, so the README can show them.

Runs only in CI, only when the HF_TOKEN secret is set. Unchanged charts make no new commit.
"""

import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = "vaibhav-4-ai/codemix-nlu"

if not os.environ.get("HF_TOKEN"):
    print("no HF_TOKEN secret - charts not uploaded")
else:
    api = HfApi()
    for stage in sorted(Path("artifacts").iterdir()):
        if stage.is_dir() and list(stage.glob("*.png")):
            api.upload_folder(
                folder_path=str(stage),
                path_in_repo=f"charts/{stage.name}",
                repo_id=REPO,
                allow_patterns=["*.png", "*.md", "*.csv", "*.json"],
                commit_message=f"CI: {stage.name} charts and report",
            )
            print(f"uploaded artifacts/{stage.name} -> charts/{stage.name}")
