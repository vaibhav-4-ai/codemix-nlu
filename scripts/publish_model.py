"""Upload the int8 model (for the demo server) and the export + transfer reports to Hugging Face.

Runs only in CI, only when the HF_TOKEN secret is set.
"""

import os

from huggingface_hub import HfApi

REPO = "vaibhav-4-ai/codemix-nlu"

if not os.environ.get("HF_TOKEN"):
    print("no HF_TOKEN secret - nothing uploaded")
else:
    api = HfApi()
    api.upload_folder(
        folder_path="artifacts/export",
        path_in_repo="serving",
        repo_id=REPO,
        commit_message="CI: int8 ONNX model for CPU serving",
    )
    api.upload_folder(
        folder_path="artifacts/transfer",
        path_in_repo="charts/transfer",
        repo_id=REPO,
        commit_message="CI: Hinglish-TOP transfer report",
    )
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=REPO,
        commit_message="CI: model card",
    )
    print("uploaded serving/, charts/transfer/ and the model card")
