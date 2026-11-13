import argparse

# One subcommand per stage of the build. Each one will import its own code inside
# its branch, so `python -m cmx --help` stays fast and never loads torch.
COMMANDS = (
    "fetch-data",  # Stage 1: download MASSIVE, check its checksum, extract Hindi + English
    "check-annotations",  # Stage 1b: validate your typed Hinglish rows
    "eda",  # Stage 2: data analysis + engineered features
    "baselines",  # Stage 3: TF-IDF/LogReg, LightGBM, CRF
    "train",  # Stage 4: the hand-written PyTorch training loop
    "tune",  # Stage 5: Optuna writes best_params.json
    "calibrate",  # Stage 6: temperature scaling, ECE per code-mixing band
    "evaluate",  # Stage 7: the degradation curve, CIs, error analysis
    "transfer",  # Stage 7b: evaluate on Google's Hinglish-TOP
    "export",  # Stage 8: ONNX + int8 for serving
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cmx")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "fetch-data":
        from cmx.data import fetch

        fetch()
    else:
        print(f"{args.command}: not implemented")


if __name__ == "__main__":
    main()
