import argparse

# One subcommand per stage of the build. Each one will import its own code inside
# its branch, so `python -m cmx --help` stays fast and never loads torch.
COMMANDS = (
    "quick",  # the quick version: first rough number on your 50 sentences
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
        p = sub.add_parser(name)
        if name == "quick":
            p.add_argument("--export", help="folder to save the demo page's models into")
        if name == "train":
            p.add_argument("--recipe", choices=["plain", "fixed"], default="fixed")
            p.add_argument("--seed", type=int, default=13)
            p.add_argument("--backbone", default="l3cube-pune/hing-mbert")
    args = parser.parse_args()
    if args.command == "train":
        from cmx.train import train

        train(args.recipe, args.seed, args.backbone)
    elif args.command == "quick":
        from cmx.quick import run

        run(args.export)
    elif args.command == "calibrate":
        from cmx.calibrate import run

        run()
    elif args.command == "evaluate":
        from cmx.evaluate import run

        run()
    elif args.command == "fetch-data":
        from cmx.data import fetch

        fetch()
    else:
        print(f"{args.command}: not implemented")


if __name__ == "__main__":
    main()
