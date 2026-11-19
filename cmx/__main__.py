import argparse

# One subcommand per stage of the build. Each one imports its own code inside its branch,
# so `python -m cmx --help` stays fast and never loads torch.
COMMANDS = (
    "quick",  # the quick version: first rough number, TF-IDF, seconds on CPU
    "fetch-data",  # Stage 1: download MASSIVE, check its checksum, extract Hindi + English
    "pick-sets",  # Stage 1: choose the test (10 per intent) and dev (7 per intent) sentences
    "check-annotations",  # Stage 1b: validate the Hinglish rewrites, build the final files
    "eda",  # Stage 2: data analysis + code-mixing features, with charts
    "baselines",  # Stage 3: TF-IDF/LogReg, LightGBM on features, CRF, nested CV
    "train",  # Stage 4: the hand-written PyTorch training loop
    "tune",  # Stage 5: Optuna writes configs/best_params.json
    "calibrate",  # Stage 6: temperature scaling, ECE per code-mixing band
    "evaluate",  # Stage 7: the degradation curve, CIs, ablations, error analysis
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
            p.add_argument(
                "--recipe", choices=["plain", "romanized", "english", "fixed", "noisy"], default="fixed"
            )
            p.add_argument("--seed", type=int, default=13)
            p.add_argument("--backbone", default="l3cube-pune/hing-mbert")
        if name == "tune":
            p.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    if args.command == "train":
        from cmx.train import train

        train(args.recipe, args.seed, args.backbone)
    elif args.command == "tune":
        from cmx.tune import run

        run(args.trials)
    elif args.command == "quick":
        from cmx.quick import run

        run(args.export)
    elif args.command == "fetch-data":
        from cmx.data import fetch

        fetch()
    elif args.command == "pick-sets":
        from cmx.data import pick_sets

        pick_sets()
    elif args.command == "check-annotations":
        from cmx.annotations import build

        build()
    elif args.command == "eda":
        from cmx.eda import run

        run()
    elif args.command == "baselines":
        from cmx.baselines import run

        run()
    elif args.command == "calibrate":
        from cmx.calibrate import run

        run()
    elif args.command == "evaluate":
        from cmx.evaluate import run

        run()
    else:
        print(f"{args.command}: not implemented")


if __name__ == "__main__":
    main()
