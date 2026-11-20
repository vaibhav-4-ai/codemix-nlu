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
    # For anyone who installs the package (no torch needed):
    "cmi",  # code-mixing numbers for one sentence
    "predict",  # this project's int8 model on one sentence (needs the [model] extra)
    "benchmark-inputs",  # write the 560 x 2 paired sentences your model should predict
    "predict-file",  # run this project's int8 model on that file
    "benchmark",  # score a predictions file: accuracy, macro-F1, slot F1, the drop, per CMI band
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
        if name in ("cmi", "predict"):
            p.add_argument("text", help='one sentence, e.g. "kal subah 7 baje alarm laga do"')
        if name == "benchmark-inputs":
            p.add_argument("out", help="where to write the sentences, e.g. inputs.jsonl")
        if name == "predict-file":
            p.add_argument("inputs")
            p.add_argument("out")
        if name == "benchmark":
            p.add_argument("predictions", help="your predictions, one JSON line per sentence and form")
    args = parser.parse_args()
    if args.command == "cmi":
        import json

        from cmx.features import code_mixing

        print(json.dumps(code_mixing(args.text), indent=2))
    elif args.command == "predict":
        from cmx.harness import predict_text

        predict_text(args.text)
    elif args.command == "benchmark-inputs":
        from cmx.harness import write_inputs

        write_inputs(args.out)
    elif args.command == "predict-file":
        from cmx.harness import predict_file

        predict_file(args.inputs, args.out)
    elif args.command == "benchmark":
        from cmx.harness import benchmark

        benchmark(args.predictions)
    elif args.command == "train":
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
    elif args.command == "export":
        from cmx.export import run

        run()
    elif args.command == "transfer":
        from cmx.transfer import run

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
