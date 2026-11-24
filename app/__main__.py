"""Command-line entrypoint: ``python -m app <command>``.

``serve`` (the default) runs the HTTP service; ``download`` fills the data
directory with training data; ``train`` fits the ML pickers and writes
``model.joblib``; ``evaluate`` scores the current extractor. In the container this
is wired as ``ENTRYPOINT ["python", "-m", "app"]`` with a default ``CMD`` of
``serve``, so ``docker run <image>`` serves and
``docker run -v ./data:/data <image> train`` trains.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import ConfigError, load_data_dir, load_settings, validate_settings


def _serve() -> None:
    import uvicorn

    try:
        validate_settings(load_settings())
    except ConfigError as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)  # noqa: S104


def _download() -> None:
    from .download import run_download

    data_dir = load_data_dir()
    print(f"Downloading datasets into {data_dir} ...")
    counts = run_download(data_dir)
    for source, count in counts.items():
        print(f"  {source}: {count} new example(s)")


def _evaluate() -> None:
    from .evaluate import evaluate_data_dir, format_report, rules_predict
    from .model import load_cached_model

    data_dir = load_data_dir()
    model = load_cached_model(data_dir)
    which = "ML pickers" if model is not None else "simple rules"
    print(f"Evaluating the extractor ({which}) on {data_dir} ...\n")
    predict = model.predict if model is not None else rules_predict
    result = evaluate_data_dir(data_dir, predict=predict)
    print(format_report(result))


def _train() -> None:
    from .train import train_model
    from .evaluate import format_report

    data_dir = load_data_dir()
    print(f"Training ML pickers on {data_dir} ...\n")
    result = train_model(data_dir)
    print(
        f"Trained on {result.train_receipts} receipt(s) "
        f"({result.store_rows} store rows, {result.total_rows} total rows), "
        f"held out {result.holdout_receipts}.\n"
    )
    print("Held-out score:")
    print(format_report(result.evaluation))
    print(f"\nWrote model to {result.model_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="receipt-scan")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Run the HTTP service (default).")
    sub.add_parser("download", help="Fill the data directory with training data.")
    sub.add_parser("train", help="Fit the ML pickers and write model.joblib.")
    sub.add_parser("evaluate", help="Score the current extractor on the data directory.")
    args = parser.parse_args(argv)

    command = args.command or "serve"
    if command == "download":
        _download()
    elif command == "train":
        _train()
    elif command == "evaluate":
        _evaluate()
    else:
        _serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
