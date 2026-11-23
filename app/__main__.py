"""Command-line entrypoint: ``python -m app <command>``.

``serve`` (the default) runs the HTTP service; ``download`` fills the data
directory with training data. In the container this is wired as
``ENTRYPOINT ["python", "-m", "app"]`` with a default ``CMD`` of ``serve``, so
``docker run <image>`` serves and ``docker run -v ./data:/data <image> download``
downloads.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_data_dir


def _serve() -> None:
    import uvicorn

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
    from .evaluate import evaluate_data_dir, format_report

    data_dir = load_data_dir()
    print(f"Evaluating the extractor on {data_dir} ...\n")
    result = evaluate_data_dir(data_dir)
    print(format_report(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="receipt-scan")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Run the HTTP service (default).")
    sub.add_parser("download", help="Fill the data directory with training data.")
    sub.add_parser("evaluate", help="Score the current extractor on the data directory.")
    args = parser.parse_args(argv)

    command = args.command or "serve"
    if command == "download":
        _download()
    elif command == "evaluate":
        _evaluate()
    else:
        _serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
