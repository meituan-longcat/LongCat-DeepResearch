from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from . import runtime
from .backend import reset_default_backend
from .pipeline import LongCatDeepResearch


def _read_question(args: argparse.Namespace) -> str:
    if args.question is not None:
        return args.question
    if args.question_file is not None:
        return Path(args.question_file).expanduser().read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("provide --question, --question-file, or stdin")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a citation-backed research report"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--question", "-q")
    source.add_argument("--question-file")
    parser.add_argument("--run-root", default=str(runtime.DEFAULT_RUN_ROOT))
    parser.add_argument("--model", default=runtime.MODEL)
    parser.add_argument(
        "--backend-module",
        default=os.getenv("DR_BACKEND_MODULE", "local_backend"),
        help="Python module exporting create_backend() or BACKEND",
    )
    args = parser.parse_args(argv)

    os.environ["DR_BACKEND_MODULE"] = args.backend_module
    runtime.MODEL = args.model
    reset_default_backend()
    try:
        result = LongCatDeepResearch().run(
            _read_question(args),
            run_root=args.run_root,
        )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result.final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
