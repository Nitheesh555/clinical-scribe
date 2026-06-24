"""Subcommand dispatcher: ``python -m clinical_scribe <prepare|train|evaluate|export|serve> ...``."""

from __future__ import annotations

import sys

from .cli import evaluate_main, export_main, prepare_data_main, serve_main, train_main

_COMMANDS = {
    "prepare": prepare_data_main,
    "train": train_main,
    "evaluate": evaluate_main,
    "export": export_main,
    "serve": serve_main,
}


def main(argv: list[str] | None = None) -> int:
    """Route ``argv[0]`` to the matching ``*_main`` and pass the rest through."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _COMMANDS:
        sys.stderr.write(f"usage: python -m clinical_scribe {{{'|'.join(_COMMANDS)}}} [options]\n")
        return 2
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
