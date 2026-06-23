#!/usr/bin/env python
"""Thin CLI wrapper: build JSONL datasets. Logic lives in clinical_scribe."""

import sys

from clinical_scribe.cli import prepare_data_main

if __name__ == "__main__":
    raise SystemExit(prepare_data_main(sys.argv[1:]))
