"""Explicit opt-in guard for the retired semantic-control workflow."""

from __future__ import annotations

import argparse


LEGACY_SEMANTIC_MESSAGE = (
    "the semantic segmentation workflow is retired from the default path and "
    "must not be mixed with YOLO caches or reports; pass "
    "--allow-legacy-semantic only for an intentional control run"
)


def add_legacy_semantic_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-legacy-semantic",
        action="store_true",
        help="explicitly opt into the retired semantic-control workflow",
    )


def require_legacy_semantic_opt_in(enabled: bool) -> None:
    if not enabled:
        raise SystemExit(LEGACY_SEMANTIC_MESSAGE)
