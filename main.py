"""Command-line entry point for the City Hotel ML project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__version__ = "0.1.0"

PLANNED_COMMANDS = (
    "analyze",
    "train",
    "tune",
    "evaluate",
    "cluster",
    "predict",
)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the project command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "City Hotel booking-demand ML project CLI. "
            "Phase 0 provides infrastructure only; ML workflows are pending."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command in PLANNED_COMMANDS:
        subparser = subparsers.add_parser(
            command,
            help="planned command (implementation pending)",
            description=(
                f"The '{command}' workflow is planned but is not implemented "
                "during Phase 0."
            ),
        )
        subparser.set_defaults(handler=_pending_command)

    return parser


def _pending_command(args: argparse.Namespace) -> int:
    """Report that a reserved future command is not implemented."""
    print(
        f"The '{args.command}' command is not implemented yet; "
        "no machine-learning work was run."
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments without running unimplemented workflows."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

