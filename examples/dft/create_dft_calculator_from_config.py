#!/usr/bin/env python
"""Create a DFT ASE calculator from YAML without running a calculation."""

from __future__ import annotations

import argparse

from ase_calculator_kit import get_calculator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calculator", choices=["vasp", "qe", "espresso", "quantum-espresso"])
    parser.add_argument("config")
    parser.add_argument("--directory", help="override the working directory")
    parser.add_argument(
        "--write-resolved-config",
        action="store_true",
        help="write resolved_calculator_config.yaml into the working directory",
    )
    args = parser.parse_args()

    overrides = {"directory": args.directory} if args.directory else None
    calc = get_calculator(
        args.calculator,
        config=args.config,
        overrides=overrides,
        write_resolved_config=args.write_resolved_config,
    )
    print(f"Created {type(calc).__module__}.{type(calc).__name__}")


if __name__ == "__main__":
    main()
