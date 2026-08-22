#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from Cython import Tempita as tempita


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "template",
        type=Path,
        help="path to a template cython module",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="path to a json configuration file containing templated substitutions"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        help="path to the output file",
    )
    args = parser.parse_args()

    with args.config.open("r") as f:
        template_subs = json.load(f)

    for k, v in list(template_subs.items()):
        # tuples are serialized as lists
        if "VERSION" in k and isinstance(v, list):
            template_subs[k] = tuple(v)

    assert args.template.suffix == ".template"
    args.output_file.write_text(
        tempita.sub(args.template.read_text(), **template_subs),
        encoding="utf-8",
    )
    print(f"wrote {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
