#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import yaml


def pick_latest(paths: list[str], expect: str) -> str:
    if expect == "dir":
        paths = [path for path in paths if Path(path).is_dir()]
    elif expect == "file":
        paths = [path for path in paths if Path(path).is_file()]

    if not paths:
        raise SystemExit("No matching paths found.")

    def sort_key(path: str) -> tuple[int, str]:
        match = re.search(r"(\d+)(?!.*\d)", Path(path).name)
        numeric = int(match.group(1)) if match else -1
        return numeric, path

    return sorted(paths, key=sort_key)[-1]


def update_config(config_path: Path, dotted_key: str, value: str | None) -> None:
    data = yaml.safe_load(config_path.read_text())
    target = data
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update a value in config/ates/default.yaml")
    parser.add_argument("--config", default="config/ates/default.yaml")
    parser.add_argument("--key", required=True)
    parser.add_argument("--value")
    parser.add_argument("--glob")
    parser.add_argument("--expect", choices=["any", "file", "dir"], default="any")
    args = parser.parse_args()

    if bool(args.value) == bool(args.glob):
        raise SystemExit("Provide exactly one of --value or --glob.")

    resolved_value = args.value
    if args.glob:
        resolved_value = pick_latest(glob.glob(args.glob, recursive=True), args.expect)

    update_config(Path(args.config), args.key, resolved_value)
    print(f"{args.key} -> {resolved_value}")


if __name__ == "__main__":
    main()
