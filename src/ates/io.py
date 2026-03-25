from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)


def dedupe_image_dicts(image_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[int | str] = set()
    unique_dicts: list[dict[str, Any]] = []
    for image_dict in image_dicts:
        image_id = image_dict["id"]
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        unique_dicts.append(image_dict)
    return unique_dicts

