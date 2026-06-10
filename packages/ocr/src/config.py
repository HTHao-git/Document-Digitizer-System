from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml
from jsonschema import ValidationError, validate


def load_config(config_path: Union[str, Path] = "config.yaml", schema_path: Union[str, Path] = "schemas/config.schema.json") -> dict:
    current_dir = Path(__file__).resolve().parent
    package_root = current_dir.parent

    config_path = Path(config_path)
    if schema_path == "schemas/config.schema.json":
        schema_path = package_root / "schemas" / "config.schema.json"
    else:
        schema_path = Path(schema_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        validate(instance=cfg, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Config schema invalid: {e.message}")

    return cfg