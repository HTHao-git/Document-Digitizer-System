from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from jsonschema import ValidationError, validate


def validate_output(result: dict, schema_path: Union[str, Path] = "schemas/ocr.schema.json") -> None:
    schema_path = Path(schema_path)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        validate(instance=result, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Output JSON invalid: {e.message}")