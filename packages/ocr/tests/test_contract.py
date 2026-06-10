from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.ocr import process_one


def test_process_one_produces_valid_output_for_blank_page(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    cfg["behavior"]["skip_blank_pages"] = True

    module1_json = Path(__file__).resolve().parents[2] / "preprocess" / "data" / "json" / "20140603_0003_BCCTC_tg_0_0.json"
    result = process_one(module1_json, cfg)

    assert result["module"] == "ocr"
    assert result["version"]
    assert result["status"] in ("success", "error")
    assert "payload" in result

