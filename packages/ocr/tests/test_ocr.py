from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.ocr import process_one
import src.ocr as ocr_mod


ASSETS = Path("tests/assets")


def _ensure_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    img_path = ASSETS / "text.png"
    if not img_path.exists():
        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Hello OCR", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imwrite(str(img_path), img)


def test_process_one_blank_page_skipped(tmp_path: Path):
    cfg = load_config("config.yaml")
    cfg["behavior"]["skip_blank_pages"] = True

    module1 = {
        "request_id": "r1",
        "document_id": "d1",
        "status": "success",
        "payload": {"page": 1, "is_blank": True, "output_image": "does_not_matter.png"},
    }
    jp = tmp_path / "in.json"
    jp.write_text(json.dumps(module1), encoding="utf-8")

    out = process_one(jp, cfg)
    assert out["status"] == "success"
    assert out["payload"]["text"] == ""
    assert out["payload"]["blocks"] == []
    assert out["payload"]["meta"]["skipped"] == "blank"


def test_process_one_qa_flags_thresholds(tmp_path: Path, monkeypatch):
    _ensure_assets()
    img_path = (ASSETS / "text.png").resolve()

    cfg = load_config("config.yaml")
    cfg["behavior"]["skip_blank_pages"] = False
    cfg["behavior"]["detector"] = "craft"
    cfg["behavior"]["confidence_threshold_yellow"] = 97
    cfg["behavior"]["confidence_threshold_red"] = 90
    cfg["vintern"]["device"] = "cpu"
    cfg["vintern"]["model_name"] = "dummy"
    cfg["vintern"]["trust_remote_code"] = False
    cfg["vintern"]["max_new_tokens"] = 16
    cfg["vintern"]["temperature"] = 0.0
    cfg["vintern"]["prompt"] = "p"

    module1 = {
        "request_id": "r2",
        "document_id": "d2",
        "status": "success",
        "payload": {"page": 1, "output_image": str(img_path)},
    }
    jp = tmp_path / "in.json"
    jp.write_text(json.dumps(module1), encoding="utf-8")

    monkeypatch.setattr(
        ocr_mod,
        "_get_bounding_boxes_with_meta",
        lambda _img, device="auto": (
            [[10, 10, 200, 60], [10, 80, 200, 140], [10, 160, 200, 220]],
            {"method": "craft_poly"},
        ),
    )

    confidences = [92.0, 96.0, 98.0]
    idx = {"i": 0}

    def _fake_run_vintern(_img, **_kwargs):
        i = idx["i"]
        idx["i"] += 1
        return type("R", (), {"text": f"t{i}", "meta": {"overall_confidence": confidences[i]}})()

    monkeypatch.setattr(ocr_mod, "run_vintern", _fake_run_vintern)

    out = process_one(jp, cfg)

    qa = out["payload"]["meta"]["qa_flags"]
    assert qa["flagged_blocks"] == 2
    assert qa["flagged_level_counts"]["yellow"] == 2
    assert qa["flagged_level_counts"]["red"] == 0
