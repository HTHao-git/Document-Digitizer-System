from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import src.ocr as ocr_mod


def test_qa_flags_thresholds_accept_ratio(monkeypatch, tmp_path: Path):
    module1 = {
        "request_id": "r1",
        "document_id": "d1",
        "status": "success",
        "payload": {"page": 1, "output_image": "dummy.png"},
    }
    jp = tmp_path / "in.json"
    jp.write_text(json.dumps(module1), encoding="utf-8")

    cfg = {
        "engine": {"name": "vintern", "language": "vie"},
        "vintern": {"device": "cpu", "prompt": "p", "model_name": "m", "trust_remote_code": False, "max_new_tokens": 16, "temperature": 0.0},
        "behavior": {
            "skip_blank_pages": False,
            "detector": "craft",
            "bbox_pad_x_ratio": 0.0,
            "bbox_pad_y_ratio": 0.0,
            "bbox_pad_px": 0,
            "min_ink_ratio": 0.0,
            "vintern_crop_max_num": 1,
            "vintern_crop_max_new_tokens": 16,
            "confidence_threshold_yellow": 0.97,
            "confidence_threshold_red": 0.90,
        },
    }

    img = np.zeros((400, 600, 3), dtype=np.uint8)
    monkeypatch.setattr(ocr_mod, "_read_image", lambda _p: img)
    monkeypatch.setattr(
        ocr_mod,
        "_get_bounding_boxes_with_meta",
        lambda _img, device="auto": (
            [[55, 33, 406, 94], [42, 77, 584, 148], [9, 145, 599, 367], [20, 319, 89, 364]],
            {"method": "craft_poly"},
        ),
    )

    confidences = [82.26149784898243, 96.4819950145695, 98.59033791213955, 94.92372198065077]
    texts = ["t1", "t2", "t3", "t4"]
    idx = {"i": 0}

    def _fake_run_vintern(_img, **_kwargs):
        i = idx["i"]
        idx["i"] += 1
        return type(
            "R",
            (),
            {"text": texts[i], "meta": {"overall_confidence": confidences[i], "model_name": "m", "device": "cpu"}},
        )()

    monkeypatch.setattr(ocr_mod, "run_vintern", _fake_run_vintern)

    out = ocr_mod.process_one(jp, cfg)

    qa = out["payload"]["meta"]["qa_flags"]
    assert qa["flagged_blocks"] == 3
    assert qa["flagged_level_counts"]["yellow"] == 2
    assert qa["flagged_level_counts"]["red"] == 1
    assert qa["page_flag"]["level"] == "yellow"
    
