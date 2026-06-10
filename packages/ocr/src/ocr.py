from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import cv2
import numpy as np

from .engines.vintern_engine import run_vintern
from .schema import validate_output

SCHEMA_VERSION = "1.0.0"

_CRAFT_MODELS: Dict[Tuple[bool, str], Any] = {}


def _clean_ocr_text(text: str, prompt: str) -> str:
    s = unicodedata.normalize("NFC", str(text or "")).strip()
    if not s:
        return ""

    prompt_s = unicodedata.normalize("NFC", str(prompt or ""))
    prompt_lines = [ln.strip() for ln in prompt_s.splitlines() if ln.strip()]

    out_lines: List[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t in prompt_lines:
            continue
        if prompt_lines:
            best = 0.0
            for p in prompt_lines:
                r = SequenceMatcher(a=t.casefold(), b=p.casefold()).ratio()
                if r > best:
                    best = r
            if best >= 0.88:
                continue
        t = re.sub(r"\s+", " ", t)
        out_lines.append(t)

    return "\n".join(out_lines).strip()


def _is_valid_text(s: str) -> bool:
    t = unicodedata.normalize("NFC", str(s or "")).strip()
    if len(t) < 1:
        return False
    if not any(ch.isalnum() for ch in t):
        return False
    return True


def _ink_ratio(img_bgr) -> float:
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mean = float(np.mean(gray))
    if mean < 127:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = thr > 0
    else:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg = thr > 0
    return float(np.mean(fg))


def _get_craft_model(device: str = "auto", *, crop_type: str = "poly"):
    global _CRAFT_MODELS

    try:
        import torchvision.models.vgg as tv_vgg

        urls = getattr(tv_vgg, "model_urls", None)
        if not isinstance(urls, dict):
            urls = {}
            tv_vgg.model_urls = urls
        urls.setdefault("vgg16_bn", "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")
        urls.setdefault("vgg16", "https://download.pytorch.org/models/vgg16-397923af.pth")
    except Exception:
        pass

    try:
        from craft_text_detector import Craft
    except ImportError:
        raise RuntimeError("Missing craft-text-detector. Install with: pip install craft-text-detector")

    use_cuda = False
    if device in ("auto", "cuda"):
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            pass

    key = (use_cuda, str(crop_type))
    m = _CRAFT_MODELS.get(key)
    if m is None:
        m = Craft(output_dir=None, crop_type=str(crop_type), cuda=use_cuda)
        _CRAFT_MODELS[key] = m
    return m


def _get_bounding_boxes(img_bgr, device="auto") -> List[List[int]]:
    boxes, _ = _get_bounding_boxes_with_meta(img_bgr, device=device)
    return boxes


def _get_bounding_boxes_with_meta(img_bgr, device="auto") -> Tuple[List[List[int]], Dict[str, Any]]:
    craft_error = None
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return [], {"method": "empty_image"}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _is_inhomogeneous_shape_error(exc: Exception) -> bool:
        msg = str(exc)
        return "inhomogeneous shape" in msg or "setting an array element with a sequence" in msg

    def _is_empty_boxes(v: Any) -> bool:
        if v is None:
            return True
        try:
            return len(v) == 0
        except Exception:
            return False

    try:
        craft = _get_craft_model(device, crop_type="poly")
        try:
            prediction_result = craft.detect_text(img_rgb)
            boxes = prediction_result.get("boxes", [])
            method = "craft_poly"
        except ValueError as e:
            if _is_inhomogeneous_shape_error(e):
                craft = _get_craft_model(device, crop_type="box")
                prediction_result = craft.detect_text(img_rgb)
                boxes = prediction_result.get("boxes", [])
                method = "craft_box"
            else:
                raise
    except Exception as e:
        craft_error = str(e)
        boxes = []
        method = None

    if _is_empty_boxes(boxes):
        cv_boxes = _get_bounding_boxes_cv(img_bgr)
        if cv_boxes:
            meta = {"method": "cv_fallback"}
            if craft_error:
                meta["craft_error"] = craft_error
            return cv_boxes, meta
        meta = {"method": "no_boxes"}
        if method:
            meta["craft_method"] = method
        if craft_error:
            meta["craft_error"] = craft_error
        return [], meta

    raw_boxes: List[List[int]] = []
    for box in boxes:
        xs = box[:, 0]
        ys = box[:, 1]
        x1 = max(0, int(math.floor(np.min(xs))))
        y1 = max(0, int(math.floor(np.min(ys))))
        x2 = min(img_bgr.shape[1], int(math.ceil(np.max(xs))))
        y2 = min(img_bgr.shape[0], int(math.ceil(np.max(ys))))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        raw_boxes.append([x1, y1, x2, y2])

    if not raw_boxes:
        meta = {"method": method or "craft_empty"}
        if craft_error:
            meta["craft_error"] = craft_error
        return [], meta

    raw_boxes.sort(key=lambda b: b[1])

    lines: List[List[List[int]]] = []
    current_line = [raw_boxes[0]]
    for box in raw_boxes[1:]:
        _, y1, _, _ = box
        _, prev_y1, _, _ = current_line[-1]
        h_avg = sum(b[3] - b[1] for b in current_line) / len(current_line)
        if abs(y1 - prev_y1) < h_avg * 0.5:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    if current_line:
        lines.append(current_line)

    result_boxes: List[List[int]] = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        min_x = min(b[0] for b in line)
        min_y = min(b[1] for b in line)
        max_x = max(b[2] for b in line)
        max_y = max(b[3] for b in line)
        result_boxes.append([min_x, min_y, max_x, max_y])

    meta = {"method": method}
    if craft_error:
        meta["craft_error"] = craft_error
    return result_boxes, meta


def _get_bounding_boxes_cv(img_bgr) -> List[List[int]]:
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return []
    h, w = img_bgr.shape[:2]
    if h < 10 or w < 10:
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mean = float(np.mean(gray))

    if mean < 127:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def _extract(thr_img) -> List[List[int]]:
        kh = max(30, int(w * 0.08))
        kv = max(30, int(h * 0.08))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
        horizontal = cv2.morphologyEx(thr_img, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(thr_img, cv2.MORPH_OPEN, v_kernel, iterations=1)
        grid = cv2.bitwise_or(horizontal, vertical)
        cleaned = cv2.bitwise_and(thr_img, cv2.bitwise_not(grid))

        kx = max(20, int(w * 0.06))
        ky = max(5, int(h * 0.012))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        dil = cv2.dilate(cleaned, kernel, iterations=1)

        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes: List[List[int]] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 20 or ch < 12:
                continue
            if cw * ch > int(w * h * 0.92):
                continue
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + cw))
            y2 = min(h, int(y + ch))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            raw_boxes.append([x1, y1, x2, y2])
        if not raw_boxes:
            return []

        raw_boxes.sort(key=lambda b: b[1])
        lines: List[List[List[int]]] = []
        current_line = [raw_boxes[0]]
        for box in raw_boxes[1:]:
            _, y1, _, _ = box
            _, prev_y1, _, _ = current_line[-1]
            h_avg = sum(b[3] - b[1] for b in current_line) / len(current_line)
            if abs(y1 - prev_y1) < h_avg * 0.8:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        if current_line:
            lines.append(current_line)

        result_boxes: List[List[int]] = []
        for line in lines:
            min_x = min(b[0] for b in line)
            min_y = min(b[1] for b in line)
            max_x = max(b[2] for b in line)
            max_y = max(b[3] for b in line)
            result_boxes.append([min_x, min_y, max_x, max_y])
        return result_boxes

    boxes = _extract(thr)
    if boxes:
        return boxes

    blk = max(25, int(min(h, w) / 20) | 1)
    if mean < 127:
        thr2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, -5)
    else:
        thr2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 5)
    return _extract(thr2)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_error(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def _resolve_image_path(image_path: str, module1_json_path: Path) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p

    candidates: List[Path] = []
    candidates.append(module1_json_path.parent / p)
    for i in range(1, min(6, len(module1_json_path.parents))):
        candidates.append(module1_json_path.parents[i] / p)

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def process_one(module1_json_path: Path, cfg: dict) -> Dict[str, Any]:
    module1 = _load_json(module1_json_path)

    request_id = str(module1.get("request_id") or uuid.uuid4())
    document_id = str(module1.get("document_id") or "")
    page = int((module1.get("payload") or {}).get("page") or 0)

    status = str(module1.get("status") or "error")
    if status != "success":
        result = {
            "request_id": request_id,
            "document_id": document_id,
            "module": "ocr",
            "version": SCHEMA_VERSION,
            "timestamp": _utc_now_iso(),
            "status": "error",
            "error": _make_error("UPSTREAM_ERROR", "Module1 status != success"),
            "payload": {
                "page": page,
                "input_image": "",
                "engine": str(cfg["engine"]["name"]),
                "language": str(cfg["engine"]["language"]),
                "text": "",
                "blocks": [],
                "meta": {"upstream_json": str(module1_json_path)},
            },
        }
        validate_output(result)
        return result

    payload = module1.get("payload") or {}
    is_blank = bool(payload.get("is_blank"))
    input_image = str(payload.get("output_image") or payload.get("input_image") or "")

    if cfg["behavior"].get("skip_blank_pages", True) and is_blank:
        result = {
            "request_id": request_id,
            "document_id": document_id,
            "module": "ocr",
            "version": SCHEMA_VERSION,
            "timestamp": _utc_now_iso(),
            "status": "success",
            "error": None,
            "payload": {
                "page": page,
                "input_image": input_image,
                "engine": str(cfg["engine"]["name"]),
                "language": str(cfg["engine"]["language"]),
                "text": "",
                "blocks": [],
                "meta": {"skipped": "blank"},
            },
        }
        validate_output(result)
        return result

    if not input_image:
        result = {
            "request_id": request_id,
            "document_id": document_id,
            "module": "ocr",
            "version": SCHEMA_VERSION,
            "timestamp": _utc_now_iso(),
            "status": "error",
            "error": _make_error("MISSING_INPUT", "Missing payload.output_image"),
            "payload": {
                "page": page,
                "input_image": "",
                "engine": str(cfg["engine"]["name"]),
                "language": str(cfg["engine"]["language"]),
                "text": "",
                "blocks": [],
                "meta": {"upstream_json": str(module1_json_path)},
            },
        }
        validate_output(result)
        return result

    img_path = _resolve_image_path(input_image, module1_json_path)
    engine = str(cfg["engine"]["name"])
    img = _read_image(img_path)
    language = str(cfg["engine"]["language"])
    try:
        if engine == "vintern":
            vcfg = cfg.get("vintern") or {}
            device = str(vcfg.get("device") or "auto")
            prompt = str(vcfg.get("prompt") or "Trích xuất văn bản trong ảnh. Chỉ trả về văn bản.")
            model_name = str(vcfg.get("model_name") or "5CD-AI/Vintern-1B-v3_5")
            trust_remote_code = bool(vcfg.get("trust_remote_code", True))
            max_new_tokens = int(vcfg.get("max_new_tokens") or 1024)
            temperature = float(vcfg.get("temperature") or 0.0)
            craft_device = device
        else:
            raise RuntimeError(f"Unsupported engine: {engine}")

        detector_name = str((cfg.get("behavior") or {}).get("detector", "craft") or "craft").strip().lower()

        pad_x_ratio = float((cfg.get("behavior") or {}).get("bbox_pad_x_ratio", 0.05))
        pad_y_ratio = float((cfg.get("behavior") or {}).get("bbox_pad_y_ratio", 0.10))
        pad_px = int((cfg.get("behavior") or {}).get("bbox_pad_px", 0) or 0)
        min_ink_ratio = float((cfg.get("behavior") or {}).get("min_ink_ratio", 0.0) or 0.0)

        thr_yellow = float((cfg.get("behavior") or {}).get("confidence_threshold_yellow", 0.0) or 0.0)
        thr_red = float((cfg.get("behavior") or {}).get("confidence_threshold_red", 0.0) or 0.0)
        if 0.0 < thr_yellow <= 1.0:
            thr_yellow = thr_yellow * 100.0
        if 0.0 < thr_red <= 1.0:
            thr_red = thr_red * 100.0
        if thr_red and thr_yellow and thr_red > thr_yellow:
            thr_red, thr_yellow = thr_yellow, thr_red

        crop_max_num = int((cfg.get("behavior") or {}).get("vintern_crop_max_num", 2) or 2)
        crop_max_new_tokens = int((cfg.get("behavior") or {}).get("vintern_crop_max_new_tokens") or max_new_tokens)

        if detector_name == "cv":
            bboxes = _get_bounding_boxes_cv(img)
            bb_info = {"method": "cv"}
        elif detector_name == "craft":
            bboxes, bb_info = _get_bounding_boxes_with_meta(img, device=craft_device)
        else:
            raise RuntimeError(f"Unsupported detector: {detector_name}. Supported: craft, cv")

        bb_meta = {"enabled": True, "count": int(len(bboxes))}
        if bb_info:
            bb_meta.update(bb_info)

        blocks: List[Dict[str, Any]] = []
        full_text_lines: List[str] = []

        if not bboxes:
            vres = run_vintern(
                img,
                prompt=prompt,
                model_name=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            text = _clean_ocr_text(vres.text, prompt)
            meta = dict(vres.meta or {})
            meta["bounding_boxes"] = bb_meta
            if thr_yellow or thr_red:
                c = meta.get("overall_confidence")
                page_flag = None
                if isinstance(c, (int, float)):
                    cf = float(c)
                    if thr_red and cf < thr_red:
                        page_flag = {"flagged": True, "level": "red", "confidence": cf}
                    elif thr_yellow and cf < thr_yellow:
                        page_flag = {"flagged": True, "level": "yellow", "confidence": cf}
                meta["qa_flags"] = {
                    "threshold_yellow": thr_yellow or None,
                    "threshold_red": thr_red or None,
                    "flagged_blocks": 0,
                    "flagged_level_counts": {"yellow": 0, "red": 0},
                    "page_flag": page_flag,
                }
        else:
            overall_meta: Optional[Dict[str, Any]] = None
            confidences: List[float] = []
            flagged_level_counts = {"yellow": 0, "red": 0}
            for box in bboxes:
                x1, y1, x2, y2 = box
                pad_x = int((x2 - x1) * pad_x_ratio) + pad_px
                pad_y = int((y2 - y1) * pad_y_ratio) + pad_px

                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(img.shape[1], x2 + pad_x)
                py2 = min(img.shape[0], y2 + pad_y)

                cropped_img = img[py1:py2, px1:px2]
                if cropped_img.size == 0:
                    continue

                if min_ink_ratio > 0.0 and _ink_ratio(cropped_img) < min_ink_ratio:
                    continue

                vres = run_vintern(
                    cropped_img,
                    prompt=prompt,
                    model_name=model_name,
                    device=device,
                    trust_remote_code=trust_remote_code,
                    max_new_tokens=crop_max_new_tokens,
                    temperature=temperature,
                    max_num=crop_max_num,
                )

                if overall_meta is None:
                    overall_meta = vres.meta

                box_text = _clean_ocr_text(vres.text, prompt)
                if box_text and _is_valid_text(box_text):
                    confidence = vres.meta.get("overall_confidence") if vres.meta else None
                    if isinstance(confidence, (int, float)):
                        confidences.append(float(confidence))
                    block: Dict[str, Any] = {
                        "level": "line",
                        "text": box_text,
                        "bbox": box,
                        "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
                    }
                    if (thr_yellow or thr_red) and isinstance(confidence, (int, float)):
                        cf = float(confidence)
                        if thr_red and cf < thr_red:
                            block["qa_flag"] = {"flagged": True, "level": "red", "threshold": thr_red}
                            flagged_level_counts["red"] += 1
                        elif thr_yellow and cf < thr_yellow:
                            block["qa_flag"] = {"flagged": True, "level": "yellow", "threshold": thr_yellow}
                            flagged_level_counts["yellow"] += 1
                    blocks.append(block)
                    full_text_lines.append(box_text)

            text = "\n".join(full_text_lines)

            meta = dict(overall_meta or {})
            meta["bounding_boxes"] = bb_meta
            page_conf = None
            if confidences:
                page_conf = sum(confidences) / len(confidences)
                meta["overall_confidence"] = page_conf
            if thr_yellow or thr_red:
                page_flag = None
                if isinstance(page_conf, (int, float)):
                    cf = float(page_conf)
                    if thr_red and cf < thr_red:
                        page_flag = {"flagged": True, "level": "red", "confidence": cf}
                    elif thr_yellow and cf < thr_yellow:
                        page_flag = {"flagged": True, "level": "yellow", "confidence": cf}
                meta["qa_flags"] = {
                    "threshold_yellow": thr_yellow or None,
                    "threshold_red": thr_red or None,
                    "flagged_blocks": int(flagged_level_counts["yellow"] + flagged_level_counts["red"]),
                    "flagged_level_counts": flagged_level_counts,
                    "page_flag": page_flag,
                }

        result = {
            "request_id": request_id,
            "document_id": document_id,
            "module": "ocr",
            "version": SCHEMA_VERSION,
            "timestamp": _utc_now_iso(),
            "status": "success",
            "error": None,
            "payload": {
                "page": page,
                "input_image": str(img_path),
                "engine": engine,
                "language": language,
                "text": text,
                "blocks": blocks,
                "meta": meta,
            },
        }
        validate_output(result)
        return result
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        result = {
            "request_id": request_id,
            "document_id": document_id,
            "module": "ocr",
            "version": SCHEMA_VERSION,
            "timestamp": _utc_now_iso(),
            "status": "error",
            "error": _make_error("OCR_FAILED", str(e)),
            "payload": {
                "page": page,
                "input_image": str(img_path),
                "engine": engine,
                "language": language,
                "text": "",
                "blocks": [],
                "meta": {"exception": str(e), "traceback": tb},
            },
        }
        validate_output(result)
        return result
