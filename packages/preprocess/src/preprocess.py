deep_mergeSCHEMA_VERSION = "1.0.0"

from pathlib import Path
import json, uuid
from datetime import datetime
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import yaml
from jsonschema import validate, ValidationError
import logging
import time

from .utils import resize_to_width, four_point_transform, count_black_ratio

from importlib.metadata import version as pkg_version

REQUIRED_KEYS = [
    "io", "resize", "denoise", "crop", "deskew",
    "normalize", "binarize", "morphology", "blank_detect", "osd"
]

DEFAULT_CONFIG = {
    "io": {"input_dir": "./data/input", "output_dir": "./data/output", "json_dir": "./data/json"},
    "resize": {"target_width": 2000},
    "denoise": {"method": "fastNlMeans", "h": 10},
    "crop": {"enable": True, "min_area_ratio": 0.20, "min_w_ratio": 0.40, "min_h_ratio": 0.40, "padding": 20},
    "deskew": {"enable": True},
    "normalize": {"method": "clahe", "clip_limit": 2.0, "tile_grid": [8, 8]},
    "binarize": {"method": "adaptive", "block_size": 31, "C": 15},
    "morphology": {"kernel": [3, 3]},
    "blank_detect": {"threshold": 0.005},
    "osd": {"enable": True, "min_width": 600, "min_height": 200, "min_black_ratio": 0.01}
}

def deep_merge(defaults, user_cfg):
    for k, v in user_cfg.items():
        if isinstance(v, dict) and k in defaults:
            defaults[k] = deep_merge(defaults[k], v)
        else:
            defaults[k] = v
    return defaults

def get_version():
    try:
        return pkg_version("module1-preprocess")
    except Exception:
        return "1.0.0"

def validate_config(cfg):
    for k in REQUIRED_KEYS:
        if k not in cfg:
            raise ValueError(f"Missing config section: '{k}'")

    if "target_width" not in cfg["resize"]:
        raise ValueError("Missing resize.target_width in config.yaml")
    if "threshold" not in cfg["blank_detect"]:
        raise ValueError("Missing blank_detect.threshold in config.yaml")
    if "min_width" not in cfg["osd"] or "min_height" not in cfg["osd"]:
        raise ValueError("Missing osd.min_width/min_height in config.yaml")

def load_config(config_path="config.yaml", schema_path="schemas/config.schema.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        validate(instance=cfg, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Config schema invalid: {e.message}")

    return cfg

def validate_output(result, schema_path="schemas/preprocess.schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        validate(instance=result, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Output JSON invalid: {e.message}")

def setup_logger(log_path=None):
    logger = logging.getLogger("module1")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file (optional)
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def verify_rotation_by_ocr(img_0, img_rotated):
    def get_ocr_score(image):
        h, w = image.shape[:2]

        crop = image[h//4:3*h//4, w//4:3*w//4]

        data = pytesseract.image_to_data(crop, output_type=Output.DICT, config="--psm 11 --dpi 300")

        conf_scores = [int(conf) for conf in data['conf'] if int(conf) > 10]
        return sum(conf_scores), len(conf_scores)

    score0, count0 = get_ocr_score(img_0)
    scoreRot, countRot = get_ocr_score(img_rotated)

    if countRot > count0 or scoreRot > score0:
        return True
    
    return False

def get_osd_info(img, min_conf=2.0):
    res = {"success": False, "rotate_deg": 0, "conf": 0, "message": ""}
    tess_config = '--psm 0 --dpi 300'

    # Lần 1: Thử tiêu chuẩn
    try:
        data = pytesseract.image_to_osd(img, config=tess_config, output_type=Output.DICT)
        res.update({"success": True, "rotate_deg": data["rotate"], "conf": data["orientation_conf"]})
    except Exception:
        # Lần 2: Phóng to
        try:
            h, w = img.shape[:2]
            upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            # Nhị phân hóa tạm thời để nổi bật chữ
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            data = pytesseract.image_to_osd(gray, config=tess_config, output_type=Output.DICT)
            res.update({"success": True, "rotate_deg": data["rotate"], "conf": data["orientation_conf"], "message": "Double-Pass Success"})
        except Exception as e:
            res["message"] = f"OSD Failed: {str(e)}"
    
    return res

def apply_preprocessing(img, cfg):
    meta = {
        "original_size": img.shape[:2],
        "rotation_detected": 0,
        "deskew_angle": 0.0,
        "crop_applied": False,
        "is_blank": False,
        "black_ratio": 0.0,
        "osd": {}
    }

    # Bước A: Resize ban đầu
    work_img = resize_to_width(img, cfg['resize']['target_width'])

    # Bước B: Xoay hướng (OSD + Trọng tài OCR)
    osd = get_osd_info(work_img, cfg["osd"].get("min_confidence", 2.0))
    meta["osd"] = osd
    
    if osd["success"] and osd["rotate_deg"] != 0:
        rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        temp_rot = cv2.rotate(work_img, rot_map[osd["rotate_deg"]])
        
        # Chỉ xoay nếu tin cậy cao HOẶC được Trọng tài OCR thông qua
        if osd["conf"] >= cfg["osd"].get("min_confidence", 2.0) or verify_rotation_by_ocr(work_img, temp_rot):
            work_img = temp_rot
            meta["rotation_detected"] = osd["rotate_deg"]
        else:
            osd["message"] += " | Rotation rejected by OCR arbitrator"

    # Bước C: Cắt biên (Document Crop)
    if cfg["crop"]["enable"]:
        # Tận dụng code từ sent.txt của bạn
        gray_temp = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.GaussianBlur(gray_temp, (5, 5), 0), 75, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                area_ratio = cv2.contourArea(c) / (work_img.shape[0] * work_img.shape[1])
                if area_ratio > cfg["crop"]["min_area_ratio"]:
                    work_img = four_point_transform(work_img, approx.reshape(4, 2))
                    meta["crop_applied"] = True
                    break

    # Bước D: Chỉnh nghiêng (Deskew dùng Hough Lines - Tránh lỗi Hình 05)
    gray_ds = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_ds, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        angle = np.median(angles)
        if abs(angle) < 45: # Chỉ xoay nếu là góc nghiêng văn bản
            (h, w) = work_img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            work_img = cv2.warpAffine(work_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            meta["deskew_angle"] = angle

# Bước E: Tạo thành phẩm
    clean_gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clean_img = clahe.apply(clean_gray)

    bin_img = cv2.adaptiveThreshold(cv2.medianBlur(clean_img, 3), 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                                    cfg["binarize"]["block_size"], cfg["binarize"]["C"])

    # Tính toán tỷ lệ đen và kiểm tra Blank
    ratio = count_black_ratio(bin_img)
    meta["black_ratio"] = float(ratio) # Lưu vào meta
    meta["is_blank"] = ratio < cfg["blank_detect"]["threshold"]

    return bin_img, clean_img, meta

def process_folder(input_dir: Path, output_dir: Path, json_dir: Path, config_path="config.yaml", log_path=None):
    cfg = load_config(config_path)
    logger = setup_logger(log_path)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for idx, p in enumerate(sorted(input_dir.iterdir()), start=1):
        if p.suffix.lower() not in exts:
            continue

        start = time.time()

        try:
            img = cv2.imread(str(p))
            if img is None: continue
            # SỬA THỨ TỰ NHẬN: bin_img trước, clean_img sau
            bin_img, clean_img, meta = apply_preprocessing(img, cfg)

            out_clean = output_dir / f"{p.stem}_clean.png"
            out_bin = output_dir / f"{p.stem}_bin.png"
            cv2.imwrite(str(out_clean), clean_img)
            cv2.imwrite(str(out_bin), bin_img)

            result = {
                "request_id": str(uuid.uuid4()),
                "document_id": f"local_doc_{idx}",
                "module": "preprocess",
                "version": get_version(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "success",
                "error": None,
                "payload": {
                    "page": idx,
                    "input_image": str(p),
                    "output_image": str(out_clean),
                    "width": int(clean_img.shape[1]),
                    "height": int(clean_img.shape[0]),
                    "rotation": meta["rotation_detected"],
                    "is_blank": bool(meta["is_blank"]),
                    "blank_ratio": meta.get("black_ratio", 0.0),
                    "crop_applied": meta["crop_applied"],
                    "deskew_angle": meta["deskew_angle"],
                    "osd": meta["osd"]
                }
            }

            validate_output(result)
            status = "SUCCESS"

        except Exception as e:
            result = {
                "request_id": str(uuid.uuid4()),
                "document_id": f"local_doc_{idx}",
                "module": "preprocess",
                "version": get_version(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "error",
                "error": {
                    "code": "PREPROCESS_ERROR",
                    "message": str(e)
                },
                "payload": {
                    "page": idx,
                    "input_image": str(p),
                    "output_image": "",
                    "width": 0,
                    "height": 0,
                    "rotation": 0,
                    "is_blank": False
                }
            }
            status = "ERROR"

        json_path = json_dir / f"{p.stem}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        elapsed = time.time() - start
        if status == "SUCCESS":
            logger.info(f"{p.name} | time={elapsed:.3f}s | blank={result['payload']['is_blank']} | rot={result['payload']['rotation']} | deskew={result['payload']['deskew_angle']}")
        else:
            logger.error(f"{p.name} | time={elapsed:.3f}s | error={result['error']['message']}")