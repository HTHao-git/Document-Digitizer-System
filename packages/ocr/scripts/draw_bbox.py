import json
import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _resolve_image_path(img_path_str: str, *, json_path: Path) -> Path:
    img_path = Path(img_path_str)
    if img_path.is_absolute():
        return img_path

    cwd_img = (Path.cwd() / img_path_str).resolve()
    if cwd_img.exists():
        return cwd_img

    return (json_path.parent / img_path_str).resolve()


def _draw_one(json_path: Path, *, image_override: Optional[str], out_path: Path, draw_text: bool, max_label_len: int):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "success":
        print(f"Skip: {json_path.name} status='{data.get('status')}'")
        return

    payload = data.get("payload", {})
    blocks = payload.get("blocks", [])

    img_path_str = image_override or payload.get("input_image")
    if not img_path_str:
        print(f"Skip: {json_path.name} missing payload.input_image (and no --image)")
        return

    img_path = _resolve_image_path(str(img_path_str), json_path=json_path)
    if not img_path.exists():
        print(f"Skip: {json_path.name} image not found: {img_path}")
        return

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Skip: {json_path.name} cannot read image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for b in blocks:
        bbox = b.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        if draw_text:
            text = str(b.get("text", "") or "").strip()
            conf = b.get("confidence", None)
            label = text
            if isinstance(conf, (int, float)):
                label = f"{text} ({float(conf):.1f}%)"
            if max_label_len > 0 and len(label) > max_label_len:
                label = label[: max(0, max_label_len - 1)] + "…"
            if label:
                draw.text((x1, max(0, y1 - 15)), label, fill="blue")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_path))
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw OCR Bounding Boxes on Image")
    parser.add_argument("--json", required=True, help="Path to OCR output JSON")
    parser.add_argument("--image", required=False, help="Path to input image (if different from JSON payload)")
    parser.add_argument("--output", required=True, help="Path to save the visualized image")
    parser.add_argument("--no-text", action="store_true", help="Draw boxes only (no text/confidence label)")
    parser.add_argument("--max-label-len", type=int, default=120, help="Truncate long labels to this length (0 = no limit)")
    args = parser.parse_args()

    json_path = Path(args.json)
    out_path = Path(args.output)
    draw_text = not bool(args.no_text)
    max_label_len = int(args.max_label_len)

    if json_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        for jp in sorted(json_path.glob("*.json")):
            op = out_path / (jp.stem + ".png")
            _draw_one(jp, image_override=args.image, out_path=op, draw_text=draw_text, max_label_len=max_label_len)
        return

    _draw_one(json_path, image_override=args.image, out_path=out_path, draw_text=draw_text, max_label_len=max_label_len)

if __name__ == "__main__":
    main()
