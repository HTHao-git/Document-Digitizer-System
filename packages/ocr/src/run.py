from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

from .config import load_config
from .logger import setup_logger
from .ocr import process_one


def _iter_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted([p for p in input_dir.glob("*.json") if p.is_file()])

def _setup_runtime() -> None:
    tc = os.environ.get("TRANSFORMERS_CACHE")
    if tc and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = tc
    if "TRANSFORMERS_CACHE" in os.environ:
        os.environ.pop("TRANSFORMERS_CACHE", None)

    warnings.filterwarnings(
        "ignore",
        message=r"The parameter 'pretrained' is deprecated.*",
        category=UserWarning,
        module=r"torchvision\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Arguments other than a weight enum.*",
        category=UserWarning,
        module=r"torchvision\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Using `TRANSFORMERS_CACHE` is deprecated.*",
        category=FutureWarning,
        module=r"transformers\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
        module=r"timm\..*",
    )


def main() -> None:
    _setup_runtime()
    parser = argparse.ArgumentParser(description="Module 2 - OCR")
    parser.add_argument("--input-json", required=False, default=None, help="Folder containing Module 1 JSON outputs")
    parser.add_argument("--output-json", required=False, default=None, help="Folder to write Module 2 JSON outputs")
    parser.add_argument("--config", required=False, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", required=False, default=None, help="Optional log file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("module2", args.log)

    input_dir = Path(args.input_json or cfg["io"]["input_json_dir"])
    
    output_dir = Path(args.output_json or cfg["io"]["output_json_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_json_files(input_dir)
    if not files:
        logger.info(f"No JSON files found in {input_dir}")
        return

    ok = 0
    err = 0
    confidences: list[float] = []
    flagged_pages = 0
    flagged_blocks = 0
    flagged_level_counts = {"yellow": 0, "red": 0}
    for p in files:
        try:
            result = process_one(p, cfg)
            out_path = output_dir / p.name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if result.get("status") == "success":
                ok += 1
                payload = result.get("payload") or {}
                meta = payload.get("meta") or {}
                c = meta.get("overall_confidence")
                if isinstance(c, (int, float)):
                    confidences.append(float(c))
                qa = meta.get("qa_flags") or {}
                if isinstance(qa, dict):
                    pb = qa.get("flagged_blocks")
                    if isinstance(pb, (int, float)):
                        flagged_blocks += int(pb)
                    plc = qa.get("flagged_level_counts")
                    if isinstance(plc, dict):
                        for k in ("yellow", "red"):
                            v = plc.get(k)
                            if isinstance(v, (int, float)):
                                flagged_level_counts[k] += int(v)
                    page_flag = qa.get("page_flag")
                    if isinstance(page_flag, dict) and page_flag.get("flagged") is True:
                        flagged_pages += 1
                    elif isinstance(pb, (int, float)) and int(pb) > 0:
                        flagged_pages += 1
            else:
                err += 1
        except Exception as e:
            err += 1
            logger.info(f"Failed {p.name}: {e}")

    avg_conf = (sum(confidences) / len(confidences)) if confidences else None
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "success": ok,
        "error": err,
        "pages_with_confidence": len(confidences),
        "average_overall_confidence": avg_conf,
        "qa_flagged_pages": flagged_pages,
        "qa_flagged_blocks": flagged_blocks,
        "qa_flagged_level_counts": flagged_level_counts,
    }
    try:
        with (output_dir / "_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if avg_conf is None:
        logger.info(f"Done. success={ok}, error={err}, output={output_dir}")
    else:
        logger.info(f"Done. success={ok}, error={err}, avg_conf={avg_conf:.2f}, output={output_dir}")


if __name__ == "__main__":
    main()
