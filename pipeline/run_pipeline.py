#!/usr/bin/env python3
"""
Pipeline Orchestrator Script
Wires together preprocess -> ocr -> extraction modules.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Setup system paths to allow clean programmatic imports from packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Orchestrator logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [MASTER-PIPELINE] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pipeline_orchestrator")


def bootstrap_environment():
    """Execute pre-flight patches for specific upstream library issues."""
    logger.info("Initializing pre-flight environment checks...")
    try:
        # Dynamically import and run the craft detector patch package
        from packages.ocr.scripts import patch_craft_text_detector
        patch_craft_text_detector.main()
        logger.info("Upstream system dependencies patched successfully.")
    except ImportError:
        logger.warning("patch_craft_text_detector not found or already handled. Skipping patch step.")
    except Exception as e:
        logger.error(f"Pre-flight setup failed: {e}")


def run_pipeline(input_dir: Path, workspace_dir: Path, config_overrides: dict):
    """Orchestrates structural pipelines sequentially."""
    start_time = time.time()
    
    # 1. Define stage-isolated directories
    prep_img_dir = workspace_dir / "01_preprocess" / "images"
    prep_json_dir = workspace_dir / "01_preprocess" / "json"
    ocr_json_dir = workspace_dir / "02_ocr" / "json"
    extraction_json_dir = workspace_dir / "03_extraction" / "json"

    # Ensure work spaces exist
    for folder in [prep_img_dir, prep_json_dir, ocr_json_dir, extraction_json_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute strings to ensure seamless communication across module boundaries
    input_dir_abs = input_dir.resolve()
    prep_img_abs = prep_img_dir.resolve()
    prep_json_abs = prep_json_dir.resolve()
    ocr_json_abs = ocr_json_dir.resolve()
    extraction_json_abs = extraction_json_dir.resolve()

    # =========================================================================
    # STAGE 1: Preprocessing
    # =========================================================================
    logger.info("=== STAGE 1: Launching Image Preprocessing ===")
    from packages.preprocess.src.preprocess import process_folder as run_preprocess
    
    preprocess_config = config_overrides.get("preprocess", "packages/preprocess/config.yaml")
    run_preprocess(
        input_dir=input_dir_abs,
        output_dir=prep_img_abs,
        json_dir=prep_json_abs,
        config_path=preprocess_config,
        log_path=None
    )

    # =========================================================================
    # STAGE 2: Optical Character Recognition (OCR)
    # =========================================================================
    logger.info("=== STAGE 2: Launching Document Text Detection & OCR ===")
    
    # Import the individual processor and config loader from the OCR package
    from packages.ocr.src.config import load_config as load_ocr_config
    from packages.ocr.src.ocr import process_one as run_ocr_one
    from packages.ocr.src.run import _iter_json_files
    
    ocr_config_path = config_overrides.get("ocr", "packages/ocr/config.yaml")
    
    # 1. Load configuration and dynamically resolve its schema path to avoid Errno 2
    ocr_package_root = PROJECT_ROOT / "packages" / "ocr"
    ocr_schema_path = ocr_package_root / "schemas" / "config.schema.json"
    cfg_ocr = load_ocr_config(ocr_config_path, schema_path=ocr_schema_path)
    
    # 2. Gather the JSON file outputs produced by Stage 1
    input_jsons = _iter_json_files(prep_json_abs)
    
    if not input_jsons:
        logger.warning("No preprocessing JSON metadata files found for Stage 2 OCR processing.")
    
    # 3. Process each document through the engine sequentially
    for json_path in input_jsons:
        logger.info(f"Running OCR Engine for metadata trace: {json_path.name}")
        try:
            # Resolves output JSON target
            out_json_path = ocr_json_abs / json_path.name
            
            # Fire the single page engine execution logic
            result = run_ocr_one(json_path, cfg_ocr)
            
            # Save validated output to our isolated Stage 2 directory workspace
            with open(out_json_path, "w", encoding="utf-8") as f:
                import json
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed processing OCR layer on {json_path.name}: {e}")

    # =========================================================================
    # STAGE 3: Semantic Extraction
    # =========================================================================
    logger.info("=== STAGE 3: Launching Structural Dynamic Extraction ===")
    from packages.extraction.src import process_folder as run_extraction
    
    extraction_config = config_overrides.get("extraction", "packages/extraction/config.yaml")
    run_extraction(
        input_dir=ocr_json_abs,
        output_dir=extraction_json_abs,
        config_path=extraction_config,
        schema_path="packages/extraction/schemas/extraction.schema.json",
        config_schema_path="packages/extraction/schemas/config.schema.json",
        max_workers=4
    )

    total_time = time.time() - start_time
    logger.info(f"Pipeline completed successfully execution run in {total_time:.2f} seconds.")
    logger.info(f"Final structured extraction outputs can be found here: {extraction_json_abs}")


def main():
    parser = argparse.ArgumentParser(description="Unified OCR Extraction Master Orchestrator")
    parser.add_argument("--input", default="data/00_input", help="Directory container holding original image source documents.")
    parser.add_argument("--workspace", default="data", help="Root directory outputting multi-tier pipeline lifecycle structures.")
    parser.add_argument("--prep-cfg", default="packages/preprocess/config.yaml", help="Path to preprocessing yaml settings.")
    parser.add_argument("--ocr-cfg", default="packages/ocr/config.yaml", help="Path to text detector and transcription setting paths.")
    parser.add_argument("--extract-cfg", default="packages/extraction/config.yaml", help="Path to parsing configurations.")
    parser.add_argument("--visualize", action="store_true", help="Enable rendering bounding box overlays for quality validation.")
    
    args = parser.parse_args()

    # Pre-flight library patching sequence
    bootstrap_environment()

    overrides = {
        "preprocess": args.prep_cfg,
        "ocr": args.ocr_cfg,
        "extraction": args.extract_cfg,
        "visualize_ocr": args.visualize
    }

    run_pipeline(
        input_dir=Path(args.input),
        workspace_dir=Path(args.workspace),
        config_overrides=overrides
    )


if __name__ == "__main__":
    main()