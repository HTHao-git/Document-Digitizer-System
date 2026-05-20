import os
import sys
import yaml
import json
import argparse
import subprocess
import logging
import time
from jsonschema import validate, ValidationError
from glob import glob
from collections import defaultdict


EXIT_SUCCESS = 0
EXIT_PREPROCESS_FAILED = 10
EXIT_OCR_FAILED = 20
EXIT_CONFIG_INVALID = 30
EXIT_UNEXPECTED = 40

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".pdf"]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg, schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate(instance=cfg, schema=schema)


def validate_report(report, schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate(instance=report, schema=schema)


def setup_logger(log_path=""):
    handlers = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers
    )


def update_module_config(module_cfg_path, updates: dict):
    with open(module_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in updates.items():
        if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    with open(module_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)


def write_report(path, report: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def count_files(root_dir, exts=None):
    if not root_dir or not os.path.exists(root_dir):
        return 0
    if exts is None:
        return sum(1 for _ in glob(os.path.join(root_dir, "**", "*"), recursive=True)
                   if os.path.isfile(_))
    exts_lower = set(e.lower() for e in exts)
    count = 0
    for p in glob(os.path.join(root_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts_lower:
            count += 1
    return count


def count_by_extension(root_dir):
    result = defaultdict(int)
    if not root_dir or not os.path.exists(root_dir):
        return dict(result)
    for p in glob(os.path.join(root_dir, "**", "*"), recursive=True):
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower() or "<no_ext>"
            result[ext] += 1
    return dict(result)


def add_warning(report, message):
    report.setdefault("warnings", [])
    report["warnings"].append(message)


def timestamp_str():
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_report_path(path, report=None):
    # Nếu là thư mục (có / hoặc \), hoặc đã tồn tại là dir,
    # hoặc không có extension => coi là folder
    if path.endswith(("/", "\\")) or os.path.isdir(path) or os.path.splitext(path)[1] == "":
        if os.path.splitext(path)[1] == "" and not path.endswith(("/", "\\")) and not os.path.isdir(path):
            msg = "report_path has no extension; treating as directory."
            logging.warning("[PIPELINE] " + msg)
            if report is not None:
                add_warning(report, msg)
        return os.path.join(path, f"pipeline_report_{timestamp_str()}.json")
    return path


def abs_path(base_dir, p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def module_config_exists(path):
    return path and os.path.isfile(path)


def record_skip(report, module_name, reason):
    report.setdefault("skipped_modules", [])
    report["skipped_modules"].append({"module": module_name, "reason": reason})
    add_warning(report, f"{module_name} skipped: {reason}")


def load_test_overrides():
    raw = os.environ.get("PIPELINE_TEST_OVERRIDES")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("[PIPELINE] Invalid PIPELINE_TEST_OVERRIDES JSON")
        return {}


def get_count_overrides(test_overrides, module_name):
    return (test_overrides or {}).get("count_overrides", {}).get(module_name, {})


def run_step(name, cmd, cwd, stop_on_error, exit_code_on_fail, report, step_meta=None, dry_run=False, force_fail=False):
    t0 = time.time()
    status = "success"
    return_code = 0
    stdout_text = ""
    stderr_text = ""

    if dry_run:
        report["steps"].append({
            "name": name,
            "status": "dry-run",
            "return_code": 0,
            "duration_sec": 0.0,
            "stdout": "",
            "stderr": "",
            **(step_meta or {})
        })
        return

    try:
        if force_fail:
            status = "failed"
            return_code = 1
            stderr_text = "forced failure for testing"
            if stop_on_error:
                raise SystemExit(exit_code_on_fail)
        else:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            return_code = result.returncode
            stdout_text = (result.stdout or "").strip()
            stderr_text = (result.stderr or "").strip()

            if return_code != 0:
                status = "failed"
                if stop_on_error:
                    raise SystemExit(exit_code_on_fail)
    except SystemExit:
        status = "failed"
        raise
    finally:
        duration = round(time.time() - t0, 3)
        step = {
            "name": name,
            "status": status,
            "return_code": return_code,
            "duration_sec": duration,
            "stdout": stdout_text,
            "stderr": stderr_text
        }
        if step_meta:
            step.update(step_meta)
        report["steps"].append(step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        validate_config(cfg, os.path.join("schemas", "pipeline.schema.json"))
    except ValidationError as e:
        logging.error("[PIPELINE] Config validation failed: %s", e)
        raise SystemExit(EXIT_CONFIG_INVALID)

    setup_logger(cfg["runtime"].get("log_path", ""))

    modules = cfg["modules"]
    stop_on_error = cfg["runtime"]["stop_on_error"]

    pipeline_start = time.time()

    # base_dir = thư mục pipeline
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # repo_root = thư mục dự án
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))

    test_overrides = load_test_overrides()
    force_fail_step = test_overrides.get("force_fail_step")

    report = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "status": "running",
        "error_code": None,
        "environment": {
            "python": sys.executable,
            "platform": sys.platform
        },
        "io": {
            "input_dir": abs_path(base_dir, cfg["io"]["input_dir"]),
            "output_dir": abs_path(base_dir, cfg["io"]["output_dir"]),
            "json_dir": abs_path(base_dir, cfg["io"]["json_dir"])
        },
        "steps": [],
        "warnings": [],
        "skipped_modules": []
    }

    report_path = resolve_report_path(cfg["runtime"]["report_path"], report)

    try:
        if modules.get("preprocess", {}).get("enable", False):
            logging.info("[PIPELINE] Preprocess enabled")
            preprocess_cfg_path = abs_path(base_dir, modules["preprocess"]["config_path"])
            preprocess_out = abs_path(base_dir, modules["preprocess"]["output_dir"])
            preprocess_json = abs_path(base_dir, modules["preprocess"]["json_dir"])

            if not module_config_exists(preprocess_cfg_path):
                record_skip(report, "preprocess", f"config not found: {preprocess_cfg_path}")
                report["steps"].append({
                    "name": "preprocess",
                    "status": "skipped",
                    "return_code": 0,
                    "duration_sec": 0.0,
                    "stdout": "",
                    "stderr": "",
                    "reason": "config_missing"
                })
            else:
                if not args.dry_run:
                    update_module_config(
                        preprocess_cfg_path,
                        {"io": {"output_dir": preprocess_out, "json_dir": preprocess_json}},
                    )

                input_dir = report["io"]["input_dir"]
                input_count = count_files(input_dir, IMAGE_EXTS)
                input_by_ext = count_by_extension(input_dir)

                if input_count == 0:
                    add_warning(report, "preprocess input_files = 0")

                run_step(
                    name="preprocess",
                    cmd=[sys.executable, "-m", "src.run", "--config", "config.yaml"],
                    cwd=os.path.join(repo_root, "packages", "preprocess"),
                    stop_on_error=stop_on_error,
                    exit_code_on_fail=EXIT_PREPROCESS_FAILED,
                    report=report,
                    step_meta={
                        "input_dir": input_dir,
                        "output_dir": preprocess_out,
                        "json_dir": preprocess_json,
                        "input_files": input_count,
                        "input_by_extension": input_by_ext
                    },
                    dry_run=args.dry_run,
                    force_fail=(force_fail_step == "preprocess")
                )

                if not args.dry_run:
                    report["steps"][-1]["output_files"] = count_files(preprocess_out, IMAGE_EXTS)
                    report["steps"][-1]["output_by_extension"] = count_by_extension(preprocess_out)
                    report["steps"][-1]["json_files"] = count_files(preprocess_json)
                    report["steps"][-1]["json_by_extension"] = count_by_extension(preprocess_json)

                    overrides = get_count_overrides(test_overrides, "preprocess")
                    if "output_files" in overrides:
                        report["steps"][-1]["output_files"] = overrides["output_files"]
                    if "json_files" in overrides:
                        report["steps"][-1]["json_files"] = overrides["json_files"]

                    if report["steps"][-1]["output_files"] == 0:
                        add_warning(report, "preprocess output_files = 0")
                    if report["steps"][-1]["json_files"] == 0:
                        add_warning(report, "preprocess json_files = 0")
                    if report["steps"][-1]["output_files"] < report["steps"][-1]["input_files"]:
                        add_warning(report, "preprocess output_files < input_files")

        if modules.get("ocr", {}).get("enable", False):
            logging.info("[PIPELINE] OCR enabled")
            ocr_cfg_path = abs_path(base_dir, modules["ocr"]["config_path"])
            ocr_input = abs_path(base_dir, modules["ocr"]["input_dir"])
            ocr_json = abs_path(base_dir, modules["ocr"]["json_dir"])
            ocr_line = abs_path(base_dir, modules["ocr"]["line_dir"])

            if not module_config_exists(ocr_cfg_path):
                record_skip(report, "ocr", f"config not found: {ocr_cfg_path}")
                report["steps"].append({
                    "name": "ocr",
                    "status": "skipped",
                    "return_code": 0,
                    "duration_sec": 0.0,
                    "stdout": "",
                    "stderr": "",
                    "reason": "config_missing"
                })
            else:
                if not args.dry_run:
                    update_module_config(
                        ocr_cfg_path,
                        {"io": {"input_dir": ocr_input, "json_dir": ocr_json, "line_dir": ocr_line}},
                    )

                input_count = count_files(ocr_input, IMAGE_EXTS)
                input_by_ext = count_by_extension(ocr_input)

                if input_count == 0:
                    add_warning(report, "ocr input_files = 0")

                run_step(
                    name="ocr",
                    cmd=[sys.executable, "-m", "src.run", "--config", "config.yaml"],
                    cwd=os.path.join(repo_root, "packages", "ocr"),
                    stop_on_error=stop_on_error,
                    exit_code_on_fail=EXIT_OCR_FAILED,
                    report=report,
                    step_meta={
                        "input_dir": ocr_input,
                        "json_dir": ocr_json,
                        "line_dir": ocr_line,
                        "input_files": input_count,
                        "input_by_extension": input_by_ext
                    },
                    dry_run=args.dry_run,
                    force_fail=(force_fail_step == "ocr")
                )

                if not args.dry_run:
                    report["steps"][-1]["json_files"] = count_files(ocr_json)
                    report["steps"][-1]["json_by_extension"] = count_by_extension(ocr_json)

                    overrides = get_count_overrides(test_overrides, "ocr")
                    if "json_files" in overrides:
                        report["steps"][-1]["json_files"] = overrides["json_files"]

                    if report["steps"][-1]["json_files"] == 0:
                        add_warning(report, "ocr json_files = 0")
                    if report["steps"][-1]["json_files"] < report["steps"][-1]["input_files"]:
                        add_warning(report, "ocr json_files < input_files")

        report["status"] = "success"

    except SystemExit as e:
        report["status"] = "failed"
        report["error_code"] = int(getattr(e, "code", EXIT_UNEXPECTED))
        raise
    except Exception:
        report["status"] = "failed"
        report["error_code"] = EXIT_UNEXPECTED
        raise
    finally:
        report["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report["duration_sec"] = round(time.time() - pipeline_start, 3)

        report["summary"] = {
            "total_steps": len(report["steps"]),
            "total_input_files": sum(s.get("input_files", 0) for s in report["steps"]),
            "total_output_files": sum(s.get("output_files", 0) for s in report["steps"]),
            "total_json_files": sum(s.get("json_files", 0) for s in report["steps"])
        }

        try:
            validate_report(report, os.path.join("schemas", "report.schema.json"))
        except ValidationError as e:
            logging.error("[PIPELINE] Report validation failed: %s", e)
            report["status"] = "failed"
            report["error_code"] = EXIT_UNEXPECTED

        write_report(report_path, report)
        logging.info("[PIPELINE] Report saved: %s", report_path)


if __name__ == "__main__":
    main()