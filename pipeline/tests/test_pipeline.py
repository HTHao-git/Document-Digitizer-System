import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from jsonschema import validate, ValidationError


PIPELINE_DIR = Path(__file__).resolve().parents[1]
BASE_CONFIG = PIPELINE_DIR / "config" / "pipeline.yaml"
RUN_PIPELINE = PIPELINE_DIR / "run_pipeline.py"
REPORT_SCHEMA = PIPELINE_DIR / "schemas" / "report.schema.json"

TMP_DIR = PIPELINE_DIR / "tests" / "_tmp"
REPORT_DIR = TMP_DIR / "reports"
DATA_DIR = TMP_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
JSON_DIR = DATA_DIR / "json"
LOG_PATH = TMP_DIR / "pipeline.log"

EXIT_PREPROCESS_FAILED = 10

PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5eW2cAAAAASUVORK5CYII="
)


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_report_schema():
    with open(REPORT_SCHEMA, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_report_schema(report):
    schema = load_report_schema()
    validate(instance=report, schema=schema)


def write_config(cfg, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)


def resolve_module_path(p: str) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (PIPELINE_DIR / p))


def latest_report(report_dir: Path):
    reports = sorted(report_dir.glob("pipeline_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        return None, None
    with open(reports[0], "r", encoding="utf-8") as f:
        return reports[0], json.load(f)


def run_pipeline(cfg_path: Path, dry_run: bool, report_dir: Path, overrides=None):
    ensure_dir(report_dir)
    for f in report_dir.glob("pipeline_report_*.json"):
        f.unlink()

    cmd = [sys.executable, str(RUN_PIPELINE), "--config", str(cfg_path)]
    if dry_run:
        cmd.append("--dry-run")

    env = dict(**dict(**vars(__import__("os")).get("environ")))
    if overrides is not None:
        env["PIPELINE_TEST_OVERRIDES"] = json.dumps(overrides)

    result = subprocess.run(cmd, cwd=str(PIPELINE_DIR), capture_output=True, text=True, env=env)
    report_path, report = latest_report(report_dir)
    return result, report_path, report


def make_base_cfg(report_path: Path):
    cfg = load_config()

    cfg["io"]["input_dir"] = str(INPUT_DIR)
    cfg["io"]["output_dir"] = str(OUTPUT_DIR)
    cfg["io"]["json_dir"] = str(JSON_DIR)

    cfg["runtime"]["report_path"] = str(report_path)
    cfg["runtime"]["log_path"] = str(LOG_PATH)

    cfg["modules"]["preprocess"]["config_path"] = resolve_module_path(cfg["modules"]["preprocess"]["config_path"])
    cfg["modules"]["ocr"]["config_path"] = resolve_module_path(cfg["modules"]["ocr"]["config_path"])
    return cfg


def write_sample_input():
    ensure_dir(INPUT_DIR)
    with open(INPUT_DIR / "sample.png", "wb") as f:
        f.write(PNG_1x1)


def record(results, name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def assert_report_schema(results, test_name, report):
    if not report:
        return record(results, test_name, "FAIL", "report not found for schema validation")

    try:
        validate_report_schema(report)
    except ValidationError as e:
        return record(results, test_name, "FAIL", f"report schema invalid: {e.message}")
    return None


def test_dry_run(results):
    cfg = make_base_cfg(REPORT_DIR)
    cfg_path = TMP_DIR / "dry_run.yaml"
    write_config(cfg, cfg_path)

    result, _, report = run_pipeline(cfg_path, dry_run=True, report_dir=REPORT_DIR)
    if result.returncode != 0:
        return record(results, "dry_run", "FAIL", result.stderr.strip())

    schema_fail = assert_report_schema(results, "dry_run_schema", report)
    if schema_fail:
        return

    if report.get("status") != "success":
        return record(results, "dry_run", "FAIL", "report status != success")

    if report.get("steps"):
        allowed = {"dry-run", "skipped"}
        if not all(s.get("status") in allowed for s in report["steps"]):
            return record(results, "dry_run", "FAIL", "not all steps are dry-run or skipped")

    return record(results, "dry_run", "PASS")


def test_missing_module_config(results):
    cfg = make_base_cfg(REPORT_DIR)
    cfg["modules"]["preprocess"]["config_path"] = str(TMP_DIR / "missing_preprocess.yaml")
    cfg["modules"]["ocr"]["config_path"] = str(TMP_DIR / "missing_ocr.yaml")

    cfg_path = TMP_DIR / "missing_module.yaml"
    write_config(cfg, cfg_path)

    result, _, report = run_pipeline(cfg_path, dry_run=True, report_dir=REPORT_DIR)
    if result.returncode != 0:
        return record(results, "missing_module_config", "FAIL", result.stderr.strip())

    schema_fail = assert_report_schema(results, "missing_module_schema", report)
    if schema_fail:
        return

    skipped = report.get("skipped_modules", []) if report else []
    if not any(s.get("module") == "preprocess" for s in skipped):
        return record(results, "missing_module_config", "FAIL", "missing preprocess skip")
    if not any(s.get("module") == "ocr" for s in skipped):
        return record(results, "missing_module_config", "FAIL", "missing ocr skip")

    return record(results, "missing_module_config", "PASS")


def test_input_empty(results):
    cfg = make_base_cfg(REPORT_DIR)
    cfg["modules"]["ocr"]["enable"] = False

    preprocess_cfg = Path(cfg["modules"]["preprocess"]["config_path"])
    if not preprocess_cfg.exists():
        return record(results, "input_empty", "SKIP", "preprocess config not found")

    cfg_path = TMP_DIR / "input_empty.yaml"
    write_config(cfg, cfg_path)

    ensure_clean_dir(INPUT_DIR)

    result, _, report = run_pipeline(cfg_path, dry_run=True, report_dir=REPORT_DIR)
    if result.returncode != 0:
        return record(results, "input_empty", "FAIL", result.stderr.strip())

    schema_fail = assert_report_schema(results, "input_empty_schema", report)
    if schema_fail:
        return

    warnings = report.get("warnings", []) if report else []
    if not any("preprocess input_files = 0" in w for w in warnings):
        return record(results, "input_empty", "FAIL", "missing input_files=0 warning")

    return record(results, "input_empty", "PASS")


def test_output_less_than_input(results):
    cfg = make_base_cfg(REPORT_DIR)
    cfg["modules"]["ocr"]["enable"] = False

    preprocess_cfg = Path(cfg["modules"]["preprocess"]["config_path"])
    if not preprocess_cfg.exists():
        return record(results, "output_less_than_input", "SKIP", "preprocess config not found")

    cfg_path = TMP_DIR / "output_less.yaml"
    write_config(cfg, cfg_path)

    ensure_clean_dir(INPUT_DIR)
    write_sample_input()

    overrides = {"count_overrides": {"preprocess": {"output_files": 0}}}

    result, _, report = run_pipeline(cfg_path, dry_run=False, report_dir=REPORT_DIR, overrides=overrides)
    if result.returncode != 0:
        return record(results, "output_less_than_input", "FAIL", result.stderr.strip())

    schema_fail = assert_report_schema(results, "output_less_schema", report)
    if schema_fail:
        return

    warnings = report.get("warnings", []) if report else []
    if any("preprocess output_files < input_files" in w for w in warnings):
        return record(results, "output_less_than_input", "PASS")

    return record(results, "output_less_than_input", "FAIL", "missing output < input warning")


def test_report_path_warning(results):
    report_path = TMP_DIR / "reports_no_ext"
    if report_path.exists():
        shutil.rmtree(report_path)

    cfg = make_base_cfg(report_path)
    cfg_path = TMP_DIR / "report_path_warn.yaml"
    write_config(cfg, cfg_path)

    result, _, report = run_pipeline(cfg_path, dry_run=True, report_dir=report_path)
    if result.returncode != 0:
        return record(results, "report_path_warning", "FAIL", result.stderr.strip())

    schema_fail = assert_report_schema(results, "report_path_schema", report)
    if schema_fail:
        return

    warnings = report.get("warnings", []) if report else []
    if any("report_path has no extension" in w for w in warnings):
        return record(results, "report_path_warning", "PASS")

    return record(results, "report_path_warning", "FAIL", "missing report_path warning")


def test_stop_on_error(results):
    cfg = make_base_cfg(REPORT_DIR)
    cfg["modules"]["ocr"]["enable"] = False

    preprocess_cfg = Path(cfg["modules"]["preprocess"]["config_path"])
    if not preprocess_cfg.exists():
        return record(results, "stop_on_error", "SKIP", "preprocess config not found")

    cfg_path = TMP_DIR / "stop_on_error.yaml"
    write_config(cfg, cfg_path)

    overrides = {"force_fail_step": "preprocess"}

    result, _, report = run_pipeline(cfg_path, dry_run=False, report_dir=REPORT_DIR, overrides=overrides)
    if result.returncode != EXIT_PREPROCESS_FAILED:
        return record(results, "stop_on_error", "FAIL", f"return_code={result.returncode}")

    schema_fail = assert_report_schema(results, "stop_on_error_schema", report)
    if schema_fail:
        return

    if report and report.get("error_code") == EXIT_PREPROCESS_FAILED and report.get("status") == "failed":
        return record(results, "stop_on_error", "PASS")

    return record(results, "stop_on_error", "FAIL", "report status/error_code mismatch")


def main():
    ensure_clean_dir(TMP_DIR)
    ensure_dir(REPORT_DIR)
    ensure_dir(DATA_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(JSON_DIR)

    results = []
    test_dry_run(results)
    test_missing_module_config(results)
    test_input_empty(results)
    test_output_less_than_input(results)
    test_report_path_warning(results)
    test_stop_on_error(results)

    fails = [r for r in results if r[1] == "FAIL"]
    print("\n=== Summary ===")
    for name, status, detail in results:
        print(f"{status:5} | {name}" + (f" | {detail}" if detail else ""))

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()