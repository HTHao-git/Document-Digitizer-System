from __future__ import annotations

from pathlib import Path


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    s = path.read_text(encoding="utf-8")
    out = s
    for old, new in replacements:
        if old in out:
            out = out.replace(old, new)
    if out == s:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    import craft_text_detector

    pkg_dir = Path(craft_text_detector.__file__).resolve().parent
    predict_py = pkg_dir / "predict.py"
    craft_utils_py = pkg_dir / "craft_utils.py"

    changed = False
    changed |= _patch_file(
        predict_py,
        [
            ("boxes_as_ratio = np.array(boxes_as_ratio)", "boxes_as_ratio = np.array(boxes_as_ratio, dtype=object)"),
            ("polys_as_ratio = np.array(polys_as_ratio)", "polys_as_ratio = np.array(polys_as_ratio, dtype=object)"),
        ],
    )
    changed |= _patch_file(
        craft_utils_py,
        [
            ("polys = np.array(polys)", "polys = np.array(polys, dtype=object)"),
        ],
    )

    print({"package_dir": str(pkg_dir), "changed": bool(changed)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

