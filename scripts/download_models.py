"""Download official OpenCV Zoo models for this application."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow invocation from the repository root or the scripts directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppPaths
from core.errors import FaceApplicationError
from core.model_manager import MODEL_SPECS, ModelManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download official OpenCV Zoo YuNet and SFace ONNX models.")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), help="Download only one model.")
    parser.add_argument("--models-dir", type=Path, help="Destination model directory.")
    parser.add_argument("--force", action="store_true", help="Replace a model that already exists.")
    args = parser.parse_args(argv)
    manager = ModelManager(AppPaths(models_dir=args.models_dir))
    try:
        for spec in ([MODEL_SPECS[args.model]] if args.model else MODEL_SPECS.values()):
            target = manager.download(spec, force=args.force)
            print(f"Installed {spec.key}: {target}")
            print(f"SHA-256: {manager.verify(spec)}")
            print(f"License: {spec.license_name}; source: {spec.source_page}")
        return 0
    except FaceApplicationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
