"""Command-line entry point for Syntecxhub Face Detection & Recognition."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2

from config import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, DEFAULT_DETECTION_THRESHOLD, DEFAULT_RECOGNITION_THRESHOLD, AppPaths
from core.errors import FaceApplicationError, ModelMissingError
from core.face_database import FaceDatabase
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.face_registration import FaceRegistrationService
from core.image_processor import ImageProcessor
from core.model_manager import MODEL_SPECS, ModelManager, check_opencv_features
from core.video_processor import VideoProcessor
from core.webcam_processor import WebcamProcessor


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local OpenCV YuNet/SFace face detection and recognition for consenting participants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=("webcam", "image", "video"), help="Recognition input source.")
    parser.add_argument("--input", help="Image/video input path (required for image or video source).")
    parser.add_argument("--output", help="Annotated image/video output path.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--data-dir", type=Path, help="Private local data directory.")
    parser.add_argument("--models-dir", type=Path, help="Directory that stores downloaded ONNX models.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_RECOGNITION_THRESHOLD, help="Minimum cosine similarity for a known identity.")
    parser.add_argument("--detection-threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD, help="YuNet face detection score threshold.")
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT, help="Requested webcam height.")
    parser.add_argument("--display", action="store_true", help="Show annotated recorded-video frames while processing.")
    parser.add_argument("--debug", action="store_true", help="Show exception tracebacks for troubleshooting.")

    operations = parser.add_argument_group("management and enrollment")
    operations.add_argument("--models", choices=("status", "download", "verify"), help="Show, install, or verify local models.")
    operations.add_argument("--model", choices=tuple(MODEL_SPECS), help="Model affected by --models download/verify.")
    operations.add_argument("--force", action="store_true", help="Replace an existing model when downloading.")
    operations.add_argument("--register", choices=("webcam", "image"), help="Register a consenting person.")
    operations.add_argument("--name", help="Display name for --register.")
    operations.add_argument("--samples", type=int, default=10, help="Webcam enrollment sample count.")
    operations.add_argument("--append", action="store_true", help="Add a sample to an existing matching name.")
    operations.add_argument("--no-retain-source", action="store_true", help="Do not retain aligned enrollment crops for rebuilds.")
    operations.add_argument("--list-people", action="store_true", help="List local registrations.")
    operations.add_argument("--delete", metavar="PERSON_ID_OR_NAME", help="Delete one local registration.")
    operations.add_argument("--yes", action="store_true", help="Confirm the requested deletion.")
    operations.add_argument("--rebuild", action="store_true", help="Validate or recompute retained-source embeddings into a new database generation.")
    operations.add_argument("--verify-only", action="store_true", help="Validate the current embedding database without writing a generation.")
    operations.add_argument("--strict", action="store_true", help="Fail rebuild if any retained source cannot be recomputed.")
    return parser


def make_paths(args: argparse.Namespace) -> AppPaths:
    return AppPaths(data_dir=args.data_dir, models_dir=args.models_dir, output_dir=Path(args.output).parent if args.output else None)


def _detector(paths: AppPaths, args: argparse.Namespace) -> FaceDetector:
    return FaceDetector(paths.yunet_model_path, score_threshold=args.detection_threshold, allow_haar_fallback=True)


def _recognizer(paths: AppPaths) -> FaceRecognizer:
    manager = ModelManager(paths)
    manager.require_recognition_models()
    return FaceRecognizer(paths.sface_model_path)


def _image_processor(paths: AppPaths, args: argparse.Namespace, require_recognition: bool = True) -> ImageProcessor:
    database = FaceDatabase(paths)
    recognizer = _recognizer(paths) if require_recognition else None
    return ImageProcessor(_detector(paths, args), recognizer, database, args.threshold)


def dispatch(args: argparse.Namespace) -> int:
    paths = make_paths(args)
    paths.ensure_runtime_directories()
    database = FaceDatabase(paths)
    manager = ModelManager(paths)

    if args.models:
        specs = [MODEL_SPECS[args.model]] if args.model else list(MODEL_SPECS.values())
        for spec in specs:
            if args.models == "download":
                path = manager.download(spec, force=args.force)
                print(f"Installed {spec.key}: {path}")
                print(f"SHA-256: {manager.verify(spec)}")
            else:
                status = manager.model_status(spec)
                if args.models == "verify":
                    status["sha256"] = manager.verify(spec)
                    status["verified"] = True
                print(f"{spec.key}: {status}")
        print(f"OpenCV: {cv2.__version__}")
        print(f"FaceDetectorYN: {hasattr(cv2, 'FaceDetectorYN')}")
        print(f"FaceRecognizerSF: {hasattr(cv2, 'FaceRecognizerSF')}")
        return 0

    if args.list_people:
        people = database.list_people()
        if not people:
            print("No people are registered in the local database.")
        for person in people:
            print(f"{person.person_id} | {person.display_name} | samples: {person.sample_count} | registered: {person.created_at}")
        return 0

    if args.delete:
        if not args.yes:
            raise FaceApplicationError("Deletion is permanent. Re-run with --yes to confirm.")
        person = database.delete_person(args.delete)
        print(f"Deleted '{person.display_name}' ({person.person_id}) and {person.sample_count} embedding sample(s).")
        return 0

    if args.rebuild or args.verify_only:
        if args.verify_only:
            people, vectors = database.snapshot()
            print(f"Database validation passed: {len(people)} people, {len(vectors)} embeddings.")
            return 0
        recognizer = _recognizer(paths)
        result = database.rebuild_database(_detector(paths, args), recognizer, strict=args.strict)
        print(
            f"Database rebuilt: {result['people']} people, {result['embeddings']} embeddings; "
            f"recomputed: {result['rebuilt']}, skipped: {result['skipped']}."
        )
        return 0

    if args.register:
        if not args.name:
            raise FaceApplicationError("--name is required with --register.")
        recognizer = _recognizer(paths)
        service = FaceRegistrationService(paths, database, _detector(paths, args), recognizer)
        if args.register == "image":
            if not args.input:
                raise FaceApplicationError("--input is required with --register image.")
            result = service.register_image(args.name, args.input, append=args.append, retain_source=not args.no_retain_source)
            print(f"Registered {result.person.display_name}: sample {result.sample.sample_id}, quality {result.quality_score:.3f}")
        else:
            results = service.capture_webcam(args.name, args.camera, args.samples, args.append, not args.no_retain_source)
            print(f"Registered {results[0].person.display_name} with {len(results)} sample(s).")
        return 0

    if not args.source:
        raise FaceApplicationError("Choose --source webcam, image, or video; use --help for all management commands.")
    if args.threshold < -1 or args.threshold > 1:
        raise FaceApplicationError("--threshold must be between -1 and 1.")

    processor = _image_processor(paths, args)
    if args.source == "image":
        if not args.input:
            raise FaceApplicationError("--input is required with --source image.")
        _, summary = processor.process(args.input, args.output)
        print(f"Detected Faces: {summary.detections}")
        print(f"Recognized Faces: {summary.recognized}")
        print(f"Unknown Faces: {summary.unknown}")
        if summary.detection_only:
            print(f"Detection-only Faces: {summary.detection_only}")
        for result in summary.results:
            print(f"- {result.display_name or 'Unknown' if result.status != 'detection_only' else 'Detected only'}")
        if summary.output_path:
            print(f"Saved annotated image: {summary.output_path}")
        return 0
    if args.source == "video":
        if not args.input:
            raise FaceApplicationError("--input is required with --source video.")
        if not args.output:
            raise FaceApplicationError("--output is required with --source video.")
        summary = VideoProcessor(processor).process(args.input, args.output, display=args.display)
        print(f"Saved {summary.frames_processed} annotated frame(s) to {summary.output_path}; faces detected: {summary.faces_detected}.")
        return 0
    summary = WebcamProcessor(processor, paths, args.width, args.height).run(args.camera)
    print(f"Webcam session ended: {summary.frames_processed} frame(s), {summary.screenshots_saved} screenshot(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return dispatch(args)
    except FaceApplicationError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception:
        LOGGER.exception("Unexpected application failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
