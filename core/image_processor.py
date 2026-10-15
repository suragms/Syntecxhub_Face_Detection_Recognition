"""Recognition of all detectable faces in a still image."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.errors import MediaError
from core.face_database import FaceDatabase
from core.face_detector import FaceDetector
from core.face_recognizer import FaceRecognizer, RecognitionResult
from core.rendering import annotate_frame


@dataclass(frozen=True)
class ImageProcessingSummary:
    detections: int
    recognized: int
    unknown: int
    detection_only: int
    results: tuple[RecognitionResult, ...]
    output_path: Path | None


class ImageProcessor:
    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer | None, database: FaceDatabase, threshold: float) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.database = database
        self.threshold = threshold

    def process(self, input_path: str | Path, output_path: str | Path | None = None) -> tuple[np.ndarray, ImageProcessingSummary]:
        source = Path(input_path)
        if not source.is_file():
            raise MediaError(f"Image does not exist: {source}")
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise MediaError("Supported image formats are JPG, JPEG, and PNG.")
        frame = cv2.imread(str(source))
        if frame is None or frame.size == 0:
            raise MediaError(f"OpenCV could not read image: {source}")
        return self.process_frame(frame, output_path)

    def process_frame(self, frame: np.ndarray, output_path: str | Path | None = None) -> tuple[np.ndarray, ImageProcessingSummary]:
        detections = self.detector.detect(frame)
        candidates = self.database.load_embeddings()
        results: list[RecognitionResult] = []
        for detection in detections:
            if self.recognizer is None:
                results.append(
                    RecognitionResult(
                        detection=detection,
                        status="detection_only",
                        reason="No SFace recognizer was loaded.",
                    )
                )
            else:
                results.append(self.recognizer.recognize(frame, detection, candidates, self.threshold))
        annotated = annotate_frame(frame, results)
        target = Path(output_path) if output_path else None
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), annotated):
                raise MediaError(f"Could not save annotated image: {target}")
        summary = ImageProcessingSummary(
            detections=len(results),
            recognized=sum(result.status == "recognized" for result in results),
            unknown=sum(result.status == "unknown" for result in results),
            detection_only=sum(result.status == "detection_only" for result in results),
            results=tuple(results),
            output_path=target,
        )
        return annotated, summary
