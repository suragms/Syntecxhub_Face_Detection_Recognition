"""Consent-based enrollment from local files or an interactive webcam."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from config import DEFAULT_BLUR_THRESHOLD, AppPaths
from core.errors import CameraOpenError, EnrollmentError, MultipleFacesDetectedError, NoFaceDetectedError
from core.face_database import FaceDatabase, Person, SampleMetadata
from core.face_detector import FaceDetection, FaceDetector
from core.face_recognizer import FaceRecognizer


@dataclass(frozen=True)
class EnrollmentResult:
    person: Person
    sample: SampleMetadata
    quality_score: float


class FaceRegistrationService:
    """Validate one-face samples before they can enter the local database."""

    def __init__(
        self,
        paths: AppPaths,
        database: FaceDatabase,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    ) -> None:
        self.paths = paths
        self.database = database
        self.detector = detector
        self.recognizer = recognizer
        self.blur_threshold = blur_threshold

    def register_image(
        self,
        name: str,
        image_path: str | Path,
        append: bool = False,
        retain_source: bool = True,
    ) -> EnrollmentResult:
        path = Path(image_path)
        if not path.is_file():
            raise EnrollmentError(f"Enrollment image does not exist: {path}")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise EnrollmentError("Enrollment images must be JPG, JPEG, or PNG files.")
        frame = cv2.imread(str(path))
        if frame is None or frame.size == 0:
            raise EnrollmentError(f"OpenCV could not read enrollment image: {path}")
        return self.register_frame(name, frame, append=append, retain_source=retain_source)

    def register_frame(
        self,
        name: str,
        frame: np.ndarray,
        append: bool = False,
        retain_source: bool = True,
    ) -> EnrollmentResult:
        detection = self._single_enrollment_detection(frame)
        quality = self.quality_score(frame, detection)
        if quality <= 0:
            raise EnrollmentError("Enrollment sample is too small, too blurry, or has low detector confidence.")
        embedding = self.recognizer.embedding_from_detection(frame, detection)
        source_file = self._write_normalized_crop(frame, detection) if retain_source else None
        person, sample = self.database.add_person(
            display_name=name,
            embedding=embedding,
            quality_score=quality,
            source_file=str(source_file) if source_file else None,
            allow_existing=append,
        )
        return EnrollmentResult(person=person, sample=sample, quality_score=quality)

    def capture_webcam(
        self,
        name: str,
        camera_index: int = 0,
        samples: int = 10,
        append: bool = False,
        retain_source: bool = True,
        min_interval_seconds: float = 0.65,
    ) -> list[EnrollmentResult]:
        if samples < 1:
            raise EnrollmentError("Enrollment sample count must be at least one.")
        capture = _open_camera(camera_index)
        captured: list[tuple[np.ndarray, FaceDetection]] = []
        last_capture_time = 0.0
        try:
            while len(captured) < samples:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise CameraOpenError("Camera stopped returning frames during enrollment.")
                preview = frame.copy()
                message = "Show one face clearly"
                try:
                    detections = self.detector.detect(frame)
                    if len(detections) == 1 and detections[0].supports_recognition:
                        quality = self.quality_score(frame, detections[0])
                        message = "Press SPACE to capture" if quality > 0 else "Improve lighting or focus"
                        _draw_enrollment_box(preview, detections[0], message)
                    else:
                        message = "Need exactly one YuNet face"
                except Exception as exc:  # Preview remains recoverable; failures are visible.
                    message = f"Detection error: {exc}"
                cv2.putText(preview, f"{len(captured)}/{samples}  {message}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.putText(preview, "SPACE: capture  Q: cancel", (12, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("Syntecxhub enrollment", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    raise EnrollmentError("Registration cancelled by user.")
                if key == ord(" ") and time.monotonic() - last_capture_time >= min_interval_seconds:
                    detection = self._single_enrollment_detection(frame)
                    if self.quality_score(frame, detection) <= 0:
                        raise EnrollmentError("Captured frame did not satisfy enrollment quality checks.")
                    captured.append((frame.copy(), detection))
                    last_capture_time = time.monotonic()
        finally:
            capture.release()
            cv2.destroyWindow("Syntecxhub enrollment")

        # Validate/embed every capture before one atomic multi-sample commit.
        prepared: list[tuple[np.ndarray, float, str | None]] = []
        for frame, detection in captured:
            quality = self.quality_score(frame, detection)
            embedding = self.recognizer.embedding_from_detection(frame, detection)
            source_file = self._write_normalized_crop(frame, detection) if retain_source else None
            prepared.append((embedding, quality, str(source_file) if source_file else None))
        person, stored_samples = self.database.add_samples(name, prepared, allow_existing=append)
        return [
            EnrollmentResult(person=person, sample=sample, quality_score=sample.quality_score)
            for sample in stored_samples
        ]

    def _single_enrollment_detection(self, frame: np.ndarray) -> FaceDetection:
        detections = self.detector.detect(frame)
        if not detections:
            raise NoFaceDetectedError("No face was detected in the enrollment sample.")
        if len(detections) != 1:
            raise MultipleFacesDetectedError("Enrollment requires exactly one detectable face.")
        detection = detections[0]
        if not detection.supports_recognition:
            raise EnrollmentError(
                "Only Haar detection is available. Enrollment requires YuNet landmarks and the SFace model."
            )
        return detection

    def quality_score(self, frame: np.ndarray, detection: FaceDetection) -> float:
        x, y, width, height = detection.bbox
        area_ratio = (width * height) / max(1, frame.shape[0] * frame.shape[1])
        if min(width, height) < 60 or area_ratio < 0.015:
            return 0.0
        gray = cv2.cvtColor(frame[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self.blur_threshold:
            return 0.0
        confidence = detection.confidence or 0.0
        return round(min(1.0, area_ratio * 12.0) * min(1.0, blur / (self.blur_threshold * 2)) * confidence, 4)

    def _write_normalized_crop(self, frame: np.ndarray, detection: FaceDetection) -> Path:
        self.paths.registered_faces_dir.mkdir(parents=True, exist_ok=True)
        # SFace alignment is intentionally driven by the same verified YuNet row.
        try:
            aligned = self.recognizer._recognizer.alignCrop(frame, detection.raw)  # noqa: SLF001 - same guarded invariant
        except cv2.error as exc:
            raise EnrollmentError(f"Could not save aligned enrollment source: {exc}") from exc
        target = self.paths.registered_faces_dir / f"{time.time_ns()}.png"
        if not cv2.imwrite(str(target), aligned):
            raise EnrollmentError(f"Could not save enrollment source image: {target}")
        return target


def _open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise CameraOpenError(
            f"Unable to open camera index {index}. Check camera permissions, availability, and index."
        )
    return capture


def _draw_enrollment_box(frame: np.ndarray, detection: FaceDetection, label: str) -> None:
    x, y, width, height = detection.bbox
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
    cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
