"""Consent-based enrollment from local files or an interactive webcam."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from config import DEFAULT_BLUR_THRESHOLD, DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, AppPaths
from core.camera import CameraConfig, open_camera, safe_destroy_window
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
        width: int = DEFAULT_CAMERA_WIDTH,
        height: int = DEFAULT_CAMERA_HEIGHT,
    ) -> list[EnrollmentResult]:
        if samples < 1:
            raise EnrollmentError("Enrollment sample count must be at least one.")
        session = open_camera(camera_index, CameraConfig(width=width, height=height))
        captured: list[tuple[np.ndarray, FaceDetection]] = []
        last_capture_time = 0.0
        window_shown = False
        try:
            while len(captured) < samples:
                frame = session.read()
                preview = frame.copy()
                message = "Show one face clearly"
                detection: FaceDetection | None = None
                try:
                    detections = self.detector.detect(frame)
                    if len(detections) == 1 and detections[0].supports_recognition:
                        detection = detections[0]
                        quality = self.quality_score(frame, detection)
                        message = "Press SPACE to capture" if quality > 0 else "Improve lighting or focus"
                    elif len(detections) > 1:
                        message = "Only one face may be enrolled"
                    else:
                        message = "Need exactly one YuNet face"
                except Exception as exc:  # Preview remains recoverable; failures are visible.
                    message = f"Detection error: {exc}"
                _compose_enrollment_preview(preview, detection, f"Samples: {len(captured)}/{samples}", message)
                cv2.imshow("Syntecxhub enrollment", preview)
                window_shown = True
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
            session.release()
            if window_shown:
                safe_destroy_window("Syntecxhub enrollment")

        # Validate/embed every capture before one atomic multi-sample commit.
        prepared: list[tuple[np.ndarray, float, str | None]] = []
        created_sources: list[Path] = []
        try:
            for frame, detection in captured:
                quality = self.quality_score(frame, detection)
                embedding = self.recognizer.embedding_from_detection(frame, detection)
                source_file = self._write_normalized_crop(frame, detection) if retain_source else None
                if source_file:
                    created_sources.append(source_file)
                prepared.append((embedding, quality, str(source_file) if source_file else None))
            person, stored_samples = self.database.add_samples(name, prepared, allow_existing=append)
        except Exception:
            # No database commit is attempted after an earlier capture failure;
            # clean up any source crops that were staged for this cancelled run.
            for source in created_sources:
                source.unlink(missing_ok=True)
            raise
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


def _compose_enrollment_preview(
    frame: np.ndarray,
    detection: FaceDetection | None,
    progress: str,
    status: str,
) -> np.ndarray:
    """Render one bounded, non-overlapping enrollment UI onto a fresh frame."""
    height, width = frame.shape[:2]
    margin = max(6, min(14, width // 40))
    header = _truncate_text(progress, max(1, width - 2 * margin), 0.62, 2)
    footer = _truncate_text("SPACE: capture   Q: cancel", max(1, width - 2 * margin), 0.52, 1)
    status_text = _truncate_text(status, max(1, width - 2 * margin), 0.52, 1)
    header_size, header_baseline = _text_size(header, 0.62, 2)
    footer_size, footer_baseline = _text_size(footer, 0.52, 1)
    header_bottom = min(height, margin + header_size[1] + header_baseline + margin)
    footer_top = max(0, height - margin - footer_size[1] - footer_baseline - margin)

    _draw_panel_text(frame, header, (margin, min(height - 1, margin + header_size[1])), (0, 220, 255), 0.62, 2)
    if detection is not None:
        box = _clip_bbox(detection.bbox, width, height)
        if box is not None:
            x, y, box_width, box_height = box
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 220, 0), 2)
            status_text = _place_status(frame, status_text, box, header_bottom, footer_top)
    if detection is None:
        _draw_panel_text(frame, status_text, (margin, min(footer_top - 4, header_bottom + _text_size(status_text, 0.52, 1)[0][1] + 8)), (0, 220, 255), 0.52, 1)
    _draw_panel_text(frame, footer, (margin, max(0, min(height - 1, height - margin - footer_baseline))), (255, 255, 255), 0.52, 1)
    return frame


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = bbox
    left, top = max(0, x), max(0, y)
    right, bottom = min(width - 1, x + box_width), min(height - 1, y + box_height)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _text_size(text: str, scale: float, thickness: int) -> tuple[tuple[int, int], int]:
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)


def _truncate_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    if _text_size(text, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    value = text
    while value and _text_size(value + suffix, scale, thickness)[0][0] > max_width:
        value = value[:-1]
    return (value + suffix) if value else suffix


def _draw_panel_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float, thickness: int) -> None:
    x, baseline = origin
    text_size, _ = _text_size(text, scale, thickness)
    x = max(0, min(x, max(0, frame.shape[1] - text_size[0])))
    baseline = max(text_size[1], min(baseline, frame.shape[0] - 1))
    cv2.putText(frame, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _place_status(frame: np.ndarray, text: str, box: tuple[int, int, int, int], header_bottom: int, footer_top: int) -> str:
    x, y, width, height = box
    scale, thickness = 0.52, 1
    text_size, baseline = _text_size(text, scale, thickness)
    text_width, text_height = text_size
    candidates = [
        (x, y - 6),
        (x, y + height + text_height + baseline + 6),
        (x + 4, max(header_bottom + text_height + baseline + 4, y + text_height + baseline + 4)),
    ]
    for origin_x, origin_baseline in candidates:
        left = max(0, min(origin_x, frame.shape[1] - text_width))
        top = origin_baseline - baseline - text_height
        bottom = origin_baseline + baseline
        if top >= header_bottom and bottom <= footer_top and bottom < frame.shape[0]:
            _draw_panel_text(frame, text, (left, origin_baseline), (0, 220, 255), scale, thickness)
            return text
    # A compact, bounded body fallback is preferable to drawing outside the image.
    fallback_baseline = max(header_bottom + text_height + baseline + 2, min(footer_top - 2, frame.shape[0] - baseline - 1))
    _draw_panel_text(frame, text, (4, fallback_baseline), (0, 220, 255), scale, thickness)
    return text
