"""Face detection adapters for YuNet and a clearly limited Haar fallback."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from core.errors import DetectorError, ModelMissingError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetection:
    """One detected face.

    ``raw`` preserves YuNet's 15-value row so it can be passed unchanged to
    ``FaceRecognizerSF.alignCrop``. Haar detections deliberately have no raw
    landmark row and therefore cannot be used to create an SFace embedding.
    """

    bbox: tuple[int, int, int, int]
    confidence: float | None
    detector: Literal["yunet", "haar"]
    landmarks: np.ndarray | None = None
    raw: np.ndarray | None = None

    @property
    def supports_recognition(self) -> bool:
        return self.detector == "yunet" and self.landmarks is not None and self.raw is not None


class YuNetDetector:
    """OpenCV YuNet detector with image-size-aware inference."""

    name = "yunet"
    supports_landmarks = True

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.90,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelMissingError(
                f"YuNet model is missing: {path}. Run: python scripts/download_models.py"
            )
        if not hasattr(cv2, "FaceDetectorYN"):
            raise DetectorError(
                "This OpenCV installation does not expose cv2.FaceDetectorYN. "
                "Install the project's opencv-contrib-python dependency."
            )
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold, nms_threshold, top_k
            )
        except cv2.error as exc:
            raise DetectorError(f"Unable to load YuNet model '{path}': {exc}") from exc
        self._input_size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        _validate_frame(frame)
        height, width = frame.shape[:2]
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
        try:
            _, faces = self._detector.detect(frame)
        except cv2.error as exc:
            raise DetectorError(f"YuNet detection failed: {exc}") from exc
        if faces is None:
            return []

        detections: list[FaceDetection] = []
        for row in np.asarray(faces):
            detection = self._parse_row(row, width, height)
            if detection is not None:
                detections.append(detection)
        return detections

    @staticmethod
    def _parse_row(row: np.ndarray, image_width: int, image_height: int) -> FaceDetection | None:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 15 or not np.isfinite(values[:15]).all():
            LOGGER.warning("Ignoring malformed YuNet detection row")
            return None

        x, y, width, height = values[:4]
        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))
        x2 = min(image_width, int(np.ceil(x + width)))
        y2 = min(image_height, int(np.ceil(y + height)))
        if x2 <= x1 or y2 <= y1:
            return None

        landmarks = values[4:14].reshape(5, 2).copy()
        if not np.isfinite(landmarks).all():
            return None
        # Landmarks can be a few pixels outside on a boundary face, but values
        # far outside indicate malformed model output.
        if (
            (landmarks[:, 0] < -image_width).any()
            or (landmarks[:, 0] > image_width * 2).any()
            or (landmarks[:, 1] < -image_height).any()
            or (landmarks[:, 1] > image_height * 2).any()
        ):
            return None

        return FaceDetection(
            bbox=(x1, y1, x2 - x1, y2 - y1),
            confidence=float(values[14]),
            detector="yunet",
            landmarks=landmarks,
            raw=values[:15].reshape(1, 15).copy(),
        )


class HaarCascadeDetector:
    """Detection-only fallback used when YuNet is unavailable.

    Haar cascade output has no five-point landmarks. It must never be routed to
    SFace alignment or recognition.
    """

    name = "haar"
    supports_landmarks = False

    def __init__(
        self,
        cascade_path: str | Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
    ) -> None:
        path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.is_file():
            raise DetectorError(f"Haar cascade is missing: {path}")
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise DetectorError(f"Unable to load Haar cascade: {path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        _validate_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            boxes = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
        except cv2.error as exc:
            raise DetectorError(f"Haar detection failed: {exc}") from exc
        return [
            FaceDetection(
                bbox=tuple(int(value) for value in box),
                confidence=None,
                detector="haar",
            )
            for box in boxes
        ]


class FaceDetector:
    """Select YuNet where possible and transparently label Haar fallback."""

    def __init__(
        self,
        yunet_model_path: str | Path | None,
        score_threshold: float = 0.90,
        allow_haar_fallback: bool = True,
    ) -> None:
        self.initialization_error: str | None = None
        self.backend: YuNetDetector | HaarCascadeDetector
        if yunet_model_path:
            try:
                self.backend = YuNetDetector(yunet_model_path, score_threshold=score_threshold)
                return
            except (DetectorError, ModelMissingError) as exc:
                self.initialization_error = str(exc)
                LOGGER.warning("YuNet unavailable; %s", exc)
        if not allow_haar_fallback:
            raise DetectorError(self.initialization_error or "YuNet model path was not provided.")
        self.backend = HaarCascadeDetector()

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def supports_landmarks(self) -> bool:
        return self.backend.supports_landmarks

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        return self.backend.detect(frame)


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise DetectorError("Expected a non-empty OpenCV image frame.")
    if frame.ndim not in (2, 3):
        raise DetectorError("Expected a two- or three-dimensional image frame.")
    if frame.ndim == 3 and frame.shape[2] not in (3, 4):
        raise DetectorError("Expected a grayscale, BGR, or BGRA image frame.")
