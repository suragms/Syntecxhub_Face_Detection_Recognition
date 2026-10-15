"""OpenCV SFace embedding creation and local cosine-similarity matching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

from core.errors import LandmarksRequiredError, ModelMissingError, RecognitionError
from core.face_detector import FaceDetection


@dataclass(frozen=True)
class StoredEmbedding:
    """A single locally registered embedding and its identity metadata."""

    person_id: str
    display_name: str
    sample_id: str
    vector: np.ndarray


@dataclass(frozen=True)
class RecognitionResult:
    """A genuine matching result for one detection."""

    detection: FaceDetection
    status: Literal["recognized", "unknown", "detection_only"]
    person_id: str | None = None
    display_name: str | None = None
    similarity: float | None = None
    reason: str | None = None


class FaceRecognizer:
    """Generate normalized SFace vectors and match them against local samples."""

    embedding_dimension = 128

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelMissingError(
                f"SFace model is missing: {path}. Run: python scripts/download_models.py"
            )
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RecognitionError(
                "This OpenCV installation does not expose cv2.FaceRecognizerSF. "
                "Install the project's opencv-contrib-python dependency."
            )
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")
        except cv2.error as exc:
            raise RecognitionError(f"Unable to load SFace model '{path}': {exc}") from exc

    def embedding_from_detection(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Align a YuNet face and return an L2-normalized 128-value embedding."""
        if not detection.supports_recognition:
            raise LandmarksRequiredError(
                "SFace embeddings require a YuNet detection with five landmarks. "
                "Haar fallback detections are detection-only."
            )
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RecognitionError("Cannot create an embedding from an empty frame.")
        try:
            aligned = self._recognizer.alignCrop(frame, detection.raw)
            vector = self._recognizer.feature(aligned)
        except cv2.error as exc:
            raise RecognitionError(f"SFace alignment or feature extraction failed: {exc}") from exc
        return self.normalize_embedding(vector)

    @classmethod
    def normalize_embedding(cls, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if values.size != cls.embedding_dimension:
            raise RecognitionError(
                f"Expected a {cls.embedding_dimension}-value SFace embedding, got {values.size}."
            )
        if not np.isfinite(values).all():
            raise RecognitionError("SFace returned an embedding containing non-finite values.")
        norm = float(np.linalg.norm(values))
        if norm <= np.finfo(np.float32).eps:
            raise RecognitionError("SFace returned a zero-length embedding.")
        return np.ascontiguousarray(values / norm, dtype=np.float32)

    @classmethod
    def cosine_similarity(cls, probe: np.ndarray, candidate: np.ndarray) -> float:
        probe_vector = cls.normalize_embedding(probe)
        candidate_vector = cls.normalize_embedding(candidate)
        return float(np.clip(np.dot(probe_vector, candidate_vector), -1.0, 1.0))

    def recognize(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        candidates: Iterable[StoredEmbedding],
        threshold: float,
    ) -> RecognitionResult:
        if not detection.supports_recognition:
            return RecognitionResult(
                detection=detection,
                status="detection_only",
                reason="Haar fallback has no facial landmarks; SFace matching was skipped.",
            )

        probe = self.embedding_from_detection(frame, detection)
        best_by_person: dict[str, tuple[StoredEmbedding, float]] = {}
        for candidate in candidates:
            try:
                similarity = self.cosine_similarity(probe, candidate.vector)
            except RecognitionError:
                # A corrupt persisted vector must not make all recognition fail.
                continue
            current = best_by_person.get(candidate.person_id)
            if current is None or similarity > current[1]:
                best_by_person[candidate.person_id] = (candidate, similarity)

        if not best_by_person:
            return RecognitionResult(
                detection=detection,
                status="unknown",
                reason="No valid registered embeddings are available.",
            )
        best, score = max(best_by_person.values(), key=lambda item: item[1])
        if score < threshold:
            return RecognitionResult(
                detection=detection,
                status="unknown",
                similarity=score,
                reason="Best local similarity is below the recognition threshold.",
            )
        return RecognitionResult(
            detection=detection,
            status="recognized",
            person_id=best.person_id,
            display_name=best.display_name,
            similarity=score,
        )
