from __future__ import annotations

import numpy as np
import pytest

from core.errors import LandmarksRequiredError
from core.face_recognizer import FaceRecognizer, StoredEmbedding


def test_cosine_similarity_of_identical_vectors_is_one(normalized_embedding):
    assert FaceRecognizer.cosine_similarity(normalized_embedding, normalized_embedding) == pytest.approx(1.0)


def test_normalize_embedding_rejects_wrong_dimension():
    with pytest.raises(Exception, match="128"):
        FaceRecognizer.normalize_embedding(np.ones(12, dtype=np.float32))


def test_haar_detection_cannot_create_embedding(haar_detection):
    recognizer = object.__new__(FaceRecognizer)
    with pytest.raises(LandmarksRequiredError):
        recognizer.embedding_from_detection(np.zeros((100, 100, 3), dtype=np.uint8), haar_detection)


def test_match_selects_best_sample_per_identity_and_applies_threshold(yunet_detection, normalized_embedding):
    recognizer = object.__new__(FaceRecognizer)
    recognizer.embedding_from_detection = lambda frame, detection: normalized_embedding  # type: ignore[method-assign]
    known = StoredEmbedding("one", "One", "sample-one", normalized_embedding)
    opposite = StoredEmbedding("two", "Two", "sample-two", -normalized_embedding)
    result = recognizer.recognize(np.zeros((8, 8, 3), dtype=np.uint8), yunet_detection, [opposite, known], 0.5)
    assert result.status == "recognized"
    assert result.display_name == "One"
    rejected = recognizer.recognize(np.zeros((8, 8, 3), dtype=np.uint8), yunet_detection, [opposite], 0.5)
    assert rejected.status == "unknown"


def test_haar_recognition_is_explicitly_detection_only(haar_detection, normalized_embedding):
    recognizer = object.__new__(FaceRecognizer)
    result = recognizer.recognize(
        np.zeros((8, 8, 3), dtype=np.uint8),
        haar_detection,
        [StoredEmbedding("one", "One", "sample", normalized_embedding)],
        0.363,
    )
    assert result.status == "detection_only"
    assert "landmarks" in result.reason.lower()
