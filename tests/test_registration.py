from __future__ import annotations

import numpy as np
import pytest

from core.errors import MultipleFacesDetectedError, NoFaceDetectedError
from core.face_registration import FaceRegistrationService, _compose_enrollment_preview


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class FakeRecognizer:
    def __init__(self, vector):
        self.vector = vector

    def embedding_from_detection(self, frame, detection):
        return self.vector


def test_registration_rejects_empty_and_multiple_face_samples(paths, normalized_embedding, yunet_detection):
    from core.face_database import FaceDatabase

    database = FaceDatabase(paths)
    frame = np.full((240, 240, 3), 255, dtype=np.uint8)
    none = FaceRegistrationService(paths, database, FakeDetector([]), FakeRecognizer(normalized_embedding))
    with pytest.raises(NoFaceDetectedError):
        none.register_frame("Ada", frame, retain_source=False)
    multiple = FaceRegistrationService(paths, database, FakeDetector([yunet_detection, yunet_detection]), FakeRecognizer(normalized_embedding))
    with pytest.raises(MultipleFacesDetectedError):
        multiple.register_frame("Ada", frame, retain_source=False)


def test_registration_saves_a_local_embedding(paths, normalized_embedding, yunet_detection):
    from core.face_database import FaceDatabase

    database = FaceDatabase(paths)
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    pattern = np.indices((100, 90)).sum(axis=0)
    pattern = ((pattern % 2) * 255).astype(np.uint8)
    frame[15:115, 10:100] = np.repeat(pattern[:, :, None], 3, axis=2)
    service = FaceRegistrationService(paths, database, FakeDetector([yunet_detection]), FakeRecognizer(normalized_embedding))
    result = service.register_frame("Ada", frame, retain_source=False)
    assert result.person.display_name == "Ada"
    assert len(database.load_embeddings()) == 1


def test_enrollment_compositor_draws_status_once_and_clips_box(monkeypatch, yunet_detection):
    import core.face_registration as registration

    calls = []
    monkeypatch.setattr(registration.cv2, "putText", lambda _frame, text, point, *_args: calls.append((text, point)))
    monkeypatch.setattr(registration.cv2, "rectangle", lambda _frame, start, end, *_args: calls.append(("rectangle", start, end)))
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    oversized = type(yunet_detection)(
        bbox=(-30, -20, 200, 130),
        confidence=yunet_detection.confidence,
        detector="yunet",
        landmarks=yunet_detection.landmarks,
        raw=yunet_detection.raw,
    )
    status = "This is a deliberately long status message"
    _compose_enrollment_preview(frame, oversized, "Samples: 0/10", status)
    labels = [text for text, _point, *rest in calls if text != "rectangle"]
    assert len(labels) == 3  # header, one status, footer
    rectangle = next(item for item in calls if item[0] == "rectangle")
    assert all(0 <= coordinate < 120 for coordinate in (rectangle[1][0], rectangle[2][0]))
    assert all(0 <= coordinate < 80 for coordinate in (rectangle[1][1], rectangle[2][1]))
    for item in calls:
        if item[0] != "rectangle":
            _text, (x, y), *_ = item
            assert 0 <= x < 120
            assert 0 <= y < 80
