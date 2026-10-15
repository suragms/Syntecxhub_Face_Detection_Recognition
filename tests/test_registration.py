from __future__ import annotations

import numpy as np
import pytest

from core.errors import MultipleFacesDetectedError, NoFaceDetectedError
from core.face_registration import FaceRegistrationService


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
