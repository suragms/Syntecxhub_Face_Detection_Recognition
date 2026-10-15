from __future__ import annotations

import numpy as np
import pytest

from core.errors import MediaError
from core.face_database import FaceDatabase
from core.face_recognizer import RecognitionResult
from core.image_processor import ImageProcessor


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class FakeRecognizer:
    def recognize(self, frame, detection, candidates, threshold):
        return RecognitionResult(detection=detection, status="unknown", similarity=0.1)


def test_image_processor_reports_multiple_faces(paths, yunet_detection):
    processor = ImageProcessor(FakeDetector([yunet_detection, yunet_detection]), FakeRecognizer(), FaceDatabase(paths), 0.363)
    _, summary = processor.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))
    assert summary.detections == 2
    assert summary.unknown == 2


def test_image_processor_rejects_missing_input(paths):
    processor = ImageProcessor(FakeDetector([]), None, FaceDatabase(paths), 0.363)
    with pytest.raises(MediaError, match="does not exist"):
        processor.process(paths.root / "missing.png")
