from __future__ import annotations

import numpy as np
import pytest

from core.errors import DetectorError
from core.face_detector import FaceDetection, YuNetDetector


def test_yunet_row_parsing_preserves_landmarks_and_raw():
    row = np.array([10, 15, 100, 90, 30, 35, 75, 35, 52, 55, 35, 80, 70, 80, 0.94], dtype=np.float32)
    detection = YuNetDetector._parse_row(row, 200, 160)
    assert detection is not None
    assert detection.bbox == (10, 15, 100, 90)
    assert detection.landmarks.shape == (5, 2)
    assert detection.raw.shape == (1, 15)
    assert detection.supports_recognition


def test_yunet_rejects_invalid_detection_rows():
    assert YuNetDetector._parse_row(np.array([1, 2], dtype=np.float32), 100, 100) is None
    invalid = np.full(15, np.nan, dtype=np.float32)
    assert YuNetDetector._parse_row(invalid, 100, 100) is None


def test_detection_contract_makes_haar_nonrecognizable(haar_detection):
    assert not haar_detection.supports_recognition
    assert haar_detection.landmarks is None


def test_invalid_frame_is_rejected():
    detector = object.__new__(YuNetDetector)
    detector._detector = None
    detector._input_size = None
    with pytest.raises(DetectorError):
        detector.detect(np.array([]))
