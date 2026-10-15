from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import AppPaths
from core.face_detector import FaceDetection


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    return AppPaths(root=tmp_path, data_dir=tmp_path / "private-data", output_dir=tmp_path / "output", models_dir=tmp_path / "models")


@pytest.fixture
def normalized_embedding() -> np.ndarray:
    return np.ones(128, dtype=np.float32) / np.sqrt(128)


@pytest.fixture
def yunet_detection() -> FaceDetection:
    row = np.array([[10, 15, 90, 100, 30, 45, 70, 45, 50, 65, 35, 90, 65, 90, 0.95]], dtype=np.float32)
    return FaceDetection(
        bbox=(10, 15, 90, 100),
        confidence=0.95,
        detector="yunet",
        landmarks=row[0, 4:14].reshape(5, 2),
        raw=row,
    )


@pytest.fixture
def haar_detection() -> FaceDetection:
    return FaceDetection(bbox=(10, 15, 90, 100), confidence=None, detector="haar")
