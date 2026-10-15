"""Application configuration and filesystem locations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTION_THRESHOLD = 0.90
DEFAULT_NMS_THRESHOLD = 0.30
DEFAULT_RECOGNITION_THRESHOLD = 0.363
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_BLUR_THRESHOLD = 45.0


@dataclass(frozen=True)
class AppPaths:
    """Paths owned by this local-only application.

    ``data_dir`` is deliberately configurable so users can keep private biometric
    data outside a cloned repository.  The default remains inside this project
    for a straightforward internship demonstration.
    """

    root: Path = PROJECT_ROOT
    data_dir: Path | None = None
    output_dir: Path | None = None
    models_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "data_dir", Path(self.data_dir or self.root / "data").resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir or self.root / "outputs").resolve())
        object.__setattr__(self, "models_dir", Path(self.models_dir or self.root / "models").resolve())

    @property
    def registered_faces_dir(self) -> Path:
        return self.data_dir / "registered_faces"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def database_dir(self) -> Path:
        return self.embeddings_dir / "database"

    @property
    def manifest_path(self) -> Path:
        return self.database_dir / "manifest.json"

    @property
    def generations_dir(self) -> Path:
        return self.database_dir / "generations"

    @property
    def yunet_model_path(self) -> Path:
        return self.models_dir / "face_detection_yunet_2026may.onnx"

    @property
    def sface_model_path(self) -> Path:
        return self.models_dir / "face_recognition_sface_2021dec.onnx"

    def ensure_runtime_directories(self) -> None:
        for directory in (
            self.models_dir,
            self.registered_faces_dir,
            self.embeddings_dir,
            self.database_dir,
            self.generations_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
