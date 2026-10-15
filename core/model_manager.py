"""Official local model metadata, download, integrity, and OpenCV checks."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2

from config import AppPaths
from core.errors import ModelIntegrityError, ModelMissingError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    source_page: str
    license_name: str
    sha256_digest: str | None = None


# Direct official OpenCV Zoo media endpoints. SHA-256 values were calculated
# from the official artifacts downloaded on 2026-10-15 and make later local
# verification meaningful. Update URL and digest together after upstream model
# changes, then document the provenance update.
YUNET = ModelSpec(
    key="yunet",
    filename="face_detection_yunet_2026may.onnx",
    url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
    source_page="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    license_name="MIT",
    sha256_digest="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
)
SFACE = ModelSpec(
    key="sface",
    filename="face_recognition_sface_2021dec.onnx",
    url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    source_page="https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface",
    license_name="Apache-2.0",
    sha256_digest="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
MODEL_SPECS = {spec.key: spec for spec in (YUNET, SFACE)}


class ModelManager:
    """Manage models without silently accepting incomplete artifacts."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def path_for(self, spec: ModelSpec) -> Path:
        return self.paths.models_dir / spec.filename

    def model_status(self, spec: ModelSpec) -> dict[str, object]:
        path = self.path_for(spec)
        status: dict[str, object] = {
            "key": spec.key,
            "path": str(path),
            "exists": path.is_file(),
            "verified": False,
            "sha256": None,
            "license": spec.license_name,
        }
        if path.is_file():
            digest = _hash_file(path)
            status["sha256"] = digest
            status["verified"] = spec.sha256_digest is None or digest == spec.sha256_digest
            status["size_bytes"] = path.stat().st_size
        return status

    def verify(self, spec: ModelSpec) -> str:
        path = self.path_for(spec)
        if not path.is_file():
            raise ModelMissingError(f"Model is missing: {path}")
        if path.stat().st_size == 0:
            raise ModelIntegrityError(f"Model is empty: {path}")
        digest = _hash_file(path)
        if spec.sha256_digest and digest != spec.sha256_digest:
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {spec.filename}. Expected {spec.sha256_digest}, got {digest}."
            )
        return digest

    def download(self, spec: ModelSpec, force: bool = False, timeout: float = 30.0) -> Path:
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(spec)
        if destination.exists() and not force:
            self.verify(spec)
            return destination

        descriptor, temp_name = tempfile.mkstemp(prefix=f".{spec.key}-", suffix=".download", dir=self.paths.models_dir)
        temporary = Path(temp_name)
        digest = sha256()
        byte_count = 0
        try:
            with os.fdopen(descriptor, "wb") as target:
                request = Request(spec.url, headers={"User-Agent": "Syntecxhub-Face-Recognition/1.0"})
                try:
                    with urlopen(request, timeout=timeout) as response:
                        while chunk := response.read(1024 * 1024):
                            target.write(chunk)
                            digest.update(chunk)
                            byte_count += len(chunk)
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    raise ModelIntegrityError(f"Could not download {spec.filename}: {exc}") from exc
                target.flush()
                os.fsync(target.fileno())
            if byte_count < 1024:
                raise ModelIntegrityError(f"Downloaded {spec.filename} is implausibly small ({byte_count} bytes).")
            first_bytes = temporary.read_bytes()[:200]
            if b"git-lfs" in first_bytes or b"version https://git-lfs.github.com/spec" in first_bytes:
                raise ModelIntegrityError(f"Downloaded {spec.filename} is a Git LFS pointer, not an ONNX model.")
            observed = digest.hexdigest()
            if spec.sha256_digest and observed != spec.sha256_digest:
                raise ModelIntegrityError(
                    f"SHA-256 mismatch for {spec.filename}. Expected {spec.sha256_digest}, got {observed}."
                )
            os.replace(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    def require_recognition_models(self) -> tuple[Path, Path]:
        yunet = self.path_for(YUNET)
        sface = self.path_for(SFACE)
        self.verify(YUNET)
        self.verify(SFACE)
        check_opencv_features(require_recognition=True)
        return yunet, sface


def check_opencv_features(require_recognition: bool = False) -> None:
    missing = []
    if not hasattr(cv2, "FaceDetectorYN"):
        missing.append("cv2.FaceDetectorYN")
    if require_recognition and not hasattr(cv2, "FaceRecognizerSF"):
        missing.append("cv2.FaceRecognizerSF")
    if missing:
        raise ModelIntegrityError(
            "The installed OpenCV package lacks " + ", ".join(missing) + ". "
            "Install exactly one compatible opencv-contrib-python wheel."
        )


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
