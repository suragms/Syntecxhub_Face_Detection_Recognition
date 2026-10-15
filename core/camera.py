"""Shared, defensive OpenCV camera setup for Windows capture workflows."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from typing import Iterable

import cv2
import numpy as np

from config import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH
from core.errors import CameraOpenError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    """Capture policy shared by enrollment and live recognition."""

    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    backend_names: tuple[str, ...] = ("dshow", "msmf", "any")
    warmup_attempts: int = 6
    read_retries: int = 2

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CameraOpenError("Camera width and height must both be positive.")
        if self.warmup_attempts < 1:
            raise CameraOpenError("Camera warm-up attempts must be at least one.")
        if self.read_retries < 0:
            raise CameraOpenError("Camera read retries cannot be negative.")


class CameraSession:
    """An opened camera with a buffered initial valid frame and safe cleanup."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        camera_index: int,
        backend_name: str,
        config: CameraConfig,
        initial_frame: np.ndarray,
    ) -> None:
        self._capture = capture
        self.camera_index = camera_index
        self.backend_name = backend_name
        self.config = config
        self._initial_frames: deque[np.ndarray] = deque([initial_frame])
        self._released = False
        self._last_signature: tuple[tuple[int, ...], int, int] | None = None
        self._identical_frame_count = 0
        self._stale_warning_emitted = False

    @property
    def actual_size(self) -> tuple[int, int]:
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray:
        """Return one structurally valid OpenCV frame or raise a clear error."""
        if self._released:
            raise CameraOpenError("The camera session has already been released.")
        if self._initial_frames:
            frame = self._initial_frames.popleft()
            self._observe_frame(frame)
            return frame

        failures: list[str] = []
        for attempt in range(self.config.read_retries + 1):
            ok, frame = self._capture.read()
            if _is_valid_frame(ok, frame):
                self._observe_frame(frame)
                return frame
            failures.append(_frame_failure_reason(ok, frame))
            LOGGER.warning(
                "Camera index %s (%s) returned an invalid frame on read attempt %s/%s: %s",
                self.camera_index,
                self.backend_name,
                attempt + 1,
                self.config.read_retries + 1,
                failures[-1],
            )
        raise CameraOpenError(
            f"Camera index {self.camera_index} using {self.backend_name} stopped returning valid frames "
            f"after {len(failures)} attempt(s): {'; '.join(failures)}."
        )

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._capture.release()
        except cv2.error as exc:
            LOGGER.warning("Camera release failed for index %s (%s): %s", self.camera_index, self.backend_name, exc)

    def _observe_frame(self, frame: np.ndarray) -> None:
        # A dark or static view can be legitimate. This inexpensive signature is
        # diagnostic only; it must not reject valid user frames or switch backend.
        signature = (frame.shape, int(frame.flat[0]), int(frame.flat[-1]))
        if signature == self._last_signature:
            self._identical_frame_count += 1
        else:
            self._identical_frame_count = 0
            self._last_signature = signature
        if self._identical_frame_count >= 30 and not self._stale_warning_emitted:
            LOGGER.warning(
                "Camera index %s (%s) has produced similar valid frame signatures repeatedly. "
                "The preview remains active because a dark or static scene may be legitimate.",
                self.camera_index,
                self.backend_name,
            )
            self._stale_warning_emitted = True


def open_camera(camera_index: int, config: CameraConfig | None = None) -> CameraSession:
    """Open a camera backend that can deliver a valid frame after warm-up.

    An all-black image is structurally valid and remains visible. Only malformed,
    empty, or absent frames cause a backend candidate to be rejected.
    """
    camera_config = config or CameraConfig()
    failures: list[str] = []
    for backend_name, backend_value in _backend_candidates(camera_config.backend_names):
        LOGGER.info("Opening camera index %s with OpenCV backend %s", camera_index, backend_name)
        capture = _create_capture(camera_index, backend_value)
        owns_capture = False
        try:
            if not capture.isOpened():
                failures.append(f"{backend_name}: did not open")
                LOGGER.warning("Camera index %s did not open with %s", camera_index, backend_name)
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            LOGGER.info(
                "Camera index %s opened with %s; requested %sx%s, driver reports %sx%s",
                camera_index,
                backend_name,
                camera_config.width,
                camera_config.height,
                actual_width,
                actual_height,
            )
            initial_frame, warmup_failures = _warm_up_capture(capture, camera_config.warmup_attempts)
            if initial_frame is not None:
                LOGGER.info("Camera index %s is ready using %s", camera_index, backend_name)
                owns_capture = True
                return CameraSession(capture, camera_index, backend_name, camera_config, initial_frame)
            failures.append(f"{backend_name}: no valid warm-up frame ({'; '.join(warmup_failures)})")
            LOGGER.warning("Camera index %s produced no valid warm-up frame with %s", camera_index, backend_name)
        except cv2.error as exc:
            failures.append(f"{backend_name}: OpenCV error ({exc})")
            LOGGER.warning("Camera index %s failed using %s: %s", camera_index, backend_name, exc)
        finally:
            # The successful session owns its handle. Every other candidate is
            # released before another Windows backend attempts the same device.
            if not owns_capture:
                _safe_release(capture)
        # A returned session exits the function before this point.

    attempted = ", ".join(failures) or "no available capture backend"
    raise CameraOpenError(
        f"Unable to open camera index {camera_index}. Tried Windows camera backends: {attempted}. "
        "Check camera permissions, privacy shutter, cable, device index, and whether another app is using the camera."
    )


def safe_destroy_window(window_name: str) -> None:
    """Best-effort HighGUI cleanup that cannot hide the original failure."""
    try:
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        LOGGER.debug("OpenCV window '%s' was already unavailable during cleanup: %s", window_name, exc)


def _create_capture(camera_index: int, backend_value: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(camera_index, backend_value)


def _backend_candidates(names: Iterable[str]) -> list[tuple[str, int]]:
    mapping = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "any": getattr(cv2, "CAP_ANY", 0),
    }
    candidates: list[tuple[str, int]] = []
    used: set[int] = set()
    for requested in names:
        name = requested.casefold()
        value = mapping.get(name)
        if value is not None and value not in used:
            candidates.append((name.upper(), int(value)))
            used.add(int(value))
    if not candidates:
        candidates.append(("ANY", int(getattr(cv2, "CAP_ANY", 0))))
    return candidates


def _warm_up_capture(capture: cv2.VideoCapture, attempts: int) -> tuple[np.ndarray | None, list[str]]:
    failures: list[str] = []
    for _ in range(attempts):
        ok, frame = capture.read()
        if _is_valid_frame(ok, frame):
            return frame, failures
        failures.append(_frame_failure_reason(ok, frame))
    return None, failures


def _is_valid_frame(ok: bool, frame: object) -> bool:
    if not ok or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    if frame.ndim == 2:
        return frame.shape[0] > 0 and frame.shape[1] > 0
    return frame.ndim == 3 and frame.shape[0] > 0 and frame.shape[1] > 0 and frame.shape[2] in (1, 3, 4)


def _frame_failure_reason(ok: bool, frame: object) -> str:
    if not ok:
        return "read returned False"
    if frame is None:
        return "frame is None"
    if not isinstance(frame, np.ndarray):
        return f"frame type is {type(frame).__name__}"
    if frame.size == 0:
        return "frame is empty"
    return f"unsupported frame shape {frame.shape}"


def _safe_release(capture: cv2.VideoCapture) -> None:
    try:
        capture.release()
    except cv2.error as exc:
        LOGGER.debug("OpenCV capture release failed during backend fallback: %s", exc)

