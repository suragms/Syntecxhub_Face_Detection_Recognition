from __future__ import annotations

import cv2
import numpy as np
import pytest

from core import camera
from core.camera import CameraConfig, CameraSession, open_camera, safe_destroy_window
from core.errors import CameraOpenError


class FakeCapture:
    def __init__(self, frames, opened=True, width=640, height=480):
        self.frames = list(frames)
        self.opened = opened
        self.width = width
        self.height = height
        self.set_calls = []
        self.release_calls = 0

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.set_calls.append((property_id, value))
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            self.width = value
        elif property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = value
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        return False, None

    def release(self):
        self.release_calls += 1


def test_open_camera_tries_next_backend_after_invalid_warmup(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dshow = FakeCapture([(False, None), (True, np.empty((0, 0, 3), dtype=np.uint8))])
    msmf = FakeCapture([(True, frame)])
    captures = {int(getattr(cv2, "CAP_DSHOW")): dshow, int(getattr(cv2, "CAP_MSMF")): msmf}
    attempts = []

    def fake_create(index, backend):
        attempts.append(backend)
        return captures[backend]

    monkeypatch.setattr(camera, "_create_capture", fake_create)
    session = open_camera(0, CameraConfig(width=640, height=480, warmup_attempts=2))
    try:
        assert attempts[:2] == [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        assert dshow.release_calls == 1
        assert session.backend_name == "MSMF"
        assert np.array_equal(session.read(), frame)
        assert (cv2.CAP_PROP_FRAME_WIDTH, 640) in msmf.set_calls
        assert (cv2.CAP_PROP_FRAME_HEIGHT, 480) in msmf.set_calls
    finally:
        session.release()
    assert msmf.release_calls == 1


def test_dark_valid_frame_is_accepted(monkeypatch):
    dark = np.zeros((32, 48, 3), dtype=np.uint8)
    capture = FakeCapture([(True, dark)])
    monkeypatch.setattr(camera, "_create_capture", lambda index, backend: capture)
    session = open_camera(0, CameraConfig(backend_names=("dshow",), warmup_attempts=1))
    try:
        assert np.array_equal(session.read(), dark)
    finally:
        session.release()


def test_all_backends_are_released_and_reported(monkeypatch):
    failed = [FakeCapture([], opened=False) for _ in range(3)]
    iterator = iter(failed)
    monkeypatch.setattr(camera, "_create_capture", lambda index, backend: next(iterator))
    with pytest.raises(CameraOpenError, match="Unable to open camera index 4"):
        open_camera(4, CameraConfig())
    assert all(item.release_calls == 1 for item in failed)


def test_active_invalid_reads_are_bounded():
    capture = FakeCapture([(True, np.zeros((8, 8, 2), dtype=np.uint8))])
    session = CameraSession(capture, 0, "TEST", CameraConfig(warmup_attempts=1, read_retries=1), np.zeros((8, 8, 3), dtype=np.uint8))
    session.read()  # consume the buffered warm-up frame
    with pytest.raises(CameraOpenError, match="stopped returning valid frames"):
        session.read()


def test_safe_destroy_window_does_not_raise(monkeypatch):
    def fail(_name):
        raise cv2.error("window already closed")

    monkeypatch.setattr(cv2, "destroyWindow", fail)
    safe_destroy_window("missing")
