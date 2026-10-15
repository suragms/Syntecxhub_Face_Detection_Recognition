"""Interactive real-time face recognition using an OpenCV webcam window."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from config import AppPaths
from core.errors import CameraOpenError
from core.image_processor import ImageProcessor
from core.rendering import annotate_frame


@dataclass(frozen=True)
class WebcamSummary:
    frames_processed: int
    screenshots_saved: int


class WebcamProcessor:
    def __init__(self, image_processor: ImageProcessor, paths: AppPaths, width: int = 640, height: int = 480) -> None:
        self.image_processor = image_processor
        self.paths = paths
        self.width = width
        self.height = height

    def run(self, camera_index: int = 0) -> WebcamSummary:
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise CameraOpenError(
                f"Unable to open camera index {camera_index}. Check camera permissions, availability, and index."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        frames = 0
        screenshots = 0
        last_tick = time.perf_counter()
        fps = 0.0
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise CameraOpenError("Camera disconnected or stopped returning frames.")
                _, summary = self.image_processor.process_frame(frame)
                now = time.perf_counter()
                delta = max(now - last_tick, 1e-9)
                fps = 0.9 * fps + 0.1 * (1.0 / delta) if frames else 1.0 / delta
                last_tick = now
                annotated = annotate_frame(frame, list(summary.results), fps=fps)
                cv2.putText(annotated, "Q: quit   S: screenshot", (12, annotated.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("Syntecxhub Face Detection & Recognition", annotated)
                frames += 1
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    target = self.paths.output_dir / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    if not cv2.imwrite(str(target), annotated):
                        raise CameraOpenError(f"Could not save screenshot: {target}")
                    screenshots += 1
        finally:
            capture.release()
            cv2.destroyWindow("Syntecxhub Face Detection & Recognition")
        return WebcamSummary(frames_processed=frames, screenshots_saved=screenshots)
