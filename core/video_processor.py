"""Frame-by-frame processing of local recorded videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from core.errors import MediaError
from core.image_processor import ImageProcessor


@dataclass(frozen=True)
class VideoProcessingSummary:
    input_path: Path
    output_path: Path
    frames_processed: int
    faces_detected: int


class VideoProcessor:
    def __init__(self, image_processor: ImageProcessor) -> None:
        self.image_processor = image_processor

    def process(self, input_path: str | Path, output_path: str | Path, display: bool = False) -> VideoProcessingSummary:
        source = Path(input_path)
        target = Path(output_path)
        if not source.is_file():
            raise MediaError(f"Video does not exist: {source}")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise MediaError(f"OpenCV could not open video: {source}")
        writer = None
        frames = 0
        faces = 0
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if width <= 0 or height <= 0:
                raise MediaError("The input video has invalid frame dimensions.")
            if not fps or fps != fps or fps < 1:
                fps = 30.0
            target.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise MediaError(
                    f"Could not create video output '{target}' using mp4v. "
                    "Install a compatible Windows video codec or choose another output location."
                )
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, summary = self.image_processor.process_frame(frame)
                writer.write(annotated)
                frames += 1
                faces += summary.detections
                if display:
                    cv2.imshow("Syntecxhub video recognition", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyWindow("Syntecxhub video recognition")
        if frames == 0:
            raise MediaError("The input video contained no decodable frames.")
        return VideoProcessingSummary(source, target, frames, faces)
