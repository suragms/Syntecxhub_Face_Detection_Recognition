"""Truthful OpenCV annotations shared by image, video, and webcam modes."""
from __future__ import annotations

import cv2
import numpy as np

from core.face_recognizer import RecognitionResult


COLORS = {
    "recognized": (40, 190, 40),
    "unknown": (0, 165, 255),
    "detection_only": (0, 0, 255),
}


def annotate_frame(frame: np.ndarray, results: list[RecognitionResult], fps: float | None = None) -> np.ndarray:
    """Return an annotated copy without changing recognition decisions."""
    output = frame.copy()
    for result in results:
        x, y, width, height = result.detection.bbox
        color = COLORS[result.status]
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        if result.status == "recognized":
            label = f"{result.display_name}  score {result.similarity:.3f}"
        elif result.status == "unknown":
            label = f"Unknown  score {result.similarity:.3f}" if result.similarity is not None else "Unknown"
        else:
            label = "Detected only (Haar; no landmarks)"
        baseline_y = y - 10 if y > 25 else y + height + 20
        cv2.putText(output, label, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(output, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return output
