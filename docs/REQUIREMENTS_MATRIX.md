# Official requirement traceability

| Requirement | Implementation | Verification | Status |
| --- | --- | --- | --- |
| 1. OpenCV detects faces in images and videos | `core/face_detector.py`, `core/image_processor.py`, `core/video_processor.py`; YuNet primary and Haar fallback | `tests/test_detector.py`, image/video manual checklist | Implemented in code; model/media manual run depends on local model/media availability |
| 2. Facial embeddings identify known faces | `core/face_recognizer.py` uses SFace `alignCrop` and `feature`, then cosine matching | `tests/test_recognizer.py`; consented enrollment/recognition manual test | Implemented in code; requires verified SFace/YuNet models for live verification |
| 3. Boxes/labels support multiple faces | `core/rendering.py`; processors iterate every detection | `tests/test_processors.py`; multi-face manual test | Implemented and unit-tested with mocked detections |
| 4. Capture/register/recognize new people | `core/face_registration.py`, `scripts/register_person.py`, `main.py --register`; local database | `tests/test_registration.py`, webcam/photo manual checklist | Implemented in code; webcam path requires camera and models |

## Additional objectives

- Local JSON/NPZ persistence, UUID identities, list/delete, retained-source embedding rebuild, and recovery are implemented in `core/face_database.py`.
- Model download/status/integrity is implemented in `core/model_manager.py` and `scripts/download_models.py`.
- Generated private photos, embeddings, models, and outputs are Git-ignored.
- Similarity scores are displayed as scores, not probabilities; Haar output is never represented as recognition.

## Remaining limitations

No benchmark accuracy claim is made. The default 0.363 threshold is an OpenCV tutorial baseline and needs context-specific calibration. Physical camera, model, codec, and Windows UI validation must be performed on the target machine and reported honestly.
