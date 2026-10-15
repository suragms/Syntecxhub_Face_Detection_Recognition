# Architecture

The application is deliberately layered so detection, recognition, persistence, enrollment, and media I/O can be tested independently.

## Detection and recognition boundary

`core/face_detector.py` returns `FaceDetection` objects. YuNet rows contain bounding box, confidence, five landmarks, and the untouched 15-value row required by OpenCV SFace alignment. Haar results contain only a box and are marked `detector="haar"`.

`core/face_recognizer.py` is the only component that calls `FaceRecognizerSF.alignCrop` and `feature`. It refuses a detection without YuNet landmarks. A Haar fallback therefore remains honest detection-only output, never a fake embedding or identity.

## Data flow

```text
image/video/webcam frame
        ↓
FaceDetector (YuNet, Haar fallback only if YuNet unavailable)
        ↓
FaceDetection[]
        ↓ (YuNet detections only)
SFace alignCrop → feature → normalized 128-value vector
        ↓
FaceDatabase.load_embeddings()
        ↓
best cosine score per identity → recognized name or Unknown
        ↓
shared renderer → box, label, score, output media
```

## Persistence

Registration creates a new immutable JSON/NPZ generation. `manifest.json` is atomically replaced only after both artifacts exist. It points to the active and previous generation. Readers validate dimensions, IDs, metadata, finite values, and cardinality; if the active generation is damaged, the previous generation is used. `--rebuild` recomputes embeddings from retained local aligned crops only when an image yields exactly one YuNet face; it never fabricates a replacement for an unavailable source. Embeddings, names, and source crops remain local and are Git-ignored.

## Enrollment

`FaceRegistrationService` is shared by photo and webcam workflows. It requires exactly one YuNet detection, checks size/sharpness/confidence, creates an SFace embedding, optionally retains an aligned crop, and commits only a valid sample. Names are validated and UUIDs are used for internal identity IDs.

## GUI scope

The initial deliverable is a tested CLI and backend. A Tkinter UI is intentionally deferred so it cannot duplicate model or persistence logic; a future UI must call these same services.
