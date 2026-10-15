# Syntecxhub Face Detection & Recognition

> **Syntecxhub AI Internship — Task 3, Project 1**  
> A local, consent-based Python/OpenCV application for face detection and embedding-based recognition.

## Project overview

This application detects all faces in an image, recorded video, or live webcam frame with **OpenCV YuNet**, then recognizes previously enrolled consenting people with landmark-aligned **OpenCV SFace** embeddings. It is a real computer-vision pipeline—not a static-image demonstration or simulated identity output.

**Pipeline:** input frame → YuNet detection + five landmarks → SFace alignment → 128-value embedding → cosine comparison against a private local database → recognized name or `Unknown`.

A Haar cascade is available only as an explicit automatic detector fallback when YuNet cannot load. Haar has no compatible landmarks, so those detections are truthfully marked **Detected only** and are never passed into SFace.

## Official task requirements

| Mandatory requirement | Delivered implementation |
| --- | --- |
| OpenCV detects faces in images and videos | `core/face_detector.py`, `core/image_processor.py`, `core/video_processor.py`, `core/webcam_processor.py` |
| Facial recognition identifies known faces | `core/face_recognizer.py` creates real SFace embeddings and compares cosine similarity against local samples |
| Bounding boxes and labels support multiple people | Shared `core/rendering.py` processes every detection in each frame |
| Capture/register/recognize new people | `core/face_registration.py`, `scripts/register_person.py`, and CLI modes |

Read the detailed, current status matrix in [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md).

## Features

- YuNet multi-face detection with boxes, confidence, and five landmarks.
- SFace aligned embeddings, multi-sample matching per person, configurable cosine threshold, and `Unknown` rejection.
- Local private enrollment through JPG/JPEG/PNG files or a webcam.
- Image, recorded-video, and real-time webcam processing; annotated image/video/screenshot saving.
- Multiple-face support; actual rolling webcam FPS; `Q` to quit and `S` to save a screenshot.
- Snapshot-based local JSON + NPZ database with UUID identities, deletion, and previous-generation recovery.
- Model downloader and verification using official OpenCV Zoo sources.
- Haar fallback that never misrepresents detection as recognition.
- Pytest tests using mocks/NumPy frames—no webcam required.

## Technology stack

- Python 3.11
- OpenCV (`opencv-contrib-python`; install **one** OpenCV wheel flavor only)
- NumPy
- pytest
- Official OpenCV Zoo YuNet and SFace ONNX models

## Architecture

```text
main.py
  ├─ config.py
  └─ core/
      ├─ model_manager.py      official local model installation/status
      ├─ face_detector.py      YuNet primary detector / Haar detection-only fallback
      ├─ face_recognizer.py    landmark alignment, embeddings, cosine matching
      ├─ face_database.py      local private JSON/NPZ snapshot storage
      ├─ face_registration.py  webcam/file enrollment and quality checks
      ├─ image_processor.py    still-image pipeline
      ├─ video_processor.py    recorded-video pipeline
      ├─ webcam_processor.py   live-camera pipeline
      └─ rendering.py          truthful annotations
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Repository structure

```text
├── main.py
├── config.py
├── core/
├── scripts/
├── models/                 # Model documentation; ONNX files remain local/ignored
├── data/
│   ├── registered_faces/   # Optional aligned enrollment crops; ignored
│   └── embeddings/         # Local JSON/NPZ biometric data; ignored
├── outputs/                # Generated annotations; ignored
├── tests/
├── docs/
├── requirements.txt
└── .gitignore
```

## Prerequisites

- Windows 10 or Windows 11
- Python **3.11** (`python --version`)
- A webcam only for live recognition/enrollment
- Internet access only once to download the two models
- Microsoft Visual C++ 2015+ Redistributable if `import cv2` fails
- On Windows N/KN, install the Media Feature Pack if video codecs are unavailable

Do **not** combine `opencv-python`, `opencv-contrib-python`, and any `*-headless` package in the same environment. This project requires UI windows for webcam enrollment/recognition.

## Installation

### Windows PowerShell

```bash
python --version
```

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_models.py
```

```bash
python main.py --help
```

### Windows Command Prompt

```bash
python --version
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate.bat
```

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_models.py
```

```bash
python main.py --help
```

If PowerShell blocks activation, use Command Prompt or apply your organization’s approved execution policy. Do not bypass organization security rules.

## Models

```bash
python scripts/download_models.py
```

```bash
python main.py --models status
```

```bash
python main.py --models verify
```

Model files operate offline after installation. Their source, licenses, downloader behavior, and manual-placement guidance are in [models/README.md](models/README.md).

## Register a consenting person

### Webcam (recommended: 10 samples)

```bash
python scripts/register_person.py --name "Person Name" --camera 0 --samples 10
```

- Position exactly one consenting person in view.
- When quality feedback says capture is available, press **Space** for each sample.
- Press **Q** or **Esc** to cancel.
- The capture only accepts YuNet face detections with valid landmarks; it rejects zero/multiple faces and blurred/small samples.

### Existing photograph

```bash
python main.py --register image --name "Person Name" --input C:\photos\person.jpg
```

Only a JPG, JPEG, or PNG containing exactly one usable YuNet face is accepted.

### Add a sample to an existing person

```bash
python main.py --register image --name "Person Name" --input C:\photos\another.jpg --append
```

For privacy, use `--no-retain-source` if you do not want to retain the aligned crop required for a future source-based rebuild.

## Recognition usage

### Webcam

```bash
python main.py --source webcam --camera 0
```

Press **Q** to stop; press **S** to save an annotated screenshot in `outputs/`.

### Image

```bash
python main.py --source image --input sample.jpg --output outputs\annotated_sample.jpg
```

It reports detection, recognized, unknown, and detection-only counts. If no output is supplied, it returns the summary without writing an image.

### Video

```bash
python main.py --source video --input sample.mp4 --output outputs\annotated_sample.mp4
```

Use `--display` to preview while processing. The program validates that the selected MP4 writer opens rather than silently creating a corrupted file.

### Management

```bash
python scripts/list_people.py
```

```bash
python main.py --delete "PERSON_ID" --yes
```

```bash
python scripts/build_embeddings.py
```

`--rebuild` recomputes embeddings from retained local aligned crops, validates the result, and writes a fresh storage generation. It skips samples registered with `--no-retain-source`; use `--strict` to make any skipped/missing source a failure. Use `--verify-only` to validate the current snapshot without writing.

## Command-line reference

```bash
python main.py --help
```

The main examples required for the task are supported:

```bash
python main.py --source webcam
```

```bash
python main.py --source webcam --camera 0
```

```bash
python main.py --source image --input sample.jpg
```

```bash
python main.py --source video --input sample.mp4 --output outputs\annotated_sample.mp4
```

```bash
python scripts/register_person.py --name "Person Name"
```

```bash
python scripts/build_embeddings.py
```

```bash
python scripts/list_people.py
```

## Testing

```bash
pytest -q
```

The tests mock cameras, model APIs, and video writers; they do not require installed models, a webcam, network access, face photos, or a display. The exact latest execution status belongs in [docs/TESTING.md](docs/TESTING.md); never treat the manual checklist as automated verification.

## Troubleshooting

| Problem | Action |
| --- | --- |
| `FaceRecognizerSF`/`FaceDetectorYN` missing | Activate the correct virtual environment; uninstall conflicting OpenCV wheels; reinstall the declared `opencv-contrib-python`. |
| ONNX model missing | Run `python scripts/download_models.py` and then `python main.py --models verify`. |
| Camera will not open | Close apps using the camera; permit desktop camera access in Windows Settings; try `--camera 1`. |
| No/multiple face enrollment error | Use one well-lit, front-facing consenting person; avoid tiny, blurry, profile, or obstructed faces. |
| Video output cannot open | Choose a writable `.mp4` path; install Windows media support/codecs; the program intentionally stops rather than writing corrupt output. |
| Database error | Keep the `data/` directory intact. The snapshot manifest can fall back to its previous intact generation after an interrupted write. |

## Privacy, consent, and limitations

- Use photographs and video only with the person’s informed consent.
- This project does not scrape images, record secretly, upload data, or contact remote identity services.
- Names, aligned enrollment crops (unless disabled), and embeddings are stored locally in `data/` and excluded from Git.
- Delete a registration with `--delete ID --yes`; delete the local data directory to remove all local records.
- Similarity scores are **not probabilities**, accuracy claims, identity proof, authorization decisions, or a security guarantee.
- Performance varies with lighting, pose, occlusion, image quality, camera hardware, and threshold selection. The baseline 0.363 cosine threshold needs consented, context-specific calibration.
- YuNet/SFace do not make this safe for surveillance, access control, or high-impact decisions.

## Optional GUI

`app.py` is intentionally an honest CLI-first placeholder: the optional Tkinter interface is deferred until a real, responsive GUI can be manually verified. All mandatory internship functionality is available now through `main.py`; no recognition logic is duplicated in the placeholder.

## Future enhancements

- A Tkinter desktop UI that calls the same tested `core` services.
- Source-image-based embedding recomputation with explicit retention choices.
- Configurable video codec fallbacks after platform testing.
- Formal, consented accuracy/fairness evaluation and threshold calibration—not fabricated benchmarks.

## Developer information and internship attribution

Built for **Syntecxhub AI Internship – Task 3: Face Detection & Recognition**. The implementation emphasizes transparent local processing, safe error handling, real embedding matching, and reproducibility on Windows.

## License

This project source is released under the [MIT License](LICENSE). The downloaded YuNet and SFace models have their own upstream MIT and Apache-2.0 terms, respectively; see [models/README.md](models/README.md).
