# Model installation and provenance

This directory is intentionally empty in Git. The application downloads two official OpenCV Zoo ONNX models locally; model binaries are ignored because they are generated/downloaded assets.

```bash
python scripts/download_models.py
```

| Model | File | Purpose | Official source | License |
| --- | --- | --- | --- | --- |
| YuNet | `face_detection_yunet_2026may.onnx` | Face detection and five landmarks | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> | MIT |
| SFace | `face_recognition_sface_2021dec.onnx` | Aligned face embeddings | <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> | Apache-2.0 |

The downloader uses direct `media.githubusercontent.com` URLs from those official OpenCV Zoo paths, writes to a temporary file, rejects empty/Git-LFS-pointer responses, verifies the following SHA-256 values, and replaces the final model only after the download completes:

- YuNet 2026may: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
- SFace 2021dec: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

Verify installed models with:

```bash
python main.py --models verify
```

The model files work offline after installation. Preserve YuNet's MIT notice and SFace's Apache-2.0 notice if redistributing their ONNX artifacts.

> Recognition requires YuNet landmarks and SFace. The Haar cascade bundled with OpenCV is a detection-only fallback and cannot make valid SFace embeddings.
