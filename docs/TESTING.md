# Testing

## Automated tests

Run from the repository root after installing requirements:

```bash
pytest -q
```

Tests use temporary directories, synthetic NumPy arrays, and fakes for detector/recognizer/media boundaries. They do not need a webcam, ONNX model, network connection, real face photograph, or GUI display.

Coverage includes:

- YuNet row validation and multiple-face contracts.
- Haar landmark absence and SFace gating.
- Embedding dimension/normalization, cosine matching, best-sample identity selection, and unknown rejection.
- Database persistence, duplicate names, deletion, generation fallback, and embedding alignment.
- Registration no-face/multiple-face rejection and local persistence.
- Image input validation and multi-face summary.
- CLI parser/error behavior and model hash verification.

## Manual Windows checklist

Run only with informed consent and local test media:

1. Register one consenting person from a webcam.
2. Recognize that person in a webcam frame.
3. Register at least two people and verify both labels in one frame.
4. Show a registered and unregistered person together; verify the latter is `Unknown`.
5. Show an unknown person only.
6. Process an image containing no face.
7. Disconnect/close the camera and verify an actionable error.
8. Restart the application and verify the stored identity remains available.
9. Delete an identity and verify it is no longer matched.
10. Process and save an annotated video.

Manual checks requiring models, a camera, and media are not represented as automated test passes. Record actual observations before submission.
