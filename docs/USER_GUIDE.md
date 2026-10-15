# User guide

## Register

Use `python scripts/register_person.py --name "Name" --camera 0 --samples 10`, then press Space for quality-approved frames. Or use `python main.py --register image --name "Name" --input path\to\photo.jpg`.

Registration requires exactly one YuNet face and SFace-compatible landmarks. Use `--append` to add a sample to an existing display name. Use `--no-retain-source` to avoid retaining an aligned crop.

## Recognize

- Webcam: `python main.py --source webcam --camera 0` (Q quits, S saves a screenshot).
- Image: `python main.py --source image --input sample.jpg --output outputs\annotated.jpg`.
- Video: `python main.py --source video --input sample.mp4 --output outputs\annotated.mp4`.

Green boxes are recognized people and include a cosine similarity score. Orange boxes are `Unknown`; red boxes indicate Haar detection-only output, never a claimed identity.

## Manage records

`python scripts/list_people.py` lists UUID, name, and sample count. Delete with `python main.py --delete PERSON_ID --yes`. Run `python scripts/build_embeddings.py` to recompute embeddings from retained local crops and write a database generation. Add `--strict` to reject skipped/missing sources, or `--verify-only` to validate without writing.

## Threshold

The default cosine threshold is 0.363, based on the OpenCV SFace tutorial baseline. It is a similarity cutoff, not an identity probability. Adjust it only after testing on consented data representative of the intended camera and environment.
