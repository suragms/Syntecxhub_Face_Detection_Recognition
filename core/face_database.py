"""Private, local persistence for face embeddings and registration metadata."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterable
from uuid import uuid4

import numpy as np

from config import AppPaths
from core.errors import AmbiguousPersonError, DatabaseError, DuplicatePersonError
from core.face_recognizer import FaceRecognizer, StoredEmbedding

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    captured_at: str
    quality_score: float
    source_file: str | None = None
    rebuildable: bool = False


@dataclass(frozen=True)
class Person:
    person_id: str
    display_name: str
    created_at: str
    updated_at: str
    samples: tuple[SampleMetadata, ...] = field(default_factory=tuple)

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class PersonSummary:
    person_id: str
    display_name: str
    sample_count: int
    created_at: str
    updated_at: str
    rebuildable_samples: int


class FaceDatabase:
    """Snapshot-based storage that avoids in-place metadata/embedding updates.

    Each commit writes a fresh JSON + NPZ generation and switches the manifest
    last. A crash before that point leaves the previous complete generation as
    the reader-visible database.
    """

    def __init__(self, paths: AppPaths, lock_timeout_seconds: float = 5.0) -> None:
        self.paths = paths
        self.lock_timeout_seconds = lock_timeout_seconds
        self.paths.ensure_runtime_directories()

    def list_people(self) -> list[PersonSummary]:
        people, _ = self._load_active()
        return [
            PersonSummary(
                person_id=person.person_id,
                display_name=person.display_name,
                sample_count=person.sample_count,
                created_at=person.created_at,
                updated_at=person.updated_at,
                rebuildable_samples=sum(sample.rebuildable for sample in person.samples),
            )
            for person in sorted(people, key=lambda value: (value.display_name.casefold(), value.person_id))
        ]

    def get_person(self, person_id_or_name: str) -> Person | None:
        people, _ = self._load_active()
        matches = self._find_people(people, person_id_or_name)
        if len(matches) > 1:
            raise AmbiguousPersonError(
                f"More than one registered person is named '{person_id_or_name}'. Use a person ID."
            )
        return matches[0] if matches else None

    def add_person(
        self,
        display_name: str,
        embedding: np.ndarray,
        quality_score: float,
        source_file: str | None = None,
        allow_existing: bool = False,
    ) -> tuple[Person, SampleMetadata]:
        name = _validate_display_name(display_name)
        vector = FaceRecognizer.normalize_embedding(embedding)
        with self._write_lock():
            people, vectors = self._load_active()
            matches = [person for person in people if person.display_name.casefold() == name.casefold()]
            if matches and not allow_existing:
                raise DuplicatePersonError(
                    f"A registration named '{name}' already exists. Use --append to add a sample."
                )
            if len(matches) > 1:
                raise AmbiguousPersonError(
                    f"More than one registration is named '{name}'. Use its ID to manage it."
                )

            now = _utc_now()
            sample = SampleMetadata(
                sample_id=str(uuid4()),
                captured_at=now,
                quality_score=float(quality_score),
                source_file=_relative_source_path(source_file, self.paths),
                rebuildable=source_file is not None,
            )
            if matches:
                original = matches[0]
                updated = Person(
                    person_id=original.person_id,
                    display_name=original.display_name,
                    created_at=original.created_at,
                    updated_at=now,
                    samples=(*original.samples, sample),
                )
                people = [updated if person.person_id == original.person_id else person for person in people]
            else:
                updated = Person(
                    person_id=str(uuid4()),
                    display_name=name,
                    created_at=now,
                    updated_at=now,
                    samples=(sample,),
                )
                people.append(updated)
            new_vectors = dict(vectors)
            new_vectors[sample.sample_id] = vector
            self._commit(people, new_vectors)
            return updated, sample

    def add_samples(
        self,
        display_name: str,
        samples: Iterable[tuple[np.ndarray, float, str | None]],
        allow_existing: bool = False,
    ) -> tuple[Person, tuple[SampleMetadata, ...]]:
        """Atomically add one or more already-validated samples for one person."""
        name = _validate_display_name(display_name)
        prepared = [
            (FaceRecognizer.normalize_embedding(vector), float(quality), _relative_source_path(source, self.paths))
            for vector, quality, source in samples
        ]
        if not prepared:
            raise DatabaseError("At least one embedding sample is required.")
        with self._write_lock():
            people, vectors = self._load_active()
            matches = [person for person in people if person.display_name.casefold() == name.casefold()]
            if matches and not allow_existing:
                raise DuplicatePersonError(
                    f"A registration named '{name}' already exists. Use --append to add a sample."
                )
            if len(matches) > 1:
                raise AmbiguousPersonError(
                    f"More than one registration is named '{name}'. Use its ID to manage it."
                )
            now = _utc_now()
            created_samples = tuple(
                SampleMetadata(
                    sample_id=str(uuid4()),
                    captured_at=now,
                    quality_score=quality,
                    source_file=source,
                    rebuildable=source is not None,
                )
                for _vector, quality, source in prepared
            )
            if matches:
                original = matches[0]
                updated = Person(
                    person_id=original.person_id,
                    display_name=original.display_name,
                    created_at=original.created_at,
                    updated_at=now,
                    samples=(*original.samples, *created_samples),
                )
                people = [updated if person.person_id == original.person_id else person for person in people]
            else:
                updated = Person(
                    person_id=str(uuid4()),
                    display_name=name,
                    created_at=now,
                    updated_at=now,
                    samples=created_samples,
                )
                people.append(updated)
            new_vectors = dict(vectors)
            for sample, (vector, _quality, _source) in zip(created_samples, prepared, strict=True):
                new_vectors[sample.sample_id] = vector
            self._commit(people, new_vectors)
            return updated, created_samples

    def snapshot(self) -> tuple[list[Person], dict[str, np.ndarray]]:
        """Return a validated read-only-style copy for controlled rebuild work."""
        people, vectors = self._load_active()
        return list(people), {sample_id: vector.copy() for sample_id, vector in vectors.items()}

    def update_person(self, person_id: str, display_name: str) -> Person:
        name = _validate_display_name(display_name)
        with self._write_lock():
            people, vectors = self._load_active()
            person = next((item for item in people if item.person_id == person_id), None)
            if person is None:
                raise DatabaseError(f"No registered person has ID '{person_id}'.")
            duplicate = next(
                (item for item in people if item.person_id != person_id and item.display_name.casefold() == name.casefold()),
                None,
            )
            if duplicate:
                raise DuplicatePersonError(f"A registration named '{name}' already exists.")
            updated = Person(
                person_id=person.person_id,
                display_name=name,
                created_at=person.created_at,
                updated_at=_utc_now(),
                samples=person.samples,
            )
            self._commit([updated if item.person_id == person_id else item for item in people], vectors)
            return updated

    def delete_person(self, person_id_or_name: str) -> Person:
        with self._write_lock():
            people, vectors = self._load_active()
            matches = self._find_people(people, person_id_or_name)
            if not matches:
                raise DatabaseError(f"No registered person matches '{person_id_or_name}'.")
            if len(matches) > 1:
                raise AmbiguousPersonError(
                    f"More than one registered person is named '{person_id_or_name}'. Use a person ID."
                )
            person = matches[0]
            remaining = [item for item in people if item.person_id != person.person_id]
            kept_vectors = {key: value for key, value in vectors.items() if key not in {sample.sample_id for sample in person.samples}}
            self._commit(remaining, kept_vectors)
            return person

    def load_embeddings(self) -> list[StoredEmbedding]:
        people, vectors = self._load_active()
        embeddings: list[StoredEmbedding] = []
        for person in people:
            for sample in person.samples:
                vector = vectors.get(sample.sample_id)
                if vector is not None:
                    embeddings.append(
                        StoredEmbedding(
                            person_id=person.person_id,
                            display_name=person.display_name,
                            sample_id=sample.sample_id,
                            vector=vector,
                        )
                    )
        return embeddings

    def save_embeddings(self, people: Iterable[Person], vectors: dict[str, np.ndarray]) -> None:
        with self._write_lock():
            self._commit(list(people), vectors)

    def rebuild_database(
        self,
        detector: object | None = None,
        recognizer: object | None = None,
        strict: bool = False,
    ) -> dict[str, int]:
        """Validate storage or recompute embeddings from retained local crops.

        Calling without a detector/recognizer validates and rewrites the current
        snapshot. Supplying both reprocesses only samples marked rebuildable;
        strict mode fails if any retained source is unavailable or invalid.
        """
        with self._write_lock():
            people, vectors = self._load_active()
            rebuilt = 0
            skipped = 0
            if (detector is None) != (recognizer is None):
                raise DatabaseError("Database rebuild requires both a detector and a recognizer.")
            if detector is not None and recognizer is not None:
                for person in people:
                    for sample in person.samples:
                        if not sample.rebuildable or not sample.source_file:
                            skipped += 1
                            continue
                        source = (self.paths.data_dir / sample.source_file).resolve()
                        if not source.is_file():
                            if strict:
                                raise DatabaseError(f"Missing retained enrollment source: {source}")
                            skipped += 1
                            continue
                        try:
                            import cv2

                            frame = cv2.imread(str(source))
                            detections = detector.detect(frame) if frame is not None else []
                            if len(detections) != 1 or not detections[0].supports_recognition:
                                raise DatabaseError(f"Retained source cannot produce one YuNet face: {source}")
                            vectors[sample.sample_id] = recognizer.embedding_from_detection(frame, detections[0])
                            rebuilt += 1
                        except (DatabaseError, OSError, ValueError) as exc:
                            if strict:
                                raise DatabaseError(f"Could not rebuild sample '{sample.sample_id}': {exc}") from exc
                            skipped += 1
            self._commit(people, vectors)
            return {"people": len(people), "embeddings": len(vectors), "rebuilt": rebuilt, "skipped": skipped}

    def _load_active(self) -> tuple[list[Person], dict[str, np.ndarray]]:
        manifest = self._read_manifest()
        if manifest is None:
            return [], {}
        generation_ids = [manifest.get("active_generation"), manifest.get("previous_generation")]
        failures: list[str] = []
        for generation_id in generation_ids:
            if not generation_id:
                continue
            try:
                return self._load_generation(str(generation_id))
            except DatabaseError as exc:
                failures.append(str(exc))
        raise DatabaseError("No valid embedding generation is available. " + " ".join(failures))

    def _read_manifest(self) -> dict[str, object] | None:
        if not self.paths.manifest_path.exists():
            return None
        try:
            payload = json.loads(self.paths.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatabaseError(f"Cannot read database manifest: {exc}") from exc
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise DatabaseError("Database manifest schema version is unsupported.")
        return payload

    def _load_generation(self, generation_id: str) -> tuple[list[Person], dict[str, np.ndarray]]:
        metadata_path, embeddings_path = self._generation_paths(generation_id)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(embeddings_path, allow_pickle=False) as archive:
                sample_ids = [str(value) for value in archive["sample_ids"].tolist()]
                raw_vectors = np.asarray(archive["embeddings"], dtype=np.float32)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise DatabaseError(f"Cannot load database generation '{generation_id}': {exc}") from exc
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("generation_id") != generation_id:
            raise DatabaseError(f"Database generation '{generation_id}' has invalid metadata.")
        if raw_vectors.ndim != 2 or raw_vectors.shape[1:] != (FaceRecognizer.embedding_dimension,):
            raise DatabaseError(f"Database generation '{generation_id}' has invalid embedding dimensions.")
        if len(sample_ids) != len(raw_vectors) or len(set(sample_ids)) != len(sample_ids):
            raise DatabaseError(f"Database generation '{generation_id}' has misaligned embeddings.")
        vectors = {sample_id: FaceRecognizer.normalize_embedding(vector) for sample_id, vector in zip(sample_ids, raw_vectors, strict=True)}
        people = [_person_from_dict(value) for value in metadata.get("people", [])]
        metadata_ids = {sample.sample_id for person in people for sample in person.samples}
        if metadata_ids != set(vectors):
            raise DatabaseError(f"Database generation '{generation_id}' metadata and embeddings do not match.")
        return people, vectors

    def _commit(self, people: list[Person], vectors: dict[str, np.ndarray]) -> None:
        _validate_database_content(people, vectors)
        generation_id = f"generation-{uuid4()}"
        metadata_path, embeddings_path = self._generation_paths(generation_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": generation_id,
            "created_at": _utc_now(),
            "people": [_person_to_dict(person) for person in people],
        }
        sample_ids = sorted(vectors)
        matrix = np.stack([FaceRecognizer.normalize_embedding(vectors[sample_id]) for sample_id in sample_ids]) if sample_ids else np.empty((0, FaceRecognizer.embedding_dimension), dtype=np.float32)
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        temp_embeddings = embeddings_path.with_suffix(".npz.tmp")
        try:
            _atomic_text_write(temp_metadata, json.dumps(metadata, indent=2, sort_keys=True))
            with temp_embeddings.open("wb") as handle:
                np.savez_compressed(handle, sample_ids=np.asarray(sample_ids, dtype="U36"), embeddings=matrix)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_metadata, metadata_path)
            os.replace(temp_embeddings, embeddings_path)
            previous = None
            manifest = self._read_manifest()
            if manifest:
                previous = manifest.get("active_generation")
            new_manifest = {
                "schema_version": SCHEMA_VERSION,
                "active_generation": generation_id,
                "previous_generation": previous,
                "updated_at": _utc_now(),
            }
            _atomic_text_write(self.paths.manifest_path, json.dumps(new_manifest, indent=2, sort_keys=True))
            self._prune_generations(keep={generation_id, str(previous) if previous else ""})
        finally:
            temp_metadata.unlink(missing_ok=True)
            temp_embeddings.unlink(missing_ok=True)

    def _generation_paths(self, generation_id: str) -> tuple[Path, Path]:
        if not generation_id.startswith("generation-") or any(char in generation_id for char in "\\/"):
            raise DatabaseError("Invalid generation identifier in database manifest.")
        return (
            self.paths.generations_dir / f"{generation_id}.json",
            self.paths.generations_dir / f"{generation_id}.npz",
        )

    def _prune_generations(self, keep: set[str]) -> None:
        for path in self.paths.generations_dir.glob("generation-*"):
            generation_id = path.stem
            if generation_id not in keep:
                path.unlink(missing_ok=True)

    @contextmanager
    def _write_lock(self):
        lock_directory = self.paths.database_dir / ".write.lock"
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                lock_directory.mkdir()
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise DatabaseError("Timed out waiting for another database write to finish.")
                time.sleep(0.05)
        try:
            yield
        finally:
            shutil.rmtree(lock_directory, ignore_errors=True)

    @staticmethod
    def _find_people(people: Iterable[Person], person_id_or_name: str) -> list[Person]:
        exact_id = [person for person in people if person.person_id == person_id_or_name]
        if exact_id:
            return exact_id
        needle = person_id_or_name.strip().casefold()
        return [person for person in people if person.display_name.casefold() == needle]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_display_name(value: str) -> str:
    name = " ".join(value.split())
    if not name or len(name) > 100:
        raise DatabaseError("A display name must contain 1 to 100 visible characters.")
    if any(character in name for character in "\\/:*?\"<>|") or any(ord(character) < 32 for character in name):
        raise DatabaseError("The display name contains characters that are unsafe for local file metadata.")
    return name


def _relative_source_path(source_file: str | None, paths: AppPaths) -> str | None:
    if source_file is None:
        return None
    path = Path(source_file).resolve()
    try:
        return str(path.relative_to(paths.data_dir))
    except ValueError as exc:
        raise DatabaseError("Stored enrollment images must stay inside the configured data directory.") from exc


def _person_to_dict(person: Person) -> dict[str, object]:
    return {
        "person_id": person.person_id,
        "display_name": person.display_name,
        "created_at": person.created_at,
        "updated_at": person.updated_at,
        "samples": [asdict(sample) for sample in person.samples],
    }


def _person_from_dict(payload: object) -> Person:
    if not isinstance(payload, dict):
        raise DatabaseError("Database person metadata is malformed.")
    try:
        samples = tuple(SampleMetadata(**sample) for sample in payload.get("samples", []))
        return Person(
            person_id=str(payload["person_id"]),
            display_name=_validate_display_name(str(payload["display_name"])),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            samples=samples,
        )
    except (KeyError, TypeError, DatabaseError) as exc:
        raise DatabaseError(f"Database person metadata is malformed: {exc}") from exc


def _validate_database_content(people: list[Person], vectors: dict[str, np.ndarray]) -> None:
    person_ids = [person.person_id for person in people]
    if len(person_ids) != len(set(person_ids)):
        raise DatabaseError("Database contains duplicate person IDs.")
    sample_ids = [sample.sample_id for person in people for sample in person.samples]
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(vectors):
        raise DatabaseError("Database samples and embeddings do not match.")
    for vector in vectors.values():
        FaceRecognizer.normalize_embedding(vector)


def _atomic_text_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".write")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
