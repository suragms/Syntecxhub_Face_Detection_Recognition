from __future__ import annotations

import json

import numpy as np
import pytest

from core.errors import DuplicatePersonError
from core.face_database import FaceDatabase


def test_database_persists_reload_and_delete(paths, normalized_embedding):
    database = FaceDatabase(paths)
    person, sample = database.add_person("Ada Lovelace", normalized_embedding, 0.8)
    assert database.list_people()[0].display_name == "Ada Lovelace"
    reloaded = FaceDatabase(paths)
    embeddings = reloaded.load_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0].person_id == person.person_id
    assert np.allclose(embeddings[0].vector, normalized_embedding)
    removed = reloaded.delete_person(person.person_id)
    assert removed.person_id == person.person_id
    assert reloaded.list_people() == []


def test_database_rejects_duplicate_name_unless_append(paths, normalized_embedding):
    database = FaceDatabase(paths)
    database.add_person("Grace Hopper", normalized_embedding, 0.7)
    with pytest.raises(DuplicatePersonError):
        database.add_person("Grace Hopper", normalized_embedding, 0.7)
    person, _ = database.add_person("Grace Hopper", normalized_embedding, 0.7, allow_existing=True)
    assert person.sample_count == 2


def test_database_empty_generation_can_be_rebuilt(paths):
    database = FaceDatabase(paths)
    result = database.rebuild_database()
    assert result["people"] == 0
    assert result["embeddings"] == 0
    assert result["rebuilt"] == 0
    assert database.list_people() == []


def test_database_falls_back_to_previous_generation(paths, normalized_embedding):
    database = FaceDatabase(paths)
    first, _ = database.add_person("First", normalized_embedding, 0.7)
    second_vector = np.roll(normalized_embedding, 1)
    database.add_person("Second", second_vector, 0.7)
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    active = paths.generations_dir / f"{manifest['active_generation']}.json"
    active.write_text("not valid json", encoding="utf-8")
    # The previous snapshot still contains the original registration.
    restored = FaceDatabase(paths).list_people()
    assert [item.display_name for item in restored] == [first.display_name]
