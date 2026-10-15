from __future__ import annotations

from hashlib import sha256

import pytest

from core.errors import ModelIntegrityError
from core.model_manager import ModelManager, ModelSpec


def test_verify_hashes_local_model(paths):
    payload = b"model-data" * 200
    spec = ModelSpec("test", "test.onnx", "https://example.invalid/model", "https://example.invalid", "Test", sha256(payload).hexdigest())
    target = paths.models_dir / spec.filename
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    assert ModelManager(paths).verify(spec) == sha256(payload).hexdigest()


def test_verify_rejects_wrong_hash(paths):
    spec = ModelSpec("test", "test.onnx", "https://example.invalid/model", "https://example.invalid", "Test", "0" * 64)
    target = paths.models_dir / spec.filename
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model-data" * 200)
    with pytest.raises(ModelIntegrityError, match="mismatch"):
        ModelManager(paths).verify(spec)
