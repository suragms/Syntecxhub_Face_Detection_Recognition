from __future__ import annotations

from main import build_parser, main


def test_main_requires_operation():
    assert main([]) == 2


def test_parser_accepts_documented_image_mode():
    args = build_parser().parse_args(["--source", "image", "--input", "face.jpg"])
    assert args.source == "image"
    assert args.input == "face.jpg"


def test_delete_requires_confirmation(tmp_path):
    assert main(["--data-dir", str(tmp_path / "data"), "--delete", "somebody"]) == 2
