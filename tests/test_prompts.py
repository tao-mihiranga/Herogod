"""Unit tests for prompt composition and size resolution."""
from __future__ import annotations

import pytest
from PIL import Image

from app.prompts import RequestProblem, parse_seed, resolve_prompt, target_size


def test_resolve_prompt_appends_styles():
    resolved = resolve_prompt("put her in a red dress", ["cinematic", "portrait"])
    assert resolved.user_prompt == "put her in a red dress"
    assert resolved.prompt.startswith("put her in a red dress, ")
    assert "cinematic still" in resolved.prompt
    assert "studio portrait" in resolved.prompt
    assert "flat lighting" in resolved.negative
    assert resolved.style_ids == ["cinematic", "portrait"]


def test_resolve_prompt_ignores_none_and_duplicates():
    resolved = resolve_prompt("test", ["none", "anime", "anime", ""])
    assert resolved.style_ids == ["anime"]
    assert resolved.prompt.count("anime illustration") == 1


def test_resolve_prompt_rejects_empty():
    with pytest.raises(RequestProblem):
        resolve_prompt("   ", [])


def test_resolve_prompt_rejects_unknown_style():
    with pytest.raises(RequestProblem):
        resolve_prompt("x", ["not-a-style"])


def test_resolve_prompt_limits_style_count():
    with pytest.raises(RequestProblem):
        resolve_prompt("x", ["cinematic", "portrait", "anime", "noir", "fantasy"])


def test_resolve_prompt_keeps_user_negative():
    resolved = resolve_prompt("x", ["anime"], "blurry, extra fingers")
    assert resolved.negative.startswith("blurry, extra fingers")


def test_target_size_matches_source_aspect():
    source = Image.new("RGB", (1200, 800))
    width, height = target_size(source, "match", None, None, 1024 * 1024)
    assert (width, height) == (1248, 832)
    assert width * height <= 1024 * 1024
    assert width % 8 == 0 and height % 8 == 0


def test_target_size_uses_aspect_preset():
    source = Image.new("RGB", (1000, 1000))
    width, height = target_size(source, "16:9", None, None, 1024 * 1024)
    assert width > height
    assert width % 8 == 0 and height % 8 == 0


def test_target_size_explicit_dimensions_are_clamped():
    source = Image.new("RGB", (512, 512))
    width, height = target_size(source, "match", 4096, 4096, 1024 * 1024)
    assert width * height <= 1024 * 1024
    assert width % 8 == 0


def test_target_size_unknown_aspect():
    with pytest.raises(RequestProblem):
        target_size(Image.new("RGB", (64, 64)), "banana", None, None, 1024 * 1024)


def test_parse_seed():
    assert parse_seed(-5) == -1
    assert parse_seed("42") == 42
    assert parse_seed(None) == -1
    assert parse_seed("abc") == -1
    assert parse_seed(10**12) <= 2_147_483_647
