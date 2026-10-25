"""Prompt composition, size resolution, and image preparation helpers.

Fooocus-style "styles" are plain text fragments appended to the user's
instruction, so nothing is hidden from the operator: `/api/config` returns the
exact fragments and the UI shows them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from PIL import Image

from .config import ASPECT_BY_ID, STYLE_BY_ID


class RequestProblem(ValueError):
    """A client mistake that should surface as HTTP 4xx."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ResolvedPrompt:
    prompt: str
    negative: str
    style_ids: list[str]
    user_prompt: str


def _dedupe_join(parts: Iterable[str]) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part or "").strip(" ,;")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(cleaned)
    return ", ".join(kept)


def resolve_prompt(prompt: str, style_ids: list[str] | None, negative: str = "") -> ResolvedPrompt:
    user_prompt = (prompt or "").strip()
    if not user_prompt:
        raise RequestProblem("Describe the edit you want to make.", 422)
    if len(user_prompt) > 2000:
        raise RequestProblem("The instruction is too long (2000 characters maximum).", 413)

    chosen: list[str] = []
    for raw in style_ids or []:
        style_id = (raw or "").strip()
        if not style_id or style_id == "none":
            continue
        if style_id not in STYLE_BY_ID:
            raise RequestProblem(f"Unknown style: {style_id}", 400)
        if style_id not in chosen:
            chosen.append(style_id)
    if len(chosen) > 4:
        raise RequestProblem("Pick at most 4 styles so the instruction stays coherent.", 400)

    style_prompts = [STYLE_BY_ID[style_id]["prompt"] for style_id in chosen]
    style_negatives = [STYLE_BY_ID[style_id]["negative"] for style_id in chosen]
    return ResolvedPrompt(
        prompt=_dedupe_join([user_prompt, *style_prompts]),
        negative=_dedupe_join([negative or "", *style_negatives]),
        style_ids=chosen,
        user_prompt=user_prompt,
    )


def target_size(
    source: Image.Image,
    aspect: str | None,
    width: int | None,
    height: int | None,
    max_pixels: int,
    multiple: int = 8,
) -> tuple[int, int]:
    """Work out the output canvas.

    Priority: explicit width/height, then an aspect preset, then the source
    aspect. Everything is clamped to the pixel budget and rounded to the model's
    VAE multiple.
    """
    def clamp(width_px: float, height_px: float) -> tuple[int, int]:
        w = max(multiple * 4, int(round(width_px)))
        h = max(multiple * 4, int(round(height_px)))
        budget = max(multiple * multiple * 16, max_pixels)
        scale = min(1.0, (budget / float(w * h)) ** 0.5)
        # floor, so the rounded pair never exceeds the pixel budget
        w = max(multiple * 4, int(w * scale // multiple) * multiple)
        h = max(multiple * 4, int(h * scale // multiple) * multiple)
        return w, h

    if width and height and width >= 64 and height >= 64:
        return clamp(width, height)

    ratio: float | None = None
    aspect_key = (aspect or "match").strip()
    if aspect_key and aspect_key != "match":
        preset = ASPECT_BY_ID.get(aspect_key)
        if not preset:
            raise RequestProblem(f"Unknown aspect ratio: {aspect_key}", 400)
        ratio = preset["ratio"]
    if ratio is None:
        ratio = source.width / float(max(1, source.height))

    side = (max_pixels / ratio) ** 0.5
    return clamp(side * ratio, side)


def fit_to_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Aspect-preserving resize + center pad, so padded areas are copied context."""
    if image.size == size:
        return image
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def parse_seed(seed: Any) -> int:
    try:
        value = int(seed)
    except (TypeError, ValueError):
        return -1
    if value < 0:
        return -1
    return min(value, 2_147_483_647)
