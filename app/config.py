"""Runtime configuration and the public catalogs (styles, presets, sizes).

Everything here is intentionally import-light: the web server and the API must
start without torch installed, and `/api/config` has to be able to describe the
studio to the browser before any weights are loaded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MODELS_DIR = Path(os.getenv("HEROGOD_MODELS_DIR") or (ROOT / "models")).expanduser().resolve()

APP_NAME = "Herogod"
APP_TAGLINE = "Qwen Image Edit Studio"
APP_VERSION = "2.0.0"

# The LoRA this studio is built around. Both values are also exposed in the UI
# and can be overridden with environment variables before the server starts.
BASE_MODEL = os.getenv("HEROGOD_BASE_MODEL", "Qwen/Qwen-Image-Edit-2511").strip()
LORA_REPO = os.getenv("HEROGOD_LORA_REPO", "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora").strip()
LORA_WEIGHT = os.getenv("HEROGOD_LORA_WEIGHT", "qwen-image-edit-plus-nsfw-lora.safetensors").strip()
LORA_ADAPTER = os.getenv("HEROGOD_LORA_ADAPTER", "mcnl-nsfw-v1").strip()
LORA_LABEL = os.getenv("HEROGOD_LORA_LABEL", "MCNL NSFW v1").strip()


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = env_str(name, default).lower()
    return value if value in allowed else default


@dataclass
class Settings:
    """Server settings, resolved once at import time."""

    host: str = field(default_factory=lambda: env_str("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: env_int("PORT", 8000))
    api_token: str = field(default_factory=lambda: env_str("HEROGOD_API_TOKEN"))
    hf_token: str = field(default_factory=lambda: env_str("HF_TOKEN"))
    mock: bool = field(default_factory=lambda: env_bool("HEROGOD_MOCK"))
    base_model: str = BASE_MODEL
    lora_repo: str = LORA_REPO
    lora_weight: str = LORA_WEIGHT
    lora_adapter: str = LORA_ADAPTER
    dtype: str = field(default_factory=lambda: env_choice("HEROGOD_DTYPE", "auto", {"auto", "bf16", "fp16", "fp32"}))
    quant: str = field(default_factory=lambda: env_choice("HEROGOD_QUANT", "none", {"none", "bnb4"}))
    offload: str = field(
        default_factory=lambda: env_choice("HEROGOD_OFFLOAD", "none", {"none", "model", "sequential"})
    )
    vae_tiling: bool = field(default_factory=lambda: env_bool("HEROGOD_VAE_TILING", True))
    max_pixels: int = field(default_factory=lambda: env_int("HEROGOD_MAX_PIXELS", 1024 * 1024))
    max_refs: int = field(default_factory=lambda: env_int("HEROGOD_MAX_REFS", 3))
    default_lora_scale: float = field(default_factory=lambda: env_float("HEROGOD_LORA_SCALE", 1.0))
    default_steps: int = field(default_factory=lambda: env_int("HEROGOD_STEPS", 20))
    default_true_cfg: float = field(default_factory=lambda: env_float("HEROGOD_TRUE_CFG", 4.0))
    max_steps: int = 60
    max_seed: int = 2_147_483_647
    max_image_bytes: int = 15 * 1024 * 1024
    max_request_bytes: int = 64 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    keep_jobs: int = 40
    max_concurrency: int = field(default_factory=lambda: env_int("HEROGOD_MAX_CONCURRENCY", 1))

    def public(self) -> dict[str, Any]:
        return {
            "app": APP_NAME,
            "tagline": APP_TAGLINE,
            "version": APP_VERSION,
            "base_model": self.base_model,
            "lora_repo": self.lora_repo,
            "mock": self.mock,
            "limits": {
                "max_refs": self.max_refs,
                "max_steps": self.max_steps,
                "max_pixels": self.max_pixels,
                "max_image_bytes": self.max_image_bytes,
            },
            "defaults": {
                "steps": min(self.default_steps, self.max_steps),
                "true_cfg": self.default_true_cfg,
                "lora_scale": self.default_lora_scale,
                "size_mode": "match",
                "seed": -1,
            },
        }


SETTINGS = Settings()

# ---------------------------------------------------------------------------
# Style presets (Fooocus-style chips)
# ---------------------------------------------------------------------------

STYLES: list[dict[str, str]] = [
    {"id": "none", "label": "No style", "prompt": "", "negative": ""},
    {
        "id": "photographic",
        "label": "Photographic",
        "prompt": "professional photograph, natural window light, 50mm lens, shallow depth of field, fine skin texture, true-to-life color",
        "negative": "illustration, painting, cartoon, low detail",
    },
    {
        "id": "cinematic",
        "label": "Cinematic",
        "prompt": "cinematic still, anamorphic composition, dramatic key light, filmic color grade, subtle film grain",
        "negative": "flat lighting, snapshot, low contrast",
    },
    {
        "id": "portrait",
        "label": "Studio portrait",
        "prompt": "studio portrait, beauty dish lighting, seamless backdrop, crisp detail, professional retouching",
        "negative": "harsh shadows, clutter, snapshot",
    },
    {
        "id": "editorial",
        "label": "Editorial",
        "prompt": "high fashion editorial, magazine composition, styled wardrobe, refined color palette, editorial lighting",
        "negative": "amateur, low effort, busy background",
    },
    {
        "id": "anime",
        "label": "Anime",
        "prompt": "anime illustration, clean lineart, cel shading, expressive eyes, vibrant color",
        "negative": "photorealistic, 3d render, muddy colors",
    },
    {
        "id": "digital-art",
        "label": "Digital art",
        "prompt": "polished digital painting, painterly brushwork, dramatic lighting, rich color harmony",
        "negative": "photograph, low detail, flat shading",
    },
    {
        "id": "noir",
        "label": "Film noir",
        "prompt": "black and white film noir, hard side light, deep shadows, 35mm grain",
        "negative": "colorful, flat lighting, digital look",
    },
    {
        "id": "fantasy",
        "label": "Fantasy",
        "prompt": "epic fantasy artwork, volumetric atmosphere, ornate detail, dramatic rim light",
        "negative": "modern clothing, mundane setting, low detail",
    },
    {
        "id": "cyberpunk",
        "label": "Cyberpunk",
        "prompt": "cyberpunk scene, neon rim light, wet reflective surfaces, moody haze",
        "negative": "daylight, rustic, low saturation",
    },
    {
        "id": "watercolor",
        "label": "Watercolor",
        "prompt": "watercolor painting, soft pigment bleed, textured paper, loose expressive strokes",
        "negative": "photograph, hard edges, digital render",
    },
    {
        "id": "product",
        "label": "Product shot",
        "prompt": "clean product photography, softbox lighting, seamless studio background, sharp focus",
        "negative": "clutter, harsh shadow, motion blur",
    },
]

PERFORMANCE_PRESETS: list[dict[str, Any]] = [
    {"id": "turbo", "label": "Turbo", "hint": "Quick look, lower fidelity", "steps": 8, "true_cfg": 2.5},
    {"id": "balanced", "label": "Balanced", "hint": "Recommended default", "steps": 20, "true_cfg": 4.0},
    {"id": "quality", "label": "Quality", "hint": "Slow, best detail", "steps": 32, "true_cfg": 4.5},
]

# Aspect ratio presets. Values are pixel budgets used to derive width/height.
ASPECT_PRESETS: list[dict[str, Any]] = [
    {"id": "match", "label": "Match input", "ratio": None},
    {"id": "1:1", "label": "1:1", "ratio": 1.0},
    {"id": "4:3", "label": "4:3", "ratio": 4 / 3},
    {"id": "3:4", "label": "3:4", "ratio": 3 / 4},
    {"id": "16:9", "label": "16:9", "ratio": 16 / 9},
    {"id": "9:16", "label": "9:16", "ratio": 9 / 16},
]

ASPECT_BY_ID = {item["id"]: item for item in ASPECT_PRESETS}
STYLE_BY_ID = {item["id"]: item for item in STYLES}


def catalog() -> dict[str, Any]:
    return {
        "styles": STYLES,
        "performance": PERFORMANCE_PRESETS,
        "aspects": ASPECT_PRESETS,
        "lora": {
            "repo": SETTINGS.lora_repo,
            "weight": SETTINGS.lora_weight,
            "adapter": SETTINGS.lora_adapter,
            "label": LORA_LABEL,
        },
        "notice": {
            "adults_only": True,
            "summary": (
                "Herogod runs an adult-capable open-weights model locally. It is for adults only. "
                "Never create sexual content involving minors, non-consensual imagery, or content that "
                "violates the law where you live."
            ),
        },
    }
