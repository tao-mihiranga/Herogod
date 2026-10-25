"""The image-editing engine: Qwen-Image-Edit-2511 plus the MCNL LoRA adapter.

Design notes
------------
* Heavy imports (torch / diffusers) happen inside methods, so the server, the
  tests and the browser UI all work on a CPU-only machine.
* One lazily-loaded pipeline per process, guarded by a lock, with adapter
  switching for the LoRA. Nothing is downloaded until the first generation.
* ``MockEngine`` renders a deterministic collage instead of running a model so
  the frontend can be developed and tested without a GPU.
"""
from __future__ import annotations

import inspect
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageFilter

from .prompts import fit_to_canvas
from .config import (
    LORA_ADAPTER,
    LORA_LABEL,
    LORA_REPO,
    LORA_WEIGHT,
    MODELS_DIR,
    SETTINGS,
)

logger = logging.getLogger("herogod.engine")


class EngineError(RuntimeError):
    """An error message that is safe to show in the UI."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class GenerationRequest:
    images: list[Image.Image]
    prompt: str
    negative: str = ""
    steps: int = 20
    true_cfg: float = 4.0
    lora_scale: float = 1.0
    adapter: str = LORA_ADAPTER
    seed: int = -1
    size: tuple[int, int] | None = None
    progress: Callable[[int, int], None] | None = None
    should_cancel: Callable[[], bool] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    image: Image.Image
    seed: int
    adapter: str
    lora_scale: float
    meta: dict[str, Any]


def resolve_seed(seed: int) -> int:
    return secrets.randbelow(2_147_483_647) if seed is None or seed < 0 else seed


def _pick(call_params: Iterable[str], **candidates: Any) -> dict[str, Any]:
    """Keep only kwargs the loaded pipeline version actually accepts."""
    allowed = set(call_params)
    return {key: value for key, value in candidates.items() if key in allowed}


class BaseEngine:
    name = "base"

    def status(self) -> dict[str, Any]:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def unload(self) -> dict[str, Any]:
        return {"unloaded": False, "detail": "This engine holds no model in memory."}

    def adapters(self) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Real engine
# ---------------------------------------------------------------------------


class QwenEngine(BaseEngine):
    name = "qwen"

    def __init__(self) -> None:
        self._torch: Any = None
        self._pipe: Any = None
        self._lock = threading.RLock()
        self._state = "not_loaded"
        self._detail = "The model loads on the first generation."
        self._active_adapter: str | None = None
        self._adapter_scale: float = SETTINGS.default_lora_scale
        self._loaded_adapters: dict[str, dict[str, Any]] = {}
        self._infer_lock = threading.Lock()
        self._load_seconds: float | None = None
        self._device_name: str | None = None

    # -- helpers ---------------------------------------------------------
    def torch(self) -> Any:
        if self._torch is None:
            try:
                import torch  # noqa: PLC0415 - deliberately lazy
            except ImportError as error:  # pragma: no cover - env dependent
                raise EngineError(
                    "PyTorch is not installed. Run: pip install -r requirements-local.txt", 503
                ) from error
            self._torch = torch
        return self._torch

    def _dtype(self) -> Any:
        torch = self.torch()
        want = SETTINGS.dtype
        if want == "fp32":
            return torch.float32
        if want == "fp16":
            return torch.float16
        if want == "bf16":
            return torch.bfloat16
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32

    def _local_dir(self, repo_id: str) -> str | None:
        """Prefer a snapshot that the Colab/CLI downloader already fetched."""
        override = os.getenv("HEROGOD_BASE_MODEL_PATH", "").strip()
        if override and Path(override).expanduser().is_dir():
            return str(Path(override).expanduser())
        candidate = MODELS_DIR / repo_id.replace("/", "--")
        if (candidate / "model_index.json").is_file() or (candidate / "config.json").is_file():
            return str(candidate)
        return None

    def _adapter_catalog(self) -> list[dict[str, Any]]:
        catalog = [
            {
                "name": LORA_ADAPTER,
                "label": LORA_LABEL,
                "repo": LORA_REPO,
                "weight": LORA_WEIGHT,
                "default_scale": SETTINGS.default_lora_scale,
            }
        ]
        for raw in os.getenv("HEROGOD_EXTRA_LORAS", "").split(","):
            entry = raw.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 2:
                logger.warning("Ignoring malformed HEROGOD_EXTRA_LORAS entry: %s", entry)
                continue
            repo, weight = parts[0], parts[1]
            name = parts[2] if len(parts) > 2 else f"{repo.split('/')[-1]}-{Path(weight).stem}"
            catalog.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").replace("-", " ").title(),
                    "repo": repo,
                    "weight": weight,
                    "default_scale": 1.0,
                }
            )
        return catalog

    def adapters(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item["name"],
                "label": item["label"],
                "repo": item["repo"],
                "weight": item["weight"],
                "default_scale": item["default_scale"],
                "loaded": item["name"] in self._loaded_adapters,
            }
            for item in self._adapter_catalog()
        ]

    # -- loading ---------------------------------------------------------
    @staticmethod
    def _from_pretrained(pipeline_class: Any, source: str, dtype: Any, **extra: Any) -> Any:
        """Load a pipeline across diffusers versions.

        Recent releases renamed ``torch_dtype`` to ``dtype``; try the modern name
        first and fall back instead of pinning one spelling.
        """
        try:
            return pipeline_class.from_pretrained(source, dtype=dtype, **extra)
        except TypeError as error:
            if "dtype" not in str(error):
                raise
            logger.info("Falling back to the legacy torch_dtype argument")
            return pipeline_class.from_pretrained(source, torch_dtype=dtype, **extra)

    def _build_pipeline(self) -> Any:
        try:
            from diffusers import QwenImageEditPlusPipeline  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - env dependent
            raise EngineError(
                "Diffusers with Qwen-Image-Edit-Plus support is required. "
                "Run: pip install -r requirements-local.txt",
                503,
            ) from error

        torch = self.torch()
        source = self._local_dir(SETTINGS.base_model) or SETTINGS.base_model
        dtype = self._dtype()
        token = SETTINGS.hf_token or None
        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token

        if SETTINGS.quant == "bnb4":
            try:
                from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig  # noqa: PLC0415

                kwargs["quantization_config"] = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": dtype,
                    },
                    components_to_quantize=["transformer", "text_encoder"],
                )
                kwargs.pop("torch_dtype", None)
            except Exception as error:  # pragma: no cover - optional path
                raise EngineError(
                    f"4-bit quantisation was requested but could not be configured: {error}. "
                    "Install bitsandbytes or unset HEROGOD_QUANT.",
                    503,
                ) from error
            logger.info("Loading %s with bitsandbytes 4-bit quantisation", source)
            pipe = self._from_pretrained(QwenImageEditPlusPipeline, source, dtype, **kwargs)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(device)
        else:
            logger.info("Loading %s (dtype=%s)", source, dtype)
            pipe = self._from_pretrained(QwenImageEditPlusPipeline, source, dtype, **kwargs)
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        if SETTINGS.offload == "model" and torch.cuda.is_available():
            logger.info("Enabling model CPU offload")
            pipe.enable_model_cpu_offload()
        elif SETTINGS.offload == "sequential" and torch.cuda.is_available():
            logger.info("Enabling sequential CPU offload")
            pipe.enable_sequential_cpu_offload()

        if SETTINGS.vae_tiling:
            try:
                pipe.vae.enable_tiling()
            except Exception:  # pragma: no cover - version dependent
                logger.debug("VAE tiling is unavailable on this pipeline", exc_info=True)
        for attribute in ("transformer", "text_encoder"):
            module = getattr(pipe, attribute, None)
            if module is not None and hasattr(module, "config"):
                module.config.use_cache = False
        return pipe

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            torch = self.torch()
            if not torch.cuda.is_available() and not SETTINGS.mock:
                logger.warning("No CUDA device detected; generation will run on CPU and be very slow.")
            self._state = "loading"
            self._detail = f"Downloading/loading {SETTINGS.base_model}"
            started = time.perf_counter()
            try:
                pipe = self._build_pipeline()
            except EngineError:
                self._state = "error"
                self._detail = "Model failed to load."
                raise
            except Exception as error:  # pragma: no cover - network/GPU dependent
                self._state = "error"
                self._detail = f"Model failed to load: {error}"
                raise EngineError(
                    "Could not load Qwen-Image-Edit. Check disk space, free GPU memory, your Hugging Face "
                    f"token/network access, and the installed diffusers version. Details: {error}",
                    503,
                ) from error

            self._pipe = pipe
            self._load_seconds = round(time.perf_counter() - started, 1)
            if torch.cuda.is_available():
                self._device_name = torch.cuda.get_device_name(0)
            else:
                self._device_name = "cpu"
            self._active_adapter = None
            self._loaded_adapters = {}
            self._state = "ready"
            self._detail = f"Loaded in {self._load_seconds}s on {self._device_name}"
            return pipe

    def load(self) -> dict[str, Any]:
        self._ensure_loaded()
        return self.status()

    # -- adapters --------------------------------------------------------
    def _activate_adapter(self, name: str, scale: float) -> str | None:
        pipe = self._pipe
        if pipe is None:
            return None
        wanted = (name or "").strip() or LORA_ADAPTER
        catalog = {item["name"]: item for item in self._adapter_catalog()}
        if wanted not in catalog:
            wanted = LORA_ADAPTER
        info = catalog[wanted]
        with self._lock:
            if wanted not in self._loaded_adapters:
                kwargs: dict[str, Any] = {"adapter_name": wanted}
                if SETTINGS.hf_token and not Path(info["repo"]).expanduser().exists():
                    kwargs["token"] = SETTINGS.hf_token
                if info["weight"]:
                    kwargs["weight_name"] = info["weight"]
                logger.info("Loading LoRA adapter %s from %s", wanted, info["repo"])
                try:
                    pipe.load_lora_weights(info["repo"], **kwargs)
                except Exception as error:  # pragma: no cover - network dependent
                    raise EngineError(
                        f"Could not load the LoRA adapter {info['repo']}. It is a 590 MB download; check your "
                        f"connection, free disk space and Hugging Face access. Details: {error}",
                        503,
                    ) from error
                self._loaded_adapters[wanted] = {"repo": info["repo"], "weight": info["weight"]}
            try:
                pipe.set_adapters([wanted], adapter_weights=[float(scale)])
            except Exception as error:  # pragma: no cover - version dependent
                raise EngineError(f"Could not apply LoRA scale: {error}", 500) from error
            self._active_adapter = wanted
            self._adapter_scale = float(scale)
        return wanted

    # -- generation ------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        torch = self.torch()
        with self._infer_lock:  # one GPU, one image at a time
            return self._generate_locked(torch, request)

    def _generate_locked(self, torch: Any, request: GenerationRequest) -> GenerationResult:
        pipe = self._ensure_loaded()
        if not request.images:
            raise EngineError("At least one image is required.", 400)

        seed = resolve_seed(request.seed)
        adapter = self._activate_adapter(request.adapter, request.lora_scale)
        generator = torch.Generator(device=self._device_of(pipe)).manual_seed(seed)

        params = inspect.signature(pipe.__call__).parameters
        images = request.images
        width, height = request.size or images[0].size
        # keep the aspect ratio and pad instead of squashing references
        prepared = [fit_to_canvas(image, (width, height)) for image in images]

        progress = request.progress

        def on_step(_pipe: Any, step: int, _timestep: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if progress:
                progress(step + 1, request.steps)
            if request.should_cancel and request.should_cancel():
                raise CancelledError("Generation cancelled by the client.")
            return kwargs

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "true_cfg_scale": request.true_cfg,
            "guidance_scale": request.true_cfg,
            "generator": generator,
            "width": width,
            "height": height,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }
        if request.negative and request.negative.strip():
            call_kwargs["negative_prompt"] = request.negative
        if "callback_on_step_end" in params:
            call_kwargs["callback_on_step_end"] = on_step
        if len(prepared) > 1 and "images" in params:
            call_kwargs["images"] = prepared
        elif "images" in params:
            call_kwargs["images"] = prepared
        elif "image" in params:
            call_kwargs["image"] = prepared[0]
        else:  # pragma: no cover - defensive
            raise EngineError("The installed pipeline does not accept image input.", 500)

        kwargs = _pick(params, **call_kwargs)
        started = time.perf_counter()
        with torch.inference_mode():
            output = pipe(**kwargs)
        elapsed = time.perf_counter() - started

        result_image = output.images[0] if hasattr(output, "images") else output[0]
        result_image = result_image.convert("RGB")
        if result_image.size != (width, height):
            result_image = result_image.resize((width, height), Image.Resampling.LANCZOS)
        return GenerationResult(
            image=result_image,
            seed=seed,
            adapter=adapter or LORA_ADAPTER,
            lora_scale=float(request.lora_scale),
            meta={
                "width": width,
                "height": height,
                "steps": request.steps,
                "true_cfg": request.true_cfg,
                "seconds": round(elapsed, 2),
                "seconds_per_step": round(elapsed / max(1, request.steps), 3),
                "device": self._device_name,
                "dtype": str(self._dtype()),
                "refs": len(prepared),
            },
        )

    def _device_of(self, pipe: Any) -> str:
        torch = self.torch()
        if torch.cuda.is_available() and SETTINGS.offload == "none":
            return "cuda"
        device = getattr(pipe, "_execution_device", None)
        if isinstance(device, str):
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def status(self) -> dict[str, Any]:
        torch_installed = True
        cuda = False
        gpu = None
        try:
            torch = self.torch()
            cuda = bool(torch.cuda.is_available())
            if cuda:
                gpu = torch.cuda.get_device_name(0)
        except EngineError:
            torch_installed = False
        return {
            "engine": self.name,
            "state": self._state,
            "detail": self._detail,
            "base_model": SETTINGS.base_model,
            "lora_repo": SETTINGS.lora_repo,
            "active_adapter": self._active_adapter,
            "lora_scale": self._adapter_scale,
            "loaded_adapters": list(self._loaded_adapters),
            "torch_installed": torch_installed,
            "hf_token": bool(SETTINGS.hf_token),
            "cuda": cuda,
            "gpu": gpu,
            "device": self._device_name,
            "dtype": SETTINGS.dtype,
            "quant": SETTINGS.quant,
            "offload": SETTINGS.offload,
            "load_seconds": self._load_seconds,
            "adapters": self.adapters(),
        }

    def unload(self) -> dict[str, Any]:
        with self._lock:
            had_model = self._pipe is not None
            self._pipe = None
            self._active_adapter = None
            self._loaded_adapters = {}
            self._state = "not_loaded"
            self._detail = "Model unloaded. It reloads on the next generation."
            if self._torch is not None:
                import gc

                gc.collect()
                try:
                    if self._torch.cuda.is_available():
                        self._torch.cuda.empty_cache()
                        self._torch.cuda.ipc_collect()
                except Exception:  # pragma: no cover
                    logger.debug("GPU cache cleanup failed", exc_info=True)
            return {"unloaded": had_model, "detail": self._detail}


class CancelledError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------


class MockEngine(BaseEngine):
    """Deterministic placeholder renderer used for UI work and CI."""

    name = "mock"

    def status(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "state": "ready",
            "detail": "Mock engine: no model is loaded and no GPU is used.",
            "base_model": "mock/preview",
            "lora_repo": "mock/preview",
            "active_adapter": LORA_ADAPTER,
            "lora_scale": 1.0,
            "loaded_adapters": [LORA_ADAPTER],
            "torch_installed": False,
            "hf_token": bool(SETTINGS.hf_token),
            "cuda": False,
            "gpu": None,
            "device": "none",
            "dtype": "none",
            "quant": "none",
            "offload": "none",
            "load_seconds": 0,
            "adapters": [
                {
                    "name": LORA_ADAPTER,
                    "label": f"{LORA_LABEL} (mock)",
                    "repo": LORA_REPO,
                    "weight": LORA_WEIGHT,
                    "default_scale": 1.0,
                    "loaded": True,
                }
            ],
        }

    def adapters(self) -> list[dict[str, Any]]:
        return self.status()["adapters"]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        seed = resolve_seed(request.seed)
        size = request.size or request.images[0].size
        base = request.images[0].resize(size, Image.Resampling.LANCZOS).convert("RGB")
        stage = base.copy()
        draw = ImageDraw.Draw(stage)
        band = max(24, size[1] // 14)
        draw.rectangle([0, size[1] - band, size[0], size[1]], fill=(10, 12, 20))
        draw.text((12, size[1] - band + band // 4), "MOCK PREVIEW - no model was run", fill=(255, 111, 97))
        glow = stage.filter(ImageFilter.GaussianBlur(6))
        stage = Image.blend(stage, glow, 0.25)
        for step in range(1, max(2, min(request.steps, 8)) + 1):
            if request.progress:
                request.progress(step, max(2, min(request.steps, 8)))
            time.sleep(0.02)
        elapsed = time.perf_counter() - started
        return GenerationResult(
            image=stage,
            seed=seed,
            adapter=request.adapter or LORA_ADAPTER,
            lora_scale=float(request.lora_scale),
            meta={
                "width": size[0],
                "height": size[1],
                "steps": request.steps,
                "true_cfg": request.true_cfg,
                "seconds": round(elapsed, 2),
                "seconds_per_step": 0.0,
                "device": "mock",
                "dtype": "none",
                "refs": len(request.images),
                "mock": True,
            },
        )


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_ENGINE_LOCK = threading.Lock()
_ENGINE: BaseEngine | None = None


def get_engine() -> BaseEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = MockEngine() if SETTINGS.mock else QwenEngine()
    return _ENGINE


def set_engine(engine: BaseEngine) -> None:
    """Test seam."""
    global _ENGINE
    _ENGINE = engine
