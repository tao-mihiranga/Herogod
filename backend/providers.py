"""Image-editing providers.

Heavy ML dependencies are imported lazily so the web server and API-only setups stay
small. No provider safety settings are modified by this module.
"""
from __future__ import annotations

import io
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any

from PIL import Image

INSTRUCT_PIX2PIX = "timbrooks/instruct-pix2pix"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-image"
LOCAL_MODELS = {INSTRUCT_PIX2PIX, SDXL}
HF_MODELS = {INSTRUCT_PIX2PIX}


class ProviderError(RuntimeError):
    """An error safe to show in the client, with an appropriate HTTP status."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class EditResult:
    image: Image.Image
    provider: str
    model: str
    seed: int | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return value.lower() in {"1", "true", "yes", "on"} if value is not None else default


def _clean_error(error: Exception) -> str:
    """Keep useful provider context without leaking configured credentials."""
    message = str(error).strip() or error.__class__.__name__
    for variable in ("HF_TOKEN", "GEMINI_API_KEY", "HEROGOD_API_TOKEN"):
        token = os.getenv(variable)
        if token:
            message = message.replace(token, "[redacted]")
    return message[:700]


def _resize_for_model(image: Image.Image, max_size: int) -> Image.Image:
    """Downsize very large inputs and keep dimensions compatible with diffusion VAEs."""
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    width = max(64, (image.width // 8) * 8)
    height = max(64, (image.height // 8) * 8)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class LocalModelManager:
    """Lazy, process-local Diffusers pipelines with a single model-load lock."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}
        self._devices: dict[str, str] = {}
        self._load_lock = threading.Lock()
        self.status = "not_loaded"
        self.last_error: str | None = None

    def _get_pipeline(self, model_id: str) -> tuple[Any, Any, str]:
        if model_id not in LOCAL_MODELS:
            raise ProviderError(f"Unsupported local model: {model_id}", 400)
        if model_id in self._pipelines:
            return self._pipelines[model_id], self._torch(), self._devices[model_id]

        with self._load_lock:
            if model_id in self._pipelines:
                return self._pipelines[model_id], self._torch(), self._devices[model_id]
            self.status = "loading"
            self.last_error = None
            try:
                torch = self._torch()
                import diffusers

                cuda = torch.cuda.is_available()
                device = "cuda" if cuda else "cpu"
                dtype = torch.float16 if cuda else torch.float32
                common = {
                    "pretrained_model_name_or_path": model_id,
                    "torch_dtype": dtype,
                    "use_safetensors": True,
                }
                if model_id == INSTRUCT_PIX2PIX:
                    pipeline = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(**common)
                else:
                    if cuda:
                        common["variant"] = "fp16"
                    pipeline = diffusers.AutoPipelineForImage2Image.from_pretrained(**common)

                pipeline.set_progress_bar_config(disable=True)
                if cuda and _env_flag("HEROGOD_CPU_OFFLOAD", model_id == SDXL):
                    pipeline.enable_model_cpu_offload()
                else:
                    pipeline.to(device)
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                if model_id == SDXL and hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()

                self._pipelines[model_id] = pipeline
                self._devices[model_id] = device
                self.status = "ready"
                return pipeline, torch, device
            except Exception as error:
                self.status = "error"
                self.last_error = _clean_error(error)
                raise

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
            return torch
        except ImportError as error:
            raise ProviderError(
                "Local model dependencies are missing. Run: pip install -r requirements-local.txt",
                503,
            ) from error

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        model_id: str,
        steps: int,
        guidance: float,
        image_guidance: float,
        strength: float,
        seed: int,
    ) -> EditResult:
        try:
            pipeline, torch, device = self._get_pipeline(model_id)
            actual_seed = secrets.randbelow(2_147_483_647) if seed < 0 else seed
            generator = torch.Generator(device=device).manual_seed(actual_seed)
            max_default = 768 if model_id == INSTRUCT_PIX2PIX else 1024
            max_size = int(os.getenv("HEROGOD_MAX_IMAGE_SIZE", str(max_default)))
            max_size = min(1536, max(256, max_size))
            prepared = _resize_for_model(image, max_size)

            arguments: dict[str, Any] = {
                "prompt": prompt,
                "image": prepared,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
            }
            if model_id == INSTRUCT_PIX2PIX:
                arguments["image_guidance_scale"] = image_guidance
            else:
                arguments["strength"] = strength

            with torch.inference_mode():
                output = pipeline(**arguments).images[0]
            return EditResult(output.convert("RGB"), "local", model_id, actual_seed)
        except ProviderError:
            raise
        except ImportError as error:
            raise ProviderError(
                "Diffusers is not installed. Run: pip install -r requirements-local.txt", 503
            ) from error
        except Exception as error:
            name = error.__class__.__name__
            if "OutOfMemory" in name:
                message = (
                    "The GPU ran out of memory. Use a smaller HEROGOD_MAX_IMAGE_SIZE, "
                    "set HEROGOD_CPU_OFFLOAD=1, or restart the runtime and try the lighter model."
                )
            else:
                message = f"Local inference failed: {_clean_error(error)}"
            raise ProviderError(message, 500) from error


LOCAL_MANAGER = LocalModelManager()


def local_edit(*args: Any, **kwargs: Any) -> EditResult:
    return LOCAL_MANAGER.edit(*args, **kwargs)


def huggingface_edit(
    image: Image.Image,
    prompt: str,
    model_id: str,
    steps: int,
    guidance: float,
) -> EditResult:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise ProviderError("HF_TOKEN is not configured on the backend. See the API key guide.", 503)
    if model_id not in HF_MODELS:
        raise ProviderError(
            f"Unsupported Hugging Face model: {model_id}. Choose {INSTRUCT_PIX2PIX} or run a local model.", 400
        )
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            provider=os.getenv("HEROGOD_HF_PROVIDER", "auto"),
            api_key=token,
            timeout=float(os.getenv("HEROGOD_PROVIDER_TIMEOUT", "240")),
        )
        output = client.image_to_image(
            _png_bytes(image),
            prompt=prompt,
            model=model_id,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        if not isinstance(output, Image.Image):
            output = Image.open(io.BytesIO(output))
        return EditResult(output.convert("RGB"), "huggingface", model_id)
    except ProviderError:
        raise
    except ImportError as error:
        raise ProviderError("huggingface_hub is not installed. Run: pip install -r requirements.txt", 503) from error
    except Exception as error:
        raise ProviderError(
            "Hugging Face inference failed. The model may be unavailable, loading, or outside your remaining credits. "
            f"Details: {_clean_error(error)}",
            502,
        ) from error


def gemini_edit(image: Image.Image, prompt: str, model_id: str) -> EditResult:
    token = os.getenv("GEMINI_API_KEY", "").strip()
    if not token:
        raise ProviderError("GEMINI_API_KEY is not configured on the backend. See the API key guide.", 503)
    configured_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    # The frontend's default option follows a backend override when Google changes model names.
    if model_id == DEFAULT_GEMINI_MODEL:
        model_id = configured_model
    allowed_models = {DEFAULT_GEMINI_MODEL, configured_model}
    if model_id not in allowed_models:
        raise ProviderError(f"Unsupported Gemini model: {model_id}", 400)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=token)
        response = client.models.generate_content(
            model=model_id,
            contents=[
                prompt,
                types.Part.from_bytes(data=_png_bytes(image), mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None)
                if data:
                    output = Image.open(io.BytesIO(data))
                    output.load()
                    return EditResult(output.convert("RGB"), "gemini", model_id)
        raise ProviderError(
            "Gemini returned no image. It may have declined the request or produced text only; review the request and try a different edit.",
            422,
        )
    except ProviderError:
        raise
    except ImportError as error:
        raise ProviderError("google-genai is not installed. Run: pip install -r requirements.txt", 503) from error
    except Exception as error:
        raise ProviderError(f"Gemini image editing failed: {_clean_error(error)}", 502) from error
