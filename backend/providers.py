"""Image-editing providers and the curated Hugging Face model registry.

Heavy ML dependencies are imported lazily so the web server and API-only setups stay
small. No provider safety settings are modified by this module. Gated models require
the user to accept their model-card conditions and provide their own HF token.
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
FLUX3_KLEIN = "black-forest-labs/flux-3-image-klein"
FIBO2_TURBO = "briaai/Fibo-2.0-edit-turbo"
QWEN_EDIT_2511 = "Qwen/Qwen-Image-Edit-2511"
FLUX3_PUP = "black-forest-labs/flux-3-image-klein-pup"
FLUX3_REQUIRED_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "transformer",
    "scheduler",
    "vae",
    "image_processor",
)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-image"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    kind: str
    default_steps: int
    default_guidance: float
    default_max_size: int = 1024
    default_image_guidance: float = 1.5
    default_strength: float = 0.65
    size_multiple: int = 8
    gated: bool = False
    requires_cuda: bool = False
    hosted: bool = False
    cpu_offload_default: bool = False
    hardware: str = "T4 GPU recommended"
    license_name: str = "See model card"
    prompt_note: str = "Describe the edit you want."


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.model_id: spec
    for spec in (
        ModelSpec(
            INSTRUCT_PIX2PIX,
            "InstructPix2Pix",
            "instruct",
            default_steps=20,
            default_guidance=7.5,
            default_max_size=768,
            hosted=True,
            license_name="See model card",
            prompt_note="Direct instruction editing; the lightest default.",
        ),
        ModelSpec(
            SDXL,
            "SDXL img2img",
            "auto_img2img",
            default_steps=25,
            default_guidance=7.5,
            cpu_offload_default=True,
            license_name="OpenRAIL++ — review model card",
            prompt_note="Creative img2img restyling with adjustable strength.",
        ),
        ModelSpec(
            FLUX3_KLEIN,
            "FLUX 3 Image Klein",
            "flux3",
            default_steps=16,
            default_guidance=1.0,
            size_multiple=16,
            gated=True,
            requires_cuda=True,
            hardware="24GB+ GPU for BF16; more with prompt upsampler",
            license_name="FLUX Kommunity License — gated",
            prompt_note="Use a full resulting-scene caption, or enable the official prompt upsampler.",
        ),
        ModelSpec(
            FIBO2_TURBO,
            "FIBO 2.0 Edit Turbo",
            "fibo2",
            default_steps=4,
            default_guidance=1.0,
            size_multiple=16,
            gated=True,
            requires_cuda=True,
            hosted=True,
            cpu_offload_default=True,
            hardware="High-memory CUDA GPU; CPU offload supported",
            license_name="CC BY-NC 4.0 — gated, non-commercial",
            prompt_note="Four-step instruction editing. Plain instructions or compatible VGL JSON are accepted.",
        ),
        ModelSpec(
            QWEN_EDIT_2511,
            "Qwen Image Edit 2511",
            "qwen_edit",
            default_steps=40,
            default_guidance=4.0,
            size_multiple=16,
            requires_cuda=True,
            cpu_offload_default=True,
            hardware="Very high-memory CUDA GPU; CPU offload supported",
            license_name="Apache 2.0",
            prompt_note="High-fidelity editing with true CFG guidance.",
        ),
    )
}
LOCAL_MODELS = set(MODEL_SPECS)
HF_MODELS = {model_id for model_id, spec in MODEL_SPECS.items() if spec.hosted}


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


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as error:
        raise ProviderError(f"Unsupported local model: {model_id}", 400) from error


def model_catalog(token_available: bool = False) -> list[dict[str, Any]]:
    """Public model metadata. Never includes the supplied token."""
    return [
        {
            "id": spec.model_id,
            "label": spec.label,
            "kind": spec.kind,
            "default_steps": spec.default_steps,
            "default_guidance": spec.default_guidance,
            "default_image_guidance": spec.default_image_guidance,
            "default_strength": spec.default_strength,
            "default_max_size": spec.default_max_size,
            "gated": spec.gated,
            "requires_hf_token": spec.gated,
            "token_available": token_available,
            "requires_cuda": spec.requires_cuda,
            "hosted": spec.hosted,
            "hardware": spec.hardware,
            "license_name": spec.license_name,
            "prompt_note": spec.prompt_note,
            "prompt_upsampling": _env_flag("HEROGOD_FLUX_UPSAMPLING") if spec.kind == "flux3" else False,
        }
        for spec in MODEL_SPECS.values()
    ]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return value.lower() in {"1", "true", "yes", "on"} if value is not None else default


def _clean_error(error: Exception, extra_secrets: tuple[str | None, ...] = ()) -> str:
    """Keep useful provider context without leaking configured credentials."""
    message = str(error).strip() or error.__class__.__name__
    for variable in ("HF_TOKEN", "GEMINI_API_KEY", "HEROGOD_API_TOKEN"):
        token = os.getenv(variable)
        if token:
            message = message.replace(token, "[redacted]")
    for token in extra_secrets:
        if token:
            message = message.replace(token, "[redacted]")
    return message[:700]


def _resize_for_model(image: Image.Image, max_size: int, multiple: int = 8) -> Image.Image:
    """Downsize very large inputs and keep dimensions compatible with model VAEs."""
    image = image.copy()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    width = max(multiple * 4, (image.width // multiple) * multiple)
    height = max(multiple * 4, (image.height // multiple) * multiple)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FluxPromptUpsampler:
    """Optional official FLUX 3 PUP, loaded only when explicitly enabled."""

    def __init__(self) -> None:
        self._processor: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self, torch: Any, hf_token: str | None) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

                model_id = os.getenv("HEROGOD_FLUX_PUP_MODEL", FLUX3_PUP)
                processor = AutoProcessor.from_pretrained(model_id, token=hf_token or None)
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    token=hf_token or None,
                ).eval()
                self._processor = processor
                self._model = model
            except Exception as error:
                raise ProviderError(
                    "Could not load the FLUX 3 prompt upsampler. It needs a recent Transformers version, substantial "
                    f"additional memory, and access to {os.getenv('HEROGOD_FLUX_PUP_MODEL', FLUX3_PUP)}: "
                    f"{_clean_error(error, (hf_token,))}",
                    503,
                ) from error

    def upsample(self, prompt: str, image: Image.Image, torch: Any, hf_token: str | None) -> str:
        self._load(torch, hf_token)
        # Use Diffusers' official FLUX 3 helper so the chat scaffold, system prompt,
        # reference preparation, greedy generation, and caption extraction stay upstream-compatible.
        from diffusers.modular_pipelines.flux3.prompt_upsampler import upsample_prompt

        with torch.no_grad():
            caption = upsample_prompt(
                self._model,
                self._processor,
                request=prompt,
                images=[image],
                max_new_tokens=3584,
            ).strip()
        if not caption:
            raise ProviderError("The FLUX 3 prompt upsampler returned an empty caption.", 502)
        return caption


class LocalModelManager:
    """Lazy, process-local pipelines with model-specific loading and inference."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}
        self._devices: dict[str, str] = {}
        self._assets: dict[str, dict[str, Any]] = {}
        self._load_lock = threading.Lock()
        self._flux_upsampler = FluxPromptUpsampler()
        self.status = "not_loaded"
        self.last_error: str | None = None

    def _get_pipeline(self, model_id: str, hf_token: str | None = None) -> tuple[Any, Any, str]:
        spec = get_model_spec(model_id)
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
                if spec.requires_cuda and not cuda:
                    raise ProviderError(
                        f"{spec.label} requires a CUDA GPU in this integration. Choose InstructPix2Pix for CPU use "
                        "or run on a suitable Colab GPU.",
                        503,
                    )
                device = "cuda" if cuda else "cpu"
                dtype = torch.bfloat16 if spec.requires_cuda else (torch.float16 if cuda else torch.float32)
                token = hf_token or None

                if spec.kind == "flux3":
                    if not hasattr(diffusers, "ModularPipeline"):
                        raise ProviderError("FLUX 3 requires diffusers>=0.42. Reinstall requirements-local.txt.", 503)
                    pipeline = diffusers.ModularPipeline.from_pretrained(model_id, token=token)
                    # The upstream default graph includes a separate 9B prompt upsampler. Herogod
                    # uses full captions by default and its explicitly enabled PUP adapter instead,
                    # so do not load or execute that block a second time.
                    sub_blocks = getattr(getattr(pipeline, "blocks", None), "sub_blocks", None)
                    if sub_blocks is None or "prompt_upsampler" not in sub_blocks:
                        raise ProviderError(
                            "This FLUX 3 repository has an incompatible modular graph; expected a prompt_upsampler block.",
                            503,
                        )
                    sub_blocks.pop("prompt_upsampler")
                    pipeline.load_components(
                        names=list(FLUX3_REQUIRED_COMPONENTS), dtype=torch.bfloat16, token=token
                    )
                    # Diffusers 0.42 logs component load failures as warnings. Verify every
                    # editing component explicitly before moving a partial pipeline to CUDA.
                    missing = [
                        name for name in FLUX3_REQUIRED_COMPONENTS
                        if getattr(pipeline, name, None) is None
                    ]
                    if missing:
                        raise ProviderError(
                            "FLUX 3 could not load required editing components: "
                            f"{', '.join(missing)}. Check model access, dependencies, and the Colab/server log.",
                            503,
                        )
                    pipeline.to("cuda")
                else:
                    common: dict[str, Any] = {
                        "pretrained_model_name_or_path": model_id,
                        "torch_dtype": dtype,
                        "use_safetensors": True,
                        "token": token,
                    }
                    if spec.kind == "instruct":
                        pipeline = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(**common)
                    elif spec.kind == "auto_img2img":
                        if cuda:
                            common["variant"] = "fp16"
                        pipeline = diffusers.AutoPipelineForImage2Image.from_pretrained(**common)
                    elif spec.kind == "fibo2":
                        if not hasattr(diffusers, "BriaFibo2Pipeline"):
                            raise ProviderError("FIBO 2 requires diffusers>=0.42. Reinstall requirements-local.txt.", 503)
                        pipeline = diffusers.BriaFibo2Pipeline.from_pretrained(**common)
                    elif spec.kind == "qwen_edit":
                        if not hasattr(diffusers, "QwenImageEditPlusPipeline"):
                            raise ProviderError("Qwen Image Edit 2511 requires a current Diffusers version.", 503)
                        pipeline = diffusers.QwenImageEditPlusPipeline.from_pretrained(**common)
                    else:
                        raise ProviderError(f"No loader is registered for {spec.label}.", 500)

                    use_offload = cuda and _env_flag("HEROGOD_CPU_OFFLOAD", spec.cpu_offload_default)
                    if use_offload and hasattr(pipeline, "enable_model_cpu_offload"):
                        pipeline.enable_model_cpu_offload()
                    else:
                        pipeline.to(device)

                if hasattr(pipeline, "set_progress_bar_config"):
                    pipeline.set_progress_bar_config(disable=True)
                if hasattr(pipeline, "enable_attention_slicing") and spec.kind not in {"flux3", "fibo2", "qwen_edit"}:
                    pipeline.enable_attention_slicing()
                if hasattr(pipeline, "enable_vae_slicing") and spec.kind in {"auto_img2img", "fibo2", "qwen_edit"}:
                    pipeline.enable_vae_slicing()

                if spec.kind == "fibo2":
                    from huggingface_hub import hf_hub_download

                    self._assets[model_id] = {
                        "fonts": [
                            hf_hub_download(model_id, name, subfolder="fonts", token=token)
                            for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")
                        ]
                    }

                self._pipelines[model_id] = pipeline
                self._devices[model_id] = device
                self.status = "ready"
                return pipeline, torch, device
            except ProviderError as error:
                self.status = "error"
                self.last_error = str(error)
                raise
            except Exception as error:
                self.status = "error"
                self.last_error = _clean_error(error, (hf_token,))
                if spec.gated and any(text in str(error).lower() for text in ("gated", "401", "403", "access")):
                    raise ProviderError(
                        f"Access to {spec.label} was denied. Accept its conditions on the Hugging Face model page, "
                        "then provide an HF token with read access in Connection settings or Colab Secrets.",
                        403,
                    ) from error
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
        hf_token: str | None = None,
    ) -> EditResult:
        spec = get_model_spec(model_id)
        try:
            pipeline, torch, device = self._get_pipeline(model_id, hf_token)
            actual_seed = secrets.randbelow(2_147_483_647) if seed < 0 else seed
            generator = torch.Generator(device=device).manual_seed(actual_seed)
            max_size = int(os.getenv("HEROGOD_MAX_IMAGE_SIZE") or spec.default_max_size)
            max_size = min(1536, max(256, max_size))
            prepared = _resize_for_model(image, max_size, spec.size_multiple)
            actual_steps = steps or spec.default_steps
            actual_guidance = guidance or spec.default_guidance

            if spec.kind == "flux3":
                caption = prompt
                if _env_flag("HEROGOD_FLUX_UPSAMPLING"):
                    caption = self._flux_upsampler.upsample(prompt, prepared, torch, hf_token)
                elif len(prompt.strip()) < 120:
                    raise ProviderError(
                        "FLUX 3 expects a full caption of the resulting scene (subject, setting, composition, lighting, "
                        "and style), not a short edit instruction. Write a fuller caption, choose another model, or set "
                        "HEROGOD_FLUX_UPSAMPLING=1 on a runtime with enough memory for the official upsampler.",
                        422,
                    )
                with torch.inference_mode():
                    output = pipeline(
                        prompt=caption,
                        image=prepared,
                        num_inference_steps=actual_steps,
                        generator=generator,
                        output="images",
                    )[0]
            else:
                arguments: dict[str, Any] = {
                    "prompt": prompt,
                    "image": prepared,
                    "num_inference_steps": actual_steps,
                    "generator": generator,
                }
                if spec.kind == "instruct":
                    arguments["guidance_scale"] = actual_guidance
                    arguments["image_guidance_scale"] = image_guidance or spec.default_image_guidance
                elif spec.kind == "auto_img2img":
                    arguments["guidance_scale"] = actual_guidance
                    arguments["strength"] = strength or spec.default_strength
                elif spec.kind == "fibo2":
                    arguments["guidance_scale"] = 1.0
                    arguments["fonts"] = self._assets.get(model_id, {}).get("fonts")
                elif spec.kind == "qwen_edit":
                    arguments["true_cfg_scale"] = actual_guidance
                    arguments["guidance_scale"] = 1.0
                    arguments["negative_prompt"] = " "
                    arguments["num_images_per_prompt"] = 1
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
                    f"{spec.label} ran out of memory. {spec.hardware}. Lower HEROGOD_MAX_IMAGE_SIZE, restart the "
                    "runtime, or choose the lighter InstructPix2Pix model. For supported standard pipelines, "
                    "HEROGOD_CPU_OFFLOAD=1 can also help."
                )
            elif spec.gated and any(text in str(error).lower() for text in ("gated", "401", "403")):
                message = (
                    f"Access to {spec.label} was denied. Accept the model-card conditions and check your HF token."
                )
            else:
                message = f"Local inference failed: {_clean_error(error, (hf_token,))}"
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
    hf_token: str | None = None,
) -> EditResult:
    token = (hf_token or os.getenv("HF_TOKEN", "")).strip()
    if not token:
        raise ProviderError(
            "Add your HF token in Connection settings or set HF_TOKEN on the backend. Hosted inference also needs "
            "Inference Providers permission.",
            503,
        )
    spec = get_model_spec(model_id)
    if model_id not in HF_MODELS:
        raise ProviderError(
            f"{spec.label} is not enabled for hosted inference in Herogod. Select its Local / Colab option instead.",
            400,
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
            num_inference_steps=steps or spec.default_steps,
            guidance_scale=1.0 if spec.kind == "fibo2" else (guidance or spec.default_guidance),
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
            "Hugging Face inference failed. Check model/provider availability, license acceptance, token permissions, "
            f"and remaining credits. Details: {_clean_error(error, (token,))}",
            502,
        ) from error


def gemini_edit(image: Image.Image, prompt: str, model_id: str) -> EditResult:
    token = os.getenv("GEMINI_API_KEY", "").strip()
    if not token:
        raise ProviderError("GEMINI_API_KEY is not configured on the backend. See the API key guide.", 503)
    configured_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
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
