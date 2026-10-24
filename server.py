"""Herogod web and API server.

Run with: python server.py
The same process serves the dependency-free frontend and the JSON image API.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import importlib.util
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from backend import providers
from backend.face_swap import face_swap_configured, swap_face
from backend.providers import ProviderError

ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_PIL_FORMATS = {"PNG", "JPEG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("herogod")

app = FastAPI(
    title="Herogod API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Herogod-Token", "X-HF-Token"],
    max_age=86400,
)
try:
    INFERENCE_LIMITER = asyncio.Semaphore(max(1, int(os.getenv("HEROGOD_MAX_CONCURRENCY", "1"))))
except ValueError:
    INFERENCE_LIMITER = asyncio.Semaphore(1)


@app.middleware("http")
async def security_and_size_headers(request: Request, call_next: Any) -> Any:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request is too large. Each image must be 15 MB or smaller."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path == "/api/docs":
        # FastAPI's Swagger page uses its official jsDelivr assets and inline bootstrap.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://cdn.jsdelivr.net; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            "connect-src 'self' http: https:; font-src 'self'; object-src 'none'; base-uri 'self'"
        )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


async def authorize(request: Request) -> None:
    expected = os.getenv("HEROGOD_API_TOKEN", "").strip()
    if not expected:
        return
    authorization = request.headers.get("authorization", "")
    bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    supplied = request.headers.get("x-herogod-token", "").strip() or bearer
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="A valid Herogod access token is required.")


@app.exception_handler(ProviderError)
async def provider_error_handler(_request: Request, error: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})


def module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def supplied_hf_token(request: Request) -> str | None:
    token = request.headers.get("x-hf-token", "").strip()
    if len(token) > 512:
        raise HTTPException(status_code=400, detail="The HF token header is too long.")
    return token or None


@app.get("/api/health", dependencies=[Depends(authorize)])
async def health(request: Request) -> dict[str, Any]:
    hf_token = supplied_hf_token(request)
    environment_hf_token = os.getenv("HF_TOKEN", "").strip()
    local_ready = module_installed("torch") and module_installed("diffusers")
    huggingface_ready = bool(hf_token or environment_hf_token) and module_installed("huggingface_hub")
    gemini_ready = bool(os.getenv("GEMINI_API_KEY")) and module_installed("google.genai")
    return {
        "app": "Herogod",
        "version": "1.0.0",
        "status": "ok",
        "providers": [
            {
                "id": "local",
                "name": "Local / Colab",
                "configured": local_ready,
                "default_model": os.getenv("HEROGOD_MODEL", providers.INSTRUCT_PIX2PIX),
                "requires_api_key": False,
                "provider_filters": False,
            },
            {
                "id": "huggingface",
                "name": "Hugging Face",
                "configured": huggingface_ready,
                "default_model": providers.INSTRUCT_PIX2PIX,
                "requires_api_key": True,
                "provider_filters": True,
            },
            {
                "id": "gemini",
                "name": "Gemini",
                "configured": gemini_ready,
                "default_model": os.getenv("GEMINI_MODEL", providers.DEFAULT_GEMINI_MODEL),
                "requires_api_key": True,
                "provider_filters": True,
            },
        ],
        "capabilities": {
            "image_edit": True,
            "face_swap": face_swap_configured() and module_installed("insightface"),
            "server_side_history": False,
        },
        "models": providers.model_catalog(token_available=bool(hf_token or environment_hf_token)),
        "runtime": {"local_model_status": providers.LOCAL_MANAGER.status},
    }


async def read_image(upload: UploadFile, label: str = "Image") -> Image.Image:
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        await upload.close()
        raise HTTPException(status_code=415, detail=f"{label} must be a PNG, JPG, or WebP file.")
    try:
        data = await upload.read(MAX_IMAGE_BYTES + 1)
    finally:
        # Starlette may spool large uploads to a temporary file; close it promptly.
        await upload.close()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} is larger than 15 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_PIL_FORMATS:
            raise HTTPException(status_code=415, detail=f"{label} format is not supported.")
        if image.width < 16 or image.height < 16:
            raise HTTPException(status_code=422, detail=f"{label} must be at least 16 × 16 pixels.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail=f"{label} has too many pixels (maximum {MAX_IMAGE_PIXELS:,}).")
        image = ImageOps.exif_transpose(image)
        image.load()
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError as error:
        raise HTTPException(status_code=413, detail=f"{label} is too large to decode safely.") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"{label} could not be decoded as a valid image.") from error


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


@app.post("/api/edit", dependencies=[Depends(authorize)])
async def edit_image(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1, max_length=2000),
    provider: str = Form("local"),
    model: str = Form(providers.INSTRUCT_PIX2PIX, max_length=200),
    steps: int = Form(0, ge=0, le=50),
    guidance: float = Form(0, ge=0, le=15),
    image_guidance: float = Form(1.5, ge=0.5, le=3),
    strength: float = Form(0.65, ge=0.1, le=1),
    seed: int = Form(-1, ge=-1, le=2_147_483_647),
) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Describe the edit you want to make.")
    if provider not in {"local", "huggingface", "gemini"}:
        raise HTTPException(status_code=400, detail="Unknown provider. Choose local, huggingface, or gemini.")
    source = await read_image(image)
    hf_token = supplied_hf_token(request)
    started = time.perf_counter()
    async with INFERENCE_LIMITER:
        if provider == "local":
            result = await asyncio.to_thread(
                providers.local_edit,
                image=source,
                prompt=prompt,
                model_id=model,
                steps=steps,
                guidance=guidance,
                image_guidance=image_guidance,
                strength=strength,
                seed=seed,
                hf_token=hf_token,
            )
        elif provider == "huggingface":
            result = await asyncio.to_thread(
                providers.huggingface_edit,
                image=source,
                prompt=prompt,
                model_id=model,
                steps=steps,
                guidance=guidance,
                hf_token=hf_token,
            )
        else:
            result = await asyncio.to_thread(providers.gemini_edit, image=source, prompt=prompt, model_id=model)
        encoded = await asyncio.to_thread(encode_image, result.image)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info("Image edit complete provider=%s model=%s elapsed_ms=%s", provider, model, elapsed_ms)
    return {
        "image": encoded,
        "provider": result.provider,
        "model": result.model,
        "seed": result.seed,
        "elapsed_ms": elapsed_ms,
    }


@app.post("/api/face-swap", dependencies=[Depends(authorize)])
async def face_swap(
    image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    consent: bool = Form(False),
) -> dict[str, Any]:
    if not consent:
        raise HTTPException(status_code=400, detail="Consent confirmation is required for face swapping.")
    target = await read_image(image, "Target image")
    reference = await read_image(reference_image, "Reference image")
    started = time.perf_counter()
    async with INFERENCE_LIMITER:
        result = await asyncio.to_thread(swap_face, target=target, reference=reference)
        encoded = await asyncio.to_thread(encode_image, result)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info("Face swap complete elapsed_ms=%s", elapsed_ms)
    return {
        "image": encoded,
        "provider": "local",
        "model": "insightface/inswapper_128",
        "seed": None,
        "elapsed_ms": elapsed_ms,
    }


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/styles.css", include_in_schema=False)
async def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def javascript() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="text/javascript")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(ROOT / "favicon.svg", media_type="image/svg+xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Herogod web and API server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("HEROGOD_API_TOKEN"):
        logger.warning("The server is network-accessible without HEROGOD_API_TOKEN. Set a token before using a public tunnel.")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=os.getenv("LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()
