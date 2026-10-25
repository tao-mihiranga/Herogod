"""Herogod server: FastAPI JSON API + the static studio frontend.

    python server.py                  # http://0.0.0.0:8000
    python server.py --mock           # no torch, scripted preview output

The same process serves `web/`, so the browser talks to relative URLs and no
CORS or build step is involved.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import APP_NAME, APP_VERSION, SETTINGS, WEB_DIR, catalog
from app.engine import EngineError, get_engine
from app.jobs import JOBS, start_job
from app.prompts import RequestProblem, parse_seed, resolve_prompt, target_size

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("herogod")

MAX_IMAGE_BYTES = SETTINGS.max_image_bytes
MAX_REQUEST_BYTES = SETTINGS.max_request_bytes
MAX_IMAGE_PIXELS = SETTINGS.max_image_pixels
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_PIL_FORMATS = {"PNG", "JPEG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title=f"{APP_NAME} API",
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Herogod-Token"],
    max_age=86400,
)

@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Any:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request is too large. Each image must be 15 MB or smaller."},
                )
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    is_image = re.search(r"^/api/jobs/[\w-]+/(image|thumb)$", request.url.path)
    if request.url.path.startswith("/api/") and not is_image:
        response.headers["Cache-Control"] = "no-store"
    elif is_image:
        response.headers["Cache-Control"] = "private, max-age=86400"
    if request.url.path in {"/", "/index.html"}:
        response.headers["Cache-Control"] = "no-cache"
    if request.url.path.startswith("/api/docs"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://cdn.jsdelivr.net; "
            "connect-src 'self'; object-src 'none'; base-uri 'self'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self' http: https:; media-src 'self' blob:; "
            "font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors *"
        )
    return response


def token_valid(supplied: str) -> bool:
    expected = SETTINGS.api_token
    if not expected:
        return True
    supplied = (supplied or "").strip()
    return bool(supplied) and hmac.compare_digest(supplied, expected)


async def authorize(request: Request, token: str | None = Query(default=None)) -> None:
    if not SETTINGS.api_token:
        return
    header = request.headers.get("x-herogod-token", "").strip()
    authorization = request.headers.get("authorization", "")
    bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if token_valid(header or bearer or (token or "")):
        return
    raise HTTPException(status_code=401, detail="A valid Herogod access token is required.")


@app.exception_handler(EngineError)
async def engine_error_handler(_request: Request, error: EngineError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})


@app.exception_handler(RequestProblem)
async def request_problem_handler(_request: Request, error: RequestProblem) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})


def read_image_bytes(data: bytes, label: str) -> Image.Image:
    if not data:
        raise RequestProblem(f"{label} is empty.", 400)
    if len(data) > MAX_IMAGE_BYTES:
        raise RequestProblem(f"{label} is larger than 15 MB.", 413)
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_PIL_FORMATS:
            raise RequestProblem(f"{label} must be a PNG, JPG or WebP file.", 415)
        if image.width < 32 or image.height < 32:
            raise RequestProblem(f"{label} must be at least 32 x 32 pixels.", 422)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise RequestProblem(f"{label} has too many pixels (maximum {MAX_IMAGE_PIXELS:,}).", 413)
        image = ImageOps.exif_transpose(image)
        image.load()
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")
    except RequestProblem:
        raise
    except Image.DecompressionBombError as error:
        raise RequestProblem(f"{label} is too large to decode safely.", 413) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RequestProblem(f"{label} could not be decoded as an image.", 422) from error


async def read_upload(upload: UploadFile, label: str) -> Image.Image:
    if upload.content_type and upload.content_type not in ALLOWED_IMAGE_TYPES:
        await upload.close()
        raise RequestProblem(f"{label} must be a PNG, JPG or WebP file.", 415)
    try:
        data = await upload.read(MAX_IMAGE_BYTES + 1)
    finally:
        await upload.close()
    return await asyncio.to_thread(read_image_bytes, data, label)


def parse_styles(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RequestProblem("The styles payload is not valid JSON.", 400) from error
        if not isinstance(value, list):
            raise RequestProblem("The styles payload must be a list.", 400)
        return [str(item) for item in value]
    return [part.strip() for part in raw.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", include_in_schema=False)
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "token_required": bool(SETTINGS.api_token),
        "mock": SETTINGS.mock,
        "engine": get_engine().status()["state"],
    }


@app.get("/api/config", dependencies=[Depends(authorize)])
async def config() -> dict[str, Any]:
    data = SETTINGS.public()
    data.update(catalog())
    return data


@app.get("/api/engine", dependencies=[Depends(authorize)])
async def engine_status() -> dict[str, Any]:
    return await asyncio.to_thread(get_engine().status)


@app.post("/api/engine/load", dependencies=[Depends(authorize)])
async def engine_load() -> dict[str, Any]:
    engine = get_engine()
    await asyncio.to_thread(engine._ensure_loaded)
    return await asyncio.to_thread(engine.status)


@app.post("/api/engine/unload", dependencies=[Depends(authorize)])
async def engine_unload() -> dict[str, Any]:
    engine = get_engine()
    result = await asyncio.to_thread(engine.unload)
    result["status"] = await asyncio.to_thread(engine.status)
    return result


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.post("/api/jobs", dependencies=[Depends(authorize)], status_code=202)
async def create_job(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    refs: list[UploadFile] = File(default=[]),
    negative: str = Form(""),
    styles: str = Form(""),
    steps: int = Form(20),
    true_cfg: float = Form(4.0),
    lora_scale: float = Form(1.0),
    adapter: str = Form(""),
    seed: int = Form(-1),
    aspect: str = Form("match"),
    width: int = Form(0),
    height: int = Form(0),
) -> dict[str, Any]:
    resolved = await asyncio.to_thread(resolve_prompt, prompt, await asyncio.to_thread(parse_styles, styles), negative)

    steps = max(1, min(int(steps), SETTINGS.max_steps))
    true_cfg = max(0.0, min(float(true_cfg), 20.0))
    lora_scale = max(0.0, min(float(lora_scale), 2.0))

    if len(refs) > SETTINGS.max_refs:
        raise RequestProblem(f"At most {SETTINGS.max_refs} reference images are supported.", 400)

    source = await read_upload(image, "Base image")
    reference_images = [await read_upload(upload, f"Reference {index + 1}") for index, upload in enumerate(refs)]
    images = [source, *reference_images]

    size = await asyncio.to_thread(
        target_size, source, aspect, int(width) or None, int(height) or None, SETTINGS.max_pixels
    )

    settings = {
        "steps": steps,
        "true_cfg": round(true_cfg, 2),
        "lora_scale": round(lora_scale, 2),
        "adapter": adapter.strip(),
        "seed": parse_seed(seed),
        "aspect": aspect,
        "size": [size[0], size[1]],
        "styles": resolved.style_ids,
        "engine": get_engine().name,
    }
    labels = ["base", *[f"ref{i + 1}" for i in range(len(reference_images))]]
    job = start_job(
        prompt=resolved.prompt,
        negative=resolved.negative,
        images=images,
        settings=settings,
        ref_labels=labels,
    )
    logger.info("Job %s queued (%sx%s, %s steps, adapter=%s)", job.id, size[0], size[1], steps, settings["adapter"])
    return job.public()


@app.get("/api/jobs", dependencies=[Depends(authorize)])
async def list_jobs() -> dict[str, Any]:
    jobs = [job.public() for job in reversed(JOBS.all())]
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(authorize)])
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return job.public()


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job.")
    job.cancel()
    return job.public()


@app.get("/api/jobs/{job_id}/image", dependencies=[Depends(authorize)], include_in_schema=False)
async def job_image(job_id: str) -> Response:
    return image_response(job_id, "full")


@app.get("/api/jobs/{job_id}/thumb", dependencies=[Depends(authorize)], include_in_schema=False)
async def job_thumb(job_id: str) -> Response:
    return image_response(job_id, "thumb")


def image_response(job_id: str, variant: str) -> Response:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job.")
    data = job.image_bytes(variant)
    if data is None:
        raise HTTPException(status_code=404, detail="This job has no image yet.")
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


@app.delete("/api/jobs", dependencies=[Depends(authorize)])
async def clear_jobs() -> dict[str, Any]:
    removed = JOBS.clear_finished()
    return {"removed": removed, "remaining": len(JOBS.all())}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


def static_file(relative: str, media_type: str | None = None) -> FileResponse:
    path = (WEB_DIR / relative).resolve()
    if WEB_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path, media_type=media_type)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return static_file("index.html", "text/html; charset=utf-8")


@app.get("/index.html", include_in_schema=False)
async def index_alias() -> FileResponse:
    return static_file("index.html", "text/html; charset=utf-8")


@app.get("/styles.css", include_in_schema=False)
async def styles() -> FileResponse:
    return static_file("styles.css", "text/css; charset=utf-8")


@app.get("/app.js", include_in_schema=False)
async def app_js() -> FileResponse:
    return static_file("app.js", "text/javascript; charset=utf-8")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    path = Path(__file__).resolve().parent / "favicon.svg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path, media_type="image/svg+xml")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run the {APP_NAME} studio")
    parser.add_argument("--host", default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    parser.add_argument("--mock", action="store_true", help="Use the mock engine (no GPU, no downloads)")
    parser.add_argument("--no-token", action="store_true", help="Disable the access token check (trusted networks only)")
    parser.add_argument("--preload", action="store_true", help="Load the model during startup instead of on first use")
    args = parser.parse_args()

    if args.mock:
        os.environ["HEROGOD_MOCK"] = "1"
        SETTINGS.mock = True
    if args.no_token:
        SETTINGS.api_token = ""

    logging.getLogger("herogod").info("%s %s starting (engine=%s)", APP_NAME, APP_VERSION, "mock" if SETTINGS.mock else "qwen")
    if SETTINGS.api_token:
        logger.info("Access token protection is ON.")
    elif args.host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning("No HEROGOD_API_TOKEN set: the API is open to anyone who can reach this port.")

    if args.preload and not SETTINGS.mock:
        async def preload() -> None:
            try:
                await asyncio.to_thread(get_engine()._ensure_loaded)
                logger.info("Model preloaded.")
            except Exception as error:  # pragma: no cover - startup diagnostics
                logger.error("Model preload failed: %s", error)

        @app.on_event("startup")
        async def _startup() -> None:
            asyncio.create_task(preload())

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=os.getenv("LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()
