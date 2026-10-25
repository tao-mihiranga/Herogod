"""In-memory generation jobs.

A job record tracks progress so the browser can poll while a slow diffusion
run happens, and keeps the finished gallery alive for as long as the server
lives. Nothing is written to disk: restarting Colab clears the gallery.
"""
from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .config import SETTINGS
from .engine import CancelledError, EngineError, GenerationRequest, get_engine


def encode_png(image: Image.Image) -> str:
    import base64

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def thumbnail(image: Image.Image, size: int = 512) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    return copy


@dataclass
class Job:
    id: str
    prompt: str
    negative: str
    settings: dict[str, Any]
    refs: list[str] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    step: int = 0
    steps: int = 0
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    elapsed: float | None = None
    seed: int | None = None
    adapter: str | None = None
    lora_scale: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _full: Image.Image | None = field(default=None, repr=False)
    _thumb: Image.Image | None = field(default=None, repr=False)

    # -- state helpers ---------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def set_progress(self, step: int, steps: int) -> None:
        with self._lock:
            self.step = step
            self.steps = max(1, steps)
            self.progress = min(1.0, step / float(max(1, steps)))

    def attach_image(self, image: Image.Image) -> None:
        with self._lock:
            self._full = image
            self._thumb = thumbnail(image)

    def image_bytes(self, variant: str = "full") -> bytes | None:
        with self._lock:
            source = self._full if variant == "full" else self._thumb
            if source is None:
                return None
            buffer = io.BytesIO()
            source.save(buffer, format="PNG", compress_level=6)
            return buffer.getvalue()

    def thumbnail_data_url(self) -> str | None:
        with self._lock:
            if self._thumb is None:
                return None
            return encode_png(self._thumb)

    def public(self, base_url: str = "") -> dict[str, Any]:
        with self._lock:
            has_image = self._full is not None
            data: dict[str, Any] = {
                "id": self.id,
                "status": self.status,
                "prompt": self.prompt,
                "negative": self.negative,
                "settings": self.settings,
                "refs": self.refs,
                "progress": round(self.progress, 4),
                "step": self.step,
                "steps": self.steps,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "elapsed": self.elapsed,
                "seed": self.seed,
                "adapter": self.adapter,
                "lora_scale": self.lora_scale,
                "error": self.error,
                "meta": self.meta,
                "has_image": has_image,
            }
            if has_image:
                suffix = f"?v={int(self.created_at)}"
                data["url"] = f"{base_url}/api/jobs/{self.id}/image{suffix}"
                data["thumb_url"] = f"{base_url}/api/jobs/{self.id}/thumb{suffix}"
            return data


class JobStore:
    def __init__(self, keep: int = 40) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._keep = max(4, keep)

    def create(
        self,
        prompt: str,
        negative: str,
        settings: dict[str, Any],
        refs: list[str],
        steps: int,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            prompt=prompt,
            negative=negative,
            settings=settings,
            refs=refs,
            steps=steps,
        )
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                oldest_id, oldest = next(iter(self._jobs.items()))
                if oldest.status in {"queued", "running"} and len(self._jobs) <= self._keep:
                    break
                self._jobs.pop(oldest_id, None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def clear_finished(self) -> int:
        with self._lock:
            removable = [job_id for job_id, job in self._jobs.items() if job.status in {"done", "error", "cancelled"}]
            for job_id in removable:
                self._jobs.pop(job_id, None)
            return len(removable)


JOBS = JobStore(SETTINGS.keep_jobs)


def start_job(
    *,
    prompt: str,
    negative: str,
    images: list[Image.Image],
    settings: dict[str, Any],
    ref_labels: list[str],
) -> Job:
    job = JOBS.create(prompt, negative, settings, ref_labels, int(settings.get("steps", 20)))
    engine = get_engine()

    def run() -> None:
        job.status = "running"
        started = time.perf_counter()
        request = GenerationRequest(
            images=images,
            prompt=prompt,
            negative=negative,
            steps=int(settings.get("steps", 20)),
            true_cfg=float(settings.get("true_cfg", 4.0)),
            lora_scale=float(settings.get("lora_scale", 1.0)),
            adapter=str(settings.get("adapter") or ""),
            seed=int(settings.get("seed", -1)),
            size=tuple(settings.get("size") or images[0].size),  # type: ignore[arg-type]
            progress=job.set_progress,
            should_cancel=job.cancelled,
        )
        try:
            result = engine.generate(request)
            job.attach_image(result.image)
            job.seed = result.seed
            job.adapter = result.adapter
            job.lora_scale = result.lora_scale
            job.meta = result.meta
            job.progress = 1.0
            job.status = "done"
        except CancelledError:
            job.status = "cancelled"
        except EngineError as error:
            job.status = "error"
            job.error = str(error)
        except Exception as error:  # pragma: no cover - defensive
            job.status = "error"
            job.error = f"Generation failed: {error}"
        finally:
            job.elapsed = round(time.perf_counter() - started, 2)
            job.finished_at = time.time()

    threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
    return job

