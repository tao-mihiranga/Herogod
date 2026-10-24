"""Optional, local InsightFace face swapping.

The caller must explicitly attest to consent. Model weights are downloaded by the
provided setup script, not silently at request time. Review InsightFace's licenses
before use, especially for commercial projects.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .providers import ProviderError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "inswapper_128.onnx"


def face_model_path() -> Path:
    return Path(os.getenv("HEROGOD_FACE_MODEL", str(DEFAULT_MODEL_PATH))).expanduser().resolve()


def face_swap_configured() -> bool:
    return face_model_path().is_file()


class FaceSwapEngine:
    def __init__(self) -> None:
        self._analyser: Any = None
        self._swapper: Any = None
        self._cv2: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._analyser is not None and self._swapper is not None:
            return
        with self._lock:
            if self._analyser is not None and self._swapper is not None:
                return
            model_path = face_model_path()
            if not model_path.is_file():
                raise ProviderError(
                    "Face-swap weights are not installed. Run python scripts/setup_face_swap.py on the backend, then restart it.",
                    503,
                )
            try:
                import cv2
                import insightface
                import onnxruntime
            except ImportError as error:
                raise ProviderError(
                    "Face-swap dependencies are missing. Run: pip install -r requirements-face.txt", 503
                ) from error

            try:
                available = onnxruntime.get_available_providers()
                providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
                if not providers:
                    providers = ["CPUExecutionProvider"]
                use_gpu = "CUDAExecutionProvider" in providers
                insightface_home = os.getenv("HEROGOD_INSIGHTFACE_HOME", str(PROJECT_ROOT / "models" / "insightface"))
                analyser = insightface.app.FaceAnalysis(
                    name=os.getenv("HEROGOD_FACE_DETECTOR", "buffalo_l"),
                    root=insightface_home,
                    providers=providers,
                )
                analyser.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
                swapper = insightface.model_zoo.get_model(str(model_path), providers=providers)
                self._cv2 = cv2
                self._analyser = analyser
                self._swapper = swapper
            except Exception as error:
                raise ProviderError(f"Could not initialize InsightFace: {str(error)[:500]}", 500) from error

    @staticmethod
    def _largest_face(faces: list[Any]) -> Any:
        return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))

    def _to_bgr(self, image: Image.Image, max_size: int = 2048) -> Any:
        resized = image.copy()
        resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return self._cv2.cvtColor(np.asarray(resized.convert("RGB")), self._cv2.COLOR_RGB2BGR)

    def swap(self, target: Image.Image, reference: Image.Image) -> Image.Image:
        self._load()
        try:
            target_bgr = self._to_bgr(target)
            reference_bgr = self._to_bgr(reference, max_size=1024)
            reference_faces = self._analyser.get(reference_bgr)
            target_faces = self._analyser.get(target_bgr)
            if not reference_faces:
                raise ProviderError("No face was detected in the reference image. Use a clear, front-facing photo.", 422)
            if not target_faces:
                raise ProviderError("No face was detected in the target image. Use a clear, front-facing photo.", 422)
            source_face = self._largest_face(reference_faces)
            target_face = self._largest_face(target_faces)
            result = self._swapper.get(target_bgr, target_face, source_face, paste_back=True)
            return Image.fromarray(self._cv2.cvtColor(result, self._cv2.COLOR_BGR2RGB))
        except ProviderError:
            raise
        except Exception as error:
            raise ProviderError(f"Face swap failed: {str(error)[:500]}", 500) from error


ENGINE = FaceSwapEngine()


def swap_face(target: Image.Image, reference: Image.Image) -> Image.Image:
    return ENGINE.swap(target, reference)
