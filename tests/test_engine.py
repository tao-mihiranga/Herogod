"""Engine unit tests that never load a model."""
from __future__ import annotations

from PIL import Image

from app.engine import GenerationRequest, GenerationResult, MockEngine, QwenEngine, _pick, resolve_seed


def test_resolve_seed_is_random_for_negative():
    seeds = {resolve_seed(-1) for _ in range(20)}
    assert len(seeds) > 1
    assert all(0 <= seed < 2_147_483_647 for seed in seeds)
    assert resolve_seed(1234) == 1234


def test_pick_filters_unsupported_kwargs():
    params = ["prompt", "images", "num_inference_steps"]
    picked = _pick(params, prompt="a", images=[1], num_inference_steps=4, guidance_scale=7)
    assert picked == {"prompt": "a", "images": [1], "num_inference_steps": 4}


def test_mock_engine_generates_and_reports_progress():
    engine = MockEngine()
    steps: list[tuple[int, int]] = []
    request = GenerationRequest(
        images=[Image.new("RGB", (256, 256), (10, 20, 30))],
        prompt="test",
        steps=4,
        size=(320, 256),
        seed=7,
        progress=lambda step, total: steps.append((step, total)),
    )
    result = engine.generate(request)
    assert isinstance(result, GenerationResult)
    assert result.image.size == (320, 256)
    assert result.image.mode == "RGB"
    assert result.seed == 7
    assert result.meta["mock"] is True
    assert steps and steps[-1] == (steps[-1][0], steps[-1][1])
    assert engine.status()["state"] == "ready"


def test_mock_engine_unload_is_a_noop():
    engine = MockEngine()
    assert engine.unload()["unloaded"] is False


def test_qwen_engine_status_without_torch():
    status = QwenEngine().status()
    assert status["engine"] == "qwen"
    assert status["state"] == "not_loaded"
    assert status["adapters"]
    assert status["adapters"][0]["name"]
    assert isinstance(status["torch_installed"], bool)
    assert status["hf_token"] in (True, False)  # a flag, never the token itself
