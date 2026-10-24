from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from backend import providers


class FakePipeline:
    def __init__(self):
        self.calls = []

    def set_progress_bar_config(self, **kwargs):
        self.calls.append(("progress", kwargs))

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def enable_attention_slicing(self):
        self.calls.append(("attention",))

    def enable_vae_slicing(self):
        self.calls.append(("vae",))

    def enable_model_cpu_offload(self):
        self.calls.append(("offload",))


def fake_torch(monkeypatch: pytest.MonkeyPatch, cuda: bool):
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_instruct_pix2pix_uses_stable_diffusion_pipeline(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=False)
    diffusers = ModuleType("diffusers")
    captured = {}

    class PipelineFactory:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return FakePipeline()

    diffusers.StableDiffusionInstructPix2PixPipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    manager = providers.LocalModelManager()
    pipeline, _torch, device = manager._get_pipeline(providers.INSTRUCT_PIX2PIX)

    assert device == "cpu"
    assert captured["pretrained_model_name_or_path"] == providers.INSTRUCT_PIX2PIX
    assert captured["use_safetensors"] is True
    assert captured["torch_dtype"] == "float32"
    assert "safety_checker" not in captured
    assert ("to", "cpu") in pipeline.calls
    assert manager.status == "ready"


def test_sdxl_uses_fp16_variant_and_offload_on_cuda(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    diffusers = ModuleType("diffusers")
    captured = {}

    class PipelineFactory:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return FakePipeline()

    diffusers.AutoPipelineForImage2Image = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    manager = providers.LocalModelManager()
    pipeline, _torch, device = manager._get_pipeline(providers.SDXL)

    assert device == "cuda"
    assert captured["variant"] == "fp16"
    assert captured["torch_dtype"] == "float16"
    assert ("offload",) in pipeline.calls
    assert ("vae",) in pipeline.calls


def test_unknown_local_model_fails_before_ml_imports():
    manager = providers.LocalModelManager()
    with pytest.raises(providers.ProviderError) as error:
        manager._get_pipeline("example/not-supported")
    assert error.value.status_code == 400
    assert "Unsupported local model" in str(error.value)
