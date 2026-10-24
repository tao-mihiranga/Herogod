from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from backend import providers


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.modular = False
        self.blocks = SimpleNamespace(sub_blocks={"prompt_upsampler": object()})
        self.missing_components = set()

    def load_components(self, **kwargs):
        self.calls.append(("load_components", kwargs))
        if self.modular:
            for name in providers.FLUX3_REQUIRED_COMPONENTS:
                if name not in self.missing_components:
                    setattr(self, name, object())

    def __call__(self, **kwargs):
        self.calls.append(("invoke", kwargs))
        output = Image.new("RGB", (32, 32), "green")
        return [output] if self.modular else SimpleNamespace(images=[output])

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
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)

    class Generator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class InferenceContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    torch.Generator = Generator
    torch.inference_mode = lambda: InferenceContext()
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


def test_model_catalog_contains_requested_models_without_exposing_token():
    catalog = providers.model_catalog(token_available=False)
    by_id = {model["id"]: model for model in catalog}
    assert providers.FLUX3_KLEIN in by_id
    assert providers.FIBO2_TURBO in by_id
    assert providers.QWEN_EDIT_2511 in by_id
    assert by_id[providers.FLUX3_KLEIN]["default_steps"] == 16
    assert by_id[providers.FLUX3_KLEIN]["requires_hf_token"] is True
    assert by_id[providers.FLUX3_KLEIN]["token_available"] is False
    assert by_id[providers.FIBO2_TURBO]["default_steps"] == 4
    assert by_id[providers.FIBO2_TURBO]["hosted"] is True
    assert by_id[providers.QWEN_EDIT_2511]["default_guidance"] == 4.0
    assert "token" not in by_id[providers.FLUX3_KLEIN]


def test_flux3_modular_loader_and_distilled_inference(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    monkeypatch.delenv("HEROGOD_FLUX_UPSAMPLING", raising=False)
    monkeypatch.delenv("HEROGOD_MAX_IMAGE_SIZE", raising=False)
    diffusers = ModuleType("diffusers")
    captured = {}

    class PipelineFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["from_pretrained"] = (args, kwargs)
            pipeline = FakePipeline()
            pipeline.modular = True
            return pipeline

    diffusers.ModularPipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    manager = providers.LocalModelManager()
    pipeline, _torch, device = manager._get_pipeline(providers.FLUX3_KLEIN, "test-hf-token")
    assert device == "cuda"
    assert captured["from_pretrained"][1]["token"] == "test-hf-token"
    load_call = next(call for call in pipeline.calls if call[0] == "load_components")
    assert load_call[1]["dtype"] == "bfloat16"
    assert load_call[1]["token"] == "test-hf-token"
    assert load_call[1]["names"] == list(providers.FLUX3_REQUIRED_COMPONENTS)
    assert "prompt_upsampler" not in pipeline.blocks.sub_blocks

    caption = (
        "A detailed cinematic photograph of the same subject in the same mountain meadow, now at blue hour, "
        "with cool moonlight, soft mist, preserved framing, natural textures, and a quiet atmospheric mood."
    )
    result = manager.edit(
        image=Image.new("RGB", (64, 48), "red"), prompt=caption, model_id=providers.FLUX3_KLEIN,
        steps=0, guidance=0, image_guidance=1.5, strength=0.65, seed=7, hf_token="test-hf-token",
    )
    invoke = next(call for call in pipeline.calls if call[0] == "invoke")[1]
    assert invoke["prompt"] == caption
    assert invoke["num_inference_steps"] == 16
    assert "guidance_scale" not in invoke
    assert invoke["output"] == "images"
    assert result.seed == 7


def test_flux3_rejects_partially_loaded_modular_pipeline(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    diffusers = ModuleType("diffusers")

    class PipelineFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            pipeline = FakePipeline()
            pipeline.modular = True
            pipeline.missing_components = {"vae"}
            return pipeline

    diffusers.ModularPipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    manager = providers.LocalModelManager()
    with pytest.raises(providers.ProviderError) as error:
        manager._get_pipeline(providers.FLUX3_KLEIN)
    assert error.value.status_code == 503
    assert "vae" in str(error.value)
    assert manager.status == "error"


def test_flux3_rejects_short_instruction_without_upsampler(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    monkeypatch.delenv("HEROGOD_FLUX_UPSAMPLING", raising=False)
    diffusers = ModuleType("diffusers")

    class PipelineFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            pipeline = FakePipeline()
            pipeline.modular = True
            return pipeline

    diffusers.ModularPipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    manager = providers.LocalModelManager()
    with pytest.raises(providers.ProviderError) as error:
        manager.edit(
            image=Image.new("RGB", (64, 48)), prompt="make it blue", model_id=providers.FLUX3_KLEIN,
            steps=0, guidance=0, image_guidance=1.5, strength=0.65, seed=1,
        )
    assert error.value.status_code == 422
    assert "full caption" in str(error.value)


def test_fibo2_loader_and_turbo_defaults(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    diffusers = ModuleType("diffusers")
    captured = {}

    class PipelineFactory:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return FakePipeline()

    diffusers.BriaFibo2Pipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda _repo, name, **_kwargs: f"/models/{name}")
    manager = providers.LocalModelManager()
    result = manager.edit(
        image=Image.new("RGB", (64, 48)), prompt="make the background snowy", model_id=providers.FIBO2_TURBO,
        steps=0, guidance=0, image_guidance=1.5, strength=0.65, seed=11, hf_token="test-hf-token",
    )
    pipeline = manager._pipelines[providers.FIBO2_TURBO]
    invoke = next(call for call in pipeline.calls if call[0] == "invoke")[1]
    assert captured["token"] == "test-hf-token"
    assert captured["torch_dtype"] == "bfloat16"
    assert invoke["num_inference_steps"] == 4
    assert invoke["guidance_scale"] == 1.0
    assert invoke["fonts"] == ["/models/Aileron-Regular.otf", "/models/DejaVuSans.ttf"]
    assert result.model == providers.FIBO2_TURBO


def test_qwen_edit_uses_plus_pipeline(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    diffusers = ModuleType("diffusers")
    captured = {}

    class PipelineFactory:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return FakePipeline()

    diffusers.QwenImageEditPlusPipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    manager = providers.LocalModelManager()
    pipeline, _torch, device = manager._get_pipeline(providers.QWEN_EDIT_2511)
    assert device == "cuda"
    assert captured["torch_dtype"] == "bfloat16"
    assert ("offload",) in pipeline.calls
    assert ("vae",) in pipeline.calls


def test_flux_prompt_upsampler_uses_token_and_official_helper(monkeypatch: pytest.MonkeyPatch):
    fake_torch(monkeypatch, cuda=True)
    torch = sys.modules["torch"]
    torch.no_grad = torch.inference_mode
    captured = {}

    class Processor:
        pass

    class Model:
        def eval(self):
            return self

    class ProcessorFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["processor"] = (model_id, kwargs)
            return Processor()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["model"] = (model_id, kwargs)
            return Model()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = ProcessorFactory
    transformers.Qwen3_5ForConditionalGeneration = ModelFactory
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    for name in ("diffusers", "diffusers.modular_pipelines", "diffusers.modular_pipelines.flux3"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    helper = ModuleType("diffusers.modular_pipelines.flux3.prompt_upsampler")

    def upsample_prompt(model, processor, **kwargs):
        captured["helper"] = (model, processor, kwargs)
        return "Expanded full scene caption"

    helper.upsample_prompt = upsample_prompt
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    source = Image.new("RGB", (32, 32))
    caption = providers.FluxPromptUpsampler().upsample("make it blue", source, torch, "test-hf-token")

    assert caption == "Expanded full scene caption"
    assert captured["processor"][1]["token"] == "test-hf-token"
    assert captured["model"][1]["token"] == "test-hf-token"
    assert captured["model"][1]["device_map"] == "auto"
    assert captured["model"][1]["dtype"] == "bfloat16"
    assert captured["helper"][2]["images"] == [source]
    assert captured["helper"][2]["request"] == "make it blue"


def test_hosted_fibo_uses_hf_token_and_turbo_settings(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeInferenceClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def image_to_image(self, _image, **kwargs):
            captured["request"] = kwargs
            return Image.new("RGB", (32, 32), "blue")

    monkeypatch.setattr("huggingface_hub.InferenceClient", FakeInferenceClient)
    result = providers.huggingface_edit(
        Image.new("RGB", (64, 48)), "make it blue", providers.FIBO2_TURBO,
        steps=0, guidance=0, hf_token="test-hf-token",
    )
    assert captured["client"]["api_key"] == "test-hf-token"
    assert captured["request"]["num_inference_steps"] == 4
    assert captured["request"]["guidance_scale"] == 1.0
    assert result.provider == "huggingface"
