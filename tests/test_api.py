from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from backend.providers import EditResult


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HEROGOD_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(server.app) as test_client:
        yield test_client


def upload(color: str = "red", size: tuple[int, int] = (64, 48)) -> tuple[str, io.BytesIO, str]:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return "source.png", buffer, "image/png"


def test_index_and_health(client: TestClient):
    index = client.get("/")
    assert index.status_code == 200
    assert "Herogod" in index.text
    assert "app.js" in index.text

    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "Herogod"
    assert payload["status"] == "ok"
    assert payload["capabilities"]["server_side_history"] is False
    assert {provider["id"] for provider in payload["providers"]} == {"local", "huggingface", "gemini"}
    assert all("credential" not in provider for provider in payload["providers"])


def test_access_token_is_enforced(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEROGOD_API_TOKEN", "test-secret")
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/api/health", headers={"X-Herogod-Token": "test-secret"})
    assert response.status_code == 200


def test_hf_token_enables_provider_metadata_without_being_returned(client: TestClient):
    response = client.get("/api/health", headers={"X-HF-Token": "test-hf-token"})
    assert response.status_code == 200
    payload = response.json()
    hf = next(provider for provider in payload["providers"] if provider["id"] == "huggingface")
    assert hf["configured"] is True
    flux = next(model for model in payload["models"] if model["id"] == "black-forest-labs/flux-3-image-klein")
    assert flux["requires_hf_token"] is True
    assert flux["token_available"] is True
    assert "test-hf-token" not in response.text


def test_rejects_unsupported_upload_and_blank_prompt(client: TestClient):
    unsupported = client.post(
        "/api/edit",
        files={"image": ("hello.txt", io.BytesIO(b"not an image"), "text/plain")},
        data={"prompt": "make it blue"},
    )
    assert unsupported.status_code == 415

    blank = client.post(
        "/api/edit",
        files={"image": upload()},
        data={"prompt": "   ", "provider": "local"},
    )
    assert blank.status_code == 422


def test_rejects_unknown_provider(client: TestClient):
    response = client.post(
        "/api/edit",
        files={"image": upload()},
        data={"prompt": "make it blue", "provider": "unknown"},
    )
    assert response.status_code == 400


def test_local_edit_returns_png_data_url(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_edit(**kwargs):
        captured.update(kwargs)
        return EditResult(Image.new("RGB", (32, 24), "blue"), "local", kwargs["model_id"], 1234)

    monkeypatch.setattr(server.providers, "local_edit", fake_edit)
    response = client.post(
        "/api/edit",
        files={"image": upload()},
        data={
            "prompt": "make the background blue",
            "provider": "local",
            "model": "timbrooks/instruct-pix2pix",
            "steps": "20",
            "guidance": "7.5",
            "image_guidance": "1.5",
            "strength": "0.65",
            "seed": "-1",
        },
        headers={"X-HF-Token": "test-hf-token"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["provider"] == "local"
    assert payload["seed"] == 1234
    assert payload["elapsed_ms"] >= 0
    assert captured["prompt"] == "make the background blue"
    assert captured["image"].size == (64, 48)
    assert captured["hf_token"] == "test-hf-token"
    assert "test-hf-token" not in response.text
    prefix, encoded = payload["image"].split(",", 1)
    assert prefix == "data:image/png;base64"
    result = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert result.size == (32, 24)
    assert result.format == "PNG"


def test_missing_hosted_provider_key_is_a_clear_error(client: TestClient):
    response = client.post(
        "/api/edit",
        files={"image": upload()},
        data={"prompt": "make it blue", "provider": "gemini", "model": "gemini-2.5-flash-image"},
    )
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_face_swap_requires_consent(client: TestClient):
    response = client.post(
        "/api/face-swap",
        files={"image": upload("red"), "reference_image": upload("green")},
        data={"consent": "false"},
    )
    assert response.status_code == 400
    assert "Consent" in response.json()["detail"]


def test_consented_face_swap_returns_image(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fake_swap(target: Image.Image, reference: Image.Image) -> Image.Image:
        assert target.size == (64, 48)
        assert reference.size == (64, 48)
        return Image.new("RGB", target.size, "purple")

    monkeypatch.setattr(server, "swap_face", fake_swap)
    response = client.post(
        "/api/face-swap",
        files={"image": upload("red"), "reference_image": upload("green")},
        data={"consent": "true"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == "insightface/inswapper_128"
    assert payload["image"].startswith("data:image/png;base64,")
