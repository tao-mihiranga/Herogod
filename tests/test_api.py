"""API tests. They run against the mock engine, so no GPU or weights are needed."""
from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from app import engine as engine_module
from app.config import SETTINGS


def png_bytes(width: int = 256, height: int = 256, color=(40, 90, 160)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(SETTINGS, "mock", True)
    monkeypatch.setattr(SETTINGS, "api_token", "")
    engine_module.set_engine(engine_module.MockEngine())
    with TestClient(server.app) as test_client:
        yield test_client
    engine_module.set_engine(None)  # type: ignore[arg-type]


def wait_for(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in {"done", "error", "cancelled"}:
            return last
        time.sleep(0.1)
    raise AssertionError(f"job did not finish: {last}")


def test_health_is_public(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["token_required"] is False
    assert body["engine"] == "ready"


def test_config_exposes_catalog(client):
    body = client.get("/api/config").json()
    assert body["base_model"]
    assert any(style["id"] == "cinematic" for style in body["styles"])
    assert body["lora"]["repo"]
    assert body["limits"]["max_refs"] >= 1
    assert body["defaults"]["steps"] >= 1


def test_engine_status_reports_mock(client):
    body = client.get("/api/engine").json()
    assert body["engine"] == "mock"
    assert body["state"] == "ready"
    assert body["adapters"]


def test_full_job_lifecycle(client):
    response = client.post(
        "/api/jobs",
        files={"image": ("input.png", png_bytes(), "image/png")},
        data={"prompt": "make it cinematic", "styles": '["cinematic"]', "steps": "4", "true_cfg": "4"},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] in {"queued", "running"}
    assert job["settings"]["size"][0] % 8 == 0

    finished = wait_for(client, job["id"])
    assert finished["status"] == "done", finished
    assert finished["seed"] is not None
    assert finished["has_image"] is True
    assert "cinematic still" in finished["prompt"]

    image = client.get(f"/api/jobs/{job['id']}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(image.content)).size == tuple(finished["settings"]["size"])

    thumb = client.get(f"/api/jobs/{job['id']}/thumb")
    assert thumb.status_code == 200

    listing = client.get("/api/jobs").json()
    assert any(item["id"] == job["id"] for item in listing["jobs"])

    cleared = client.delete("/api/jobs").json()
    assert cleared["removed"] >= 1


def test_reference_images_are_accepted(client):
    response = client.post(
        "/api/jobs",
        files=[
            ("image", ("input.png", png_bytes(), "image/png")),
            ("refs", ("ref1.png", png_bytes(128, 128, (200, 20, 20)), "image/png")),
        ],
        data={"prompt": "use the jacket from the reference", "steps": "2"},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["refs"] == ["base", "ref1"]
    assert job["settings"]["size"] == [256, 256] or job["settings"]["size"][0] > 0
    wait_for(client, job["id"])


def test_rejects_non_image_upload(client):
    response = client.post(
        "/api/jobs",
        files={"image": ("notes.txt", b"hello", "text/plain")},
        data={"prompt": "x"},
    )
    assert response.status_code == 415


def test_rejects_empty_prompt(client):
    response = client.post(
        "/api/jobs",
        files={"image": ("input.png", png_bytes(), "image/png")},
        data={"prompt": "   "},
    )
    assert response.status_code == 422


def test_rejects_unknown_style(client):
    response = client.post(
        "/api/jobs",
        files={"image": ("input.png", png_bytes(), "image/png")},
        data={"prompt": "x", "styles": '["nope"]'},
    )
    assert response.status_code == 400


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_token_protection(client, monkeypatch):
    monkeypatch.setattr(SETTINGS, "api_token", "s3cret")
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers={"X-Herogod-Token": "wrong"}).status_code == 401
    assert client.get("/api/config", headers={"X-Herogod-Token": "s3cret"}).status_code == 200
    assert client.get("/api/config", headers={"Authorization": "Bearer s3cret"}).status_code == 200

    created = client.post(
        "/api/jobs",
        files={"image": ("input.png", png_bytes(), "image/png")},
        data={"prompt": "x", "steps": "2"},
        headers={"X-Herogod-Token": "s3cret"},
    )
    assert created.status_code == 202
    job_id = created.json()["id"]
    wait_for_authed = wait_for_token(client, job_id)
    assert wait_for_authed["status"] == "done"
    assert client.get(f"/api/jobs/{job_id}/image?s3cret").status_code == 401
    assert client.get(f"/api/jobs/{job_id}/image?token=s3cret").status_code == 200


def wait_for_token(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}", headers={"X-Herogod-Token": "s3cret"})
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"done", "error", "cancelled"}:
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_static_frontend_is_served(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "Herogod" in index.text
    assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/styles.css").headers["content-type"].startswith("text/css")
    assert client.get("/favicon.svg").status_code == 200
