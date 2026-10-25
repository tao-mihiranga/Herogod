"""Tests for scripts/download_models.py — no network, no huggingface_hub needed."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_models.py"


def load_script():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture()
def script():
    return load_script()


class FakeGatedRepoError(Exception):
    pass


@pytest.fixture()
def stub_hub(monkeypatch):
    """Install a fake huggingface_hub module for check_access()."""
    hub = types.ModuleType("huggingface_hub")
    behaviour = {"mode": "ok"}

    class Info:
        gated = False
        private = False
        lastModified = "2026-01-01"

    class Api:
        def repo_info(self, repo_id, token=None, files_metadata=False):
            if behaviour["mode"] == "gated":
                raise FakeGatedRepoError("Access to model is restricted (403). You must accept the conditions.")
            if behaviour["mode"] == "missing":
                raise FakeGatedRepoError("404 Client Error: Repository Not Found for url")
            return Info()

    hub.HfApi = Api
    hub.snapshot_download = lambda **kwargs: str(kwargs["local_dir"])
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return behaviour


def test_explain_gated_mentions_accepting_conditions(script):
    message = script.explain(FakeGatedRepoError("GatedRepoError: 403 gated repo"), "org/model", None)
    assert "gated" in message.lower()
    assert "accept" in message.lower()
    assert "org/model" in message


def test_explain_invalid_token(script):
    message = script.explain(Exception("401 Client Error: invalid token"), "org/model", "hf_x")
    assert "rejected" in message.lower() or "401" in message


def test_explain_includes_no_token_hint(script):
    message = script.explain(Exception("boom"), "org/model", None)
    assert "HF_TOKEN" in message


def test_check_access_success(script, stub_hub):
    info = script.check_access("org/model", "hf_x")
    assert info["repo"] == "org/model"
    assert info["gated"] is False


def test_check_access_gated_raises(script, stub_hub):
    stub_hub["mode"] = "gated"
    with pytest.raises(script.DownloadError) as error:
        script.check_access("org/model", "hf_x")
    assert "accept" in str(error.value).lower()


def test_check_access_missing_raises(script, stub_hub):
    stub_hub["mode"] = "missing"
    with pytest.raises(script.DownloadError):
        script.check_access("org/model", None)


def test_fetch_wraps_errors(script):
    def boom(**_kwargs):
        raise RuntimeError("403 gated")

    with pytest.raises(script.DownloadError) as error:
        script.fetch("org/model", Path("/tmp/x"), "hf_x", snapshot_download=boom)
    assert "accept" in str(error.value).lower()


def test_fetch_passes_patterns_and_token(script, tmp_path):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    result = script.fetch(
        "org/lora",
        tmp_path,
        "hf_secret",
        allow_patterns=["w.safetensors"],
        max_workers=2,
        snapshot_download=fake,
    )
    assert result == str(tmp_path)
    assert seen["repo_id"] == "org/lora"
    assert seen["token"] == "hf_secret"
    assert seen["allow_patterns"] == ["w.safetensors"]
    assert seen["max_workers"] == 2


def test_check_access_mode_returns_zero(script, stub_hub, monkeypatch, capsys):
    monkeypatch.setattr(script, "check_access", lambda repo_id, token: {"repo": repo_id, "gated": True, "private": False})
    code = script.main(["--check-access", "--token", "hf_x"])
    out = capsys.readouterr().out
    assert code == 0
    assert "gated" in out
    assert "hf_x" not in out  # the token itself must never be printed
    assert "provided (hidden)" in out


def test_check_access_mode_failure(script, stub_hub, monkeypatch, capsys):
    def fail(repo_id, token):
        raise script.DownloadError("no access")

    monkeypatch.setattr(script, "check_access", fail)
    code = script.main(["--check-access"])
    assert code == 3
    assert "Access check failed" in capsys.readouterr().err


def test_main_aborts_without_disk_space(script, monkeypatch, capsys):
    monkeypatch.setattr(script, "free_space", lambda path: 1024)
    code = script.main(["--dest", "/tmp/does-not-matter", "--min-free-gb", "45"])
    assert code == 2
    assert "Not enough disk space" in capsys.readouterr().err


def test_main_downloads_and_prints_env(script, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(script, "free_space", lambda path: 500 * 1024**3)
    calls = []

    def fake_fetch(repo_id, target, token, **kwargs):
        calls.append((repo_id, target, token, kwargs.get("allow_patterns")))
        return str(target)

    monkeypatch.setattr(script, "fetch", fake_fetch)
    code = script.main(["--dest", str(tmp_path), "--token", "hf_secret"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(calls) == 2
    assert calls[0][0] == script.DEFAULT_BASE
    assert calls[1][0] == script.DEFAULT_LORA
    assert calls[1][3] == [script.LORA_WEIGHT, "*.json", "*.md"]
    assert "HEROGOD_BASE_MODEL_PATH" in out
    assert "HEROGOD_LORA_REPO" in out
    assert "hf_secret" not in out


def test_main_reports_download_failure(script, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(script, "free_space", lambda path: 500 * 1024**3)

    def blow_up(repo_id, target, token, **kwargs):
        raise script.DownloadError(f"Could not fetch {repo_id}: gated")

    monkeypatch.setattr(script, "fetch", blow_up)
    code = script.main(["--dest", str(tmp_path), "--skip-lora"])
    assert code == 4
    assert "Could not fetch" in capsys.readouterr().err


def test_main_skips_everything(script):
    assert script.main(["--skip-base", "--skip-lora"]) == 2
