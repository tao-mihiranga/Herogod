#!/usr/bin/env python3
"""Download (or just verify access to) the Herogod model bundle.

The studio can pull weights on demand, but pre-downloading is much friendlier on
Colab: you see the progress, you can reuse the disk between sessions on a
persistent runtime, and a failed download never looks like a broken app.

    python scripts/download_models.py
    python scripts/download_models.py --dest /content/models --skip-base
    python scripts/download_models.py --check-access          # token/repo check only
    python scripts/download_models.py --lora ScottzillaSystems/qwen-image-edit-plus-nsfw-lora

A Hugging Face token is optional for public repos but required for gated ones
(the MCNL LoRA carries a Not-For-All-Audiences gate) and recommended for higher
download rate limits. Pass it with --token or HF_TOKEN; it is never printed.

Prints the environment variables the server needs when it finishes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_BASE = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_LORA = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
LORA_WEIGHT = "qwen-image-edit-plus-nsfw-lora.safetensors"


class DownloadError(RuntimeError):
    """A failure with an actionable, human-readable explanation."""


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def free_space(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = os.statvfs(probe)
    return usage.f_bavail * usage.f_frsize


def explain(error: Exception, repo_id: str, token: str | None) -> str:
    """Turn a huggingface_hub exception into something a user can act on."""
    text = f"{error.__class__.__name__}: {error}"
    lowered = text.lower()
    hints: list[str] = []

    if "gated" in lowered or "awaiting a review" in lowered or "access to model" in lowered:
        hints.append(
            f"Access to {repo_id} is gated. Open https://huggingface.co/{repo_id} while logged in, "
            "accept its conditions (the LoRA also has an age/NSFW gate), then retry."
        )
    if "401" in lowered or "invalid username or password" in lowered or "token" in lowered and "invalid" in lowered:
        hints.append(
            "The token was rejected. Create a fine-grained token with 'Read' access to public gated repos at "
            "https://huggingface.co/settings/tokens and pass it again."
        )
    if "403" in lowered or "forbidden" in lowered:
        hints.append(
            f"403 from {repo_id}: the token is valid but not allowed here. Accept the repo conditions with this "
            "same account, or check the token's repository permissions."
        )
    if "404" in lowered or "repository not found" in lowered:
        hints.append(f"{repo_id} was not found or is private and invisible to this token.")
    if "429" in lowered or "rate limit" in lowered:
        hints.append(
            "Rate limited by Hugging Face. Authenticating (or setting HF_HUB_ENABLE_HF_TRANSFER=1) raises the limits."
        )
    if "no space" in lowered or "disk" in lowered:
        hints.append("Out of disk space. Free space under the destination, or run on a bigger runtime.")
    if "connection" in lowered or "timeout" in lowered or "resolve" in lowered:
        hints.append("Network problem while talking to huggingface.co. Retry; partial downloads resume.")
    if not token:
        hints.append("No token was supplied. Set HF_TOKEN or pass --token for gated repos and higher limits.")

    hints.append("Full error: " + text[:600])
    return "\n  - ".join([f"Could not fetch {repo_id}:"] + hints)


def check_access(repo_id: str, token: str | None) -> dict[str, Any]:
    """Return metadata about a repo, raising DownloadError when inaccessible."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise DownloadError("huggingface_hub is missing. Run: pip install -r requirements.txt") from error

    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, token=token or None, files_metadata=False)
    except Exception as error:
        raise DownloadError(explain(error, repo_id, token)) from error
    return {
        "repo": repo_id,
        # repo_info exposes `.gated` on every modern huggingface_hub; fall back to None
        "gated": getattr(info, "gated", None),
        "private": bool(getattr(info, "private", False)),
        "updated": str(getattr(info, "lastModified", "") or ""),
    }


def fetch(
    repo_id: str,
    target: Path,
    token: str | None,
    *,
    allow_patterns: Iterable[str] | None = None,
    max_workers: int = 8,
    snapshot_download: Callable[..., str] | None = None,
) -> str:
    """Download one repo snapshot, translating failures into DownloadError."""
    downloader = snapshot_download
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise DownloadError("huggingface_hub is missing. Run: pip install -r requirements.txt") from error
        downloader = hf_snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(target),
        "token": token or None,
        "max_workers": max_workers,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    try:
        return downloader(**kwargs)
    except Exception as error:
        raise DownloadError(explain(error, repo_id, token)) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Qwen-Image-Edit-2511 and the MCNL LoRA")
    parser.add_argument("--dest", default=os.getenv("HEROGOD_MODELS_DIR", "models"), help="destination directory")
    parser.add_argument("--base", default=os.getenv("HEROGOD_BASE_MODEL", DEFAULT_BASE), help="base model repo id")
    parser.add_argument("--lora", default=os.getenv("HEROGOD_LORA_REPO", DEFAULT_LORA), help="LoRA repo id")
    parser.add_argument("--lora-weight", default=os.getenv("HEROGOD_LORA_WEIGHT", LORA_WEIGHT))
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "")
    parser.add_argument("--skip-base", action="store_true", help="only fetch the LoRA")
    parser.add_argument("--skip-lora", action="store_true", help="only fetch the base model")
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="verify the token and repo access without downloading anything",
    )
    parser.add_argument("--min-free-gb", type=float, default=45.0, help="abort below this much free disk")
    args = parser.parse_args(argv)

    token = (args.token or "").strip() or None
    repos = []
    if not args.skip_base:
        repos.append(("base model", args.base, None, 8))
    if not args.skip_lora:
        repos.append(("LoRA", args.lora, [args.lora_weight, "*.json", "*.md"], 4))

    if not repos:
        print("Nothing to do: both --skip-base and --skip-lora were given.", file=sys.stderr)
        return 2

    print(f"Token       : {'provided (hidden)' if token else 'none (public repos only)'}")

    # -- access check ---------------------------------------------------
    if args.check_access:
        failures = 0
        for label, repo_id, _patterns, _workers in repos:
            try:
                info = check_access(repo_id, token)
            except DownloadError as error:
                failures += 1
                print(f"\n[FAIL] {label}: {error}", file=sys.stderr)
                continue
            gate = "gated" if info["gated"] else ("private" if info["private"] else "public")
            print(f"[ok]   {label}: {repo_id} ({gate})")
        if failures:
            print(
                "\nAccess check failed. Accept the model conditions on the Hugging Face page with the same "
                "account that owns the token, then retry.",
                file=sys.stderr,
            )
            return 3
        print("\nAll repos are accessible with this token.")
        return 0

    # -- download -------------------------------------------------------
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    available = free_space(dest)
    print(f"Destination : {dest}")
    print(f"Free disk   : {human(available)}")
    if available < args.min_free_gb * 1024**3:
        print(
            f"Not enough disk space: {human(available)} free, {args.min_free_gb:.0f} GB recommended.\n"
            "Qwen-Image-Edit-2511 is roughly 40 GB in bf16. Use a bigger runtime or --skip-base.",
            file=sys.stderr,
        )
        return 2

    started = time.time()
    results: dict[str, str] = {}
    total = len(repos)

    for index, (label, repo_id, patterns, workers) in enumerate(repos, start=1):
        target = dest / repo_id.replace("/", "--")
        print(f"\n[{index}/{total}] {label}: {repo_id} -> {target}")
        if label == "base model":
            print("      This is the big download (~40 GB). It can take a while.")
        try:
            results[label] = fetch(repo_id, target, token, allow_patterns=patterns, max_workers=workers)
        except DownloadError as error:
            print(f"\n{error}", file=sys.stderr)
            return 4
        print(f"      done: {results[label]}")

    elapsed = time.time() - started
    print(f"\nFinished in {elapsed / 60:.1f} min")
    print("\nExport these before starting the server:")
    if "base model" in results:
        print(f'  export HEROGOD_BASE_MODEL_PATH="{results["base model"]}"')
    if "LoRA" in results:
        print(f'  export HEROGOD_LORA_REPO="{results["LoRA"]}"')
        print(f'  export HEROGOD_LORA_WEIGHT="{args.lora_weight}"')
    print(f'  export HEROGOD_MODELS_DIR="{dest}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
