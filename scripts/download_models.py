#!/usr/bin/env python3
"""Download the Herogod model bundle into a local directory.

The studio can pull weights on demand, but pre-downloading is much friendlier on
Colab: you see the progress, you can reuse the disk between sessions on a
persistent runtime, and a failed download never looks like a broken app.

    python scripts/download_models.py
    python scripts/download_models.py --dest /content/models --skip-base
    python scripts/download_models.py --lora ScottzillaSystems/qwen-image-edit-plus-nsfw-lora

Prints the environment variables the server needs when it finishes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_LORA = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
LORA_WEIGHT = "qwen-image-edit-plus-nsfw-lora.safetensors"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen-Image-Edit-2511 and the MCNL LoRA")
    parser.add_argument("--dest", default=os.getenv("HEROGOD_MODELS_DIR", "models"), help="destination directory")
    parser.add_argument("--base", default=os.getenv("HEROGOD_BASE_MODEL", DEFAULT_BASE), help="base model repo id")
    parser.add_argument("--lora", default=os.getenv("HEROGOD_LORA_REPO", DEFAULT_LORA), help="LoRA repo id")
    parser.add_argument("--lora-weight", default=os.getenv("HEROGOD_LORA_WEIGHT", LORA_WEIGHT))
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""), help="HF token (defaults to $HF_TOKEN)")
    parser.add_argument("--skip-base", action="store_true", help="only fetch the LoRA")
    parser.add_argument("--skip-lora", action="store_true", help="only fetch the base model")
    parser.add_argument("--min-free-gb", type=float, default=45.0, help="abort below this much free disk")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

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
    token = args.token or None
    base_dir = None
    lora_dir = None

    if not args.skip_base:
        print(f"\n[1/2] {args.base} -> {dest / args.base.replace('/', '--')}")
        print("      This is the big download (~40 GB). It can take a while.")
        base_dir = snapshot_download(
            repo_id=args.base,
            local_dir=str(dest / args.base.replace("/", "--")),
            token=token,
            max_workers=8,
        )
        print(f"      done: {base_dir}")

    if not args.skip_lora:
        print(f"\n[2/2] {args.lora} -> {dest / args.lora.replace('/', '--')}")
        lora_dir = snapshot_download(
            repo_id=args.lora,
            local_dir=str(dest / args.lora.replace("/", "--")),
            token=token,
            allow_patterns=[args.lora_weight, "*.json", "*.md"],
            max_workers=4,
        )
        print(f"      done: {lora_dir}")

    elapsed = time.time() - started
    print(f"\nFinished in {elapsed / 60:.1f} min")
    print("\nExport these before starting the server:")
    if base_dir:
        print(f'  export HEROGOD_BASE_MODEL_PATH="{base_dir}"')
    if lora_dir:
        print(f'  export HEROGOD_LORA_REPO="{lora_dir}"')
        print(f'  export HEROGOD_LORA_WEIGHT="{args.lora_weight}"')
    print(f'  export HEROGOD_MODELS_DIR="{dest}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
