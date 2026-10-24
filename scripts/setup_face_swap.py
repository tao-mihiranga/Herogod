#!/usr/bin/env python3
"""Download the optional INSwapper weights with integrity verification.

InsightFace distributes models for non-commercial research. Review the current
license before downloading or using the weights: https://github.com/deepinsight/insightface
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx"
EXPECTED_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
CHUNK_SIZE = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "inswapper_128.onnx")
    parser.add_argument("--force", action="store_true", help="Download again even if the file exists")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file() and not args.force:
        print(f"Verifying existing model: {output}")
        if sha256(output) == EXPECTED_SHA256:
            print("Model is already installed and its checksum is valid.")
            return
        print("Existing checksum does not match; downloading a fresh copy.")

    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Herogod-Model-Setup/1.0"})
    print("Downloading INSwapper weights (about 529 MB)…")
    print("By continuing, you are responsible for reviewing and following the model license.")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            next_report = 10
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent >= next_report:
                        print(f"  {min(percent, 100)}%")
                        next_report += 10
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"Checksum mismatch: expected {EXPECTED_SHA256}, received {actual}")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Installed and verified: {output}")
    print("InsightFace will fetch the buffalo_l detector pack on first use if it is not already cached.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Download cancelled.")
