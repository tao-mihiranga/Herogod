# Herogod

**A chat-based, self-hostable AI image studio.** Upload an image, ask for a change, and iterate on the result. Run open models on your own GPU or Google Colab, or connect your own free-tier Hugging Face / Gemini credentials.

Herogod is dependency-free in the browser: **double-click `index.html` and the interface works**. A backend is required for actual AI inference; the app never pretends a disconnected edit succeeded.

## What is included

- Modern ChatGPT/Gemini-style chat UI with drag-and-drop uploads
- Iterative editing: **Use as input** applies another prompt to the latest result
- Local browser chat/image history using IndexedDB (up to 20 creations)
- **InstructPix2Pix** for lightweight instruction-based local edits
- **SDXL img2img** for local restyling with adjustable strength
- **FLUX 3 Image Klein** with modular Diffusers, full-caption editing, and optional official prompt upsampling
- **FIBO 2.0 Edit Turbo** with its required 4-step/no-CFG defaults, locally or through supported HF Inference Providers
- **Qwen Image Edit 2511** with its dedicated Plus pipeline and true-CFG controls
- A model registry with per-model defaults, gated-model token handling, and hardware/license guidance
- Optional **InsightFace / INSwapper** face swapping with an explicit consent step
- Hugging Face image-to-image inference using your own token and available credits
- Gemini image editing using your own Google AI Studio key
- Colab notebook with a temporary HTTPS Cloudflare tunnel and generated access token
- PNG downloads, full-screen previews, seeds, steps, guidance, and image-adherence controls
- No framework, npm install, CDN, tracking, prompt log, or server-side image history

> **What “uncensored” means here:** the self-hosted pipeline does not add a cloud moderation layer or secretly rewrite prompts. FLUX 3’s official prompt upsampler is optional and clearly indicated. Herogod does **not** jailbreak or bypass third-party API safeguards. Hosted provider rules, model licenses, and the law still apply. Do not use it for non-consensual intimate imagery, sexual content involving minors, harassment, fraud, or deception.

## Fastest start: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tao-mihiranga/Herogod/blob/main/Herogod_Colab.ipynb)

1. Open the notebook and select **Runtime → Change runtime type → T4 GPU** for InstructPix2Pix or SDXL. Free GPU availability varies. FLUX 3, FIBO 2, and Qwen are high-memory checkpoints and are not all practical on a free T4.
2. Run the cells in order. Keep the default `timbrooks/instruct-pix2pix` model for the easiest T4 setup, or choose another registered model in **Runtime choices**.
3. The launcher prints a `https://…trycloudflare.com` studio/backend URL and a random Herogod access token.
4. Open the studio URL. In **Connection settings**, paste the access token. If you opened `index.html` separately, paste both the URL and token.
5. Add an image, describe the edit, and send.

The first request can take several minutes while model weights download. Keep Colab running. Closing/resetting the runtime stops the backend and invalidates the temporary tunnel.

### Optional Colab API keys

For a shared runtime, add `HF_TOKEN` and/or `GEMINI_API_KEY` in Colab Secrets and allow notebook access **before** running the configuration cell. Values are loaded into the backend environment and never printed. Alternatively, paste your HF token into **Connection settings → Your HF token**; it is stored only in that browser tab and forwarded to your configured backend for hosted inference or gated downloads.

Ungated InstructPix2Pix, SDXL, and Qwen checkpoints need no AI key. FLUX 3 and FIBO 2 are gated: accept their conditions on Hugging Face first, then provide a token with read access. Hosted HF Inference additionally needs Inference Providers permission.

### Optional face swap in Colab

Set `ENABLE_FACE_SWAP = True` in the configuration cell before the installation cell. The notebook installs InsightFace and verifies the INSwapper model checksum. Review its license before use; InsightFace weights are generally distributed for non-commercial research.

## Run locally

### 1. Open just the interface

Open `index.html` in any modern browser. No server or build step is needed for the UI. It defaults to `http://127.0.0.1:8000` for AI requests.

### 2. Start the API-only backend

This supports hosted providers; local Diffusers requires the next step too.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HEROGOD_API_TOKEN="a-long-random-string"
python server.py
```

Open **http://127.0.0.1:8000**. Enter your Herogod access token in Connection settings. The server binds to `0.0.0.0` by default so it also works in containers and tunnels; use `--host 127.0.0.1` if you want loopback only.

### 3. Enable local open models

Use **Python 3.11+** and a PyTorch environment appropriate for your GPU. Colab already includes PyTorch. The local requirements use Diffusers 0.42+ for FLUX 3/FIBO 2 and Transformers 5.8+ for their Qwen3-VL/Qwen3.5 components, including the optional FLUX prompt upsampler.

```bash
pip install -r requirements-local.txt
python server.py
```

For CUDA, install the matching PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) if your environment does not already have it. CPU inference is supported for InstructPix2Pix/SDXL but slow; FLUX 3, FIBO 2, and Qwen require CUDA in this integration.

### 4. Enable optional face swap

```bash
pip install -r requirements-face.txt
python scripts/setup_face_swap.py
python server.py
```

The setup script downloads roughly 529 MB and verifies SHA-256 integrity. The `buffalo_l` detector pack is fetched by InsightFace on first use if not cached. Model files are ignored by Git.

## Free options at a glance

| Option | AI key needed? | Cost model | Content controls | Best for |
|---|---:|---|---|---|
| Local / Colab InstructPix2Pix | No | Your hardware or free Colab quota | No extra Herogod cloud filter | Direct instruction edits |
| Local / Colab SDXL img2img | No | Your hardware or free Colab quota | No extra Herogod cloud filter | Creative restyling |
| Local FLUX 3 Image Klein | Yes, gated | Your hardware; 24GB+ BF16 GPU | FLUX Kommunity License | Full-caption edits, 16 steps |
| Local FIBO 2 Edit Turbo | Yes, gated | Your high-memory hardware | CC BY-NC 4.0 | Non-commercial 4-step edits |
| Local Qwen Image Edit 2511 | No | Your very-high-memory hardware | Apache 2.0 | High-fidelity edits, true CFG |
| Local InsightFace face swap | No | Your hardware or free Colab quota | Consent required by app | Permitted face replacement |
| HF Inference: InstructPix2Pix / FIBO 2 | Yes | Provider's current credits; limits vary | Provider/model rules apply | Supported hosted edits |
| Gemini Flash Image | Yes | Current eligible free tier; limits vary | Google safety systems apply | Conversational hosted editing |

**No option is unlimited.** Colab runtimes, hosted free quotas, eligible models, and provider availability can change. Herogod does not include shared keys, sell credits, or guarantee free access. See [GUIDE.md](GUIDE.md) for setup links, prompt tips, licensing notes, and troubleshooting.

## Configuration

Set these environment variables on the **backend**. As an alternative for a trusted backend, the UI can keep your HF token in tab-only `sessionStorage` and send it through the `X-HF-Token` request header. Never embed credentials in source files.

| Variable | Default | Purpose |
|---|---|---|
| `HEROGOD_API_TOKEN` | empty | Protects all `/api` routes. Strongly recommended for any network/public server |
| `HF_TOKEN` | empty | Enables hosted HF inference and gated downloads; UI token is an alternative |
| `GEMINI_API_KEY` | empty | Enables Gemini inference |
| `GEMINI_MODEL` | `gemini-2.5-flash-image` | Override the Gemini model when needed |
| `HEROGOD_MODEL` | `timbrooks/instruct-pix2pix` | Default local model for new browser sessions (an explicit composer choice takes priority) |
| `HEROGOD_MAX_IMAGE_SIZE` | model-dependent | Maximum local input dimension; clamped to 256–1536; unset/empty uses model default |
| `HEROGOD_CPU_OFFLOAD` | model-dependent | Reduce CUDA memory for supported standard pipelines at the cost of speed/RAM |
| `HEROGOD_FLUX_UPSAMPLING` | `0` | Explicitly enable the official FLUX 3 prompt upsampler; needs substantial extra memory |
| `HEROGOD_FLUX_PUP_MODEL` | `black-forest-labs/flux-3-image-klein-pup` | Override the compatible prompt-upsampler repository |
| `HEROGOD_MAX_CONCURRENCY` | `1` | Simultaneous inference jobs; keep low on free GPUs |
| `HEROGOD_HF_PROVIDER` | `auto` | Hugging Face inference provider routing |
| `HEROGOD_FACE_MODEL` | `models/inswapper_128.onnx` | Optional INSwapper weight path |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | Web server bind address |

Copy [.env.example](.env.example) as a reference. The app does not automatically load `.env`; export variables in your shell or use your process manager. Never commit real credentials.

## API

Interactive OpenAPI documentation is available at `/api/docs` when the server is running.

- `GET /api/health` — provider and feature readiness; never returns credentials
- `POST /api/edit` — multipart image + prompt + provider/model/settings; returns a PNG data URL
- `POST /api/face-swap` — multipart target + reference + `consent=true`; returns a PNG data URL

If `HEROGOD_API_TOKEN` is configured, send either:

```text
Authorization: Bearer YOUR_HEROGOD_TOKEN
```

or `X-Herogod-Token: YOUR_HEROGOD_TOKEN`. This token is different from an AI provider key.

For hosted HF inference or gated local downloads, the UI can also send `X-HF-Token: YOUR_HF_TOKEN`. It overrides `HF_TOKEN` only for that request, is never returned by health responses, and is not written into server environment variables. Use HTTPS and only send it to a backend you trust. CORS permits the standalone HTML file to connect to your chosen backend.

## Project structure

```text
index.html                 Standalone interface and in-app guide
styles.css                Responsive dark studio design
app.js                    Chat, uploads, IndexedDB, providers, API client
server.py                 FastAPI server, validation, authentication, static hosting
backend/providers.py      Lazy Diffusers / Hugging Face / Gemini adapters
backend/face_swap.py      Optional local InsightFace engine
scripts/setup_face_swap.py Verified face-weight downloader
Herogod_Colab.ipynb       Colab installer, server, and Cloudflare tunnel
GUIDE.md                  Detailed free setup and usage guide
tests/                    API and validation tests (no GPU required)
```

## Privacy and security notes

- Images are decoded, processed in memory, re-encoded, and returned. Herogod does not save request images or prompts to a database. Large multipart uploads can use short-lived Starlette/OS temporary files, which are closed promptly after reading.
- The browser stores conversations and image data locally in IndexedDB. Delete a creation to remove it from that browser; download anything you want to keep.
- Gemini keys remain in backend environment variables/Colab Secrets. The Herogod tunnel token and optional UI-supplied HF token are stored in `sessionStorage`, not persistent local storage. The HF token is forwarded only to your configured backend; use Colab Secrets instead on shared runtimes.
- Uploaded files are limited to 15 MB each and 40 MP; only valid PNG, JPEG, and WebP images are accepted. EXIF orientation is applied and metadata is not carried into output.
- A public tunnel is still public. Always use a long random `HEROGOD_API_TOKEN`, keep provider keys in Colab Secrets, and stop the tunnel when finished.
- Cloudflare and any selected hosted AI provider have their own logging/data policies. “Self-hosted” refers to the Herogod pipeline, not the tunnel infrastructure.

## Development checks

The API/provider tests mock inference, so they need neither a GPU nor provider keys:

```bash
pip install -r requirements-dev.txt
pytest -q
node --check app.js
```

The optional DOM smoke test uses jsdom for development only—the frontend itself still needs no npm/build step:

```bash
npm install --prefix /tmp/herogod-jsdom jsdom@26
NODE_PATH=/tmp/herogod-jsdom/node_modules node tests/frontend_smoke.cjs
```

## Responsible creation

Only upload media you may modify. Get clear permission before swapping a recognizable person's face, label synthetic edits when context could mislead, and never create exploitative or non-consensual imagery. Model pricing does not determine model licensing—review every model card before commercial use.
