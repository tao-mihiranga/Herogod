# Herogod — Qwen Image Edit Studio

A self-hosted image **editing** studio for people who would rather run the model than rent an API.
Drop in a photo, describe the change, and iterate — with Fooocus-style styles, reference images,
true-CFG control and a before/after canvas.

It is built around one pipeline:

| Piece | Value |
| --- | --- |
| Base model | [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (`QwenImageEditPlusPipeline`) |
| Adapter | [`ScottzillaSystems/qwen-image-edit-plus-nsfw-lora`](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora) — MCNL v1, 590 MB |
| Adapter name | `mcnl-nsfw-v1` |
| Licences | Apache-2.0 (base) · OpenRAIL++ (LoRA) |

No cloud API, no moderation layer, no prompt log. Everything runs on the machine (or Colab runtime)
that serves the page. The adapter's own model card lists the trigger words its training used; the
studio passes your instruction through unchanged, so include them yourself if you want those concepts.

> **Adults only.** The default adapter is trained for adult content. Do not use this project to make
> sexual content involving minors, non-consensual intimate imagery, or content that harasses,
> impersonates or defrauds anyone. Follow the law where you live and the licences of the weights.
> You are responsible for what you generate.

---

## Quick start on Google Colab

Two notebooks, same model (`Qwen-Image-Edit-2511` + the MCNL LoRA). Pick one:

| Notebook | Use it when |
| --- | --- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tao-mihiranga/Herogod/blob/main/Herogod_Colab.ipynb) [`Herogod_Colab.ipynb`](Herogod_Colab.ipynb) | Anonymous downloads are fine and the repo gates are already accepted. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tao-mihiranga/Herogod/blob/main/Herogod_Colab_HF_Token.ipynb) [`Herogod_Colab_HF_Token.ipynb`](Herogod_Colab_HF_Token.ipynb) | **Recommended.** Uses your own Hugging Face token: unlocks the gated LoRA, higher rate limits, `hf_transfer` speedups, and verifies access *before* the 40 GB download. |

### Common flow

1. Pick **Runtime → Change runtime type → GPU** (L4/A100 preferred; T4 works at a lower pixel budget).
2. Run the cells in order. The configuration form sets dtype, quantisation, offload, pixel budget and steps.
3. The download cell fetches the weights into `/content/models` (~40 GB base + 590 MB LoRA).
   Skip it and the model downloads lazily on the first edit instead.
4. The launcher starts the server, opens a Cloudflare quick tunnel and prints:
   ```
   Studio:         https://something.trycloudflare.com
   One-click link: https://something.trycloudflare.com/?token=…
   Access token:   …
   ```
5. Open the one-click link (or paste the token into **Engine → Connection**).
6. Add an image, type an instruction, press **Generate**.

Set `PUBLIC_TUNNEL = False` to use Colab's private proxy instead of a public URL.

The first generation after loading a 40 GB model takes several minutes on a T4. Keep the runtime alive;
closing it kills the tunnel and the model.

### About the Hugging Face token notebook

The token is read from Colab Secrets (`HF_TOKEN`), a hidden prompt, or the environment. It is:

* validated with `whoami` and a repo access check (`scripts/download_models.py --check-access`) **before**
  any large download, with actionable messages for 401 / 403 / gate-not-accepted / rate-limit cases;
* used for the base model *and* the LoRA (the LoRA repo carries a Not-For-All-Audiences gate);
* never printed, written to disk or exposed to the studio frontend — only a short fingerprint is shown so
  you can tell which token is loaded. The studio access token is a separate, run-specific value.

Optional: enable `ENABLE_HF_TRANSFER` in the first cell to install `hf_transfer` (faster parallel downloads),
and use the final smoke-test cell to prove the token, gates, pipeline and LoRA all work end to end.

> The token is only needed to *download* weights. Inference is fully local — nothing about an edit leaves
> your runtime.

---

## Run locally

```bash
git clone https://github.com/tao-mihiranga/Herogod.git
cd Herogod

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # server + UI only

# optional: try the UI with no GPU and no downloads
python server.py --mock
```

For real inference install the diffusion stack matching your CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu126 / rocm / cpu
pip install -r requirements-local.txt
pip install -r requirements-quant.txt      # optional, for HEROGOD_QUANT=bnb4

python scripts/download_models.py --dest models    # pre-download ~40 GB
python server.py                                   # http://localhost:8000
```

If you downloaded into `models/`, the server finds the snapshot automatically. Otherwise `HEROGOD_BASE_MODEL_PATH`
points it at a directory.

Typical standalone flags:

```bash
python server.py --host 0.0.0.0 --port 8000        # LAN access
python server.py --preload                          # load the model at boot, not on first edit
python server.py --mock                             # placeholder engine for UI work
HEROGOD_API_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(30))") python server.py
```

---

## The interface

```
┌──────────────────────────────────────────────────────────────┬────────────────────┐
│ canvas: Result · Compare · Input     seed / size / steps     │ instruction        │
│                                                              │ styles (Fooocus)   │
│   drop zone → input image → generated result                 │ reference images   │
│   drag the compare handle for before/after                   │ steps · CFG · LoRA │
│                                                              │ aspect · seed      │
│ Save PNG · Use as input · Copy prompt · Full size · Stop     │ Generate           │
├──────────────────────────────────────────────────────────────┴────────────────────┤
│ Gallery drawer (server memory)          Engine drawer (runtime, VRAM, adapter)    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

* **Compare** overlays the input under the result with a draggable seam (arrow keys work too).
* **Use as input** chains edits without re-uploading; **Chain edits automatically** in Engine settings
  does it on every result.
* **Styles** append a fixed prompt fragment (hover a chip to read it exactly). Up to four, combined.
* **Reference images** (max 3) go to the pipeline's multi-image input: subject, garment, style cue.
* **Aspect** presets or "match input" pick an output canvas inside the pixel budget.
* **Seed** `-1` is random; **Last** reuses the previous seed for a controlled variation.
* Everything you change is remembered in `localStorage`; **Clear local settings** wipes it.
* Keyboard: `Ctrl/⌘ + Enter` generates, `G` toggles the gallery, `Esc` closes drawers.

---

## Configuration

Copy `.env.example` and export what you need, or pass environment variables to the server/Colab cell.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HEROGOD_API_TOKEN` | *(empty)* | Shared secret checked on every API call. **Set it for any public URL.** Empty token in the query string works for `<img>` loads. |
| `HEROGOD_BASE_MODEL` | `Qwen/Qwen-Image-Edit-2511` | Base repo id, or a local path. |
| `HEROGOD_BASE_MODEL_PATH` | *(unset)* | Local snapshot directory (wins over the repo id). |
| `HEROGOD_LORA_REPO` | `ScottzillaSystems/qwen-image-edit-plus-nsfw-lora` | LoRA repo, or a local directory. |
| `HEROGOD_LORA_WEIGHT` | `qwen-image-edit-plus-nsfw-lora.safetensors` | Weight filename inside the repo. |
| `HEROGOD_LORA_ADAPTER` | `mcnl-nsfw-v1` | Adapter name reported to diffusers. |
| `HEROGOD_EXTRA_LORAS` | *(empty)* | `repo:weight:name,repo2:weight2:name2` — extra adapters in the dropdown. |
| `HEROGOD_DTYPE` | `auto` | `auto` = bf16 on Ampere+, fp16 on T4/Turing, fp32 on CPU. |
| `HEROGOD_QUANT` | `none` | `bnb4` enables bitsandbytes NF4 on transformer + text encoder. |
| `HEROGOD_OFFLOAD` | `none` | `model` (whole modules) or `sequential` (layer by layer, slowest, smallest). |
| `HEROGOD_VAE_TILING` | `1` | Tile the VAE decode to save VRAM at high resolutions. |
| `HEROGOD_MAX_PIXELS` | `1048576` | Output pixel budget (1024×1024). Lower it first when you hit OOM. |
| `HEROGOD_MAX_REFS` | `3` | Reference-image slots. |
| `HEROGOD_STEPS` / `HEROGOD_TRUE_CFG` / `HEROGOD_LORA_SCALE` | `20` / `4.0` / `1.0` | UI defaults. |
| `HEROGOD_MOCK` | `0` | Placeholder engine, no torch. |
| `HEROGOD_MODELS_DIR` | `./models` | Where snapshots live. |
| `HEROGOD_MAX_CONCURRENCY` | `1` | Parallel generations (keep at 1 on one GPU). |
| `HF_TOKEN` | *(empty)* | Needed only for gated/private repos; also speeds up downloads. |
| `HOST` / `PORT` / `LOG_LEVEL` | `0.0.0.0` / `8000` / `INFO` | Server binding. |

### Hardware reality check

| GPU | Practical setting |
| --- | --- |
| RTX 4090 / A100 / L40S (24–40 GB+) | bf16, 1024², 20 steps — the intended experience |
| L4 / A10 (24 GB) | bf16, 1024², 16–20 steps |
| T4 (16 GB, Colab free) | fp16, `MAX_PIXELS=786432`, 12–20 steps, ~2–4 min/image |
| 8–12 GB | `HEROGOD_QUANT=bnb4` + `HEROGOD_OFFLOAD=model`, 768² |
| CPU only | works, but minutes to tens of minutes per image |

Qwen-Image-Edit-2511 is a large MMDiT. There is no way to make it small; start by shrinking the pixel
budget, then quantise, then offload.

---

## CLI

```bash
python scripts/download_models.py --help
  --dest models              destination (default $HEROGOD_MODELS_DIR)
  --base  Qwen/…             base repo id
  --lora  Scottzilla…        LoRA repo id
  --token hf_…               or set HF_TOKEN (never printed)
  --check-access             verify token + repo access, download nothing
  --skip-base / --skip-lora  partial download
  --min-free-gb 45           refuse to start when the disk is too small

python scripts/download_models.py --check-access            # gated/private access check
HF_TOKEN=hf_… python scripts/download_models.py --dest models

python server.py  [--host H] [--port P] [--mock] [--no-token] [--preload]
```

## API

All endpoints under `/api` require `X-Herogod-Token` (or `Authorization: Bearer …`, or `?token=…`
on GETs) when `HEROGOD_API_TOKEN` is set. `GET /api/health` is always public.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness, version, whether a token is required, mock flag |
| `GET` | `/api/config` | Styles, performance presets, aspects, LoRA info, limits, defaults |
| `GET` | `/api/engine` | Runtime state: device, dtype, quant, offload, adapters, load time |
| `POST` | `/api/engine/load` | Load the pipeline now |
| `POST` | `/api/engine/unload` | Drop the pipeline and empty the CUDA cache |
| `POST` | `/api/jobs` | Multipart: `image`, `refs[]`, `prompt`, `negative`, `styles` (JSON), `steps`, `true_cfg`, `lora_scale`, `adapter`, `seed`, `aspect` → `202 {id}` |
| `GET` | `/api/jobs` | Job list (newest first) |
| `GET` | `/api/jobs/{id}` | Status, progress, step counter, seed, settings, error |
| `POST` | `/api/jobs/{id}/cancel` | Cooperative cancel between steps |
| `GET` | `/api/jobs/{id}/image` · `/thumb` | PNG output, full and 512 px thumbnail |
| `DELETE` | `/api/jobs` | Clear finished jobs from memory |

Images are returned as PNG bytes (not base64) so the browser can cache them normally.
Interactive docs: `/api/docs`.

```bash
curl -s -X POST http://localhost:8000/api/jobs \
  -H "X-Herogod-Token: $HEROGOD_API_TOKEN" \
  -F image=@photo.jpg \
  -F prompt="swap the background for a rainy neon street, keep the face" \
  -F styles='["cinematic"]' \
  -F steps=20 -F true_cfg=4 -F seed=-1
```

---

## Development

```bash
pip install -r requirements-dev.txt
pytest -q                        # API, prompt, engine, downloader and notebook tests; mock engine, no GPU
node tests/frontend_smoke.cjs    # CSP, asset, id and CSS sanity checks

# optional: real DOM tests in jsdom (starts its own mock server)
cd tests/ui && npm install && npm test

python server.py --mock          # iterate on the UI with instant placeholder renders
```

The frontend is plain HTML/CSS/JS in `web/` — no bundler, no npm packages, no CDN. The server
streams those files as static routes and sets a strict CSP (`script-src 'self'`, no inline handlers).

```
Herogod/
├── server.py              FastAPI app, auth, upload validation, static routes
├── app/
│   ├── config.py          settings, style/preset catalogs
│   ├── prompts.py         style composition, size math, seed parsing
│   ├── engine.py          Qwen pipeline + LoRA adapters + mock engine
│   └── jobs.py            in-memory job store with progress + cancel
├── web/                   index.html · styles.css · app.js
├── scripts/download_models.py   token-aware downloader + access checker
├── tests/                 pytest suites + node/jsdom frontend tests
├── Herogod_Colab.ipynb          anonymous Colab runtime
├── Herogod_Colab_HF_Token.ipynb Colab runtime with your HF token (gates, rate limits)
└── requirements*.txt
```

### How a request flows

1. `POST /api/jobs` validates uploads (type, size, pixel count, EXIF orientation), resolves styles,
   clamps steps/CFG/LoRA scale and computes the output canvas.
2. `jobs.start_job` runs generation on a worker thread so the event loop stays responsive.
3. `QwenEngine` lazily loads `QwenImageEditPlusPipeline`, then loads/switches the LoRA adapter
   (`load_lora_weights` once, `set_adapters` per request) and calls it with `true_cfg_scale`,
   a seeded generator and a step callback feeding job progress.
4. The browser polls `GET /api/jobs/{id}` and swaps in `/api/jobs/{id}/image` when it is done.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Cannot reach the backend` | The page can't reach the API: server down, wrong **Backend URL**, or a tunnel expired. |
| `401` / token rejected | Token mismatch: copy it from the Colab cell into **Engine → Connection**. |
| `Model failed to load` | Disk or network: base model needs ~40 GB free. Check the log (`/tmp/herogod-server.log` on Colab). |
| `CUDA out of memory` | Lower `HEROGOD_MAX_PIXELS`, set `HEROGOD_OFFLOAD=model`, or `HEROGOD_QUANT=bnb4`; restart the runtime to clear fragmentation. |
| Black/NaN output on a T4 | Force `HEROGOD_DTYPE=fp16` (T4 has no bf16). |
| Slow first run | Expected: weights download once, then load into VRAM. Use `--preload` to pay it up front. |
| `diffusers` has no `QwenImageEditPlusPipeline` | Upgrade: `pip install -U "diffusers>=0.36"`. |
| Colab tunnel URL changes | Quick tunnels are per-run; re-run the tunnel cell and update the token in the studio. |
| `Access to model … is restricted` | Gated repo: accept its conditions on Hugging Face with the token's account, then use the [token notebook](Herogod_Colab_HF_Token.ipynb) (`--check-access` tells you which repo failed). |
| HF downloads rate-limited | Authenticate with `HF_TOKEN` and set `HF_HUB_ENABLE_HF_TRANSFER=1` (needs `pip install hf_transfer`). |

## Privacy, safety, legal

* No analytics, no telemetry, no prompt log, no server-side image history on disk. Jobs live in RAM and
  disappear when the process stops. Only **Save PNG** writes a file, and only to your downloads folder.
* A Cloudflare quick tunnel publishes your runtime on a public URL. Keep `HEROGOD_API_TOKEN` set; the
  tunnel is the only thing between the internet and your GPU.
* Herogod does not add a moderation layer, and equally does not help you evade one on other platforms.
* Adults only. No sexual content involving minors, no non-consensual intimate imagery, no impersonation,
  no fraud or harassment. Respect the OpenRAIL++ use restrictions of the LoRA and the law where you live.
* Face and identity edits of real people require their consent.

## Credits

* [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) — Alibaba Qwen team (Apache-2.0).
* [MCNL v1 NSFW LoRA](https://huggingface.co/ScottzillaSystems/qwen-image-edit-plus-nsfw-lora) —
  ScottzillaSystems (OpenRAIL++), a mirror of `aiunivers/qwen-image-edit-plus-nsfw-lora`.
* [Diffusers](https://github.com/huggingface/diffusers) for the pipeline, [FastAPI](https://fastapi.tiangolo.com/)
  for the server.
* Style presets are inspired by Fooocus' style chips; the fragments here are original text.
