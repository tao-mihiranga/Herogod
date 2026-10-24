# Herogod free setup & usage guide

This guide separates **free open weights**, **free compute**, and **free-tier API access**. They are not the same thing:

- An open model can have no per-request fee but still needs a GPU/CPU and has a license.
- Colab can provide no-cost compute when available, but sessions and accelerators are limited.
- A hosted API can provide free credits or a free tier, but requires an account, has quotas, and keeps its own usage/safety policies.

Herogod does not include shared API keys or promise unlimited generation. Check the linked official pages before relying on a current quota.

## 1. Choose your route

### A. Completely self-hosted inference — recommended

**Models included:** InstructPix2Pix, SDXL img2img, FLUX 3 Image Klein, FIBO 2.0 Edit Turbo, Qwen Image Edit 2511, and optional INSwapper.

- **AI token:** not required for ungated repositories. FLUX 3 and FIBO 2 require your HF token and prior acceptance of their model-page conditions.
- **API cost:** none for local inference. Compute, storage, and bandwidth are still your responsibility.
- **Compute:** your machine or Colab. A free T4 is a good default for InstructPix2Pix, not a guarantee for every registered checkpoint.
- **Moderation:** Herogod adds no cloud moderation layer or hidden prompt rewrite. The FLUX prompt upsampler is optional and explicit. Hosted-provider safety controls are not bypassed.
- **Best default:** `timbrooks/instruct-pix2pix` on a T4 GPU.
- **License:** review each current model card. “Open weights” does not mean unrestricted commercial use; FIBO 2 is non-commercial and FLUX 3 has its own Kommunity License.

| Model and official card | Herogod defaults | Token | Runtime guidance |
|---|---|---|---|
| [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) | 20 steps, guidance 7.5, image adherence 1.5 | No | Lightweight default; T4 recommended |
| [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 25 steps, guidance 7.5, strength 0.65 | No | T4 with conservative settings/offload |
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | 16 steps, no CFG | **Yes, gated** | 24GB+ GPU for BF16; official upsampler needs substantial extra memory |
| [FIBO 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) | 4 steps, guidance 1.0 | **Yes, gated** | High-memory CUDA GPU; standard-pipeline CPU offload supported; CC BY-NC 4.0 |
| [Qwen Image Edit 2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | 40 steps, true CFG 4.0 | No | Very high-memory CUDA GPU; offload requires sufficient system RAM; Apache 2.0 |

Additional references:

- [FLUX 3 prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) — optional in Herogod; required only when you do not provide a full resulting-scene caption.
- [Diffusers documentation](https://huggingface.co/docs/diffusers/index) — version 0.42+ includes the FLUX 3 and FIBO 2 pipeline classes used here.
- [InsightFace repository and licensing](https://github.com/deepinsight/insightface) for optional face swap.

Herogod currently enables hosted HF Inference for InstructPix2Pix and FIBO 2 Turbo when a provider is available. At implementation time, the FIBO 2 card listed WaveSpeed; the FLUX 3 card listed no Inference Provider deployment. The Qwen card listed a provider, but Herogod keeps Qwen local-only because its dedicated Plus/true-CFG parameter mapping has not been validated through that hosted route. Check each current card before relying on availability or free credits.

### B. Hugging Face Inference Providers

- **AI token:** a Hugging Face user access token with the appropriate inference permission.
- **Cost:** uses your account's currently available inference credits. Free credit amounts and provider support can change.
- **Editing support:** the selected model must actually be available for image-to-image inference. A model existing on the Hub does not guarantee serverless deployment.
- **Controls:** provider and model rules remain active.

Official resources:

- [Inference Providers documentation](https://huggingface.co/docs/inference-providers/index)
- [Image-to-image task documentation](https://huggingface.co/docs/inference-providers/en/tasks/image-to-image)
- [Access token settings](https://huggingface.co/settings/tokens)
- [Inference pricing and credits](https://huggingface.co/pricing)

Setup:

1. Create/sign in to a Hugging Face account.
2. Create your own token. Read access is enough for gated local downloads; hosted inference also needs Inference Providers permission. Use the least permission needed.
3. For FLUX 3 or FIBO 2, open the exact model page while logged in and accept its conditions before trying to download.
4. Either set `HF_TOKEN` in the backend environment/Colab Secrets, or paste it into **Connection settings → Your HF token**. The UI option is kept in tab-only `sessionStorage` and sent only to your configured backend through `X-HF-Token`. Use HTTPS and a backend you trust.
5. Never embed a token in HTML/JavaScript source, Git, screenshots, or a chat prompt. Prefer Colab Secrets on shared runtimes.
6. Select an **HF Inference** option for hosted credits, or a **Local / Colab** option for weight downloads. If hosted inference is unavailable, Herogod reports it rather than silently substituting another paid service.

### C. Google AI Studio / Gemini

- **AI token:** your own Google AI Studio API key.
- **Cost:** use only a model/tier currently marked as free for your account and region. Rate limits and model eligibility can change.
- **Editing support:** configured for `gemini-2.5-flash-image` by default; override `GEMINI_MODEL` if Google changes the available model name.
- **Controls:** Google's safety systems and API policies remain active. Herogod does not alter safety settings.

Official resources:

- [Google AI Studio](https://aistudio.google.com/)
- [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini API pricing and free-tier tables](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

Setup:

1. Create an API key in Google AI Studio in an eligible region.
2. Check that the selected image model currently has a free tier and review its limits.
3. Set `GEMINI_API_KEY` in the backend environment or Colab Secrets.
4. Optionally set `GEMINI_MODEL` to a newer supported image model.
5. Start Herogod and select **Gemini Flash Image**. If a request is declined or returns only text, Herogod shows an error; it does not attempt to evade the provider.

## 2. Colab: detailed walkthrough

1. Open [`Herogod_Colab.ipynb`](Herogod_Colab.ipynb) in Colab.
2. Choose **Runtime → Change runtime type → T4 GPU** for lightweight models. If no free GPU is available, wait or run locally. FLUX 3, FIBO 2, and Qwen need substantially more memory and are not all practical on a free T4.
3. In **Runtime choices**, start with:
   - `MODEL_ID = timbrooks/instruct-pix2pix`
   - `MAX_IMAGE_SIZE = 0` to use each model's recommended size automatically.
   - `ENABLE_FLUX_UPSAMPLING = False` unless you selected FLUX 3 on a runtime with enough extra GPU/RAM for the official upsampler.
   - `ENABLE_FACE_SWAP = False` unless you need it and have reviewed its license.
4. For hosted APIs or gated models, add `HF_TOKEN` or `GEMINI_API_KEY` under Colab's Secrets icon and enable notebook access. Never paste keys into a code cell. You can instead paste only the HF token into the UI's Connection settings for the current browser tab.
5. Run cells from top to bottom. Installation and the first model download can take time.
6. Copy the printed studio URL and Herogod access token. The token protects the tunnel; it is not an AI key.
7. Open the studio URL, go to **Connection settings**, and paste the Herogod access token. Add your HF token too if you are using gated models or HF Inference and did not configure it through Colab Secrets. Click **Test connection**, then **Save & connect**.
8. Alternatively, double-click the repository's `index.html`; paste both the tunnel backend URL and token there.
9. Upload a PNG/JPG/WebP (up to 15 MB), describe a specific edit, and send.

The temporary Cloudflare URL changes when you restart the tunnel. Keep the notebook runtime alive while editing. Do not share the access token publicly.

## 3. Local setup choices

### Interface only

Open `index.html`. HTML, CSS, JavaScript, icons, and the guide are all local files. No internet is needed to browse the interface, but inference needs a reachable backend.

### Local API + hosted providers

```bash
pip install -r requirements.txt
export HEROGOD_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(30))')"
export HF_TOKEN="your-own-token"                 # optional
export GEMINI_API_KEY="your-own-google-key"     # optional
python server.py --host 127.0.0.1
```

### Local open models

Use Python 3.11+ and install a PyTorch build matching your CUDA/CPU environment, then:

```bash
pip install -r requirements.txt -r requirements-local.txt
python server.py --host 127.0.0.1
```

The first run downloads model weights from the model hub. Later runs can use the local cache. GPU memory needs depend on image size, model, and settings.

### Hosting the frontend online

The frontend is static. Host `index.html`, `styles.css`, `app.js`, and `favicon.svg` on any static host, then point Settings at an HTTPS Herogod backend. Alternatively, `python server.py` serves both frontend and API from one origin.

For an HTTPS-hosted page, use an HTTPS backend tunnel to avoid mixed-content restrictions. Never embed provider keys in static source or public hosting environment variables intended for browser code. The optional HF token field keeps the value only in the current tab and forwards it to your backend; Gemini remains backend-only.

## 4. Prompting guide

Instruction editors perform better with one clear objective than a long paragraph of unrelated requests.

### Direct scene changes

```text
Change the background to a rainy Tokyo alley at night, with soft neon reflections. Keep the person, pose, clothing, and facial features unchanged.
```

### Restyling

```text
Turn this photograph into hand-painted gouache concept art. Preserve the composition and main shapes; use muted teal and warm ochre colors.
```

### FLUX 3 full resulting-scene captions

With prompt upsampling off, do not send a short instruction such as “make it blue.” Describe the complete image you want returned, including the preserved subject and setting:

```text
A detailed photorealistic portrait of the same person in the same coastal viewpoint and framing, now at blue hour. Preserve their identity, pose, clothing, and natural skin texture. Replace the sunny sky with a deep indigo twilight sky, add soft cool moonlight from the left, retain gentle ocean detail in the background, and use a calm cinematic color grade with realistic shadows.
```

You can enable `HEROGOD_FLUX_UPSAMPLING=1` to run the official vision-language upsampler and use shorter instructions instead. It is a separate large model and needs substantial additional GPU/RAM. Herogod shows when this explicit option is active.

FIBO 2 Turbo accepts a direct instruction such as “replace the jacket with a red leather one.” Advanced users can instead provide compatible VGL JSON with an `edit_instruction`; Herogod does not require a separate Gemini/VGL key for direct edits.

### Product cleanup

```text
Place this product on a clean warm-gray studio surface. Add a soft shadow and even lighting. Keep its label, proportions, and colors accurate.
```

### Iterative editing

1. Ask for the main change.
2. Inspect the result.
3. Click **Use as input**.
4. Ask for a smaller refinement, such as “make the lighting softer, keep everything else the same.”

### Useful controls

- **Steps:** keep each model's default: 20 for InstructPix2Pix, 25 for SDXL, 16 for FLUX 3, 4 for FIBO 2 Turbo, and 40 for Qwen. More is not always better, especially for turbo models.
- **Prompt guidance:** start at 7.5 for InstructPix2Pix/SDXL. FLUX 3 and FIBO 2 do not use standard CFG in this integration, so Herogod hides the irrelevant controls.
- **True CFG (Qwen):** defaults to 4.0; use Qwen's Advanced settings to adjust it rather than standard prompt guidance.
- **Image adherence (InstructPix2Pix):** higher values retain more of the source. Start at 1.5.
- **Change strength (SDXL):** lower values stay closer to the input; higher values allow bigger changes. Start around 0.65.
- **Seed:** `-1` selects a random seed. Reuse a reported seed when you want comparable variations.

Results can distort text, hands, logos, and identity. Always inspect before use.

## 5. Face-swap guide

Face swap is separate from prompt-based editing and always runs on the local/Colab InsightFace backend.

1. Install `requirements-face.txt` and run `scripts/setup_face_swap.py`, or enable the Colab option.
2. Choose **Face swap** in the composer.
3. Add the **target image** (the scene/body to modify).
4. Choose **Add face reference** and add the image supplying the identity.
5. Confirm that you own the images or have permission from every recognizable person.
6. Send. Herogod uses the largest detected face in each image and returns a PNG.

Use clear, front-facing, well-lit faces. No detection or poor alignment can reduce quality. The app requires a consent checkbox, but that checkbox is not a substitute for actually obtaining permission.

**Never** use face swap for non-consensual intimate imagery, sexual depiction of minors, impersonation, fraud, harassment, or presenting fabricated events as genuine. Label synthetic media when sharing could otherwise mislead. Review current InsightFace/INSwapper licensing, especially its non-commercial research restrictions.

## 6. Troubleshooting

### “Backend offline”

- Confirm `python server.py` is still running or Colab has not disconnected.
- Test `http://127.0.0.1:8000/api/health` locally (with your Herogod token if configured).
- For Colab, check whether the `trycloudflare.com` URL changed.
- An HTTPS frontend generally needs an HTTPS backend.
- Check the Herogod access token; a 401 means authentication failed. For gated HF models, a 403 can instead mean you have not accepted the model conditions or your HF token lacks access.

### Local model is downloading or slow

The first request loads weights and initializes the pipeline. Watch the terminal or `/tmp/herogod-server.log` in Colab. A T4 can still take time per edit. Do not start many concurrent requests; the default concurrency is one.

### CUDA out of memory

- Restart the runtime to clear other model allocations.
- Use InstructPix2Pix instead of a larger checkpoint. Offload and resizing do not make every 9B/20B model fit a free T4.
- Lower `HEROGOD_MAX_IMAGE_SIZE` to `512` or `640` for models that support smaller inputs.
- Set `HEROGOD_CPU_OFFLOAD=1` for SDXL, FIBO 2, or Qwen and ensure sufficient system RAM. The FLUX 3 modular pipeline does not expose standard CPU-offload methods in this integration.
- Disable `HEROGOD_FLUX_UPSAMPLING` and provide a full caption instead; the PUP is a separate large model.
- Keep steps at model defaults and concurrency at one.

### Hugging Face model unavailable

Serverless inference support is independent of model download availability. Check the model's inference widget/provider support, token permission, and remaining credits. Local Diffusers is the fallback included for supported hosted options.

For a local gated-model 401/403, confirm that `HF_TOKEN` or the UI's HF token belongs to the same account that accepted the model conditions. Herogod only tests whether a token is present during health checks; the Hub validates it during the actual download.

### Gemini returned no image

The provider can return text only or decline a request. Confirm that you selected an image-capable model, the key has access, and free-tier quota remains. Herogod does not retry with altered prompts to evade a refusal.

### No face detected

Use a larger, unobstructed, front-facing face with even lighting. Try a different reference/target. Ensure `models/inswapper_128.onnx` exists and the InsightFace detector pack downloaded successfully.

### Browser history is missing

IndexedDB behavior for `file://` pages varies by browser and privacy mode. Serve the UI through `python server.py` for a stable origin. Download important images; browser history is convenience storage, not a backup.

## 7. Safety, licensing, and privacy checklist

- [ ] I have the right to upload and modify every image.
- [ ] Every recognizable person has consented where identity/face editing is involved.
- [ ] I reviewed the model license and the compute/API provider's terms.
- [ ] I understand that free quotas and Colab runtimes can end or change.
- [ ] I will not create exploitative, deceptive, harassing, or non-consensual content.
- [ ] I will label generated edits when presentation as an authentic photo could mislead.
- [ ] My provider keys are only in backend secrets, and my public tunnel has a strong Herogod token.
- [ ] I downloaded important results and will stop the public tunnel when finished.
