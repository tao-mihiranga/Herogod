# Herogod free setup & usage guide

This guide separates **free open weights**, **free compute**, and **free-tier API access**. They are not the same thing:

- An open model can have no per-request fee but still needs a GPU/CPU and has a license.
- Colab can provide no-cost compute when available, but sessions and accelerators are limited.
- A hosted API can provide free credits or a free tier, but requires an account, has quotas, and keeps its own usage/safety policies.

Herogod does not include shared API keys or promise unlimited generation. Check the linked official pages before relying on a current quota.

## 1. Choose your route

### A. Completely self-hosted inference — recommended

**Models included:** InstructPix2Pix, SDXL img2img, and optional INSwapper.

- **AI token:** none.
- **API cost:** none.
- **Compute:** your machine or a free Colab runtime, subject to availability.
- **Moderation:** Herogod adds no cloud moderation layer or hidden prompt rewrite. It also does not remove or bypass a hosted provider's safety controls.
- **Best default:** `timbrooks/instruct-pix2pix` on a T4 GPU.
- **License:** review each current model card before use. “Open” and “free to download” do not automatically mean unrestricted commercial use.

Model links:

- [InstructPix2Pix model card](https://huggingface.co/timbrooks/instruct-pix2pix)
- [SDXL 1.0 base model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [InsightFace repository and licensing](https://github.com/deepinsight/insightface)
- [Diffusers documentation](https://huggingface.co/docs/diffusers/index)

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

1. Create/sign in to a free Hugging Face account.
2. Create your own token with Inference Providers permission. Use the least permission needed.
3. Set `HF_TOKEN` in the backend environment or Colab Secrets. Do not put it in HTML, JavaScript, Git, or a chat prompt.
4. Start Herogod and select **Hugging Face** in the model picker.
5. If inference says the model is not deployed or supported, use local InstructPix2Pix instead. Herogod reports this rather than silently substituting a paid service.

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
2. Choose **Runtime → Change runtime type → T4 GPU**. If no free GPU is available, wait or run locally; CPU diffusion is much slower.
3. In **Runtime choices**, keep:
   - `MODEL_ID = timbrooks/instruct-pix2pix`
   - `MAX_IMAGE_SIZE = 768`
   - `ENABLE_FACE_SWAP = False` unless you need it and have reviewed its license.
4. For hosted APIs, add `HF_TOKEN` or `GEMINI_API_KEY` under Colab's key/Secrets icon and enable notebook access. Never paste keys into a code cell.
5. Run cells from top to bottom. Installation and the first model download can take time.
6. Copy the printed studio URL and Herogod access token. The token protects the tunnel; it is not an AI key.
7. Open the studio URL, go to **Connection settings**, and paste the token. Click **Test connection**, then **Save & connect**.
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

Install a PyTorch build matching your CUDA/CPU environment, then:

```bash
pip install -r requirements.txt -r requirements-local.txt
python server.py --host 127.0.0.1
```

The first run downloads model weights from the model hub. Later runs can use the local cache. GPU memory needs depend on image size, model, and settings.

### Hosting the frontend online

The frontend is static. Host `index.html`, `styles.css`, `app.js`, and `favicon.svg` on any static host, then point Settings at an HTTPS Herogod backend. Alternatively, `python server.py` serves both frontend and API from one origin.

For an HTTPS-hosted page, use an HTTPS backend tunnel to avoid mixed-content restrictions. Do not expose provider keys in static hosting environment variables intended for browser code; this project only reads them in Python.

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

- **Steps:** more is not always better. Start at 20; increase only if details need work.
- **Prompt guidance:** higher values follow the text more strongly but can reduce naturalness.
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
- Check the access token; a 401/403 means authentication failed, not that the model is missing.

### Local model is downloading or slow

The first request loads weights and initializes the pipeline. Watch the terminal or `/tmp/herogod-server.log` in Colab. A T4 can still take time per edit. Do not start many concurrent requests; the default concurrency is one.

### CUDA out of memory

- Restart the runtime to clear other model allocations.
- Use InstructPix2Pix instead of SDXL.
- Lower `HEROGOD_MAX_IMAGE_SIZE` to `512` or `640`.
- Set `HEROGOD_CPU_OFFLOAD=1`.
- Keep steps moderate and concurrency at one.

### Hugging Face model unavailable

Serverless inference support is independent of model download availability. Check the model's inference widget/provider support, token permission, and remaining credits. Local Diffusers is the reliable fallback included for this case.

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
