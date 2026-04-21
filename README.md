# KotenAI Image Generator

AI image generation tool untuk pembuatan konten visual sosial media.
Dibangun di atas Modal.com dengan GPU serverless dan model open-source dari Hugging Face.

---

## Stack

| Layer | Tech |
|---|---|
| Inference | `baidu/ERNIE-Image-Turbo` atau `Tongyi-MAI/Z-Image-Turbo` via Diffusers |
| Backend | FastAPI, Modal.com (serverless GPU) |
| Frontend | Vanilla HTML/CSS/JS (single-file template) |
| Storage | Modal Volume (HF model cache) |
| Auth | HTTP Basic Auth |
| GPU | L40S 48GB (default) |

---

## Project Structure

```
.
├── main.py              # Modal App — inference classes & FastAPI web function
├── config.py            # Model registry, aspect ratios, style presets, examples
├── frontend/
│   ├── index.html       # UI template dengan __TOKEN__ placeholders
│   ├── builder.py       # Render HTML dari config ke string siap serve
│   └── __init__.py
└── .env                 # Kredensial lokal (tidak di-commit)
```

---

## Architecture

```
Browser
  │  GET /          → HTML (rendered dari builder.py)
  │  POST /api/generate → JSON payload
  ▼
web()  [FastAPI, CPU container]
  │  Verifikasi Basic Auth
  │  Validasi GenRequest (Pydantic)
  │  Route model_id → generator class
  ▼
ERNIEGenerator / ZImageGenerator  [GPU container: L40S]
  │  @enter: load pipeline + warmup pass
  │  @method generate(): inference → base64 PNG list
  ▼
modal.Volume  (/root/.cache/huggingface)
  └─ Model weights cached — tidak re-download setiap cold start
```

Request flow: `fetch('/api/generate')` → FastAPI → `generator.generate.remote.aio()` → base64 images → render gallery di browser.

Dua generator class terpisah (`ERNIEGenerator`, `ZImageGenerator`) agar masing-masing menjaga container dan weights-nya sendiri di GPU memory secara independen.

---

## Setup

**Prerequisites:** Python 3.12+, akun [Modal.com](https://modal.com), `uv`

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & setup
git clone https://github.com/fahmiaziz98/kotenai-imagen.git
cd kotenai-imagen
uv sync

# Login Modal
uv run modal setup
```

**Secrets:**

```bash
uv run modal secret create app-auth \
  APP_USERNAME=admin \
  APP_PASSWORD=your_password

# Untuk model private/gated
uv run modal secret create huggingface-secret \
  HF_TOKEN=hf_xxxxxxxx
```

**Download model weights ke Volume (jalankan sekali):**

```bash
# ERNIE-Image-Turbo (default)
uv run modal run main.py::download_weights

# Z-Image-Turbo (opsional)
uv run modal run main.py::download_weights --model-id Tongyi-MAI/Z-Image-Turbo
```

**Serve & Deploy:**

```bash
uv run modal serve main.py    # dev mode, auto-reload
uv run modal deploy main.py   # production
```

URL setelah deploy: `https://<workspace>--kotenai-image-gen-web.modal.run`

---

## Configuration

Semua pengaturan ada di `config.py`. `main.py` tidak perlu disentuh untuk ganti model, GPU, atau preset.

**GPU:**
```python
GPU_TYPE = "L40S"       # 48GB — default, cost-effective untuk inference
GPU_TYPE = "A100-40GB"  # 40GB, fallback jika L40S tidak tersedia
GPU_TYPE = "A100-80GB"  # 80GB, untuk batch 4 gambar tanpa CPU offload
GPU_TYPE = "H100"       # Tercepat, paling mahal
```

**Model:**
```python
# Di MODELS dict — tambah entry baru untuk model lain
DEFAULT_MODEL_ID = "baidu/ERNIE-Image-Turbo"
```

**Scale:**
```python
# Di @app.cls decorator di main.py
max_containers = 3      # Concurrent GPU containers
scaledown_window = 300  # Detik sebelum container sleep (hemat biaya)
```

---

## Cost Estimate

Asumsi: ERNIE-Image-Turbo, L40S, 1024x1024, 8 steps, ~15 detik/generate.

| Skenario | Request/hari | Biaya/hari | Biaya/bulan |
|---|---|---|---|
| Ringan | 50 | ~$0.40 | ~$12 |
| Normal | 200 | ~$1.56 | ~$47 |
| Aktif (4 img/req) | 200 | ~$4.68 | ~$140 |
| Heavy | 500 | ~$3.90 | ~$117 |

Scale-to-zero: tidak ada request = $0.

---

## Prompt Guide

**ERNIE-Image-Turbo dengan PE enabled** — prompt singkat cukup, model expand sendiri:
```
Foto produk kopi, meja kayu, warm lighting
```

**PE disabled atau Z-Image** — tulis lebih detail:
```
Photorealistic product photography, artisan coffee in ceramic mug,
rustic wooden table, morning light, shallow depth of field, 8k
```

Template umum:
```
[subjek], [latar], [pencahayaan], [mood], [kualitas/style]
```

Catatan model: ERNIE text rendering sangat kuat, cocok untuk poster dan grafis dengan tulisan. Z-Image lebih ringan (6B vs 8B), cocok untuk volume tinggi dengan budget terbatas. Guidance scale 1.0 + 8 steps adalah sweet spot untuk kedua model.

---

## Troubleshooting

| Error | Solusi |
|---|---|
| `ModuleNotFoundError: ErnieImagePipeline` | `pip install git+https://github.com/huggingface/diffusers.git` |
| `CUDA out of memory` | Kode sudah handle fallback sequential. Jika tetap OOM, naikkan GPU ke A100-80GB |
| `401 Unauthorized` | Cek value `APP_USERNAME` / `APP_PASSWORD` di Modal secret `app-auth` |
| Cold start sangat lambat | Pastikan `download_weights` sudah dijalankan sebelum deploy |
| Z-Image tidak support `use_pe` | Normal — PE hanya ada di ERNIE pipeline, UI otomatis menyembunyikan toggle |

---

## Roadmap

- Image-to-image (img2img dengan strength control)
- Image-to-video
- Text-to-video
