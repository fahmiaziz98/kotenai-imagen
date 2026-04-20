# KotenAI Image Generator
**Production-ready image generator untuk konten sosial media**
Model: `baidu/ERNIE-Image-Turbo` · Deploy: Modal.com · GPU: L40S (48GB)

---

## 📂 Project Structure
```text
.
├── main.py          # Entry point (Modal App & Logic)
├── config.py        # Konfigurasi, Preset, & Model Settings
├── frontend/        # Folder Frontend UI
│   ├── index.html   # Template HTML/CSS/JS
│   ├── builder.py   # Helper untuk render HTML
│   └── __init__.py  # Package export
└── .env             # Environment variables (Local only)
```

---

## 📋 PRD — Product Requirements Document

### Objective
Tool AI image generation sederhana, cepat, dan production-ready untuk content creator,
freelancer, dan UMKM Indonesia yang ingin membuat visual untuk IG, FB, Threads, dan artikel.

### MVP Features ✅
| Fitur | Detail |
|-------|--------|
| Text-to-Image | Prompt bebas, Bahasa Indonesia & Inggris |
| Multi-Image | Generate 1–4 gambar sekaligus |
| Aspect Ratio | 7 preset sesuai platform (IG Post, Story, FB, Artikel, dll.) |
| Style Preset | 10 gaya siap pakai (Fotorealistik, Sinematik, Anime, dll.) |
| Prompt Enhancer | Toggle PE bawaan ERNIE untuk kualitas lebih baik |
| Gallery | Grid view, klik untuk lightbox, download per gambar / semua |
| Auth | HTTP Basic Auth (username + password) |

### Optional Features ✅ (sudah diimplementasi)
- Seed control (reproducible results)
- Advanced: steps (4–20) & guidance scale (0–10)
- Loading messages sequence (UX feel)
- Error toast notifications
- Mobile responsive layout

### Success Metrics
| Metrik | Target |
|--------|--------|
| Waktu generate 1 gambar 1024x1024 | < 15 detik (L40S, 8 steps) |
| Waktu generate 4 gambar | < 45 detik |
| Biaya per 100 generate/hari | ~$1.17/hari (~$35/bulan) |
| Biaya per 500 generate/hari | ~$5.83/hari (~$175/bulan) |
| Cold start time | < 60 detik (model cached di Volume) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODAL.COM INFRASTRUCTURE                      │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │  web()               │    │  ImageGenerator              │   │
│  │  @asgi_app (CPU)     │    │  @cls (GPU: L40S 48GB)       │   │
│  │                      │    │                              │   │
│  │  FastAPI             │    │  @enter → load_model()       │   │
│  │  ├── GET  /          │───▶│    ErnieImagePipeline        │   │
│  │  ├── POST /api/gen   │    │    + Warmup pass             │   │
│  │  └── GET  /health    │    │                              │   │
│  │                      │    │  @method → generate()        │   │
│  │  HTTP Basic Auth     │    │    → returns base64 images   │   │
│  │  max_containers=1    │    │                              │   │
│  │  concurrent=20       │    │                              │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│           │                              │                       │
│           │     Modal RPC (.remote())    │                       │
│           └──────────────────────────────┘                       │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                 │
│  │  modal.Volume "kontenai-image-gen-models"   │                 │
│  │  Mount: /root/.cache/huggingface            │                 │
│  │  → HF weights cached, tidak download ulang  │                 │
│  └─────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
         ▲
         │ HTTPS
         │
┌────────────────┐
│  Browser User  │
│  index.html    │◀── File terpisah di frontend/ (builder.py)
│  fetch() API   │
└────────────────┘
```

**Request Flow:**
```
User → GET / (Browser auth dialog) → HTML page load
     → POST /api/generate (JSON payload)
     → web() → ImageGenerator().generate.remote()
     → GPU container: ErnieImagePipeline inference
     → base64 PNG list → render gallery di browser
```

---

## 🚀 Deployment Guide — Step by Step

### Prerequisites
- Python 3.12+
- Akun Modal.com (daftar gratis, dapat $30 kredit/bulan)
- (Opsional) Akun Hugging Face untuk private models

### Step 1: Install uv & Modal CLI
```bash
# Install uv (modern python manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone project
git clone https://github.com/fahmiaziz98/kotenai-imagen.git
cd kotenai-imagen

# Setup environment & dependencies
uv sync

# Login to Modal
uv run modal setup
```

### Step 2: Buat Secrets di Modal
```bash
# Gunakan 'uv run' untuk menjalankan perintah modal dalam environment proyek
uv run modal secret create app-auth \
  APP_USERNAME=admin \
  APP_PASSWORD=passwordyangkuat

uv run modal secret create huggingface-secret \
  HF_TOKEN=hf_xxxxxxxx
```

### Step 3: Pre-download model weights
```bash
uv run modal run main.py::download_weights
```

### Step 4: Development mode
```bash
uv run modal serve main.py
```

### Step 5: Production deploy
```bash
uv run modal deploy main.py
```

### Verifikasi deployment
```bash
# Cek status app
modal app list

# Lihat logs realtime
modal app logs kontenai-image-gen

# Cek health endpoint
curl https://your-url.modal.run/health
```

---

## 🔧 Konfigurasi Penting

Semua pengaturan utama ada di **`config.py`**. Tidak perlu mengedit `main.py` untuk sekadar ganti model atau preset.

### Ganti GPU (di config.py)
```python
GPU_TYPE = "L40S"      # 48GB, $1.95/hr — RECOMMENDED
GPU_TYPE = "A100-40GB" # 40GB, $2.10/hr
```

### Ganti Model (di config.py)
```python
MODEL_ID = "baidu/ERNIE-Image-Turbo"  # Default — 8B, 8 steps
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo" # Alternatif
```
> Jika ganti model ke Z-Image-Turbo, ganti juga `ErnieImagePipeline` → `ZImagePipeline`
> dan sesuaikan `use_pe` parameter (mungkin tidak ada di Z-Image).

### Scale untuk traffic lebih tinggi
```python
# Di @app.cls decorator:
max_containers=3      # Naikkan untuk parallel users
scaledown_window=300  # Turunkan untuk hemat biaya
```

---

## 💰 Estimasi Biaya

### Asumsi: ERNIE-Image-Turbo, L40S, 1024×1024, 8 steps

| Skenario | Request/hari | Waktu GPU/req | Biaya/hari | Biaya/bulan |
|----------|-------------|---------------|------------|-------------|
| Demo ringan | 50 | ~15 detik | ~$0.40 | ~$12 |
| Penggunaan normal | 200 | ~15 detik | ~$1.56 | ~$47 |
| Aktif (4 img/req) | 200 | ~45 detik | ~$4.68 | ~$140 |
| Heavy usage | 500 | ~15 detik | ~$3.90 | ~$117 |

> **Scale-to-zero**: Tidak ada request = $0. Modal sangat cost-effective untuk workload bursty.

---

## ✍️ Prompt Engineering Tips

### ERNIE-Image-Turbo Best Practices

**Dengan PE enabled (use_pe=True):** Prompt singkat sudah cukup.
```
Foto produk kopi, meja kayu, hangat
```
PE akan expand menjadi deskripsi detail otomatis.

**Dengan PE disabled (use_pe=False):** Tulis prompt lebih detail.
```
Photorealistic product photography of artisan coffee in a ceramic mug,
rustic wooden table background, warm morning light, shallow depth of field,
steam rising, 8k quality, professional food photography
```

**Template untuk konten IG:**
```
[SUBJEK], [LATAR], [GAYA LIGHTING], [MOOD/ATMOSPHERE], [KUALITAS]

Contoh:
"Portrait wanita pengusaha 30an, background gedung Jakarta modern,
 golden hour lighting, confident dan elegan, professional headshot, 8k"
```

**Template untuk foto produk UMKM:**
```
"Product photography [NAMA PRODUK], [LATAR], [PENCAHAYAAN],
 commercial style, clean background, high detail, [WARNA TEMA]"
```

**Template untuk konten artikel:**
```
"[TOPIK ARTIKEL] concept, [GAYA VISUAL], editorial photography style,
 [MOOD], professional quality, [ASPECT]"
```

**Catatan khusus ERNIE-Image-Turbo:**
- **Text rendering sangat kuat** — bisa generate gambar dengan tulisan yang terbaca
- Untuk poster dengan teks: tambahkan teks yang diinginkan langsung di prompt
- Guidance scale 1.0 (default) optimal untuk 8 steps
- Resolusi 1024×1024 memberikan kualitas terbaik; resolusi lain juga bagus

### Z-Image-Turbo (Alternatif)
- Sama-sama 8 steps, sangat ringan (6B, cukup 16GB VRAM)
- Lebih hemat biaya (bisa pakai A10G atau L4)
- Kualitas sedikit di bawah ERNIE tapi sangat cepat


---

## 🐛 Troubleshooting

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError: ErnieImagePipeline` | diffusers belum update ke main branch, tunggu atau pin ke git |
| `CUDA out of memory` | Code sudah handle fallback sequential, tapi jika tetap error → naikkan GPU ke A100-80GB |
| `401 Unauthorized` | Cek APP_USERNAME/APP_PASSWORD di Modal secret |
| `Container cold start lambat` | Jalankan `download_weights` terlebih dahulu |
| `torch.compile error` | Diabaikan otomatis (non-fatal), inferensi tetap berjalan |
| `PE (Prompt Enhancer) lambat` | Matikan PE di UI atau `use_pe=False` default |