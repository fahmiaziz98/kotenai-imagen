from __future__ import annotations

APP_NAME: str = "kotenai-image-gen"
APP_TITLE: str = "KotenAI"

HF_CACHE_PATH: str = "/root/.cache/huggingface"

# GPU options: "L40S" | "A100-40GB" | "A100-80GB" | "H100"
GPU_TYPE: str = "L40S"

# ── Model registry ─────────────────────────────────────────────────────────
MODELS: dict[str, dict] = {
    "baidu/ERNIE-Image-Turbo": {
        "label": "ERNIE-Image-Turbo  —  Baidu · 8B",
        "description": "8 langkah · teks dalam gambar sangat akurat · Apache 2.0",
        "pipeline": "ErnieImagePipeline",
        "default_steps": 8,
        "default_cfg": 1.0,
        "supports_pe": True,
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "label": "Z-Image-Turbo  —  Alibaba · 6B",
        "description": "8 langkah · lebih ringan & cepat · Apache 2.0",
        "pipeline": "ZImagePipeline",
        "default_steps": 8,
        "default_cfg": 1.0,
        "supports_pe": False,
    },
}

DEFAULT_MODEL_ID: str = "baidu/ERNIE-Image-Turbo"

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1  —  IG Post / Threads": (1024, 1024),
    "4:3  —  Artikel / Blog": (1200, 896),
    "3:4  —  Portrait": (896, 1200),
    "16:9  —  Facebook / YouTube": (1376, 768),
    "9:16  —  IG Story / Reels": (768, 1376),
    "3:2  —  Landscape": (1264, 848),
    "2:3  —  Portrait Tall": (848, 1264),
}

STYLE_PRESETS: dict[str, str] = {
    "Default": "",
    "Fotorealistik": "photorealistic, professional photography, high resolution, sharp focus, 8k quality, natural lighting",
    "Ilustrasi Digital": "digital illustration, vibrant colors, detailed artwork, professional design, clean linework",
    "Anime": "anime style, manga art, vibrant colors, expressive, clean linework, Japanese animation",
    "Sinematik": "cinematic photography, dramatic lighting, anamorphic lens, film grain, epic composition",
    "Konten Sosial Media": "social media content, eye-catching, modern aesthetic, clean composition, professional quality",
    "Foto Produk": "product photography, studio lighting, white background, commercial, clean, minimal shadows",
    "Poster & Grafis": "professional poster design, bold visual hierarchy, clean layout, advertising graphics",
    "Minimalis Modern": "minimalist design, modern aesthetic, clean composition, ample negative space, contemporary",
    "Seni Lukis": "oil painting style, impressionistic brushstrokes, painterly, fine art, gallery quality",
}

EXAMPLE_PROMPTS: list[str] = [
    "Foto produk kopi artisan di atas meja kayu rustic, cangkir espresso dengan latte art, steam mengepul, bokeh, warm lighting",
    "Portrait profesional wanita pengusaha muda, background kantor modern Jakarta, confident smile, business attire, golden hour",
    "Banner promosi Hari Raya Idul Fitri, elemen batik modern, warna emas dan hijau zamrud, typography elegan, premium",
    "Konten carousel resep masakan: mie ayam Jakarta, bahan-bahan segar di atas meja marmer putih, flat lay, appetizing",
    "Suasana kafe aesthetic di Semarang, tanaman hijau, dinding bata ekspos, buku dan laptop, cozy vibes, film photography",
    "Ilustrasi digital city branding Semarang: Lawang Sewu, Simpang Lima, Tugu Muda, vibrant colors, modern flat design",
]
