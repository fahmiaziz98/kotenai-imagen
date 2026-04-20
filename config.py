"""
KotenAI Image Generator — Configuration
All constants, model settings and UI presets live here.
"""

from __future__ import annotations


APP_NAME: str = "kotenai-image-gen"
APP_TITLE: str = "KotenAI Image Generator"

# Supported: "baidu/ERNIE-Image-Turbo" | "Tongyi-MAI/Z-Image-Turbo"
MODEL_ID: str = "baidu/ERNIE-Image-Turbo"

HF_CACHE_PATH: str = "/root/.cache/huggingface"

# GPU options: "L40S" (recommended) | "A100-40GB" | "A100-80GB" | "H100"
GPU_TYPE: str = "L40S"

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 — IG Post / Threads": (1024, 1024),
    "4:3 — Artikel / Blog": (1200, 896),
    "3:4 — Portrait": (896, 1200),
    "16:9 — Facebook / YouTube": (1376, 768),
    "9:16 — IG Story / Reels": (768, 1376),
    "3:2 — Landscape": (1264, 848),
    "2:3 — Portrait Tall": (848, 1264),
}

STYLE_PRESETS: dict[str, str] = {
    "🎯 Default": "",
    "📸 Realistic": "photorealistic, professional photography, high resolution, sharp focus, 8k quality, natural lighting",
    "🎨 Digital Illustration": "digital illustration, vibrant colors, detailed artwork, professional design, clean linework",
    "🌸 Anime": "anime style, manga art, vibrant colors, expressive, clean linework, Japanese animation",
    "🎬 Cinematic": "cinematic photography, dramatic lighting, anamorphic lens, film grain, epic composition",
    "📱 Social Media Content": "social media content, eye-catching, modern aesthetic, clean composition, professional quality",
    "🛍️ Product Photography": "product photography, studio lighting, white background, commercial, clean, minimal shadows",
    "✏️ Poster & Grafis": "professional poster design, bold visual hierarchy, clean layout, advertising graphics",
    "🏙️ Minimalist": "minimalist design, modern aesthetic, clean composition, ample negative space, contemporary",
    "🎭 Oil Painting": "oil painting style, impressionistic brushstrokes, painterly, fine art, gallery quality",
}

EXAMPLE_PROMPTS: list[str] = [
    "Foto produk kopi artisan di atas meja kayu rustic, cangkir espresso dengan latte art, steam mengepul, bokeh, warm lighting",
    "Portrait profesional wanita pengusaha muda, background kantor modern Jakarta, confident smile, business attire, golden hour",
    "Banner promosi Hari Raya Idul Fitri, elemen batik modern, warna emas dan hijau zamrud, typography elegan, premium",
    "Konten carousel resep masakan: mie ayam Jakarta, bahan-bahan segar di atas meja marmer putih, flat lay, appetizing",
    "Suasana kafe aesthetic di Semarang, tanaman hijau, dinding bata ekspos, buku dan laptop, cozy vibes, film photography",
    "Ilustrasi digital city branding Semarang: Lawang Sewu, Simpang Lima, Tugu Muda, vibrant colors, modern flat design",
]
