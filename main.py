"""
KotenAI Image Generator — Main Entry Point

Two Modal classes, one per model family, so each keeps its own
warm container and model weights in GPU memory independently:

  ERNIEGenerator  → baidu/ERNIE-Image-Turbo
  ZImageGenerator → Tongyi-MAI/Z-Image-Turbo
  web             → FastAPI + HTML frontend  (CPU only)
"""

from __future__ import annotations

import modal
from loguru import logger
from pydantic import BaseModel, Field

from config import (
    APP_NAME,
    DEFAULT_MODEL_ID,
    GPU_TYPE,
    HF_CACHE_PATH,
)

model_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)

_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.51.0",
        "accelerate>=1.2.0",
        "loguru>=0.7.0",
        "safetensors>=0.4.5",
        "huggingface_hub[hf_transfer]>=0.30.0",
        "sentencepiece>=0.2.0",
        "Pillow>=11.0.0",
        "pydantic>=2.10.0",
        "ninja",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": HF_CACHE_PATH,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("config")
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.10.0",
        "loguru>=0.7.0",
    )
    .add_local_python_source("config")
    .add_local_python_source("frontend")
    .add_local_dir("frontend", remote_path="/root/frontend")
)

app = modal.App(APP_NAME)


class GenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    width: int = Field(1024, ge=512, le=1376)
    height: int = Field(1024, ge=512, le=1376)
    num_images: int = Field(1, ge=1, le=4)
    num_inference_steps: int = Field(8, ge=4, le=20)
    guidance_scale: float = Field(1.0, ge=0.0, le=10.0)
    seed: int | None = None
    use_pe: bool = True
    style_prefix: str = ""


def _run_inference(pipe, kwargs: dict) -> dict:
    """Core generation logic, shared between both generator classes."""
    import io
    import base64
    import random
    import time
    import torch

    prompt = kwargs["prompt"]
    style_prefix = kwargs.get("style_prefix", "")
    width = kwargs["width"]
    height = kwargs["height"]
    num_images = kwargs["num_images"]
    num_inference_steps = kwargs["num_inference_steps"]
    guidance_scale = kwargs["guidance_scale"]
    seed = kwargs.get("seed")
    use_pe = kwargs.get("use_pe", True)

    final_prompt = f"{style_prefix.rstrip(', ')}, {prompt}" if style_prefix else prompt

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    logger.info(
        f"[GEN] '{final_prompt[:65]}…' | {width}x{height} | "
        f"{num_images}img | steps={num_inference_steps} | seed={seed}"
    )

    t0 = time.time()
    torch.cuda.empty_cache()

    # use_pe is an ERNIE-specific param; skip it for other pipelines
    supports_pe = "Ernie" in type(pipe).__name__
    call_kw = dict(
        prompt=final_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator,
    )
    if supports_pe:
        call_kw["use_pe"] = use_pe

    try:
        with torch.inference_mode():
            images = pipe(**call_kw).images

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("[GEN] OOM — falling back to sequential generation")
        images = []
        for i in range(num_images):
            call_kw["num_images_per_prompt"] = 1
            call_kw["generator"] = torch.Generator("cuda").manual_seed(seed + i)
            with torch.inference_mode():
                images.append(pipe(**call_kw).images[0])
            torch.cuda.empty_cache()

    elapsed = round(time.time() - t0, 1)

    b64 = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    logger.info(f"[GEN] Done — {len(images)} image(s) in {elapsed}s")
    return {
        "images": b64,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "prompt_used": final_prompt,
        "dimensions": f"{width}×{height}",
    }


_ERNIE_ID = "baidu/ERNIE-Image-Turbo"


@app.cls(
    gpu=GPU_TYPE,
    image=_base_image,
    volumes={HF_CACHE_PATH: model_volume},
    scaledown_window=600,
    timeout=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ERNIEGenerator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import ErnieImagePipeline

        logger.info(f"[ERNIE] Loading {_ERNIE_ID}…")
        self.pipe = ErnieImagePipeline.from_pretrained(
            _ERNIE_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        logger.info("[ERNIE] Warmup pass…")
        try:
            with torch.inference_mode():
                _ = self.pipe(
                    prompt="warmup",
                    height=1024,
                    width=1024,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    use_pe=False,
                )
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[ERNIE] Warmup failed (non-fatal): {e}")

        model_volume.commit()
        logger.info("[ERNIE] Ready ✓")

    @modal.method()
    def generate(self, **kw) -> dict:
        return _run_inference(self.pipe, kw)


_ZIMG_ID = "Tongyi-MAI/Z-Image-Turbo"


@app.cls(
    gpu=GPU_TYPE,
    image=_base_image,
    volumes={HF_CACHE_PATH: model_volume},
    scaledown_window=600,
    timeout=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ZImageGenerator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import ZImagePipeline

        logger.info(f"[ZIMG] Loading {_ZIMG_ID}…")
        self.pipe = ZImagePipeline.from_pretrained(
            _ZIMG_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        logger.info("[ZIMG] Warmup pass…")
        try:
            with torch.inference_mode():
                _ = self.pipe(
                    prompt="warmup",
                    height=1024,
                    width=1024,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                )
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[ZIMG] Warmup failed (non-fatal): {e}")

        model_volume.commit()
        logger.info("[ZIMG] Ready ✓")

    @modal.method()
    def generate(self, **kw) -> dict:
        return _run_inference(self.pipe, kw)


# ── Web / FastAPI ──────────────────────────────────────────────────────────────
@app.function(
    image=web_image,
    max_containers=1,
    scaledown_window=300,
    timeout=900,
    secrets=[modal.Secret.from_name("app-auth")],
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os
    import secrets as _sec

    from fastapi import FastAPI, HTTPException, Depends, Body
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from fastapi.exceptions import RequestValidationError

    from frontend import build_html
    from config import (
        APP_TITLE,
        MODELS,
        DEFAULT_MODEL_ID,
        ASPECT_RATIOS,
        STYLE_PRESETS,
        EXAMPLE_PROMPTS,
    )

    ADMIN_USER = os.environ.get("APP_USERNAME", "admin")
    ADMIN_PASS = os.environ.get("APP_PASSWORD", "demo1234")
    VALID_MODELS = set(MODELS)

    security = HTTPBasic()
    api = FastAPI(title=APP_TITLE, version="2.0.0")

    def verify(creds: HTTPBasicCredentials = Depends(security)) -> str:
        ok = _sec.compare_digest(
            creds.username.encode(), ADMIN_USER.encode()
        ) and _sec.compare_digest(creds.password.encode(), ADMIN_PASS.encode())
        if not ok:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )
        return creds.username

    @api.exception_handler(RequestValidationError)
    async def _val_err(request, exc):
        logger.error(f"[VALIDATION] {exc.errors()}")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    # Build HTML once per container lifetime
    _html = build_html(
        app_title=APP_TITLE,
        models=MODELS,
        default_model_id=DEFAULT_MODEL_ID,
        aspect_ratios=ASPECT_RATIOS,
        style_presets=STYLE_PRESETS,
        examples=EXAMPLE_PROMPTS,
    )

    @api.get("/", response_class=HTMLResponse)
    async def home(_: str = Depends(verify)):
        return HTMLResponse(content=_html)

    @api.post("/api/generate")
    async def generate(req: GenRequest = Body(...), _: str = Depends(verify)):
        if req.model_id not in VALID_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"model_id tidak valid. Pilihan: {sorted(VALID_MODELS)}",
            )
        try:
            kw = dict(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_images=req.num_images,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                use_pe=req.use_pe,
                style_prefix=req.style_prefix,
            )
            if req.model_id == _ERNIE_ID:
                result = await ERNIEGenerator().generate.remote.aio(**kw)
            else:
                result = await ZImageGenerator().generate.remote.aio(**kw)

            return JSONResponse(content=result)

        except Exception as exc:
            logger.error(f"[GENERATE] {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.get("/health")
    async def health():
        return {"status": "ok", "app": APP_TITLE, "models": sorted(VALID_MODELS)}

    return api


@app.function(
    image=_base_image,
    volumes={HF_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_weights(model_id: str = DEFAULT_MODEL_ID) -> None:
    """
    Pre-download weights to the shared Volume (run before first deploy).

        modal run main.py::download_weights
        modal run main.py::download_weights --model-id Tongyi-MAI/Z-Image-Turbo
    """
    from huggingface_hub import snapshot_download

    logger.info(f"[DOWNLOAD] {model_id}…")
    snapshot_download(
        repo_id=model_id,
        local_dir=f"{HF_CACHE_PATH}/hub/models--{model_id.replace('/', '--')}",
        ignore_patterns=["*.gguf", "*.bin"],
    )
    model_volume.commit()
    logger.success(f"[DOWNLOAD] Done — {model_id}")
