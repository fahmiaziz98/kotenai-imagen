"""
KotenAI Image Generator — Main Entry Point

Modal app comprising two functions:
  • ImageGenerator  – GPU-accelerated inference (ERNIE / Z-Image)
  • web             – FastAPI frontend (HTML + REST API)
"""

from __future__ import annotations

import modal
from loguru import logger
from pydantic import BaseModel, Field

from config import (
    APP_NAME,
    APP_TITLE,
    MODEL_ID,
    GPU_TYPE,
    HF_CACHE_PATH,
)

model_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)

inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "wget", "git")
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
        "pydantic>=2.10.0",
        "safetensors>=0.4.5",
        "huggingface_hub[hf_transfer]>=0.30.0",
        "sentencepiece>=0.2.0",
        "Pillow>=11.0.0",
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
    width: int = Field(1024, ge=512, le=1376)
    height: int = Field(1024, ge=512, le=1376)
    num_images: int = Field(1, ge=1, le=4)
    num_inference_steps: int = Field(8, ge=4, le=20)
    guidance_scale: float = Field(1.0, ge=0.0, le=10.0)
    seed: int | None = None
    use_pe: bool = True
    style_prefix: str = ""


@app.cls(
    gpu=GPU_TYPE,
    image=inference_image,
    volumes={HF_CACHE_PATH: model_volume},
    scaledown_window=600,
    timeout=900,
    max_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ImageGenerator:
    @modal.enter()
    def load_model(self) -> None:
        """Load the diffusion pipeline when the container starts."""
        import torch
        from diffusers import ErnieImagePipeline, ZImagePipeline

        logger.info(f"[INIT] Loading {MODEL_ID} → {GPU_TYPE}...")

        if MODEL_ID == "baidu/ERNIE-Image-Turbo":
            self.pipe = ErnieImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        elif MODEL_ID == "Tongyi-MAI/Z-Image-Turbo":
            self.pipe = ZImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        else:
            raise ValueError(f"Unsupported MODEL_ID: {MODEL_ID!r}")

        # torch.compile is intentionally disabled.
        # Re-enable once traffic is stable and GPU stays warm:
        #   self.pipe.transformer = torch.compile(
        #       self.pipe.transformer,
        #       mode="default",   # or "reduce-overhead"
        #       fullgraph=False,  # True is stricter but slower to compile
        #   )
        logger.info("[INIT] torch.compile skipped ✓")

        # Warmup — trigger CUDA kernel init before the first real request
        logger.info("[INIT] Running warmup pass...")
        try:
            with torch.inference_mode():
                _ = self.pipe(
                    prompt="warmup pass",
                    height=1024,
                    width=1024,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    use_pe=False,
                )
            torch.cuda.empty_cache()
            logger.info("[INIT] Warmup done ✓")
        except Exception as exc:
            logger.error(f"[INIT] Warmup failed (non-fatal): {exc}")

        # Persist cached weights to the volume for subsequent containers
        model_volume.commit()
        logger.info(f"[INIT] {APP_TITLE} ready! ✓")

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        use_pe: bool = True,
        style_prefix: str = "",
    ) -> dict:
        """
        Generate images from a text prompt.

        Returns:
            dict with keys:
                images          – list of base64-encoded PNG data-URIs
                elapsed_seconds – float, wall-clock generation time
                seed            – int, seed actually used
                prompt_used     – str, final prompt after style prefix is applied
                dimensions      – str, e.g. "1024×1024"
        """
        import io
        import base64
        import random
        import time
        import torch

        final_prompt = (
            f"{style_prefix.rstrip(', ')}, {prompt}" if style_prefix else prompt
        )

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(seed)

        logger.info(
            f"[GEN] '{final_prompt[:70]}…' | {width}×{height} | "
            f"{num_images}img | steps={num_inference_steps} | seed={seed}"
        )

        t_start = time.time()
        torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                output = self.pipe(
                    prompt=final_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    use_pe=use_pe,
                )
            images = output.images

        except torch.cuda.OutOfMemoryError:
            # Fallback: generate one image at a time to recover from OOM
            torch.cuda.empty_cache()
            logger.warning("[GEN] OOM — falling back to sequential generation")
            images = []
            for i in range(num_images):
                g = torch.Generator("cuda").manual_seed(seed + i)
                with torch.inference_mode():
                    out = self.pipe(
                        prompt=final_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=1,
                        generator=g,
                        use_pe=use_pe,
                    )
                images.append(out.images[0])
                torch.cuda.empty_cache()

        elapsed = round(time.time() - t_start, 1)

        b64_images: list[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            b64_images.append(f"data:image/png;base64,{b64}")

        logger.info(f"[GEN] ✓ {len(images)} image(s) in {elapsed}s")

        return {
            "images": b64_images,
            "elapsed_seconds": elapsed,
            "seed": seed,
            "prompt_used": final_prompt,
            "dimensions": f"{width}×{height}",
        }


# ── Web / FastAPI ─────────────────────────────────────────────────────────────
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
    import secrets as _secrets

    from fastapi import FastAPI, HTTPException, Depends, Body
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from fastapi.exceptions import RequestValidationError

    from frontend import build_html
    from config import APP_TITLE, ASPECT_RATIOS, STYLE_PRESETS, EXAMPLE_PROMPTS

    ADMIN_USER = os.environ.get("APP_USERNAME", "admin")
    ADMIN_PASS = os.environ.get("APP_PASSWORD", "demo1234")

    security = HTTPBasic()
    api = FastAPI(title=APP_TITLE, version="1.0.0")

    def verify(credentials: HTTPBasicCredentials = Depends(security)) -> str:
        ok_u = _secrets.compare_digest(
            credentials.username.encode(), ADMIN_USER.encode()
        )
        ok_p = _secrets.compare_digest(
            credentials.password.encode(), ADMIN_PASS.encode()
        )
        if not (ok_u and ok_p):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username

    @api.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        logger.error(f"[VALIDATION] {exc.errors()}")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @api.get("/", response_class=HTMLResponse)
    async def homepage(_: str = Depends(verify)):
        html = build_html(APP_TITLE, ASPECT_RATIOS, STYLE_PRESETS, EXAMPLE_PROMPTS)
        return HTMLResponse(content=html)

    @api.post("/api/generate")
    async def generate_endpoint(
        req: GenRequest = Body(...),
        _: str = Depends(verify),
    ):
        try:
            result = await ImageGenerator().generate.remote.aio(
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
            return JSONResponse(content=result)
        except Exception as exc:
            logger.error(f"[GENERATE] {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID, "app": APP_TITLE}

    return api


# ── One-time weight download utility ─────────────────────────────────────────
@app.function(
    image=inference_image,
    volumes={HF_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_weights() -> None:
    """
    Pre-download model weights to the shared Volume.

    Run once before deploying:
        modal run main.py::download_weights
    """
    from huggingface_hub import snapshot_download

    logger.info(f"[DOWNLOAD] Downloading {MODEL_ID}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=f"{HF_CACHE_PATH}/hub/models--{MODEL_ID.replace('/', '--')}",
        ignore_patterns=["*.gguf", "*.bin"],
    )
    model_volume.commit()
    logger.success("[DOWNLOAD] ✓ Model weights downloaded successfully")
