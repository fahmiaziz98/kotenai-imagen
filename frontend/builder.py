"""
KotenAI Image Generator — Frontend HTML Builder
"""

from __future__ import annotations

import json
from pathlib import Path

_CONTAINER_PATH = Path("/root/frontend/index.html")
_LOCAL_PATH = Path(__file__).parent / "index.html"


def _get_template() -> str:
    path = _CONTAINER_PATH if _CONTAINER_PATH.exists() else _LOCAL_PATH
    return path.read_text(encoding="utf-8")


def build_html(
    app_title: str,
    models: dict[str, dict],
    default_model_id: str,
    aspect_ratios: dict[str, tuple[int, int]],
    style_presets: dict[str, str],
    examples: list[str],
) -> str:
    """Render the HTML template by substituting __TOKEN__ placeholders."""

    # <select> for model picker
    model_options = "\n".join(
        f'<option value="{mid}"{" selected" if mid == default_model_id else ""}>'
        f"{meta['label']}</option>"
        for mid, meta in models.items()
    )

    # JS object: model_id → metadata needed by the UI
    model_meta_js = json.dumps(
        {
            mid: {
                "description": m["description"],
                "supports_pe": m["supports_pe"],
                "default_steps": m["default_steps"],
                "default_cfg": m["default_cfg"],
            }
            for mid, m in models.items()
        },
        ensure_ascii=False,
    )

    default_desc = models.get(default_model_id, {}).get("description", "")

    ar_options = "\n".join(
        f'<option value="{w},{h}"{" selected" if i == 0 else ""}>{label}</option>'
        for i, (label, (w, h)) in enumerate(aspect_ratios.items())
    )

    style_options = "\n".join(
        f'<option value="{prefix}"{" selected" if i == 0 else ""}>{label}</option>'
        for i, (label, prefix) in enumerate(style_presets.items())
    )

    examples_js = json.dumps(examples, ensure_ascii=False)

    html = _get_template()
    html = html.replace("__APP_TITLE__", app_title)
    html = html.replace("__MODEL_OPTIONS__", model_options)
    html = html.replace("__MODEL_META_JS__", model_meta_js)
    html = html.replace("__DEFAULT_MODEL_DESC__", default_desc)
    html = html.replace("__AR_OPTIONS__", ar_options)
    html = html.replace("__STYLE_OPTIONS__", style_options)
    html = html.replace("__EXAMPLES_JS__", examples_js)
    return html
