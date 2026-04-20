"""
KotenAI Image Generator — Frontend HTML Builder

Loads `index.html` from the container filesystem and injects
dynamic content (aspect-ratio options, style presets, example prompts).

The HTML template uses simple `__TOKEN__` markers instead of Python
f-strings so that CSS `{...}` and JS `${...}` syntax remains untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE_PATH = Path("/root/frontend/index.html")

_LOCAL_TEMPLATE_PATH = Path(__file__).parent / "index.html"


def _get_template() -> str:
    """Read the HTML template file. Works in both Modal container and locally."""
    path = _TEMPLATE_PATH if _TEMPLATE_PATH.exists() else _LOCAL_TEMPLATE_PATH
    return path.read_text(encoding="utf-8")


def build_html(
    app_title: str,
    aspect_ratios: dict[str, tuple[int, int]],
    style_presets: dict[str, str],
    examples: list[str],
) -> str:
    """
    Render the HTML template by substituting `__TOKEN__` placeholders.

    Args:
        app_title:     Title shown in the browser tab.
        aspect_ratios: Dict of label → (width, height).
        style_presets: Dict of label → prompt prefix string.
        examples:      List of example prompt strings.

    Returns:
        Fully rendered HTML string ready to serve.
    """
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
    html = html.replace("__AR_OPTIONS__", ar_options)
    html = html.replace("__STYLE_OPTIONS__", style_options)
    html = html.replace("__EXAMPLES_JS__", examples_js)
    return html
