"""Estimate image tokens from the selected model's documented sizing rules."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit

from astrbot.core.agent.context.token_counter import IMAGE_TOKEN_ESTIMATE
from PIL import Image


def deepseek_image_tokens(width, height):
    """Match the official V4.1 Flash calculator, checked on 2026-10-24.

    Source: https://api-docs.deepseek.com/zh-cn/quick_start/token_usage/
    This estimates visual tokens, not transmitted Base64 text or billed usage.
    """
    if width <= 0 or height <= 0:
        raise ValueError("invalid_image_dimensions")

    def resize(w, h):
        if w * h < 295936:
            scale = math.sqrt(295936 / (w * h))
            w, h = int(w * scale), int(h * scale)
        rounded_w = ((w + 13) // 14) * 14
        rounded_h = ((h + 13) // 14) * 14
        cols, rows = (rounded_w // 14 + 2) // 3, (rounded_h // 14 + 2) // 3
        tokens = rows * (cols + 1) + 2
        if tokens > 1024:
            ratio = h / w
            target_cols = math.sqrt(1022 / ratio + 0.25) - 0.5
            target_rows = target_cols * ratio
            if target_cols < 1:
                rounded_w, rounded_h = 42, (1022 // 2) * 42
            elif target_rows < 1:
                rounded_w, rounded_h = (1022 - 1) * 42, 42
            else:
                scale = min(int(target_cols) * 42 / w, int(target_rows) * 42 / h)
                rounded_w = int(w * scale / 14) * 14
                rounded_h = int(h * scale / 14) * 14
            cols, rows = (rounded_w // 14 + 2) // 3, (rounded_h // 14 + 2) // 3
            tokens = rows * (cols + 1) + 2
        return rounded_w, rounded_h, rows, cols, tokens

    current = resize(width, height)
    for _ in range(1, 10):
        following = resize(current[0], current[1])
        if following == current:
            return min(1024, current[-1])
        current = following
    return 1024


class ImageTokenEstimator:
    """Read local image headers only; never download a URL to estimate tokens."""

    def __init__(self, model):
        self.deepseek = str(model).lower().rsplit("/", 1)[-1] in {
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        }
        self.method = "deepseek-v4.1-dimensions" if self.deepseek else "astrbot-generic"
        self.cache = {}

    def count(self, part):
        if not self.deepseek:
            return IMAGE_TOKEN_ESTIMATE
        image = part.get("image_url", {})
        reference = image.get("url", "") if isinstance(image, dict) else image
        detail = image.get("detail", "auto") if isinstance(image, dict) else "auto"
        key = (str(reference), str(detail))
        if key in self.cache:
            return self.cache[key]
        tokens = 1024
        try:
            if detail == "low":
                tokens = deepseek_image_tokens(512, 512)
            elif isinstance(reference, str) and reference.startswith("data:image/"):
                raw = base64.b64decode(reference.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as source:
                    tokens = deepseek_image_tokens(*source.size)
            elif isinstance(reference, str) and not reference.startswith(
                ("http://", "https://")
            ):
                path = (
                    unquote(urlsplit(reference).path)
                    if reference.startswith("file:")
                    else reference
                )
                with Image.open(Path(path)) as source:
                    tokens = deepseek_image_tokens(*source.size)
        except (OSError, ValueError, IndexError, Image.DecompressionBombError):
            pass
        self.cache[key] = tokens
        return tokens
