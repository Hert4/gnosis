"""
smartsearch/render.py — PDF page rendering utilities.

Standalone functions (no class dependency) for rendering PDF pages
as base64 PNG images for VLM analysis.
"""

from __future__ import annotations

import base64
import io
import math

import fitz
from PIL import Image, ImageDraw


def render_page(
    pdf_path: str,
    page_num: int,
    dpi: int = 250,
    cache: dict[int, str] | None = None,
) -> str:
    """Render a single PDF page at given DPI, return base64 PNG.

    Args:
        pdf_path: Path to PDF file.
        page_num: 1-indexed page number.
        dpi: Render resolution (default 250).
        cache: Optional dict to cache rendered pages.

    Returns:
        Base64-encoded PNG string, or "" on error.
    """
    if cache is not None and page_num in cache:
        return cache[page_num]
    try:
        with fitz.open(pdf_path) as doc:
            p = doc[page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
        if cache is not None:
            cache[page_num] = b64
        return b64
    except Exception as e:
        print(f"  [Render] Page {page_num}: {e}")
        return ""


def render_grid(
    pdf_path: str,
    pages: list[int],
    total_pages: int,
    thumb_width: int = 1000,
    cols: int = 3,
    max_pages: int = 10,
) -> tuple[str, list[int]]:
    """Render multiple PDF pages as a labeled grid image.

    Args:
        pdf_path: Path to PDF file.
        pages: List of 1-indexed page numbers to render.
        total_pages: Total pages in document (for bounds checking).
        thumb_width: Width of each thumbnail in pixels.
        cols: Number of columns in grid.
        max_pages: Maximum pages per grid.

    Returns:
        (base64_png, actual_pages_rendered)
    """
    thumbs = []
    actual = []

    doc = fitz.open(pdf_path)
    for page_num in pages[:max_pages]:
        if page_num < 1 or page_num > total_pages:
            continue
        page = doc[page_num - 1]
        scale = thumb_width / page.rect.width
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Label page number
        draw = ImageDraw.Draw(img)
        label = f"P.{page_num}"
        draw.rectangle([0, 0, 55, 18], fill="black")
        draw.text((4, 2), label, fill="white")

        thumbs.append(img)
        actual.append(page_num)
    doc.close()

    if not thumbs:
        return "", []

    grid_cols = min(cols, len(thumbs))
    rows = math.ceil(len(thumbs) / grid_cols)
    thumb_h = max(t.height for t in thumbs)
    grid = Image.new("RGB", (grid_cols * thumb_width, rows * thumb_h), "white")

    for i, img in enumerate(thumbs):
        r, c = divmod(i, grid_cols)
        grid.paste(img, (c * thumb_width, r * thumb_h))

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), actual


def render_zoom(
    pdf_path: str,
    page_num: int,
    region: str,
    dpi: int = 250,
) -> str:
    """Render a cropped region of a page at 2x DPI.

    Args:
        pdf_path: Path to PDF file.
        page_num: 1-indexed page number.
        region: "top", "bottom", "left", "right", or "center".
        dpi: Base DPI (will be doubled for zoom).

    Returns:
        Base64-encoded PNG of the cropped region.
    """
    zoom_dpi = dpi * 2
    try:
        with fitz.open(pdf_path) as doc:
            p = doc[page_num - 1]
            mat = fitz.Matrix(zoom_dpi / 72, zoom_dpi / 72)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except Exception as e:
        return ""

    w, h = img.size
    crops = {
        "top":    (0,     0,     w,     h // 2),
        "bottom": (0,     h // 2, w,     h),
        "left":   (0,     0,     w // 2, h),
        "right":  (w // 2, 0,     w,     h),
        "center": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
    }
    box = crops.get(region, crops["center"])
    cropped = img.crop(box)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
