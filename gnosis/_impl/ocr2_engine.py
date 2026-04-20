"""
ocr2_engine.py — Nanonets-OCR2-3B OCR engine.

Two backends (auto-selected):
  1. sglang API (fast, if server running)
  2. Local transformers (fallback, loads model into GPU directly)

Produces verbatim structured markdown (headings, HTML tables) instead of
VLM-generated descriptions.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Callable

import fitz
import requests


class OCR2Engine:
    """Nanonets-OCR2-3B OCR with sglang API or local transformers fallback."""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:30000",
        model_name: str = "nanonets/Nanonets-OCR2-3B",
        max_tokens: int = 4096,
        timeout: int = 120,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self._mode: str = ""  # "api" or "local"
        self._local_model = None
        self._local_processor = None

    def _headers(self) -> dict[str, str]:
        h = dict(self.extra_headers)
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def is_available(self) -> bool:
        """Check if OCR2 is usable (either API or local)."""
        # Once mode is set, don't re-check
        if self._mode:
            return True
        # Try API first
        if self._check_api():
            self._mode = "api"
            return True
        # Try local model
        if self._check_local():
            self._mode = "local"
            return True
        return False

    def _check_api(self) -> bool:
        # sglang/vLLM expose /health (no auth). OpenAI-compatible endpoints
        # (OpenRouter, vLLM with auth, LM Studio, ...) use /v1/models with Bearer.
        try:
            r = requests.get(
                f"{self.api_base}/health", timeout=3, headers=self._headers()
            )
            if r.status_code == 200:
                return True
        except Exception:
            pass
        try:
            r = requests.get(
                f"{self.api_base}/v1/models", timeout=5, headers=self._headers()
            )
            return r.status_code == 200
        except Exception:
            return False

    def _check_local(self) -> bool:
        """Check if model can be loaded locally via transformers."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            # Check if model is cached (don't download, just check)
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(self.model_name, "config.json")
            return result is not None and result != "sentinel"  # sentinel = not cached
        except Exception:
            return False

    def _ensure_local_model(self) -> None:
        """Load model into GPU (lazy, one-time)."""
        if self._local_model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"  [OCR2] Loading {self.model_name} locally...")
        self._local_model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto",
        )
        self._local_model.eval()
        self._local_processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"  [OCR2] Model loaded on {self._local_model.device}")

    def ocr_image(
        self,
        image,
        prompt: str = (
            "Extract the text from the above document as if you were reading it naturally. "
            "Return the tables in html format. Return the equations in LaTeX representation. "
            "If there is an image in the document and image caption is not present, "
            "add a small description of the image inside the <img></img> tag; "
            "otherwise, add the image caption inside <img></img>. "
            "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. "
            "Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number>. "
            "Prefer using ☐ and ☑ for check boxes."
        ),
    ) -> str:
        """OCR a PIL Image, return structured markdown."""
        if self._mode == "api":
            return self._ocr_image_api(image, prompt)
        return self._ocr_image_local(image, prompt)

    def _ocr_image_api(self, image, prompt: str) -> str:
        """OCR via sglang API."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self.api_base}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
            headers=self._headers(),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OCR2 API error {resp.status_code}: {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"]

    def _ocr_image_local(self, image, prompt: str) -> str:
        """OCR via local transformers model."""
        import torch

        self._ensure_local_model()
        model = self._local_model
        processor = self._local_processor

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            torch.cuda.empty_cache()
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0]

    def ocr_pdf_pages(
        self,
        pdf_path: str,
        pages: list[int],
        dpi: int = 250,
        prompt: str = (
            "Extract the text from the above document as if you were reading it naturally. "
            "Return the tables in html format. Return the equations in LaTeX representation. "
            "If there is an image in the document and image caption is not present, "
            "add a small description of the image inside the <img></img> tag; "
            "otherwise, add the image caption inside <img></img>. "
            "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. "
            "Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number>. "
            "Prefer using ☐ and ☑ for check boxes."
        ),
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> dict[int, str]:
        """Render PDF pages to images then OCR each.

        Args:
            pdf_path: Path to the PDF file.
            pages: 1-indexed page numbers to OCR.
            dpi: Rendering resolution.
            prompt: Prompt for the OCR model.
            on_progress: Callback(page_num, current_idx, total).

        Returns:
            {page_num: markdown_text} for pages with non-empty results.
        """
        from PIL import Image

        results: dict[int, str] = {}
        total = len(pages)
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        for i, page_num in enumerate(pages, 1):
            try:
                import sys as _sys
                print(f"  [OCR2] Page {page_num} ({i}/{total})...", end="", flush=True)
                page = doc[page_num - 1]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

                text = self.ocr_image(img, prompt)
                if text:
                    results[page_num] = text
                    print(f" OK ({len(text)} chars)", flush=True)
                else:
                    print(" (empty)", flush=True)

                if on_progress:
                    on_progress(page_num, i, total)
            except Exception as e:
                print(f" ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

        doc.close()
        return results

    def unload(self) -> None:
        """Free GPU memory by unloading local model."""
        if self._local_model is not None:
            import torch
            del self._local_model
            del self._local_processor
            self._local_model = None
            self._local_processor = None
            torch.cuda.empty_cache()
