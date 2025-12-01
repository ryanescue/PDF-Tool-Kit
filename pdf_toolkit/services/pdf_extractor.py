"""
Utilities for extracting text from PDFs using multiple backends.

This module exposes a single entry point, ``extract_pdf_text``. The implementation
is shared between the CLI helper script and the Django view so the extraction logic
stays in one place.
"""

from __future__ import annotations

import re  # stdlib regex helpers for normalization
from io import BytesIO  # stdlib in-memory buffer for rendered pages
from pathlib import Path  # stdlib path helper for file handling
from typing import Iterable, List, Literal, Optional, Sequence

import pypdfium2 as pdfium  # PDF renderer used for OCR backends
from PIL import Image  # Pillow image object used for OCR preprocessing
from PyPDF2 import PdfReader  # Reading PDF text layer when available

Method = Literal["pypdf", "tesseract", "easyocr", "auto"]

__all__ = [
    "convert_pdf_to_images",
    "extract_pdf_text",
    "extract_text_pypdf",
    "extract_text_tesseract",
    "extract_text_easyocr",
    "normalize_text",
]


def _lazy_import_tesseract():
    from pytesseract import image_to_string  # OCR helper from the pytesseract wrapper

    return image_to_string


def _lazy_import_easyocr():
    from easyocr import Reader  # EasyOCR model loader for multilingual OCR

    return Reader(["en"])


def convert_pdf_to_images(pdf_path: Path, *, scale: float = 300 / 72) -> List[dict]:
    """Render each page of the PDF to a JPEG image in memory."""
    pdf_file = pdfium.PdfDocument(str(pdf_path))
    images: List[dict] = []
    for i in range(len(pdf_file)):
        page = pdf_file.get_page(i)
        pil_image = page.render(scale=scale).to_pil()
        buf = BytesIO()
        pil_image.save(buf, format="jpeg", optimize=True)
        images.append({"page": i, "bytes": buf.getvalue()})
    return images


def extract_text_pypdf(pdf_path: Path) -> str:
    """Extract the built-in text layer via PyPDF."""
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_text_from_images(
    images: Iterable[dict], extractor
) -> str:
    chunks: List[str] = []
    for item in images:
        img = Image.open(BytesIO(item["bytes"]))
        chunks.append(str(extractor(img)))
    return "\n".join(chunks)


def extract_text_tesseract(images: Sequence[dict]) -> str:
    """Run Tesseract OCR on each rendered page Image."""
    image_to_string = _lazy_import_tesseract()
    return _extract_text_from_images(images, image_to_string)


def extract_text_easyocr(images: Sequence[dict]) -> str:
    """Run EasyOCR on each rendered page Image."""
    reader = _lazy_import_easyocr()
    chunks: List[str] = []
    for item in images:
        img = Image.open(BytesIO(item["bytes"]))
        res = reader.readtext(img)
        chunks.append("\n".join(r[1] for r in res))
    return "\n".join(chunks)


def normalize_text(text: str) -> str:
    """Collapse single newlines to spaces while preserving paragraph breaks."""
    text = text.replace("\r", "")
    placeholder = "__PARA_BREAK__"
    text = text.replace("\n\n", placeholder)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(placeholder, "\n\n").strip()
    return text


def extract_pdf_text(
    source: Path,
    *,
    method: Method = "auto",
    scale: float = 300 / 72,
    normalize: bool = False,
) -> str:
    """
    Extract text from ``source`` using the requested backend.

    ``method``:
        - ``pypdf``: read the text layer directly.
        - ``tesseract``: OCR via pytesseract (requires the Tesseract binary).
        - ``easyocr``: OCR via EasyOCR.
        - ``auto``: try PyPDF first, fallback to Tesseract if nothing was read.
    """
    if not source.exists():
        raise FileNotFoundError(source)

    requires_images = method in {"tesseract", "easyocr", "auto"}
    images: Optional[Sequence[dict]] = None
    if requires_images:
        images = convert_pdf_to_images(source, scale=scale)

    text = ""

    if method in {"pypdf", "auto"}:
        text = extract_text_pypdf(source)
        if method == "pypdf" or text.strip():
            pass  # keep PyPDF result
        else:
            if images is None:
                images = convert_pdf_to_images(source, scale=scale)
            text = extract_text_tesseract(images)
    elif method == "tesseract":
        if images is None:
            images = convert_pdf_to_images(source, scale=scale)
        text = extract_text_tesseract(images)
    elif method == "easyocr":
        if images is None:
            images = convert_pdf_to_images(source, scale=scale)
        text = extract_text_easyocr(images)
    else:  # pragma: no cover - validated by Literal
        raise ValueError(f"Unknown extraction method: {method}")

    if normalize:
        text = normalize_text(text)

    return text
