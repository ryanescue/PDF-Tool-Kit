"""
Utilities for extracting text from PDFs using multiple backends.

This module exposes a single entry point, ``extract_pdf_text``. The implementation
is shared between the CLI helper script and the Django view so the extraction logic
stays in one place.
"""

from __future__ import annotations

import re #helpers for normalization
from io import BytesIO #in-memory buffer for rendered pages
from pathlib import Path #path helper for file handling
from typing import Iterable, List, Literal, Optional, Sequence
#OCR = Optical Character Recognition
import pypdfium2 as pdfium #PDF renderer
from PIL import Image, ImageFilter, ImageOps #OCR preprocessing
from PyPDF2 import PdfReader #Reading PDF text layer when available

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
    from pytesseract import image_to_string  #OCR helper from the pytesseract wrapper
    return image_to_string

def _lazy_import_easyocr():
    from easyocr import Reader  # EasyOCR model loader for multilingual OCR
    return Reader(["en"])

def convert_pdf_to_images(
    pdf_path: Path, *, scale: float = 300 / 72, pages: Optional[Sequence[int]] = None
) -> List[dict]:
    #render selected pages of the pdf to jpg images in memory
    pdf_file=pdfium.PdfDocument(str(pdf_path)) #used to open pdf
    indices= _prepare_page_indexes(pages, len(pdf_file))
    images: List[dict]=[]
    target_pages=indices if indices is not None else range(len(pdf_file))
    for i in target_pages: #writes that image into a bytesio buffer in jpeg format
        page =pdf_file.get_page(i)
        pil_image= page.render(scale=scale).to_pil()
        buf= BytesIO()
        pil_image.save(buf, format="jpeg", optimize=True)
        images.append({"page": i, "bytes": buf.getvalue()})
    return images


def extract_text_pypdf(pdf_path: Path, pages: Optional[Sequence[int]] = None) -> str:
    reader = PdfReader(str(pdf_path))
    indices = _prepare_page_indexes(pages, len(reader.pages))
    parts: List[str] = []
    target_pages = indices if indices is not None else range(len(reader.pages))
    for idx in target_pages: #figures out which page numbs are wanted & loops through them calling page.extract_text()
        page = reader.pages[idx]
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _preprocess_scan_image(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    contrasted = ImageOps.autocontrast(gray)
    filtered = contrasted.filter(ImageFilter.MedianFilter(size=3))
    thresholded = filtered.point(lambda p: 255 if p > 165 else 0, mode="1")
    return thresholded.convert("L")


def _extract_text_from_images(images: Iterable[dict], extractor, *, preprocess_scans: bool = False) -> str: #helper that feeds each rendered page through the given OCR
    chunks: List[str]= []
    for item in images:
        img = Image.open(BytesIO(item["bytes"]))
        if preprocess_scans:
            img = _preprocess_scan_image(img)
        chunks.append(str(extractor(img)))
    return "\n".join(chunks)


def extract_text_tesseract(images: Sequence[dict], *, preprocess_scans: bool = False) -> str:#feeder to concatenat recognized text
    image_to_string = _lazy_import_tesseract()
    return _extract_text_from_images(images, image_to_string, preprocess_scans=preprocess_scans)


def extract_text_easyocr(images: Sequence[dict], *, preprocess_scans: bool = False) -> str: #runs EasyOCR
    reader= _lazy_import_easyocr()
    chunks: List[str]= []
    for item in images:
        img= Image.open(BytesIO(item["bytes"]))
        if preprocess_scans:
            img = _preprocess_scan_image(img)
        res= reader.readtext(img)
        chunks.append("\n".join(r[1] for r in res))
    return "\n".join(chunks)


def normalize_text(text: str) -> str: #output messy?  This should clean the OCR output
    """Collapse single newlines to spaces while preserving paragraph breaks."""
    text = text.replace("\r", "")
    placeholder = "__PARA_BREAK__"
    text= text.replace("\n\n", placeholder) #replaces double newlines with a placeholder
    text= re.sub(r"\s*\n\s*", " ", text) #single newlines to spaces 
    text= re.sub(r"\s+", " ", text)
    text= text.replace(placeholder, "\n\n").strip()
    return text

                    #filesystem       Choose method            DPI Control
def extract_pdf_text(source: Path, *, method: Method = "auto", scale: float = 300 / 72, normalize: bool = False, pages: Optional[Sequence[int]] = None, preprocess_scans: bool = False, ) -> str: #extration logic from the slected PDF
    if not source.exists(): #bad path checker
        raise FileNotFoundError(source)

    requires_images = method in {"tesseract", "easyocr", "auto"}  #rasterize pages
    images: Optional[Sequence[dict]] = None #initalize buffer - Ryan 
    if requires_images:
        images = convert_pdf_to_images(source, scale=scale, pages=pages)
    text = ""
    if method in {"pypdf", "auto"}:
        text = extract_text_pypdf(source, pages=pages)
        if method == "pypdf" or text.strip():
            pass  #keep PyPDF result
        else:
            if images is None: #render pages if havent yet
                images = convert_pdf_to_images(source, scale=scale, pages=pages)
            text = extract_text_tesseract(images, preprocess_scans=preprocess_scans)
    elif method == "tesseract":
        if images is None:
            images = convert_pdf_to_images(source, scale=scale, pages=pages)
        text = extract_text_tesseract(images, preprocess_scans=preprocess_scans)
    elif method == "easyocr":
        if images is None:
            images = convert_pdf_to_images(source, scale=scale, pages=pages)
        text = extract_text_easyocr(images, preprocess_scans=preprocess_scans)
    else:  #pragma: no cover - validated by Literal
        raise ValueError(f"Unknown extraction method: {method}") #collapses whitespace, was just getting line by line

    if normalize:
        text = normalize_text(text)

    return text


def _prepare_page_indexes(pages: Optional[Sequence[int]], total_pages: int) -> Optional[List[int]]: #This should clean up the optional list of page numbers a user typed in
    if not pages:
        return None
    cleaned= sorted({p for p in pages if 1 <= p <= total_pages})
    if not cleaned:
        return None
    return [p - 1 for p in cleaned]
