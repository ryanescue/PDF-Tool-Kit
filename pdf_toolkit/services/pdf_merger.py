"""
Helpers for building PDFs from heterogeneous input files.

The core entry point is ``merge_uploaded_files`` which accepts a sequence of
``UploadedFile`` objects (the order they are provided is preserved) and returns
the path to a temporary PDF that contains all uploaded pages merged together.
"""

from __future__ import annotations

import os
import re
import tempfile
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Sequence

from django.core.files.uploadedfile import UploadedFile
from docx import Document as DocxDocument
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEXT_EXTENSIONS = {".txt"}
DOCX_EXTENSIONS = {".docx"}
DOC_EXTENSIONS = {".doc"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | DOCX_EXTENSIONS | DOC_EXTENSIONS | {".pdf"}


@dataclass
class MergeFile:
    """Lightweight helper to track temporary file metadata."""

    path: Path
    original_name: str

    def cleanup(self) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass


def merge_uploaded_files(files: Sequence[UploadedFile], *, output_label: str) -> Path:
    """
    Merge ``files`` in-place order and return a path to the compiled PDF.

    Each ``UploadedFile`` is written to a temporary file first to avoid holding
    everything in-memory. The caller is responsible for removing the returned
    path when appropriate.
    """
    if not files:
        raise ValueError("Select at least one file before merging.")

    writer = PdfWriter()
    saved_files: List[MergeFile] = []
    try:
        for uploaded in files:
            temp_file = _persist_upload(uploaded)
            saved_files.append(temp_file)
            _append_to_writer(writer, temp_file)
    finally:
        # Always delete the temporary source files
        for saved in saved_files:
            saved.cleanup()

    suffix = ".pdf" if not output_label.lower().endswith(".pdf") else ""
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".pdf")
    try:
        writer.write(output_file)
    finally:
        output_file.close()

    return Path(output_file.name)


def _persist_upload(uploaded: UploadedFile) -> MergeFile:
    suffix = Path(uploaded.name).suffix or ""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in uploaded.chunks():
        temp.write(chunk)
    temp.flush()
    temp.close()
    return MergeFile(Path(temp.name), uploaded.name)


def _append_to_writer(writer: PdfWriter, merge_file: MergeFile) -> None:
    suffix = merge_file.path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file '{merge_file.original_name}'. "
            "Allowed types: PDF, DOC, DOCX, TXT, JPG, JPEG, PNG."
        )

    if suffix == ".pdf":
        reader = PdfReader(str(merge_file.path))
    elif suffix in IMAGE_EXTENSIONS:
        reader = PdfReader(_image_to_pdf_stream(merge_file.path))
    elif suffix in DOCX_EXTENSIONS:
        text = _extract_docx_text(merge_file.path)
        reader = PdfReader(_text_to_pdf_stream(text, merge_file.original_name))
    elif suffix in DOC_EXTENSIONS:
        text = _extract_doc_text(merge_file.path)
        reader = PdfReader(_text_to_pdf_stream(text, merge_file.original_name))
    elif suffix in TEXT_EXTENSIONS:
        text = merge_file.path.read_text(encoding="utf-8", errors="ignore")
        reader = PdfReader(_text_to_pdf_stream(text, merge_file.original_name))
    else:  # pragma: no cover - defensive, SUPPORTED_EXTENSIONS gate prevents this
        raise ValueError(f"File type '{suffix}' is not supported.")

    for page in reader.pages:
        writer.add_page(page)


def _image_to_pdf_stream(path: Path) -> BytesIO:
    image = Image.open(path)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    stream = BytesIO()
    image.save(stream, format="PDF", resolution=300.0)
    stream.seek(0)
    return stream


def _text_to_pdf_stream(text: str, label: str) -> BytesIO:
    """
    Render ``text`` into a simple paginated PDF stream.

    This intentionally keeps layout simple (monospace-like) so that any
    combination of inputs ends up with predictable output.
    """
    cleaned = text.strip() or "[Empty document]"
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    pdf.setTitle(label)
    width, height = LETTER
    left_margin = 54
    right_margin = width - 54
    top_margin = height - 72
    y = top_margin
    line_height = 14

    def _draw_line(line: str, cursor_y: int) -> int:
        pdf.drawString(left_margin, cursor_y, line)
        return cursor_y - line_height

    wrap_width = int((right_margin - left_margin) / 7)
    for raw_line in cleaned.splitlines():
        wrapped = textwrap.wrap(raw_line, width=wrap_width) or [""]
        for line in wrapped:
            if y <= 72:
                pdf.showPage()
                pdf.setTitle(label)
                y = top_margin
            y = _draw_line(line, y)

    pdf.save()
    buffer.seek(0)
    return buffer


def _extract_docx_text(path: Path) -> str:
    document = DocxDocument(str(path))
    parts: List[str] = []
    for paragraph in document.paragraphs:
        parts.append(paragraph.text)

    table_text = _docx_tables_to_text(document)
    if table_text:
        parts.append(table_text)

    return "\n".join(parts)


def _docx_tables_to_text(document: DocxDocument) -> str:
    rows: List[str] = []
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cell for cell in cells if cell))
        rows.append("")
    return "\n".join(row for row in rows if row).strip()


def _extract_doc_text(path: Path) -> str:
    """
    Rough text extraction for legacy .doc files.

    Binary Word documents are tricky to parse without external utilities,
    so this method focuses on extracting readable ASCII/UTF encoded strings.
    """
    data = path.read_bytes()
    # Try a couple of decoders before falling back to ascii-ish filtering
    text: str
    for encoding in ("utf-16", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        filtered = (chr(b) if 32 <= b <= 126 else " " for b in data)
        text = "".join(filtered)

    text = text.replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
