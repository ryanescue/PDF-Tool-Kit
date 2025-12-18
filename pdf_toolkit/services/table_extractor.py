"""Utilities for converting detected tables into CSV output.

This module piggybacks on the existing ``convert_pdf_to_images`` helper so we
can reuse the high-quality rasterization pipeline already used for OCR.  Given
one or more pages from a PDF, we run Tesseract in data mode to retrieve the
bounding boxes for every recognized word.  The bounding boxes are grouped into
rows/columns heuristically to build a lightweight representation of the table
that can be exported to CSV or rendered as an HTML preview.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional, Sequence

import re

import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from .pdf_extractor import convert_pdf_to_images

HEADER_KEYWORDS = {
    "description",
    "qty",
    "quantity",
    "estimate",
    "tax",
    "replacement",
    "depreciation",
    "value",
    "paid",
    "pending",
    "payment",
    "age",
    "cond",
    "cost",
}

@dataclass
class CsvExtractionResult:
    """Container for the structured table output from a PDF."""

    csv_text: str
    headers: List[str]
    preview_rows: List[dict]
    total_rows: int
    page_count: int
    column_count: int


def extract_pdf_tables_as_csv(
    pdf_path: Path,
    *,
    pages: Optional[Sequence[int]] = None,
    scale: float = 300 / 72,
    min_confidence: int = 45,
    max_preview_rows: int = 12,
    preprocess_scans: bool = False,
) -> Optional[CsvExtractionResult]:
    """Return a structured CSV for the provided PDF if a table is detected.

    The function returns ``None`` when no rows could be detected so callers can
    fall back to plain text extraction without surfacing errors to the user.
    """

    images = convert_pdf_to_images(pdf_path, scale=scale, pages=pages)
    if not images:
        return None

    output = StringIO()
    writer = csv.writer(output)
    header: List[str] = []
    preview_rows: List[dict] = []
    total_rows = 0
    pages_with_rows = 0

    header_roles = {}

    for page_number, payload in enumerate(images, start=1):
        rows = _table_rows_from_image(
            payload["bytes"],
            min_confidence=min_confidence,
            preprocess_scans=preprocess_scans,
        )
        if not rows:
            continue
        pages_with_rows += 1

        row_index = 0
        if not header:
            header_index = _detect_header_row(rows)
            if header_index is None:
                continue
            header = rows[header_index]
            header_roles = _classify_headers(header)
            writer.writerow(["Page"] + header)
            row_index = header_index + 1

        for row in rows[row_index:]:
            if header and _rows_match(header, row):
                # Many scanned PDFs repeat the header on every page. Skip dupes.
                continue
            total_rows += 1
            normalized_row = _normalize_row(row, len(header))
            normalized_row = _align_special_columns(
                normalized_row, header_roles, row
            )
            writer.writerow([page_number] + normalized_row)
            if len(preview_rows) < max_preview_rows:
                preview_rows.append({"page": page_number, "cells": normalized_row})

    if not header or total_rows == 0:
        return None

    return CsvExtractionResult(
        csv_text=output.getvalue(),
        headers=header,
        preview_rows=preview_rows,
        total_rows=total_rows,
        page_count=pages_with_rows,
        column_count=len(header),
    )


def _table_rows_from_image(
    image_bytes: bytes, *, min_confidence: int, preprocess_scans: bool
) -> List[List[str]]:
    from pytesseract import Output, image_to_data

    image = Image.open(BytesIO(image_bytes))
    if preprocess_scans:
        image = _preprocess_scan_image(image)
    image = image.convert("L")
    data_frame = image_to_data(image, output_type=Output.DATAFRAME)
    if data_frame is None or not isinstance(data_frame, pd.DataFrame):
        return []

    # Clean up the OCR output: drop empty tokens and low-confidence hits.
    frame = (
        data_frame.dropna(subset=["text"])
        .assign(text=lambda df: df["text"].str.strip())
    )
    frame = frame[(frame["text"] != "") & (frame["conf"] >= min_confidence)]
    if frame.empty:
        return []

    frame = frame.assign(
        center_x=frame["left"] + frame["width"] / 2.0,
        center_y=frame["top"] + frame["height"] / 2.0,
    )

    row_threshold = max(8.0, float(frame["height"].median()) * 0.75)
    rows: List[dict] = []
    for word in frame.sort_values(["top", "left"]).to_dict("records"):
        placed = False
        for bucket in rows:
            if abs(word["center_y"] - bucket["center_y"]) <= row_threshold:
                bucket["words"].append(word)
                count = bucket["count"] + 1
                bucket["center_y"] = (
                    bucket["center_y"] * bucket["count"] + word["center_y"]
                ) / count
                bucket["count"] = count
                placed = True
                break
        if not placed:
            rows.append({"center_y": word["center_y"], "count": 1, "words": [word]})

    if not rows:
        return []

    column_centers = _estimate_columns(rows)
    if not column_centers:
        column_centers = [rows[0]["words"][0]["center_x"]]

    rendered_rows: List[List[str]] = []
    for row in sorted(rows, key=lambda bucket: bucket["center_y"]):
        words = sorted(row["words"], key=lambda item: (item["left"], item["center_y"]))
        cells = ["" for _ in column_centers]
        for word in words:
            idx = _closest_column(word["center_x"], column_centers)
            if idx is None:
                continue
            text = word["text"].strip()
            if not text:
                continue
            cells[idx] = f"{cells[idx]} {text}".strip() if cells[idx] else text
        if any(cell for cell in cells):
            rendered_rows.append([cell.strip() for cell in cells])

    return rendered_rows


def _estimate_columns(rows: List[dict]) -> List[float]:
    centers: List[float] = []
    widths: List[float] = []
    for row in rows:
        for word in row["words"]:
            centers.append(word["center_x"])
            widths.append(word["width"])
    if not centers:
        return []
    centers.sort()
    median_width = statistics.median(widths) if widths else 40.0
    threshold = max(18.0, median_width * 1.4)
    columns: List[float] = []
    for value in centers:
        if not columns or abs(value - columns[-1]) > threshold:
            columns.append(value)
        else:
            columns[-1] = (columns[-1] + value) / 2.0
    return columns


def _closest_column(value: float, columns: Sequence[float]) -> Optional[int]:
    if not columns:
        return None
    nearest = None
    min_distance = float("inf")
    for idx, target in enumerate(columns):
        distance = abs(value - target)
        if distance < min_distance:
            nearest = idx
            min_distance = distance
    return nearest


def _rows_match(a: Sequence[str], b: Sequence[str]) -> bool:
    norm_a = [segment.strip().lower() for segment in a]
    norm_b = [segment.strip().lower() for segment in b]
    return norm_a == norm_b


def _classify_headers(header: Sequence[str]) -> dict:
    roles: dict[str, int] = {}
    for idx, cell in enumerate(header):
        value = cell.lower()
        if "description" in value and "description" not in roles:
            roles["description"] = idx
        if ("qty" in value or "quantity" in value) and "qty" not in roles:
            roles["qty"] = idx
    return roles


def _align_special_columns(row: List[str], roles: dict, original_row: Sequence[str]) -> List[str]:
    desc_idx = roles.get("description")
    qty_idx = roles.get("qty")
    if desc_idx is not None and qty_idx is not None and qty_idx < len(row):
        cell = original_row[qty_idx] if qty_idx < len(original_row) else row[qty_idx]
        prefix, qty_value = _split_qty_cell(cell)
        if prefix and desc_idx < len(row):
            row[desc_idx] = f"{row[desc_idx]} {prefix}".strip()
        if qty_value:
            row[qty_idx] = qty_value
    return row


def _preprocess_scan_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    contrasted = ImageOps.autocontrast(gray)
    filtered = contrasted.filter(ImageFilter.MedianFilter(size=3))
    thresholded = filtered.point(lambda p: 255 if p > 165 else 0, mode="1")
    return thresholded.convert("L")


def _detect_header_row(rows: Sequence[Sequence[str]]) -> Optional[int]:
    best_idx = None
    best_score = 0
    for idx, row in enumerate(rows):
        score = 0
        for cell in row:
            value = cell.lower().strip()
            if not value:
                continue
            for keyword in HEADER_KEYWORDS:
                if keyword in value:
                    score += 1
                    break
        if score > best_score or (score == best_score and best_idx is not None and len(row) > len(rows[best_idx])):
            best_idx = idx
            best_score = score
    if best_idx is None:
        return None
    if best_score >= 2 or len(rows[best_idx]) >= 6:
        return best_idx
    return None


def _normalize_row(row: Sequence[str], width: int) -> List[str]:
    cells = list(row)
    if len(cells) < width:
        cells = cells + [""] * (width - len(cells))
    elif len(cells) > width:
        cells = cells[:width]
    return [cell.strip() for cell in cells]


def _split_qty_cell(value: str) -> tuple[str, str]:
    if not value:
        return "", ""
    original = value.strip()
    if not original:
        return "", ""
    normalized = _prepare_qty_string(original)
    match = re.search(r"(\d+(?:\.\d+)?)\s*([A-Z]{1,4})?$", normalized)
    if not match:
        digits = re.findall(r"\d", normalized)
        if not digits:
            return "", original.strip()
        number = "".join(digits)
        qty_value = _format_qty_number(number)
        return "", qty_value
    start, end = match.span()
    desc_extra = original[:start].strip()
    number = match.group(1)
    unit = (match.group(2) or "").strip()
    number = _format_qty_number(number)
    if unit and not unit.startswith(" "):
        unit = f" {unit}"
    qty_value = f"{number}{unit}".strip()
    return desc_extra, qty_value


def _prepare_qty_string(value: str) -> str:
    translation = str.maketrans({"O": "0", "I": "1", "L": "1"})
    sanitized_chars = []
    for ch in value:
        if ch.isalnum() or ch in {".", " ", "/"}:
            sanitized_chars.append(ch.upper())
        else:
            sanitized_chars.append(" ")
    sanitized = "".join(sanitized_chars).translate(translation)
    return sanitized


def _format_qty_number(raw: str) -> str:
    digits = raw.replace(" ", "")
    if not digits:
        return ""
    if "." in digits:
        return digits
    if len(digits) == 1:
        return f"{digits}.00"
    if len(digits) == 2:
        return f"{digits}.00"
    return f"{digits[:-2]}.{digits[-2:]}"
