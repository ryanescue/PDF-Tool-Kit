#!/usr/bin/env python3
"""
Extract structured insurance estimate tables from a PDF into a clean XLSX file.

Primary strategy: use text-based table extractors (pdfplumber/camelot/tabula).
Fallback strategy: OCR with pytesseract + simple spatial alignment.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from pdf_toolkit.services.pdf_extractor import convert_pdf_to_images, enhance_ocr_image


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
    "condition",
    "cost",
    "remaining",
}

NUMERIC_RE = re.compile(r"^\(?-?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$")
PLAIN_NUMERIC_RE = re.compile(r"^\(?-?\d+(?:\.\d+)?\)?$")


@dataclass
class ExtractedTable:
    header: List[str]
    rows: List[List[str]]


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def _clean_cell(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _looks_numeric(value: str) -> bool:
    if not value:
        return False
    text = value.strip()
    if "/" in text and re.search(r"[A-Za-z]", text):
        return False
    return bool(NUMERIC_RE.match(text) or PLAIN_NUMERIC_RE.match(text))


def _parse_number(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace("$", "").replace(",", "").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _is_header_row(row: Sequence[str]) -> bool:
    if not row:
        return False
    keyword_hits = 0
    alpha_cells = 0
    numeric_cells = 0
    for cell in row:
        value = cell.lower().strip()
        if not value:
            continue
        if _looks_numeric(value):
            numeric_cells += 1
        if re.search(r"[A-Za-z]", value):
            alpha_cells += 1
        for keyword in HEADER_KEYWORDS:
            if keyword in value:
                keyword_hits += 1
                break
    if keyword_hits >= 2:
        return True
    if alpha_cells >= max(2, int(len(row) * 0.6)) and numeric_cells == 0:
        return True
    return False


def _merge_header_rows(rows: Sequence[Sequence[str]]) -> List[str]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    merged: List[str] = []
    for col in range(width):
        parts = []
        for row in rows:
            cell = row[col] if col < len(row) else ""
            cell = _clean_cell(cell)
            if cell:
                parts.append(cell)
        merged.append(" ".join(parts).strip())
    return merged


def _normalize_rows(rows: Sequence[Sequence[str]], width: int) -> List[List[str]]:
    normalized: List[List[str]] = []
    for row in rows:
        cells = [_clean_cell(cell) for cell in row]
        if len(cells) < width:
            cells += [""] * (width - len(cells))
        elif len(cells) > width:
            cells = cells[:width]
        if any(cells):
            normalized.append(cells)
    return normalized


def _rows_match(a: Sequence[str], b: Sequence[str]) -> bool:
    if len(a) != len(b):
        return False
    return [c.strip().lower() for c in a] == [c.strip().lower() for c in b]


def _detect_header(rows: Sequence[Sequence[str]]) -> Tuple[List[str], int]:
    header_indices: List[int] = []
    for idx, row in enumerate(rows[:3]):
        if _is_header_row(row):
            header_indices.append(idx)
        elif header_indices:
            break
    if not header_indices and rows:
        header_indices = [0]
    header_rows = [rows[i] for i in header_indices]
    header = _merge_header_rows(header_rows)
    return header, (header_indices[-1] + 1 if header_indices else 0)


def _normalize_table(rows: Sequence[Sequence[str]]) -> Optional[ExtractedTable]:
    cleaned_rows = [[_clean_cell(cell) for cell in row] for row in rows]
    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    if not cleaned_rows:
        return None
    header, start_idx = _detect_header(cleaned_rows)
    if not header:
        return None
    data_rows = _normalize_rows(cleaned_rows[start_idx:], len(header))
    data_rows = [row for row in data_rows if not _rows_match(header, row)]
    if not data_rows:
        return None
    return ExtractedTable(header=header, rows=data_rows)


def _combine_tables(tables: Iterable[ExtractedTable]) -> pd.DataFrame:
    combined_rows: List[List[str]] = []
    header: Optional[List[str]] = None
    for table in tables:
        if header is None:
            header = table.header
        if header != table.header:
            # Align to initial header width if the same columns repeat with minor diffs.
            width = len(header)
            adjusted = _normalize_rows(table.rows, width)
            combined_rows.extend(adjusted)
        else:
            combined_rows.extend(table.rows)
    if header is None:
        return pd.DataFrame()
    return pd.DataFrame(combined_rows, columns=header)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        series = df[column].fillna("").astype(str).str.strip()
        non_empty = series[series != ""]
        if non_empty.empty:
            continue
        numeric_ratio = non_empty.apply(_looks_numeric).mean()
        if numeric_ratio >= 0.7:
            df[column] = series.apply(_parse_number)
    return df


def _extract_with_pdfplumber(pdf_path: Path, pages: Optional[str]) -> List[ExtractedTable]:
    pdfplumber = _try_import("pdfplumber")
    if pdfplumber is None:
        return []
    results: List[ExtractedTable] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_numbers = _parse_page_selection(pages, len(pdf.pages))
        for page_idx in page_numbers:
            page = pdf.pages[page_idx]
            tables = []
            for settings in (
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
            ):
                try:
                    tables = page.extract_tables(table_settings=settings) or []
                    if tables:
                        break
                except Exception:
                    continue
            for table in tables:
                normalized = _normalize_table(table)
                if normalized:
                    results.append(normalized)
    return results


def _extract_with_camelot(pdf_path: Path, pages: Optional[str]) -> List[ExtractedTable]:
    camelot = _try_import("camelot")
    if camelot is None:
        return []
    results: List[ExtractedTable] = []
    page_spec = pages or "all"
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=page_spec, flavor=flavor)
        except Exception:
            continue
        for table in tables:
            normalized = _normalize_table(table.df.values.tolist())
            if normalized:
                results.append(normalized)
        if results:
            break
    return results


def _extract_with_tabula(pdf_path: Path, pages: Optional[str]) -> List[ExtractedTable]:
    tabula = _try_import("tabula")
    if tabula is None:
        return []
    results: List[ExtractedTable] = []
    try:
        tables = tabula.read_pdf(str(pdf_path), pages=pages or "all", multiple_tables=True)
    except Exception:
        return []
    for table in tables:
        normalized = _normalize_table(table.values.tolist())
        if normalized:
            results.append(normalized)
    return results


def _parse_page_selection(pages: Optional[str], total_pages: int) -> List[int]:
    if not pages:
        return list(range(total_pages))
    selected: List[int] = []
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            for idx in range(start, end + 1):
                if 1 <= idx <= total_pages:
                    selected.append(idx - 1)
        else:
            try:
                idx = int(part)
            except ValueError:
                continue
            if 1 <= idx <= total_pages:
                selected.append(idx - 1)
    return sorted(set(selected))


def _ocr_rows_from_pdf(pdf_path: Path, pages: Optional[str]) -> List[ExtractedTable]:
    from pytesseract import Output, image_to_data

    results: List[ExtractedTable] = []
    images = convert_pdf_to_images(pdf_path, pages=_pages_for_images(pdf_path, pages))
    for payload in images:
        from PIL import Image
        from io import BytesIO

        image = Image.open(BytesIO(payload["bytes"]))
        image = enhance_ocr_image(image, preprocess_scans=True)
        data_frame = image_to_data(image, output_type=Output.DATAFRAME)
        if data_frame is None or data_frame.empty:
            continue
        frame = (
            data_frame.dropna(subset=["text"])
            .assign(text=lambda df: df["text"].str.strip())
        )
        frame = frame[(frame["text"] != "") & (frame["conf"] >= 45)]
        if frame.empty:
            continue
        frame = frame.assign(
            center_x=frame["left"] + frame["width"] / 2.0,
            center_y=frame["top"] + frame["height"] / 2.0,
        )
        row_threshold = max(8.0, float(frame["height"].median()) * 0.75)
        buckets: List[dict] = []
        for word in frame.sort_values(["top", "left"]).to_dict("records"):
            placed = False
            for bucket in buckets:
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
                buckets.append({"center_y": word["center_y"], "count": 1, "words": [word]})
        if not buckets:
            continue
        column_centers = _estimate_columns(buckets)
        if not column_centers:
            continue
        rendered_rows: List[List[str]] = []
        for bucket in sorted(buckets, key=lambda item: item["center_y"]):
            words = sorted(bucket["words"], key=lambda item: (item["left"], item["center_y"]))
            cells = ["" for _ in column_centers]
            for word in words:
                idx = _closest_column(word["center_x"], column_centers)
                if idx is None:
                    continue
                text = word["text"].strip()
                if not text:
                    continue
                cells[idx] = f"{cells[idx]} {text}".strip() if cells[idx] else text
            if any(cells):
                rendered_rows.append(cells)
        normalized = _normalize_table(rendered_rows)
        if normalized:
            results.append(normalized)
    return results


def _estimate_columns(buckets: List[dict]) -> List[float]:
    centers: List[float] = []
    widths: List[float] = []
    for bucket in buckets:
        for word in bucket["words"]:
            centers.append(word["center_x"])
            widths.append(word["width"])
    if not centers:
        return []
    centers.sort()
    median_width = float(pd.Series(widths).median()) if widths else 40.0
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


def _pages_for_images(pdf_path: Path, pages: Optional[str]) -> Optional[Sequence[int]]:
    if not pages:
        return None
    from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    return [p + 1 for p in _parse_page_selection(pages, len(reader.pages))]


def _write_excel(df: pd.DataFrame, output_path: Path) -> None:
    try:
        import openpyxl  # noqa: F401
        engine = "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except Exception as exc:
            raise RuntimeError(
                "XLSX output requires openpyxl or xlsxwriter. "
                "Install one of them to continue."
            ) from exc
    df.to_excel(output_path, index=False, engine=engine)


def extract_tables_to_excel(pdf_path: Path, output_path: Path, pages: Optional[str]) -> None:
    tables: List[ExtractedTable] = []
    tables.extend(_extract_with_pdfplumber(pdf_path, pages))
    if not tables:
        tables.extend(_extract_with_camelot(pdf_path, pages))
    if not tables:
        tables.extend(_extract_with_tabula(pdf_path, pages))
    if not tables:
        tables.extend(_ocr_rows_from_pdf(pdf_path, pages))
    if not tables:
        raise RuntimeError("No tables detected in PDF.")
    df = _combine_tables(tables)
    if df.empty:
        raise RuntimeError("No usable rows detected after normalization.")
    df = _coerce_numeric_columns(df)
    _write_excel(df, output_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract insurance estimate tables from a PDF into XLSX."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF file.")
    parser.add_argument("output_path", type=Path, help="Path to output XLSX file.")
    parser.add_argument(
        "--pages",
        help="Optional page selection like '1-3,5'. Defaults to all pages.",
    )
    args = parser.parse_args(argv)

    if not args.pdf_path.exists():
        print(f"Input PDF not found: {args.pdf_path}", file=sys.stderr)
        return 2
    try:
        extract_tables_to_excel(args.pdf_path, args.output_path, args.pages)
    except Exception as exc:
        print(f"Failed to extract tables: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote Excel output to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
