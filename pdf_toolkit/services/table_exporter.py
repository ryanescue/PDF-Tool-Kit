"""Extract structured tables from PDFs and export to Excel."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .pdf_extractor import convert_pdf_to_images, enhance_ocr_image

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
INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


@dataclass
class ExcelExtractionResult:
    excel_bytes: bytes
    headers: List[str]
    total_rows: int
    page_count: int
    column_count: int


@dataclass
class ExtractedTable:
    header: List[str]
    rows: List[List[str]]
    page: Optional[int] = None


def extract_pdf_tables_as_xlsx(
    pdf_path: Path,
    *,
    pages: Optional[Sequence[int]] = None,
    scale: float = 300 / 72,
    preprocess_scans: bool = False,
    deskew: bool = False,
    min_confidence: int = 45,
) -> Optional[ExcelExtractionResult]:
    """Return Excel bytes with detected tables or None when no table is found."""

    tables: List[ExtractedTable] = []
    tables.extend(_extract_with_pdfplumber(pdf_path, pages))
    if not tables:
        tables.extend(_extract_with_camelot(pdf_path, pages))
    if not tables:
        tables.extend(_extract_with_tabula(pdf_path, pages))
    if not tables:
        tables.extend(
            _extract_with_ocr(
                pdf_path,
                pages=pages,
                scale=scale,
                preprocess_scans=preprocess_scans,
                deskew=deskew,
                min_confidence=min_confidence,
            )
        )

    if not tables:
        return None

    df, page_count = _combine_tables(tables)
    if df.empty:
        return None
    df = _coerce_numeric_columns(df)
    info_rows = _extract_info_rows(pdf_path, pages)
    df = _merge_info_with_table(info_rows, df)
    df = _sanitize_dataframe_for_excel(df)

    output = BytesIO()
    engine = _select_excel_engine()
    df.to_excel(output, index=False, engine=engine)
    output.seek(0)
    return ExcelExtractionResult(
        excel_bytes=output.getvalue(),
        headers=list(df.columns),
        total_rows=len(df.index),
        page_count=page_count,
        column_count=len(df.columns),
    )


def _select_excel_engine() -> str:
    try:
        import openpyxl  # noqa: F401

        return "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa: F401

            return "xlsxwriter"
        except Exception as exc:
            raise RuntimeError(
                "XLSX export requires openpyxl or xlsxwriter. "
                "Install one of them to continue."
            ) from exc


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


def _normalize_table(rows: Sequence[Sequence[str]], *, page: Optional[int] = None) -> Optional[ExtractedTable]:
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
    header, data_rows = _collapse_description_columns(header, data_rows)
    return ExtractedTable(header=header, rows=data_rows, page=page)


def _combine_tables(tables: Iterable[ExtractedTable]) -> Tuple[pd.DataFrame, int]:
    combined_rows: List[List[str]] = []
    header: Optional[List[str]] = None
    pages = {table.page for table in tables if table.page is not None}
    for table in tables:
        if header is None:
            header = table.header
        if header != table.header:
            width = len(header)
            adjusted = _normalize_rows(table.rows, width)
            combined_rows.extend(adjusted)
        else:
            combined_rows.extend(table.rows)
    if header is None:
        return pd.DataFrame(), len(pages)
    header = _make_unique_headers(header)
    return pd.DataFrame(combined_rows, columns=header), len(pages)


def _collapse_description_columns(
    header: List[str],
    rows: List[List[str]],
) -> Tuple[List[str], List[List[str]]]:
    qty_idx = None
    for idx, cell in enumerate(header):
        value = cell.lower().strip()
        if "qty" in value or "quantity" in value:
            qty_idx = idx
            break
    if qty_idx is None or qty_idx <= 1:
        return header, rows

    desc_header = header[0].strip() or "Description"
    merged_header = [desc_header] + header[qty_idx:]

    merged_rows: List[List[str]] = []
    for row in rows:
        desc_parts = [part for part in row[:qty_idx] if part]
        desc_value = " ".join(desc_parts).strip()
        merged_rows.append([desc_value] + row[qty_idx:])
    return merged_header, merged_rows


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for idx, column in enumerate(df.columns):
        series = df.iloc[:, idx].fillna("").astype(str).str.strip()
        non_empty = series[series != ""]
        if non_empty.empty:
            continue
        numeric_ratio = non_empty.apply(_looks_numeric).mean()
        if numeric_ratio >= 0.7:
            df.iloc[:, idx] = series.apply(_parse_number)
    return df


def _extract_with_pdfplumber(pdf_path: Path, pages: Optional[Sequence[int]]) -> List[ExtractedTable]:
    pdfplumber = _try_import("pdfplumber")
    if pdfplumber is None:
        return []
    results: List[ExtractedTable] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_indexes = _pages_to_zero_based(pages, len(pdf.pages))
        for page_number in page_indexes:
            page = pdf.pages[page_number]
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
                normalized = _normalize_table(table, page=page_number + 1)
                if normalized:
                    results.append(normalized)
    return results


def _extract_with_camelot(pdf_path: Path, pages: Optional[Sequence[int]]) -> List[ExtractedTable]:
    camelot = _try_import("camelot")
    if camelot is None:
        return []
    results: List[ExtractedTable] = []
    page_spec = _pages_to_spec(pages)
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


def _extract_with_tabula(pdf_path: Path, pages: Optional[Sequence[int]]) -> List[ExtractedTable]:
    tabula = _try_import("tabula")
    if tabula is None:
        return []
    results: List[ExtractedTable] = []
    try:
        tables = tabula.read_pdf(
            str(pdf_path),
            pages=_pages_to_spec(pages),
            multiple_tables=True,
        )
    except Exception:
        return []
    for table in tables:
        normalized = _normalize_table(table.values.tolist())
        if normalized:
            results.append(normalized)
    return results


def _extract_with_ocr(
    pdf_path: Path,
    *,
    pages: Optional[Sequence[int]],
    scale: float,
    preprocess_scans: bool,
    deskew: bool,
    min_confidence: int,
) -> List[ExtractedTable]:
    from pytesseract import Output, image_to_data
    from PIL import Image

    results: List[ExtractedTable] = []
    images = convert_pdf_to_images(pdf_path, scale=scale, pages=pages)
    for payload in images:
        image = Image.open(BytesIO(payload["bytes"]))
        image = enhance_ocr_image(
            image, preprocess_scans=preprocess_scans, deskew=deskew
        ).convert("L")
        data_frame = image_to_data(image, output_type=Output.DATAFRAME)
        if data_frame is None or data_frame.empty:
            continue
        frame = (
            data_frame.dropna(subset=["text"])
            .assign(text=lambda df: df["text"].str.strip())
        )
        frame = frame[(frame["text"] != "") & (frame["conf"] >= min_confidence)]
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
                buckets.append(
                    {"center_y": word["center_y"], "count": 1, "words": [word]}
                )
        if not buckets:
            continue
        column_centers = _estimate_columns(buckets)
        if not column_centers:
            continue
        rendered_rows: List[List[str]] = []
        for bucket in sorted(buckets, key=lambda item: item["center_y"]):
            words = sorted(
                bucket["words"], key=lambda item: (item["left"], item["center_y"])
            )
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
        page_number = payload.get("page")
        normalized = _normalize_table(
            rendered_rows,
            page=(page_number + 1) if isinstance(page_number, int) else None,
        )
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


def _pages_to_zero_based(pages: Optional[Sequence[int]], total_pages: int) -> List[int]:
    if not pages:
        return list(range(total_pages))
    cleaned = sorted({p for p in pages if 1 <= p <= total_pages})
    return [p - 1 for p in cleaned]


def _pages_to_spec(pages: Optional[Sequence[int]]) -> str:
    if not pages:
        return "all"
    return ",".join(str(p) for p in sorted(set(pages)))


def _make_unique_headers(headers: Sequence[str]) -> List[str]:
    seen: dict[str, int] = {}
    unique: List[str] = []
    for header in headers:
        base = _sanitize_excel_text(header).strip() or "Column"
        count = seen.get(base, 0)
        if count:
            unique.append(f"{base} ({count + 1})")
        else:
            unique.append(base)
        seen[base] = count + 1
    return unique


def _sanitize_excel_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        from openpyxl.utils.cell import ILLEGAL_CHARACTERS_RE
    except Exception:
        return INVALID_XML_RE.sub("", text)
    return ILLEGAL_CHARACTERS_RE.sub("", text)


def _sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    for idx, _column in enumerate(df.columns):
        series = df.iloc[:, idx]
        if series.dtype != object:
            continue
        df.iloc[:, idx] = series.apply(_sanitize_excel_text)
    return df


def _extract_info_rows(pdf_path: Path, pages: Optional[Sequence[int]]) -> List[List[str]]:
    lines: List[str] = []
    pdfplumber = _try_import("pdfplumber")
    if pdfplumber is not None:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_indexes = _pages_to_zero_based(pages, len(pdf.pages))
            for page_number in page_indexes:
                page = pdf.pages[page_number]
                text = page.extract_text() or ""
                lines.extend(text.splitlines())
    else:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            return []
        reader = PdfReader(str(pdf_path))
        page_indexes = _pages_to_zero_based(pages, len(reader.pages))
        for page_number in page_indexes:
            text = reader.pages[page_number].extract_text() or ""
            lines.extend(text.splitlines())

    cleaned: List[str] = []
    seen = set()
    for line in lines:
        normalized = _clean_cell(line)
        if not normalized:
            continue
        if _looks_like_table_row(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return [_split_info_line(line) for line in cleaned]


def _looks_like_table_row(line: str) -> bool:
    normalized = line.strip()
    if re.match(r"^\d+\.\s+", normalized):
        return True
    tokens = normalized.split()
    if not tokens:
        return False
    numeric_tokens = sum(1 for token in tokens if _looks_numeric(token))
    return numeric_tokens >= max(2, int(len(tokens) * 0.4))


def _split_info_line(line: str, max_columns: int = 4, chunk_size: int = 55) -> List[str]:
    chunks = [line[i : i + chunk_size] for i in range(0, len(line), chunk_size)]
    if len(chunks) > max_columns:
        chunks = chunks[: max_columns - 1] + [" ".join(chunks[max_columns - 1 :])]
    while len(chunks) < max_columns:
        chunks.append("")
    return chunks


def _merge_info_with_table(info_rows: List[List[str]], table_df: pd.DataFrame) -> pd.DataFrame:
    table_headers = list(table_df.columns)
    info_headers = ["Info A", "Info B", "Info C", "Info D"]
    combined_rows: List[List[object]] = []

    for row in info_rows:
        combined_rows.append(row + [""] * len(table_headers))
    for _, row in table_df.iterrows():
        combined_rows.append([""] * len(info_headers) + row.tolist())

    return pd.DataFrame(combined_rows, columns=info_headers + table_headers)
