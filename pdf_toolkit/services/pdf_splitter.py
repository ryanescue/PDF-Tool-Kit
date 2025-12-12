"""Utilities for splitting PDFs into smaller chunks."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from PyPDF2 import PdfReader, PdfWriter


class SplitError(ValueError):
    """Raised when invalid split points are provided."""


@dataclass
class SplitSegment:
    """Information about a generated split segment."""

    path: Path
    start_page: int  # 1-indexed
    end_page: int  # inclusive
    index: int


def split_pdf(source: Path, split_points: Sequence[int]) -> Tuple[List[SplitSegment], int]:
    """
    Split ``source`` at each page listed in ``split_points``.

    ``split_points`` should contain 1-indexed page numbers that mark the
    beginning of a new segment (e.g., splitting at page 5 will produce segments
    1-4 and 5-end).
    """
    reader = PdfReader(str(source))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise SplitError("The PDF appears to be empty.")

    sanitized_points = _sanitize_split_points(split_points, total_pages)
    if not sanitized_points:
        raise SplitError("Add at least one split point between page 2 and the last page.")

    boundaries = [0]
    for point in sanitized_points:
        boundaries.append(point - 1)
    if boundaries[-1] != total_pages:
        boundaries.append(total_pages)

    segments: List[SplitSegment] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if start >= end:
            continue
        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            writer.write(temp_file)
        finally:
            temp_file.close()
        segments.append(
            SplitSegment(
                path=Path(temp_file.name),
                start_page=start + 1,
                end_page=end,
                index=idx + 1,
            )
        )

    return segments, total_pages


def _sanitize_split_points(points: Sequence[int], total_pages: int) -> List[int]:
    valid_points: List[int] = []
    seen = set()
    for point in points:
        if not isinstance(point, int):
            continue
        if point <= 1 or point > total_pages:
            continue
        if point in seen:
            continue
        seen.add(point)
        valid_points.append(point)
    valid_points.sort()
    return valid_points
