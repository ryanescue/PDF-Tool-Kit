from __future__ import annotations

import tempfile #creates tempfiles for holding
from dataclasses import dataclass 
from pathlib import Path
from typing import List, Sequence, Tuple #pagecount and func signitures
from PyPDF2 import PdfReader, PdfWriter


class SplitError(ValueError):#error for when bad numbers are given like if given page 10 but only 8 pages
    pass

@dataclass
class SplitSegment:#info about a generated split segment
    path: Path
    start_page: int #1-indexed
    end_page: int #inclusive
    index: int

                #split 'source' at each page listed in 'split_points'
def split_pdf(source: Path, split_points: Sequence[int]) -> Tuple[List[SplitSegment], int]:
    #'split_points' have to onlt have 1-indexed page numbers that mark the beginning of a new segment for example splitting at page 5 will produce segments 1-4 and 5-end will be the other segment
    reader=PdfReader(str(source)) #opens pdf
    total_pages=len(reader.pages)
    if total_pages == 0: #error if no pages are found
        raise SplitError("The PDF appears to be empty.") 
    sanitized_points= _sanitize_split_points(split_points, total_pages)
    if not sanitized_points:
        raise SplitError("Add at least one split point between page 2 and the last page.")
    #Starts to build boundary indexes that mark where each segment starts/ends
    boundaries=[0] #start with 0
    for point in sanitized_points:
        boundaries.append(point - 1) #for each SP add p-1
    if boundaries[-1] != total_pages:
        boundaries.append(total_pages)

    segments: List[SplitSegment]=[] #prepares output list
    for idx in range(len(boundaries) - 1):
        start= boundaries[idx]
        end= boundaries[idx + 1]
        if start >= end: #just in case skip
            continue
        writer= PdfWriter() #creates new PdfWriter
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        temp_file =tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            writer.write(temp_file) #write to temp pdf
        finally: #learned how to use finally :)
            temp_file.close()
        segments.append( SplitSegment(path=Path(temp_file.name), start_page=start + 1, end_page=end, index=idx + 1,))
        #records metadata about seg + Temp path + start/end page numbers
    return segments, total_pages


def _sanitize_split_points(points: Sequence[int], total_pages: int) -> List[int]: #cleans user input split marker for Split_pdf
    valid_points: List[int] = []
    seen = set()
    for point in points: #checks all points user provides
        if not isinstance(point, int): #skips non int
            continue
        if point <= 1 or point > total_pages: #cant split at 1 or last page
            continue
        if point in seen:
            continue
        seen.add(point)
        valid_points.append(point)
    valid_points.sort()
    return valid_points
