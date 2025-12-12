"""Views for the PDF toolkit app."""

import datetime  # timestamp formatting for merge history
import os  # stdlib helper to delete tmp files
import re  # filename normalization
import tempfile  # stdlib helper for creating temp files
import uuid  # merge history identifiers
from pathlib import Path  # stdlib path helper for temp files

import requests  # hTTP client used for diagnostics in server_info
from django.conf import settings  # django settings object for diagnostics
from django.http import FileResponse, Http404, HttpResponse  # HTTP responses
from django.shortcuts import render  # template rendering helper
from django.views.decorators.http import require_http_methods  # restrict HTTP verbs

from .services.pdf_extractor import Method, extract_pdf_text  # shared PDF extraction logic
from .services.pdf_merger import merge_uploaded_files  # PDF merge helpers
from .services.pdf_splitter import SplitError, split_pdf  # PDF split helpers

MERGE_HISTORY_SESSION_KEY = "pdf_merge_history"
MERGE_HISTORY_LIMIT = 5
SPLIT_HISTORY_SESSION_KEY = "pdf_split_history"
SPLIT_HISTORY_LIMIT = 5


def home(request):
    return render(request, "home.html")


def server_info(request):
    server_geodata = requests.get("https://ipwhois.app/json/").json()
    settings_dump = settings.__dict__
    return HttpResponse(f"{server_geodata}{settings_dump}")


@require_http_methods(["GET", "POST"])
def extract_view(request):
    """
    Upload view that accepts PDFs and returns the extracted text.

    Heavy OCR work happens in the shared service to keep the view lean.
    """

    method: Method = request.POST.get("method", "auto")  # type: ignore[assignment]
    normalize_requested = bool(request.POST.get("normalize"))

    context = {
        "selected_method": method,
        "normalize": normalize_requested,
        "text": "",
        "error": "",
    }

    if request.method == "POST":
        uploaded_file = request.FILES.get("pdf")
        if not uploaded_file:
            context["error"] = "Please upload a PDF before submitting"
        else:
            suffix = Path(uploaded_file.name).suffix or ".pdf"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_file.flush()
                temp_path = Path(temp_file.name)
                extracted_text = extract_pdf_text( temp_path, method=method, normalize=normalize_requested,)
                context["text"] = extracted_text
            except Exception as exc:  
                context["error"] = f"Failed to extract text: {exc}"
            finally:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    return render(request, "pdf_toolkit/extract.html", context)


def merge_create_view(request):
    """Upload + ordering workflow for merging heterogeneous files into PDF."""

    history = request.session.get(MERGE_HISTORY_SESSION_KEY, [])
    allowed_attr = ".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
    context = {
        "history": history,
        "success": "",
        "error": "",
        "output_name": request.POST.get("output_name", "Merged-Pack.pdf") if request.method == "POST" else "Merged-Pack.pdf",
        "allowed_types": allowed_attr,
        "allowed_types_display": allowed_attr.replace(",", ", "),
    }

    if request.method == "POST":
        files = request.FILES.getlist("files")
        desired_name = request.POST.get("output_name", "").strip()
        safe_name = _normalize_output_name(desired_name)
        context["output_name"] = safe_name

        if not files:
            context["error"] = "Add at least one document or image before merging."
        else:
            try:
                merged_path = merge_uploaded_files(files, output_label=safe_name)
                updated_history = _record_merge_history(
                    request,
                    output_name=safe_name,
                    merged_path=merged_path,
                    total=len(files),
                )
                context["history"] = updated_history
                context["success"] = f"Merged {len(files)} {'file' if len(files) == 1 else 'files'} into {safe_name}"
            except ValueError as exc:
                context["error"] = str(exc)

    return render(request, "pdf_toolkit/merge.html", context)


def download_merge_result(request, record_id: str):
    """Download handler for items stored in the merge history session bucket."""

    history = request.session.get(MERGE_HISTORY_SESSION_KEY, [])
    for record in history:
        if record.get("id") == record_id:
            target = Path(record.get("path", ""))
            if not target.exists():
                raise Http404("That file is no longer available.")
            filename = record.get("name", "merged.pdf")
            return FileResponse(target.open("rb"), as_attachment=True, filename=filename)

    raise Http404("Merge result not found.")


@require_http_methods(["GET", "POST"])
def splitter_view(request):
    """Split PDFs by user-selected page markers."""

    history = request.session.get(SPLIT_HISTORY_SESSION_KEY, [])
    context = {
        "history": history,
        "error": "",
        "success": "",
        "split_points": request.POST.get("split_points", ""),
        "activity_history": _build_activity_history(request),
    }

    if request.method == "POST":
        uploaded_file = request.FILES.get("pdf")
        split_points = _parse_split_points(request.POST.get("split_points", ""))

        if not uploaded_file:
            context["error"] = "Upload a PDF before splitting."
        elif not split_points:
            context["error"] = "Add at least one split marker between page 2 and the last page."
        else:
            suffix = Path(uploaded_file.name).suffix or ".pdf"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_file.flush()
                temp_path = Path(temp_file.name)
                segments, total_pages = split_pdf(temp_path, split_points)
                updated_history = _record_split_history(
                    request,
                    uploaded_name=uploaded_file.name,
                    segments=segments,
                    split_points=split_points,
                    total_pages=total_pages,
                )
                context["history"] = updated_history
                context["success"] = f"Split {uploaded_file.name} into {len(segments)} file(s)."
                context["split_points"] = ",".join(str(p) for p in split_points)
            except SplitError as exc:
                context["error"] = str(exc)
            except Exception as exc:
                context["error"] = f"Failed to split PDF: {exc}"
            finally:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    context["activity_history"] = _build_activity_history(request)
    return render(request, "pdf_toolkit/split.html", context)


def download_split_result(request, record_id: str, segment_id: str):
    """Download handler for split segments."""

    history = request.session.get(SPLIT_HISTORY_SESSION_KEY, [])
    for record in history:
        if record.get("id") != record_id:
            continue
        for segment in record.get("segments", []):
            if segment.get("id") == segment_id:
                target = Path(segment.get("path", ""))
                if not target.exists():
                    raise Http404("Split file is no longer available.")
                filename = segment.get("name", "split.pdf")
                return FileResponse(target.open("rb"), as_attachment=True, filename=filename)
        break

    raise Http404("Split result not found.")


def _record_merge_history(request, *, output_name: str, merged_path: Path, total: int):
    history = request.session.get(MERGE_HISTORY_SESSION_KEY, [])
    record = {
        "id": uuid.uuid4().hex,
        "name": output_name,
        "path": str(merged_path),
        "total": total,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    history.insert(0, record)

    while len(history) > MERGE_HISTORY_LIMIT:
        expired = history.pop()
        _safe_remove(expired.get("path"))

    request.session[MERGE_HISTORY_SESSION_KEY] = history
    request.session.modified = True
    return history


def _record_split_history(
    request,
    *,
    uploaded_name: str,
    segments,
    split_points,
    total_pages: int,
):
    history = request.session.get(SPLIT_HISTORY_SESSION_KEY, [])
    base_name = Path(uploaded_name).stem or "split"
    record = {
        "id": uuid.uuid4().hex,
        "source_name": uploaded_name,
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_pages": total_pages,
        "split_points": sorted(set(int(p) for p in split_points if isinstance(p, int))),
        "segments": [],
    }

    for segment in segments:
        segment_id = uuid.uuid4().hex
        segment_name = _normalize_output_name(
            f"{base_name}-part-{segment.index}-pages-{segment.start_page}-{segment.end_page}"
        )
        record["segments"].append(
            {
                "id": segment_id,
                "name": segment_name,
                "label": f"Pages {segment.start_page} - {segment.end_page}",
                "path": str(segment.path),
            }
        )

    history.insert(0, record)

    while len(history) > SPLIT_HISTORY_LIMIT:
        expired = history.pop()
        _cleanup_split_segments(expired)

    request.session[SPLIT_HISTORY_SESSION_KEY] = history
    request.session.modified = True
    return history


def _safe_remove(path_value):
    if not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_split_segments(record):
    for segment in record.get("segments", []):
        _safe_remove(segment.get("path"))


def _normalize_output_name(raw_name: str) -> str:
    base = raw_name or "merged-pack"
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "merged-pack"
    return f"{sanitized}.pdf"


def _parse_split_points(raw_value: str) -> list[int]:
    if not raw_value:
        return []
    candidates = re.split(r"[,\s]+", raw_value)
    points: list[int] = []
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            value = int(cand)
        except ValueError:
            continue
        points.append(value)
    return points


def _build_activity_history(request):
    activity = []
    merge_history = request.session.get(MERGE_HISTORY_SESSION_KEY, [])
    for record in merge_history:
        created = record.get("created", "")
        total = record.get("total", 0)
        activity.append(
            {
                "type": "merge",
                "id": record.get("id"),
                "title": record.get("name", "Merged PDF"),
                "created": created,
                "meta": f"Merged {total} file{'s' if total != 1 else ''}",
                "total": total,
            }
        )

    split_history = request.session.get(SPLIT_HISTORY_SESSION_KEY, [])
    for record in split_history:
        created = record.get("created", "")
        points = record.get("split_points", [])
        activity.append(
            {
                "type": "split",
                "id": record.get("id"),
                "title": record.get("source_name", "Split PDF"),
                "created": created,
                "meta": f"{record.get('total_pages', 0)} pages split",
                "split_points": points,
                "segments": record.get("segments", []),
            }
        )

    activity.sort(key=lambda item: item.get("created", ""), reverse=True)
    return activity
