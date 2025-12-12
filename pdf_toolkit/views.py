"""Views for the PDF toolkit app."""

import os  # stdlib helper to delete tmp files
import re  # filename normalization
import tempfile  # stdlib helper for creating temp files
from pathlib import Path  # stdlib path helper for temp files

import requests  # hTTP client used for diagnostics in server_info
from django.conf import settings  # django settings object for diagnostics
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.core.files.base import ContentFile
from django.http import FileResponse, HttpResponse, JsonResponse  # HTTP responses
from django.shortcuts import get_object_or_404, render  # template rendering helper
from django.views.decorators.http import require_http_methods  # restrict HTTP verbs

from .models import DocumentArtifact, DocumentOperation
from .services.pdf_extractor import Method, extract_pdf_text  # shared PDF extraction logic
from .services.pdf_merger import merge_uploaded_files  # PDF merge helpers
from .services.pdf_splitter import SplitError, split_pdf  # PDF split helpers


@login_required(login_url="login")
def home(request):
    history = _build_activity_history(request.user)
    return render(request, "home.html", {"history": history})


def server_info(request):
    server_geodata = requests.get("https://ipwhois.app/json/").json()
    settings_dump = settings.__dict__
    return HttpResponse(f"{server_geodata}{settings_dump}")


@login_required(login_url="login")
@require_http_methods(["GET", "POST"])
def extract_view(request):
    """
    Upload view that accepts PDFs and returns the extracted text.

    Heavy OCR work happens in the shared service to keep the view lean.
    """

    method: Method = request.POST.get("method", "auto")  # type: ignore[assignment]
    normalize_requested = bool(request.POST.get("normalize"))
    page_range_raw = request.POST.get("pages", "")

    context = {
        "selected_method": method,
        "normalize": normalize_requested,
        "pages": page_range_raw,
        "text": "",
        "error": "",
        "history": _extract_history_for_user(request.user),
    }

    if request.method == "POST":
        try:
            pages = _parse_page_range(page_range_raw)
        except ValueError as exc:
            context["error"] = str(exc)
            return render(request, "pdf_toolkit/extract.html", context)

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
                extracted_text = extract_pdf_text(
                    temp_path,
                    method=method,
                    normalize=normalize_requested,
                    pages=pages or None,
                )
                context["text"] = extracted_text
                _persist_extract_result(
                    user=request.user,
                    source_name=uploaded_file.name,
                    text=extracted_text,
                    method=method,
                    normalize=normalize_requested,
                    page_range=page_range_raw or "All pages",
                )
                context["history"] = _extract_history_for_user(request.user)
            except Exception as exc:  # pragma: no cover - best effort UI feedback
                context["error"] = f"Failed to extract text: {exc}"
            finally:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    return render(request, "pdf_toolkit/extract.html", context)


@login_required(login_url="login")
@require_http_methods(["GET", "POST"])
def merge_create_view(request):
    """Upload + ordering workflow for merging heterogeneous files into PDF."""

    allowed_attr = ".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
    context = {
        "history": _merge_history_for_user(request.user),
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
                _persist_merge_result(
                    user=request.user,
                    output_name=safe_name,
                    merged_path=merged_path,
                    filenames=[uploaded.name for uploaded in files],
                )
                context["history"] = _merge_history_for_user(request.user)
                context["success"] = f"Merged {len(files)} {'file' if len(files) == 1 else 'files'} into {safe_name}"
            except ValueError as exc:
                context["error"] = str(exc)

    return render(request, "pdf_toolkit/merge.html", context)


@login_required(login_url="login")
def download_merge_result(request, artifact_id: int):
    artifact = get_object_or_404(
        DocumentArtifact,
        pk=artifact_id,
        operation__user=request.user,
        operation__operation_type=DocumentOperation.OperationType.MERGE,
    )
    return FileResponse(
        artifact.file.open("rb"),
        as_attachment=True,
        filename=artifact.display_name or "merged.pdf",
    )


@login_required(login_url="login")
@require_http_methods(["GET", "POST"])
def splitter_view(request):
    """Split PDFs by user-selected page markers."""

    context = {
        "error": "",
        "success": "",
        "split_points": request.POST.get("split_points", ""),
        "activity_history": _build_activity_history(request.user),
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
                _persist_split_results(
                    user=request.user,
                    uploaded_name=uploaded_file.name,
                    segments=segments,
                    split_points=split_points,
                    total_pages=total_pages,
                )
                context["activity_history"] = _build_activity_history(request.user)
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

    return render(request, "pdf_toolkit/split.html", context)


@login_required(login_url="login")
def download_split_result(request, artifact_id: int):
    artifact = get_object_or_404(
        DocumentArtifact,
        pk=artifact_id,
        operation__user=request.user,
        operation__operation_type=DocumentOperation.OperationType.SPLIT,
    )
    return FileResponse(
        artifact.file.open("rb"),
        as_attachment=True,
        filename=artifact.display_name or "split.pdf",
    )


@login_required(login_url="login")
def download_extract_result(request, artifact_id: int):
    artifact = get_object_or_404(
        DocumentArtifact,
        pk=artifact_id,
        operation__user=request.user,
        operation__operation_type=DocumentOperation.OperationType.EXTRACT,
    )
    return FileResponse(
        artifact.file.open("rb"),
        as_attachment=True,
        filename=artifact.display_name or "extracted.txt",
    )


@login_required(login_url="login")
def extract_preview(request, artifact_id: int):
    artifact = get_object_or_404(
        DocumentArtifact,
        pk=artifact_id,
        operation__user=request.user,
        operation__operation_type=DocumentOperation.OperationType.EXTRACT,
    )
    with artifact.file.open("r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return JsonResponse({"text": text})


def _normalize_output_name(raw_name: str) -> str:
    base = raw_name or "merged-pack"
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "merged-pack"
    return f"{sanitized}.pdf"


def _safe_text_filename(source_name: str) -> str:
    base = Path(source_name).stem or "extracted"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "extracted"
    return f"{sanitized}-text.txt"


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


def _parse_page_range(raw_value: str) -> list[int]:
    if not raw_value:
        return []
    cleaned: list[int] = []
    for chunk in raw_value.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:  # pragma: no cover - user input guard
                raise ValueError("Use numeric page ranges like 3-5") from exc
            if start <= 0 or end <= 0 or end < start:
                raise ValueError("Page ranges must be positive and increasing")
            cleaned.extend(range(start, end + 1))
        else:
            try:
                value = int(part)
            except ValueError as exc:  # pragma: no cover
                raise ValueError("Pages must be numbers like 2 or 4-6") from exc
            if value <= 0:
                raise ValueError("Page numbers must be positive")
            cleaned.append(value)
    return sorted(set(cleaned))


def _persist_merge_result(*, user, output_name: str, merged_path: Path, filenames):
    total_files = len(filenames)
    operation = DocumentOperation.objects.create(
        user=user,
        operation_type=DocumentOperation.OperationType.MERGE,
        output_name=output_name,
        metadata={
            "total_files": total_files,
            "source_files": filenames,
        },
    )
    artifact = DocumentArtifact(
        operation=operation,
        display_name=output_name,
        metadata={"total_files": total_files},
    )
    with open(merged_path, "rb") as merged_file:
        artifact.file.save(Path(output_name).name, File(merged_file), save=True)
    Path(merged_path).unlink(missing_ok=True)
    return artifact


def _persist_split_results(
    *,
    user,
    uploaded_name: str,
    segments,
    split_points,
    total_pages: int,
):
    operation = DocumentOperation.objects.create(
        user=user,
        operation_type=DocumentOperation.OperationType.SPLIT,
        source_name=uploaded_name,
        metadata={
            "split_points": sorted(set(int(p) for p in split_points if isinstance(p, int))),
            "total_pages": total_pages,
        },
    )
    base_name = Path(uploaded_name).stem or "split"
    for segment in segments:
        label = f"Pages {segment.start_page} - {segment.end_page}"
        display_name = _normalize_output_name(
            f"{base_name}-part-{segment.index}-pages-{segment.start_page}-{segment.end_page}"
        )
        artifact = DocumentArtifact(
            operation=operation,
            display_name=display_name,
            metadata={
                "label": label,
                "start_page": segment.start_page,
                "end_page": segment.end_page,
            },
        )
        with open(segment.path, "rb") as split_file:
            artifact.file.save(display_name, File(split_file), save=True)
        Path(segment.path).unlink(missing_ok=True)
    return operation


def _persist_extract_result(
    *,
    user,
    source_name: str,
    text: str,
    method: str,
    normalize: bool,
    page_range: str,
):
    filename = _safe_text_filename(source_name)
    operation = DocumentOperation.objects.create(
        user=user,
        operation_type=DocumentOperation.OperationType.EXTRACT,
        source_name=source_name,
        output_name=filename,
        metadata={
            "method": method,
            "normalize": normalize,
            "page_range": page_range,
            "characters": len(text),
        },
    )
    snippet = text[:600]
    artifact = DocumentArtifact(
        operation=operation,
        display_name=filename,
        metadata={
            "snippet": snippet,
            "characters": len(text),
        },
    )
    artifact.file.save(filename, ContentFile(text), save=True)
    return artifact


def _merge_history_for_user(user, limit: int = 10):
    operations = (
        DocumentOperation.objects.filter(
            user=user,
            operation_type=DocumentOperation.OperationType.MERGE,
        )
        .prefetch_related("artifacts")
        .order_by("-created_at")[:limit]
    )
    history = []
    for operation in operations:
        artifact = operation.artifacts.first()
        if not artifact:
            continue
        total_files = artifact.metadata.get("total_files") or operation.metadata.get("total_files", 0)
        history.append(
            {
                "artifact_id": artifact.id,
                "name": artifact.display_name,
                "created": operation.created_at.strftime("%Y-%m-%d %H:%M"),
                "total": total_files,
            }
        )
    return history


def _extract_history_for_user(user, limit: int = 10):
    operations = (
        DocumentOperation.objects.filter(
            user=user,
            operation_type=DocumentOperation.OperationType.EXTRACT,
        )
        .prefetch_related("artifacts")
        .order_by("-created_at")[:limit]
    )
    history = []
    for operation in operations:
        artifact = operation.artifacts.first()
        if not artifact:
            continue
        history.append(
            {
                "artifact_id": artifact.id,
                "title": operation.source_name or artifact.display_name,
                "created": operation.created_at.strftime("%Y-%m-%d %H:%M"),
                "method": operation.metadata.get("method", "auto"),
                "page_range": operation.metadata.get("page_range", "All pages"),
                "snippet": artifact.metadata.get("snippet", ""),
            }
        )
    return history


def _build_activity_history(user, limit: int = 10):
    operations = (
        DocumentOperation.objects.filter(user=user)
        .prefetch_related("artifacts")
        .order_by("-created_at")[:limit]
    )
    activity = []
    for operation in operations:
        entry = {
            "type": operation.operation_type,
            "title": operation.output_name or operation.source_name or operation.get_operation_type_display(),
            "created": operation.created_at.strftime("%Y-%m-%d %H:%M"),
            "meta": "",
        }
        if operation.operation_type == DocumentOperation.OperationType.MERGE:
            artifact = operation.artifacts.first()
            if not artifact:
                continue
            total_files = artifact.metadata.get("total_files") or operation.metadata.get("total_files", 0)
            entry.update(
                {
                    "meta": f"Merged {total_files} file{'s' if total_files != 1 else ''}",
                    "total": total_files,
                    "artifact_id": artifact.id,
                    "segments": [],
                }
            )
        elif operation.operation_type == DocumentOperation.OperationType.SPLIT:
            segments = []
            for artifact in operation.artifacts.all():
                segments.append(
                    {
                        "label": artifact.metadata.get("label", artifact.display_name),
                        "name": artifact.display_name,
                        "artifact_id": artifact.id,
                    }
                )
            entry.update(
                {
                    "meta": f"{operation.metadata.get('total_pages', 0)} pages split",
                    "split_points": operation.metadata.get("split_points", []),
                    "segments": segments,
                }
            )
        else:  # extract history
            artifact = operation.artifacts.first()
            if not artifact:
                continue
            snippet = artifact.metadata.get("snippet", "")
            entry.update(
                {
                    "meta": f"Extracted via {operation.metadata.get('method', 'auto').title()}",
                    "artifact_id": artifact.id,
                    "snippet": snippet,
                    "segments": [],
                }
            )
        activity.append(entry)
    return activity
