import os
import re #filename normalization
import socket
import tempfile #creating temp files
from pathlib import Path 
import requests 
from django.conf import settings 
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.core.files.base import ContentFile
from django.http import FileResponse, JsonResponse 
from django.shortcuts import get_object_or_404, render  #template rendering
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from pytesseract import TesseractNotFoundError

from .models import DocumentArtifact, DocumentOperation
from .services.pdf_extractor import extract_pdf_text #PDF extraction logic
from .services.pdf_merger import merge_uploaded_files #PDF merge helpers
from .services.pdf_splitter import SplitError, split_pdf #PDF split helpers
from .services.table_extractor import extract_pdf_tables_as_csv
from .services.table_exporter import extract_pdf_tables_as_xlsx


@login_required(login_url="login")
def home(request):#home = recent activity for each user
    history = _build_activity_history(request.user)
    return render(request, "home.html", {"history": history})

_METADATA_ROOT = "http://metadata/computeMetadata/v1/"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def _query_metadata(path: str) -> str:
    """Best-effort helper to read a metadata attribute from GCE."""
    try:
        response = requests.get(
            f"{_METADATA_ROOT}{path}", headers=_METADATA_HEADERS, timeout=0.25
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return ""


def server_info(request):
    """Lightweight diagnostics endpoint required by the assignment spec."""
    zone = _query_metadata("instance/zone")
    instance_id = _query_metadata("instance/id")
    internal_ip = _query_metadata("instance/network-interfaces/0/ip")
    external_ip = _query_metadata("instance/network-interfaces/0/access-configs/0/external-ip")

    data = {
        "hostname": socket.gethostname(),
        "instance_id": instance_id or "unavailable",
        "zone": zone.rsplit("/", 1)[-1] if zone else "unavailable",
        "project_id": _query_metadata("project/project-id") or "unavailable",
        "internal_ip": internal_ip or "unavailable",
        "external_ip": external_ip or "unavailable",
        "client_ip": request.META.get("REMOTE_ADDR", ""),
        "server_time_utc": timezone.now().isoformat(),
        "app_version": os.environ.get("APP_VERSION", os.environ.get("COMMIT_SHA", "unknown")),
        "settings_debug": settings.DEBUG,
        "healthy": True,
    }
    return JsonResponse(data)


@login_required(login_url="login")
@require_http_methods(["GET", "POST"])
def extract_view(request): #Upload view
    method=request.POST.get("method", "auto")
    normalize_requested=bool(request.POST.get("normalize"))
    page_range_raw=request.POST.get("pages", "")
    export_csv_requested=bool(request.POST.get("export_csv"))
    export_xlsx_requested=bool(request.POST.get("export_xlsx"))
    scanned_hint=bool(request.POST.get("scanned_hint"))
    deskew_hint=bool(request.POST.get("deskew_hint"))

    context = { #base template
        "selected_method": method,
        "normalize": normalize_requested,
        "pages": page_range_raw,
        "export_csv": export_csv_requested,
        "export_xlsx": export_xlsx_requested,
        "scanned_hint": scanned_hint,
        "deskew_hint": deskew_hint,
        "text": "",
        "error": "",
        "csv_preview": None,
        "text_artifact_id": None,
        "csv_error": "",
        "xlsx_error": "",
        "xlsx_artifact_id": None,
        "history": _extract_history_for_user(request.user),
    }
    if request.method == "POST":
        try:
            pages = _parse_page_range(page_range_raw)
        except ValueError as exc:
            context["error"] = str(exc)
            return render(request, "pdf_toolkit/extract.html", context)

        uploaded_file = request.FILES.get("pdf") #grabs the uploaded file from request.FILES["pdf"]
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
                scan_scale = (450 / 72) if scanned_hint else (300 / 72)
                extracted_text = extract_pdf_text(
                    temp_path,
                    method=method,
                    normalize=normalize_requested,
                    pages=pages or None,
                    scale=scan_scale,
                    preprocess_scans=scanned_hint,
                    deskew=deskew_hint,
                )
                context["text"] = extracted_text
                csv_result = None
                if export_csv_requested:
                    try:
                        csv_result = extract_pdf_tables_as_csv(
                            temp_path,
                            pages=pages or None,
                            scale=scan_scale,
                            preprocess_scans=scanned_hint,
                            deskew=deskew_hint,
                        )
                    except TesseractNotFoundError:
                        context["csv_error"] = (
                            "CSV export requires the Tesseract OCR binary. "
                            "Install it (e.g. `sudo apt install tesseract-ocr`) "
                            "or uncheck the CSV checkbox."
                        )
                        csv_result = None
                    except Exception as exc:
                        context["csv_error"] = f"Unable to detect tables: {exc}"
                        csv_result = None

                xlsx_result = None
                if export_xlsx_requested:
                    try:
                        xlsx_result = extract_pdf_tables_as_xlsx(
                            temp_path,
                            pages=pages or None,
                            scale=scan_scale,
                            preprocess_scans=scanned_hint,
                            deskew=deskew_hint,
                        )
                        if xlsx_result is None:
                            context["xlsx_error"] = "Unable to detect tables for Excel export."
                    except TesseractNotFoundError:
                        context["xlsx_error"] = (
                            "Excel export requires the Tesseract OCR binary for scanned PDFs. "
                            "Install it (e.g. `sudo apt install tesseract-ocr`) "
                            "or uncheck the Excel checkbox."
                        )
                        xlsx_result = None
                    except Exception as exc:
                        context["xlsx_error"] = f"Unable to export Excel: {exc}"
                        xlsx_result = None

                text_artifact, csv_artifact, xlsx_artifact = _persist_extract_result(
                    user=request.user,
                    source_name=uploaded_file.name,
                    text=extracted_text,
                    method=method,
                    normalize=normalize_requested,
                    page_range=page_range_raw or "All pages",
                    csv_result=csv_result,
                    xlsx_result=xlsx_result,
                    scan_hint=scanned_hint,
                    deskew_hint=deskew_hint,
                )
                context["text_artifact_id"] = text_artifact.id if text_artifact else None
                if csv_result and csv_artifact:
                    context["csv_preview"] = {
                        "artifact_id": csv_artifact.id,
                        "headers": ["Page"] + csv_result.headers,
                        "rows": [
                            [row["page"]] + row["cells"]
                            for row in csv_result.preview_rows
                        ],
                        "count": len(csv_result.preview_rows),
                        "total": csv_result.total_rows,
                    }
                if xlsx_artifact:
                    context["xlsx_artifact_id"] = xlsx_artifact.id
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
def merge_create_view(request):#upload + ordering workflow for merging heterogeneous files into PDF
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
def download_merge_result(request, artifact_id): #merged PDF artifact belonging to the current user
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
def splitter_view(request): #split PDFs by user-selected page markers Used AI to assist
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
                for chunk in uploaded_file.chunks(): #AI assist here as i couldnt get it to split correctly
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
def download_split_result(request, artifact_id): #1 of the split segment PDFs after verifying ownership
    artifact= get_object_or_404(
        DocumentArtifact,
        pk=artifact_id,
        operation__user=request.user,
        operation__operation_type=DocumentOperation.OperationType.SPLIT,)
    return FileResponse(
        artifact.file.open("rb"),
        as_attachment=True,
        filename=artifact.display_name or "split.pdf",)


@login_required(login_url="login")
def download_extract_result(request, artifact_id): #return the extracted text as a downloadable file for the user
    artifact= get_object_or_404( DocumentArtifact, pk=artifact_id, operation__user=request.user, operation__operation_type=DocumentOperation.OperationType.EXTRACT,)
    return FileResponse( artifact.file.open("rb"), as_attachment=True, filename=artifact.display_name or "extracted.txt",)


@login_required(login_url="login")
def extract_preview(request, artifact_id): #Return JSON containing the text snippet for preview card
    artifact= get_object_or_404(DocumentArtifact, pk=artifact_id, operation__user=request.user, operation__operation_type=DocumentOperation.OperationType.EXTRACT,)
    with artifact.file.open("rb") as handle:
        text_bytes = handle.read()
    text = text_bytes.decode("utf-8", errors="replace")
    return JsonResponse({"text": text})


def _normalize_output_name(raw_name: str) -> str:
    #AI to help me sanitise
    base = raw_name or "merged-pack"
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "merged-pack"
    return f"{sanitized}.pdf"


def _safe_text_filename(source_name: str) -> str: #generates filename
    base = Path(source_name).stem or "extracted"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "extracted"
    return f"{sanitized}-text.txt"


def _safe_csv_filename(source_name: str) -> str:
    base = Path(source_name).stem or "extracted"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "extracted"
    return f"{sanitized}-table.csv"


def _safe_xlsx_filename(source_name: str) -> str:
    base = Path(source_name).stem or "extracted"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.") or "extracted"
    return f"{sanitized}-table.xlsx"


def _parse_split_points(raw_value): #turns input into ints
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


def _parse_page_range(raw_value):
    #inputs like '1-3,8' into a sorted list
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
            except ValueError as exc:  # AI Assisted here - pragma: no cover - user input guard
                raise ValueError("Use numeric page ranges like 3-5") from exc
            if start <= 0 or end <= 0 or end < start:
                raise ValueError("Page ranges must be positive and increasing")
            cleaned.extend(range(start, end + 1))
        else:
            try:
                value = int(part)
            except ValueError as exc:  # AI assisted here - pragma: no cover
                raise ValueError("Pages must be numbers like 2 or 4-6") from exc
            if value <= 0:
                raise ValueError("Page numbers must be positive")
            cleaned.append(value)
    return sorted(set(cleaned))


def _persist_merge_result(*, user, output_name, merged_path, filenames): # Persist merged metadata & the merged PDF file/arti
    total_files = len(filenames)
    operation = DocumentOperation.objects.create(user=user, operation_type=DocumentOperation.OperationType.MERGE, output_name=output_name,
        metadata={
            "total_files": total_files,
            "source_files": filenames,
        },)
    artifact = DocumentArtifact( operation=operation, display_name=output_name, metadata={"total_files": total_files},)
    with open(merged_path, "rb") as merged_file:
        artifact.file.save(Path(output_name).name, File(merged_file), save=True)
    Path(merged_path).unlink(missing_ok=True)
    return artifact


def _persist_split_results(*, user, uploaded_name, segments, split_points, total_pages,): #stores each split chunk as its own file
    operation = DocumentOperation.objects.create( user=user, operation_type=DocumentOperation.OperationType.SPLIT, source_name=uploaded_name,
        metadata={
            "split_points": sorted(set(int(p) for p in split_points if isinstance(p, int))),
            "total_pages": total_pages,
        },)
    base_name= Path(uploaded_name).stem or "split" #AI Assisted - I was struggleing on the persistant data I Was loosing it
    for segment in segments:
        label= f"Pages {segment.start_page} - {segment.end_page}"
        display_name= _normalize_output_name(
            f"{base_name}-part-{segment.index}-pages-{segment.start_page}-{segment.end_page}"
        )
        artifact = DocumentArtifact( operation=operation, display_name=display_name,
            metadata={
                "label": label,
                "start_page": segment.start_page,
                "end_page": segment.end_page,
            },)
        with open(segment.path, "rb") as split_file:
            artifact.file.save(display_name, File(split_file), save=True)
        Path(segment.path).unlink(missing_ok=True)
    return operation


def _persist_extract_result(
    *,
    user,
    source_name,
    text,
    method,
    normalize,
    page_range,
    csv_result=None,
    xlsx_result=None,
    scan_hint=False,
    deskew_hint=False,
): #extracted text as a plaintext file & metadata row
    filename= _safe_text_filename(source_name)
    operation_metadata={
        "method": method,
        "normalize": normalize,
        "page_range": page_range,
        "characters": len(text),
        "has_csv": bool(csv_result),
        "scan_hint": bool(scan_hint),
        "deskew_hint": bool(deskew_hint),
    }
    if csv_result:
        operation_metadata["csv_rows"] = csv_result.total_rows
        operation_metadata["csv_columns"] = csv_result.column_count
    if xlsx_result:
        operation_metadata["xlsx_rows"] = xlsx_result.total_rows
        operation_metadata["xlsx_columns"] = xlsx_result.column_count
    operation= DocumentOperation.objects.create( user=user, operation_type=DocumentOperation.OperationType.EXTRACT, source_name=source_name, output_name=filename,
        metadata=operation_metadata,)
    snippet= text[:600]
    text_artifact= DocumentArtifact(operation=operation, display_name=filename,
        metadata={
            "snippet": snippet,
            "characters": len(text),
            "kind": "text",
        },)
    text_artifact.file.save(filename, ContentFile(text), save=True)

    csv_artifact = None
    if csv_result:
        csv_filename = _safe_csv_filename(source_name)
        csv_artifact = DocumentArtifact(
            operation=operation,
            display_name=csv_filename,
            metadata={
                "kind": "csv",
                "headers": csv_result.headers,
                "total_rows": csv_result.total_rows,
                "page_count": csv_result.page_count,
            },
        )
        csv_artifact.file.save(csv_filename, ContentFile(csv_result.csv_text), save=True)

    xlsx_artifact = None
    if xlsx_result:
        xlsx_filename = _safe_xlsx_filename(source_name)
        xlsx_artifact = DocumentArtifact(
            operation=operation,
            display_name=xlsx_filename,
            metadata={
                "kind": "xlsx",
                "headers": xlsx_result.headers,
                "total_rows": xlsx_result.total_rows,
                "page_count": xlsx_result.page_count,
            },
        )
        xlsx_artifact.file.save(
            xlsx_filename,
            ContentFile(xlsx_result.excel_bytes),
            save=True,
        )

    return text_artifact, csv_artifact, xlsx_artifact


def _extract_operation_artifacts(operation):
    artifacts = list(operation.artifacts.order_by("created_at"))
    text_artifact = next(
        (artifact for artifact in artifacts if artifact.metadata.get("kind") == "text"),
        None,
    )
    csv_artifact = next(
        (artifact for artifact in artifacts if artifact.metadata.get("kind") == "csv"),
        None,
    )
    xlsx_artifact = next(
        (artifact for artifact in artifacts if artifact.metadata.get("kind") == "xlsx"),
        None,
    )
    if not text_artifact and artifacts:
        text_artifact = artifacts[0]
    return text_artifact, csv_artifact, xlsx_artifact


def _merge_history_for_user(user, limit=10): #simplified list of recent merge operations
    operations= (DocumentOperation.objects.filter(user=user, operation_type=DocumentOperation.OperationType.MERGE,).prefetch_related("artifacts").order_by("-created_at")[:limit])
    history= []
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


def _extract_history_for_user(user, limit=10):#history specific to extraction
    operations = (DocumentOperation.objects.filter( user=user, operation_type=DocumentOperation.OperationType.EXTRACT,) .prefetch_related("artifacts") .order_by("-created_at")[:limit])
    history = []
    for operation in operations:
        text_artifact, csv_artifact, xlsx_artifact = _extract_operation_artifacts(operation)
        if not text_artifact:
            continue
        history.append(
            {
                "artifact_id": text_artifact.id,
                "csv_artifact_id": csv_artifact.id if csv_artifact else None,
                "xlsx_artifact_id": xlsx_artifact.id if xlsx_artifact else None,
                "title": operation.source_name or text_artifact.display_name,
                "created": operation.created_at.strftime("%Y-%m-%d %H:%M"),
                "method": operation.metadata.get("method", "auto"),
                "page_range": operation.metadata.get("page_range", "All pages"),
                "snippet": text_artifact.metadata.get("snippet", ""),
            }
        )
    return history


def _build_activity_history(user, limit=10): #AI assisted 
    entries = []
    qs = DocumentOperation.objects.filter(user=user).prefetch_related("artifacts").order_by("-created_at")[:limit]
    for op in qs:
        base = {
            "type": op.operation_type,
            "title": op.output_name or op.source_name or op.get_operation_type_display(),
            "created": op.created_at.strftime("%Y-%m-%d %H:%M"),
            "segments": [],
        }
        if op.operation_type == DocumentOperation.OperationType.MERGE: #AI
            artifact = op.artifacts.first()
            if not artifact:
                continue
            total = artifact.metadata.get("total_files") or op.metadata.get("total_files", 0)
            base["meta"] = f"Merged {total} file{'s' if total != 1 else ''}"
            base["artifact_id"] = artifact.id
        elif op.operation_type == DocumentOperation.OperationType.SPLIT:
            base["meta"] = f"{op.metadata.get('total_pages', 0)} pages split"
            base["split_points"] = op.metadata.get("split_points", [])
            for art in op.artifacts.all():
                base["segments"].append(
                    {
                        "label": art.metadata.get("label", art.display_name),
                        "name": art.display_name,
                        "artifact_id": art.id,
                    }
                )
        else: #AI
            text_artifact, csv_artifact, xlsx_artifact = _extract_operation_artifacts(op)
            if not text_artifact:
                continue
            base["artifact_id"] = text_artifact.id
            base["meta"] = f"Extracted via {op.metadata.get('method', 'auto').title()}"
            base["snippet"] = text_artifact.metadata.get("snippet", "")
            if csv_artifact:
                base["csv_artifact_id"] = csv_artifact.id
            if xlsx_artifact:
                base["xlsx_artifact_id"] = xlsx_artifact.id
        entries.append(base)
    return entries
