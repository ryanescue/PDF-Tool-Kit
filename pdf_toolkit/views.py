"""Views for the PDF toolkit app."""

import os #stdlib helper to delete tmp files
import tempfile #stdlib helper for creating temp files
from pathlib import Path #stdlib path helper for temp files

import requests  #hTTP client used for diagnostics in server_info
from django.conf import settings  #django settings object for diagnostics
from django.http import HttpResponse  #basic HTTP responses
from django.shortcuts import render  #template rendering helper
from django.views.decorators.http import require_http_methods  #restrict HTTP verbs

from .services.pdf_extractor import Method, extract_pdf_text  #shared PDF extraction logic


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
