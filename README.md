# Document-Extractor-Tool-CSCI-565-

## OCR prerequisites

Some extractor features (scanned-PDF text extraction, EasyOCR, and the optional CSV/table detection) rely on the native [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) binary. Install it on your system before enabling these options:

```bash
sudo apt update
sudo apt install -y tesseract-ocr
```

On macOS use `brew install tesseract`, and on Windows install the official Tesseract package and make sure `tesseract.exe` is on your `PATH`.

When working with scanned PDFs inside the extractor UI, enable the “This is a scanned document” checkbox to run the higher DPI render plus image preprocessing pipeline before OCR. This improves accuracy for faint digits and helps the CSV/table detector stay aligned.
