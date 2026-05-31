"""
Text normalization utilities for document preprocessing.
Standardizes unicode, removes line-break noise, and reduces formatting clutter before downstream storage or vectorization.
"""

from io import BytesIO
from pathlib import Path
import re
import unicodedata


def normalize_text(text: str) -> str:
    if text is None:
        raise ValueError("Text cannot be empty")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    normalized = re.sub(r"[\t\f\v]+", " ", normalized)
    normalized = re.sub(r"[ ]{2,}", " ", normalized)

    lines = []
    previous_was_empty = False
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if not previous_was_empty:
                lines.append("")
            previous_was_empty = True
            continue

        line = re.sub(r"\s{2,}", " ", line)
        lines.append(line)
        previous_was_empty = False

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def prepare_document_text(text: str) -> str:
    """Prepare uploaded document text for storage and downstream vectorization."""
    normalized = normalize_text(text)

    # Reduce common formatting noise from pasted documents and PDFs.
    normalized = re.sub(r"(?m)^\s*Page \d+\s*$", "", normalized)
    normalized = re.sub(r"(?m)^\s*\d+\s*$", "", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{2,}", "\n\n", normalized)
    return normalized.strip()


def extract_uploaded_document_text(file_name: str, content: bytes) -> str:
    if not file_name or not file_name.strip():
        raise ValueError("File name cannot be empty")
    if content is None or not content:
        raise ValueError("File content cannot be empty")

    extension = Path(file_name.strip()).suffix.lower()

    if extension == ".txt":
        try:
            raw_text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raw_text = content.decode("latin-1")
        return prepare_document_text(raw_text)

    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        extracted_text = "\n\n".join(extracted_pages).strip()
        if not extracted_text:
            raise ValueError("No text could be extracted from the PDF file")
        return prepare_document_text(extracted_text)

    if extension == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text and paragraph.text.strip()]
        extracted_text = "\n\n".join(paragraphs).strip()
        if not extracted_text:
            raise ValueError("No text could be extracted from the DOCX file")
        return prepare_document_text(extracted_text)

    raise ValueError("Only TXT, PDF, and DOCX files are allowed")
