""" 
CMPT 310 - ResuMatch

This module extracts text from uploaded resume files.
Supported formats: 
- PDF
- DOCX
- TXT

"""

from pathlib import Path
from typing import BinaryIO, Union

import pdfplumber
from docx import Document

FileInput = Union[str, Path, BinaryIO]

def read_docx(file: FileInput) -> str:
    """Extract and return text from a DOCX file."""

    document = Document(file)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)

def read_pdf(file: FileInput) -> str:
    """Extract and return text from a text-based PDF file."""

    pages = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            if text:
                pages.append(text.strip())
    return "\n".join(pages)


def read_txt(file: FileInput) -> str:
    """Extract and return text from a TXT file."""

    if isinstance(file, (str, Path)):
        return Path(file).read_text(
            encoding = "utf-8",
            errors = "ignore",
        )

    file.seek(0)
    content = file.read()

    if isinstance(content, bytes):
        return content.decode("utf-8", errors = "ignore")

    return content 

def extract_resume_text(file: FileInput, filename: str | None = None) -> str:
    """
    Detect the resume file type and return its extracted text.

    For Streamlit uploaded files, the filename is taken from file.name.
    """

    detected_name = filename or getattr(file, "name", None)

    if not detected_name:
        raise ValueError("The filename is required to detect the file type")

    extension = Path(detected_name).suffix.lower()

    if hasattr(file, "seek"):
        file.seek(0)

    if extension == ".pdf":
        text = read_pdf(file)
    elif extension == ".docx":
        text = read_docx(file)
    elif extension == ".txt":
        text = read_txt(file)
    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF, DOCX, or TXT file."
        )
    if not text.strip():
        raise ValueError(
            "No readable text was found in the uploaded resume"
        )
    return text.strip()


if __name__ == "__main__":
    resume_text = extract_resume_text("test.docx")
    print(resume_text[:1000])

