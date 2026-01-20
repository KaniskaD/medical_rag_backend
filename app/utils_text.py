import os
import json
import csv
import io
import hashlib
from typing import Optional

import pdfplumber
from PIL import Image
import pytesseract
from docx import Document  # for .docx

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".txt":
            return _extract_text_from_txt(file_path)
        elif ext == ".pdf":
            return _extract_text_from_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            return _extract_text_from_image(file_path)
        elif ext == ".docx":
            return _extract_text_from_docx(file_path)
        elif ext == ".json":
            return _extract_text_from_json(file_path)
        elif ext == ".csv":
            return _extract_text_from_csv(file_path)
        else:
            return ""
    except Exception:
        return ""

def _extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()

def _extract_text_from_image(file_path: str) -> str:
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def _extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras).strip()

def _extract_text_from_json(file_path: str) -> str:
    """
    Converts structured JSON medical data into readable text.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""

    if isinstance(data, dict):
        return "\n".join(f"{k}: {v}" for k, v in data.items())

    if isinstance(data, list):
        blocks = []
        for item in data:
            if isinstance(item, dict):
                block = "\n".join(f"{k}: {v}" for k, v in item.items())
                blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    return ""

def _extract_text_from_csv(file_path: str) -> str:
    """
    Converts CSV lab data into readable medical text.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return ""

    if not rows:
        return ""

    blocks = []
    for row in rows:
        lines = [f"{k}: {v}" for k, v in row.items()]
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)

def calculate_content_hash(file_bytes: bytes) -> str:
    """
    Generates a SHA-256 hash of file content.
    Used to prevent duplicate report uploads.
    """
    return hashlib.sha256(file_bytes).hexdigest()