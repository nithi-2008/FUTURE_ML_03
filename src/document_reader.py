"""
Document Reader Module for extracting text from PDF, DOCX, and TXT resumes.
Supports both file paths on disk and Streamlit UploadedFile / BytesIO objects.
"""

import io
import os
from typing import BinaryIO, Dict, Tuple, Union
import docx
import pypdf


class DocumentReader:
    """
    Extracts raw text and metadata from uploaded or local resume files (.pdf, .docx, .txt).
    """

    @staticmethod
    def read_pdf(file_source: Union[str, BinaryIO, bytes]) -> str:
        """
        Extracts text from a PDF file using pypdf.
        """
        text_parts = []
        try:
            if isinstance(file_source, bytes):
                stream = io.BytesIO(file_source)
                reader = pypdf.PdfReader(stream)
            elif isinstance(file_source, str):
                if not os.path.exists(file_source):
                    raise FileNotFoundError(f"File not found: {file_source}")
                reader = pypdf.PdfReader(file_source)
            else:
                reader = pypdf.PdfReader(file_source)

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text.strip())

        except Exception as e:
            raise ValueError(f"Failed to read PDF document: {str(e)}")

        return "\n\n".join(filter(None, text_parts))

    @staticmethod
    def read_docx(file_source: Union[str, BinaryIO, bytes]) -> str:
        """
        Extracts text from a DOCX file using python-docx.
        """
        text_parts = []
        try:
            if isinstance(file_source, bytes):
                stream = io.BytesIO(file_source)
                doc = docx.Document(stream)
            elif isinstance(file_source, str):
                if not os.path.exists(file_source):
                    raise FileNotFoundError(f"File not found: {file_source}")
                doc = docx.Document(file_source)
            else:
                doc = docx.Document(file_source)

            # Paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text.strip())

            # Tables
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        text_parts.append(" | ".join(row_cells))

        except Exception as e:
            raise ValueError(f"Failed to read DOCX document: {str(e)}")

        return "\n".join(text_parts)

    @staticmethod
    def read_txt(file_source: Union[str, BinaryIO, bytes]) -> str:
        """
        Extracts text from a plain TXT file with fallback encodings.
        """
        if isinstance(file_source, bytes):
            raw_bytes = file_source
        elif isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise FileNotFoundError(f"File not found: {file_source}")
            with open(file_source, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = file_source.read()

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue

        return raw_bytes.decode("utf-8", errors="replace")

    @classmethod
    def extract_text(cls, file_source: Union[str, BinaryIO, bytes], filename: str = "") -> Tuple[str, Dict]:
        """
        Generic entry point to extract text and basic metadata from PDF, DOCX, or TXT.
        """
        if isinstance(file_source, str) and not filename:
            filename = os.path.basename(file_source)

        ext = os.path.splitext(filename)[1].lower() if filename else ""

        if ext == ".pdf":
            text = cls.read_pdf(file_source)
            file_type = "PDF"
        elif ext in [".docx", ".doc"]:
            text = cls.read_docx(file_source)
            file_type = "DOCX"
        elif ext in [".txt", ".md", ".rtf"] or not ext:
            text = cls.read_txt(file_source)
            file_type = "TXT"
        else:
            # Attempt TXT decoding as fallback
            try:
                text = cls.read_txt(file_source)
                file_type = ext.upper()
            except Exception:
                raise ValueError(f"Unsupported file format '{ext}'. Please upload PDF, DOCX, or TXT.")

        cleaned_text = text.strip()
        word_count = len(cleaned_text.split())
        char_count = len(cleaned_text)

        metadata = {
            "filename": filename or "uploaded_document",
            "file_type": file_type,
            "word_count": word_count,
            "char_count": char_count,
            "is_empty": word_count == 0
        }

        return cleaned_text, metadata


reader = DocumentReader()
