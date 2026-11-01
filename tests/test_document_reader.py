"""
Unit tests for DocumentReader.
"""

import os
import pytest
from src.document_reader import DocumentReader


@pytest.fixture
def reader():
    return DocumentReader()


def test_read_txt_file(reader, tmp_path):
    txt_file = tmp_path / "test_resume.txt"
    txt_file.write_text("Experienced Python and AWS Engineer\nPhone: 123-456-7890", encoding="utf-8")

    text, metadata = reader.extract_text(str(txt_file))
    assert "Experienced Python and AWS Engineer" in text
    assert metadata["file_type"] == "TXT"
    assert metadata["word_count"] > 0
    assert not metadata["is_empty"]


def test_read_sample_pdf(reader):
    pdf_path = os.path.join("data", "sample_resumes", "resume_alex_chen_ml.pdf")
    if os.path.exists(pdf_path):
        text, meta = reader.extract_text(pdf_path)
        assert len(text) > 100
        assert meta["file_type"] == "PDF"
        assert "Alex Chen" in text or "Machine Learning" in text


def test_empty_input_handling(reader):
    text, meta = reader.extract_text(b"", filename="empty.txt")
    assert text == ""
    assert meta["is_empty"] is True
