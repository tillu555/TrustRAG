
import fitz
import pytest

from src.ingestion.pdf_loader import PDFLoader


def test_pdf_loader_extracts_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "TrustRAG scientific document extraction test."
    )
    document.save(pdf_path)
    document.close()

    result = PDFLoader().load(pdf_path)

    assert result["page_count"] == 1
    assert result["word_count"] > 0
    assert "TrustRAG" in result["full_text"]
    assert result["pages"][0]["page_number"] == 1


def test_pdf_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        PDFLoader().load("missing.pdf")


def test_pdf_loader_rejects_non_pdf(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF")

    with pytest.raises(ValueError):
        PDFLoader().load(file_path)
