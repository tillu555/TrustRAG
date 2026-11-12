from src.ingestion.section_parser import SectionParser


def test_section_parser_identifies_sections():
    document = {
        "file_name": "research.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "ABSTRACT\n"
                    "This paper studies reliable AI.\n"
                    "1. INTRODUCTION\n"
                    "We investigate evidence retrieval.\n"
                    "2. METHODOLOGY\n"
                    "We use hybrid retrieval.\n"
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "3. RESULTS\n"
                    "Our evaluation measures reliability.\n"
                    "4. CONCLUSION\n"
                    "We discuss future research.\n"
                ),
            },
        ],
    }

    result = SectionParser().parse(document)

    titles = [section["title"] for section in result["sections"]]

    assert "Abstract" in titles
    assert "Introduction" in titles
    assert "Methodology" in titles
    assert "Results" in titles
    assert "Conclusion" in titles


def test_section_parser_preserves_page_numbers():
    document = {
        "file_name": "research.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "ABSTRACT\nFirst page content.",
            },
            {
                "page_number": 2,
                "text": "Second page content.",
            },
        ],
    }

    result = SectionParser().parse(document)

    abstract = result["sections"][0]

    assert abstract["title"] == "Abstract"
    assert abstract["page_numbers"] == [1, 2]


def test_section_parser_handles_empty_document():
    document = {
        "file_name": "empty.pdf",
        "pages": [],
    }

    result = SectionParser().parse(document)

    assert result["section_count"] == 0
