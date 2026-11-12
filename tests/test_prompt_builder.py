import pytest

from src.generation.prompt_builder import EvidencePromptBuilder


@pytest.fixture
def evidence():
    return [
        {
            "document": {
                "chunk_id": "chunk_001",
                "document_name": "research.pdf",
                "section": "Methodology",
                "page_numbers": [5, 6],
                "text": "The proposed system combines dense and sparse retrieval.",
            }
        }
    ]


def test_prompt_contains_research_question(evidence):
    result = EvidencePromptBuilder().build(
        "How does the system retrieve evidence?",
        evidence,
    )

    assert "How does the system retrieve evidence?" in result["user_prompt"]


def test_prompt_contains_source_citation(evidence):
    result = EvidencePromptBuilder().build("Explain retrieval.", evidence)

    assert "[S1]" in result["user_prompt"]
    assert result["sources"][0]["source_id"] == "S1"


def test_prompt_preserves_page_metadata(evidence):
    result = EvidencePromptBuilder().build("Explain retrieval.", evidence)

    assert result["sources"][0]["page_numbers"] == [5, 6]


def test_prompt_rejects_empty_query(evidence):
    with pytest.raises(ValueError):
        EvidencePromptBuilder().build("", evidence)


def test_prompt_rejects_empty_evidence():
    with pytest.raises(ValueError):
        EvidencePromptBuilder().build("Explain retrieval.", [])
