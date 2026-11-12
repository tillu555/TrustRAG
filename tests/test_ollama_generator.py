import json

import pytest

from src.generation.ollama_generator import OllamaGenerator


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.data).encode("utf-8")


@pytest.fixture
def evidence():
    return [
        {
            "document": {
                "chunk_id": "chunk_001",
                "document_name": "paper.pdf",
                "section": "Results",
                "page_numbers": [8],
                "text": "The system achieved improved retrieval performance.",
            }
        }
    ]


def test_generator_configuration():
    generator = OllamaGenerator()

    assert generator.model == "qwen2.5:3b"
    assert generator.base_url == "http://localhost:11434"


def test_generator_returns_answer(monkeypatch, evidence):
    import src.generation.ollama_generator as module

    def mock_urlopen(request, timeout):
        return FakeResponse({
            "model": "qwen2.5:3b",
            "message": {
                "content": "The system improves retrieval [S1]."
            },
            "done": True,
        })

    monkeypatch.setattr(module, "urlopen", mock_urlopen)

    generator = OllamaGenerator()
    result = generator.generate("What improved?", evidence)

    assert result["answer"] == "The system improves retrieval [S1]."
    assert result["sources"][0]["source_id"] == "S1"
    assert result["done"] is True


def test_generator_rejects_empty_evidence():
    generator = OllamaGenerator()

    with pytest.raises(ValueError):
        generator.generate("Explain the paper.", [])


def test_generator_rejects_empty_model():
    with pytest.raises(ValueError):
        OllamaGenerator(model="")


def test_generator_rejects_empty_model_response(monkeypatch, evidence):
    import src.generation.ollama_generator as module

    def mock_urlopen(request, timeout):
        return FakeResponse({
            "message": {"content": ""}
        })

    monkeypatch.setattr(module, "urlopen", mock_urlopen)

    with pytest.raises(RuntimeError):
        OllamaGenerator().generate("Explain results.", evidence)
