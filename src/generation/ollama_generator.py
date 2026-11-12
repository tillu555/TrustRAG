import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.generation.prompt_builder import EvidencePromptBuilder


class OllamaGenerator:
    """Generate evidence-grounded answers using a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        prompt_builder: Any = None,
    ):
        if not model.strip():
            raise ValueError("Model name cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be positive.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.prompt_builder = (
            prompt_builder or EvidencePromptBuilder()
        )

    def generate(
        self,
        query: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:

        prepared = self.prompt_builder.build(query, evidence)

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": prepared["system_prompt"],
                },
                {
                    "role": "user",
                    "content": prepared["user_prompt"],
                },
            ],
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 512,
            },
        }

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as error:
            raise RuntimeError(
                f"Ollama HTTP error: {error.code}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                "Cannot connect to Ollama. Check that Ollama is running."
            ) from error

        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(
                "Ollama returned an invalid response."
            ) from error

        message = result.get("message", {})
        answer = message.get("content", "")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The model returned an empty answer.")

        return {
            "answer": answer.strip(),
            "model": result.get("model", self.model),
            "sources": prepared["sources"],
            "done": result.get("done", False),
        }
