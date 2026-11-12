from typing import Any


class NLIVerifier:
    """Classify premise-hypothesis pairs using a pretrained NLI model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
    ):
        if device != "cpu":
            raise ValueError("TrustRAG currently supports CPU NLI inference only.")

        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._label_indices = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        model.to("cpu")
        model.eval()

        id2label = model.config.id2label
        labels = {
            str(label).lower().replace("-", "_"): int(index)
            for index, label in id2label.items()
        }

        required = {"entailment", "contradiction", "neutral"}
        if not required.issubset(labels):
            raise RuntimeError(
                f"Unexpected NLI labels in {self.model_name}: {id2label}"
            )

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._label_indices = {
            name: labels[name] for name in required
        }

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        if not premise.strip() or not hypothesis.strip():
            raise ValueError("Premise and hypothesis cannot be empty.")

        self._load()

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)[0]

        return {
            label: float(probabilities[index].item())
            for label, index in self._label_indices.items()
        }
