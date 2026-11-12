from typing import Any

from src.verification.claim_extractor import ClaimExtractor
from src.verification.evidence_matcher import EvidenceMatcher
from src.verification.nli_verifier import NLIVerifier


class ClaimVerificationPipeline:
    """Verify generated claims against retrieved scientific passages."""

    def __init__(
        self,
        nli_verifier: Any = None,
        entailment_threshold: float = 0.65,
        contradiction_threshold: float = 0.65,
    ):
        if not 0 < entailment_threshold <= 1:
            raise ValueError("Invalid entailment threshold.")
        if not 0 < contradiction_threshold <= 1:
            raise ValueError("Invalid contradiction threshold.")

        self.claim_extractor = ClaimExtractor()
        self.evidence_matcher = EvidenceMatcher()
        self.nli_verifier = nli_verifier or NLIVerifier()
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def verify(
        self,
        answer: str,
        evidence: list[dict],
        sources: list[dict],
    ) -> dict:
        claims = self.claim_extractor.extract(answer)
        passages = self.evidence_matcher.prepare(evidence, sources)

        if not claims or not passages:
            return {
                "verification_status": "insufficient_evidence",
                "claims": claims,
                "verified_claims": 0,
                "total_claims": len(claims),
            }

        results = []

        for claim in claims:
            cited = set(claim["cited_sources"])
            candidates = [
                passage for passage in passages
                if not cited or passage["source_id"] in cited
            ]

            if not candidates:
                candidates = passages

            best_entailment = None
            best_contradiction = None

            for passage in candidates:
                scores = self.nli_verifier.predict(
                    premise=passage["text"],
                    hypothesis=claim["text"],
                )

                record = {
                    "source_id": passage["source_id"],
                    "document_name": passage["document_name"],
                    "section": passage["section"],
                    "page_numbers": passage["page_numbers"],
                    "entailment": scores["entailment"],
                    "contradiction": scores["contradiction"],
                    "neutral": scores["neutral"],
                }

                if (
                    best_entailment is None
                    or record["entailment"] > best_entailment["entailment"]
                ):
                    best_entailment = record

                if (
                    best_contradiction is None
                    or record["contradiction"] > best_contradiction["contradiction"]
                ):
                    best_contradiction = record

            entailment = best_entailment["entailment"]
            contradiction = best_contradiction["contradiction"]

            if (
                entailment >= self.entailment_threshold
                and entailment > contradiction
            ):
                status = "SUPPORTED"
                supporting_sources = [best_entailment]
            elif (
                contradiction >= self.contradiction_threshold
                and contradiction > entailment
            ):
                status = "CONTRADICTED"
                supporting_sources = [best_contradiction]
            else:
                status = "NOT ENOUGH EVIDENCE"
                supporting_sources = []

            results.append({
                **claim,
                "status": status,
                "supporting_evidence": supporting_sources,
                "entailment_score": entailment,
                "contradiction_score": contradiction,
            })

        statuses = [item["status"] for item in results]

        if all(status == "SUPPORTED" for status in statuses):
            overall = "verified"
        elif "CONTRADICTED" in statuses:
            overall = "contradicted"
        elif "SUPPORTED" in statuses:
            overall = "partially_supported"
        else:
            overall = "insufficient_evidence"

        return {
            "verification_status": overall,
            "claims": results,
            "verified_claims": statuses.count("SUPPORTED"),
            "total_claims": len(results),
        }
