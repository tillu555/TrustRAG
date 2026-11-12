from src.verification.claim_extractor import ClaimExtractor
from src.verification.evidence_matcher import EvidenceMatcher
from src.verification.verification_pipeline import ClaimVerificationPipeline


class FakeNLI:
    def predict(self, premise, hypothesis):
        if "RAG retrieves evidence" in premise:
            return {
                "entailment": 0.91,
                "contradiction": 0.02,
                "neutral": 0.07,
            }

        return {
            "entailment": 0.02,
            "contradiction": 0.03,
            "neutral": 0.95,
        }


def test_claim_extractor_extracts_claims_and_citations():
    claims = ClaimExtractor().extract(
        "RAG retrieves evidence before generation [S1]. "
        "It uses external documents to support answers [S2]."
    )

    assert len(claims) == 2
    assert claims[0]["cited_sources"] == ["S1"]
    assert claims[1]["cited_sources"] == ["S2"]


def test_evidence_matcher_preserves_source_ids():
    evidence = [{"document": {"text": "RAG retrieves evidence."}}]
    sources = [{"source_id": "S1", "document_name": "paper.pdf"}]

    result = EvidenceMatcher().prepare(evidence, sources)

    assert result[0]["source_id"] == "S1"
    assert result[0]["text"] == "RAG retrieves evidence."


def test_verification_marks_supported_claim():
    verifier = ClaimVerificationPipeline(nli_verifier=FakeNLI())

    result = verifier.verify(
        answer="RAG retrieves evidence before generation [S1].",
        evidence=[{"document": {"text": "RAG retrieves evidence before generation."}}],
        sources=[{"source_id": "S1"}],
    )

    assert result["verification_status"] == "verified"
    assert result["claims"][0]["status"] == "SUPPORTED"
    assert result["verified_claims"] == 1


def test_verification_does_not_overstate_neutral_evidence():
    verifier = ClaimVerificationPipeline(nli_verifier=FakeNLI())

    result = verifier.verify(
        answer="A completely unrelated scientific claim is presented here.",
        evidence=[{"document": {"text": "Unrelated passage about another subject."}}],
        sources=[{"source_id": "S1"}],
    )

    assert result["verification_status"] == "insufficient_evidence"
    assert result["claims"][0]["status"] == "NOT ENOUGH EVIDENCE"
