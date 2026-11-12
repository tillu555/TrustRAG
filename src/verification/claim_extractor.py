import re


class ClaimExtractor:
    """Extract sentence-level claims from a generated answer."""

    def extract(self, answer: str) -> list[dict]:
        if not isinstance(answer, str) or not answer.strip():
            return []

        parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
        claims = []

        for part in parts:
            text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip()
            if len(text.split()) < 4:
                continue

            citations = list(dict.fromkeys(
                re.findall(r"\[(S\d+)\]", text)
            ))

            claims.append({
                "claim_id": f"C{len(claims) + 1}",
                "text": text,
                "cited_sources": citations,
            })

        return claims
