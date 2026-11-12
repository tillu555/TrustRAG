
import re
from typing import Any


class SectionParser:
    """Detect scientific headings embedded in PDF text lines."""

    SECTION_NAMES = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "related work": "Related Work",
        "background": "Background",
        "literature review": "Literature Review",
        "methodology": "Methodology",
        "methods": "Methods",
        "materials and methods": "Materials and Methods",
        "experimental setup": "Experimental Setup",
        "experiments": "Experiments",
        "results": "Results",
        "discussion": "Discussion",
        "results and discussion": "Results and Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "future work": "Future Work",
        "limitations": "Limitations",
        "acknowledgements": "Acknowledgements",
        "acknowledgments": "Acknowledgements",
        "references": "References",
        "bibliography": "References",
        "fog computing architecture": "Architecture",
        
    }

    ROMAN_NUMERAL = r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)"

    def _identify_heading(self, line: str) -> str | None:
        """Identify a heading when it occupies the entire line."""
        cleaned = line.strip()

        if not cleaned or len(cleaned) > 150:
            return None

        cleaned = re.sub(
            rf"^(?:\d+(?:\.\d+)*|{self.ROMAN_NUMERAL})[.)]?\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = cleaned.strip(" .:-\t—").lower()
        return self.SECTION_NAMES.get(cleaned)

    def _split_heading_line(
        self,
        line: str,
    ) -> tuple[str, str | None, str]:
        """Split a line into text before, heading, and text after."""

        # Abstract is commonly formatted as "Abstract—..."
        abstract_match = re.match(
            r"^\s*abstract\s*[—–-]\s*",
            line,
            flags=re.IGNORECASE,
        )
        if abstract_match:
            return (
                "",
                "Abstract",
                line[abstract_match.end():].strip(),
            )

        names = sorted(
            self.SECTION_NAMES.keys(),
            key=len,
            reverse=True,
        )

        alternatives = "|".join(
            re.escape(name) for name in names
        )

        # Detect numbered/Roman-numeral headings even when they
        # occur after text from another PDF column.
        numbered_pattern = re.compile(
            rf"(?<![A-Za-z])"
            rf"(?:{self.ROMAN_NUMERAL}|\d+(?:\.\d+)*)"
            rf"\s*[.)]\s*"
            rf"({alternatives})"
            rf"(?=\s|$)",
            flags=re.IGNORECASE,
        )

        match = numbered_pattern.search(line)

        if match:
            heading = self.SECTION_NAMES[
                match.group(1).lower()
            ]
            prefix = line[:match.start()].strip()
            suffix = line[match.end():].strip()
            return prefix, heading, suffix

        # Detect standalone headings at the beginning of a line.
        plain_pattern = re.compile(
            rf"^\s*({alternatives})\s*(?:[.:—–-]\s*)?$",
            flags=re.IGNORECASE,
        )

        match = plain_pattern.match(line)

        if match:
            heading = self.SECTION_NAMES[
                match.group(1).lower()
            ]
            return "", heading, ""

        return line, None, ""

    def parse(self, document: dict[str, Any]) -> dict[str, Any]:
        sections = []

        current_section = {
            "title": "Front Matter",
            "text": "",
            "page_numbers": [],
            "page_texts": [],
        }

        def add_text(text: str, page_number: int) -> None:
            if not text.strip():
                return

            current_section["text"] += text + "\n"

            if page_number not in current_section["page_numbers"]:
                current_section["page_numbers"].append(page_number)

            if (
                not current_section["page_texts"]
                or current_section["page_texts"][-1]["page_number"]
                != page_number
            ):
                current_section["page_texts"].append({
                    "page_number": page_number,
                    "text": text,
                })
            else:
                current_section["page_texts"][-1]["text"] += "\n" + text

        def save_current() -> None:
            text = current_section["text"].strip()

            if text:
                sections.append({
                    "title": current_section["title"],
                    "text": text,
                    "page_numbers": sorted(
                        set(current_section["page_numbers"])
                    ),
                    "page_texts": [
                        item.copy()
                        for item in current_section["page_texts"]
                        if item["text"].strip()
                    ],
                })

        for page in document["pages"]:
            page_number = page["page_number"]

            for line in page["text"].splitlines():
                prefix, heading, suffix = self._split_heading_line(line)

                if heading:
                    # Preserve any text before an embedded heading.
                    add_text(prefix, page_number)
                    save_current()

                    current_section = {
                        "title": heading,
                        "text": "",
                        "page_numbers": [],
                        "page_texts": [],
                    }

                    # Preserve text following the heading.
                    add_text(suffix, page_number)
                else:
                    add_text(prefix, page_number)

        save_current()

        return {
            "file_name": document["file_name"],
            "sections": sections,
            "section_count": len(sections),
        }


if __name__ == "__main__":
    print("TrustRAG Section Parser is ready.")
