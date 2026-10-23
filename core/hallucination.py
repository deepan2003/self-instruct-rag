from typing import List, Tuple, Dict, Any
import re
from core.models import GroundingEvaluation, ConfidenceLevel, SourceCitation
from core.document import Document

class HallucinationPreventer:
    """
    Evaluates factual grounding and prevents hallucinations:
    1) Verifies claim support against retrieved passages.
    2) Verifies numerical consistency (ensures numbers in claims match context).
    3) Computes a quantitative faithfulness score.
    4) Flags ungrounded or speculative claims.
    """

    def __init__(self, minimum_faithfulness_threshold: float = 0.60):
        self.minimum_threshold = minimum_faithfulness_threshold

    def extract_claims(self, text: str) -> List[str]:
        """Extracts individual factual statements from generated answer."""
        lines = text.split("\n")
        claims = []
        for line in lines:
            line_clean = line.strip()
            # Ignore headers or markdown decoration
            if not line_clean or line_clean.startswith("###") or line_clean.startswith("==="):
                continue
            # Strip bullet markdown
            clean_claim = re.sub(r'^[-*•\d\.\)\s]+', '', line_clean).strip()
            # Remove existing citation tags if already present
            clean_claim = re.sub(r'\[Page \d+.*?\]:?', '', clean_claim).strip()
            if len(clean_claim) > 15:
                claims.append(clean_claim)
        return claims

    def verify_grounding(
        self,
        claims: List[str],
        retrieved_docs: List[Tuple[Document, float]]
    ) -> GroundingEvaluation:
        if not retrieved_docs or not claims:
            return GroundingEvaluation(
                is_grounded=False,
                faithfulness_score=0.0,
                confidence=ConfidenceLevel.UNGROUNDED,
                grounded_claims=[],
                unsupported_claims=claims,
                verification_notes="No supporting context retrieved from the paper to ground these claims."
            )

        # Build full combined context text for verification
        combined_context = " ".join([doc.page_content for doc, _ in retrieved_docs]).lower()
        context_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?%?\b', combined_context))

        grounded = []
        unsupported = []

        for claim in claims:
            claim_lower = claim.lower()
            claim_words = set(re.findall(r'\w{4,}', claim_lower)) - {"about", "which", "there", "these", "their", "after", "before", "using", "paper", "model"}

            # Check word overlap with context
            overlap_words = [w for w in claim_words if w in combined_context]
            overlap_ratio = len(overlap_words) / max(len(claim_words), 1)

            # Check numbers in claim: if numbers are mentioned, they must exist in the context!
            claim_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?%?\b', claim_lower))
            numbers_valid = True
            if claim_numbers:
                numbers_valid = claim_numbers.issubset(context_numbers)

            # Grounded if lexical overlap is strong and all referenced numbers exist in source text
            if overlap_ratio >= 0.50 and numbers_valid:
                grounded.append(claim)
            else:
                unsupported.append(claim)

        total_claims = len(claims)
        faithfulness = len(grounded) / max(total_claims, 1)
        is_grounded = faithfulness >= self.minimum_threshold

        if faithfulness >= 0.85:
            conf = ConfidenceLevel.HIGH
        elif faithfulness >= 0.60:
            conf = ConfidenceLevel.MEDIUM
        elif faithfulness > 0.0:
            conf = ConfidenceLevel.LOW
        else:
            conf = ConfidenceLevel.UNGROUNDED

        notes = (
            f"Verified {len(grounded)}/{total_claims} claims directly against retrieved paper text. "
            f"Faithfulness score: {faithfulness * 100:.1f}%."
        )
        if unsupported:
            notes += f" Flagged {len(unsupported)} ungrounded or weak claims to prevent hallucination."

        return GroundingEvaluation(
            is_grounded=is_grounded,
            faithfulness_score=round(faithfulness, 4),
            confidence=conf,
            grounded_claims=grounded,
            unsupported_claims=unsupported,
            verification_notes=notes
        )
