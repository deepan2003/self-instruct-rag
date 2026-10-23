from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class GuardrailStatus(str, Enum):
    PASSED = "PASSED"
    BLOCKED_PROMPT_INJECTION = "BLOCKED_PROMPT_INJECTION"
    BLOCKED_OUT_OF_SCOPE = "BLOCKED_OUT_OF_SCOPE"
    BLOCKED_GIBBERISH = "BLOCKED_GIBBERISH"
    BLOCKED_HARMFUL = "BLOCKED_HARMFUL"

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNGROUNDED = "UNGROUNDED"

class SourceCitation(BaseModel):
    page: int = Field(..., description="The exact page number from the research paper (1-23)")
    section: str = Field(..., description="The section name or table/appendix identifier")
    title: str = Field(..., description="Section or chunk title")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Cosine relevance similarity score")
    quote_snippet: str = Field(..., description="Verifiable excerpt from the paper supporting the answer")

class RAGAnswer(BaseModel):
    """
    Dedicated Pydantic structure for strictly grounded RAG outputs to prevent hallucination.
    Used exclusively for grounded RAG responses, containing the factual answer and verified citations.
    """
    answer: str = Field(..., description="Factually grounded answer synthesized from the document context")
    citations: List[SourceCitation] = Field(default_factory=list, description="Verified paper citations with page numbers and quotes")

class GuardrailReport(BaseModel):
    is_safe: bool = Field(..., description="Whether the user input passed all security checks")
    status: GuardrailStatus = Field(default=GuardrailStatus.PASSED, description="Detailed guardrail status")
    passed_injection_check: bool = Field(..., description="Passed adversarial prompt injection check")
    passed_scope_check: bool = Field(..., description="Passed domain scope and relevance check")
    flagged_patterns: List[str] = Field(default_factory=list, description="List of detected suspicious patterns or triggers")
    message: str = Field(..., description="Human-readable safety assessment")

class GroundingEvaluation(BaseModel):
    is_grounded: bool = Field(..., description="Whether the claims are strictly supported by retrieved context")
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="Quantified grounding confidence (0.0 to 1.0)")
    confidence: ConfidenceLevel = Field(..., description="Categorical confidence grade")
    grounded_claims: List[str] = Field(default_factory=list, description="Specific factual claims verified against the source text")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims flagged as ungrounded or speculative")
    verification_notes: str = Field(..., description="Explanation of factual verification against retrieved passages")

class StructuredRAGResponse(BaseModel):
    query: str = Field(..., description="Original user prompt or sanitized query")
    answer: str = Field(..., description="Clear, factual, hallucination-free answer")
    summary: str = Field(..., description="One-sentence executive summary")
    key_findings: List[str] = Field(default_factory=list, description="Bullet points of verified facts")
    sources: List[SourceCitation] = Field(default_factory=list, description="Citations strictly mapped to paper pages and sections")
    guardrails: GuardrailReport = Field(..., description="Pre-retrieval safety and injection validation report")
    grounding: GroundingEvaluation = Field(..., description="Post-generation anti-hallucination verification report")
    latency_ms: float = Field(..., description="Total pipeline processing time in milliseconds")
    status: str = Field(default="SUCCESS", description="Execution status: SUCCESS, CONVERSATIONAL, BLOCKED_GUARDRAILS, or UNSUPPORTED_CONTEXT")

    @property
    def citations(self) -> List[SourceCitation]:
        """Direct access to verified citations to match the 'answer + citations' contract."""
        return self.sources

    def to_rag_answer(self) -> Optional[RAGAnswer]:
        """
        Converts grounded RAG outputs into a strictly typed RAGAnswer (answer + citations).
        Returns None for simple conversation queries or blocked requests.
        """
        if self.status == "CONVERSATIONAL" or not self.sources:
            return None
        return RAGAnswer(answer=self.answer, citations=self.sources)
