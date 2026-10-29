from typing import List, Dict, Any, Optional, Tuple
import time
import re
from core.document import Document, load_documents_from_json
from core.chunker import TextChunker
from core.embeddings import TFIDFEmbeddingEngine, BaseEmbeddingEngine
from core.vector_store import VectorStore
from core.retriever import Retriever
from core.generator import RAGGenerator
from core.guardrails import InputGuardrails
from core.hallucination import HallucinationPreventer
from core.conversation import ConversationHandler
from core.tools import ToolRouter, TamilNaduWeatherTool, CodeExecutionTool, TavilySearchTool, GmailDraftTool
from core.agent import ReActAgent
from core.models import StructuredRAGResponse, GuardrailReport, GroundingEvaluation, ConfidenceLevel, GuardrailStatus

class RAGPipeline:
    """
    End-to-end Guardrailed, Anti-Hallucinating RAG Pipeline with Main ReAct Agent:
    1) Input Guardrails (Prompt Injection, Harmful, Exfiltration & Scope Filter)
    2) Main Agent ReAct Loop (openai/gpt-oss-120b on Groq)
       - Reasons which sub-agent / tool to invoke
       - Executes: rag_search, tamil_nadu_weather, python_code_runner, tavily_live_news
       - Synthesizes final grounded response
    3) Anti-Hallucination & Faithfulness Verification
    4) Strongly-typed Pydantic Output with exact Page Sources
    """
    def __init__(
        self,
        embedding_engine: Optional[BaseEmbeddingEngine] = None,
        generator: Optional[RAGGenerator] = None,
        chunk_size: int = 1800,
        chunk_overlap: int = 250,
        groq_api_key: Optional[str] = None
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_engine = embedding_engine or TFIDFEmbeddingEngine()
        self.vector_store = VectorStore(self.embedding_engine)
        self.retriever = Retriever(self.vector_store)
        self.generator = generator or RAGGenerator()
        self.guardrails = InputGuardrails()
        self.hallucination_preventer = HallucinationPreventer()
        self.agent = ReActAgent(api_key=groq_api_key)
        self.is_indexed = False

    def ingest_json(self, json_path: str) -> Dict[str, int]:
        """Loads, chunks, and indexes documents from a JSON file."""
        raw_docs = load_documents_from_json(json_path)
        chunks = self.chunker.split_documents(raw_docs)
        self.vector_store.add_documents(chunks)
        self.is_indexed = True
        return {
            "raw_documents": len(raw_docs),
            "chunks_created": len(chunks)
        }

    def retrieve(self, query: str, top_k: int = 4, threshold: float = 0.10) -> List[Tuple[Document, float]]:
        """Retrieves top scoring chunks for a query."""
        if not self.is_indexed:
            raise ValueError("Pipeline has not indexed any documents. Call ingest_json first.")
        return self.retriever.retrieve(query, top_k=top_k, threshold=threshold)

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        threshold: float = 0.10,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        tavily_api_key: Optional[str] = None,
        file_data: Optional[Dict[str, Any]] = None
    ) -> StructuredRAGResponse:
        """
        Executes the full protected RAG pipeline through the Main ReAct Agent.
        """
        start_time = time.time()

        if file_data and not query_text.strip():
            query_text = "Please analyze and summarize this uploaded document / image."

        # Step 1: Input Guardrails Check
        guardrail_report = self.guardrails.inspect(query_text)
        if not guardrail_report.is_safe and not file_data:
            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query_text,
                answer=f"🛡️ **Request Blocked by Security Guardrails**: {guardrail_report.message}",
                summary="Query rejected due to guardrail security violation.",
                key_findings=[],
                sources=[],
                guardrails=guardrail_report,
                grounding=GroundingEvaluation(
                    is_grounded=False,
                    faithfulness_score=0.0,
                    confidence=ConfidenceLevel.UNGROUNDED,
                    grounded_claims=[],
                    unsupported_claims=[],
                    verification_notes="Processing terminated early by safety guardrails."
                ),
                latency_ms=round(latency, 2),
                status="BLOCKED_GUARDRAILS"
            )

        # Step 1.1: Conversational Memory Intent Handling
        intent = ConversationHandler.classify_intent(query_text)
        if intent == "memory" and not file_data:
            latency = (time.time() - start_time) * 1000
            return ConversationHandler.generate_response(
                query_text,
                intent,
                latency_ms=latency,
                chat_history=chat_history
            )

        # Step 2: Main ReAct Agent Loop (Thought ➔ Action ➔ Observation ➔ Synthesis)
        return self.agent.run(
            query=query_text,
            rag_pipeline=self,
            chat_history=chat_history,
            top_k=top_k,
            threshold=threshold,
            tavily_api_key=tavily_api_key,
            file_data=file_data
        )

    def save_index(self, path: str):
        self.vector_store.save(path)

    def load_index(self, path: str):
        self.vector_store = VectorStore.load(path)
        self.retriever = Retriever(self.vector_store)
        self.is_indexed = True
