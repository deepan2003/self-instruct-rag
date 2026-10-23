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
from core.tools import ToolRouter, TamilNaduWeatherTool, CodeExecutionTool, TavilySearchTool
from core.models import StructuredRAGResponse, GuardrailReport, GroundingEvaluation, ConfidenceLevel, GuardrailStatus

class RAGPipeline:
    """
    End-to-end Guardrailed, Anti-Hallucinating RAG Pipeline with Tool Support:
    1) Input Guardrails (Prompt Injection, Harmful, Exfiltration & Scope Filter)
    2) Tool Router (Tamil Nadu Live Weather, Python Code Execution, Tavily Live News Search)
    3) Conversational & Memory Handling
    4) Semantic Retrieval & Context-Augmented Generation
    5) Hallucination & Faithfulness Verification
    6) Strongly-typed Pydantic Output with exact Page Sources
    """
    def __init__(
        self,
        embedding_engine: Optional[BaseEmbeddingEngine] = None,
        generator: Optional[RAGGenerator] = None,
        chunk_size: int = 1800,
        chunk_overlap: int = 250
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_engine = embedding_engine or TFIDFEmbeddingEngine()
        self.vector_store = VectorStore(self.embedding_engine)
        self.retriever = Retriever(self.vector_store)
        self.generator = generator or RAGGenerator()
        self.guardrails = InputGuardrails()
        self.hallucination_preventer = HallucinationPreventer()
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
        tavily_api_key: Optional[str] = None
    ) -> StructuredRAGResponse:
        """
        Executes the full protected RAG pipeline, tool router, and anti-hallucination checks.
        """
        start_time = time.time()

        # Step 1: Input Guardrails Check
        guardrail_report = self.guardrails.inspect(query_text)
        if not guardrail_report.is_safe:
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

        # Step 1.2: Tool Router (Live Weather, Code Execution, Tavily Live News)
        detected_tool = ToolRouter.identify_tool(query_text)

        if detected_tool == "weather":
            weather_data = TamilNaduWeatherTool.get_weather(query_text)
            answer_md = TamilNaduWeatherTool.format_markdown(weather_data)
            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query_text,
                answer=answer_md,
                summary=f"Live weather report for {weather_data.get('location', 'Tamil Nadu')}.",
                key_findings=[
                    f"Location: {weather_data.get('location')}, Tamil Nadu",
                    f"Temperature: {weather_data.get('temperature_c')}°C ({weather_data.get('temperature_f')}°F)",
                    f"Condition: {weather_data.get('condition')}",
                    f"Humidity: {weather_data.get('humidity')}%"
                ],
                sources=[],
                guardrails=guardrail_report,
                grounding=GroundingEvaluation(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    confidence=ConfidenceLevel.HIGH,
                    grounded_claims=["Live weather data retrieved from regional weather stations."],
                    unsupported_claims=[],
                    verification_notes="Real-time Tamil Nadu meteorological telemetry."
                ),
                latency_ms=round(latency, 2),
                status="TOOL_WEATHER"
            )

        elif detected_tool == "code_execution":
            code_to_run = query_text
            # Clean common command prefixes
            code_to_run = re.sub(r'^(?:run\s+code:\s*|execute:\s*|python:\s*|run\s+(?:this\s+)?(?:python\s+)?code\s*:?\s*)', '', code_to_run, flags=re.IGNORECASE)
            exec_res = CodeExecutionTool.execute(code_to_run)
            answer_md = CodeExecutionTool.format_markdown(exec_res)
            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query_text,
                answer=answer_md,
                summary=f"Python code execution completed (Exit code: {exec_res.get('exit_code')}).",
                key_findings=[
                    f"Status: {'Succeeded' if exec_res.get('success') else 'Failed'}",
                    f"Execution Time: {exec_res.get('execution_time_ms')} ms",
                    f"Exit code: {exec_res.get('exit_code')}"
                ],
                sources=[],
                guardrails=guardrail_report,
                grounding=GroundingEvaluation(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    confidence=ConfidenceLevel.HIGH,
                    grounded_claims=["Local Python subprocess execution completed."],
                    unsupported_claims=[],
                    verification_notes="Code executed in local sandboxed Python interpreter."
                ),
                latency_ms=round(latency, 2),
                status="TOOL_CODE"
            )

        elif detected_tool == "tavily_search":
            search_res = TavilySearchTool.search(query_text, api_key=tavily_api_key)
            answer_md = TavilySearchTool.format_markdown(search_res)
            latency = (time.time() - start_time) * 1000
            key_findings = [f"Result: {r.get('title')}" for r in search_res.get("results", [])[:3]]
            if not key_findings:
                key_findings = ["Live web search executed."]

            return StructuredRAGResponse(
                query=query_text,
                answer=answer_md,
                summary=search_res.get("answer") or "Tavily live search results.",
                key_findings=key_findings,
                sources=[],
                guardrails=guardrail_report,
                grounding=GroundingEvaluation(
                    is_grounded=search_res.get("success", False),
                    faithfulness_score=1.0 if search_res.get("success") else 0.5,
                    confidence=ConfidenceLevel.HIGH if search_res.get("success") else ConfidenceLevel.MEDIUM,
                    grounded_claims=["Tavily search API intelligence."],
                    unsupported_claims=[],
                    verification_notes="Web intelligence retrieved via Tavily."
                ),
                latency_ms=round(latency, 2),
                status="TOOL_TAVILY"
            )

        # Step 1.5: Conversational & Memory Intent Handling
        intent = ConversationHandler.classify_intent(query_text)
        if intent:
            latency = (time.time() - start_time) * 1000
            return ConversationHandler.generate_response(
                query_text,
                intent,
                latency_ms=latency,
                chat_history=chat_history
            )

        # Optimize search query if user combined a conversational greeting with a substantive question
        search_query = query_text
        substantive = ConversationHandler.extract_substantive_query(query_text)
        if substantive:
            search_query = substantive

        # Step 2: Retrieval
        scored_docs = self.retrieve(search_query, top_k=top_k, threshold=threshold)

        # Step 3: Handle Queries Outside the Paper Conversationally Like a Friend
        if not scored_docs:
            latency = (time.time() - start_time) * 1000
            return ConversationHandler.generate_friendly_chat(
                query_text,
                chat_history=chat_history,
                latency_ms=latency
            )

        # Step 4: Generation
        gen_result = self.generator.generate(query_text, scored_docs)
        sources = gen_result.get("sources", [])

        # Step 5: Hallucination & Faithfulness Verification
        extracted_claims = self.hallucination_preventer.extract_claims(gen_result["answer"])
        grounding_eval = self.hallucination_preventer.verify_grounding(extracted_claims, scored_docs)

        latency = (time.time() - start_time) * 1000

        return StructuredRAGResponse(
            query=query_text,
            answer=gen_result["answer"],
            summary=gen_result.get("summary", ""),
            key_findings=gen_result.get("key_findings", []),
            sources=sources,
            guardrails=guardrail_report,
            grounding=grounding_eval,
            latency_ms=round(latency, 2),
            status="SUCCESS"
        )

    def save_index(self, path: str):
        self.vector_store.save(path)

    def load_index(self, path: str):
        self.vector_store = VectorStore.load(path)
        self.retriever = Retriever(self.vector_store)
        self.is_indexed = True
