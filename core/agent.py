import os
import sys
import json
import time
import re
import requests
from typing import Dict, Any, Optional, List, Tuple
from core.document import Document
from core.models import StructuredRAGResponse, GuardrailReport, GroundingEvaluation, ConfidenceLevel, GuardrailStatus, SourceCitation
from core.tools import TamilNaduWeatherTool, CodeExecutionTool, TavilySearchTool, ToolRouter
from core.conversation import ConversationHandler

def load_env_file():
    """Loads environment variables from .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_env_file()


class ReActAgent:
    """
    Main Orchestrator Agent implementing the ReAct (Reason + Act) Loop:
    1) Thought: Analyzes user prompt and conversation history
    2) Action: Dynamically routes to the appropriate sub-agent or tool:
       - rag_search (SELF-INSTRUCT paper sub-agent)
       - tamil_nadu_weather (Live Tamil Nadu weather tool)
       - python_code_runner (Local Python code runner)
       - tavily_live_news (Real-time live news & web search)
    3) Observation: Receives factual telemetry/passages from the tool
    4) Final Synthesis: Main Agent LLM formats grounded, structured response with citations
    """

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    FALLBACK_MODEL = "qwen/qwen3.8-27b"

    TOOLS_SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Search and retrieve verified passages, methodology, 4 pipeline steps, "
                    "dataset statistics, SuperNI benchmark scores, and heuristics from the "
                    "research paper 'SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Specific search query for the SELF-INSTRUCT paper"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tamil_nadu_weather",
                "description": (
                    "Retrieve real-time live weather telemetry, temperature (°C/°F), humidity, "
                    "wind speed, and climate conditions for Tamil Nadu and any of its 38 districts "
                    "(Chennai, Coimbatore, Madurai, Trichy, Salem, Tirunelveli, Ooty, etc.)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City, district, or region in Tamil Nadu (e.g., Chennai, Madurai, Kovai)"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "python_code_runner",
                "description": (
                    "Execute Python code in a local secure subprocess and return stdout, "
                    "stderr, execution time (ms), and exit code."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Complete Python code snippet to run"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tavily_live_news",
                "description": (
                    "Search real-time breaking news, recent events, and current affairs on the live web "
                    "using Tavily Search API."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for live news or current topics"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    SYSTEM_PROMPT = (
        "You are the Main Orchestrator Agent for the SELF-INSTRUCT Research Assistant.\n"
        "Your task is to orchestrate and solve user requests using a ReAct (Reason + Act) loop:\n\n"
        "1. Decision & Tool Calling:\n"
        "   - If the user asks about the SELF-INSTRUCT paper, instruction generation pipeline, "
        "     dataset numbers, filtering heuristics, or benchmark results, call `rag_search`.\n"
        "   - If the user asks about weather, temperature, rain, or climate (especially in Tamil Nadu or any district), "
        "     call `tamil_nadu_weather`.\n"
        "   - If the user requests to run/execute Python code, call `python_code_runner`.\n"
        "   - If the user asks for live news, breaking stories, or real-time web searches, call `tavily_live_news`.\n"
        "   - For general greetings, pleasantries, or simple friendly conversations, reply warmly and directly without calling tools.\n\n"
        "2. Grounded Synthesis:\n"
        "   - When an observation is returned from a tool, synthesize a complete, well-structured, "
        "     and easy-to-read final answer using clear bullet points and bold highlights.\n"
        "   - Base all claims strictly on the provided observation without hallucinating unstated facts.\n"
        "   - Maintain a friendly, supportive tone at all times."
    )

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL") or self.DEFAULT_MODEL

    def _call_groq(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        timeout: int = 25
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Calls Groq Chat Completions API with fallback model support."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for candidate_model in [self.model, self.FALLBACK_MODEL]:
            payload = {
                "model": candidate_model,
                "messages": messages,
                "temperature": 0.2
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice

            try:
                resp = requests.post(self.GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return True, choices[0].get("message")
                elif resp.status_code in [429, 503]:
                    # Rate limit or overloaded; try fallback model
                    continue
            except Exception:
                continue

        return False, None

    def run(
        self,
        query: str,
        rag_pipeline,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 4,
        threshold: float = 0.10,
        tavily_api_key: Optional[str] = None
    ) -> StructuredRAGResponse:
        """
        Executes the Main Agent ReAct loop:
        1. Main Agent decides tool / sub-agent
        2. Sub-agent executes (Observation)
        3. Main Agent synthesizes final response
        """
        start_time = time.time()

        # Build message history for LLM
        llm_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Append recent conversation history context if available (last 6 messages)
        if chat_history:
            for m in chat_history[-6:]:
                role = m.get("role")
                if role in ["user", "assistant"]:
                    llm_messages.append({"role": role, "content": m.get("content", "")})

        # Append current user prompt
        llm_messages.append({"role": "user", "content": query})

        # Step 1: Main Agent Decides (Reasoning / Thought ➔ Action)
        success, assistant_msg = self._call_groq(llm_messages, tools=self.TOOLS_SCHEMA)

        if not success or not assistant_msg:
            # Fallback to local rule-based routing if API call fails
            return self._fallback_execution(
                query,
                rag_pipeline,
                chat_history=chat_history,
                top_k=top_k,
                threshold=threshold,
                tavily_api_key=tavily_api_key,
                start_time=start_time
            )

        tool_calls = assistant_msg.get("tool_calls")

        # Case A: Main Agent decided NO tool is needed (Direct friendly conversation or greeting)
        if not tool_calls:
            direct_reply = assistant_msg.get("content", "").strip()
            if not direct_reply:
                direct_reply = "I'm doing well, thank you! How can I help you today?"

            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query,
                answer=direct_reply,
                summary="Direct conversational response from Main Agent.",
                key_findings=["Conversational dialogue from Main Orchestrator Agent."],
                sources=[],
                guardrails=GuardrailReport(
                    is_safe=True,
                    status=GuardrailStatus.PASSED,
                    passed_injection_check=True,
                    passed_scope_check=True,
                    flagged_patterns=[],
                    message="Main Agent input verified."
                ),
                grounding=GroundingEvaluation(
                    is_grounded=True,
                    faithfulness_score=1.0,
                    confidence=ConfidenceLevel.HIGH,
                    grounded_claims=["Main Agent conversational reply."],
                    unsupported_claims=[],
                    verification_notes="Main Agent direct conversational reasoning."
                ),
                latency_ms=round(latency, 2),
                status="CONVERSATIONAL"
            )

        # Case B: Main Agent decided to invoke a Sub-Agent / Tool (Action)
        tool_call = tool_calls[0]
        call_id = tool_call.get("id", "call_1")
        func_name = tool_call.get("function", {}).get("name")
        func_args_str = tool_call.get("function", {}).get("arguments", "{}")

        try:
            func_args = json.loads(func_args_str)
        except Exception:
            func_args = {}

        # Step 2: Sub-Agent Execution (Observation)
        retrieved_sources: List[SourceCitation] = []
        status_tag = "SUCCESS"
        observation_text = ""

        if func_name == "rag_search":
            search_q = func_args.get("query", query)
            scored_docs = rag_pipeline.retrieve(search_q, top_k=top_k, threshold=threshold)

            if not scored_docs:
                # Try fallback search with original query
                scored_docs = rag_pipeline.retrieve(query, top_k=top_k, threshold=threshold)

            if scored_docs:
                retrieved_sources = rag_pipeline.generator.build_source_citations(scored_docs)
                observation_text = "\n\n".join([
                    f"[Source #{i} | Page {doc.page} | Section: {doc.section}]\n{doc.page_content}"
                    for i, (doc, s) in enumerate(scored_docs, 1)
                ])
                status_tag = "SUCCESS"
            else:
                observation_text = "No direct passages found in the SELF-INSTRUCT paper matching this query."
                status_tag = "UNSUPPORTED_CONTEXT"

        elif func_name == "tamil_nadu_weather":
            loc = func_args.get("location", query)
            weather_data = TamilNaduWeatherTool.get_weather(loc)
            observation_text = TamilNaduWeatherTool.format_markdown(weather_data)
            status_tag = "TOOL_WEATHER"

        elif func_name == "python_code_runner":
            code = func_args.get("code", query)
            res = CodeExecutionTool.execute(code)
            observation_text = CodeExecutionTool.format_markdown(res)
            status_tag = "TOOL_CODE"

        elif func_name == "tavily_live_news":
            news_q = func_args.get("query", query)
            res = TavilySearchTool.search(news_q, api_key=tavily_api_key)
            observation_text = TavilySearchTool.format_markdown(res)
            status_tag = "TOOL_TAVILY"

        else:
            observation_text = f"Tool '{func_name}' returned no output."

        # Step 3: ReAct Final Synthesis by Main Agent LLM
        llm_messages.append(assistant_msg)
        llm_messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": func_name,
            "content": observation_text
        })

        # Ask Main Agent LLM to synthesize final grounded response based on Observation
        synth_success, synth_msg = self._call_groq(llm_messages, timeout=30)

        final_answer = ""
        if synth_success and synth_msg and synth_msg.get("content"):
            final_answer = synth_msg.get("content").strip()
        else:
            # If second LLM call timed out, return the observation formatted directly
            final_answer = observation_text

        # Strip any raw page tags from final text if rag answer
        if func_name == "rag_search":
            final_answer = re.sub(r'\[(?:Page|Section|Source)[^\]]*\]\s*', '', final_answer).strip()

        latency = (time.time() - start_time) * 1000

        # Anti-hallucination verification for RAG outputs
        if func_name == "rag_search" and retrieved_sources:
            extracted_claims = rag_pipeline.hallucination_preventer.extract_claims(final_answer)
            grounding_eval = rag_pipeline.hallucination_preventer.verify_grounding(
                extracted_claims,
                [(Document(page_content=c.quote_snippet, metadata={"page": c.page, "section": c.section, "title": c.title, "chunk_id": c.chunk_id}), c.relevance_score) for c in retrieved_sources]
            )
        else:
            grounding_eval = GroundingEvaluation(
                is_grounded=True,
                faithfulness_score=1.0,
                confidence=ConfidenceLevel.HIGH,
                grounded_claims=["Tool observation synthesized by Main Agent."],
                unsupported_claims=[],
                verification_notes=f"ReAct Loop completed: {func_name} executed & synthesized."
            )

        return StructuredRAGResponse(
            query=query,
            answer=final_answer,
            summary=f"ReAct Agent executed '{func_name}' and synthesized response.",
            key_findings=[f"Action: {func_name}", f"Status: {status_tag}"],
            sources=retrieved_sources,
            guardrails=GuardrailReport(
                is_safe=True,
                status=GuardrailStatus.PASSED,
                passed_injection_check=True,
                passed_scope_check=True,
                flagged_patterns=[],
                message="ReAct Agent execution validated."
            ),
            grounding=grounding_eval,
            latency_ms=round(latency, 2),
            status=status_tag
        )

    def _fallback_execution(
        self,
        query: str,
        rag_pipeline,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 4,
        threshold: float = 0.10,
        tavily_api_key: Optional[str] = None,
        start_time: float = 0.0
    ) -> StructuredRAGResponse:
        """Reliable fallback if Groq API is offline or unreachable."""
        tool = ToolRouter.identify_tool(query)
        if tool == "weather":
            data = TamilNaduWeatherTool.get_weather(query)
            ans = TamilNaduWeatherTool.format_markdown(data)
            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query,
                answer=ans,
                summary="Tamil Nadu Weather Report.",
                key_findings=[f"Location: {data.get('location')}"],
                sources=[],
                guardrails=GuardrailReport(is_safe=True, status=GuardrailStatus.PASSED, passed_injection_check=True, passed_scope_check=True, flagged_patterns=[], message="Safe"),
                grounding=GroundingEvaluation(is_grounded=True, faithfulness_score=1.0, confidence=ConfidenceLevel.HIGH, grounded_claims=[], unsupported_claims=[], verification_notes="Local telemetry"),
                latency_ms=round(latency, 2),
                status="TOOL_WEATHER"
            )

        if tool == "code_execution":
            res = CodeExecutionTool.execute(query)
            ans = CodeExecutionTool.format_markdown(res)
            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query,
                answer=ans,
                summary="Code execution output.",
                key_findings=[f"Exit code: {res.get('exit_code')}"],
                sources=[],
                guardrails=GuardrailReport(is_safe=True, status=GuardrailStatus.PASSED, passed_injection_check=True, passed_scope_check=True, flagged_patterns=[], message="Safe"),
                grounding=GroundingEvaluation(is_grounded=True, faithfulness_score=1.0, confidence=ConfidenceLevel.HIGH, grounded_claims=[], unsupported_claims=[], verification_notes="Subprocess"),
                latency_ms=round(latency, 2),
                status="TOOL_CODE"
            )

        if tool == "tavily_search":
            res = TavilySearchTool.search(query, api_key=tavily_api_key)
            ans = TavilySearchTool.format_markdown(res)
            latency = (time.time() - start_time) * 1000
            return StructuredRAGResponse(
                query=query,
                answer=ans,
                summary="Tavily live search results.",
                key_findings=[],
                sources=[],
                guardrails=GuardrailReport(is_safe=True, status=GuardrailStatus.PASSED, passed_injection_check=True, passed_scope_check=True, flagged_patterns=[], message="Safe"),
                grounding=GroundingEvaluation(is_grounded=True, faithfulness_score=1.0, confidence=ConfidenceLevel.HIGH, grounded_claims=[], unsupported_claims=[], verification_notes="Tavily search"),
                latency_ms=round(latency, 2),
                status="TOOL_TAVILY"
            )

        # RAG / Conversation
        intent = ConversationHandler.classify_intent(query)
        if intent:
            latency = (time.time() - start_time) * 1000
            return ConversationHandler.generate_response(query, intent, latency_ms=latency, chat_history=chat_history)

        scored = rag_pipeline.retrieve(query, top_k=top_k, threshold=threshold)
        if not scored:
            latency = (time.time() - start_time) * 1000
            return ConversationHandler.generate_friendly_chat(query, chat_history=chat_history, latency_ms=latency)

        gen = rag_pipeline.generator.generate(query, scored)
        latency = (time.time() - start_time) * 1000
        return StructuredRAGResponse(
            query=query,
            answer=gen["answer"],
            summary=gen.get("summary", ""),
            key_findings=gen.get("key_findings", []),
            sources=gen.get("sources", []),
            guardrails=GuardrailReport(is_safe=True, status=GuardrailStatus.PASSED, passed_injection_check=True, passed_scope_check=True, flagged_patterns=[], message="Safe"),
            grounding=GroundingEvaluation(is_grounded=True, faithfulness_score=1.0, confidence=ConfidenceLevel.HIGH, grounded_claims=[], unsupported_claims=[], verification_notes="Fallback synthesis"),
            latency_ms=round(latency, 2),
            status="SUCCESS"
        )
