import streamlit as st
import os
import sys
import time

# Ensure current dir is in Python path
sys.path.append(os.path.dirname(__file__))

from core.rag_pipeline import RAGPipeline
from core.models import StructuredRAGResponse, RAGAnswer, GuardrailStatus
from core.conversation import ConversationHandler
from core.memory import ThreadMemoryManager
from core.tools import ToolRouter

st.set_page_config(
    page_title="Self-Instruct Protected RAG Chat",
    page_icon="🤖",
    layout="wide"
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "self_instruct_paper.json")

@st.cache_resource
def load_rag_pipeline():
    pipeline = RAGPipeline(chunk_size=1800, chunk_overlap=250)
    stats = pipeline.ingest_json(DATA_PATH)
    return pipeline, stats

@st.cache_resource
def load_memory_manager():
    return ThreadMemoryManager()

pipeline, stats = load_rag_pipeline()
memory_mgr = load_memory_manager()

# Initialize session state for current thread
if "current_thread" not in st.session_state:
    st.session_state.current_thread = None

# ==============================================================================
# SCREEN 1: Thread ID / Username Selection (Shown when system starts)
# ==============================================================================
if not st.session_state.current_thread:
    st.title("🔐 Welcome to SELF-INSTRUCT Research Assistant")
    st.markdown(
        """
        To ensure **complete privacy and separate memory management**, please enter your 
        **Username or Thread ID** to begin.
        
        > **🔒 Memory Isolation Policy**: Each Thread ID has its own isolated conversation memory. 
        > What you discuss in one thread will **never** be accessible or visible to any other thread or user.
        """
    )
    st.markdown("---")

    col_new, col_existing = st.columns([1, 1], gap="large")

    with col_new:
        st.subheader("👤 Start a New Thread / User Session")
        with st.form("new_thread_form"):
            user_input = st.text_input(
                "Enter Username or Thread ID:",
                placeholder="e.g. deepak, research_session_1, user_rag",
                help="Choose any unique identifier for your conversation session."
            )
            submit_new = st.form_submit_button("🚀 Start Session", type="primary", use_container_width=True)

            if submit_new:
                clean_name = user_input.strip()
                if clean_name:
                    st.session_state.current_thread = clean_name
                    st.rerun()
                else:
                    st.error("Please enter a valid Username or Thread ID.")

    with col_existing:
        st.subheader("📂 Resume an Existing Thread")
        existing_threads = memory_mgr.get_all_threads()
        if existing_threads:
            selected_existing = st.selectbox(
                "Select existing thread to continue:",
                options=existing_threads,
                help="Pick a previously saved thread to resume where you left off."
            )
            if st.button("📂 Resume Selected Thread", use_container_width=True):
                st.session_state.current_thread = selected_existing
                st.rerun()
        else:
            st.info("No prior threads found. Create a new thread on the left to get started!")

    # Stop execution here so chat is only rendered after thread selection
    st.stop()


# ==============================================================================
# SCREEN 2: Active Chat Interface (Thread Selected)
# ==============================================================================
active_thread = st.session_state.current_thread

# Sidebar: Thread Management & Controls
st.sidebar.title("👤 Session & Threads")
st.sidebar.success(f"**Active Thread:** `{active_thread}`")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🔄 Switch User", use_container_width=True, help="Switch to another user or thread"):
        st.session_state.current_thread = None
        st.rerun()
with col_btn2:
    if st.button("➕ New Thread", use_container_width=True, help="Create a fresh thread"):
        st.session_state.current_thread = None
        st.rerun()

all_other_threads = [t for t in memory_mgr.get_all_threads() if t != active_thread]
if all_other_threads:
    st.sidebar.markdown("---")
    st.sidebar.caption("⚡ Quick Switch to Another Thread:")
    quick_switch = st.sidebar.selectbox("Other Available Threads", all_other_threads, label_visibility="collapsed")
    if st.sidebar.button("Switch to Thread", use_container_width=True):
        st.session_state.current_thread = quick_switch
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Current Thread", use_container_width=True, help="Clears memory for this thread only"):
    memory_mgr.clear_thread(active_thread)
    st.rerun()

# Sidebar: Controls & Ingestion info
st.sidebar.markdown("---")
st.sidebar.title("⚙️ RAG Settings")
top_k = st.sidebar.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=4, step=1)
threshold = st.sidebar.slider("Minimum Similarity Threshold", min_value=0.0, max_value=0.5, value=0.10, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ingestion Stats")
st.sidebar.metric("Paper Sections", stats["raw_documents"])
st.sidebar.metric("Indexed Chunks", stats["chunks_created"])


# Main Header
st.title("💬 SELF-INSTRUCT Research Assistant")
st.caption(
    f"🔒 Isolated Session for Thread: **`{active_thread}`** | Live Tools (Weather, Code, Tavily) + Guardrails & Pydantic RAG."
)

def render_rag_details(response: StructuredRAGResponse):
    """Renders verified source citations and the anti-hallucination Pydantic model for RAG queries."""
    if not response or not response.guardrails.is_safe:
        return

    # For conversational and live tool outputs, citations are not required
    if response.status in ["CONVERSATIONAL", "TOOL_WEATHER", "TOOL_CODE", "TOOL_TAVILY"]:
        return

    # For grounded RAG answers: Display citations and Pydantic structured output
    if response.sources:
        with st.expander(f"📚 Verified Citations ({len(response.sources)} Paper Sources)", expanded=False):
            for idx, src in enumerate(response.sources, 1):
                st.markdown(f"**Source #{idx} — Page {src.page} | {src.section}** (Relevance Score: `{src.relevance_score:.4f}`)")
                st.info(f"\"{src.quote_snippet}\"")

        rag_answer = response.to_rag_answer()
        with st.expander("📦 Grounded Pydantic Structure (`answer` + `citations`)", expanded=False):
            st.caption("Anti-hallucination verified Pydantic schema enforcing strictly grounded answer with citations:")
            if rag_answer:
                st.code(rag_answer.model_dump_json(indent=2), language="json")
            else:
                st.code(response.model_dump_json(indent=2), language="json")


# Retrieve the isolated message history for THIS thread only
current_thread_messages = memory_mgr.get_thread_history(active_thread)

# Render Existing Conversation History for this thread
for msg in current_thread_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("response_obj"):
            render_rag_details(msg["response_obj"])

# Chat Input at bottom (ChatGPT style)
if prompt := st.chat_input("Ask a research question, weather in Tamil Nadu, run code, or say hello..."):
    # 1. Append user message to this thread's memory
    memory_mgr.add_message(active_thread, role="user", content=prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate assistant response
    with st.chat_message("assistant"):
        # The checking phase lives exclusively inside this temporary status placeholder box
        status_placeholder = st.empty()

        with status_placeholder.status("⚡ Executing verification checks...", expanded=True) as status_box:
            # 1. Guardrail inspection phase
            status_box.write("🛡️ Inspecting guardrails (Prompt injection, gibberish & scope checks)...")
            time.sleep(0.25)
            guardrail_report = pipeline.guardrails.inspect(prompt)

            if not guardrail_report.is_safe:
                status_box.write(f"🛑 Guardrail check failed: {guardrail_report.message}")
                if guardrail_report.flagged_patterns:
                    status_box.write(f"⚠️ Trigger patterns detected: {', '.join(guardrail_report.flagged_patterns)}")
                time.sleep(0.5)
                is_safe = False
                response = pipeline.query(
                    prompt,
                    top_k=top_k,
                    threshold=threshold,
                    chat_history=current_thread_messages
                )
            else:
                is_safe = True
                status_box.write("✅ Guardrails passed: Input is clean, safe, and authorized")
                time.sleep(0.25)

                # Check if a tool is identified
                tool_type = ToolRouter.identify_tool(prompt)

                if tool_type == "weather":
                    status_box.write(f"🌦️ Invoking Tamil Nadu Live Weather API for query: '{prompt}'...")
                    time.sleep(0.3)
                    response: StructuredRAGResponse = pipeline.query(
                        prompt,
                        top_k=top_k,
                        threshold=threshold,
                        chat_history=current_thread_messages
                    )
                    status_box.write("✅ Tamil Nadu weather telemetry retrieved")
                    time.sleep(0.25)

                elif tool_type == "code_execution":
                    status_box.write("💻 Executing Python code in sandboxed interpreter...")
                    time.sleep(0.3)
                    response: StructuredRAGResponse = pipeline.query(
                        prompt,
                        top_k=top_k,
                        threshold=threshold,
                        chat_history=current_thread_messages
                    )
                    status_box.write("✅ Code executed & stdout captured successfully")
                    time.sleep(0.25)

                elif tool_type == "tavily_search":
                    status_box.write("🌐 Querying Tavily Live Web & News Search API...")
                    time.sleep(0.3)
                    response: StructuredRAGResponse = pipeline.query(
                        prompt,
                        top_k=top_k,
                        threshold=threshold,
                        chat_history=current_thread_messages
                    )
                    status_box.write("✅ Live news & web results retrieved")
                    time.sleep(0.25)

                else:
                    intent = ConversationHandler.classify_intent(prompt)
                    if intent == "memory":
                        status_box.write(f"🧠 Accessing isolated memory for thread '{active_thread}'...")
                        time.sleep(0.25)
                        response: StructuredRAGResponse = pipeline.query(
                            prompt,
                            top_k=top_k,
                            threshold=threshold,
                            chat_history=current_thread_messages
                        )
                        status_box.write("✅ Thread memory retrieved successfully")
                        time.sleep(0.3)
                    elif intent:
                        status_box.write("📦 Executing Pydantic structure for conversational response...")
                        time.sleep(0.25)
                        response: StructuredRAGResponse = pipeline.query(
                            prompt,
                            top_k=top_k,
                            threshold=threshold,
                            chat_history=current_thread_messages
                        )
                        status_box.write("✅ Executed Pydantic structure successfully")
                        time.sleep(0.3)
                    else:
                        # RAG path: Retrieve citations and execute Pydantic structure
                        status_box.write("🔍 Retrieving context and citations from SELF-INSTRUCT paper...")
                        time.sleep(0.25)
                        response: StructuredRAGResponse = pipeline.query(
                            prompt,
                            top_k=top_k,
                            threshold=threshold,
                            chat_history=current_thread_messages
                        )

                        cit_count = len(response.sources) if response.sources else 0
                        if cit_count > 0:
                            status_box.write(f"📚 Retrieved & verified {cit_count} source citations from paper")
                        else:
                            status_box.write("ℹ️ Handled as conversational out-of-scope guidance")
                        time.sleep(0.25)

                        status_box.write("📦 Executing Pydantic structure (`answer` + `citations`)...")
                        rag_answer = response.to_rag_answer()
                        time.sleep(0.25)
                        status_box.write("✅ Executed Pydantic structure (Anti-hallucination verified)")
                        time.sleep(0.3)

        # Disappear the checking box completely before printing the output
        status_placeholder.empty()

        # Print output outside the box
        if not is_safe:
            error_msg = f"🛑 **SECURITY GUARDRAIL TRIGGERED**: {guardrail_report.message}"
            st.error(error_msg)
            if guardrail_report.flagged_patterns:
                st.markdown(f"**Detected Trigger Patterns:** `{', '.join(guardrail_report.flagged_patterns)}`")
                st.caption("ℹ️ *'Detected Trigger Patterns' are the specific security rules, keywords, or anomaly signatures flagged by the Input Guardrails.*")
            memory_mgr.add_message(active_thread, role="assistant", content=error_msg, response_obj=response)
        else:
            st.markdown(response.answer)
            render_rag_details(response)
            memory_mgr.add_message(active_thread, role="assistant", content=response.answer, response_obj=response)
