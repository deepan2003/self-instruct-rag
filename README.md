# 📚 SELF-INSTRUCT Paper RAG System

A fast, modular, and grounded **Retrieval-Augmented Generation (RAG)** system built on the ACL 2023 research paper:
> **"SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions"**  
> *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi*

---

## 🌟 Key Features

1. **🛡️ Multi-Layer Input Guardrails (Safety & Security Defense)**:
   - **Harmful, Violent & Sexual Content Filtering**: Rejects dangerous prompts involving weapons, explosives, violence, self-harm, sexually explicit content, and social disorder.
   - **Data Exfiltration Defense**: Prevents bulk extraction or unauthorized full-document exfiltration attempts.
   - **Adversarial & Prompt Injection Defense**: Blocks instruction overrides (`ignore previous instructions`, `reveal system prompt`, DAN jailbreaks).
   - **Quality & Spam Defense**: Filters low-information inputs, spam, and keyboard-smash gibberish.

2. **🧠 Thread & User-Isolated Conversational Memory**:
   - Asks for a **Username / Thread ID** on startup.
   - Completely separate memory management per thread ID (conversations from one user/thread are never visible to another).
   - Dedicated memory query support: recaps past discussions for the active thread.

3. **🛠️ Real-Time Live Tools**:
   - **🌦️ Tamil Nadu Live Weather**: Real-time meteorological telemetry across all 38 districts of Tamil Nadu via Open-Meteo.
   - **💻 Python Code Execution**: Sandboxed local execution of Python code with execution timings, stdout, stderr, and exit codes.
   - **🌐 Tavily Live News Search**: Real-time breaking news and web intelligence via Tavily Search API.

4. **🎯 Anti-Hallucination & Grounding Engine**:
   - Performs claim-level extraction and checks evidence against retrieved paper passages.
   - Verifies numerical consistency and assigns a quantified `faithfulness_score` and confidence rating (`HIGH`, `MEDIUM`, `LOW`, `UNGROUNDED`).
   - Rejects speculative answers when evidence is missing.

5. **📦 Strongly Typed Pydantic Output (`RAGAnswer`)**:
   - Grounded RAG responses strictly follow the Pydantic schema enforcing `answer` and `citations` (`List[SourceCitation]`).
   - Every source citation includes exact **Page Number (1-23)**, **Section**, **Title**, **Chunk ID**, **Relevance Score**, and a **Verifiable Quote Excerpt**.

6. **💬 ChatGPT-Style Interactive Streamlit UI**:
   - Persistent message history, automatic scrolling, and a disappearing status box during guardrail & pydantic verification checks.

---

## 📁 Project Structure

```
simple_rag/
├── core/
│   ├── __init__.py           # Package initializer
│   ├── models.py             # Pydantic schemas (StructuredRAGResponse, SourceCitation, GuardrailReport)
│   ├── guardrails.py         # Multi-layer input guardrails (injection, harmful, exfiltration)
│   ├── hallucination.py      # Anti-hallucination & claim-level grounding verification
│   ├── memory.py             # Thread & user-isolated conversation memory manager
│   ├── tools.py              # Live tools (Tamil Nadu weather, Code runner, Tavily news)
│   ├── document.py           # Document dataclass and JSON/text loaders
│   ├── chunker.py            # Overlapping chunker with section headers
│   ├── embeddings.py         # Base and TF-IDF embedding engines
│   ├── vector_store.py       # Vector store with cosine similarity & persistence
│   ├── retriever.py          # Top-k retriever and citation formatter
│   ├── generator.py          # Grounded answer synthesizer and prompt builder
│   ├── conversation.py       # Conversational intent classifier and friendly handler
│   └── rag_pipeline.py       # End-to-end RAG orchestrator with tool router
├── data/
│   └── self_instruct_paper.json # Full paper content mapped by page & section
├── app.py                    # Streamlit web application with thread login & chat UI
├── cli.py                    # Interactive CLI tool & security benchmark runner
├── test_rag.py               # Comprehensive unit test suite (35 automated tests)
├── requirements.txt          # Python project dependencies
├── .env.example              # Sample environment variable template
├── .gitignore                # Git ignore rules (protects .env and caches)
└── README.md                 # System documentation
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` and add your Tavily API key:
```bash
cp .env.example .env
```

### 3. Launch the Web Chat Application
```bash
streamlit run app.py
```

### 4. Run Unit Tests
```bash
python -m unittest test_rag -v
```
