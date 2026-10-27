import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(__file__))

from core.rag_pipeline import RAGPipeline
from core.memory import ThreadMemoryManager
from core.models import StructuredRAGResponse

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "self_instruct_paper.json")

# Initialize shared in-memory pipeline & memory manager
print("[Backend] Initializing SELF-INSTRUCT RAG Pipeline...", flush=True)
pipeline = RAGPipeline(chunk_size=1800, chunk_overlap=250)
stats = pipeline.ingest_json(DATA_PATH)
print(f"[Backend] Pipeline indexed {stats['raw_documents']} sections, {stats['chunks_created']} chunks.", flush=True)

memory_mgr = ThreadMemoryManager()
print(f"[Backend] Memory manager loaded {len(memory_mgr.get_all_threads())} threads.", flush=True)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class RAGAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_cors_headers()
        self.end_headers()
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            # If opened from a browser (accepts HTML), redirect to frontend on port 3000
            accept_header = self.headers.get("Accept", "")
            if "text/html" in accept_header:
                self.send_response(302)
                self.send_header("Location", "http://localhost:3000/")
                self.end_headers()
                return
            self._send_json(200, {
                "status": "ok",
                "service": "SELF-INSTRUCT RAG Bridge",
                "stats": stats,
                "active_threads": memory_mgr.get_all_threads()
            })
            return

        if self.path == "/api/health":
            self._send_json(200, {
                "status": "ok",
                "service": "SELF-INSTRUCT RAG Bridge",
                "stats": stats,
                "active_threads": memory_mgr.get_all_threads()
            })
            return

        if self.path == "/api/threads":
            threads = memory_mgr.get_all_threads()
            self._send_json(200, {"threads": threads})
            return

        if self.path.startswith("/api/history"):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            thread_id = params.get("thread_id", [""])[0].strip()
            if not thread_id:
                self._send_json(400, {"error": "Missing 'thread_id' parameter"})
                return

            history = memory_mgr.get_thread_history(thread_id)
            serialized = []
            for m in history:
                resp_dict = None
                if m.get("response_obj"):
                    try:
                        resp_dict = m["response_obj"].model_dump()
                    except Exception:
                        resp_dict = None

                serialized.append({
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                    "response": resp_dict
                })

            self._send_json(200, {
                "thread_id": thread_id,
                "messages": serialized
            })
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req_data = json.loads(body) if body else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if self.path == "/api/query":
            query_text = req_data.get("query", "").strip()
            thread_id = req_data.get("thread_id", "default").strip()
            top_k = int(req_data.get("top_k", 4))
            threshold = float(req_data.get("threshold", 0.10))

            if not query_text:
                self._send_json(400, {"error": "Query cannot be empty"})
                return

            # Append user message to thread memory
            memory_mgr.add_message(thread_id, role="user", content=query_text)
            current_history = memory_mgr.get_thread_history(thread_id)

            # Query the RAG Pipeline through ReAct Agent
            response: StructuredRAGResponse = pipeline.query(
                query_text,
                top_k=top_k,
                threshold=threshold,
                chat_history=current_history
            )

            # Append assistant response to thread memory
            memory_mgr.add_message(thread_id, role="assistant", content=response.answer, response_obj=response)

            # Prepare Pydantic RAGAnswer if citations present
            rag_answer_dict = None
            rag_answer_obj = response.to_rag_answer()
            if rag_answer_obj:
                rag_answer_dict = rag_answer_obj.model_dump()

            res_payload = response.model_dump()
            res_payload["rag_answer"] = rag_answer_dict

            self._send_json(200, {
                "success": True,
                "thread_id": thread_id,
                "response": res_payload
            })
            return

        if self.path == "/api/clear":
            thread_id = req_data.get("thread_id", "").strip()
            if not thread_id:
                self._send_json(400, {"error": "Missing 'thread_id'"})
                return

            memory_mgr.clear_thread(thread_id)
            self._send_json(200, {
                "success": True,
                "message": f"Thread '{thread_id}' cleared.",
                "history": memory_mgr.get_thread_history(thread_id)
            })
            return

        self._send_json(404, {"error": "Endpoint not found"})

def run_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, RAGAPIHandler)
    print(f"[Backend] Python RAG Bridge HTTP Server listening on http://127.0.0.1:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Backend] Shutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(port)
