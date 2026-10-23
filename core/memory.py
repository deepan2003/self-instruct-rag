import json
import os
from typing import List, Dict, Any, Optional

DEFAULT_WELCOME_MESSAGE = (
    "Hello! 👋 I'm your interactive research assistant for the **SELF-INSTRUCT** paper. "
    "Ask me anything about the methodology, data generation pipeline, benchmark evaluations, or say hello!"
)

class ThreadMemoryManager:
    """
    Manages isolated conversational memory per thread / username.
    Ensures memory of one thread ID is strictly separated and not visible to other thread IDs.
    Supports in-memory caching with optional persistent JSON backup.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "thread_memory.json"
        )
        self._threads: Dict[str, List[Dict[str, Any]]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads thread histories from disk if available."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Ensure all messages have response_obj as None for loaded serialized items
                        for tid, msgs in data.items():
                            self._threads[tid] = []
                            for m in msgs:
                                self._threads[tid].append({
                                    "role": m.get("role", "user"),
                                    "content": m.get("content", ""),
                                    "response_obj": None
                                })
            except Exception:
                self._threads = {}

    def _save_to_disk(self):
        """Persists serializable thread message history to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            serializable_data = {}
            for tid, msgs in self._threads.items():
                serializable_data[tid] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                ]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_all_threads(self) -> List[str]:
        """Returns sorted list of all active thread IDs / usernames."""
        return sorted(list(self._threads.keys()))

    def thread_exists(self, thread_id: str) -> bool:
        return thread_id.strip() in self._threads

    def get_thread_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the isolated message history for the given thread_id.
        If the thread does not exist, initializes it with a welcome message.
        """
        clean_id = thread_id.strip()
        if clean_id not in self._threads or not self._threads[clean_id]:
            self._threads[clean_id] = [
                {
                    "role": "assistant",
                    "content": DEFAULT_WELCOME_MESSAGE,
                    "response_obj": None
                }
            ]
            self._save_to_disk()
        return self._threads[clean_id]

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        response_obj: Optional[Any] = None
    ):
        """Adds a message specifically to the given thread ID's memory."""
        clean_id = thread_id.strip()
        if clean_id not in self._threads:
            self.get_thread_history(clean_id)

        self._threads[clean_id].append({
            "role": role,
            "content": content,
            "response_obj": response_obj
        })
        self._save_to_disk()

    def clear_thread(self, thread_id: str):
        """Clears memory exclusively for the given thread ID."""
        clean_id = thread_id.strip()
        self._threads[clean_id] = [
            {
                "role": "assistant",
                "content": DEFAULT_WELCOME_MESSAGE,
                "response_obj": None
            }
        ]
        self._save_to_disk()

    def delete_thread(self, thread_id: str):
        """Deletes a thread and its memory completely."""
        clean_id = thread_id.strip()
        if clean_id in self._threads:
            del self._threads[clean_id]
            self._save_to_disk()

