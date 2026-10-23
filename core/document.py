from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
import os

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page(self) -> int:
        return self.metadata.get("page", 0)

    @property
    def section(self) -> str:
        return self.metadata.get("section", "Unknown Section")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            page_content=data["page_content"],
            metadata=data.get("metadata", {})
        )

def load_documents_from_json(file_path: str) -> List[Document]:
    """Loads documents from a structured JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    docs = []
    for idx, item in enumerate(records):
        content = item.get("content", "")
        metadata = {
            "page": item.get("page", 1),
            "section": item.get("section", ""),
            "title": item.get("title", ""),
            "authors": item.get("authors", ""),
            "doc_id": idx + 1
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs
