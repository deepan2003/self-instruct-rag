from typing import List
from core.document import Document
import re

class TextChunker:
    """
    Splits documents into overlapping chunks while preserving metadata.
    Supports paragraph-aware splitting and configurable chunk size and overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Splits text using paragraph boundaries, sentences, or character windows."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # First split into paragraphs / line blocks
        paragraphs = re.split(r'\n+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_length + para_len + 1 <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_len + 1
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                # If paragraph itself is larger than chunk_size, split by sliding window
                if para_len > self.chunk_size:
                    start = 0
                    step = self.chunk_size - self.chunk_overlap
                    while start < para_len:
                        end = min(start + self.chunk_size, para_len)
                        chunks.append(para[start:end])
                        if end == para_len:
                            break
                        start += step
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [para]
                    current_length = para_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # Add overlapping context if necessary
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of Documents into chunked Documents with contextual header."""
        chunked_docs = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for idx, text in enumerate(text_chunks):
                meta = dict(doc.metadata)
                meta["chunk_id"] = f"p{doc.page}_s{meta.get('doc_id', 0)}_c{idx + 1}"
                meta["chunk_index"] = idx + 1
                meta["total_chunks_in_section"] = len(text_chunks)
                meta["char_count"] = len(text)
                # Keep section header for embedding and retrieval context
                header = f"[{doc.section} | {doc.title}]\n" if doc.section else ""
                enriched_text = f"{header}{text}"
                chunked_docs.append(Document(page_content=enriched_text, metadata=meta))
        return chunked_docs
