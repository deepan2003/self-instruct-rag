from typing import List, Dict, Any, Tuple
from core.vector_store import VectorStore
from core.document import Document

class Retriever:
    """
    Retriever module that queries the VectorStore and returns
    scored document chunks with citation metadata.
    """
    def __init__(self, vector_store: VectorStore, default_top_k: int = 4, default_threshold: float = 0.05):
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[Tuple[Document, float]]:
        k = top_k if top_k is not None else self.default_top_k
        thresh = threshold if threshold is not None else self.default_threshold
        return self.vector_store.similarity_search(query, top_k=k, threshold=thresh)

    def format_citations(self, scored_docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """Formats scored documents into structured citations."""
        citations = []
        for doc, score in scored_docs:
            citations.append({
                "page": doc.page,
                "section": doc.section,
                "title": doc.title,
                "chunk_id": doc.chunk_id,
                "score": round(score, 4),
                "snippet": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            })
        return citations
